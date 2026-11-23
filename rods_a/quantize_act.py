"""
Per-token activation fake-quantization (doc 3.3).

Pipeline per forward:
  1. optionally rotate X on the contraction axis (orthonormal Hadamard, pad fallback)
  2. estimate a per-token scale sigma_hat (bounded-influence estimator)
  3. clip = gamma* * sigma_hat ;  step a = clip / q_max  (q_max = 2^{b-1} - 1)
  4. Xq = clip(round(Xt / a), -q_max, q_max)         [INT grid, fake-quant only]
  5. dequantize Xt_hat = a * Xq ; unrotate back to the original basis

This phase is FAKE-QUANT ONLY: it returns the dequantized FP tensor (no int4
bit-packing) so we can measure SQNR. Weights are never touched here.

Six scale estimators behind one flag:
  percentile  : robust high-quantile, Gaussian-consistent  (kernel default)
  mad         : normalized median-absolute-deviation
  huber       : Huber M-estimator of scale (theory / ablation)
  std         : per-token sample std -- a CLIPPING estimator (clip = gamma* * std)
                but with UNBOUNDED influence (one outlier inflates std and widens
                the clip). The non-robust per-token clip baseline: the ablation
                hinge that separates "clipping helps" (std beats absmax) from
                "robustness helps" (mad/huber/percentile beat std on heavy tails).
  absmax      : per-token max|.| used directly as the clip (NO gamma*) -- the
                "no-clip" baseline the Phase-1 SQNR exit must beat
  per_tensor  : a single global scale for the whole tensor -- the "per-tensor"
                baseline the Phase-1 exit must beat
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Tuple

import torch
from torch import Tensor

from rods_a.constants import beta_c, phi_inv
from rods_a.fwht import ROTATION_SCHEMES, rotate_for, unrotate_for
from rods_a import profiling as _profiling

_EPS = 1e-12

# Estimators that already produce a std-equivalent sigma_hat (so the clip is
# gamma* * sigma_hat). `absmax` is special (clip = max|.|, gamma* unused).
_SIGMA_ESTIMATORS = ("percentile", "mad", "huber", "std", "per_tensor")
ESTIMATORS = ("percentile", "mad", "huber", "std", "absmax", "per_tensor")


def _q_max(b: int) -> int:
    """Largest positive code on the signed-symmetric grid {-(2^{b-1}-1) .. 2^{b-1}-1}."""
    return (1 << (b - 1)) - 1


def huber_scale_fixedpoint(
    Xt: Tensor,
    c: float = 1.345,
    iters: int = 3,
    dim: int = -1,
) -> Tensor:
    """Per-token Huber M-estimator of scale (fixed-point).

    sigma <- sigma * sqrt( mean(psi_c(Xt/sigma)^2, dim) / beta ),  psi_c = clip(.,-c,c)
    initialised from a robust MAD estimate. Returns sigma with a kept dim.
    """
    beta = beta_c(c)
    # Robust init: 1.4826 * MAD (== MAD / 0.6745).
    med = Xt.median(dim=dim, keepdim=True).values
    mad = (Xt - med).abs().median(dim=dim, keepdim=True).values
    sigma = (1.4826 * mad).clamp_min(_EPS)
    for _ in range(iters):
        u = Xt / sigma
        psi = u.clamp(-c, c)
        sigma = sigma * torch.sqrt(psi.pow(2).mean(dim=dim, keepdim=True) / beta).clamp_min(_EPS)
        sigma = sigma.clamp_min(_EPS)
    return sigma


def estimate_scale(
    Xt: Tensor,
    estimator: str,
    *,
    percentile_p: float = 0.999,
    huber_c: float = 1.345,
) -> Tensor:
    """Per-token (per-row) std-equivalent scale sigma_hat, reduced over the last
    (channel) axis with keepdim. `Xt` is (T, C). `absmax` is handled in
    `quantize_act` (it is not a sigma estimator). Returns (T, 1)."""
    if estimator == "percentile":
        # Nearest-rank quantile of |Xt|. The idx-th SMALLEST value equals the k-th
        # LARGEST with k = C - idx + 1; for a high quantile (p=0.999) k is tiny
        # (~0.001*C), so a small `topk` returns the SAME order statistic as
        # `kthvalue` but is much cheaper on GPU (kthvalue does a full selection;
        # topk with small k is a partial reduction). Fall back to kthvalue when the
        # rank is not near an extreme (k > C/2), so low-p still works.
        C = Xt.shape[-1]
        idx = max(1, min(C, int(round(percentile_p * C))))
        k = C - idx + 1
        Xa = Xt.abs()
        if k <= max(1, C // 2):
            q = Xa.topk(k, dim=-1).values[..., -1:]            # k-th largest == idx-th smallest
        else:
            q = Xa.kthvalue(idx, dim=-1, keepdim=True).values
        # half-normal Gaussian-consistency: P(|x|<=t)=p => t = sigma*Phi^{-1}((1+p)/2)
        denom = phi_inv(0.5 * (1.0 + percentile_p))
        return (q / denom).clamp_min(_EPS)
    if estimator == "mad":
        med = Xt.median(dim=-1, keepdim=True).values
        mad = (Xt - med).abs().median(dim=-1, keepdim=True).values
        return (mad / 0.6745).clamp_min(_EPS)
    if estimator == "huber":
        return huber_scale_fixedpoint(Xt, c=huber_c, dim=-1)
    if estimator == "std":
        # Per-token sample std. Unbiased estimator at the Gaussian, but with
        # UNBOUNDED influence: a single large outlier inflates it (and hence the
        # clip), unlike the robust estimators above. This is the non-robust
        # per-token CLIP baseline.
        return Xt.std(dim=-1, keepdim=True).clamp_min(_EPS)
    if estimator == "per_tensor":
        # One global std-equivalent scale, broadcast to every token.
        sigma = Xt.std().clamp_min(_EPS)
        return sigma.expand(Xt.shape[0], 1).contiguous()
    raise ValueError(
        f"Unknown sigma estimator '{estimator}'. Choose from {_SIGMA_ESTIMATORS}."
    )


def _excess_kurtosis(x: Tensor) -> float:
    """Excess kurtosis (0 for Gaussian) over all elements."""
    xf = x.flatten().to(torch.float64)
    mu = xf.mean()
    var = xf.var(unbiased=False).clamp_min(_EPS)
    return float(((xf - mu).pow(4).mean() / var.pow(2)) - 3.0)


def quantize_act(
    X: Tensor,
    b: int,
    gamma_star: float,
    estimator: str = "percentile",
    *,
    rotate_act: bool = True,
    percentile_p: float = 0.999,
    huber_c: float = 1.345,
    return_diag: bool = True,
    instrument: bool = True,
    return_rotated: bool = False,
    center_mean: bool = False,
    rotation: str = "pad",
    rotate_impl: str = "fwht",
    rotate_dtype: Optional[torch.dtype] = None,
    passthrough: bool = False,
    rotation_seed: int = 0,
    rotate_override: Optional[Callable[[Tensor], Tensor]] = None,
    unrotate_override: Optional[Callable[..., Tensor]] = None,
) -> Tuple[Tensor, Dict[str, float]]:
    """Fake-quantize activations per-token (doc 3.3). Returns the dequantized FP
    tensor X_hat (same shape/dtype as X) and a diagnostics dict.

    Args:
        X:            (T, Cin) activations entering a linear (FP16/BF16/FP32).
        b:            activation bit-width (q_max = 2^{b-1} - 1).
        gamma_star:   optimal-clip constant in sigma units (ignored for absmax).
        estimator:    one of ESTIMATORS.
        rotate_act:   apply the orthonormal Hadamard rotation before quantizing.
        rotation:     rotation scheme, "pad" (zero-pad to pow2; default, legacy
                      bit-untouched) or "kronecker" (deployment basis, K = Cin).
                      Must match the weight side's `prepare_weight(rotation=...)`.
        center_mean:  subtract the per-token mean over the real Cin dims (in FP32)
                      BEFORE padding+rotation, so the Hadamard DC coordinate carries
                      ~0 (no mean spike to inflate the per-token scale). The removed
                      mean x_bar (T,1) is returned via diag["x_bar"] (when return_diag)
                      so the caller can restore the exact contribution downstream as
                      x_bar * colsum(W). The pad region stays zero. Default OFF
                      (numerically a no-op).
        return_rotated: if True (and rotate_act), return the dequantized activation
                      in the ROTATED basis (skip the final unrotate) so it can be
                      matmul'd directly against a rotated weight W̃ (RoDS-A Phase 2
                      composed W4A4 forward). The returned tensor then has the
                      padded width Cin_pad >= Cin. The per-token scale `a` and the
                      integer codes `Xq` are added to the diag dict (when
                      return_diag) under keys "a" / "Xq_codes" for the folded-form
                      equivalence checks. No effect when rotate_act is False.
        instrument:   compute the §6 analysis FLOAT diagnostics (kurtosis_excess,
                      residual_channel_mean_*, clip_fraction). Each is a host-sync
                      `float(...)` of a device tensor (opt-B/B0.2: 4 DtoH syncs/call).
                      Default True keeps every analysis caller unchanged; the deployed
                      int4/oracle paths thread `cfg.instrument` (default OFF) so the hot
                      forward NEVER computes them — when off they are not computed at all
                      (not computed-and-dropped). DECOUPLED from the operand tensors
                      a/Xq_codes/x_bar, which stay gated by return_rotated/center_mean and
                      are device-resident (no host transfer). Only meaningful when
                      return_diag is True.
        passthrough:  skip the quantization itself (scale estimate + round/clamp)
                      while keeping centering + rotation identical: X̂t = X̃t exactly,
                      a = 1 per token, Xq_codes = None. The U2.2 "W4A16-on-W̃"
                      decomposition arm (FP activations against the quantized W̃_hat).
                      Default OFF (bit-identical legacy path). Codes-domain consumers
                      (int4 / kernel-semantics) must not request passthrough.
    """
    if estimator not in ESTIMATORS:
        raise ValueError(f"Unknown estimator '{estimator}'. Choose from {ESTIMATORS}.")

    if rotate_impl not in ("fwht", "dense"):
        raise ValueError(f"rotate_impl must be 'fwht' or 'dense', got {rotate_impl!r}")
    if rotate_override is not None:
        # R1.4 discrepancy sweep: run the LOCKED quant path on a CUSTOM rotation
        # basis (a variant order-2^k Kronecker factor), bypassing the string
        # dispatch. Requires the matching unrotate_override. The default path
        # (rotate_override None) is byte-for-byte unchanged -> goldens unaffected.
        if unrotate_override is None:
            raise ValueError("rotate_override requires a matching unrotate_override")
        _rot = rotate_override
    else:
        if rotation not in ROTATION_SCHEMES:
            raise ValueError(f"rotation must be one of {ROTATION_SCHEMES}, got {rotation!r}")
        # rotation_seed: the ±1 diagonal seed for "kronecker_d" (U4.B); ignored by
        # pad/kronecker, so the legacy paths stay bit-identical.
        _rot = rotate_for(rotation, rotate_impl, seed=rotation_seed)

    orig_dtype = X.dtype
    Cin = X.shape[-1]
    Xf = X.to(torch.float32)
    x_bar = None
    if center_mean:
        # Per-token mean over the real Cin dims, removed before padding+rotation so
        # the Hadamard DC coordinate carries ~0. Restored downstream via x_bar*colsum.
        with _profiling.section("center"):
            x_bar = Xf.mean(dim=-1, keepdim=True)
            Xf = Xf - x_bar
    if rotate_act:
        # Rotate in `rotate_dtype` (e.g. fp16/bf16 -> tensor cores, fp32 accumulate),
        # then bring the result back to fp32 so the scale estimate + round/clamp stay
        # full-precision. rotate_dtype=None keeps the legacy all-fp32 path bit-for-bit.
        rdt = rotate_dtype or torch.float32
        with _profiling.section("rotate"):
            Xt = _rot(Xf.to(rdt)).to(torch.float32)  # (T, Cin_padded)
    else:
        Xt = Xf
    q_max = _q_max(b)

    with _profiling.section("scale_estimate"):
        if passthrough:
            a = torch.ones_like(Xt[..., :1])
        elif estimator == "absmax":
            a = (Xt.abs().amax(dim=-1, keepdim=True) / q_max).clamp_min(_EPS)
        else:
            sigma = estimate_scale(Xt, estimator, percentile_p=percentile_p, huber_c=huber_c)
            a = ((gamma_star / q_max) * sigma).clamp_min(_EPS)

    with _profiling.section("round_clamp"):
        if passthrough:
            Xq = None                  # no codes exist; the FP activation is kept exact
            Xt_hat = Xt
        else:
            Xq = torch.clamp(torch.round(Xt / a), -q_max, q_max)
            Xt_hat = a * Xq

    diag: Dict[str, float] = {}
    if return_diag:
        # §6 analysis floats: each is a host-sync float() — gated behind `instrument`
        # so the deployed forward (instrument off) computes NONE of them (opt-B/B1).
        if instrument:
            residual_channel_mean = (Xt - Xt_hat).mean(dim=0)  # (C,)
            diag["kurtosis_excess"] = _excess_kurtosis(Xt)
            diag["residual_channel_mean_absmax"] = float(residual_channel_mean.abs().max())
            diag["residual_channel_mean_l2"] = float(residual_channel_mean.norm())
            diag["clip_fraction"] = float((Xt.abs() > a * q_max).float().mean())
        # Operand tensors are DECOUPLED from the floats: device-resident, no host sync.
        if center_mean:
            diag["x_bar"] = x_bar        # (T, 1) per-token mean removed pre-rotation
        if return_rotated:
            diag["a"] = a                # (T, 1) per-token scale
            diag["Xq_codes"] = Xq        # (T, Cin_pad) integer codes (fake-quant)

    if return_rotated and rotate_act:
        # Stay in the rotated basis: caller multiplies against a rotated weight W̃,
        # so the inverse rotation that would normally restore the original basis
        # is the caller's GEMM partner (X̃_hat W̃ᵀ == X_hat Wᵀ). Returned width is
        # the padded Cin_pad.
        return Xt_hat.to(orig_dtype), diag

    _unrot = (unrotate_override if rotate_override is not None
              else unrotate_for(rotation, rotate_impl, seed=rotation_seed))
    X_hat = _unrot(Xt_hat, Cin) if rotate_act else Xt_hat
    return X_hat.to(orig_dtype), diag
