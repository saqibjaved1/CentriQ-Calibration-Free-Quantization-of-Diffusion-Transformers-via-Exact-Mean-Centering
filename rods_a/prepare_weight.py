"""
PREPARE_WEIGHT for RoDS-A Phase 2 (doc 3.2): rotate a linear's weight on the
contraction axis and re-fit RoDS on the *rotated* weight W̃ = W H.

1. Re-fit, do not transfer: H mixes input channels, so RoDS's BCD is re-run on W̃.
2. t absorbed (`fit_t=False`): the stored quantizer is a per-(output, input-group)
   affine (s, z) only — the form that folds into a grouped-INT4 kernel (doc 3.4 / 4).

Diagnostics: ρ = σ_FP/Δ median per block (healthy [2, 4]); incoherence = max/mean column-norm.

Fake-quant only: no bit-level INT4 kernel here (Phase 4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import Tensor

from rods_a.fwht import ROTATION_SCHEMES, next_pow2, rotate_for
from rods.quantizer import quantize_rods


@dataclass
class PreparedWeight:
    """Container for a rotated, RoDS-re-fit (grouped, t-absorbed) weight."""

    W_q: Tensor                 # packed integer codes of W̃
    meta: dict                  # SINQ-style meta (scale2 is None: t absorbed)
    Cin_orig: int               # pre-pad contraction width
    Cin_pad: int                # padded (rotated) contraction width
    rotated: bool               # whether W̃ = W H was applied (matches act rotate)
    out_features: int
    rho_median: float
    incoherence_W: float
    incoherence_Wt: float
    bias: Optional[Tensor] = None
    colsum: Optional[Tensor] = None  # (out_f,) Σ_j W[:,j] over real Cin (FP, mean pathway)


def max_colnorm_ratio(W: Tensor) -> float:
    """Incoherence proxy: max_k ||W[:,k]|| / mean_k ||W[:,k]||.

    1.0 means perfectly flat column norms; large values flag a few dominant
    (outlier) input channels that hurt grouped quantization.
    """
    col_norms = W.float().norm(dim=0)  # (Cin,)
    mean = col_norms.mean().clamp_min(1e-12)
    return float(col_norms.max() / mean)


def _rho_median(Wt: Tensor, meta: dict) -> float:
    """ρ = σ_FP / Δ median over (output, group) blocks.

    Δ (quant step) is the per-(output,group) scale s (codes are integer-spaced by
    1, so adjacent reconstructed levels differ by s). σ_FP is the std of W̃ within
    the same block. Higher ρ = finer relative resolution; ACIQ-healthy band ~[2,4].
    """
    out_f, in_f = Wt.shape
    group_size = int(meta["group_size"])
    n_tpr = in_f // group_size
    # std of W̃ per (out, group): (out_f, n_tpr)
    blocks = Wt.float().reshape(out_f, n_tpr, group_size)
    sigma = blocks.std(dim=-1, unbiased=False)            # (out_f, n_tpr)
    # s is stored (out_f * n_tpr, 1) in the (out, tile) major order; reshape back.
    s = meta["scale"].float().reshape(out_f, n_tpr)       # Δ per (out, group)
    rho = sigma / s.clamp_min(1e-12)
    return float(rho.median())


def prepare_weight(
    weight: Tensor,
    *,
    bias: Optional[Tensor] = None,
    bits: int = 4,
    group_size: int = 64,
    p: float = 0.7,
    K_outer: int = 3,
    K_irls: int = 5,
    tol_outer: float = 1e-3,
    tol_irls: float = 1e-3,
    device: str = "cpu",
    compute_dtype: torch.dtype = torch.float16,
    bitpack: bool = True,
    rotate_weight: bool = True,
    rotation: str = "pad",
    rho_band: tuple[float, float] = (2.0, 4.0),
    strict_rho: bool = True,
    fit_t: bool = False,
    rotation_seed: int = 0,
    rotate_fn: Optional[Callable[[Tensor], Tensor]] = None,
) -> PreparedWeight:
    """Rotate `weight` (Cout, Cin) on the contraction axis and re-fit grouped
    RoDS (t absorbed) on W̃.

    Args:
        rotate_weight: if False, skip the Hadamard rotation and re-fit RoDS on W
            directly (Cin_pad == Cin_orig). Must match the activation path's
            `rotate_act`; the installer keeps them consistent. This is the
            rotation-on/off realizability ablation isolator.
        rotation: rotation scheme when `rotate_weight` is True. "pad" (zero-pad to
            pow2; Cin_pad = next_pow2(Cin)) or "kronecker" (deployment basis,
            Cin_pad = Cin, no padding). Must match the activation path's
            `quantize_act(rotation=...)`.
        strict_rho: if True, raise when the ρ median falls outside `rho_band`
            (the Phase-2 gate). The dry-run sets this False to collect every
            layer's ρ before deciding.
        fit_t: W4A4 deployment absorbs the per-input-column dual scale t into the
            per-(out,group) s (fit_t=False) so the affine folds into the grouped-INT4
            epilogue. fit_t=True keeps the free per-column t — the Phase-3 free-t arm
            (A12), FAKEQUANT ONLY: meta["scale2"] is populated and applied by
            Quantizer.dequantize; the int4/kernel epilogue does not consume it.
        rotate_fn: optional explicit rotation callable (factored-rotation path).
            When set (and rotate_weight is True) it replaces the string-dispatched
            scheme entirely: W̃ = rotate_fn(W) and Cin_pad is read off the rotated
            width. Must be the SAME map the activation side installs via
            rotate_override so X̃ W̃ᵀ == X Wᵀ. Default None keeps the existing
            scheme dispatch byte-for-byte.
    """
    W = weight.detach().to(torch.float32)
    out_f, Cin_orig = W.shape

    if rotate_weight and rotate_fn is not None:
        # Factored-rotation injection: the caller supplies the exact orthonormal
        # map (shared with the activation side); the rotated width defines Cin_pad.
        Wt = rotate_fn(W)                                  # (out_f, Cin_pad)
        Cin_pad = Wt.shape[-1]
    elif rotate_weight:
        # W̃ = W H on the last/contraction axis, reusing the exact same orthonormal
        # map as the activation path (so X̃ W̃ᵀ == X Wᵀ in FP). "pad" zero-pads Cin
        # to a power of two; "kronecker" keeps the native width (Cin_pad == Cin).
        if rotation not in ROTATION_SCHEMES:
            raise ValueError(f"rotation must be one of {ROTATION_SCHEMES}, got {rotation!r}")
        # kronecker_d (U4.B) keeps K = Cin like kronecker; rotation_seed drives its
        # ±1 diagonal (ignored by pad/kronecker — legacy bit-identical). dense_haar
        # (R1.3 diagnostic ceiling) is a dense native-width orthogonal — also K = Cin.
        Cin_pad = (Cin_orig if rotation in ("kronecker", "kronecker_d", "dense_haar")
                   else next_pow2(Cin_orig))
        Wt = rotate_for(rotation, "fwht", seed=rotation_seed)(W)  # (out_f, Cin_pad)
    else:
        Cin_pad = Cin_orig
        Wt = W

    # Re-fit RoDS on W̃ with t absorbed (grouped affine, W4A4-realizable).
    W_q, meta = quantize_rods(
        Wt,
        nbits=bits,
        group_size=group_size,
        p=p,
        K_outer=K_outer,
        K_irls=K_irls,
        tol_outer=tol_outer,
        tol_irls=tol_irls,
        device=device,
        compute_dtype=compute_dtype,
        bitpack=bitpack,
        fit_t=fit_t,
    )
    if not fit_t:
        assert meta.get("scale2") is None, "fit_t=False must yield scale2=None"

    rho_med = _rho_median(Wt, meta)
    incoh_W = max_colnorm_ratio(W)
    incoh_Wt = max_colnorm_ratio(Wt)
    # Mean-pathway correction vector for per-token centering (DECISION 1: FP weight,
    # so the DC/mean part is carried exactly): colsum[k] = Σ_j W[k,j] over real Cin.
    colsum = W.sum(dim=1).to(compute_dtype)

    lo, hi = rho_band
    if not (lo <= rho_med <= hi):
        msg = (
            f"[prepare_weight] rho_median={rho_med:.3f} outside healthy band "
            f"[{lo}, {hi}] (out={out_f}, Cin {Cin_orig}->{Cin_pad}, gs={group_size}). "
            f"Rotation may have pushed RoDS out of resolution."
        )
        if strict_rho:
            raise ValueError(msg)
        print("WARNING: " + msg)

    return PreparedWeight(
        W_q=W_q,
        meta=meta,
        Cin_orig=Cin_orig,
        Cin_pad=Cin_pad,
        rotated=rotate_weight,
        out_features=out_f,
        rho_median=rho_med,
        incoherence_W=incoh_W,
        incoherence_Wt=incoh_Wt,
        bias=None if bias is None else bias.detach().to(compute_dtype),
        colsum=colsum,
    )


def dequantize_prepared(pack: PreparedWeight) -> Tensor:
    """Dequantize the rotated weight W̃_hat (out_f, Cin_pad) for the fake-quant
    composed forward. Uses SINQ's Quantizer.dequantize (scale2=None path)."""
    from core.sinq.quantizer import Quantizer

    return Quantizer.dequantize(pack.W_q, pack.meta).float()
