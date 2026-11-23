"""
KERNEL-SEMANTICS ORACLE (RoDS-A Step 1, O1): pure-PyTorch reference of the exact
arithmetic of the forked Nunchaku INT4 W4A4 kernel; the Step-2 packing and Step-3
GEMM parity tests compare against it. No CUDA, no packing layout, no real GEMM.

Numeric contract: rods_a/INT4_GEMM_CONTRACT.md (S-sections).
Signed remap (O0.1): Q_s = Q - 8 in {-8..7}, z' = z - 8 -- exact: s(Q-z)t == s(Q_s-z')t.
The integer reduction runs as an fp64 matmul on small integers (|Σ| well under 2^53,
so it is bit-exact); the FP epilogue runs in `accum_dtype` (FP32 to match the kernel).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from rods_a.quantize_act import quantize_act

# 4-bit unsigned->signed remap offset: Q in {0..15} -> Q_s = Q - 8 in {-8..7}.
QZERO_4BIT = 8


@dataclass
class KernelOperands:
    """The signed-domain operands the forked INT4 kernel multiplies.

    Shapes use out_f = output channels, n_tpr = Cin_pad // group_size (number of
    K-groups), gs = group_size.
    """

    Q_s: Tensor          # (out_f, n_tpr, gs) signed weight codes {-8..7}
    s: Tensor            # (out_f, n_tpr) fp32 per-(output,group) scale (kernel wmscale)
    z_signed: Tensor     # (out_f, n_tpr) fp32 signed zero-point z' = z - 8
    t_g: Tensor          # (out_f, n_tpr) fp32 per-group dual scale (fitted == 1)
    group_size: int
    out_features: int
    Cin_pad: int

    @property
    def n_tpr(self) -> int:
        return self.Q_s.shape[1]


def _unpack_weight_codes(pack) -> Tensor:
    """Unsigned weight codes Q in {0..15} as (out_f, n_tpr, gs) int64.

    Handles both bit-packed packs (meta["packing"] set) and the bitpack=False
    debug packs (raw integer codes). Row order matches rods's
    (output, tile)-major packing, so reshape((out_f, n_tpr, gs)) aligns column
    g*gs+c with the per-group s[o,g] / z[o,g] (see prepare_weight._rho_median).
    """
    meta = pack.meta
    out_f = pack.out_features
    gs = int(meta["group_size"])
    n_tpr = pack.Cin_pad // gs

    W_q = pack.W_q
    packing = meta.get("packing")
    if packing:
        from core.sinq.quantizer import Quantizer

        view_dt = meta.get("unpack_view_dtype")
        if meta.get("view_as_float", False) and view_dt is not None:
            W_q = W_q.view(view_dt)
        codes = Quantizer.unpack[packing](W_q, dtype=torch.uint8)
    else:
        codes = W_q
    # mirror Quantizer.dequantize's truncation to the valid (out_f*n_tpr) rows.
    codes = codes[: out_f * n_tpr].to(torch.int64)
    return codes.reshape(out_f, n_tpr, gs)


def to_kernel_operands(pack, *, t_g: Optional[Tensor] = None) -> KernelOperands:
    """RoDS PreparedWeight -> signed-domain kernel operands (plan O1.2).

    `t_g` is the per-group dual scale; defaults to ones (the deployed fit_t=False
    value). A grouped-t(64) upgrade passes a (out_f, n_tpr) tensor here.
    """
    meta = pack.meta
    out_f = pack.out_features
    gs = int(meta["group_size"])
    n_tpr = pack.Cin_pad // gs

    Q = _unpack_weight_codes(pack)                                # (out, G, gs) {0..15}
    Q_s = Q - QZERO_4BIT                                          # {-8..7}
    s = meta["scale"].to(torch.float32).reshape(out_f, n_tpr)
    z = meta["zero"].to(torch.float32).reshape(out_f, n_tpr)
    z_signed = z - QZERO_4BIT

    if t_g is None:
        t_g = torch.ones(out_f, n_tpr, dtype=torch.float32)
    else:
        t_g = t_g.to(torch.float32).reshape(out_f, n_tpr)

    return KernelOperands(
        Q_s=Q_s, s=s, z_signed=z_signed, t_g=t_g,
        group_size=gs, out_features=out_f, Cin_pad=pack.Cin_pad,
    )


def dequant_weight_signed(ops: KernelOperands) -> Tensor:
    """Reconstruct W̃_hat (out_f, Cin_pad) from the SIGNED operands:
    Ŵ[o,k] = s[o,g]·t_g[o,g]·(Q_s[o,k] - z'[o,g]). Used by the round-trip identity
    test (O1.2): this must equal the minus-z unsigned dequant and SINQ's
    Quantizer.dequantize.
    """
    out_f, n_tpr, gs = ops.Q_s.shape
    sg = (ops.s * ops.t_g).reshape(out_f, n_tpr, 1)              # (out, G, 1)
    z = ops.z_signed.reshape(out_f, n_tpr, 1)
    W = sg * (ops.Q_s.to(torch.float32) - z)                    # (out, G, gs)
    return W.reshape(out_f, n_tpr * gs)


def oracle_matmul(
    Xq: Tensor,
    a: Tensor,
    ops: KernelOperands,
    *,
    out_dtype: torch.dtype = torch.float16,
    accum_dtype: torch.dtype = torch.float32,
    apply_t: bool = True,
    scale_dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """The kernel-semantics INT4 GEMM + epilogue (plan O0.5/O0.9).

    Args:
        Xq:  (T, Cin_pad) activation codes in {-7..7} (float or int).
        a:   (T, 1) per-token activation scale (fp).
        ops: signed-domain weight operands.
        out_dtype:   final cast (O0.6); FP16 matches the deployed kernel.
        accum_dtype: cross-group + epilogue FP accumulation dtype (FP32 = kernel).
        apply_t:     multiply by t_g (False removes the t-rescale stage; with the
                     deployed t_g == 1 both are identical -- test a).
        scale_dtype: storage dtype of the kernel's scale OPERANDS as loaded from HBM
                     (plan P0.3). Nunchaku loads wmscales/ascales as **FP16**
                     (gemm_w4a4.cuh:75, gemm_base.cuh:181-183, :452-468), while the
                     oracle's fits are FP32. None (default) keeps the legacy FP32
                     scales (back-compat with O1.4/O2.3). `torch.float16` rounds the
                     wmscale `(s·t_g)` and the per-token `a_i` to FP16 BEFORE the FP32
                     epilogue, matching the deployed kernel so Step-3 parity is
                     apples-to-apples. Only the scale operands are rounded; the int
                     reduction and FP32 accumulation are unchanged.

    Returns (T, out_f) in out_dtype.
    """
    T = Xq.shape[0]
    out_f, n_tpr, gs = ops.Q_s.shape
    assert Xq.shape[1] == n_tpr * gs, (
        f"Xq Cin={Xq.shape[1]} != n_tpr*gs={n_tpr * gs} (Cin_pad mismatch)"
    )
    # Align the (CPU-built) weight operands to the activation device; the reference
    # path likewise runs on the activation's device.
    dev = Xq.device
    Xq_g = Xq.reshape(T, n_tpr, gs)
    # Round the scale operands to the kernel's HBM dtype first (P0.3), then promote
    # to accum_dtype: this is exactly what the kernel does (load fp16 -> fma in fp32).
    a_in = a.reshape(T, 1)
    if scale_dtype is not None:
        a_in = a_in.to(scale_dtype)
    a_f = a_in.to(accum_dtype).to(dev)
    Qs = ops.Q_s.to(dev)
    s = ops.s.to(dev)
    z_signed = ops.z_signed.to(dev)
    tg = (ops.t_g if apply_t else torch.ones_like(ops.t_g)).to(dev)
    wmscale = s * tg                                    # (out_f, n_tpr), fp32
    if scale_dtype is not None:
        wmscale = wmscale.to(scale_dtype).to(accum_dtype)

    Y = torch.zeros(T, out_f, dtype=accum_dtype, device=dev)
    for g in range(n_tpr):
        # --- EXACT int4xint4 reduction (fp64 matmul on small integers) ---
        Xqg64 = Xq_g[:, g, :].to(torch.float64)                 # (T, gs)
        Qsg64 = Qs[:, g, :].to(torch.float64)                   # (out, gs)
        Ig = Xqg64 @ Qsg64.t()                                  # (T, out) exact int32
        Pg = Xqg64.sum(dim=1, keepdim=True)                     # (T, 1) exact
        # --- epilogue in accum_dtype: z-correction folded per group ---
        corr = z_signed[:, g].to(accum_dtype).unsqueeze(0) * Pg.to(accum_dtype)
        contrib = Ig.to(accum_dtype) - corr                    # (T, out)
        # per-group scale, with a_i applied per group (Nunchaku apply_scales:
        # fsum += i2f(psum) * (ascale*wscale)). wmscale is pre-rounded to the
        # kernel's HBM dtype when scale_dtype is set (P0.3).
        g_scale = a_f * wmscale[:, g].to(accum_dtype).unsqueeze(0)
        Y = Y + contrib * g_scale
    return Y.to(out_dtype)


def oracle_linear(
    X: Tensor,
    pack,
    cfg,
    *,
    t_g: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    out_dtype: torch.dtype = torch.float16,
    accum_dtype: torch.dtype = torch.float32,
    rotate_impl: str = "dense",
    rotate_dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Single-layer oracle: FWHT -> per-token quant (reused estimator) -> kernel
    GEMM semantics (plan O1.1). `cfg` is an ActQuantConfig.

    The activation rotation follows `pack.rotated` so X̃ and W̃ live in the same
    basis. Per-token centering follows `cfg.center_act_mean` (Contract v3, Appendix
    S6): when on, the per-token mean is removed PRE-rotation (inside quantize_act)
    and the exact contribution is re-injected as the rank-1 `x̄ ⊗ colsum`, where
    `colsum` is the column sum of the ORIGINAL FP weight (DECISION 1 / `pack.colsum`),
    NOT the dequantized Ŵ — building it from Ŵ row-sums re-injects quant error into
    the protected DC channel and fails parity (R2.4). `oracle_matmul` stays the pure
    grouped GEMM; the rank-1 is added here in `accum_dtype` (fp32) before the single
    final cast to `out_dtype`.
    """
    center_mean = bool(getattr(cfg, "center_act_mean", False))
    orig_shape = X.shape
    x2d = X.reshape(-1, orig_shape[-1])
    _, diag = quantize_act(
        x2d,
        b=cfg.act_bits,
        gamma_star=cfg.gamma_star,
        estimator=cfg.scale_estimator,
        rotate_act=pack.rotated,
        percentile_p=cfg.percentile_p,
        huber_c=cfg.huber_c,
        return_diag=True,
        instrument=getattr(cfg, "instrument", False),  # §6 floats unused here (opt-B/B1)
        return_rotated=True,
        center_mean=center_mean,
        rotate_impl=rotate_impl,
        rotate_dtype=rotate_dtype,
    )
    a = diag["a"]                       # (T, 1)
    Xq = diag["Xq_codes"]              # (T, Cin_pad) integer codes (±7)

    ops = to_kernel_operands(pack, t_g=t_g)

    if bias is None and getattr(pack, "bias", None) is not None:
        bias = pack.bias

    if center_mean:
        # Accumulate matmul + rank-1 mean + bias in accum_dtype (fp32), one final cast.
        colsum = getattr(pack, "colsum", None)
        if colsum is None:
            raise ValueError(
                "center_mean=True requires pack.colsum (the FP-weight column sum; "
                "prepare_weight must precompute it). It must NOT be derived from Ŵ."
            )
        Y = oracle_matmul(Xq, a, ops, out_dtype=accum_dtype, accum_dtype=accum_dtype)
        x_bar = diag["x_bar"].to(device=Y.device, dtype=Y.dtype)          # (T, 1) fp32
        Y = Y + x_bar * colsum.to(device=Y.device, dtype=Y.dtype)        # (T, out_f)
        if bias is not None:
            Y = Y + bias.to(device=Y.device, dtype=Y.dtype)
        return Y.reshape(*orig_shape[:-1], ops.out_features).to(out_dtype)

    # --- uncentered: unchanged legacy path (bit-identical to pre-v3) ---
    Y = oracle_matmul(Xq, a, ops, out_dtype=out_dtype, accum_dtype=accum_dtype)
    if bias is not None:
        Y = Y + bias.to(device=Y.device, dtype=Y.dtype)
    return Y.reshape(*orig_shape[:-1], ops.out_features).to(out_dtype)
