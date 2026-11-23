"""
OPERAND PACKING + CORRECTION PRECOMPUTE (RoDS-A Step 2, P1/P2).

Host-only (torch, no CUDA): turns a RoDS PreparedWeight + quantized activations into
the exact operand buffers the forked Nunchaku INT4 W4A4 GEMM (Step 3) consumes, and
precomputes the offline z-correction. The fork keeps the stock Nunchaku main GEMM (the
`wmscale` slot verbatim, no per-element weight zero-point buffer) and folds the
weight zero-point into a static correction `C` plus a cheap dense side-GEMM:

    Y_ij = a_i · Σ_g (s_jg·t_g)·[ Σ_{k∈g} X^q_ik · Q^s_jk ]  −  a_i · (P Cᵀ)_ij
      wmscale_jg = s_jg · t_g          (stock Nunchaku slot; = s_jg under t≡1)
      C_jg       = s_jg · t_g · z'_jg  (static, precomputed offline, FP32 compute)
      P_ig       = Σ_{k∈g} X^q_ik      (emitted during activation packing)

so the per-(out,group) z' never enters the int4 mainloop. `verify_epilogue_identity`
(used by the P3.1 gate) checks this restructured form against the Step-1 oracle.

Contract (O0 + P0): weight codes signed-remapped Q_s = Q-8 ∈ {-8..7}, packed via the
vendored packer's `&0xF` two's-complement convention; wmscales and a_i FP16 (P0.3);
activations clamp ±7 (never -8); rotation = zero-pad to next_pow2 (P0.4), Cin_pad ÷64/÷128.

Vendored packer: rods_a._nunchaku_pack (byte-faithful copy of commit 8f41840).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from rods_a._nunchaku_pack import NunchakuWeightPacker
from rods_a.kernel_semantics import KernelOperands, to_kernel_operands
from rods_a.quantize_act import quantize_act

_WARP_N = 128  # Nunchaku default warp tile in the output (n) dimension.


# --------------------------------------------------------------------------- #
# Weight-side packing (offline path, P1)
# --------------------------------------------------------------------------- #
@dataclass
class PackedWeights:
    """Operand buffers for one W4A4 linear (offline)."""

    Qpacked: Tensor      # int8 (out_f, Cin_pad/2) — Nunchaku-packed signed int4 codes
    wmscales: Tensor     # fp16, Nunchaku wscale layout (validation view (-1, out_f))
    C: Tensor            # (out_f, n_tpr) correction s·t·z'  (dtype per `c_dtype`)
    wmscale_dense: Tensor  # fp16 (out_f, n_tpr) — pre-pack s·t_g (kept for verify/fixtures)
    meta: dict
    # Per-token centering (Contract v3, S6): the ORIGINAL FP-weight column sum
    # colsum_j = Σ_k W_fp[j,k] over the real Cin, fp32 (DECISION 1 — carried through
    # from PreparedWeight.colsum, NEVER recomputed from the dequantized Ŵ). None when
    # the prepared weight was built without centering support.
    colsum: Optional[Tensor] = None


def pack_weights(
    pack,
    *,
    t_g: Optional[Tensor] = None,
    c_dtype: torch.dtype = torch.float32,
    warp_n: int = _WARP_N,
    scheme: str = "zero-pad",
) -> PackedWeights:
    """RoDS PreparedWeight -> {Qpacked, wmscales, C, ...} (plan P1.1).

    Args:
        pack:    a PreparedWeight (prepare_weight output) with the grouped (s,z) fit.
        t_g:     per-group dual scale; defaults to ones (deployed fit_t=False).
        c_dtype: storage dtype of the correction C (P3.3 decides fp16 vs fp32; compute
                 is always FP32). Default FP32 (lossless).
        warp_n:  Nunchaku output-tile width (128).
    """
    ops = to_kernel_operands(pack, t_g=t_g)
    out_f, n_tpr, gs = ops.Q_s.shape
    Cin_pad = ops.Cin_pad
    assert n_tpr * gs == Cin_pad

    packer = NunchakuWeightPacker(bits=4, warp_n=warp_n)
    # Signed codes Q_s ∈ {-8..7} as int32 (n, k); the packer's `&0xF` keeps the
    # two's-complement nibble the kernel's signed int4 MMA reads.
    Qs_2d = ops.Q_s.reshape(out_f, Cin_pad).to(torch.int32).contiguous()
    Qpacked = packer.pack_weight(Qs_2d)

    # wmscale slot = s·t_g (stock Nunchaku weight scale). FP16 per P0.3.
    wmscale_dense = (ops.s * ops.t_g).to(torch.float16).contiguous()   # (out_f, n_tpr)
    wmscales = packer.pack_scale(wmscale_dense.clone(), group_size=gs)

    # Correction C_jg = s·t·z' computed in FP32, stored per c_dtype.
    C = (ops.s.to(torch.float32) * ops.t_g.to(torch.float32)
         * ops.z_signed.to(torch.float32)).to(c_dtype).contiguous()    # (out_f, n_tpr)

    # Carry the FP-weight column sum through unchanged (DECISION 1): it is the exact
    # mean-pathway operand, computed in prepare_weight from the ORIGINAL fp32 weight in
    # the unrotated basis over the real Cin. NEVER derive it from the dequantized Ŵ.
    colsum = getattr(pack, "colsum", None)
    if colsum is not None:
        colsum = colsum.detach().to(torch.float32).contiguous()

    meta = {
        "out_features": out_f,
        "Cin_pad": Cin_pad,
        "Cin_orig": getattr(pack, "Cin_orig", Cin_pad),
        "group_size": gs,
        "n_tpr": n_tpr,
        "warp_n": warp_n,
        "scheme": scheme,              # rotation basis: "zero-pad" (legacy) | "kronecker"
        "wmscale_dtype": "float16",
        "c_dtype": str(c_dtype),
        "qpacked_dtype": "int8",
        "has_colsum": colsum is not None,
    }
    return PackedWeights(Qpacked=Qpacked, wmscales=wmscales, C=C,
                         wmscale_dense=wmscale_dense, meta=meta, colsum=colsum)


def unpack_weight_codes(Qpacked: Tensor, out_f: int, Cin_pad: int,
                        *, warp_n: int = _WARP_N) -> Tensor:
    """Invert NunchakuWeightPacker.pack_weight -> signed codes Q_s (out_f, Cin_pad).

    Used by the P1.2 round-trip test. Reverses the int4 nibble packing and the MMA
    swizzle permute, then sign-extends the 4-bit two's-complement nibble back to
    {-8..7}.
    """
    p = NunchakuWeightPacker(bits=4, warp_n=warp_n)
    n, k = out_f, Cin_pad
    n_tiles, k_tiles = n // p.mem_n, k // p.mem_k
    # int8 (n, k/2) -> int32 (n, k/8) -> the 9-axis post-sum contiguous shape.
    w32 = Qpacked.contiguous().view(torch.int32)
    nine = w32.reshape(
        n_tiles, k_tiles, p.num_k_packs, p.num_n_packs,
        p.num_n_lanes, p.num_k_lanes, p.n_pack_size, p.k_pack_size, p.reg_n,
    )
    # Restore reg_k (8 nibbles per int32): nibble c = (w >> 4c) & 0xF.
    shift = torch.arange(0, 32, 4, dtype=torch.int32, device=Qpacked.device)
    codes = (nine.unsqueeze(-1) >> shift) & 0xF          # 10 axes, last = reg_k(8)
    # Invert the forward permute (0,5,6,1,3,8,2,7,4,9).
    inv_perm = [0, 3, 6, 4, 8, 1, 2, 7, 5, 9]
    codes = codes.permute(*inv_perm).contiguous()
    codes = codes.reshape(n, k)
    # 4-bit two's-complement nibble {0..15} -> signed {-8..7}.
    return torch.where(codes >= 8, codes - 16, codes).to(torch.int64)


def unpack_scale(packed: Tensor, out_f: int, n_tpr: int,
                 *, warp_n: int = _WARP_N) -> Tensor:
    """Invert NunchakuWeightPacker.pack_scale (regular, non-micro) -> (out_f, n_tpr)."""
    n = out_f
    s_pack_size = min(max(warp_n // 32, 2), 8)
    num_s_lanes = min(32, warp_n // s_pack_size)
    num_s_packs = warp_n // (s_pack_size * num_s_lanes)
    # Forward shape pre-permute: (A, B=num_s_packs, C=num_s_lanes//4, D=s_pack_size//2,
    #                             E=4, F=2, G=n_tpr); permute (0,6,1,2,4,3,5).
    A = n // warp_n
    B, Cc, D, E, F, G = num_s_packs, num_s_lanes // 4, s_pack_size // 2, 4, 2, n_tpr
    permuted = packed.reshape(A, G, B, Cc, E, D, F)
    inv_perm = [0, 2, 3, 5, 4, 6, 1]                     # inverse of (0,6,1,2,4,3,5)
    scale = permuted.permute(*inv_perm).contiguous().reshape(n, n_tpr)
    return scale


# --------------------------------------------------------------------------- #
# Activation-side packing (reference of the online path, P2)
# --------------------------------------------------------------------------- #
@dataclass
class PackedActivations:
    """Per-forward activation operands (reference of what the kernel emits online)."""

    Xq_packed: Tensor    # int32 (T, Cin_pad/8) — 8 signed int4 codes / uint32
    a: Tensor            # fp16 (T, 1) — per-token activation scale (ascale)
    P: Tensor            # int16 (T, n_tpr) — group sums P_ig = Σ_{k∈g} X^q_ik
    Xq_codes: Tensor     # int (T, Cin_pad) — the unpacked codes (oracle reference)
    meta: dict
    # Per-token centering (Contract v3, S6): the per-token mean removed PRE-rotation
    # over the real Cin, fp32 (T, 1). None when cfg.center_act_mean is off. Restored
    # downstream as the rank-1 x̄ ⊗ colsum(W_fp).
    x_bar: Optional[Tensor] = None


def pack_activations_ref(X: Tensor, cfg, *, group_size: int = 64,
                         rotate_impl: str = "fwht",
                         rotate_dtype: Optional[torch.dtype] = None) -> PackedActivations:
    """Reference activation packing (plan P2.1): the locked oracle quant path
    (FWHT zero-pad -> percentile σ -> a_i -> round/clamp ±7), packed to the int4
    element layout, with the group sums P emitted in the SAME pass.

    `cfg` is an ActQuantConfig; `group_size` is the weight K-group (64), used only to
    form the per-group sums P. Scope note (plan P2): the deployed fork quantizes
    activations ONLINE (the kernel owns the per-(INSN_M,INSN_K) warp swizzle of
    `packed_act_t`), so the load-bearing NEW artifacts here are `a` and `P`. This
    function packs the int4 codes at the element granularity the kernel's
    `packed_act_t` uses (8 signed int4 / uint32, `&0xF`), validated bit-exactly by
    round-trip; the full warp swizzle is confirmed at Step-3 bring-up.
    """
    x2d = X.reshape(-1, X.shape[-1])
    rotation = getattr(cfg, "rotation", "pad")
    center_mean = bool(getattr(cfg, "center_act_mean", False))
    _, diag = quantize_act(
        x2d,
        b=cfg.act_bits,
        gamma_star=cfg.gamma_star,
        estimator=cfg.scale_estimator,
        rotate_act=cfg.rotate_act,
        percentile_p=cfg.percentile_p,
        huber_c=cfg.huber_c,
        return_diag=True,
        instrument=getattr(cfg, "instrument", False),  # §6 floats unused here (opt-B/B1)
        return_rotated=True,
        center_mean=center_mean,
        rotation=rotation,
        rotate_impl=rotate_impl,
        rotate_dtype=rotate_dtype,
    )
    a = diag["a"].to(torch.float16)                       # (T, 1) fp16 (ascale, P0.3)
    Xq = diag["Xq_codes"]                                 # (T, Cin_pad) ±7 codes
    # Per-token mean removed pre-rotation (S6); restored downstream as x̄ ⊗ colsum(W_fp).
    x_bar = diag.get("x_bar") if center_mean else None    # (T, 1) fp32 or None
    T, Cin_pad = Xq.shape
    gs = int(group_size)
    assert Cin_pad % gs == 0, f"Cin_pad {Cin_pad} not divisible by group_size {gs}"
    n_tpr = Cin_pad // gs

    Xq_i = Xq.to(torch.int32)
    Xq_packed = _pack_int4_codes(Xq_i)                    # (T, Cin_pad/8) int32

    # P_ig = Σ_{k∈g} X^q_ik  in the same pass. |P| <= 64*7 = 448 < 2^15 -> int16.
    P = Xq_i.reshape(T, n_tpr, gs).sum(dim=-1).to(torch.int16)

    scheme_tag = "kronecker" if rotation == "kronecker" else "zero-pad"
    meta = {"T": T, "Cin_pad": Cin_pad, "group_size": gs, "n_tpr": n_tpr,
            "a_dtype": "float16", "p_dtype": "int16", "scheme": scheme_tag,
            "center_mean": center_mean}
    return PackedActivations(Xq_packed=Xq_packed, a=a, P=P, Xq_codes=Xq_i, meta=meta,
                             x_bar=x_bar)


def _pack_int4_codes(codes: Tensor) -> Tensor:
    """Pack signed int4 codes (..., K) -> int32 (..., K/8), 8 nibbles/word, `&0xF`."""
    *lead, K = codes.shape
    assert K % 8 == 0, f"K={K} must be divisible by 8 for int4 packing"
    c = codes.reshape(*lead, K // 8, 8).bitwise_and(0xF)
    shift = torch.arange(0, 32, 4, dtype=torch.int32, device=codes.device)
    return (c << shift).sum(dim=-1, dtype=torch.int32)


def unpack_int4_codes(packed: Tensor, K: int) -> Tensor:
    """Invert `_pack_int4_codes` -> signed codes (..., K)."""
    shift = torch.arange(0, 32, 4, dtype=torch.int32, device=packed.device)
    c = (packed.unsqueeze(-1) >> shift) & 0xF             # (..., K/8, 8)
    *lead, _, _ = c.shape
    c = c.reshape(*lead, K)
    return torch.where(c >= 8, c - 16, c).to(torch.int64)


# --------------------------------------------------------------------------- #
# Restructured-epilogue identity (the P3.1 gate)
# --------------------------------------------------------------------------- #
def restructured_epilogue(
    Xq: Tensor,
    a: Tensor,
    Qs: Tensor,
    wmscale: Tensor,
    C: Tensor,
    P: Tensor,
    *,
    group_size: int,
    out_dtype: torch.dtype = torch.float16,
    accum_dtype: torch.dtype = torch.float32,
    corr_dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Compute the fork's restructured epilogue from (un)packed buffers (plan P3.1):

        main_io = Σ_g wmscale_og · (Σ_{k∈g} Xq_ik · Q^s_ok)
        Y_io    = a_i · ( main_io − (P Cᵀ)_io )

    The two int reductions (Σ Xq·Q_s per group, and P) are exact int32; the FP
    epilogue runs in `accum_dtype`. `corr_dtype` is the STORAGE dtype of the
    correction C in the side-GEMM `P Cᵀ` (P3.3 compares fp16 vs fp32): C is rounded
    to `corr_dtype`, then the MAC accumulates in `accum_dtype` (the real kernel
    behaviour: fp16 HBM operands, fp32 accumulator). Returns (T, out_f) in `out_dtype`.
    """
    dev = Xq.device
    T = Xq.shape[0]
    out_f, n_tpr, gs = Qs.shape
    assert gs == group_size and Xq.shape[1] == n_tpr * gs
    Xq_g = Xq.reshape(T, n_tpr, gs).to(torch.float64)
    Qs64 = Qs.to(torch.float64).to(dev)
    wm = wmscale.to(accum_dtype).to(dev)                  # (out_f, n_tpr)
    a_f = a.reshape(T, 1).to(accum_dtype).to(dev)

    # --- main term (stock Nunchaku wmscale-weighted group accumulation) ---
    main = torch.zeros(T, out_f, dtype=accum_dtype, device=dev)
    for g in range(n_tpr):
        Ig = (Xq_g[:, g, :] @ Qs64[:, g, :].t()).to(accum_dtype)   # (T, out) exact
        main = main + Ig * wm[:, g].unsqueeze(0)

    # --- correction side-GEMM  (P Cᵀ), dense (T×G)(G×N). C rounded to its storage
    # dtype; P is exact (|P|<=448 is exact in fp16); MAC accumulates in accum_dtype
    # (avoids an fp16 matmul that CPU does not support, and matches the kernel).
    Pc = P.to(accum_dtype).to(dev)                        # (T, n_tpr) exact
    Cc = C.to(corr_dtype).to(accum_dtype).to(dev)         # (out_f, n_tpr) fp16-rounded
    corr = Pc @ Cc.t()                                    # (T, out_f) fp32 accumulate

    Y = a_f * (main - corr)
    return Y.to(out_dtype)
