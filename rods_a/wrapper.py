"""
RoDS-A layer wrapper (Phase 1 scaffolding).

`RoDSActLinear` wraps an existing `nn.Linear`, fake-quantizes the *input
activation* per-token (rods_a.quantize_act) and runs the original FP16 weight
matmul unchanged. Weights are never quantized or rotated this phase.

`select_act_targets` reuses the canonical RoDS layer filter
(`sinq_diffusion._is_quant_target` + per-model `ignore_layers` from
`utils.model_config`), and additionally keeps cross-attn K/V and adaLN linears
in FP16 per doc section 5 Phase 1 ("keep adaLN-linear and cross-attn K/V at
16-bit initially"). The `in_features % group_size` skip from the weight artifact
path is honoured too.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from core.layer_filter import is_quant_target
from core.model_config import get_model_config
from rods_a.constants import gamma_star as _gamma_star_lookup
from rods_a.quantize_act import quantize_act
from rods_a import profiling as _profiling


# Cross-attn K/V leaf-suffixes kept in FP16 initially (PixArt attn2). FLUX uses
# joint attention (no attn2) so this is a no-op there.
_CROSS_ATTN_KV_SUFFIXES = (".attn2.to_k", ".attn2.to_v")


# --- Lazy INT4 backend handles (Step 5). Loaded on first int4 forward so the
#     fakequant path keeps working in environments without the pinned kernel. ---
_INT4_OPS = None                 # (gemm, quantize) from the pinned Nunchaku wheel
_ASCALES_CACHE: Dict[tuple, "torch.Tensor"] = {}  # (M_pad, K, device) -> ones ascales


def _int4_ops():
    """Load + cache the pinned-wheel gemm + device quantizer (idempotent). Asserts
    the pinned kernel version (deviation is a STOP, not a fix)."""
    global _INT4_OPS
    if _INT4_OPS is None:
        from rods_a.int4_gemm import (
            assert_pinned_version,
            load_nunchaku_ops,
            load_nunchaku_quantize,
        )
        assert_pinned_version()
        _, gemm = load_nunchaku_ops()
        quantize = load_nunchaku_quantize()
        _INT4_OPS = (gemm, quantize)
    return _INT4_OPS


def _ones_ascales(quantize, m_pad: int, k: int, device) -> "torch.Tensor":
    """All-ones `ascales` matching the device quantizer's exact layout for (m_pad, k);
    cached (the buffer is all-ones, so layer-independent for a given shape)."""
    from rods_a.int4_gemm import ones_ascales_like_device
    key = (int(m_pad), int(k), str(device))
    a = _ASCALES_CACHE.get(key)
    if a is None:
        a = ones_ascales_like_device(quantize, m_pad, k, device=device)
        _ASCALES_CACHE[key] = a
    return a


# opt-B/B3: one shared CUDA-graph memory pool across ALL captured per-layer graphs.
# Layers replay sequentially (each clones its output before the next replays), so they
# can reuse pool memory — resident static-buffer cost stays ~max-over-shape-classes
# instead of 224x (B0.4). Mirrors what torch.cuda.make_graphed_callables does internally.
_GRAPH_POOL = None
_GRAPH_WARMUP = 3  # eager passes before capture: settle autotuning + populate caches


def _graph_pool():
    global _GRAPH_POOL
    if _GRAPH_POOL is None:
        _GRAPH_POOL = torch.cuda.graph_pool_handle()
    return _GRAPH_POOL


@dataclass
class ActQuantConfig:
    """Activation-quant knobs (mirrors configs/quant/rods_a.yaml)."""
    act_bits: int = 4
    scale_estimator: str = "percentile"
    clip_family: str = "gaussian"
    huber_c: float = 1.345
    percentile_p: float = 0.999
    rotate_act: bool = True
    # Rotation scheme: "pad" (zero-pad pow2) | "kronecker" (K = Cin) | "factored"
    # (explicit per-width H_p (x) H_q via rotation_factorization; installer-resolved,
    # never reaches the fwht string dispatch).
    rotation: str = "pad"
    center_act_mean: bool = False  # per-token mean-centering (DC-blowup fix); OFF by default
    # Compute backend for the composed W4A4 linear (RoDS-A Step 5):
    #   "fakequant" (default) -- the host fp matmul against the dequantized W̃_hat;
    #   "int4"               -- the real Nunchaku INT4 W4A4 GEMM (rods_a.int4_gemm).
    backend: str = "fakequant"  # "fakequant" | "int4"
    # Arithmetic the fakequant path emulates (only meaningful when backend=fakequant):
    #   "reference" -- X̃_hat @ W̃_hatᵀ (cached dequant weight);
    #   "kernel"    -- the kernel-semantics oracle (rods_a.kernel_semantics), the
    #                  exact fp32-accum reference the int4 backend is compared against.
    semantics: str = "reference"  # "reference" | "kernel"
    # Activation packing path for the int4 backend (Step 4):
    #   "reference" (default) -- pack_activations_kernel (rebuilds the swizzle map);
    #   "fast"                -- pack_activations_kernel_fast (cached reindex, BIT-
    #                            IDENTICAL output; opt-A, gated by scripts/fast_pack_gate.py).
    act_path: str = "reference"  # "reference" | "fast"
    # §6 analysis float diagnostics (kurtosis_excess, residual_channel_mean_*,
    # clip_fraction) in quantize_act. Each is a host-sync float() that stalls the
    # pipeline and blocks CUDA-graph capture (opt-B/B0.2: 4 DtoH/layer, ~0.9-1.7 ms).
    # OFF by default in deployment; turn ON only for analysis runs. Decoupled from the
    # operand tensors a/Xq_codes/x_bar (those are always available, device-resident).
    instrument: bool = False  # compute the analysis float diagnostics (host syncs)
    # opt-B/B3: capture the whole per-layer int4 forward (quantize + pack + GEMM +
    # epilogue) as a CUDA graph and replay it, amortizing the ~51 launches/layer to one
    # replay (the path is launch-bound, B0). OFF by default; only valid on the int4
    # backend with instrument=off and no shadow/SQNR capture (those host-sync, which
    # blocks capture). Bit-identical by construction (replay = identical kernels),
    # gated by scripts/graph_capture_gate.py.
    graphs: bool = False  # capture+replay the per-layer int4 forward as a CUDA graph
    # Host epilogue realization for the int4 backend (opt-C, Contract §S8):
    #   "reference" (default) -- the validated 3-fp32-pass chain (side-GEMM P·Cᵀ, the
    #                            a·(main−corr) elementwise, the +x̄·colsum rank-1); the
    #                            opt-A/B bit-identical base.
    #   "fused"               -- fold the rank-1 into the side-GEMM via C̃=[C,−colsum],
    #                            P̃=[P, x̄/a] and realize a⊙(main−P̃C̃ᵀ) as addmm+mul_,
    #                            collapsing 3 passes → side-GEMM + 1. NOT bit-identical to
    #                            "reference" (fp32 summation reordered); gated against the
    #                            ORACLE at the I2 envelope (median_rel ≤ 1.5e-3), NOT vs
    #                            the reference path. Gated by scripts/rods_a_int4_fused_gate.py.
    epilogue: str = "reference"  # "reference" | "fused"
    # U2.2 decomposition: False => the activation passes through FP (centering +
    # rotation unchanged, NO scale/round/clamp) while the weight stays quantized
    # ("W4A16-on-W̃"). fakequant+reference ONLY (codes-domain paths need Xq).
    # Default ON => bit-identical legacy behavior.
    act_quant_enabled: bool = True
    # U4.B: ±1-diagonal seed for rotation="kronecker_d" (RHT proper). Ignored by
    # pad/kronecker. The SAME seed drives both the activation rotate and the weight
    # refit (install_w4a4 -> prepare_weight), so X̃ W̃ᵀ == X Wᵀ holds.
    rotation_seed: int = 0
    # Factored rotation (rotation="factored"): explicit per-width Kronecker
    # factorization {Cin: (p, q)} rotating at width p*q (>= Cin: minimal-pad
    # candidates pad first). rotation_pad_to {Cin: p*q} is redundant but documents
    # the padding; when present it must match. Widths ABSENT from the map fall
    # back bitwise to the pad scheme. Both None by default: every existing
    # rotation scheme ignores them (bit-identical legacy behavior).
    rotation_factorization: Optional[Dict[int, Tuple]] = None  # values: int order | variant token
    rotation_pad_to: Optional[Dict[int, int]] = None
    # Reserved: sweep the H_q (x) H_p ordering axis later; must stay False.
    rotation_factor_order_swap: bool = False

    @property
    def gamma_star(self) -> float:
        return _gamma_star_lookup(self.act_bits, self.clip_family)

    @classmethod
    def from_mapping(cls, m) -> "ActQuantConfig":
        """Build from a dict / OmegaConf node, ignoring unrelated keys."""
        get = (lambda k, d: m.get(k, d)) if hasattr(m, "get") else (lambda k, d: getattr(m, k, d))

        def _width_map(v, pair: bool):
            # YAML/OmegaConf carries the width keys as (quoted) strings and the
            # pairs as lists; normalize to {int: (factor, factor)} / {int: int}.
            # A factor is an int order or a variant token like "p24" (Paley-I
            # atom — a different equivalence class than the composed order).
            # Empty map == absent (the base factored yaml ships `{}`).
            if v is None:
                return None
            items = {int(k): v[k] for k in v.keys()} if hasattr(v, "keys") else dict(v)
            if not items:
                return None

            def _factor(f):
                return int(f) if not isinstance(f, str) or f.isdigit() else str(f)

            if pair:
                return {int(k): (_factor(t[0]), _factor(t[1])) for k, t in items.items()}
            return {int(k): int(t) for k, t in items.items()}

        return cls(
            act_bits=int(get("act_bits", 4)),
            scale_estimator=str(get("scale_estimator", "percentile")),
            clip_family=str(get("clip_family", "gaussian")),
            huber_c=float(get("huber_c", 1.345)),
            percentile_p=float(get("percentile_p", 0.999)),
            rotate_act=bool(get("rotate_act", True)),
            rotation=str(get("rotation", "pad")),
            center_act_mean=bool(get("center_act_mean", False)),
            backend=str(get("backend", "fakequant")),
            semantics=str(get("semantics", "reference")),
            act_path=str(get("act_path", "reference")),
            instrument=bool(get("instrument", False)),
            graphs=bool(get("graphs", False)),
            epilogue=str(get("epilogue", "reference")),
            act_quant_enabled=bool(get("act_quant_enabled", True)),
            rotation_seed=int(get("rotation_seed", 0)),
            rotation_factorization=_width_map(get("rotation_factorization", None), pair=True),
            rotation_pad_to=_width_map(get("rotation_pad_to", None), pair=False),
            rotation_factor_order_swap=bool(get("rotation_factor_order_swap", False)),
        )


class RoDSActLinear(nn.Module):
    """Wraps an nn.Linear; fake-quantizes the input activation, FP16 weights."""

    def __init__(self, linear: nn.Linear, cfg: ActQuantConfig):
        super().__init__()
        self.linear = linear  # original module, weights untouched (FP16)
        self.cfg = cfg
        self.in_features = linear.in_features
        self.out_features = linear.out_features

    def forward(self, x: Tensor) -> Tensor:
        cfg = self.cfg
        orig_shape = x.shape
        x2d = x.reshape(-1, orig_shape[-1])
        x_hat, _ = quantize_act(
            x2d,
            b=cfg.act_bits,
            gamma_star=cfg.gamma_star,
            estimator=cfg.scale_estimator,
            rotate_act=cfg.rotate_act,
            percentile_p=cfg.percentile_p,
            huber_c=cfg.huber_c,
            rotation=cfg.rotation,
            return_diag=False,
        )
        x_hat = x_hat.reshape(orig_shape)
        return self.linear(x_hat)

    def extra_repr(self) -> str:
        c = self.cfg
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"act_bits={c.act_bits}, est={c.scale_estimator}, "
            f"family={c.clip_family}, rotate={c.rotate_act}"
        )


# opt-C a-positivity guard (§S8): x̄/a needs a>0. The percentile estimator can return
# a=0 for a pathological all-zero token; clamp ONLY in the division (the multiply-back by
# the true `a` keeps the value exact for a>0, and x̄=0 for an all-zero token so the rank-1
# contribution is 0). Far below any real per-token activation scale (>1e-4 in practice).
_EPILOGUE_EPS = 1e-12


def fused_epilogue_core(main: Tensor, P: Tensor, a: Tensor, C: Tensor, *,
                        x_bar: Optional[Tensor] = None,
                        colsum: Optional[Tensor] = None,
                        C_aug: Optional[Tensor] = None,
                        eps: float = _EPILOGUE_EPS) -> Tensor:
    """opt-C fused epilogue math (Contract §S8) — the SINGLE SOURCE OF TRUTH shared by
    RoDSAW4A4Linear._epilogue_fused and scripts/rods_a_int4_fused_gate.py.

    Realizes Y = a ⊙ (main − P̃·C̃ᵀ), fp16, with the centering rank-1 folded into the
    side-GEMM: C̃ = [C, −colsum], P̃ = [P, x̄/a]. The chain collapses from 3 fp32 (T×out)
    passes to one addmm (matmul fused with the subtract via beta/alpha) + one in-place mul.
    NOT bit-identical to the reference epilogue (fp32 summation is reordered) — gate the
    output against the ORACLE at the I2 envelope, never vs the reference path. None of
    {main, P, a, x̄} is mutated (all promotions are out-of-place; in-place ops touch only
    the local `core` copy). Centering is active iff `x_bar` is not None; pass a precomputed
    `C_aug` (deploy) or `colsum` to build it inline (gate)."""
    af = a.float()                                          # (T, 1) fp32, out-of-place
    core = main.float()                                    # (T, out_f) fp32 copy
    if x_bar is not None:
        if C_aug is None:
            C_aug = torch.cat([C.float(), -colsum.float().reshape(-1, 1)], dim=1)
        P_aug = torch.cat(
            [P.float(), x_bar.float() / af.clamp_min(eps)], dim=1
        )                                                  # (T, n_tpr+1) fp32
        core.addmm_(P_aug, C_aug.t(), beta=1.0, alpha=-1.0)  # main − P̃·C̃ᵀ
    else:
        core.addmm_(P.float(), C.float().t(), beta=1.0, alpha=-1.0)  # main − P·Cᵀ
    core.mul_(af)                                          # a ⊙ (...)
    return core.to(torch.float16)


class RoDSAW4A4Linear(nn.Module):
    """Composed W4A4 fake-quant linear (RoDS-A Phase 2, doc 3.4 RODS_A_LINEAR).

    Holds a rotated, grouped-RoDS-re-fit weight (t absorbed; `prepare_weight`)
    and fake-quantizes the input activation per-token in the *rotated* basis,
    then multiplies against the rotated dequantized weight W̃_hat. Equivalent to
    the folded epilogue form Y_ij = a_i s_j Σ_k Q^X_ik (Q^W-z)_jk, with no online
    scale on the contraction axis k.

    Fake-quant only: W̃_hat is dequantized to FP once and cached; the real
    INT4×INT4 kernel is Phase 4.
    """

    def __init__(self, pack, cfg: ActQuantConfig, *, fp_weight: Optional[Tensor] = None,
                 center_mean: bool = False, matmul_dtype: torch.dtype = torch.float16,
                 backend: Optional[str] = None, shadow: bool = False,
                 layer_name: str = "", role: str = "other",
                 device: Optional[str] = None,
                 w_hat_override: Optional[Tensor] = None,
                 rotate_override: Optional[Callable] = None,
                 unrotate_override: Optional[Callable] = None,
                 capture_dtype: torch.dtype = torch.float32):
        super().__init__()
        self.cfg = cfg
        self.pack = pack
        self.in_features = pack.Cin_orig
        self.out_features = pack.out_features
        self.Cin_pad = pack.Cin_pad
        self.rotated = pack.rotated
        self.group_size = int(pack.meta["group_size"])
        self.n_tpr = pack.Cin_pad // self.group_size
        # Backend: "fakequant" (host fp matmul) | "int4" (real Nunchaku W4A4 GEMM).
        self.backend = str(backend if backend is not None else getattr(cfg, "backend", "fakequant"))
        # Emulated arithmetic for the fakequant path: "reference" (X̃_hat @ W̃_hatᵀ) |
        # "kernel" (the kernel-semantics oracle = the int4 backend's exact reference).
        self.semantics = str(getattr(cfg, "semantics", "reference"))
        self.shadow = bool(shadow)
        self.layer_name = layer_name
        self.role = role
        # Dtype of the cached dequant weight + the X̃_hat @ W̃_hatᵀ matmul. fp16 by
        # default: it matches what the deployed (fp16 / int32->fp16) kernel does and
        # uses tensor cores, so it is both faster and more representative than fp32.
        # Set fp32 only to isolate W4A4 quant error from fp16 matmul rounding.
        self.matmul_dtype = matmul_dtype
        # Per-token mean-centering: route the DC/mean part of the activation through
        # the FP colsum (exact) so it never inflates the per-token quant scale.
        self.center_mean = bool(center_mean)
        if self.center_mean:
            if pack.colsum is None:
                raise ValueError(
                    "center_mean=True requires pack.colsum (prepare_weight must "
                    "precompute the FP colsum)."
                )
            self.register_buffer("colsum", pack.colsum.detach().to(torch.float32),
                                 persistent=False)

        # Factored-rotation overrides: the explicit (rotate, unrotate) pair the
        # installer resolved for this layer's width. fakequant+reference only —
        # the codes-domain paths (int4 packing, kernel semantics) assume the
        # pad/kron layouts. None => the string-dispatched scheme, byte-identical.
        self._rotate_override = rotate_override
        self._unrotate_override = unrotate_override
        if (rotate_override is not None) != (unrotate_override is not None):
            raise ValueError("rotate_override and unrotate_override must be set together")
        if rotate_override is not None and not (
                self.backend == "fakequant" and self.semantics == "reference"):
            raise ValueError("rotation overrides (factored basis) require "
                             "backend='fakequant' + semantics='reference'.")

        # U2.2 decomposition guards: both passthroughs are fakequant+reference only
        # (the codes-domain int4 / kernel-semantics paths need Xq and W4 codes).
        if not bool(getattr(cfg, "act_quant_enabled", True)) and not (
                self.backend == "fakequant" and self.semantics == "reference"):
            raise ValueError("act_quant_enabled=False (W4A16-on-W̃ arm) requires "
                             "backend='fakequant' + semantics='reference'.")
        if w_hat_override is not None and not (
                self.backend == "fakequant" and self.semantics == "reference"):
            raise ValueError("w_hat_override (A4-only arm) requires "
                             "backend='fakequant' + semantics='reference'.")

        # --- which operand sets this instance needs ---
        need_w_hat = self.backend == "fakequant" and self.semantics == "reference"
        need_packed = self.backend == "int4"
        # kernel-semantics oracle operands: the int4 shadow path, OR the kernel
        # fake-quant variant.
        need_ops = (self.backend == "int4" and self.shadow) or \
                   (self.backend == "fakequant" and self.semantics == "kernel")

        if need_w_hat:
            # rotate_act must match the weight rotation (X̃ W̃ᵀ == X Wᵀ only if both
            # are rotated or neither is). The installer guarantees this.
            if w_hat_override is not None:
                # U2.2 A4-only arm: the rotated FP weight stands in for W̃_hat
                # (pack.W_q is None; nothing to dequantize).
                W_hat = w_hat_override.detach().to(torch.float32)
            else:
                from rods_a.prepare_weight import dequantize_prepared
                W_hat = dequantize_prepared(pack)    # (out_f, Cin_pad), dequant in FP32
            self.register_buffer("W_hat", W_hat.to(matmul_dtype), persistent=False)
        else:
            self.W_hat = None

        if need_packed:
            # Real INT4 operands: signed-int4 packed weight, fp16 wmscales, fp16
            # correction C = s·t·z' (fp16 == the kernel HBM dtype validated in Step-3).
            from rods_a.pack_operands import pack_weights
            # scheme is metadata-only (no layout effect); label it by the actual basis.
            self._pack_scheme = ("kronecker" if str(cfg.rotation).startswith("kronecker")
                                 else "zero-pad")
            pw = pack_weights(pack, c_dtype=torch.float16, scheme=self._pack_scheme)
            self.register_buffer("Qpacked", pw.Qpacked, persistent=False)
            self.register_buffer("wmscales", pw.wmscales, persistent=False)
            self.register_buffer("C", pw.C, persistent=False)
            # opt-C (§S8): pre-build the augmented side-GEMM matrix C̃ = [C, −colsum]
            # (out_f, n_tpr+1) fp32 so the centering rank-1 rides through the side-GEMM
            # (P̃ = [P, x̄/a]). Static, built once at install; fp32 to match the reference
            # path's fp32 side-GEMM precision. Only needed when centering — without it the
            # fused path uses plain C (no extra column).
            if self.center_mean:
                self.register_buffer(
                    "C_aug",
                    torch.cat([pw.C.to(torch.float32),
                               -self.colsum.reshape(-1, 1).to(torch.float32)], dim=1),
                    persistent=False,
                )  # (out_f, n_tpr+1) fp32
            else:
                self.C_aug = None

        if need_ops:
            from rods_a.kernel_semantics import to_kernel_operands
            ops = to_kernel_operands(pack)
            # Q_s ∈ {-8..7} fits int8 -> bound the shadow memory (~1 B/param vs the
            # int64 to_kernel_operands emits). oracle_matmul upcasts per group.
            self.register_buffer("ops_Qs", ops.Q_s.to(torch.int8), persistent=False)
            self.register_buffer("ops_s", ops.s.to(torch.float32), persistent=False)
            self.register_buffer("ops_zsigned", ops.z_signed.to(torch.float32), persistent=False)
            self.register_buffer("ops_tg", ops.t_g.to(torch.float32), persistent=False)

        bias = None if pack.bias is None else pack.bias.to(matmul_dtype)
        if bias is not None:
            self.register_buffer("bias", bias, persistent=False)
        else:
            self.bias = None

        # Move the operand buffers to the install device (the fits load on CPU; the
        # transformer they are spliced into already lives on `device`). Operands and
        # activations MUST share a device (documented gotcha) -- asserted in forward.
        if device is not None:
            self.to(device)

        # Optional real-activation per-layer SQNR capture (sanity gate): keep the
        # original FP weight and accumulate signal / noise power across forwards.
        # capture_dtype fp32 is the legacy default (bit-identical); fp16 halves the
        # resident reference-copy memory for very large models (FLUX) — an exact
        # cast for bf16 weights (mantissa always fits) as long as the values stay
        # inside fp16 range, which the amax guard asserts at install.
        self.capture_sqnr = fp_weight is not None
        if self.capture_sqnr:
            if capture_dtype == torch.float16:
                amax = float(fp_weight.detach().abs().amax())
                if amax >= 65504.0:
                    raise ValueError(
                        f"capture_dtype=float16 but |W|max={amax:.1f} exceeds fp16 "
                        f"range for layer {layer_name!r}; use float32 capture."
                    )
            self.register_buffer("fp_weight", fp_weight.detach().to(capture_dtype),
                                 persistent=False)
            self._sqnr_sig = 0.0
            self._sqnr_noise = 0.0

        # opt-B/B3: whole-layer CUDA-graph capture. Only valid on the int4 backend with
        # NO host syncs in the captured region -> requires instrument=off and no
        # shadow/SQNR (both host-sync). Lazily capture one graph per input shape (T is
        # fixed at a given resolution, so this is one graph in steady state) and replay.
        self.use_graphs = (
            bool(getattr(cfg, "graphs", False))
            and self.backend == "int4"
            and not self.shadow
            and not self.capture_sqnr
            and not bool(getattr(cfg, "instrument", False))
        )
        self._graph_cache: Dict[tuple, dict] = {}

    # ----------------------------------------------------------------- helpers
    def _kernel_operands(self):
        from rods_a.kernel_semantics import KernelOperands
        return KernelOperands(
            Q_s=self.ops_Qs, s=self.ops_s, z_signed=self.ops_zsigned, t_g=self.ops_tg,
            group_size=self.group_size, out_features=self.out_features,
            Cin_pad=self.Cin_pad,
        )

    def _int4_core(self, Xq: Tensor, a: Tensor,
                   x_bar: Optional[Tensor] = None) -> Tensor:
        """Real INT4 W4A4 GEMM + host epilogue Y = a·(main − P·Cᵀ) + x̄⊗colsum, pre-bias
        (T, out_f) fp16. `Xq` (T, Cin_pad) signed codes, `a` (T, 1). The centering
        rank-1 (Contract v3, S6) shares the host-correction fp32 precision path and is
        the exact FP-weight mean contribution (DECISION 1: self.colsum is the original
        FP-weight column sum, NOT a Ŵ row-sum), added unscaled by a. Mirrors
        int4_gemm_parity when center_mean is off."""
        from rods_a.int4_gemm import (gemm_w4a4_main, pack_activations_kernel,
                                         pack_activations_kernel_fast)
        dev = self.Qpacked.device
        assert Xq.device == dev, (
            f"int4 backend: activation codes on {Xq.device} != weights on {dev}"
        )
        T = Xq.shape[0]
        # opt-A: cached-reindex packer when act_path="fast" (bit-identical output).
        _pack = (pack_activations_kernel_fast
                 if getattr(self.cfg, "act_path", "reference") == "fast"
                 else pack_activations_kernel)
        # events-only sub-brackets: pack / GEMM / epilogue split of the enclosing
        # `int4_gemm` sync lump (a nested synchronize would corrupt its numbers).
        with _profiling.section("int4_pack", events_only=True):
            act = _pack(Xq)                                     # (M_pad, K//2) uint8
        gemm, quantize = _int4_ops()
        ascales = _ones_ascales(quantize, act.shape[0], self.Cin_pad, dev)
        with _profiling.section("int4_gemm_core", events_only=True):
            main = gemm_w4a4_main(act, self.Qpacked, self.wmscales, ascales,
                                  out_features=self.out_features, gemm=gemm)[:T]
        with _profiling.section("int4_epilogue", events_only=True):
            P = Xq.reshape(T, self.n_tpr, self.group_size).sum(-1)  # (T, n_tpr) int
            if getattr(self.cfg, "epilogue", "reference") == "fused":
                return self._epilogue_fused(main, P, a, x_bar)
            # reference epilogue (default): the validated 3-fp32-pass chain (opt-A/B base).
            corr = P.float() @ self.C.float().t()               # (T, out_f)
            core = a.float() * (main.float() - corr)            # (T, out_f) fp32
            if self.center_mean:
                core = core + x_bar.float() * self.colsum.float()  # rank-1 FP-weight mean
            return core.to(torch.float16)

    def _epilogue_fused(self, main: Tensor, P: Tensor, a: Tensor,
                        x_bar: Optional[Tensor]) -> Tensor:
        """opt-C fused epilogue (Contract §S8); delegates to the shared
        `fused_epilogue_core`. Centering uses the install-time C̃ (self.C_aug); centering
        off passes plain self.C. See the module function for the full contract."""
        if self.center_mean:
            return fused_epilogue_core(main, P, a, self.C, x_bar=x_bar, C_aug=self.C_aug)
        return fused_epilogue_core(main, P, a, self.C)

    def _oracle_core(self, Xq: Tensor, a: Tensor,
                     x_bar: Optional[Tensor] = None) -> Tensor:
        """Kernel-semantics oracle output (exact fp32-accum) + x̄⊗colsum, pre-bias
        (T, out_f) fp16. Scales rounded to the kernel's fp16 HBM dtype (apples-to-apples).
        With centering the rank-1 is added in fp32 before the final cast (S6)."""
        from rods_a.kernel_semantics import oracle_matmul
        if self.center_mean:
            Y = oracle_matmul(Xq, a, self._kernel_operands(),
                              out_dtype=torch.float32, scale_dtype=torch.float16)
            Y = Y + x_bar.float() * self.colsum.float()
            return Y.to(torch.float16)
        return oracle_matmul(Xq, a, self._kernel_operands(), scale_dtype=torch.float16)

    def _fixture_cell(self, Xq: Tensor, a: Tensor,
                      y_oracle_core: Optional[Tensor] = None,
                      x_bar: Optional[Tensor] = None) -> dict:
        """Self-contained reproducer for a live (layer, step) cell, matching the Step-3
        pack_fixtures schema so it drops straight into scripts/int4_gemm_parity.py.
        Used by the I2.3 conditional STOP to dump a violating live tensor. Carries the
        centering operands (x_bar, colsum) when center_mean is on (Contract v3)."""
        from rods_a import parity_probe
        if y_oracle_core is None:
            y_oracle_core = self._oracle_core(Xq, a, x_bar)
        T = Xq.shape[0]
        P = Xq.reshape(T, self.n_tpr, self.group_size).sum(-1)
        cell = {
            "layer": self.layer_name, "step": parity_probe.current_step(),
            "out_features": self.out_features, "Cin_pad": self.Cin_pad,
            "group_size": self.group_size, "n_tpr": self.n_tpr,
            "scheme": getattr(self, "_pack_scheme", "kronecker"),
            "Qpacked": self.Qpacked.detach().cpu(),
            "wmscales": self.wmscales.detach().cpu(),
            "C": self.C.detach().cpu(),
            "a": a.detach().to(torch.float16).cpu(),
            "P": P.detach().to(torch.int64).cpu(),
            "Xq_codes": Xq.detach().to(torch.int16).cpu(),
            "y_oracle_fp16": y_oracle_core.detach().cpu(),
            "center_mean": self.center_mean,
        }
        if self.center_mean:
            cell["x_bar"] = x_bar.detach().to(torch.float32).cpu()
            cell["colsum"] = self.colsum.detach().to(torch.float32).cpu()
        return cell

    def _quantize_codes(self, x2d: Tensor):
        """Run the locked per-token quant in the rotated basis; return (Xq codes, a,
        x_bar). x_bar (T,1) is the pre-rotation per-token mean when center_mean is on
        (Contract v3), else None -- restored downstream as the rank-1 x̄⊗colsum."""
        cfg = self.cfg
        _, diag = quantize_act(
            x2d,
            b=cfg.act_bits,
            gamma_star=cfg.gamma_star,
            estimator=cfg.scale_estimator,
            rotate_act=self.rotated,
            percentile_p=cfg.percentile_p,
            huber_c=cfg.huber_c,
            return_diag=True,                      # needed for the operand tensors
            instrument=cfg.instrument,             # §6 floats OFF in deployment (opt-B/B1)
            return_rotated=self.rotated,
            center_mean=self.center_mean,
            rotation=cfg.rotation,
            rotate_impl="dense",
            rotate_dtype=self.matmul_dtype,
            rotation_seed=int(getattr(cfg, "rotation_seed", 0)),
            rotate_override=self._rotate_override,
            unrotate_override=self._unrotate_override,
        )
        x_bar = diag.get("x_bar") if self.center_mean else None
        return diag["Xq_codes"], diag["a"], x_bar               # (T, Cin_pad), (T, 1), (T,1)|None

    # ------------------------------------------------------------- B3 CUDA graphs
    def _int4_core_from_x2d(self, x2d: Tensor) -> Tensor:
        """The full deployed int4 per-layer core: quantize (instrument off) -> int4 GEMM
        + epilogue -> (T, out_f) fp16, pre-bias. This is the region captured as a CUDA
        graph (no host syncs: B2 verified)."""
        Xq, a, x_bar = self._quantize_codes(x2d)
        a = a.to(torch.float16)
        return self._int4_core(Xq, a, x_bar)

    def _capture_graph(self, x2d: Tensor) -> dict:
        """Eager-warmup (settle autotuning + caches), then capture the per-layer core
        against a static input buffer. Shares the module graph pool (B0.4)."""
        static_in = x2d.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(_GRAPH_WARMUP):
                self._int4_core_from_x2d(static_in)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=_graph_pool()):
            static_out = self._int4_core_from_x2d(static_in)
        return {"graph": g, "in": static_in, "out": static_out}

    def _graphed_core(self, x2d: Tensor) -> Tensor:
        """Replay the captured per-layer graph for this input shape (capture on first
        sight). Returns a CLONE of the static output -- the buffer is overwritten by the
        next step's replay, so downstream must not alias it."""
        key = (tuple(x2d.shape), x2d.dtype, str(x2d.device))
        entry = self._graph_cache.get(key)
        if entry is None:
            entry = self._capture_graph(x2d)
            self._graph_cache[key] = entry
        entry["in"].copy_(x2d)
        entry["graph"].replay()
        return entry["out"].clone()

    def forward(self, x: Tensor) -> Tensor:
        cfg = self.cfg
        orig_shape = x.shape
        x2d = x.reshape(-1, orig_shape[-1]).to(torch.float32)

        if self.use_graphs and x2d.is_cuda:
            # B3 fast path: replay the whole-layer graph (bit-identical to eager; the
            # captured region has no host syncs -- B2). Bias + reshape stay eager.
            y_core = self._graphed_core(x2d)
            y = y_core if self.bias is None else (y_core + self.bias.to(y_core.dtype))
            return y.reshape(*orig_shape[:-1], self.out_features).to(x.dtype)

        if self.backend == "int4" or (self.backend == "fakequant" and self.semantics == "kernel"):
            # Code-domain paths: quantize once, then the int4 GEMM and/or the oracle.
            Xq, a, x_bar = self._quantize_codes(x2d)
            # a_i is stored fp16 in the deployed contract (P0.3); cast once so the int4
            # and oracle paths see IDENTICAL a (their gap is then purely the epilogue
            # floor, matching the Step-3 fixtures which stored a as fp16).
            a = a.to(torch.float16)
            if self.backend == "int4":
                with _profiling.section("int4_gemm"):
                    y_core = self._int4_core(Xq, a, x_bar)
                if self.shadow:
                    from rods_a import parity_probe
                    if parity_probe.recording():
                        y_oracle_core = self._oracle_core(Xq, a, x_bar)
                        stats = parity_probe.record(self.layer_name, self.role,
                                                    y_core, y_oracle_core,
                                                    Cin_pad=self.Cin_pad)
                        if (stats is not None and parity_probe.capture_armed()
                                and stats["median_rel"] > parity_probe.gate()):
                            parity_probe.stash_violation(
                                self._fixture_cell(Xq, a, y_oracle_core, x_bar))
            else:  # fakequant + kernel semantics
                with _profiling.section("kernel_oracle"):
                    y_core = self._oracle_core(Xq, a, x_bar)
            y = y_core if self.bias is None else (y_core + self.bias.to(y_core.dtype))
            y = y.reshape(*orig_shape[:-1], self.out_features)
            return y.to(x.dtype)

        # --- reference fake-quant: X̃_hat @ W̃_hatᵀ (legacy default) ---
        x_hat, diag = quantize_act(
            x2d,
            b=cfg.act_bits,
            gamma_star=cfg.gamma_star,
            estimator=cfg.scale_estimator,
            rotate_act=self.rotated,
            percentile_p=cfg.percentile_p,
            huber_c=cfg.huber_c,
            return_diag=self.center_mean,          # only x_bar needed here (centered)
            instrument=cfg.instrument,             # §6 floats OFF in deployment (opt-B/B1)
            return_rotated=self.rotated,
            center_mean=self.center_mean,
            rotation=cfg.rotation,
            # Fast path: single cached-Hadamard GEMM (tensor cores) instead of the
            # Python butterfly, rotation in the matmul dtype (fp16). Scale estimate +
            # round/clamp stay fp32 inside quantize_act.
            rotate_impl="dense",
            rotate_dtype=self.matmul_dtype,
            passthrough=not bool(getattr(cfg, "act_quant_enabled", True)),
            rotation_seed=int(getattr(cfg, "rotation_seed", 0)),
            rotate_override=self._rotate_override,
            unrotate_override=self._unrotate_override,
        )
        # x_hat: (T, Cin_pad) if rotated else (T, Cin_orig); W_hat matches.
        with _profiling.section("weight_matmul"):
            y = x_hat.to(self.W_hat.dtype) @ self.W_hat.t()
        if self.center_mean:
            # Exact mean contribution: x_bar (T,1) ⊗ colsum (out_f,) -> (T, out_f).
            with _profiling.section("mean_correction"):
                y = y + diag["x_bar"].to(y.dtype) * self.colsum.to(y.dtype)
        if self.bias is not None:
            y = y + self.bias
        if self.capture_sqnr:
            with torch.no_grad():
                # .to() is a no-op for the fp32 default (same tensor object), so
                # the legacy op sequence is bit-identical; fp16 capture runs the
                # reference matmul in fp16 (tensor cores, fp32 accumulate).
                y_fp = x2d.to(self.fp_weight.dtype) @ self.fp_weight.t()
                if self.bias is not None:
                    y_fp = y_fp + self.bias
                self._sqnr_sig += float(y_fp.double().pow(2).sum())
                self._sqnr_noise += float((y_fp - y).double().pow(2).sum())
        y = y.reshape(*orig_shape[:-1], self.out_features)
        return y.to(x.dtype)

    def sqnr_db(self) -> Optional[float]:
        """Accumulated real-activation composed-W4A4-vs-FP16 output SQNR (dB), or
        None if capture was not enabled / no forward has run yet."""
        import math as _math
        if not self.capture_sqnr or self._sqnr_noise <= 0.0:
            return None
        return 10.0 * _math.log10(max(self._sqnr_sig, 1e-30) / self._sqnr_noise)

    def extra_repr(self) -> str:
        c = self.cfg
        be = self.backend if self.backend != "fakequant" else f"fakequant/{self.semantics}"
        return (
            f"in={self.in_features}, out={self.out_features}, Cin_pad={self.Cin_pad}, "
            f"W4A{c.act_bits}, backend={be}, shadow={self.shadow}, "
            f"est={c.scale_estimator}, rotated={self.rotated}, "
            f"center_mean={self.center_mean}, rho_median={self.pack.rho_median:.3f}"
        )


def select_act_targets(
    transformer: nn.Module,
    model_type: str = "pixart",
    group_size: int = 64,
    extra_ignore_suffixes: Optional[List[str]] = None,
) -> Dict[str, nn.Module]:
    """Return {full_name: nn.Linear} for activation-quant target layers.

    Filter = canonical RoDS filter, restricted to nn.Linear, minus cross-attn
    K/V and minus layers whose in_features is not divisible by group_size.
    """
    ignore_leaves = set(get_model_config(model_type).ignore_layers)
    suffixes = tuple(_CROSS_ATTN_KV_SUFFIXES) + tuple(extra_ignore_suffixes or ())

    targets: Dict[str, nn.Module] = {}
    for name, mod in transformer.named_modules():
        if type(mod) is not nn.Linear:
            continue  # activation wrapper handles Linear only this phase
        if not is_quant_target(mod):
            continue
        leaf = name.split(".")[-1]
        if leaf in ignore_leaves:
            continue
        if any(name.endswith(s) for s in suffixes):
            continue
        if group_size > 0 and mod.in_features % group_size != 0:
            continue
        targets[name] = mod
    return targets


def _set_submodule(root: nn.Module, name: str, new_mod: nn.Module) -> None:
    parent = root
    parts = name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_mod)


def wrap_transformer(
    transformer: nn.Module,
    cfg: ActQuantConfig,
    model_type: str = "pixart",
    group_size: int = 64,
    verbose: bool = True,
) -> nn.Module:
    """Replace activation-quant target Linears with RoDSActLinear in place.

    Weights stay FP16. Returns the same transformer.
    """
    targets = select_act_targets(transformer, model_type=model_type, group_size=group_size)
    for name, mod in targets.items():
        _set_submodule(transformer, name, RoDSActLinear(mod, cfg))
    if verbose:
        print(
            f"[rods_a] Wrapped {len(targets)} nn.Linear layers with RoDSActLinear "
            f"(model_type={model_type}, act_bits={cfg.act_bits}, "
            f"est={cfg.scale_estimator}, rotate={cfg.rotate_act}). Weights remain FP16."
        )
    transformer.rods_a_wrapped = True
    return transformer
