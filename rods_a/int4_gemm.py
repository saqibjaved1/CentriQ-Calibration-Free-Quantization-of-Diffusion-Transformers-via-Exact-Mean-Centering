"""
RoDS-A Step 3 — real INT4 W4A4 GEMM bring-up against the kernel-semantics oracle.

The op-level call site wired into `RoDSActLinear` (Step 5). It drives Nunchaku's
prebuilt W4A4 GEMM (`ops.gemm_w4a4`, pinned commit 8f41840 / wheel
v1.3.0dev20260306) DIRECTLY — no forked kernel (G0 Outcome A) — and applies the
z/scale/bias correction on the host (G0.4):

    main_io = Σ_g s_jg · (Σ_{k∈g} X^q_ik Q^s_jk)            # the GEMM, ascales≡1
    Y[i,o]  = a_i · (main_io − (P Cᵀ)_io) + bias_o          # host epilogue

Setting `ascales ≡ 1` (a constant buffer, layout-invariant) and pulling the
per-token scale `a_i` into the host epilogue avoids any host ascale-swizzle mirror;
`wscales` uses the Step-2 byte-verified `pack_scale`.

Import note (mirrors scripts/pack_byteref.py): `import nunchaku` executes
`nunchaku/__init__.py` → `nunchaku.models` → diffusers/transformers, which the lean
`venvs/nunchaku-int4` does not install (wheel installed `--no-deps`). We therefore
locate the package dir via `find_spec` (no __init__ execution), stub the parent
packages, and import ONLY `nunchaku._C` (the compiled op) + `nunchaku.ops.gemm`.
"""

from __future__ import annotations

import importlib.machinery
import importlib.metadata
import importlib.util
import os
import sys
import types

import torch
import torch.nn.functional as F

# --- W4A4 tile geometry (gemm_base.cuh:36-94; verified G0.2). Mirrors the device
#     quantize_w4a4_act_kernel store index and the mma.m16n8k64 A-fragment tile. ---
_WARP_SIZE = 32
_NUM_WARPS = 8
_INSN_M = 16
_INSN_K = 64
_BLOCK_M = 256
_WARP_M = _BLOCK_M // _NUM_WARPS  # 32
_WARP_K = _INSN_K  # 64
_WARP_M_TILES = _WARP_M // _INSN_M  # 2
_NIB_PER_UINT4 = 32  # packed_act_t = uint4 = 16 bytes = 32 int4 nibbles (one lane)

_ACT_SWIZZLE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs", "int4_gemm", "act_swizzle.pt",
)
_ACT_PERM_CACHE: "torch.Tensor | None" = None

# The pinned commit 8f41840 ships as wheel/dist version "1.3.0.dev20260306"
# (PEP440-normalised from tag v1.3.0dev20260306), with a local segment
# "+cu12.8torch2.10". NOTE: the in-package `nunchaku/__version__.py` attribute is a
# COARSE "1.3.0dev" (no date) — it does NOT pin the commit. The date-stamped pin
# lives only in the distribution metadata, so we verify via importlib.metadata.
PINNED_VERSION = "1.3.0.dev20260306"  # tag v1.3.0dev20260306 == commit 8f41840


# --------------------------------------------------------------------------- #
# Stub-import of the Nunchaku op (no nunchaku/__init__ → no diffusers/transformers)
# --------------------------------------------------------------------------- #
_NUNCHAKU_ROOT_CACHE: "str | None" = None


def _nunchaku_root() -> str:
    """Filesystem dir of the installed `nunchaku` package WITHOUT executing its
    `__init__.py`. Cached on first resolution: once we stub `nunchaku` in
    sys.modules (to skip its heavy __init__), a later `find_spec` would return the
    stub's spec (empty submodule_search_locations), so we must not re-resolve."""
    global _NUNCHAKU_ROOT_CACHE
    if _NUNCHAKU_ROOT_CACHE is not None:
        return _NUNCHAKU_ROOT_CACHE
    # If already stubbed, the stub carries __path__ = [root] (set by _stub_pkg).
    mod = sys.modules.get("nunchaku")
    if mod is not None and getattr(mod, "__path__", None):
        _NUNCHAKU_ROOT_CACHE = list(mod.__path__)[0]
        return _NUNCHAKU_ROOT_CACHE
    spec = importlib.util.find_spec("nunchaku")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("nunchaku is not installed in the active environment")
    _NUNCHAKU_ROOT_CACHE = list(spec.submodule_search_locations)[0]
    return _NUNCHAKU_ROOT_CACHE


def _stub_pkg(name: str, path: str) -> None:
    """Register an empty parent package so submodule import skips its __init__."""
    if name in sys.modules and getattr(sys.modules[name], "__path__", None):
        return
    m = types.ModuleType(name)
    m.__path__ = [path]
    m.__package__ = name
    m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    sys.modules[name] = m


def nunchaku_version() -> str:
    """Distribution version of the installed nunchaku (e.g.
    '1.3.0.dev20260306+cu12.8torch2.10'). Reads dist metadata — no module import,
    and carries the date-stamped pin (unlike the coarse module __version__)."""
    return importlib.metadata.version("nunchaku")


def load_nunchaku_ops():
    """Return `(ops, svdq_gemm_w4a4_cuda)` from the installed wheel, loading only
    the compiled extension + the gemm wrapper (no model stack). Idempotent."""
    root = _nunchaku_root()
    _stub_pkg("nunchaku", root)
    _stub_pkg("nunchaku.ops", os.path.join(root, "ops"))
    # torch must be imported before the CUDA extension registers its ops.
    import torch  # noqa: F401

    from nunchaku._C import ops as _ops  # loads _C.cpython-*.so
    from nunchaku.ops.gemm import svdq_gemm_w4a4_cuda  # does `from .._C import ops`

    return _ops, svdq_gemm_w4a4_cuda


def load_nunchaku_quantize():
    """Return `svdq_quantize_w4a4_act_fuse_lora_cuda` (the device activation
    quantizer) via the same lean stub-import. Used by the swizzle-recovery probe
    (G2) to produce the reference `act` operand on-device. Idempotent."""
    root = _nunchaku_root()
    _stub_pkg("nunchaku", root)
    _stub_pkg("nunchaku.ops", os.path.join(root, "ops"))
    import torch  # noqa: F401

    from nunchaku.ops.quantize import svdq_quantize_w4a4_act_fuse_lora_cuda

    return svdq_quantize_w4a4_act_fuse_lora_cuda


def quantize_act_operand(quantize, x: torch.Tensor, *, pad_size: int = _BLOCK_M):
    """Run the device W4A4 activation quantizer on x (M,K) fp16/bf16 and return
    `(act uint8 (M_pad,K//2), ascales)`.

    The wrapper requires `lora_down` (it reads `.shape[1]` for the rank), so we pass
    a ZEROS rank-16 down-projection (16 = the kernel's rank alignment). The quantized
    `act` output is independent of LoRA — LoRA is a separate additive output we
    discard. The kernel ALWAYS loads `smooth_factor` (gemm_w4a4.cuh:959), so passing
    None segfaults; we pass a neutral all-ones smooth (identity). The op pads channels
    to N = ceilDiv(K,128)*128 and requires lora_down.shape[0]==N & smooth length N;
    all our K are multiples of 128 (N==K)."""
    M, K = x.shape
    assert K % 128 == 0, f"K={K} must be a multiple of 128 (BLOCK_N) for the quantizer"
    N = K
    lora_down = torch.zeros(N, 16, dtype=x.dtype, device=x.device)
    smooth = torch.ones(N, dtype=x.dtype, device=x.device)
    out, ascales, _ = quantize(x, lora_down=lora_down, smooth=smooth, fp4=False, pad_size=pad_size)
    return out, ascales


def assert_pinned_version() -> str:
    """Hard-check that the active kernel is the pinned commit. Any deviation
    raises (G1.2)."""
    v = nunchaku_version()
    base = v.split("+", 1)[0]  # strip local '+cu12.8torch2.10' segment
    if base != PINNED_VERSION:
        raise RuntimeError(
            f"nunchaku version {v!r} (base {base!r}) != pinned {PINNED_VERSION!r} "
            f"(commit 8f41840). Refusing to proceed — re-install the pinned wheel."
        )
    return v


# --------------------------------------------------------------------------- #
# Activation operand packing (mirror of quantize_w4a4_act_kernel's MMA swizzle)
# --------------------------------------------------------------------------- #
def load_act_swizzle(path: str | None = None) -> torch.Tensor:
    """Load the within-tile (16x64) activation swizzle permutation recovered by
    scripts/recover_act_swizzle.py (G2). Cached. perm[m,k] = nibble index in
    [0,1024) of code (m,k) within one (INSN_M x INSN_K) A-fragment tile."""
    global _ACT_PERM_CACHE
    if path is None and _ACT_PERM_CACHE is not None:
        return _ACT_PERM_CACHE
    p = path or _ACT_SWIZZLE_PATH
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"activation swizzle artifact not found at {p}. Run "
            f"scripts/recover_act_swizzle.py (launch/recover_act_swizzle.sh) first."
        )
    blob = torch.load(p, map_location="cpu")
    perm = blob["within_tile_perm"].to(torch.int64)
    assert tuple(perm.shape) == (_INSN_M, _INSN_K), f"bad perm shape {tuple(perm.shape)}"
    if path is None:
        _ACT_PERM_CACHE = perm
    return perm


def pack_activations_kernel(
    codes: torch.Tensor, *, perm: torch.Tensor | None = None, group_size: int = 64
) -> torch.Tensor:
    """Pack signed int4 activation CODES (M, K) into the device `act` operand
    (uint8 (M_pad, K//2)) in the exact `packed_act_t` MMA swizzle the GEMM consumes.

    M is zero-token-padded up to a multiple of BLOCK_M (256) — harmless (G0.2).
    `codes` are in [-7,7] (activations never use -8); negatives are stored as the
    two's-complement nibble (`& 0xF`), matching the device bit pattern.
    """
    if perm is None:
        perm = load_act_swizzle()
    assert group_size == _WARP_K, f"group_size must be {_WARP_K}, got {group_size}"
    codes = codes.to(torch.int64)
    M, K = codes.shape
    assert K % _WARP_K == 0, f"K={K} must be a multiple of {_WARP_K}"
    Mpad = ((M + _BLOCK_M - 1) // _BLOCK_M) * _BLOCK_M
    if Mpad != M:
        codes = F.pad(codes, (0, 0, 0, Mpad - M))  # zero tokens at the tail

    dev = codes.device
    perm = perm.to(dev)
    i = torch.arange(Mpad, device=dev).view(Mpad, 1)
    k = torch.arange(K, device=dev).view(1, K)
    bm = i // _BLOCK_M
    rem = i % _BLOCK_M
    warpId = rem // _WARP_M
    wrem = rem % _WARP_M
    tileId = wrem // _INSN_M
    m = wrem % _INSN_M  # (Mpad,1)
    bk = k // _WARP_K
    kk = k % _WARP_K  # (1,K)

    base_uint4 = (
        (((bm * (K // _WARP_K) + bk) * _NUM_WARPS + warpId) * _WARP_M_TILES + tileId)
        * _WARP_SIZE
    )  # (Mpad,K) broadcast
    nu = perm[m.expand(Mpad, K), kk.expand(Mpad, K)]  # within-tile nibble idx (Mpad,K)
    global_nibble = base_uint4 * _NIB_PER_UINT4 + nu
    byte_idx = global_nibble // 2
    half = global_nibble % 2

    nib = codes & 0xF
    contrib = nib << (4 * half)
    buf = torch.zeros(Mpad * (K // 2), dtype=torch.int64, device=dev)
    # Each (byte, half) is unique (perm bijection per tile, tiles disjoint); the two
    # halves of a byte land on the same byte_idx with disjoint bit-shifts -> sum == OR.
    buf.scatter_add_(0, byte_idx.reshape(-1), contrib.reshape(-1))
    return buf.to(torch.uint8).reshape(Mpad, K // 2)


# --------------------------------------------------------------------------- #
# Step 4 / opt-A — cached-reindex fast activation packer (act_path="fast").
#
# `pack_activations_kernel` rebuilds the ENTIRE swizzle index/scatter map on every
# call (arange/div/mod/perm-gather over (Mpad,K) -> ~25 of its ~31 kernel launches and
# 45-59% of the int4 activation path, Q0.1). That map is INPUT-INDEPENDENT given
# (Mpad, K, perm): only `codes & 0xF` and the final scatter depend on data. We cache
# the flattened scatter index + per-element bit-shift per shape, so the hot path is
# just `nib<<shift -> scatter_add -> uint8`. This is BIT-IDENTICAL by construction:
# same `byte_idx.reshape(-1)`, same `(4*half).reshape(-1)`, same `contrib`, same
# `scatter_add_`, same cast as the reference above — proven by torch.equal in the gate.
# --------------------------------------------------------------------------- #
_PACK_INDEX_CACHE: "dict[tuple, tuple[torch.Tensor, torch.Tensor]]" = {}


def _packing_index(Mpad: int, K: int, dev: "torch.device", perm: torch.Tensor):
    """(idx_flat, shift_flat) for the swizzle scatter at shape (Mpad, K), cached.

    `idx_flat[p] == byte_idx.reshape(-1)[p]` and `shift_flat[p] == (4*half).reshape(-1)[p]`
    of `pack_activations_kernel` — i.e. the exact, input-independent half of that op.
    Keyed on (Mpad, K, device); version-locked to the singleton swizzle perm (the
    pinned-wheel layout; swizzle re-recovery is a non-goal)."""
    key = (int(Mpad), int(K), str(dev))
    hit = _PACK_INDEX_CACHE.get(key)
    if hit is not None:
        return hit
    perm = perm.to(dev)
    i = torch.arange(Mpad, device=dev).view(Mpad, 1)
    k = torch.arange(K, device=dev).view(1, K)
    bm = i // _BLOCK_M
    rem = i % _BLOCK_M
    warpId = rem // _WARP_M
    wrem = rem % _WARP_M
    tileId = wrem // _INSN_M
    m = wrem % _INSN_M
    bk = k // _WARP_K
    kk = k % _WARP_K
    base_uint4 = (
        (((bm * (K // _WARP_K) + bk) * _NUM_WARPS + warpId) * _WARP_M_TILES + tileId)
        * _WARP_SIZE
    )
    nu = perm[m.expand(Mpad, K), kk.expand(Mpad, K)]
    global_nibble = base_uint4 * _NIB_PER_UINT4 + nu
    idx_flat = (global_nibble // 2).reshape(-1).contiguous()
    shift_flat = (4 * (global_nibble % 2)).reshape(-1).contiguous()
    _PACK_INDEX_CACHE[key] = (idx_flat, shift_flat)
    return idx_flat, shift_flat


def pack_activations_kernel_fast(
    codes: torch.Tensor, *, perm: torch.Tensor | None = None, group_size: int = 64
) -> torch.Tensor:
    """Cached-reindex equivalent of `pack_activations_kernel` (Step 4, opt-A).

    Returns the byte-for-byte SAME `act` operand (uint8 (M_pad, K//2)); the only
    difference is the swizzle index/shift are precomputed once per (Mpad, K, device)
    instead of every call. Bit-identity is the spec — see the gate
    (scripts/fast_pack_gate.py)."""
    if perm is None:
        perm = load_act_swizzle()
    assert group_size == _WARP_K, f"group_size must be {_WARP_K}, got {group_size}"
    codes = codes.to(torch.int64)
    M, K = codes.shape
    assert K % _WARP_K == 0, f"K={K} must be a multiple of {_WARP_K}"
    Mpad = ((M + _BLOCK_M - 1) // _BLOCK_M) * _BLOCK_M
    if Mpad != M:
        codes = F.pad(codes, (0, 0, 0, Mpad - M))  # zero tokens at the tail
    dev = codes.device
    idx_flat, shift_flat = _packing_index(Mpad, K, dev, perm)
    nib = (codes & 0xF).reshape(-1)
    contrib = nib << shift_flat
    buf = torch.zeros(Mpad * (K // 2), dtype=torch.int64, device=dev)
    buf.scatter_add_(0, idx_flat, contrib)
    return buf.to(torch.uint8).reshape(Mpad, K // 2)


def ones_ascales_like_device(quantize, M: int, K: int, *, dtype=torch.float16, device="cuda"):
    """Build an all-ones `ascales` buffer matching the EXACT shape/dtype/layout the
    device quantizer emits (a matched read/write pair with the GEMM), so setting
    ascales == 1 makes the GEMM emit `main = Σ_g s_jg·(Σ Xq Q^s)` and the per-token
    scale a_i is applied host-side (G0.4). Derived from a one-shot device probe to
    avoid hand-computing the packed ascale layout."""
    # Probe with ones (not zeros) so the quantizer's per-(row,group) max is well
    # defined (zeros -> max 0 -> 7/0); only the returned ascales SHAPE/dtype is used.
    x = torch.ones(M, K, dtype=dtype, device=device)
    _, ascales = quantize_act_operand(quantize, x, pad_size=_BLOCK_M)
    return torch.ones_like(ascales)


def gemm_w4a4_main(
    act: torch.Tensor,
    wgt: torch.Tensor,
    wscales: torch.Tensor,
    ascales: torch.Tensor,
    *,
    out_features: int,
    out_dtype=torch.float16,
    gemm=None,
):
    """Drive the bare W4A4 GEMM with ascales == 1: returns
    `main[i,o] = Σ_g s_jg·(Σ_{k∈g} Xq_ik Q^s_jk)` as fp16 (M, out_features).
    No LoRA / qout / smooth / bias — those are None (G0.1). a_i, z-correction and
    bias are applied host-side by the caller (restructured_epilogue / G0.4)."""
    if gemm is None:
        _, gemm = load_nunchaku_ops()
    M = act.shape[0]
    out = torch.empty(M, out_features, dtype=out_dtype, device=act.device)
    gemm(
        act=act,
        wgt=wgt,
        out=out,
        ascales=ascales,
        wscales=wscales,
        lora_act_in=None, lora_up=None, lora_down=None, lora_act_out=None,
        bias=None, smooth_factor=None, qout=None, oscales=None,
        act_unsigned=False, fp4=False, alpha=1.0,
        lora_scales=[1.0],
    )
    return out
