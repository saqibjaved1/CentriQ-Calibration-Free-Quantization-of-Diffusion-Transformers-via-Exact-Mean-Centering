"""
VENDORED Nunchaku weight/scale packing utilities (RoDS-A Step 2, plan P1.0).

This is a PURE-TORCH, byte-faithful copy of the packing math from the pinned
Nunchaku clone at code/nunchaku-inspect/ (commit 8f41840):
  - `MmaWeightPackerBase` / `NunchakuWeightPacker.pack_weight` / `pack_scale`
    from nunchaku/lora/flux/packer.py (lines 14-360),
  - `ceil_divide` from nunchaku/utils.py,
  - `pad` from nunchaku/lora/flux/utils.py.

WHY VENDOR (not import): `import nunchaku` executes nunchaku/__init__.py ->
nunchaku.models -> nunchaku._C, the CUDA extension, which is not prebuilt and
cannot be built on the front node. The packing math itself is pure torch (CPU),
so we copy the self-contained methods here to keep all Step-2 packing + round-trip
tests on the CPU front node. The copy is validated **byte-identical** against the
installed package's real packer on real layer shapes via a single A100 runai job
(scripts/pack_byteref.py); if that check ever diverges, this file is stale.

Only the methods Step 2 needs are copied (pack_weight, pack_scale + the base
geometry, pad_weight/pad_scale, micro-scale guard). Low-rank/LoRA packing is out
of scope and intentionally omitted. Do NOT edit the math — re-sync from the pinned
clone instead.
"""

from __future__ import annotations

import typing as tp

import torch


# --- nunchaku/utils.py:ceil_divide (verbatim) ---
def ceil_divide(x: int, divisor: int) -> int:
    """The smallest integer >= x / divisor."""
    return (x + divisor - 1) // divisor


# --- nunchaku/lora/flux/utils.py:pad (verbatim) ---
def pad(
    tensor: tp.Optional[torch.Tensor],
    divisor: int | tp.Sequence[int],
    dim: int | tp.Sequence[int],
    fill_value: float | int = 0,
) -> torch.Tensor | None:
    """Pad `tensor` so each `dim` is divisible by `divisor` (fill with `fill_value`)."""
    if isinstance(divisor, int):
        if divisor <= 1:
            return tensor
    elif all(d <= 1 for d in divisor):
        return tensor
    if tensor is None:
        return None
    shape = list(tensor.shape)
    if isinstance(dim, int):
        assert isinstance(divisor, int)
        shape[dim] = ceil_divide(shape[dim], divisor) * divisor
    else:
        if isinstance(divisor, int):
            divisor = [divisor] * len(dim)
        for d, div in zip(dim, divisor, strict=True):
            shape[d] = ceil_divide(shape[d], div) * div
    result = torch.full(shape, fill_value, dtype=tensor.dtype, device=tensor.device)
    result[[slice(0, extent) for extent in tensor.shape]] = tensor
    return result


class MmaWeightPackerBase:
    """Base class for MMA weight packing (verbatim from packer.py:14-162)."""

    def __init__(self, bits: int, warp_n: int, comp_n: int = None, comp_k: int = None):
        self.bits = bits
        assert self.bits in (1, 4, 8, 16, 32), "weight bits should be 1, 4, 8, 16, or 32."

        self.comp_n = comp_n if comp_n is not None else 16
        self.comp_k = comp_k if comp_k is not None else 256 // self.bits
        self.insn_n = 8
        self.insn_k = self.comp_k
        assert self.insn_k * self.bits in (128, 256), (
            f"insn_k ({self.insn_k}) * bits ({self.bits}) should be 128 or 256."
        )
        assert self.comp_n % self.insn_n == 0, (
            f"comp_n ({self.comp_n}) should be divisible by insn_n ({self.insn_n})."
        )
        self.num_lanes = 32
        self.num_k_lanes = 4
        self.num_n_lanes = 8
        assert warp_n >= self.comp_n and warp_n % self.comp_n == 0, (
            f"warp_n ({warp_n}) should be divisible by comp_n({self.comp_n})."
        )
        self.warp_n = warp_n
        self.reg_k = 32 // self.bits
        self.reg_n = 1
        self.k_pack_size = self.comp_k // (self.num_k_lanes * self.reg_k)
        self.n_pack_size = self.comp_n // (self.num_n_lanes * self.reg_n)
        self.pack_size = self.k_pack_size * self.n_pack_size
        assert 1 <= self.pack_size <= 4, "pack size should be less than or equal to 4."
        assert self.k_pack_size * self.num_k_lanes * self.reg_k == self.comp_k
        assert self.n_pack_size * self.num_n_lanes * self.reg_n == self.comp_n
        self.mem_k = self.comp_k
        self.mem_n = warp_n
        self.num_k_packs = self.mem_k // (self.k_pack_size * self.num_k_lanes * self.reg_k)
        self.num_n_packs = self.mem_n // (self.n_pack_size * self.num_n_lanes * self.reg_n)


class NunchakuWeightPacker(MmaWeightPackerBase):
    """Nunchaku weight/scale packer (verbatim subset from packer.py:165-360)."""

    def __init__(self, bits: int, warp_n: int = 128):
        super().__init__(bits=bits, warp_n=warp_n)
        self.num_k_unrolls = 2

    def pack_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """Pack int32 quantized weight (n, k) -> packed int8 (n, k*bits/8)."""
        assert weight.dtype == torch.int32, (
            f"quantized weight should be torch.int32, but got {weight.dtype}."
        )
        n, k = weight.shape
        assert n % self.mem_n == 0, (
            f"output channel size ({n}) should be divisible by mem_n ({self.mem_n})."
        )
        assert k % (self.mem_k * self.num_k_unrolls) == 0, (
            f"input channel size ({k}) should be divisible by "
            f"mem_k ({self.mem_k}) * num_k_unrolls ({self.num_k_unrolls})."
        )
        n_tiles, k_tiles = n // self.mem_n, k // self.mem_k
        weight = weight.reshape(
            n_tiles,
            self.num_n_packs,
            self.n_pack_size,
            self.num_n_lanes,
            self.reg_n,
            k_tiles,
            self.num_k_packs,
            self.k_pack_size,
            self.num_k_lanes,
            self.reg_k,
        )
        weight = weight.permute(0, 5, 6, 1, 3, 8, 2, 7, 4, 9).contiguous()
        assert weight.shape[4:-2] == (8, 4, 2, 2)
        if self.bits == 4:
            weight = weight.bitwise_and_(0xF)
            shift = torch.arange(0, 32, 4, dtype=torch.int32, device=weight.device)
            weight = weight.bitwise_left_shift_(shift)
            weight = weight.sum(dim=-1, dtype=torch.int32)
        elif self.bits == 8:
            weight = weight.bitwise_and_(0xFF)
            shift = torch.arange(0, 32, 8, dtype=torch.int32, device=weight.device)
            weight = weight.bitwise_left_shift_(shift)
            weight = weight.sum(dim=-1, dtype=torch.int32)
        else:
            raise NotImplementedError(f"weight bits {self.bits} is not supported.")
        return weight.view(dtype=torch.int8).view(n, -1)  # assume little-endian

    def pack_scale(self, scale: torch.Tensor, group_size: int) -> torch.Tensor:
        """Pack fp16/bf16 scale (n, n_groups) into the MMA wscale layout."""
        if self.check_if_micro_scale(group_size=group_size):
            return self.pack_micro_scale(scale, group_size=group_size)
        assert scale.dtype in (torch.float16, torch.bfloat16), (
            "currently nunchaku only supports fp16 and bf16."
        )
        n = scale.shape[0]
        s_pack_size = min(max(self.warp_n // self.num_lanes, 2), 8)
        num_s_lanes = min(self.num_lanes, self.warp_n // s_pack_size)
        num_s_packs = self.warp_n // (s_pack_size * num_s_lanes)
        warp_s = num_s_packs * num_s_lanes * s_pack_size
        assert warp_s == self.warp_n, "warp_n for scales should be equal to warp_n for weights."
        scale = scale.reshape(n // warp_s, num_s_packs, num_s_lanes // 4, s_pack_size // 2, 4, 2, -1)
        scale = scale.permute(0, 6, 1, 2, 4, 3, 5).contiguous()
        return scale.view(-1) if group_size == -1 else scale.view(-1, n)

    def pack_micro_scale(self, scale: torch.Tensor, group_size: int) -> torch.Tensor:
        """Pack fp8 micro-scale (group_size==16 path). Not used by W4A4 gs=64."""
        assert scale.dtype in (torch.float16, torch.bfloat16), (
            "currently nunchaku only supports fp16 and bf16."
        )
        assert scale.max() <= 448, "scale should be less than 448."
        assert scale.min() >= -448, "scale should be greater than -448."
        assert group_size == 16, "currently only support group size 16."
        assert self.insn_k == 64, "insn_k should be 64."
        scale = scale.to(dtype=torch.float8_e4m3fn)
        n = scale.shape[0]
        assert self.warp_n >= 32, "currently only support warp_n >= 32."
        s_pack_size = min(max(self.warp_n // self.num_lanes, 1), 4)
        num_s_lanes = 4 * 8
        num_s_packs = ceil_divide(self.warp_n, s_pack_size * num_s_lanes)
        warp_s = num_s_packs * num_s_lanes * s_pack_size
        assert warp_s == self.warp_n, "warp_n for scales should be equal to warp_n for weights."
        scale = scale.view(n // warp_s, num_s_packs, s_pack_size, 4, 8, -1, self.insn_k // group_size)
        scale = scale.permute(0, 5, 1, 4, 3, 2, 6).contiguous()
        return scale.view(-1, n)

    def check_if_micro_scale(self, group_size: int) -> bool:
        """True when fp8 micro-scale packing applies (insn_k == group_size*4)."""
        return self.insn_k == group_size * 4

    def pad_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """Pad (n, k) to (mem_n, mem_k*num_k_unrolls) multiples."""
        assert weight.ndim == 2, "weight tensor should be 2D."
        return pad(weight, divisor=(self.mem_n, self.mem_k * self.num_k_unrolls), dim=(0, 1))

    def pad_scale(self, scale: torch.Tensor, group_size: int, fill_value: float = 0) -> torch.Tensor:
        """Pad scale to the warp_n / group structure (no-op when already aligned)."""
        if group_size > 0 and scale.numel() > scale.shape[0]:
            scale = scale.view(scale.shape[0], 1, -1, 1)
            if self.check_if_micro_scale(group_size=group_size):
                scale = pad(scale, divisor=(self.warp_n, self.insn_k // group_size), dim=(0, 2), fill_value=fill_value)
            else:
                scale = pad(scale, divisor=(self.warp_n, self.num_k_unrolls), dim=(0, 2), fill_value=fill_value)
        else:
            scale = pad(scale, divisor=self.warp_n, dim=0, fill_value=fill_value)
        return scale
