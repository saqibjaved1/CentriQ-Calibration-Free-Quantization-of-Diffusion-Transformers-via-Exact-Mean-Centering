"""
RoDS-A W4A4 for 1x1 pointwise Conv2d (SANA GLUMBConv FFN extension).

A 1x1 conv over (B, Cin, H, W) is a per-spatial-token GEMM Cin->Cout, so the
composed W4A4 path REUSES RoDSAW4A4Linear for all quant math ("tokens" = spatial
positions), adding only the NCHW <-> (N, C) reshaping.

Scope: groups=1, kernel 1x1 ONLY. Depthwise convs (groups != 1, e.g. SANA's
conv_depth 3x3) stay FP16 -- the per-output-channel weight count (kH*kW) is too
small to group-quantize (SVDQuant keeps depth_conv FP16 too).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor, nn

from rods_a.prepare_weight import prepare_weight
from rods_a.wrapper import ActQuantConfig, RoDSAW4A4Linear, _set_submodule

# The two pointwise (1x1, groups=1) convs of SANA's GLUMBConv. conv_depth is
# depthwise (groups != 1) and is intentionally absent -> stays FP16.
_FFN_CONV_SUFFIXES = (".ff.conv_inverted", ".ff.conv_point")


def classify_conv_role(name: str) -> str:
    if name.endswith("conv_inverted"):
        return "ffn_conv_in"
    if name.endswith("conv_point"):
        return "ffn_conv_out"
    return "other_conv"


def select_conv1x1_targets(
    transformer: nn.Module,
    suffixes: tuple = _FFN_CONV_SUFFIXES,
) -> Dict[str, nn.Conv2d]:
    """Return {name: Conv2d} for the 1x1 groups=1 FFN pointwise convs. Filters to
    `suffixes` so embedder 1x1 convs (e.g. patch_embed.proj) are NOT swept in."""
    targets: Dict[str, nn.Conv2d] = {}
    for name, mod in transformer.named_modules():
        if not isinstance(mod, nn.Conv2d):
            continue
        if mod.kernel_size != (1, 1) or mod.groups != 1:
            continue
        if suffixes and not any(name.endswith(s) for s in suffixes):
            continue
        targets[name] = mod
    return targets


class RoDSAW4A4Conv1x1(nn.Module):
    """Fake-quant composed W4A4 for a 1x1 pointwise Conv2d. Delegates all quant
    math to an inner RoDSAW4A4Linear (tokens = spatial positions)."""

    def __init__(self, pack, cfg: ActQuantConfig, *, fp_weight: Optional[Tensor] = None,
                 center_mean: bool = False, matmul_dtype: torch.dtype = torch.float16,
                 backend: str = "fakequant", layer_name: str = "", role: str = "ffn_conv",
                 device: Optional[str] = None,
                 rotate_override=None, unrotate_override=None,
                 capture_dtype: torch.dtype = torch.float32):
        super().__init__()
        # fp_weight for the inner linear's SQNR capture is the SQUEEZED (Cout, Cin) weight,
        # so its per-token y_fp = x2d @ Wᵀ matches the 1x1 conv output exactly.
        self.lin = RoDSAW4A4Linear(
            pack, cfg, fp_weight=fp_weight, center_mean=center_mean,
            matmul_dtype=matmul_dtype, backend=backend, layer_name=layer_name,
            role=role, device=device,
            rotate_override=rotate_override, unrotate_override=unrotate_override,
            capture_dtype=capture_dtype,
        )
        self.in_channels = pack.Cin_orig
        self.out_channels = pack.out_features
        self.role = role

    def forward(self, x: Tensor) -> Tensor:
        # (B, Cin, H, W) -> (B, H, W, Cin); the inner Linear reshapes (-1, Cin) and
        # returns (B, H, W, Cout); -> (B, Cout, H, W).
        if x.dim() != 4:
            raise ValueError(f"RoDSAW4A4Conv1x1 expects NCHW input, got {tuple(x.shape)}")
        y = self.lin(x.permute(0, 2, 3, 1).contiguous())
        return y.permute(0, 3, 1, 2).contiguous()

    def sqnr_db(self) -> Optional[float]:
        return self.lin.sqnr_db()

    @classmethod
    def from_conv(cls, conv: nn.Conv2d, cfg: ActQuantConfig, *, bits: int = 4,
                  group_size: int = 64, rotation: str = "pad", device: str = "cpu",
                  compute_dtype: torch.dtype = torch.float32, capture_sqnr: bool = False,
                  strict_rho: bool = False, rho_band: tuple = (2.0, 4.0),
                  center_mean: bool = False, matmul_dtype: torch.dtype = torch.float16,
                  backend: str = "fakequant", layer_name: str = "", role: str = "ffn_conv",
                  rotate_fn=None, rotate_override=None, unrotate_override=None,
                  capture_dtype: torch.dtype = torch.float32):
        """Prepare (rotate + grouped RoDS on W̃) the squeezed conv weight and wrap.
        Raises for non-1x1 / depthwise convs. Returns (module, pack)."""
        W = conv.weight.detach()                       # (Cout, Cin, 1, 1)
        if tuple(W.shape[2:]) != (1, 1):
            raise ValueError(f"RoDSAW4A4Conv1x1 needs a 1x1 conv, got kernel {tuple(W.shape[2:])}")
        if conv.groups != 1:
            raise ValueError("RoDSAW4A4Conv1x1 does not support depthwise (groups!=1) convs")
        W2 = W.reshape(W.shape[0], W.shape[1])         # (Cout, Cin)
        pack = prepare_weight(
            W2, bias=conv.bias, bits=bits, group_size=group_size, device=device,
            compute_dtype=compute_dtype, bitpack=True, rotate_weight=cfg.rotate_act,
            rotation=rotation, rho_band=rho_band, strict_rho=strict_rho,
            rotate_fn=rotate_fn,
        )
        fp_w = W2 if capture_sqnr else None
        mod = cls(pack, cfg, fp_weight=fp_w, center_mean=center_mean,
                  matmul_dtype=matmul_dtype, backend=backend, layer_name=layer_name,
                  role=role, device=device,
                  rotate_override=rotate_override, unrotate_override=unrotate_override,
                  capture_dtype=capture_dtype)
        return mod, pack


def install_ffn_conv_w4a4(
    transformer: nn.Module,
    cfg: ActQuantConfig,
    *,
    weight_bits: int = 4,
    group_size: int = 64,
    rotation: str = "pad",
    device: str = "cpu",
    compute_dtype: torch.dtype = torch.float32,
    matmul_dtype: torch.dtype = torch.float16,
    capture_sqnr: bool = False,
    center_act_mean: bool = False,
    verbose: bool = True,
    capture_dtype: torch.dtype = torch.float32,
) -> Dict[str, dict]:
    """Replace SANA GLUMBConv 1x1 pointwise convs with RoDSAW4A4Conv1x1 in place.
    Returns per-layer diagnostics (role, rho_median, incoherence, Cin_orig/pad).
    The depthwise conv_depth is left untouched (FP16)."""
    # Factored rotation: same installer-side resolution as install_w4a4 -- resolve
    # each conv's channel width against cfg.rotation_factorization and inject the
    # rotators; unmapped widths fall back bitwise to the pad scheme. When factored,
    # the `rotation` string param is superseded by the injected callables.
    factored = str(getattr(cfg, "rotation", "pad")) == "factored"
    if factored:
        from rods_a.factored_rotation import (factored_rotators, pad_rotators,
                                              resolve_factored)
        from rods_a.fwht import next_pow2 as _next_pow2
        from rods_a.fwht import rotate as _pad_rotate_fwht

    targets = select_conv1x1_targets(transformer)
    diags: Dict[str, dict] = {}
    for name, conv in targets.items():
        role = classify_conv_role(name)
        center = bool(center_act_mean)
        rot_w = act_rot = act_unrot = None
        conv_rotation = rotation
        fact_spec = None
        rot_desc = None
        if factored:
            width = int(conv.weight.shape[1])
            fact_spec = resolve_factored(width,
                                         getattr(cfg, "rotation_factorization", None),
                                         getattr(cfg, "rotation_pad_to", None))
            if fact_spec is not None:
                if fact_spec.pad_to % group_size != 0:
                    raise ValueError(
                        f"factored rotation for {name}: pad_to={fact_spec.pad_to} is "
                        f"not divisible by group_size={group_size}.")
                act_rot, act_unrot = factored_rotators(fact_spec)
                rot_w = act_rot
                rot_desc = (f"{fact_spec.factors[0]}x{fact_spec.factors[1]}"
                            f"@{fact_spec.pad_to}")
            else:
                act_rot, act_unrot = pad_rotators("dense")
                rot_w = _pad_rotate_fwht
                rot_desc = f"pad@{_next_pow2(width)}"
            conv_rotation = "pad"  # inert: rotate_fn/overrides carry the real map
        mod, pack = RoDSAW4A4Conv1x1.from_conv(
            conv, cfg, bits=weight_bits, group_size=group_size, rotation=conv_rotation,
            device=device, compute_dtype=compute_dtype, capture_sqnr=capture_sqnr,
            center_mean=center, matmul_dtype=matmul_dtype, layer_name=name, role=role,
            rotate_fn=rot_w, rotate_override=act_rot, unrotate_override=act_unrot,
            capture_dtype=capture_dtype,
        )
        _set_submodule(transformer, name, mod)
        diags[name] = {
            "role": role, "rho_median": pack.rho_median,
            "incoherence_W": pack.incoherence_W, "incoherence_Wt": pack.incoherence_Wt,
            "Cin_orig": pack.Cin_orig, "Cin_pad": pack.Cin_pad,
        }
        if factored:
            diags[name]["rotation_desc"] = rot_desc
            if fact_spec is not None:
                diags[name]["factors"] = list(fact_spec.factors)
                diags[name]["pad_to"] = fact_spec.pad_to
    if verbose:
        rhos = [d["rho_median"] for d in diags.values()]
        rho_med = float(torch.tensor(rhos).median()) if rhos else float("nan")
        rot_label = "factored" if factored else rotation
        print(f"[rods_a] W4A{cfg.act_bits} FFN-conv installed on {len(diags)} 1x1 convs "
              f"(depthwise left FP16), weight {weight_bits}b gs={group_size}, "
              f"rotate={cfg.rotate_act}/{rot_label}, rho_median={rho_med:.3f}.")
        if factored:
            by_class: Dict[tuple, int] = {}
            for d in diags.values():
                key = (d["Cin_orig"], d["rotation_desc"])
                by_class[key] = by_class.get(key, 0) + 1
            parts = [f"{w}->{desc} ({n} convs)"
                     for (w, desc), n in sorted(by_class.items())]
            print("[rods_a] factored rotation (convs): " + "; ".join(parts))
    return diags
