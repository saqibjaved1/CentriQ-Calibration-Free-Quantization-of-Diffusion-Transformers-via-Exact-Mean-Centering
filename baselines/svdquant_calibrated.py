"""Reimport a deepcompressor *calibrated* SVDQuant checkpoint into a diffusers
PixArt transformer — both a weight-only (W4A16/W3A16) reconstruction and a native
W4A4/W3A4 fake-quant path.

nunchaku's weight converter is FLUX-only, so we reconstruct directly from
deepcompressor artifacts.

Artifacts in the `--save-model` dir (the .pt files are symlinks into
runs/.../cache; torch.load follows them). Verified against a real PixArt-Sigma
W4A4 checkpoint:
  - model.pt  : full fp16 state_dict. Per quantized linear it holds the
                *dequantized residual* R_dq = quant(W·s − LR) (the low-rank branch
                is NOT included — it is stored separately and added at runtime).
                ff.net.2 is an output-preserving ShiftedLinear, stored as
                `<n>.shift` (scalar), `<n>.linear.weight`, `<n>.linear.bias`.
  - branch.pt : {akey: {"a.weight":[rank,in], "b.weight":[outs,rank]}}. For
                self-attention the branch is FUSED over q/k/v: b is [3·out, rank],
                keyed under to_q; to_k/to_v take the 2nd/3rd row-block. LR = b@a.
  - smooth.pt : {akey: s[in]}, the per-input-channel smoothing scale applied at
                runtime as x/s (fuse_when_possible=False). Self-attention q/k/v
                share to_q's scale; cross-attention k/v are not smoothed.

`akey` is the layer's quantized-module name: the layer name itself, or
`<name>.linear` for the shifted ff.net.2.

Weight-only (no activation quant ⇒ smoothing folds into the weight):
    W_eff   = (R_dq + LR) / s
    bias_eff = bias + (R_dq + LR) @ (shift / s)        (shift folds, ff.net.2 only)
Native W4A4/W3A4 keeps per-token-per-group INT4 activation fake-quant.
"""

import os
import torch
import torch.nn as nn


# ----------------------------------------------------------------------------- #
# Artifact loading + per-layer resolution
# ----------------------------------------------------------------------------- #

def _load_artifacts(ckpt_dir: str):
    model_path = os.path.join(ckpt_dir, "model.pt")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"deepcompressor checkpoint not found: {model_path}\n"
            f"Run launch/quant/svdquant_calib_pixart.sh first."
        )
    model_sd = torch.load(model_path, map_location="cpu", weights_only=False)
    branch = _maybe_load(os.path.join(ckpt_dir, "branch.pt"))
    smooth = _maybe_load(os.path.join(ckpt_dir, "smooth.pt"))
    return model_sd, branch, smooth


def _maybe_load(path: str):
    return torch.load(path, map_location="cpu", weights_only=False) if os.path.exists(path) else {}


def _qkv_slot(akey: str):
    """If akey ends with to_q/to_k/to_v, return (to_q_key, slot in {0,1,2})."""
    for slot, suf in enumerate((".to_q", ".to_k", ".to_v")):
        if akey.endswith(suf):
            return akey[: -len(suf)] + ".to_q", slot
    return None


def _branch_full(akey: str, branch: dict):
    """LR = b @ a for an artifact key, float32 [b_out, in], or None."""
    if akey not in branch:
        return None
    d = branch[akey]
    a = d.get("a.weight", d.get("a"))
    b = d.get("b.weight", d.get("b"))
    if a is None or b is None:
        ts = [v for v in d.values() if torch.is_tensor(v)]
        if len(ts) != 2:
            return None
        t0, t1 = ts
        a, b = (t0, t1) if t0.shape[0] <= t0.shape[1] else (t1, t0)
    return b.float() @ a.float()


def _branch_LR(akey: str, branch: dict, N: int):
    """Low-rank weight [N, in] for a layer with out_features N, handling the
    fused-qkv branch (b=[3N, rank] under to_q; to_k/to_v take row-blocks)."""
    LR = _branch_full(akey, branch)
    if LR is not None:
        if LR.shape[0] == N:
            return LR
        if LR.shape[0] == 3 * N:          # fused group, akey is to_q (slot 0)
            return LR[:N]
        return LR[:N]
    grp = _qkv_slot(akey)
    if grp is not None:
        qkey, slot = grp
        if slot != 0:
            LRq = _branch_full(qkey, branch)
            if LRq is not None and LRq.shape[0] == 3 * N:
                return LRq[slot * N:(slot + 1) * N]
    return None


def _smooth_s(akey: str, smooth: dict, branch: dict, N: int):
    """Smoothing scale [in] for a layer, or None. to_k/to_v share to_q's scale
    only when the group is the fused self-attention qkv (to_q branch is [3N,·])."""
    if akey in smooth:
        return smooth[akey]
    grp = _qkv_slot(akey)
    if grp is not None:
        qkey, slot = grp
        if slot != 0 and qkey in smooth:
            LRq = _branch_full(qkey, branch)
            if LRq is not None and LRq.shape[0] == 3 * N:
                return smooth[qkey]
    return None


def _layer_record(name: str, model_sd: dict, branch: dict, smooth: dict):
    """For a diffusers linear `name`, return (R_dq, bias, LR, s, shift) as float32,
    or None if it has no quantized weight in the checkpoint."""
    if name + ".weight" in model_sd:                      # plain linear
        akey = name
        R_dq = model_sd[name + ".weight"].float()
        bias = model_sd.get(name + ".bias")
        shift = None
    elif name + ".linear.weight" in model_sd:             # ShiftedLinear (ff.net.2)
        akey = name + ".linear"
        R_dq = model_sd[name + ".linear.weight"].float()
        bias = model_sd.get(name + ".linear.bias")
        shift = model_sd.get(name + ".shift")
    else:
        return None
    N = R_dq.shape[0]
    LR = _branch_LR(akey, branch, N)
    s = _smooth_s(akey, smooth, branch, N)
    bias = None if bias is None else bias.float()
    shift = None if shift is None else shift.reshape(-1).float()
    s = None if s is None else s.reshape(-1).float()
    return R_dq, bias, LR, s, shift


def _quant_prefixes(linear_names, model_sd, branch, smooth):
    return {n for n in linear_names if _layer_record(n, model_sd, branch, smooth) is not None}


def _passthrough(model_sd, quant_prefixes):
    """model.pt tensors that are NOT quantized-linear sub-keys (norms/embeds/…)."""
    out = {}
    for k, v in model_sd.items():
        base = k.rsplit(".", 1)[0]          # strip .weight/.bias/.shift
        base2 = k.rsplit(".", 2)[0]         # strip .linear.weight/.linear.bias
        if base in quant_prefixes or base2 in quant_prefixes:
            continue
        out[k] = v
    return out


# ----------------------------------------------------------------------------- #
# Public entry
# ----------------------------------------------------------------------------- #

def load_svdquant_calibrated_inplace(transformer, ckpt_dir: str, mode: str = "weight_only",
                                     act_bits: int = 4) -> int:
    model_sd, branch, smooth = _load_artifacts(ckpt_dir)
    target_dtype = next(transformer.parameters()).dtype
    linear_names = {n for n, m in transformer.named_modules() if isinstance(m, nn.Linear)}
    quant_prefixes = _quant_prefixes(linear_names, model_sd, branch, smooth)
    if mode == "weight_only":
        return _load_weight_only(transformer, model_sd, branch, smooth, quant_prefixes, target_dtype)
    elif mode == "w4a4":
        return _load_native(transformer, model_sd, branch, smooth, quant_prefixes, target_dtype, act_bits)
    raise ValueError(f"unknown mode {mode!r}")


# ----------------------------------------------------------------------------- #
# Weight-only reconstruction  (W4A16 / W3A16)
# ----------------------------------------------------------------------------- #

def _load_weight_only(transformer, model_sd, branch, smooth, quant_prefixes, target_dtype) -> int:
    out_sd = _passthrough(model_sd, quant_prefixes)
    n_quant = n_branch = n_smooth = n_shift = 0
    for name in sorted(quant_prefixes):
        R_dq, bias, LR, s, shift = _layer_record(name, model_sd, branch, smooth)
        M = R_dq if LR is None else (R_dq + LR)             # smoothed effective weight [out,in]
        if LR is not None:
            n_branch += 1
        if s is not None:
            M = M / s.clamp_min(1e-8).unsqueeze(0)          # fold runtime x/s into the weight
            n_smooth += 1
        b = torch.zeros(M.shape[0]) if bias is None else bias.clone()
        if shift is not None:
            sh = shift / (s.clamp_min(1e-8) if s is not None else torch.ones_like(shift))
            b = b + (R_dq if LR is None else (R_dq + LR)) @ sh   # output-preserving shift fold
            n_shift += 1
        out_sd[name + ".weight"] = M.to(target_dtype)
        if bias is not None or shift is not None:
            out_sd[name + ".bias"] = b.to(target_dtype)
    missing, unexpected = transformer.load_state_dict(out_sd, strict=False)
    if unexpected:
        print(f"  [SVDQuant-Cal] WARNING {len(unexpected)} unexpected keys, e.g. {unexpected[:5]}")
    if missing:
        print(f"  [SVDQuant-Cal] WARNING {len(missing)} missing keys, e.g. {missing[:5]}")
    print(f"  [SVDQuant-Cal] weight-only: {len(quant_prefixes)} layers "
          f"({n_branch} branch, {n_smooth} smooth, {n_shift} shifted)")
    return len(quant_prefixes)


# ----------------------------------------------------------------------------- #
# Native W4A4 / W3A4  (activation fake-quant via a wrapper module)
# ----------------------------------------------------------------------------- #

def _qdq_per_token_per_group(x_flat, K, G, bits, signed):
    """Per-token × per-input-group fake-quant. x_flat (M,K)."""
    qmax = (1 << (bits - 1)) - 1 if signed else (1 << bits) - 1
    qmin = -(1 << (bits - 1)) if signed else 0
    M = x_flat.shape[0]
    ng = (K + G - 1) // G
    pad = ng * G - K
    if pad:
        x_flat = torch.cat([x_flat, x_flat.new_zeros(M, pad)], dim=1)
    xg = x_flat.reshape(M, ng, G).float()
    s = xg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    xq = (xg / s).round().clamp_(qmin, qmax)
    return (xq * s).reshape(M, ng * G)[:, :K].to(x_flat.dtype)


class SVDQuantCalLinear(nn.Module):
    """Native SVDQuant fake-quant: y = qdq((x+shift)/s)·R_dqᵀ + ((x+shift)/s)·LRᵀ + bias."""

    def __init__(self, R_dq, LR, smooth, shift, bias, group_size, act_bits, act_unsigned):
        super().__init__()
        out_f, in_f = R_dq.shape
        self.register_buffer("R_dq", R_dq)
        self.register_buffer("LR", LR if LR is not None else None)
        self.register_buffer("smooth", smooth if smooth is not None
                             else torch.ones(in_f, dtype=R_dq.dtype))
        if shift is not None and shift.numel() == 1:                # scalar -> per-channel
            shift = shift.expand(in_f).clone()
        self.register_buffer("shift", shift)
        self.register_buffer("bias", bias)
        self.group_size = int(group_size) if group_size > 0 else in_f
        self.act_bits = int(act_bits)
        self.act_unsigned = bool(act_unsigned)
        self.out_features, self.in_features = out_f, in_f

    def forward(self, x):
        K = self.in_features
        xf = x.reshape(-1, K)
        if self.shift is not None:
            xf = xf + self.shift.view(1, K)
        Xhat = xf / self.smooth.view(1, K)
        out = _qdq_per_token_per_group(Xhat, K, self.group_size, self.act_bits,
                                       not self.act_unsigned) @ self.R_dq.t()
        if self.LR is not None:
            out = out + (Xhat.float() @ self.LR.float().t()).to(x.dtype)
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*x.shape[:-1], self.out_features).to(x.dtype)


def _load_native(transformer, model_sd, branch, smooth, quant_prefixes, target_dtype, act_bits) -> int:
    transformer.load_state_dict(_passthrough(model_sd, quant_prefixes), strict=False)
    device = next(transformer.parameters()).device
    name_to_module = dict(transformer.named_modules())
    for name in sorted(quant_prefixes):
        R_dq, bias, LR, s, shift = _layer_record(name, model_sd, branch, smooth)
        mod = SVDQuantCalLinear(
            R_dq=R_dq.to(target_dtype),
            LR=None if LR is None else LR.to(target_dtype),
            smooth=None if s is None else s.to(target_dtype),
            shift=None if shift is None else shift.to(target_dtype),
            bias=None if bias is None else bias.to(target_dtype),
            group_size=64, act_bits=act_bits,
            act_unsigned=shift is not None,        # shifted (ff.net.2) -> unsigned act quant
        ).to(device)
        parent, _, child = name.rpartition(".")
        setattr(name_to_module[parent], child, mod)
    print(f"  [SVDQuant-Cal] native A{act_bits}: {len(quant_prefixes)} layers wrapped")
    return len(quant_prefixes)
