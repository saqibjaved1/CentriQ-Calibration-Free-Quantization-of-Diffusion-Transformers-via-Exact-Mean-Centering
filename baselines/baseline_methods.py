"""Baseline weight-quantization methods (RTN + NF4).

Extracted verbatim from scripts/fid_eval.py so the eval harness AND the W4A4
additive-baseline hook (rods_a/baseline_weight.py) share one definition without
rods_a reaching into scripts/. Weight-only, in-place fake-quant; fp scales.
"""
import torch


def _baseline_layer_selected(name, ignore_layer_names, target_names) -> bool:
    """Layer-selection rule shared by quantize_rtn / quantize_nf4.

    target_names (set of fully-qualified module names) takes precedence and
    selects exactly those layers. Otherwise ignore entries match the leaf name,
    or — when the entry contains a dot (e.g. "attn2.to_k") — a dotted suffix of
    the full name, so a cross-attn-only skip is expressible without also
    skipping the self-attn twin.
    """
    if target_names is not None:
        return name in target_names
    for ign in ignore_layer_names:
        if "." in ign:
            if name == ign or name.endswith("." + ign):
                return False
        elif name.split(".")[-1] == ign:
            return False
    return True


def quantize_rtn(
    transformer,
    nbits: int = 4,
    group_size: int = 64,
    ignore_layer_names: list = None,
    target_names: set = None,
    stats_out: dict = None,
) -> int:
    """
    Naive round-to-nearest (RTN) fake quantization applied in-place.
    Per-group symmetric INT-{nbits}: quantize then immediately dequantize each
    nn.Linear weight, leaving the model in fp16 but with quantisation noise.
    Acts as a lower-bound baseline for SINQ and HQQ.
    Returns the number of quantized layers.

    target_names: optional set of fully-qualified layer names — quantize exactly
    these (ignore_layer_names is then unused). stats_out: optional dict that
    receives per-layer {"rho_median": median over groups of std(W_g)/step}
    where step = absmax/qmax is the quantization LSB (report-only sanity).
    """
    ignore_layer_names = ignore_layer_names or []
    qmax = 2 ** (nbits - 1) - 1  # 7 for INT4
    n_quantized = 0

    for name, module in transformer.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if not _baseline_layer_selected(name, ignore_layer_names, target_names):
            continue

        W = module.weight.data           # [out_features, in_features]
        W_f = W.float()
        out_features, in_features = W_f.shape

        if group_size > 0 and in_features >= group_size:
            pad = (-in_features) % group_size
            if pad:
                W_f = torch.cat([W_f, W_f.new_zeros(out_features, pad)], dim=1)
            n_groups = W_f.shape[1] // group_size
            W_g = W_f.reshape(out_features * n_groups, group_size)
            scale = W_g.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8) / qmax
            W_q = (W_g / scale).round().clamp(-(qmax + 1), qmax)
            W_dq = (W_q * scale).reshape(out_features, -1)[:, :in_features]
        else:
            W_g = W_f
            scale = W_f.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8) / qmax
            W_q = (W_f / scale).round().clamp(-(qmax + 1), qmax)
            W_dq = W_q * scale

        if stats_out is not None:
            rho = (W_g.std(dim=1, keepdim=True) / scale).squeeze(1)
            stats_out[name] = {
                "rho_median": float(rho.median()),
                "n_groups": int(rho.numel()),
            }

        module.weight.data = W_dq.to(W.dtype)
        n_quantized += 1

    return n_quantized


def quantize_nf4(
    transformer,
    group_size: int = 64,
    ignore_layer_names: list = None,
    target_names: set = None,
    stats_out: dict = None,
) -> int:
    """
    Plain NF4 fake quantization applied in-place (no Sinkhorn, no double-quant).
    Per-group absmax-scaled mapping onto the fixed 16-level NF4 quantile codebook:
    s = max(|W_g|); normalise W_g/s into [-1, 1]; snap each value to the nearest
    NF4 codebook entry; dequantise = codebook[idx] * s. Weight-only, FP (per-group)
    scales. NF4 is a fixed 4-bit format, so `nbits` does not apply.

    Mirrors quantize_rtn() (the INT4 baseline) but with a non-uniform codebook so
    the two are directly comparable at the same group_size. Returns the number of
    quantized layers.

    target_names: optional set of fully-qualified layer names — quantize exactly
    these (ignore_layer_names is then unused). stats_out: optional dict that
    receives per-layer {"sigma_over_absmax_median": median over groups of
    std(W_g)/absmax} — the NF4 codebook is non-uniform so there is no single LSB;
    this is the analogous shape statistic (report-only).
    """
    from core.sinq.dual_shift import NF4_CODEBOOK  # 16-value QLoRA NF4 codebook
    ignore_layer_names = ignore_layer_names or []
    n_quantized = 0

    for name, module in transformer.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if not _baseline_layer_selected(name, ignore_layer_names, target_names):
            continue

        W = module.weight.data           # [out_features, in_features]
        W_f = W.float()
        out_features, in_features = W_f.shape
        cb = NF4_CODEBOOK.to(W_f.device)  # [16]

        if group_size > 0 and in_features >= group_size:
            pad = (-in_features) % group_size
            if pad:
                W_f = torch.cat([W_f, W_f.new_zeros(out_features, pad)], dim=1)
            n_groups = W_f.shape[1] // group_size
            W_g = W_f.reshape(out_features * n_groups, group_size)
            scale = W_g.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
            idx = (W_g / scale).unsqueeze(-1).sub(cb).abs().argmin(dim=-1)
            W_dq = (cb[idx] * scale).reshape(out_features, -1)[:, :in_features]
        else:
            W_g = W_f
            scale = W_f.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
            idx = (W_f / scale).unsqueeze(-1).sub(cb).abs().argmin(dim=-1)
            W_dq = cb[idx] * scale

        if stats_out is not None:
            ratio = (W_g.std(dim=1, keepdim=True) / scale).squeeze(1)
            stats_out[name] = {
                "sigma_over_absmax_median": float(ratio.median()),
                "n_groups": int(ratio.numel()),
            }

        module.weight.data = W_dq.to(W.dtype)
        n_quantized += 1

    return n_quantized
