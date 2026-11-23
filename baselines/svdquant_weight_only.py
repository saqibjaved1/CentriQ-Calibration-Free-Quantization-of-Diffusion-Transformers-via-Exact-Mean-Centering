"""Weight-only (W4A16) SVDQuant fake quantization.

Reimplements the *weight* half of SVDQuant (Li et al., "SVDQuant: Absorbing
Outliers by Low-Rank Components for 4-Bit Diffusion Models") as an in-place
fake-quant pass, mirroring `scripts/fid_eval.py:quantize_rtn`.

Each linear weight is split into a 16-bit low-rank branch (the top-`rank`
singular components — the part most expensive to quantize) plus a low-bit
residual:

    W  ≈  L + Q_b(W - L),    L = U[:, :r] · diag(S[:r]) · Vh[:r, :]

The residual `W - L` is quantized with the exact group-wise symmetric INT-`b`
scheme used by the RTN baseline, then dequantized and added back to the bf16
low-rank branch. Activations are left untouched (weight-only ⇒ W4A16), so this
runs through the same fake-quant eval path as RTN/SINQ/HQQ — there is no fast
W4A4 nunchaku kernel involved.

This form is calibration-free: SVDQuant's activation *smoothing* (a per-channel
scale that migrates activation outliers into the weights) needs activation
statistics and only matters when activations are also quantized, so it is
omitted here. `rank=32` matches the `_r32` published SVDQuant checkpoints.
"""

import torch


def quantize_svdquant_inplace(
    transformer,
    nbits: int = 4,
    group_size: int = 64,
    rank: int = 32,
    ignore_layer_names: list = None,
) -> int:
    """
    Weight-only SVDQuant fake quantization applied in-place.

    For each ``nn.Linear`` weight: peel off a rank-``rank`` bf16 low-rank branch
    via ``torch.svd_lowrank`` (cheaper than a full SVD — only the top
    components are needed), quantize the residual to per-group symmetric
    INT-``nbits`` (same math as ``quantize_rtn``), then write back
    ``dequant(residual) + low_rank``.

    Returns the number of quantized layers.
    """
    ignore_layer_names = ignore_layer_names or []
    qmax = 2 ** (nbits - 1) - 1  # 7 for INT4
    n_quantized = 0

    for name, module in transformer.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if name.split(".")[-1] in ignore_layer_names:
            continue

        W = module.weight.data           # [out_features, in_features]
        W_f = W.float()
        out_features, in_features = W_f.shape

        # --- Low-rank branch (top-`rank` singular components, kept in bf16) ---
        # svd_lowrank needs q <= min(dims); clamp for small layers.
        r = min(rank, out_features, in_features)
        if r > 0:
            # q slightly above r improves the randomized approximation.
            q = min(r + 8, out_features, in_features)
            U, S, V = torch.svd_lowrank(W_f, q=q)
            U, S, V = U[:, :r], S[:r], V[:, :r]
            L = (U * S) @ V.t()          # [out_features, in_features]
        else:
            L = torch.zeros_like(W_f)

        # --- 4-bit residual (same group-wise symmetric INT scheme as RTN) ---
        R = W_f - L
        if group_size > 0 and in_features >= group_size:
            pad = (-in_features) % group_size
            if pad:
                R = torch.cat([R, R.new_zeros(out_features, pad)], dim=1)
            n_groups = R.shape[1] // group_size
            R_g = R.reshape(out_features * n_groups, group_size)
            scale = R_g.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8) / qmax
            R_q = (R_g / scale).round().clamp(-(qmax + 1), qmax)
            R_dq = (R_q * scale).reshape(out_features, -1)[:, :in_features]
        else:
            scale = R.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8) / qmax
            R_q = (R / scale).round().clamp(-(qmax + 1), qmax)
            R_dq = R_q * scale

        module.weight.data = (R_dq + L).to(W.dtype)
        n_quantized += 1

    return n_quantized
