"""In-place RTN / NF4 baseline weight fake-quant on the RoDS-A act-target set.

Used by the naive-W4A4 baseline cells: quantize the weights of EXACTLY the
layers the RoDS-A activation track targets (PixArt 224 / SANA 120 / FLUX 380;
skip-set weights stay FP16), then install the A4-only wrapper
(`install_w4a4(weight_quant_enabled=False)`) on top so the forward pass is
fakequant_act(x) @ W_baseline.T + b.

The quantizers themselves are the registered spine baselines
(`baselines.baseline_methods.quantize_rtn` / `quantize_nf4`) reused verbatim — this
module only pins the layer set and reports parity diagnostics. Report-only:
no gating on the rho statistic (W3 is expected to run hot).
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, Optional

import torch.nn as nn

from rods_a.wrapper import select_act_targets
from baselines.baseline_methods import quantize_rtn, quantize_nf4

# Family act-target counts (parity invariant; a mismatch means the layer set
# drifted from the RoDS-A runs and the cell is non-reportable).
_EXPECTED_TARGETS = {"pixart": 224, "sana": 120, "flux": 380}


def apply_baseline_weight_quant(
    transformer: nn.Module,
    *,
    method: str,
    nbits: int = 4,
    group_size: int = 64,
    model_type: str = "pixart",
    expected_n: Optional[int] = None,
) -> Dict[str, Any]:
    """Quantize-dequantize weights in place on the act-target set; return diags.

    method: "rtn" | "nf4". NF4 is a fixed 4-bit format — nbits != 4 raises
    (there is no W3 NF4 baseline anywhere, by construction).
    """
    method = str(method).lower()
    if method not in ("rtn", "nf4"):
        raise ValueError(f"baseline weight method must be 'rtn' or 'nf4', got {method!r}")
    nbits = int(nbits)
    if method == "nf4" and nbits != 4:
        raise ValueError("NF4 is a fixed 4-bit format; a W3 NF4 cell is a bug, not a baseline.")

    targets = select_act_targets(transformer, model_type=model_type, group_size=group_size)
    names = set(targets.keys())

    stats: Dict[str, Dict[str, float]] = {}
    if method == "rtn":
        n = quantize_rtn(transformer, nbits=nbits, group_size=group_size,
                         target_names=names, stats_out=stats)
        rho_key = "rho_median"  # sigma_FP / Delta (Delta = absmax/qmax LSB)
    else:
        n = quantize_nf4(transformer, group_size=group_size,
                         target_names=names, stats_out=stats)
        rho_key = "sigma_over_absmax_median"  # non-uniform codebook: no single LSB

    if n != len(names):
        raise RuntimeError(f"[baseline] quantized {n} layers but selected {len(names)} "
                           f"act targets — selection/quantizer drift.")
    exp = expected_n if expected_n is not None else _EXPECTED_TARGETS.get(model_type)
    if exp is not None and n != exp:
        raise RuntimeError(f"[baseline] {model_type}: quantized {n} layers, expected {exp} "
                           f"(act-target parity violated — cell is non-reportable).")

    vals = [v[rho_key] for v in stats.values()]
    rho_global = float(statistics.median(vals)) if vals else float("nan")
    print(f"[baseline] weight={method} W{nbits} gs{group_size} in-place on {n} "
          f"act-target layers (model={model_type}); {rho_key} global median={rho_global:.3f}")
    return {
        "method": method,
        "nbits": nbits,
        "group_size": int(group_size),
        "n_quantized": int(n),
        "rho_stat": rho_key,
        "rho_median_global": rho_global,
        "per_layer": stats,
    }
