"""
CPU SQNR harness for RoDS-A (Phase 1 scaffolding, no GPU / no model load).

Runs the activation fake-quant path over synthetic tensors shaped like the real
quantization targets (the Step-3 dimension audit) and reports, per layer and per
estimator: reconstruction SQNR X -> X_hat (dB), post-rotation excess kurtosis,
and residual channel-mean (abs-max and L2).

HOOKING NOTE
------------
Without a GPU/model we drive the path directly through `quantize_act` -- exactly
what `RoDSActLinear.forward` calls internally -- on synthetic activations. For
real GPU runs (Phase 2) the same metrics are captured via forward hooks on the
live transformer's RoDSActLinear inputs; this harness is the CPU stand-in.

wandb: wired via `wandb.config`, offline by default, no-op if unavailable.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from rods_a.constants import gamma_star as _gamma_star_lookup
from rods_a.quantize_act import ESTIMATORS, quantize_act

# Distinct quantization-target in_features per model (Step-3 dimension audit).
# (Cross-attn K/V 768 is kept FP16 by the wrapper but is still informative to
# measure, so it is included here.)
AUDITED_SHAPES: Dict[str, List[Tuple[str, int]]] = {
    "pixart": [("attn2.kv", 768), ("hidden", 1152), ("ff.down", 4608)],
    "flux":   [("hidden", 3072), ("ff.down", 24576)],
}


def sqnr_db(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """Signal-to-quantization-noise ratio in dB."""
    sig = x.double().pow(2).sum()
    noise = (x.double() - x_hat.double()).pow(2).sum().clamp_min(1e-30)
    return float(10.0 * torch.log10(sig / noise))


def synth_activations(
    T: int,
    Cin: int,
    *,
    outlier_frac: float = 0.01,
    outlier_scale: float = 20.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Synthetic activations: standard Gaussian bulk + sparse heavy-tail outliers,
    plus a small per-channel mean (adaLN-like leakage) to exercise the residual
    channel-mean diagnostic.
    """
    X = torch.randn(T, Cin, generator=generator)
    # Sparse outliers.
    mask = torch.rand(T, Cin, generator=generator) < outlier_frac
    X = X + mask * outlier_scale * torch.randn(T, Cin, generator=generator)
    # Mild per-channel mean offset.
    X = X + 0.3 * torch.randn(1, Cin, generator=generator)
    return X


def run_sqnr_harness(
    cfg,
    *,
    model_types: Sequence[str] = ("pixart", "flux"),
    shapes: Optional[Dict[str, List[Tuple[str, int]]]] = None,
    T: int = 2048,
    estimators: Sequence[str] = ESTIMATORS,
    seed: int = 0,
    compare_rotation: bool = True,
    use_wandb: bool = True,
) -> List[Dict]:
    """Run the harness and return a list of per-(model, layer, estimator, rotate)
    result dicts. Logs to wandb if available/enabled.
    """
    shapes = shapes or {m: AUDITED_SHAPES[m] for m in model_types}
    b = int(cfg.act_bits)
    family = str(cfg.clip_family)
    g = _gamma_star_lookup(b, family)
    percentile_p = float(cfg.percentile_p)
    huber_c = float(cfg.huber_c)

    rotate_opts = [True, False] if compare_rotation else [bool(cfg.rotate_act)]

    run = _maybe_init_wandb(cfg, use_wandb)
    results: List[Dict] = []
    for model_type in model_types:
        for label, Cin in shapes[model_type]:
            gen = torch.Generator().manual_seed(seed)
            X = synth_activations(T, Cin, generator=gen)
            for rot in rotate_opts:
                for est in estimators:
                    X_hat, diag = quantize_act(
                        X, b=b, gamma_star=g, estimator=est, rotate_act=rot,
                        percentile_p=percentile_p, huber_c=huber_c, return_diag=True,
                    )
                    rec = {
                        "model": model_type,
                        "layer": label,
                        "in_features": Cin,
                        "estimator": est,
                        "rotate_act": rot,
                        "act_bits": b,
                        "clip_family": family,
                        "sqnr_db": sqnr_db(X, X_hat),
                        **diag,
                    }
                    results.append(rec)
                    if run is not None:
                        run.log({f"{model_type}/{label}/rot{int(rot)}/{est}/sqnr_db": rec["sqnr_db"],
                                 f"{model_type}/{label}/rot{int(rot)}/{est}/kurtosis": rec["kurtosis_excess"],
                                 f"{model_type}/{label}/rot{int(rot)}/{est}/resid_mean_absmax": rec["residual_channel_mean_absmax"]})
    if run is not None:
        run.summary["n_records"] = len(results)
        run.finish()
    return results


def _maybe_init_wandb(cfg, use_wandb: bool):
    if not use_wandb:
        return None
    try:
        import wandb
    except ImportError:
        return None
    try:
        from omegaconf import OmegaConf
        conf = OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else dict(cfg)
    except Exception:
        conf = {k: getattr(cfg, k) for k in ("act_bits", "scale_estimator", "clip_family",
                                             "huber_c", "percentile_p", "rotate_act")
                if hasattr(cfg, k)}
    return wandb.init(
        project=os.environ.get("WANDB_PROJECT", "rods-a-smoke"),
        config=conf,
        mode=os.environ.get("WANDB_MODE", "offline"),  # CPU phase: offline by default
    )


def format_table(results: List[Dict]) -> str:
    """Pretty per-(model, layer, rotate) SQNR table across estimators."""
    ests = []
    for r in results:
        if r["estimator"] not in ests:
            ests.append(r["estimator"])
    # group by (model, layer, in_features, rotate)
    groups: Dict[Tuple, Dict[str, float]] = {}
    order: List[Tuple] = []
    for r in results:
        key = (r["model"], r["layer"], r["in_features"], r["rotate_act"])
        if key not in groups:
            groups[key] = {}
            order.append(key)
        groups[key][r["estimator"]] = r["sqnr_db"]
    head = f"{'model':7} {'layer':9} {'Cin':>6} {'rot':>3} | " + " ".join(f"{e:>10}" for e in ests)
    lines = [head, "-" * len(head)]
    for key in order:
        model, layer, Cin, rot = key
        cells = " ".join(f"{groups[key].get(e, float('nan')):10.2f}" for e in ests)
        lines.append(f"{model:7} {layer:9} {Cin:>6} {int(rot):>3} | {cells}")
    return "\n".join(lines)
