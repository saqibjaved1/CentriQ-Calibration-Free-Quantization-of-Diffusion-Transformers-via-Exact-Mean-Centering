"""CentriQ public API: quantize a diffusers pipeline to W4A4 without calibration data.

    from centriq import load_and_quantize
    pipe = load_and_quantize("flux")          # FLUX.1-schnell, paper config
    pipe("a red fox in the snow", num_inference_steps=4, guidance_scale=0.0).images[0]

This is a thin wrapper around `rods_a.install_w4a4`, called with exactly the arguments
that `scripts/rods_a_w4a4_eval.py` uses for the paper runs. Quantization is simulated
("fake-quant"): weights and activations are rounded to the 4-bit grid and the matmul runs
in 16 bits. The fused INT4 kernels used for latency are in `latency/`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from omegaconf import OmegaConf

from core.model_config import get_model_config, load_pipeline
from rods_a.install_w4a4 import install_w4a4
from rods_a.wrapper import ActQuantConfig

_ROOT = os.path.dirname(os.path.abspath(__file__))

# Configurations behind the paper's main W4A4 table.
PAPER_CONFIGS = {
    "pixart": {"quant": "rods_a_w4a4_pad",        "matmul_fp32": False, "cpu_offload": True},
    "flux":   {"quant": "rods_a_w4a4_fact_flux3", "matmul_fp32": False, "cpu_offload": False},
    "sana":   {"quant": "rods_a_w4a4_fact_sana5", "matmul_fp32": True,  "cpu_offload": True},
}


def load_quant_config(name, **overrides):
    """Load configs/quant/<name>.yaml, with optional overrides (e.g. act_bits=2)."""
    cfg = OmegaConf.load(os.path.join(_ROOT, "configs", "quant", f"{name}.yaml"))
    for k, v in overrides.items():
        cfg[k] = v
    return cfg


def quantize_transformer(transformer, model_type, quant_cfg, device="cuda:0",
                         compute_dtype=torch.float16, matmul_fp32=False):
    """Replace the transformer's linear layers by CentriQ W4A4 layers, in place."""
    actcfg = ActQuantConfig.from_mapping(quant_cfg)
    roles = OmegaConf.to_container(quant_cfg.get("roles_enabled", {}), resolve=True) or {}
    center_roles = OmegaConf.to_container(quant_cfg.get("center_roles_enabled", {}), resolve=True) or {}
    return install_w4a4(
        transformer, actcfg, model_type=model_type,
        weight_bits=int(quant_cfg.get("nbits", 4)), group_size=int(quant_cfg.get("group_size", 64)),
        roles_enabled=roles, center_act_mean=actcfg.center_act_mean,
        center_roles_enabled=center_roles, device=device, compute_dtype=compute_dtype,
        matmul_dtype=torch.float32 if matmul_fp32 else torch.float16,
        strict_rho=False, capture_sqnr=False, backend="fakequant",
        fits_path=None, expected_sha=None, fit_t=bool(quant_cfg.get("fit_t", False)),
        weight_quant_enabled=True,
        tol_outer=float(quant_cfg.get("tol_outer", 1e-3)),
        tol_irls=float(quant_cfg.get("tol_irls", 1e-3)),
        capture_dtype=torch.float32,
    )


def load_and_quantize(model_type, model_id=None, device="cuda:0", **quant_overrides):
    """Load a supported pipeline (pixart | flux | sana) and apply the paper's CentriQ config."""
    spec = PAPER_CONFIGS[model_type]
    mc = get_model_config(model_type)
    pipe = load_pipeline(mc, model_id or mc.default_model_id, device, cpu_offload=spec["cpu_offload"])
    qcfg = load_quant_config(spec["quant"], **quant_overrides)
    quantize_transformer(pipe.transformer, model_type, qcfg, device=device,
                         compute_dtype=mc.default_dtype, matmul_fp32=spec["matmul_fp32"])
    return pipe
