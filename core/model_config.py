"""
Model-type registry for cfq-diffusion.

Encapsulates all model-specific constants and pipeline/inference helpers so that
scripts never contain raw if/else branches over model_type.

Supported model_type strings:
  "pixart" — PixArt-Sigma-XL (DiT, float16, 20 steps, CFG)
  "flux"   — FLUX.1-schnell  (MM-DiT, bfloat16, 1 step, no CFG)
  "sana"   — SANA 1.6B 1024px (Linear-DiT, bfloat16, 20 steps, CFG 4.5)
"""

from dataclasses import dataclass, field
from typing import List

import torch


@dataclass
class ModelConfig:
    model_type: str
    default_model_id: str
    default_dtype: torch.dtype      # float16 (PixArt) vs bfloat16 (FLUX)
    default_steps: int              # 20 vs 1
    default_guidance_scale: float   # 7.5 vs 0.0 (FLUX schnell: distilled, no CFG)
    default_height: int
    default_width: int
    ignore_layers: List[str]        # leaf layer names to keep in FP16


# ------------------------------------------------------------------ #
# Registry entries
# ------------------------------------------------------------------ #

_PIXART_CONFIG = ModelConfig(
    model_type="pixart",
    default_model_id="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
    default_dtype=torch.float16,
    default_steps=20,
    default_guidance_scale=7.5,
    default_height=1024,
    default_width=1024,
    ignore_layers=["proj_out", "linear_1", "linear_2", "linear", "attn_2.to_k", "attn_2.to_v"],
)

_FLUX_CONFIG = ModelConfig(
    model_type="flux",
    default_model_id="black-forest-labs/FLUX.1-schnell",
    default_dtype=torch.bfloat16,
    default_steps=1,
    default_guidance_scale=0.0,
    default_height=1024,
    default_width=1024,
    ignore_layers=["proj_out", "linear_1", "linear_2", "linear", "context_embedder", "x_embedder"],
)

_SANA_CONFIG = ModelConfig(
    model_type="sana",
    default_model_id="Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
    default_dtype=torch.bfloat16,
    default_steps=20,
    default_guidance_scale=4.5,
    default_height=1024,
    default_width=1024,
    ignore_layers=["proj_out", "linear_1", "linear_2", "linear"],
)

_REGISTRY = {
    "pixart": _PIXART_CONFIG,
    "flux":   _FLUX_CONFIG,
    "sana":   _SANA_CONFIG,
}


def get_model_config(model_type: str) -> ModelConfig:
    if model_type not in _REGISTRY:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Choose from: {list(_REGISTRY)}"
        )
    return _REGISTRY[model_type]


# ------------------------------------------------------------------ #
# Pipeline helpers
# ------------------------------------------------------------------ #

def _flux_gpu_id(device: str) -> int:
    """Extract integer GPU index from a device string like 'cuda:0'."""
    return int(device.split(":")[-1]) if ":" in device else 0


def load_pipeline(model_config: ModelConfig, model_id: str, device: str, verbose: bool = False,
                  cpu_offload: bool = True):
    """Load the appropriate diffusers pipeline in the correct dtype.

    `cpu_offload` (FLUX only): when True (default) FLUX uses
    `enable_model_cpu_offload` (low VRAM, but leaves weights on CPU). The RoDS-A W4A4
    fakequant track sets it False and puts the pipeline on the GPU directly — fakequant
    keeps full-precision dequantized weights (~35 GB total: transformer ~24 GB + T5 ~9 GB),
    which fits one 80 GB A100, and avoids the offload-hook vs in-place-module-replacement
    device mismatch (weights must be on `device` before install_w4a4 / prepare_weight)."""
    dtype = model_config.default_dtype
    if model_config.model_type == "pixart":
        from diffusers import PixArtSigmaPipeline
        print(f"Loading PixArt-Sigma pipeline from {model_id} ...")
        pipe = PixArtSigmaPipeline.from_pretrained(
            model_id, torch_dtype=dtype, use_safetensors=True,
        )
        pipe = pipe.to(device)
    elif model_config.model_type == "flux":
        from diffusers import FluxPipeline
        print(f"Loading FLUX pipeline from {model_id} ...")
        pipe = FluxPipeline.from_pretrained(
            model_id, torch_dtype=dtype, use_safetensors=True,
        )
        # FLUX.1-schnell is ~34 GB in bfloat16 (transformer 24 GB + T5 9 GB + VAE/CLIP).
        # enable_model_cpu_offload keeps each component in CPU RAM and moves it to GPU
        # only during its forward pass — peak VRAM is the largest single component
        # (~24 GB transformer for FP16, ~6 GB after SINQ quantization).
        # SINQ quantization works transparently: BasePatch accesses transformer weights
        # from CPU RAM and stores W_q on GPU directly.
        # The RoDS-A W4A4 fakequant track passes cpu_offload=False: in-place module
        # replacement (install_w4a4) plus the offload hooks leave operands on mixed
        # devices, and prepare_weight needs the weight already on `device`. Putting the
        # whole pipeline on the GPU (fits 80 GB) sidesteps both.
        if cpu_offload:
            pipe.enable_model_cpu_offload(gpu_id=_flux_gpu_id(device))
        else:
            pipe = pipe.to(device)
    elif model_config.model_type == "sana":
        from diffusers import SanaPipeline
        print(f"Loading SANA pipeline from {model_id} ...")
        pipe = SanaPipeline.from_pretrained(
            model_id, torch_dtype=dtype, use_safetensors=True,
        )
        pipe = pipe.to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_config.model_type}")

    pipe.set_progress_bar_config(disable=not verbose)
    return pipe


def reload_transformer(pipe, model_config: ModelConfig, model_id: str, device: str) -> None:
    """
    Replace pipe.transformer with a freshly loaded copy in the correct dtype.
    Keeps all other pipeline components (text encoders, VAE, scheduler) untouched.
    """
    dtype = model_config.default_dtype
    print(f"\nReloading transformer from {model_id} ...")
    if model_config.model_type == "pixart":
        from diffusers import PixArtTransformer2DModel
        pipe.transformer = PixArtTransformer2DModel.from_pretrained(
            model_id, subfolder="transformer",
            torch_dtype=dtype, use_safetensors=True,
        ).to(device)
    elif model_config.model_type == "flux":
        from diffusers import FluxTransformer2DModel
        pipe.transformer = FluxTransformer2DModel.from_pretrained(
            model_id, subfolder="transformer",
            torch_dtype=dtype, use_safetensors=True,
        )
        # Re-register CPU offload hooks for the new transformer.
        # Replacing pipe.transformer removes the old hooks; calling
        # enable_model_cpu_offload again re-attaches them to all components.
        pipe.enable_model_cpu_offload(gpu_id=_flux_gpu_id(device))
    elif model_config.model_type == "sana":
        from diffusers import SanaTransformer2DModel
        pipe.transformer = SanaTransformer2DModel.from_pretrained(
            model_id, subfolder="transformer",
            torch_dtype=dtype, use_safetensors=True,
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_config.model_type}")
    pipe.transformer.eval()


def call_pipeline(pipe, model_config: ModelConfig, prompt: str, args, generator=None):
    """
    Call the pipeline with model-type-appropriate keyword arguments.

    FLUX.1-schnell requires guidance_scale=0.0 (distilled, no CFG).
    SANA accepts guidance_scale (default 4.5) and max_sequence_length (default 300).
    PixArt-Sigma uses CFG internally and does not accept guidance_scale the same way.
    """
    kwargs = dict(
        prompt=prompt,
        num_inference_steps=args.num_inference_steps,
        height=args.height,
        width=args.width,
    )
    if generator is not None:
        kwargs["generator"] = generator
    if model_config.model_type == "flux":
        kwargs["guidance_scale"] = args.guidance_scale
    elif model_config.model_type == "sana":
        kwargs["guidance_scale"] = args.guidance_scale
        kwargs["max_sequence_length"] = getattr(args, "max_sequence_length", 300)
    return pipe(**kwargs)
