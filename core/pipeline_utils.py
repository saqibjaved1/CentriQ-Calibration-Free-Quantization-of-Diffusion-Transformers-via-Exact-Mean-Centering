"""
Shared PixArt-Sigma pipeline helpers used across evaluation scripts.
"""

import torch
from diffusers import PixArtSigmaPipeline, PixArtTransformer2DModel


def load_pipeline(model_id: str, device: str, verbose: bool = False) -> PixArtSigmaPipeline:
    """Load PixArt-Sigma pipeline in FP16. Progress bar shown only if verbose=True."""
    print(f"Loading PixArt-Sigma pipeline from {model_id} ...")
    pipe = PixArtSigmaPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=not verbose)
    return pipe


def reload_transformer(pipe: PixArtSigmaPipeline, model_id: str, device: str) -> None:
    """
    Replace pipe.transformer with a freshly loaded FP16 transformer.
    The T5 encoder, VAE, tokenizer, and scheduler are left untouched —
    only the DiT backbone is reloaded (saves ~10 GB vs reloading the full pipeline).
    """
    print(f"\nReloading transformer from {model_id} ...")
    pipe.transformer = PixArtTransformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to(device)
    pipe.transformer.eval()
