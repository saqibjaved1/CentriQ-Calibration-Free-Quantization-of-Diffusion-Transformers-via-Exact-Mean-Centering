"""FLUX.1-schnell pipeline with random weights built from the model's config files (latency only: identical shapes
and compute to the real checkpoint; image content is meaningless). Used when FLUX_RANDOM=1 because the gated
checkpoint is not available on the A100 node. FLUX_CONFIG_DIR points to the copied configs + tokenizers."""
import os
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler
from transformers import CLIPTextModel, CLIPTextConfig, CLIPTokenizer, T5EncoderModel, T5Config, T5TokenizerFast


def build_random_pipe(transformer=None, dtype=torch.bfloat16, device="cuda"):
    d = os.environ["FLUX_CONFIG_DIR"]
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            if transformer is None:
                transformer = FluxTransformer2DModel.from_config(FluxTransformer2DModel.load_config(f"{d}/transformer"))
            vae = AutoencoderKL.from_config(AutoencoderKL.load_config(f"{d}/vae"))
            te = CLIPTextModel(CLIPTextConfig.from_pretrained(f"{d}/text_encoder"))
            te2 = T5EncoderModel(T5Config.from_pretrained(f"{d}/text_encoder_2"))
    finally:
        torch.set_default_dtype(prev)
    pipe = FluxPipeline(scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(f"{d}/scheduler"), vae=vae,
                        text_encoder=te, tokenizer=CLIPTokenizer.from_pretrained(f"{d}/tokenizer"),
                        text_encoder_2=te2, tokenizer_2=T5TokenizerFast.from_pretrained(f"{d}/tokenizer_2"),
                        transformer=transformer)
    return pipe.to(device)          # components were created in `dtype`; a given (quantized) transformer keeps its own
