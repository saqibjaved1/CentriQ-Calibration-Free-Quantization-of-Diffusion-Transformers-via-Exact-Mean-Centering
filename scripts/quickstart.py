#!/usr/bin/env python3
"""Quantize a DiT with CentriQ (no calibration data) and generate one image.

    python scripts/quickstart.py --model flux --prompt "a lighthouse at dusk, oil painting"
    python scripts/quickstart.py --model pixart --act_bits 2      # W4A2 (simulated)
    python scripts/quickstart.py --model flux --bf16               # 16-bit reference
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from centriq import load_and_quantize
from core.model_config import get_model_config, load_pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="flux", choices=["flux", "pixart", "sana"])
    ap.add_argument("--prompt", default="A cozy cabin in a snowy forest at night, warm light in the windows")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weight_bits", type=int, default=4)
    ap.add_argument("--act_bits", type=int, default=4)
    ap.add_argument("--bf16", action="store_true", help="skip quantization (16-bit reference)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    mc = get_model_config(a.model)
    t0 = time.time()
    if a.bf16:
        pipe = load_pipeline(mc, mc.default_model_id, a.device, cpu_offload=False)
    else:
        pipe = load_and_quantize(a.model, device=a.device, nbits=a.weight_bits, act_bits=a.act_bits)
    print(f"model ready in {time.time() - t0:.0f}s")

    steps = 4 if a.model == "flux" else mc.default_steps   # paper: FLUX.1-schnell 4 steps
    kw = dict(num_inference_steps=steps, guidance_scale=mc.default_guidance_scale,
              height=1024, width=1024, generator=torch.Generator(a.device).manual_seed(a.seed))
    img = pipe(a.prompt, **kw).images[0]
    tag = "bf16" if a.bf16 else f"centriq_w{a.weight_bits}a{a.act_bits}"
    out = a.out or f"{a.model}_{tag}.png"
    img.save(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
