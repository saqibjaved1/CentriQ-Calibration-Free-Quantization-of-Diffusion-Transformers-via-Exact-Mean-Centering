#!/usr/bin/env python3
"""Export chosen prompts as web images: assets/img/cmp2/<model>_<idx>_<key>.jpg (768 px).

    python tools/page/export_picks.py --model flux --idx 169 848 814 \
        --run ref=OUT/flux/bf16 --run centriq=OUT/flux/centriq --run svd=OUT/flux/svdquant_fp4 \
        --out docs/assets/img/cmp2
Keys used by index.html: ref, centriq, svd, convrot, hadanorm, rot.
"""
import argparse, os
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--idx", type=int, nargs="+", required=True)
ap.add_argument("--run", action="append", required=True, help="key=run_dir")
ap.add_argument("--out", default="assets/img/cmp2")
ap.add_argument("--size", type=int, default=768)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
runs = dict(r.split("=", 1) for r in a.run)
for i in a.idx:
    for k, d in runs.items():
        Image.open(os.path.join(d, "images", f"{i:04d}.png")).convert("RGB") \
            .resize((a.size, a.size), Image.LANCZOS).save(os.path.join(a.out, f"{a.model}_{i}_{k}.jpg"), quality=86)
print("exported", len(a.idx) * len(runs), "images")
