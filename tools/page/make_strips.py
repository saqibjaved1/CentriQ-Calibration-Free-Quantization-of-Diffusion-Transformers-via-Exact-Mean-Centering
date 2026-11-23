#!/usr/bin/env python3
"""Make side-by-side review strips (one row per prompt) for candidates from rank_candidates.py.

    python tools/page/make_strips.py --cands picks.json --runs OUT/flux/bf16 OUT/flux/centriq OUT/flux/convrot \
        --out strips/
"""
import argparse, json, os
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--cands", required=True)
ap.add_argument("--runs", nargs="+", required=True, help="16-bit first, then CentriQ, then others")
ap.add_argument("--out", required=True)
ap.add_argument("--tile", type=int, default=320)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
T = a.tile
for r, c in enumerate(json.load(open(a.cands))):
    i = c["idx"]; s = Image.new("RGB", (T * len(a.runs), T), "white")
    for j, run in enumerate(a.runs):
        s.paste(Image.open(os.path.join(run, "images", f"{i:04d}.png")).convert("RGB").resize((T, T)), (j * T, 0))
    s.save(os.path.join(a.out, f"{r:02d}_idx{i}.jpg"), quality=82)
print("done")
