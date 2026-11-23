#!/usr/bin/env python3
"""Rank prompts where CentriQ stays closest to the 16-bit image relative to other methods.

Each run directory holds images/{i:04d}.png and paired_vs_bf16.json (per-image LPIPS vs the
16-bit run, from the release's scripts/paired_image_metrics.py). All runs must share prompts and
seeds, so image i is the same prompt and initial noise in every run.

    python tools/page/rank_candidates.py --centriq OUT/flux/centriq \
        --others OUT/flux/svdquant_fp4 OUT/flux/convrot OUT/flux/hadanorm --top 24 --out picks.json
Score = min(LPIPS of the other methods) - LPIPS of CentriQ (higher = clearer CentriQ advantage).
"""
import argparse, json, os

def lpips(run):
    return json.load(open(os.path.join(run, "paired_vs_bf16.json")))["lpips_scores"]

ap = argparse.ArgumentParser()
ap.add_argument("--centriq", required=True)
ap.add_argument("--others", nargs="+", required=True)
ap.add_argument("--n", type=int, default=None, help="only the first n prompts")
ap.add_argument("--top", type=int, default=24)
ap.add_argument("--out", default="candidates.json")
a = ap.parse_args()
c = lpips(a.centriq); o = {r: lpips(r) for r in a.others}
n = a.n or min([len(c)] + [len(v) for v in o.values()])
print("mean LPIPS vs 16-bit:", {os.path.basename(a.centriq): round(sum(c[:n]) / n, 3)},
      {os.path.basename(r): round(sum(v[:n]) / n, 3) for r, v in o.items()})
rows = sorted(range(n), key=lambda i: -(min(v[i] for v in o.values()) - c[i]))[:a.top]
json.dump([{"idx": i, "lpips_centriq": c[i], "lpips_others": {r: o[r][i] for r in o}} for i in rows],
          open(a.out, "w"), indent=1)
print("wrote", a.out, [i for i in rows])
