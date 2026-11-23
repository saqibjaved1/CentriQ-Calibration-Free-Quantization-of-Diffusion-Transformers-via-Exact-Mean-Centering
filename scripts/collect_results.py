#!/usr/bin/env python3
"""Print one table row per run found under an outputs root.

    python scripts/collect_results.py outputs/
Reads <root>/<model>/<arm>/results.json (FID, ImageReward) and paired_vs_fp16.json (PSNR, LPIPS).
PSNR is reported as the mean and the median over images (the paper uses the median for PixArt-Sigma).
"""
import glob
import json
import os
import statistics
import sys


def main(root):
    rows = []
    for res in sorted(glob.glob(os.path.join(root, "*", "*", "results.json"))):
        arm_dir = os.path.dirname(res)
        model, arm = arm_dir.split(os.sep)[-2:]
        r = json.load(open(res))
        p = os.path.join(arm_dir, "paired_vs_fp16.json")
        psnr = lpips = psnr_med = None
        if os.path.exists(p):
            q = json.load(open(p))
            psnr = statistics.mean(q["psnr_scores"]); psnr_med = statistics.median(q["psnr_scores"])
            lpips = statistics.mean(q["lpips_scores"])
        rows.append((model, arm, r.get("fid"), r.get("imagereward_mean"), psnr, psnr_med, lpips, r.get("fid_n")))
    f = lambda v, d: "-" if v is None else f"{v:.{d}f}"
    print(f"{'model':8} {'arm':22} {'FID':>8} {'IR':>7} {'PSNR':>7} {'PSNRmed':>8} {'LPIPS':>7} {'n':>5}")
    for m, a, fid, ir, ps, pm, lp, n in rows:
        print(f"{m:8} {a:22} {f(fid,2):>8} {f(ir,3):>7} {f(ps,2):>7} {f(pm,2):>8} {f(lp,3):>7} {n or '-':>5}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "outputs")
