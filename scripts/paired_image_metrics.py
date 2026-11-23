#!/usr/bin/env python3
"""
Paired per-image LPIPS/PSNR vs the FP16 arm (U1 attribution, pad-vs-kron task).

Pairs `{i:04d}.png` by index across two image dirs (dev-slice arms share the
per-prompt seed map, so index i is the SAME prompt + SAME initial noise in every arm).
LPIPS: repo headline convention (scripts/lpips_eval.py) — 256x256 resize, net=alex,
normalize=True; `--resize 0` switches to full-res. PSNR: full-res uint8, max=255
(scripts/rods_a_int4_images.py convention). `--patch_results` patches the arm's
results.json in place: `lpips_vs_fp16` / `psnr_vs_fp16` summary slots plus the
per-prompt `*_scores` lists consumed by scripts/ablation_significance.py.
Run via RunAI (never the front node).
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image

from core.metric_core import LpipsScorer as _LPIPS, mse_uint8  # _LPIPS: external import in lpips_selected_pairs


def psnr_uint8(a: np.ndarray, b: np.ndarray) -> float:
    mse = mse_uint8(a, b)
    return 100.0 if mse == 0.0 else 10.0 * math.log10(255.0 ** 2 / mse)


def pair_paths(dir_a, dir_b, n):
    pairs = []
    for i in range(int(n)):
        pa = os.path.join(dir_a, f"{i:04d}.png")
        pb = os.path.join(dir_b, f"{i:04d}.png")
        if not os.path.exists(pa) or not os.path.exists(pb):
            raise FileNotFoundError(f"missing pair at idx {i}: {pa} / {pb} "
                                    f"(paired metrics need a complete index range)")
        pairs.append((pa, pb))
    return pairs


def compute_pair_metrics(pairs, device, net="alex", resize=256):
    lp = _LPIPS(device, net=net, resize=resize)
    lpips_scores, psnr_scores = [], []
    for k, (pa, pb) in enumerate(pairs):
        im_a = Image.open(pa).convert("RGB")
        im_b = Image.open(pb).convert("RGB")
        lpips_scores.append(lp(im_a, im_b))
        psnr_scores.append(psnr_uint8(np.asarray(im_a), np.asarray(im_b)))
        if (k + 1) % 100 == 0:
            print(f"[pair] {k + 1}/{len(pairs)}", flush=True)
    return {"lpips_scores": lpips_scores, "psnr_scores": psnr_scores,
            "n": len(pairs)}


def patch_results(results_path, lpips_scores, psnr_scores, meta):
    with open(results_path) as f:
        res = json.load(f)
    for name, scores in (("lpips_vs_fp16", lpips_scores),
                         ("psnr_vs_fp16", psnr_scores)):
        st = np.asarray(scores, float)
        res[name] = {"mean": float(st.mean()), "std": float(st.std()),
                     "median": float(np.median(st)), "n": int(len(st))}
        res[f"{name}_scores"] = [float(v) for v in scores]
    res["paired_image_metrics_meta"] = meta
    with open(results_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[patch] updated {results_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir_a", required=True, help="arm images dir")
    ap.add_argument("--dir_b", required=True, help="FP16 reference images dir")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--net", default="alex")
    ap.add_argument("--resize", type=int, default=256,
                    help="LPIPS resize (repo convention 256); 0 = full-res")
    ap.add_argument("--out", default=None, help="standalone metrics json")
    ap.add_argument("--patch_results", default=None,
                    help="arm results.json to patch in place")
    args = ap.parse_args()

    pairs = pair_paths(args.dir_a, args.dir_b, args.n)
    m = compute_pair_metrics(pairs, args.device, net=args.net, resize=args.resize)
    meta = {"dir_a": args.dir_a, "dir_b": args.dir_b, "n": m["n"],
            "lpips_net": args.net, "lpips_resize": args.resize,
            "psnr": "full-res uint8 max=255"}
    print(f"[done] n={m['n']}  LPIPS mean={np.mean(m['lpips_scores']):.4f} "
          f"median={np.median(m['lpips_scores']):.4f}  "
          f"PSNR mean={np.mean(m['psnr_scores']):.2f} dB "
          f"median={np.median(m['psnr_scores']):.2f} dB")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({**meta, **m}, f, indent=2)
        print(f"[done] wrote {args.out}")
    if args.patch_results:
        patch_results(args.patch_results, m["lpips_scores"], m["psnr_scores"], meta)


if __name__ == "__main__":
    main()
