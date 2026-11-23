"""Shared image-generation loop for the diffusion eval / generation entrypoints
(P1 gen/eval seam, c6a).

`generate_into_dir` is the single source of truth for the generator convention
used by fid_eval.generate_category and rods_a_w4a4_eval._generate. The convention
is load-bearing — a divergence desyncs seeds and changes every image — so it lives
here once and both callers wrap it, keeping generation byte-identical by
construction. Callers own their own logging and return shape.

Convention:
  * one torch.Generator seeded once with args.seed, advanced across the run;
  * an already-present VALID image is skipped and does NOT advance the generator
    (so a partial-resume run is not bit-identical to a from-scratch run);
  * paired_seed=True reseeds the generator per image from entry["seed"] (the
    persisted per-prompt seed map), making prompt->noise independent of arm
    ordering / resume-skips — the precondition for the paired significance tests.
"""
from __future__ import annotations

import os
import time
from collections import namedtuple

import torch

from core.model_config import call_pipeline
from core.mjhq_data import _image_file_ok


GenResult = namedtuple("GenResult", ["paths", "n_generated", "n_skipped", "elapsed"])


def generate_into_dir(pipe, model_config, entries, gen_dir, args, *,
                      overwrite=False, paired_seed=False, warn_truncated=False):
    """Generate images for `entries` into `gen_dir/{i:04d}.png`.

    Returns GenResult(paths, n_generated, n_skipped, elapsed). See the module
    docstring for the generator convention. overwrite=True regenerates present
    images; warn_truncated=True prints a notice when an existing file is present but
    is not a valid image (the fid_eval convention; rods_a_w4a4_eval did not print it).
    """
    os.makedirs(gen_dir, exist_ok=True)
    paths = []
    n_generated = 0
    n_skipped = 0
    t0 = time.perf_counter()
    gen = torch.Generator(device=args.device).manual_seed(args.seed)
    for i, entry in enumerate(entries):
        p = os.path.join(gen_dir, f"{i:04d}.png")
        if not overwrite and _image_file_ok(p):
            paths.append(p)
            n_skipped += 1
            continue
        if warn_truncated and os.path.exists(p):
            print(f"  Warning: truncated image {p}, regenerating")
        if paired_seed:
            gen = torch.Generator(device=args.device).manual_seed(int(entry["seed"]))
        out = call_pipeline(pipe, model_config, entry["prompt"], args, generator=gen)
        out.images[0].save(p)
        paths.append(p)
        n_generated += 1
    elapsed = time.perf_counter() - t0
    return GenResult(paths=paths, n_generated=n_generated, n_skipped=n_skipped, elapsed=elapsed)
