"""Shared IO helpers for the pre-generated-image evaluation scripts
(lpips_eval, psnr_eval, imagereward_eval, eval_pregenerated, fid_eval).

The expected on-disk layout is the one produced by the inference-path
entrypoints (demo, fid_eval generation phase, compare_eval):

    pregenerated_dir/
        <method>/images/0000.png ... NNNN.png
        <method>/eval/  (per-metric JSON files, optional)
        eval_results.json  (aggregated per-metric scores)
"""
from __future__ import annotations

import json
import os
from typing import List, Optional

from omegaconf import OmegaConf


def discover_methods(
    pregenerated_dir: str,
    exclude: Optional[str] = None,
) -> List[str]:
    """Return sorted list of method subdirs of `pregenerated_dir` that contain
    an `images/` subdirectory.

    If `exclude` is given (e.g. the LPIPS / PSNR reference method), that
    method is omitted from the result.
    """
    methods: List[str] = []
    for name in sorted(os.listdir(pregenerated_dir)):
        if exclude is not None and name == exclude:
            continue
        if os.path.isdir(os.path.join(pregenerated_dir, name, "images")):
            methods.append(name)
    return methods


# Per-method nested dicts that accumulate across separate runs (one method per
# invocation). They must be deep-merged key-by-key, otherwise a shallow update
# would replace the whole segment on disk with only the method(s) computed in
# the current process, wiping earlier ones.
DEFAULT_PER_METHOD_KEYS = (
    "fid", "lpips", "transformer_mb", "generation_time_s", "quant_time_s",
)


def merge_results(results_path, new_entries, per_method_keys=DEFAULT_PER_METHOD_KEYS):
    """Deep-merge `new_entries` into the eval_results.json at `results_path` and
    write it back; returns the merged dict.

    For each per-method key: if the new value is None but disk has one, keep disk
    (nothing computed this run, e.g. --skip_fid, must not blank the segment); if
    both are dicts, union them (new wins on collision). All other keys shallow-
    update. OmegaConf ListConfig/DictConfig values are converted to plain Python
    before the json.dump (args.categories / args.ignore_layers would otherwise
    raise). A corrupt existing file is treated as empty.
    """
    existing = {}
    if os.path.exists(results_path):
        try:
            with open(results_path) as f:
                existing = json.load(f)
        except json.JSONDecodeError:
            existing = {}
    for key in per_method_keys:
        if key not in new_entries:
            continue
        new_val = new_entries[key]
        old_val = existing.get(key)
        if new_val is None and old_val is not None:
            new_entries = {k: v for k, v in new_entries.items() if k != key}
        elif isinstance(new_val, dict) and isinstance(old_val, dict):
            merged = dict(old_val)
            merged.update(new_val)
            new_entries = {**new_entries, key: merged}
    existing.update(new_entries)
    existing = OmegaConf.to_container(OmegaConf.create(existing), resolve=True)
    with open(results_path, "w") as f:
        json.dump(existing, f, indent=2)
    return existing


def write_method_meta(method_dir, meta):
    """Write a per-method `meta.json` sidecar into `method_dir` (created if needed),
    so the run config can be recovered from the folder. Non-JSON values are
    stringified (default=str), matching the fid_eval convention."""
    os.makedirs(method_dir, exist_ok=True)
    with open(os.path.join(method_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
