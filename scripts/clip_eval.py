"""
CLIP score evaluation for pre-generated diffusion outputs on MJHQ-5K, via
torchmetrics.multimodal.CLIPScore so numbers are directly comparable to SVDQuant /
nunchaku / deepcompressor reported scores (formula max(0, 100 * cos_sim), default
backbone openai/clip-vit-large-patch14). Optionally also scores the original MJHQ
reference images as an upper bound.

Layout: pregenerated_dir/<METHOD>/images/{i:04d}.png; mjhq_dir/images/<hash>.png +
prompts.yaml. Pairing: sorted by hash filename, so {i:04d}.png pairs with sorted
entries[i]. Merges a "clip" key into pregenerated_dir/eval_results.json.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import statistics

import hydra
import torch
from omegaconf import DictConfig
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

from core.eval_io import discover_methods

scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, scripts_dir)
from imagereward_eval import load_entries


# ---------------------------------------------------------------------- #
# CLIP loading + scoring
# ---------------------------------------------------------------------- #

def load_clip(model_id: str, device: str):
    """Instantiate torchmetrics CLIPScore (downloads model on first call)."""
    from torchmetrics.multimodal import CLIPScore

    print(f"Loading torchmetrics CLIPScore: {model_id}")
    metric = CLIPScore(model_name_or_path=model_id).to(device)
    return metric


def truncate_prompts_inplace(entries: list, metric, max_total_tokens: int = 77) -> int:
    """
    Pre-truncate prompts so that after re-tokenization (which re-adds the
    BOS/EOS specials) they fit within CLIP's max_position_embeddings=77.

    CLIP's BPE tokenizer is not idempotent under decode-then-re-encode —
    a 75-token slice can come back as 76 after round-trip — so we
    iteratively shrink the budget until the actual re-tokenized length
    is safe.

    Returns the number of prompts that were shortened.
    """
    tokenizer = metric.processor.tokenizer
    shortened = 0

    for entry in entries:
        original = entry["prompt"]
        if len(tokenizer(original, add_special_tokens=True).input_ids) <= max_total_tokens:
            continue

        target = max_total_tokens - 2  # initial budget after BOS/EOS
        candidate = original
        while target > 0:
            ids = tokenizer(
                original, add_special_tokens=False,
                truncation=True, max_length=target,
            ).input_ids
            candidate = tokenizer.decode(ids, skip_special_tokens=True)
            if len(tokenizer(candidate, add_special_tokens=True).input_ids) <= max_total_tokens:
                break
            target -= 1

        entry["prompt"] = candidate
        shortened += 1

    return shortened


@torch.no_grad()
def score_method(metric, entries: list, get_path_fn, device: str) -> dict:
    """
    Score (prompt, image) pairs using torchmetrics CLIPScore.

    Formula (per torchmetrics): max(0, 100 * cos_similarity).  Identical to
    what SVDQuant / deepcompressor report, so absolute values are directly
    comparable.  Typical good range ~20-35.

    Args:
        metric: a torchmetrics.multimodal.CLIPScore instance on `device`
        entries: list of {"filename": ..., "prompt": ...}
        get_path_fn: callable(i, entry) → absolute image path
        device: torch device

    Returns:
        dict with keys: scores, skipped, mean/std/median/min/max/n.
    """
    scores = []
    skipped = 0

    for i, entry in enumerate(tqdm(entries, leave=False)):
        img_path = get_path_fn(i, entry)
        if img_path is None or not os.path.exists(img_path):
            skipped += 1
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  Warning: could not open {img_path}: {e}")
            skipped += 1
            continue

        # torchmetrics CLIPScore expects uint8 tensor of shape (3, H, W) in [0, 255]
        img_tensor = pil_to_tensor(image).to(device)

        # Per-image score: reset → update with single pair → compute returns
        # that pair's score (compute() = sum/n, and here n=1).
        metric.reset()
        metric.update(img_tensor, entry["prompt"])
        score = metric.compute().item()
        scores.append(float(score))

    result = {"scores": scores, "skipped": skipped}
    if scores:
        result["mean"]   = statistics.mean(scores)
        result["std"]    = statistics.stdev(scores) if len(scores) > 1 else 0.0
        result["median"] = statistics.median(scores)
        result["min"]    = min(scores)
        result["max"]    = max(scores)
        result["n"]      = len(scores)
    else:
        result.update({"mean": None, "std": None, "median": None,
                       "min": None, "max": None, "n": 0})
    return result


# ---------------------------------------------------------------------- #
# Output
# ---------------------------------------------------------------------- #

def print_table(results: dict) -> None:
    header = (
        f"{'Method':<20}  {'n':>6}  {'Mean':>7}  {'Std':>7}  "
        f"{'Median':>7}  {'Min':>7}  {'Max':>7}  {'Skipped':>7}"
    )
    print("\n" + "=" * len(header))
    print("CLIP Scores (cosine similarity × 100, higher is better)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for method, stats in results.items():
        if stats["n"] == 0:
            print(
                f"  {method:<18}  {'N/A':>6}  {'--':>7}  {'--':>7}  "
                f"{'--':>7}  {'--':>7}  {'--':>7}  {stats['skipped']:>7}"
            )
        else:
            print(
                f"  {method:<18}  {stats['n']:>6}  "
                f"{stats['mean']:>7.4f}  {stats['std']:>7.4f}  "
                f"{stats['median']:>7.4f}  {stats['min']:>7.4f}  "
                f"{stats['max']:>7.4f}  {stats['skipped']:>7}"
            )
    print("=" * len(header))


# ---------------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------------- #

@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="clip_eval",
)
def main(cfg: DictConfig) -> None:
    output_file = cfg.output_file or os.path.join(
        cfg.pregenerated_dir, "eval_results.json"
    )

    prompts_path = os.path.join(cfg.mjhq_dir, "prompts.yaml")
    if not os.path.exists(prompts_path):
        raise FileNotFoundError(f"prompts.yaml not found at {prompts_path}")

    images_dir = os.path.join(cfg.mjhq_dir, "images")
    if cfg.score_reference and not os.path.isdir(images_dir):
        raise FileNotFoundError(
            f"Reference images/ directory not found at {images_dir}"
        )

    cfg_methods = None if cfg.methods is None else list(cfg.methods)
    methods = cfg_methods or discover_methods(cfg.pregenerated_dir)
    if not methods:
        raise RuntimeError(
            f"No method subdirs with 'images/' found in {cfg.pregenerated_dir}. "
            "Pass methods=[...] explicitly."
        )

    print(f"Methods to evaluate: {methods}")
    if cfg.score_reference:
        print("Also scoring reference images from mjhq_dir/images/")

    entries = load_entries(prompts_path, cfg.num_samples)
    print(f"Loaded {len(entries)} entries from prompts.yaml")

    print(f"\nLoading CLIP: {cfg.model_id} on {cfg.device}")
    metric = load_clip(cfg.model_id, cfg.device)

    n_shortened = truncate_prompts_inplace(entries, metric)
    if n_shortened:
        print(f"Truncated {n_shortened} / {len(entries)} prompts to fit CLIP's 77-token limit.")

    results = {}

    if cfg.score_reference:
        print("\nScoring reference images ...")
        results["reference"] = score_method(
            metric, entries,
            get_path_fn=lambda i, e: os.path.join(images_dir, e["filename"]),
            device=cfg.device,
        )
        ref_n = results["reference"]["n"]
        ref_skip = results["reference"]["skipped"]
        print(f"  Done: {ref_n} scored, {ref_skip} skipped")

    for method in methods:
        method_dir = os.path.join(cfg.pregenerated_dir, method, "images")
        if not os.path.isdir(method_dir):
            print(f"\n[{method}] Skipping — directory not found: {method_dir}")
            results[method] = {"n": 0, "skipped": len(entries), "scores": []}
            continue

        print(f"\nScoring [{method}] from {method_dir} ...")
        results[method] = score_method(
            metric, entries,
            get_path_fn=lambda i, e, d=method_dir: os.path.join(d, f"{i:04d}.png"),
            device=cfg.device,
        )
        n = results[method]["n"]
        skip = results[method]["skipped"]
        print(f"  Done: {n} scored, {skip} skipped")

    # Merge into shared eval_results.json
    existing = {}
    if os.path.exists(output_file):
        with open(output_file) as f:
            existing = json.load(f)

    if "clip" in existing and not cfg.force:
        print(
            f"\nNote: 'clip' key already present in {output_file}. "
            f"Overwriting (pass force=false to disable)."
        )

    existing["clip"] = {
        "config": {
            "model_id": cfg.model_id,
            "mjhq_dir": cfg.mjhq_dir,
            "num_samples": cfg.num_samples,
            "score_reference": cfg.score_reference,
            "methods": methods,
            "scale": "cosine_similarity * 100",
        },
        "results": results,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nResults written to {output_file}")

    # Per-method copies under {method}/eval/
    for method_name, method_result in results.items():
        if method_name == "reference":
            continue
        method_eval_dir = os.path.join(cfg.pregenerated_dir, method_name, "eval")
        os.makedirs(method_eval_dir, exist_ok=True)
        per_method = {
            "config": existing["clip"]["config"],
            "results": {method_name: method_result},
        }
        per_method_path = os.path.join(method_eval_dir, "clip_results.json")
        with open(per_method_path, "w") as f:
            json.dump(per_method, f, indent=2)
        print(f"  [saved] {per_method_path}")

    print_table(results)


if __name__ == "__main__":
    main()
