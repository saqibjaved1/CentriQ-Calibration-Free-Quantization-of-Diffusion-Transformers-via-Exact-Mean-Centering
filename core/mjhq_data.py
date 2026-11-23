"""MJHQ-30K / MJHQ-5000 (svdquant) dataset + prompt loaders — single source of truth.

fid_eval.py re-exports these names for backward compatibility, and the standalone
generation entry (scripts/generate.py) imports them directly.

Three prompt sources are supported:

  * MJHQ-30K, per category   -> download_mjhq + load_prompts_per_category  (FID reference set)
  * MJHQ-5000 (svdquant)     -> download_mjhq_svdquant + load_prompts_svdquant (write-once test set)
  * dev-1k firewall slice    -> load_prompts_dev (built once by scripts/build_dev_slice.py)

The on-disk image layout these feed into is documented in core/eval_io.py.
"""
from __future__ import annotations

import os
import json
import zipfile
from collections import defaultdict

from PIL import Image
from huggingface_hub import hf_hub_download


MJHQ_REPO = "playgroundai/MJHQ-30K"
SVDQUANT_REPO = "mit-han-lab/svdquant-datasets"


# ---------------------------------------------------------------------- #
# Dataset helpers
# ---------------------------------------------------------------------- #

def download_mjhq(mjhq_dir: str) -> tuple:
    """
    Download MJHQ-30K from HuggingFace if not already present.
    Returns (imgs_dir, meta_path).

    Directory layout after extraction:
        mjhq_dir/
        ├── meta_data.json
        ├── animals/
        ├── art/
        └── ...   (category folders sit directly under mjhq_dir, not in a sub-folder)
    """
    os.makedirs(mjhq_dir, exist_ok=True)
    meta_path = os.path.join(mjhq_dir, "meta_data.json")
    imgs_dir = mjhq_dir

    if not os.path.exists(meta_path):
        print("Downloading meta_data.json ...")
        hf_hub_download(
            MJHQ_REPO, "meta_data.json",
            local_dir=mjhq_dir,
            repo_type="dataset",
        )

    if not os.path.exists(os.path.join(mjhq_dir, "animals")):
        zip_path = os.path.join(mjhq_dir, "mjhq30k_imgs.zip")
        if not os.path.exists(zip_path):
            print("Downloading mjhq30k_imgs.zip (3.6 GB) ...")
            hf_hub_download(
                MJHQ_REPO, "mjhq30k_imgs.zip",
                local_dir=mjhq_dir,
                repo_type="dataset",
            )
        print("Extracting mjhq30k_imgs.zip ...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(mjhq_dir)

    return imgs_dir, meta_path


def download_mjhq_svdquant(mjhq_dir: str) -> tuple:
    """
    Download MJHQ-5000.zip from mit-han-lab/svdquant-datasets (the exact 5k
    subset used in the SVDQuant paper) if not already present.
    Returns (images_dir, prompts_yaml_path).

    Layout after extraction:
        mjhq_dir/
        ├── images/          ← 5000 flat PNG files named by hash
        ├── prompts.yaml     ← {hash: prompt} mapping (no categories)
        ├── canny_images/
        ├── cropped_images/
        ├── depth_images/
        └── mask_images/
    """
    import shutil, tempfile
    os.makedirs(mjhq_dir, exist_ok=True)
    images_dir = os.path.join(mjhq_dir, "images")
    prompts_path = os.path.join(mjhq_dir, "prompts.yaml")

    if not os.path.isdir(images_dir) or not os.path.exists(prompts_path):
        zip_path = os.path.join(mjhq_dir, "MJHQ-5000.zip")
        if not os.path.exists(zip_path):
            print("Downloading MJHQ-5000.zip (~11.7 GB) from mit-han-lab/svdquant-datasets ...")
            hf_hub_download(
                SVDQUANT_REPO,
                "MJHQ-5000.zip",
                local_dir=mjhq_dir,
                repo_type="dataset",
            )
        print("Extracting MJHQ-5000.zip ...")
        with zipfile.ZipFile(zip_path, "r") as z:
            common = os.path.commonprefix(z.namelist()).rstrip("/")
            if common and all(
                n.startswith(common + "/") or n.rstrip("/") == common
                for n in z.namelist()
            ):
                with tempfile.TemporaryDirectory() as tmp:
                    z.extractall(tmp)
                    src = os.path.join(tmp, common)
                    for item in os.listdir(src):
                        dst = os.path.join(mjhq_dir, item)
                        if not os.path.exists(dst):
                            shutil.move(os.path.join(src, item), dst)
            else:
                z.extractall(mjhq_dir)

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(
            f"'images/' directory not found in {mjhq_dir} after extraction."
        )
    if not os.path.exists(prompts_path):
        raise FileNotFoundError(
            f"'prompts.yaml' not found in {mjhq_dir} after extraction."
        )

    return images_dir, prompts_path


# ---------------------------------------------------------------------- #
# Prompt loaders
# ---------------------------------------------------------------------- #

def load_prompts_per_category(meta_path: str, categories: list, n: int, seed: int) -> dict:
    """
    Parse meta_data.json and return {category: [{"filename": ..., "prompt": ...}, ...]}.
    Takes the first `n` entries in deterministic order (sorted by filename).
    """
    with open(meta_path) as f:
        meta = json.load(f)

    cat_entries = defaultdict(list)
    for fname, info in meta.items():
        cat = info.get("category", "")
        if cat in categories:
            filename = fname if os.path.splitext(fname)[1] else fname + ".jpg"
            cat_entries[cat].append({"filename": filename, "prompt": info["prompt"]})

    result = {}
    for cat in categories:
        entries = sorted(cat_entries[cat], key=lambda x: x["filename"])
        result[cat] = entries[:n]
        if len(entries) < n:
            print(
                f"  Warning: category '{cat}' has only {len(entries)} entries "
                f"(requested {n})."
            )

    return result


def load_prompts_svdquant(prompts_path: str, n: int, seed: int) -> dict:
    """
    Load prompts from prompts.yaml (mit-han-lab/svdquant-datasets format).
    Returns {"all": [{"filename": "hash.png", "prompt": "..."}, ...]}.
    Takes the first n entries in deterministic sorted order.
    """
    import yaml
    with open(prompts_path) as f:
        data = yaml.safe_load(f)

    entries = sorted(
        [{"filename": k + ".png", "prompt": v} for k, v in data.items()],
        key=lambda x: x["filename"],
    )
    if len(entries) < n:
        print(f"  Warning: prompts.yaml has only {len(entries)} entries (requested {n}).")
    return {"images": entries[:n]}


def load_prompts_dev(dev_prompts_path, n):
    """Load the Phase-3 dev slice (built by scripts/build_dev_slice.py). Returns
    {"images": [...], "ref_dir": <FID reference dir>}. Entries carry the persisted
    per-prompt seed; order is the canonical idx order. Disjoint-from-svdquant by
    construction (proof in disjointness_proof.json next to dev_prompts.json)."""
    with open(dev_prompts_path) as f:
        rec = json.load(f)
    entries = sorted(rec["prompts"], key=lambda e: e["idx"])[:int(n)]
    for e in entries:
        e.setdefault("filename", f"{e['idx']:04d}.png")
    return {"images": entries, "ref_dir": rec["meta"]["ref_images_dir"]}


def _image_file_ok(path: str) -> bool:
    """
    Return True if path exists and is a valid, non-truncated image.
    Uses PIL verify() which checks file structure without decoding pixels.
    Fast enough to run on every file (~1 ms each).
    """
    try:
        if os.path.getsize(path) == 0:
            return False
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False
