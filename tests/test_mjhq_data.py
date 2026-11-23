"""Characterization + identity tests for core/mjhq_data.py (P1 gen/eval seam, c1).

These pin the behavior of the dataset/prompt loaders that were lifted verbatim out
of scripts/fid_eval.py, and prove fid_eval re-exports the *same* objects (a shim, not
a reimplementation). CPU-only, no network: the download helpers are exercised only on
their already-present early-return path, with hf_hub_download monkeypatched to fail if
it is ever reached.
"""
import json
import os
import pathlib

import pytest
import yaml
from PIL import Image

import core.mjhq_data as md


# ----------------------------------------------------------------------------- #
# load_prompts_svdquant
# ----------------------------------------------------------------------------- #

def test_load_prompts_svdquant_structure_sorting_and_n(tmp_path):
    p = tmp_path / "prompts.yaml"
    # intentionally out of order; keys map to "<key>.png"
    p.write_text(yaml.safe_dump({"c": "third", "a": "first", "b": "second"}))

    out = md.load_prompts_svdquant(str(p), n=2, seed=0)
    assert set(out.keys()) == {"images"}
    # sorted by filename ("a.png" < "b.png" < "c.png"); first n=2
    assert out["images"] == [
        {"filename": "a.png", "prompt": "first"},
        {"filename": "b.png", "prompt": "second"},
    ]


def test_load_prompts_svdquant_n_larger_than_available(tmp_path):
    p = tmp_path / "prompts.yaml"
    p.write_text(yaml.safe_dump({"a": "one"}))
    out = md.load_prompts_svdquant(str(p), n=5, seed=0)
    assert out["images"] == [{"filename": "a.png", "prompt": "one"}]


# ----------------------------------------------------------------------------- #
# load_prompts_per_category
# ----------------------------------------------------------------------------- #

def test_load_prompts_per_category_sorting_extension_and_filter(tmp_path):
    meta = {
        # already has extension -> kept; out of order -> sorted by filename
        "z.jpg": {"category": "art", "prompt": "art-z"},
        "a.jpg": {"category": "art", "prompt": "art-a"},
        # no extension -> ".jpg" appended
        "noext": {"category": "art", "prompt": "art-noext"},
        # different category -> excluded when not requested
        "x.jpg": {"category": "food", "prompt": "food-x"},
    }
    meta_path = tmp_path / "meta_data.json"
    meta_path.write_text(json.dumps(meta))

    out = md.load_prompts_per_category(str(meta_path), categories=["art"], n=2, seed=0)
    assert list(out.keys()) == ["art"]
    # sorted by filename: "a.jpg" < "noext.jpg" < "z.jpg"; first n=2
    assert out["art"] == [
        {"filename": "a.jpg", "prompt": "art-a"},
        {"filename": "noext.jpg", "prompt": "art-noext"},
    ]


def test_load_prompts_per_category_missing_category_returns_empty(tmp_path):
    meta_path = tmp_path / "meta_data.json"
    meta_path.write_text(json.dumps({"a.jpg": {"category": "art", "prompt": "p"}}))
    out = md.load_prompts_per_category(str(meta_path), categories=["vehicles"], n=3, seed=0)
    assert out == {"vehicles": []}


# ----------------------------------------------------------------------------- #
# _image_file_ok
# ----------------------------------------------------------------------------- #

def test_image_file_ok_valid_png(tmp_path):
    p = tmp_path / "img.png"
    Image.new("RGB", (8, 8), (1, 2, 3)).save(p)
    assert md._image_file_ok(str(p)) is True


def test_image_file_ok_rejects_zero_byte_missing_and_truncated(tmp_path):
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    assert md._image_file_ok(str(empty)) is False

    assert md._image_file_ok(str(tmp_path / "does_not_exist.png")) is False

    truncated = tmp_path / "trunc.png"
    good = tmp_path / "good.png"
    Image.new("RGB", (8, 8)).save(good)
    truncated.write_bytes(good.read_bytes()[:32])  # chop the file
    assert md._image_file_ok(str(truncated)) is False


# ----------------------------------------------------------------------------- #
# download helpers — already-present early return, no network
# ----------------------------------------------------------------------------- #

def _boom(*a, **k):  # pragma: no cover - only called on failure
    raise AssertionError("hf_hub_download must not be called when data is present")


def test_download_mjhq_early_return_no_network(tmp_path, monkeypatch):
    monkeypatch.setattr(md, "hf_hub_download", _boom)
    (tmp_path / "meta_data.json").write_text("{}")
    (tmp_path / "animals").mkdir()
    imgs_dir, meta_path = md.download_mjhq(str(tmp_path))
    assert imgs_dir == str(tmp_path)
    assert meta_path == os.path.join(str(tmp_path), "meta_data.json")


def test_download_mjhq_svdquant_early_return_no_network(tmp_path, monkeypatch):
    monkeypatch.setattr(md, "hf_hub_download", _boom)
    (tmp_path / "images").mkdir()
    (tmp_path / "prompts.yaml").write_text("a: b\n")
    images_dir, prompts_path = md.download_mjhq_svdquant(str(tmp_path))
    assert images_dir == os.path.join(str(tmp_path), "images")
    assert prompts_path == os.path.join(str(tmp_path), "prompts.yaml")


# ----------------------------------------------------------------------------- #
# Shim identity — fid_eval re-exports the SAME objects (heavy import: deps required)
# ----------------------------------------------------------------------------- #

def test_fid_eval_reexports_same_objects():
    fe = pytest.importorskip("scripts.fid_eval")
    for name in (
        "download_mjhq",
        "download_mjhq_svdquant",
        "load_prompts_per_category",
        "load_prompts_svdquant",
        "_image_file_ok",
        "MJHQ_REPO",
        "SVDQUANT_REPO",
    ):
        assert getattr(fe, name) is getattr(md, name), f"{name} is not re-exported as-is"


# ----------------------------------------------------------------------------- #
# load_prompts_dev (lifted from rods_a_w4a4_eval in c2)
# ----------------------------------------------------------------------------- #

def test_load_prompts_dev_sorting_default_filename_and_ref_dir(tmp_path):
    rec = {
        "prompts": [
            {"idx": 2, "seed": 22, "prompt": "two"},
            {"idx": 0, "seed": 20, "prompt": "zero"},
            # idx 1 already carries an explicit filename -> preserved (setdefault)
            {"idx": 1, "seed": 21, "prompt": "one", "filename": "custom.png"},
        ],
        "meta": {"ref_images_dir": "/some/ref/dir"},
    }
    p = tmp_path / "dev_prompts.json"
    p.write_text(json.dumps(rec))

    out = md.load_prompts_dev(str(p), n=2)
    assert out["ref_dir"] == "/some/ref/dir"
    # sorted by idx; first n=2 -> idx 0 then idx 1
    assert [e["idx"] for e in out["images"]] == [0, 1]
    assert out["images"][0]["filename"] == "0000.png"   # default applied
    assert out["images"][1]["filename"] == "custom.png"  # explicit preserved


def test_rods_a_w4a4_eval_reexports_load_prompts_dev():
    rae = pytest.importorskip("scripts.rods_a_w4a4_eval")
    assert rae.load_prompts_dev is md.load_prompts_dev


# ----------------------------------------------------------------------------- #
# c3 — importers repointed onto core.mjhq_data (loaders decoupled from the evaluator)
# ----------------------------------------------------------------------------- #

_SCRIPTS = pathlib.Path(__file__).resolve().parent.parent / "scripts"


def test_c3_rods_a_w4a4_eval_loaders_resolve_to_core():
    rae = pytest.importorskip("scripts.rods_a_w4a4_eval")
    # _image_file_ok was an internal-use import dropped in c6a-2 (the generation
    # loop that used it moved to core.generate_core); the loaders below are still
    # used by main() (svdquant + dev paths) and must resolve to core.mjhq_data.
    for name in (
        "download_mjhq_svdquant",
        "load_prompts_svdquant",
        "load_prompts_dev",
    ):
        assert getattr(rae, name) is getattr(md, name), f"{name} did not repoint to core"
