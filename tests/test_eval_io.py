"""Characterization tests for the persistence helpers folded into core/eval_io.py
(P1 gen/eval seam, c5) — merge_results (deep-merge + preserve-on-None + OmegaConf
coercion) and write_method_meta (sidecar writer). CPU-only; pin the exact behavior
that lived in fid_eval's nested _merge_results / _write_method_meta closures.
"""
import json

from omegaconf import OmegaConf

import core.eval_io as eio


def test_merge_results_preserve_segment_when_new_is_none(tmp_path):
    p = tmp_path / "eval_results.json"
    p.write_text(json.dumps({"fid": {"m1": {"overall": 1.0}}, "seed": 42}))
    out = eio.merge_results(str(p), {"fid": None, "seed": 7})
    assert out["fid"] == {"m1": {"overall": 1.0}}  # disk segment preserved
    assert out["seed"] == 7                          # non-per-method key updates
    assert json.loads(p.read_text()) == out          # file == return


def test_merge_results_deep_merges_per_method_dicts(tmp_path):
    p = tmp_path / "eval_results.json"
    p.write_text(json.dumps({"lpips": {"m1": {"mean": 0.1}}}))
    out = eio.merge_results(str(p), {"lpips": {"m2": {"mean": 0.2}}})
    assert out["lpips"] == {"m1": {"mean": 0.1}, "m2": {"mean": 0.2}}  # union; new wins


def test_merge_results_coerces_omegaconf_values(tmp_path):
    p = tmp_path / "eval_results.json"
    # a ListConfig would raise TypeError on a plain json.dump without to_container
    out = eio.merge_results(
        str(p), {"categories": OmegaConf.create(["a", "b"]), "fid": {"m": {"overall": 3.0}}}
    )
    assert out["categories"] == ["a", "b"]
    assert json.loads(p.read_text())["categories"] == ["a", "b"]


def test_merge_results_treats_corrupt_existing_as_empty(tmp_path):
    p = tmp_path / "eval_results.json"
    p.write_text("{ not valid json")
    out = eio.merge_results(str(p), {"x": 1})
    assert out == {"x": 1}


def test_merge_results_respects_explicit_per_method_keys(tmp_path):
    p = tmp_path / "eval_results.json"
    p.write_text(json.dumps({"custom": {"a": 1}}))
    # "custom" is per-method here -> deep-merged; default keys would shallow-replace
    out = eio.merge_results(str(p), {"custom": {"b": 2}}, per_method_keys=("custom",))
    assert out["custom"] == {"a": 1, "b": 2}


def test_write_method_meta_creates_dir_and_stringifies_nonjson(tmp_path):
    d = tmp_path / "nested" / "method"
    eio.write_method_meta(str(d), {"method": "fp16", "obj": object()})
    written = json.loads((d / "meta.json").read_text())
    assert written["method"] == "fp16"
    assert isinstance(written["obj"], str)  # default=str applied
