"""CPU tests for scripts/generate.py helpers (P1 gen/eval seam, c6b).

The generation itself is GPU/model-bound and covered by a dev-slice byte-diff against
the c6a oracle; here we pin the metric-free, model-free plumbing: prompt loading for
the flat layout, the args builder (model defaults), and the manifest firewall ledger.
generate.py imports core.model_config (torch/diffusers), so the module import is guarded.
"""
import json

import pytest
from omegaconf import OmegaConf
from types import SimpleNamespace


@pytest.fixture
def gen_mod():
    return pytest.importorskip("scripts.generate")


def test_generate_importable(gen_mod):
    assert hasattr(gen_mod, "main")
    assert hasattr(gen_mod, "generate_into_dir")  # composes the shared core


def test_load_entries_dev(gen_mod, tmp_path):
    rec = {
        "prompts": [
            {"idx": 1, "seed": 11, "prompt": "one"},
            {"idx": 0, "seed": 10, "prompt": "zero"},
            {"idx": 2, "seed": 12, "prompt": "two"},
        ],
        "meta": {"ref_images_dir": "/ref"},
    }
    p = tmp_path / "dev_prompts.json"
    p.write_text(json.dumps(rec))
    cfg = OmegaConf.create({"prompt_source": "dev", "num_samples": 2, "dev_prompts_path": str(p)})
    entries = gen_mod.load_entries(cfg)
    assert [e["idx"] for e in entries] == [0, 1]          # sorted by idx, first n
    assert [e["prompt"] for e in entries] == ["zero", "one"]


def test_make_args_uses_model_defaults(gen_mod):
    mc = pytest.importorskip("core.model_config").get_model_config("pixart")
    cfg = OmegaConf.create({
        "num_inference_steps": None, "guidance_scale": None,
        "height": 1024, "width": 1024, "device": "cpu", "seed": 12345,
    })
    args = gen_mod.make_args(cfg, mc)
    assert args.num_inference_steps == mc.default_steps      # 20 for PixArt
    assert args.guidance_scale == mc.default_guidance_scale  # 7.5
    assert args.max_sequence_length == 300                   # default when unset


def test_build_manifest_paired_records_per_prompt_seed(gen_mod):
    entries = [{"idx": 5, "seed": 99, "prompt": "a"}, {"idx": 6, "seed": 100, "prompt": "b"}]
    args = SimpleNamespace(seed=12345, num_inference_steps=20, guidance_scale=7.5, height=1024, width=1024)
    cfg = OmegaConf.create({"paired_seed": True})
    m = gen_mod.build_manifest(entries, cfg, args, "fp16", "dev1k", "dev", "/out/fp16/images")
    assert m["dataset_split"] == "dev1k" and m["exposure"] == "dev"
    assert m["generator_convention"] == "paired-per-prompt"
    assert [im["seed"] for im in m["images"]] == [99, 100]            # per-prompt seed
    assert [im["filename"] for im in m["images"]] == ["0000.png", "0001.png"]
    assert len(m["prompt_set_sha"]) == 64                             # sha256 hex
    assert m["n"] == 2


def test_build_manifest_nonpaired_uses_global_seed(gen_mod):
    entries = [{"prompt": "a"}, {"prompt": "b"}]
    args = SimpleNamespace(seed=12345, num_inference_steps=20, guidance_scale=7.5, height=1024, width=1024)
    cfg = OmegaConf.create({"paired_seed": False})
    m = gen_mod.build_manifest(entries, cfg, args, "fp16", "mjhq30k", "reference", "/out/fp16/images")
    assert m["generator_convention"] == "global-advance-no-skip-advance"
    assert [im["seed"] for im in m["images"]] == [12345, 12345]       # global args.seed
