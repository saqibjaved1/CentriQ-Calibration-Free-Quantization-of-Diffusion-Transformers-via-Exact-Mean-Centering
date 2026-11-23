"""Generator-convention tests for core/generate_core.py (P1 gen/eval seam, c6a).

Pins the load-bearing behavior of generate_into_dir WITHOUT a real diffusion model:
call_pipeline is faked to draw one int from the generator per call (modelling the
pipeline consuming the generator), so the exact generator management — seed once,
skip-does-not-advance, paired reseed, overwrite — is verifiable against an inline
reference sequence. CPU-only; needs torch (no GPU, no model).
"""
import os
from types import SimpleNamespace

import torch

import core.generate_core as gc


# ----------------------------------------------------------------------------- #
# Fakes
# ----------------------------------------------------------------------------- #

class _FakeImg:
    def __init__(self, val):
        self.val = val

    def save(self, path):
        with open(path, "w") as f:
            f.write(str(self.val))  # non-empty -> looks "valid" to the patched checker


class _FakeOut:
    def __init__(self, val):
        self.images = [_FakeImg(val)]


def _draw(gen):
    return int(torch.randint(0, 2**31 - 1, (1,), generator=gen).item())


def _install(monkeypatch, recorded):
    def fake_call(pipe, model_config, prompt, args, generator):
        v = _draw(generator)
        recorded.append((prompt, v))
        return _FakeOut(v)

    monkeypatch.setattr(gc, "call_pipeline", fake_call)
    # a present, non-empty file counts as a valid image for the skip path
    monkeypatch.setattr(
        gc, "_image_file_ok", lambda p: os.path.exists(p) and os.path.getsize(p) > 0
    )


def _args(seed=12345):
    return SimpleNamespace(device="cpu", seed=seed)


def _entries(n, with_seed=False):
    return [
        {"prompt": f"p{i}", **({"seed": 1000 + i} if with_seed else {})}
        for i in range(n)
    ]


def _ref_sequence(seed, k):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return [_draw(g) for _ in range(k)]


# ----------------------------------------------------------------------------- #
# Tests
# ----------------------------------------------------------------------------- #

def test_from_scratch_seeds_once_and_advances(tmp_path, monkeypatch):
    rec = []
    _install(monkeypatch, rec)
    r = gc.generate_into_dir(None, None, _entries(3), str(tmp_path), _args(7))
    assert [v for _, v in rec] == _ref_sequence(7, 3)  # single advancing generator
    assert (r.n_generated, r.n_skipped) == (3, 0)
    assert r.paths == [os.path.join(str(tmp_path), f"{i:04d}.png") for i in range(3)]


def test_resume_skip_does_not_advance_generator(tmp_path, monkeypatch):
    rec = []
    _install(monkeypatch, rec)
    # pre-place a valid image at index 0 -> it is skipped
    with open(tmp_path / "0000.png", "w") as f:
        f.write("present")
    r = gc.generate_into_dir(None, None, _entries(3), str(tmp_path), _args(7))
    # index 0 skipped (no draw); indices 1,2 take the FIRST TWO draws of a fresh
    # Generator(7) -> the generator was NOT advanced for the skipped image.
    ref = _ref_sequence(7, 2)
    assert [v for _, v in rec] == ref
    assert (r.n_generated, r.n_skipped) == (2, 1)


def test_paired_seed_reseeds_per_entry(tmp_path, monkeypatch):
    rec = []
    _install(monkeypatch, rec)
    ents = _entries(3, with_seed=True)
    gc.generate_into_dir(None, None, ents, str(tmp_path), _args(7), paired_seed=True)
    expected = [_ref_sequence(e["seed"], 1)[0] for e in ents]
    assert [v for _, v in rec] == expected  # each image determined by its own seed


def test_overwrite_regenerates_present_images(tmp_path, monkeypatch):
    rec = []
    _install(monkeypatch, rec)
    for i in range(3):
        with open(tmp_path / f"{i:04d}.png", "w") as f:
            f.write("present")
    r = gc.generate_into_dir(None, None, _entries(3), str(tmp_path), _args(7), overwrite=True)
    assert [v for _, v in rec] == _ref_sequence(7, 3)
    assert (r.n_generated, r.n_skipped) == (3, 0)


def test_warn_truncated_only_when_requested(tmp_path, monkeypatch, capsys):
    rec = []
    _install(monkeypatch, rec)
    # an empty file is present but not "valid" (getsize 0) -> not skipped -> regenerated
    (tmp_path / "0000.png").write_bytes(b"")
    gc.generate_into_dir(None, None, _entries(1), str(tmp_path), _args(7), warn_truncated=True)
    assert "truncated image" in capsys.readouterr().out

    (tmp_path / "0000.png").write_bytes(b"")
    gc.generate_into_dir(None, None, _entries(1), str(tmp_path), _args(7), warn_truncated=False)
    assert "truncated image" not in capsys.readouterr().out
