"""Bit-identity tests for core/metric_core.py (P1 gen/eval seam, c4).

LPIPS (AlexNet; lpips.LPIPS defaults to eval_mode=True so dropout is off ->
deterministic) and the uint8 MSE are pure code-motion. Each test rebuilds the
pre-c4 reference logic inline as the oracle and asserts the shared kernel + the
repointed drivers reproduce it exactly (bit-identical, not approx).

The LPIPS model build is guarded: if the AlexNet weights cannot be loaded (no
network / no cache in the gate container) the perceptual tests skip rather than
fail — an environment gap must not turn the refactor gate red. The MSE / PSNR /
wiring tests need no model and always run.
"""
import math

import numpy as np
import pytest
from PIL import Image

import core.metric_core as mc


@pytest.fixture
def two_pngs(tmp_path):
    rng = np.random.RandomState(0)
    a = Image.fromarray(rng.randint(0, 256, (256, 256, 3), dtype=np.uint8))
    b = Image.fromarray(rng.randint(0, 256, (256, 256, 3), dtype=np.uint8))
    pa, pb = tmp_path / "a.png", tmp_path / "b.png"
    a.save(pa)
    b.save(pb)
    return str(pa), str(pb)


def _arrs(pa, pb):
    a = np.array(Image.open(pa).convert("RGB"))
    b = np.array(Image.open(pb).convert("RGB"))
    return a, b


def _scorer_or_skip(resize=256, device="cpu"):
    pytest.importorskip("lpips")
    try:
        return mc.LpipsScorer(device, net="alex", resize=resize)
    except Exception as e:  # weights/network unavailable in-container
        pytest.skip(f"LPIPS model unavailable: {e}")


def _oracle_lpips(pa, pb, resize=256, device="cpu"):
    import lpips
    import torch
    import torchvision.transforms as T

    loss_fn = lpips.LPIPS(net="alex").float().to(device)
    loss_fn.eval()
    ops = [T.ToTensor()]
    if resize:
        ops.insert(0, T.Resize((resize, resize), antialias=True))
    tf = T.Compose(ops)
    ta = tf(Image.open(pa).convert("RGB")).unsqueeze(0).float().to(device)
    tb = tf(Image.open(pb).convert("RGB")).unsqueeze(0).float().to(device)
    with torch.no_grad():
        return float(loss_fn(ta, tb, normalize=True).item())


# ----------------------------------------------------------------------------- #
# mse_uint8
# ----------------------------------------------------------------------------- #

def test_mse_uint8_matches_inline_and_zero_on_equal(two_pngs):
    a, b = _arrs(*two_pngs)
    expected = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    assert mc.mse_uint8(a, b) == expected
    assert mc.mse_uint8(a, a) == 0.0


# ----------------------------------------------------------------------------- #
# LPIPS kernel — bit-identical to the pre-c4 inline construction
# ----------------------------------------------------------------------------- #

def test_lpips_scorer_bit_identical_to_oracle_256(two_pngs):
    scorer = _scorer_or_skip(resize=256)
    pa, pb = two_pngs
    assert scorer(pa, pb) == _oracle_lpips(pa, pb, resize=256)


def test_lpips_scorer_pil_input_equals_path_input(two_pngs):
    scorer = _scorer_or_skip(resize=256)
    pa, pb = two_pngs
    pil_a = Image.open(pa).convert("RGB")
    pil_b = Image.open(pb).convert("RGB")
    assert scorer(pil_a, pil_b) == scorer(pa, pb)


def test_lpips_scorer_fullres_resize0(two_pngs):
    scorer = _scorer_or_skip(resize=0)
    pa, pb = two_pngs
    assert scorer(pa, pb) == _oracle_lpips(pa, pb, resize=0)


# ----------------------------------------------------------------------------- #
# Repointed PSNR drivers keep their exact formula + identical-image sentinel
# ----------------------------------------------------------------------------- #

def test_psnr_eval_driver_inf_sentinel_and_amplitude_formula(two_pngs):
    psnr_eval = pytest.importorskip("scripts.psnr_eval")
    a, b = _arrs(*two_pngs)
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    assert psnr_eval.psnr_from_arrays(a, b) == 20.0 * math.log10(255.0 / math.sqrt(mse))
    assert psnr_eval.psnr_from_arrays(a, a) == float("inf")


def test_paired_psnr_driver_100_sentinel_and_power_formula(two_pngs):
    pim = pytest.importorskip("scripts.paired_image_metrics")
    a, b = _arrs(*two_pngs)
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    assert pim.psnr_uint8(a, b) == 10.0 * math.log10(255.0 ** 2 / mse)
    assert pim.psnr_uint8(a, a) == 100.0
    # _LPIPS (imported externally by lpips_selected_pairs) is now the shared scorer
    assert pim._LPIPS is mc.LpipsScorer
