"""
Compose-time gates for the factored-rotation quant configs.

verify_configs.py only composes TOP-LEVEL configs, so the per-arm quant-group
yamls (quant=rods_a_w4a4_fact_*) would otherwise be unexercised until job time.
Every arm is composed here through the real eval config and its width maps are
validated end-to-end (from_mapping int-normalization, p*q >= width, group-64
divisibility, pad_to consistency via resolve_factored).
"""

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from core.hadamard import order_of
from rods_a.factored_rotation import resolve_factored
from rods_a.wrapper import ActQuantConfig

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = os.path.join(REPO, "configs")

ARM_STEMS = sorted(
    os.path.splitext(os.path.basename(p))[0]
    for p in glob.glob(os.path.join(CONFIGS, "quant", "rods_a_w4a4_fact_*.yaml"))
)
GROUP_SIZE = 64


def _compose(quant_name: str):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIGS, version_base=None):
        return compose(config_name="rods_a_w4a4_eval",
                       overrides=[f"quant={quant_name}"])


def test_expected_arm_set_present():
    # The release ships the factorized configs behind the paper's numbers.
    expected = {"rods_a_w4a4_fact_px1", "rods_a_w4a4_fact_sana5", "rods_a_w4a4_fact_flux3"}
    assert set(ARM_STEMS) == expected


@pytest.mark.parametrize("stem", ARM_STEMS)
def test_arm_config_maps_valid(stem):
    cfg = _compose(stem)
    assert str(cfg.quant.name) == stem
    act = ActQuantConfig.from_mapping(cfg.quant)
    assert act.rotation == "factored"
    assert act.rotation_factor_order_swap is False
    assert act.percentile_p == 0.9995              # uniform sweep clip
    assert act.center_act_mean is True
    fmap = act.rotation_factorization
    assert fmap, f"{stem}: empty rotation_factorization"
    for width, (p, q) in fmap.items():
        # A factor is an int order or a variant token ("p24"/"p48"); order_of
        # validates both forms.
        assert isinstance(width, int)
        spec = resolve_factored(width, fmap, act.rotation_pad_to)
        assert spec is not None and spec.factors == (p, q)
        assert spec.pad_to == order_of(p) * order_of(q) >= width
        assert spec.pad_to % GROUP_SIZE == 0, \
            f"{stem}: pad_to {spec.pad_to} not divisible by gs={GROUP_SIZE}"
    if act.rotation_pad_to:
        # Every documented pad width refers to a mapped factorization.
        assert set(act.rotation_pad_to) <= set(fmap)


def test_eval_configs_default_new_keys_off():
    for name in ["rods_a_w4a4_eval", "rods_a_w4a4_eval_sana", "rods_a_w4a4_eval_flux"]:
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=CONFIGS, version_base=None):
            cfg = compose(config_name=name)
        assert str(cfg.capture_dtype) == "float32"
        assert bool(cfg.install_ffn_conv) is False
