"""
Factored-rotation tests: width-keyed H_p (x) H_q enters the locked W4A4 path
through rotate/unrotate overrides + prepare_weight's rotate_fn, with rods_a.fwht
untouched. Pins:

(a) resolution semantics: map hit -> FactoredSpec, miss -> None (pad fallback),
    pad_to consistency, degenerate factor pairs raise.
(b) round-trip + norm preservation for native and padded specs (Ĥᵀ unrotation).
(c) the (36, 32) factored matrix is BITWISE the kronecker scheme's matrix, and a
    wrapper built on the same pack with overrides reproduces the kron wrapper's
    output exactly (torch.equal).
(d) the unmapped-width fallback is BITWISE the pad scheme, weight side
    (prepare_weight rotate_fn = fwht.rotate) and activation side
    (pad_rotators("dense")) alike -- end-to-end torch.equal vs a pad wrapper.
(e) ActQuantConfig.from_mapping normalizes the YAML string-keyed width maps.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

# Direct-function imports: `from rods_a import fwht` binds the FWHT *function*
# re-exported by rods_a/__init__, not the submodule.
from rods_a.fwht import (
    dense_kron_rotate,
    dense_rotate,
    dense_unrotate,
    rotate as pad_fwht_rotate,
    unrotate as pad_fwht_unrotate,
)
from rods_a.factored_rotation import (
    FactoredSpec,
    factored_rotators,
    pad_rotators,
    resolve_factored,
)
from rods_a.prepare_weight import prepare_weight
from rods_a.wrapper import ActQuantConfig, RoDSAW4A4Linear


# --------------------------------------------------------------------------- #
# (a) resolution
# --------------------------------------------------------------------------- #
def test_resolve_hit_and_miss():
    fmap = {1152: (20, 64), 2240: (40, 56)}
    spec = resolve_factored(1152, fmap)
    assert spec == FactoredSpec(width=1152, factors=(20, 64), pad_to=1280)
    native = resolve_factored(2240, fmap)
    assert native.pad_to == 2240 and native.factors == (40, 56)
    assert resolve_factored(4608, fmap) is None          # miss -> pad fallback
    assert resolve_factored(4608, None) is None


def test_resolve_pad_to_consistency():
    fmap = {1152: (20, 64)}
    ok = resolve_factored(1152, fmap, {1152: 1280})
    assert ok.pad_to == 1280
    with pytest.raises(ValueError):
        resolve_factored(1152, fmap, {1152: 2048})       # pad_to != p*q
    with pytest.raises(ValueError):
        resolve_factored(1152, {1152: (12, 32)})         # p*q = 384 < width
    with pytest.raises(ValueError):
        resolve_factored(1152, {1152: (12, 8, 12)})      # not a pair


# --------------------------------------------------------------------------- #
# (b) round-trip + norm preservation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("width,factors", [(2240, (40, 56)),   # native
                                           (1152, (20, 64)),   # padded to 1280
                                           (288, (36, 8))])    # native control
def test_roundtrip_and_norm(width, factors):
    torch.manual_seed(0)
    spec = resolve_factored(width, {width: factors})
    rot, unrot = factored_rotators(spec)
    x = torch.randn(16, width, dtype=torch.float32)
    xr = rot(x)
    assert xr.shape == (16, spec.pad_to)
    # Orthonormal: norms preserved (padding adds zeros, no energy).
    assert torch.allclose(xr.norm(dim=-1), x.norm(dim=-1), atol=1e-4)
    back = unrot(xr, width)
    assert back.shape == x.shape
    assert (back - x).abs().max() < 1e-4
    with pytest.raises(ValueError):
        unrot(xr, width + 1)                              # orig_dim mismatch


def test_rotator_width_checks():
    spec = resolve_factored(288, {288: (36, 8)})
    rot, unrot = factored_rotators(spec)
    with pytest.raises(ValueError):
        rot(torch.randn(4, 289))
    with pytest.raises(ValueError):
        unrot(torch.randn(4, 289))


# --------------------------------------------------------------------------- #
# (c) (36, 32)-class specs are bitwise the kronecker scheme
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("width,factors", [(1152, (36, 32)), (288, (36, 8))])
def test_factored_kron_shape_bitwise(width, factors):
    torch.manual_seed(1)
    spec = resolve_factored(width, {width: factors})
    rot, _ = factored_rotators(spec)
    x = torch.randn(8, width, dtype=torch.float32)
    assert torch.equal(rot(x), dense_kron_rotate(x))


def test_wrapper_override_reproduces_kron_wrapper():
    """Same prepared pack, kron wrapper vs factored-override wrapper: the
    activation-side override must reproduce the kron scheme output EXACTLY.
    Cin = 2304 = 36 * 64: kron-factorable AND group-64 divisible (the refit
    asserts in_f % group_size == 0)."""
    torch.manual_seed(2)
    out_f, Cin = 32, 2304
    W = torch.randn(out_f, Cin, dtype=torch.float32) * 0.05
    pack = prepare_weight(W, bits=4, group_size=64, device="cpu",
                          compute_dtype=torch.float32, bitpack=False,
                          rotate_weight=True, rotation="kronecker",
                          strict_rho=False)
    cfg_kron = ActQuantConfig(rotation="kronecker")
    lyr_kron = RoDSAW4A4Linear(pack, cfg_kron, matmul_dtype=torch.float32)

    spec = resolve_factored(Cin, {Cin: (36, 64)})
    rot, unrot = factored_rotators(spec)
    cfg_fact = ActQuantConfig(rotation="factored",
                              rotation_factorization={Cin: (36, 64)})
    lyr_fact = RoDSAW4A4Linear(pack, cfg_fact, matmul_dtype=torch.float32,
                               rotate_override=rot, unrotate_override=unrot)

    x = torch.randn(24, Cin, dtype=torch.float32)
    assert torch.equal(lyr_kron(x), lyr_fact(x))


# --------------------------------------------------------------------------- #
# (d) unmapped-width fallback is bitwise the pad scheme
# --------------------------------------------------------------------------- #
def test_pad_rotators_are_the_pad_scheme():
    rot_d, unrot_d = pad_rotators("dense")
    assert rot_d is dense_rotate and unrot_d is dense_unrotate
    rot_f, unrot_f = pad_rotators("fwht")
    assert rot_f is pad_fwht_rotate and unrot_f is pad_fwht_unrotate
    with pytest.raises(ValueError):
        pad_rotators("butterfly")


def test_prepare_weight_rotate_fn_matches_pad_branch():
    torch.manual_seed(3)
    out_f, Cin = 64, 192                                  # pads to 256
    W = torch.randn(out_f, Cin, dtype=torch.float32) * 0.05
    ref = prepare_weight(W, bits=4, group_size=64, device="cpu",
                         compute_dtype=torch.float32, bitpack=False,
                         rotation="pad", strict_rho=False)
    inj = prepare_weight(W, bits=4, group_size=64, device="cpu",
                         compute_dtype=torch.float32, bitpack=False,
                         strict_rho=False, rotate_fn=pad_fwht_rotate)
    assert inj.Cin_pad == ref.Cin_pad == 256
    assert torch.equal(inj.W_q, ref.W_q)
    assert torch.equal(inj.meta["scale"], ref.meta["scale"])
    assert torch.equal(inj.meta["zero"], ref.meta["zero"])
    assert torch.equal(inj.colsum, ref.colsum)


def test_wrapper_fallback_end_to_end_bitwise_pad():
    torch.manual_seed(4)
    out_f, Cin = 64, 192
    W = torch.randn(out_f, Cin, dtype=torch.float32) * 0.05
    pack = prepare_weight(W, bits=4, group_size=64, device="cpu",
                          compute_dtype=torch.float32, bitpack=False,
                          rotation="pad", strict_rho=False)
    lyr_pad = RoDSAW4A4Linear(pack, ActQuantConfig(rotation="pad"),
                              matmul_dtype=torch.float32)
    rot, unrot = pad_rotators("dense")
    lyr_fb = RoDSAW4A4Linear(pack, ActQuantConfig(rotation="factored"),
                             matmul_dtype=torch.float32,
                             rotate_override=rot, unrotate_override=unrot)
    x = torch.randn(24, Cin, dtype=torch.float32)
    assert torch.equal(lyr_pad(x), lyr_fb(x))


def test_override_pair_required():
    torch.manual_seed(5)
    W = torch.randn(32, 128, dtype=torch.float32) * 0.05
    pack = prepare_weight(W, bits=4, group_size=64, device="cpu",
                          compute_dtype=torch.float32, bitpack=False,
                          rotation="pad", strict_rho=False)
    rot, _ = pad_rotators("dense")
    with pytest.raises(ValueError):
        RoDSAW4A4Linear(pack, ActQuantConfig(), matmul_dtype=torch.float32,
                        rotate_override=rot)              # missing unrotate


# --------------------------------------------------------------------------- #
# (e) config normalization
# --------------------------------------------------------------------------- #
def test_from_mapping_normalizes_width_maps():
    cfg = ActQuantConfig.from_mapping({
        "rotation": "factored",
        "rotation_factorization": {"1152": [20, 64], "4608": [36, 128]},
        "rotation_pad_to": {"1152": 1280},
        "rotation_factor_order_swap": False,
    })
    assert cfg.rotation == "factored"
    assert cfg.rotation_factorization == {1152: (20, 64), 4608: (36, 128)}
    assert cfg.rotation_pad_to == {1152: 1280}
    assert cfg.rotation_factor_order_swap is False


def test_from_mapping_empty_maps_are_none():
    cfg = ActQuantConfig.from_mapping({"rotation_factorization": {},
                                       "rotation_pad_to": {}})
    assert cfg.rotation_factorization is None
    assert cfg.rotation_pad_to is None
    # Legacy configs (no new keys) stay at defaults.
    legacy = ActQuantConfig.from_mapping({"act_bits": 4})
    assert legacy.rotation_factorization is None
    assert legacy.rotation_pad_to is None
    assert legacy.rotation_factor_order_swap is False
