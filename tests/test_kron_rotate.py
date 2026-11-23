"""
Kronecker (deployment-K) rotation tests (Task K, phase K1.3).

The pad fallback inflates the contraction width K (1152->2048, 4608->8192). The
Kronecker rotation R = (H_36 (x) H_{2^k})/sqrt(Cin) keeps K = Cin. These tests
pin its exactness (all deterministic, host-only CPU, no CUDA):

K1.3(a) equality vs the dense H_kron matmul  (||.||_inf < 1e-4 fp32).
K1.3(b) inverse: kron_unrotate(kron_rotate(x)) == x  (<= 1e-6).
K1.3(c) orthogonality: ||x||_2 invariant  (<= 1e-5).
K1.3(d) both deployed dims (1152, 4608) + a synthetic m*2^k control (36*8 = 288).
K1.3(e) H_36 integer-exact: entries in {-1,+1}, symmetric, H H^T = 36 I.
K1.3(f) regression: the pad path is bit-identical to the legacy rotate / pack
        (the Kronecker work does not touch the pad scheme).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math

import pytest
import torch

from rods_a.fwht import (
    dense_hadamard,
    dense_hadamard_m,
    dense_kron_rotate,
    dense_rotate,
    factor_kron,
    kron_rotate,
    kron_unrotate,
    rotate,
    rotate_for,
    unrotate_for,
)
from rods_a.pack_operands import pack_activations_ref
from rods_a.prepare_weight import prepare_weight
from rods_a.wrapper import ActQuantConfig

DEPLOYED_DIMS = [1152, 4608]          # PixArt-Sigma: 36*32, 36*128
CONTROL_DIMS = [288]                  # synthetic m*2^k control: 36*8
ALL_DIMS = DEPLOYED_DIMS + CONTROL_DIMS


def _dense_kron_matrix(Cin: int) -> torch.Tensor:
    """Independent reference H_kron = (H_36 (x) H_{2^k})/sqrt(Cin) (fp32)."""
    m, two_k = factor_kron(Cin)
    Hm = dense_hadamard_m(m, dtype=torch.float32)
    H2 = dense_hadamard(two_k, dtype=torch.float32)
    return torch.kron(Hm, H2) / math.sqrt(Cin)


# --------------------------------------------------------------------------- #
# K1.3(e) -- H_36 integer-exact
# --------------------------------------------------------------------------- #
def test_h36_integer_exact():
    H = dense_hadamard_m(36, dtype=torch.int64)
    assert set(H.unique().tolist()) <= {-1, 1}
    assert torch.equal(H, H.t())                       # symmetric
    assert torch.equal(H @ H.t(), 36 * torch.eye(36, dtype=torch.int64))


def test_factor_kron_deployed_and_control():
    assert factor_kron(1152) == (36, 32)
    assert factor_kron(4608) == (36, 128)
    assert factor_kron(288) == (36, 8)


def test_factor_kron_rejects_unsupported():
    # 3072 = 12*256 (FLUX) -- m=12 is noted but not vendored.
    with pytest.raises(ValueError):
        factor_kron(3072)
    with pytest.raises(ValueError):
        factor_kron(1000)                              # not m*2^k for any vendored m


def test_dense_hadamard_m_rejects_unsupported():
    with pytest.raises(ValueError):
        dense_hadamard_m(12)


# --------------------------------------------------------------------------- #
# K1.3(a) -- equality vs the dense H_kron matmul
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("Cin", ALL_DIMS)
def test_kron_rotate_equals_dense_matmul(Cin):
    torch.manual_seed(0)
    X = torch.randn(17, Cin, dtype=torch.float32)
    Href = _dense_kron_matrix(Cin)
    Y_struct = kron_rotate(X)                           # FWHT + small dense H_36
    Y_dense_fn = dense_kron_rotate(X)                   # cached full-dense GEMM path
    Y_ref = X @ Href                                    # independent reference
    assert Y_struct.shape == X.shape                    # no padding: width preserved
    assert (Y_struct - Y_ref).abs().max() < 1e-4
    assert (Y_dense_fn - Y_ref).abs().max() < 1e-4


# --------------------------------------------------------------------------- #
# K1.3(b) -- inverse (self-inverse involution)
#   Exactness is asserted in float64 (the math: R R = I to ~1e-12). The fp32
#   round-trip is machine-precision-limited (a double pass over Cin ~ 1e3-1e4
#   elements accumulates a few ulps, ~1e-6 max-abs), so fp32 is held to a
#   realistic 1e-5 floor (the same envelope as the orthogonality test).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("Cin", ALL_DIMS)
def test_kron_inverse_roundtrip(Cin):
    torch.manual_seed(1)
    # float64: true involution, tight bound.
    Xd = torch.randn(13, Cin, dtype=torch.float64)
    assert (kron_unrotate(kron_rotate(Xd), Cin) - Xd).abs().max() <= 1e-10
    # float32: realistic machine-precision floor.
    Xf = torch.randn(13, Cin, dtype=torch.float32)
    assert (kron_unrotate(kron_rotate(Xf), Cin) - Xf).abs().max() <= 1e-5
    # dense path round-trips too (fp32 floor).
    assert (unrotate_for("kronecker", "dense")(dense_kron_rotate(Xf), Cin) - Xf).abs().max() <= 1e-5


def test_kron_unrotate_width_guard():
    X = torch.randn(4, 1152)
    with pytest.raises(ValueError):
        kron_unrotate(kron_rotate(X), 1024)            # wrong orig_dim -> error


# --------------------------------------------------------------------------- #
# K1.3(c) -- orthogonality (norm invariance)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("Cin", ALL_DIMS)
def test_kron_orthogonality_norm_invariant(Cin):
    torch.manual_seed(2)
    X = torch.randn(11, Cin, dtype=torch.float32)
    n0 = X.norm(dim=-1)
    n1 = kron_rotate(X).norm(dim=-1)
    assert (n1 - n0).abs().max() <= 1e-5 * (1.0 + n0.max())


# --------------------------------------------------------------------------- #
# K1.3(f) -- pad-path regression (the Kronecker work must not touch "pad")
# --------------------------------------------------------------------------- #
def test_pad_scheme_dispatch_is_legacy():
    torch.manual_seed(3)
    X = torch.randn(8, 1152, dtype=torch.float32)
    assert torch.equal(rotate_for("pad", "fwht")(X), rotate(X))
    assert torch.equal(rotate_for("pad", "dense")(X), dense_rotate(X))


def test_pad_prepare_weight_bit_identical():
    """prepare_weight(rotation='pad') must reproduce the legacy fit byte-for-byte."""
    torch.manual_seed(4)
    W = torch.randn(128, 1152, dtype=torch.float32)
    pk_default = prepare_weight(W, bits=4, group_size=64, device="cpu",
                                compute_dtype=torch.float32, strict_rho=False)
    pk_pad = prepare_weight(W, bits=4, group_size=64, device="cpu",
                            compute_dtype=torch.float32, rotation="pad", strict_rho=False)
    assert pk_default.Cin_pad == 2048                  # legacy zero-pad
    assert torch.equal(pk_default.W_q, pk_pad.W_q)
    assert torch.equal(pk_default.meta["scale"], pk_pad.meta["scale"])


def test_pad_pack_activations_scheme_tag():
    torch.manual_seed(5)
    X = torch.randn(20, 1152, dtype=torch.float32)
    cfg_pad = ActQuantConfig(act_bits=4, scale_estimator="percentile", rotation="pad")
    cfg_kron = ActQuantConfig(act_bits=4, scale_estimator="percentile", rotation="kronecker")
    pa_pad = pack_activations_ref(X, cfg_pad, group_size=64)
    pa_kron = pack_activations_ref(X, cfg_kron, group_size=64)
    assert pa_pad.meta["scheme"] == "zero-pad" and pa_pad.meta["Cin_pad"] == 2048
    assert pa_kron.meta["scheme"] == "kronecker" and pa_kron.meta["Cin_pad"] == 1152


# --------------------------------------------------------------------------- #
# Kronecker end-to-end: rotated activation X̃ . rotated weight W̃ == X . W (FP).
# Confirms the deployment basis composes identically to the FP linear, at K = Cin.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("Cin", DEPLOYED_DIMS)
def test_kron_weight_activation_compose_matches_fp(Cin):
    torch.manual_seed(6)
    out_f = 64
    W = torch.randn(out_f, Cin, dtype=torch.float32)
    X = torch.randn(32, Cin, dtype=torch.float32)
    Wt = kron_rotate(W)                                # W̃
    Xt = kron_rotate(X)                                # X̃
    assert Wt.shape == W.shape and Xt.shape == X.shape
    y_fp = X @ W.t()
    y_rot = Xt @ Wt.t()                                # X̃ W̃ᵀ == X Wᵀ (orthonormal)
    rel = (y_rot - y_fp).norm() / y_fp.norm().clamp_min(1e-12)
    assert rel < 1e-4


# --------------------------------------------------------------------------- #
# kronecker_d (U4.B): seeded RHT — orthogonality, inverse, determinism,
# X̃ W̃ᵀ == X Wᵀ invariance, and pad/kron dispatch untouched by the seed kwarg.
# --------------------------------------------------------------------------- #

def test_kron_d_inverse_and_invariance():
    from rods_a.fwht import rotate_for, unrotate_for
    torch.manual_seed(11)
    for Cin in (1152, 4608):
        X = torch.randn(32, Cin, dtype=torch.float32)
        W = torch.randn(64, Cin, dtype=torch.float32)
        rot = rotate_for("kronecker_d", "fwht", seed=7)
        unrot = unrotate_for("kronecker_d", "fwht", seed=7)
        Xt = rot(X)
        assert Xt.shape == X.shape                       # K = Cin, no pad
        # norm preserved (orthogonality)
        assert torch.allclose(Xt.norm(dim=-1), X.norm(dim=-1), atol=1e-3)
        # exact inverse (R is NOT an involution; unrotate must apply D after H)
        back = unrot(Xt, Cin)
        assert (back - X).abs().max() < 1e-4
        # invariance with the SAME seed on both sides
        rel = ((rot(X) @ rot(W).t() - X @ W.t()).norm()
               / (X @ W.t()).norm().clamp_min(1e-12))
        assert rel < 1e-4
        # a DIFFERENT seed must break the compose (sanity that d is load-bearing)
        rot2 = rotate_for("kronecker_d", "fwht", seed=8)
        rel2 = ((rot(X) @ rot2(W).t() - X @ W.t()).norm()
                / (X @ W.t()).norm().clamp_min(1e-12))
        assert rel2 > 1e-2


def test_kron_d_seed_determinism_and_dense_match():
    from rods_a.fwht import rademacher_diag, rotate_for
    d1 = rademacher_diag(1152, 3)
    d2 = rademacher_diag(1152, 3)
    d3 = rademacher_diag(1152, 4)
    assert torch.equal(d1, d2)
    assert not torch.equal(d1, d3)
    assert set(d1.unique().tolist()) <= {-1.0, 1.0}
    X = torch.randn(16, 1152, dtype=torch.float32)
    a = rotate_for("kronecker_d", "fwht", seed=3)(X)
    b = rotate_for("kronecker_d", "dense", seed=3)(X)
    assert (a - b).abs().max() < 1e-4


def test_rotate_for_seed_kwarg_ignored_by_legacy_schemes():
    from rods_a.fwht import rotate_for
    X = torch.randn(8, 1152, dtype=torch.float32)
    for scheme in ("pad", "kronecker"):
        a = rotate_for(scheme, "fwht")(X)
        b = rotate_for(scheme, "fwht", seed=123)(X)
        assert torch.equal(a, b), f"{scheme}: seed kwarg must be a no-op"


def test_kron_d_prepare_weight_quantize_act_compose():
    """End-to-end U4.B seam: prepare_weight(kronecker_d, seed) x quantize_act
    (kronecker_d, same seed) compose like the kron pair (centered, FP check via
    passthrough)."""
    from rods_a.quantize_act import quantize_act
    from rods_a.constants import gamma_star
    torch.manual_seed(12)
    Cin = 1152
    W = torch.randn(48, Cin, dtype=torch.float32)
    X = torch.randn(24, Cin, dtype=torch.float32) + 0.4
    pack = prepare_weight(W, bits=4, group_size=64, rotation="kronecker_d",
                          rotation_seed=5, strict_rho=False, bitpack=False)
    assert pack.Cin_pad == Cin
    x_hat, diag = quantize_act(
        X, b=4, gamma_star=gamma_star(4, "gaussian"), rotate_act=True,
        rotation="kronecker_d", rotation_seed=5, center_mean=True,
        return_rotated=True, return_diag=True, passthrough=True)
    from rods_a.fwht import rotate_for
    Wt = rotate_for("kronecker_d", "fwht", seed=5)(W)
    y = x_hat @ Wt.t() + diag["x_bar"] * W.sum(dim=1)
    y_fp = X @ W.t()
    rel = ((y - y_fp).norm() / y_fp.norm().clamp_min(1e-12)).item()
    assert rel < 1e-4, f"kron_d composed invariance broken: {rel:.2e}"
