"""
Integer-exactness gates for the Hadamard matrix factory (core/hadamard).

These are the real correctness gate for the factorization sweep: the goldens
firewall cannot distinguish a wrong-but-orthogonal matrix from the intended one,
so every vendored order, every Paley construction, every pinned composition and
every grid (p, q) product is checked EXACTLY:

  (i)  entries in {-1, +1};
  (ii) H @ H^T == n I in exact arithmetic (int64, or fp64 GEMM -- products/sums
       of +-1 with partial sums <= 16384 are exactly representable in fp64, so
       fp64 equality IS integer equality);
  (iii) shape n x n.

For composed sizes above _FULL_EXACT_MAX the full n^3 GEMM is replaced by a
seeded 256-row exact check PLUS the structural pin
build_kron(p,q) == kron(H_p, H_q): exactness is compositional (a Kronecker
product of exact Hadamard matrices is an exact Hadamard matrix), so nothing is
approximated.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from core.hadamard import (
    COMPOSED_ORDERS,
    CONSTRUCTED_ORDERS,
    CONSTRUCTED_VARIANTS,
    VENDORED_ORDERS,
    build_hadamard,
    build_kron,
    composition_tree,
    order_of,
)
from core.hadamard.paley import paley1_hadamard, paley2_hadamard
from rods_a.fwht import dense_hadamard, dense_hadamard_m

# Every top-level (p, q) pair of the sweep grid (all three models, natives + pads).
GRID_PAIRS = [
    # PixArt 1152 (+ pads at 1280)
    (36, 32), (24, 48), (16, 72), (12, 96), (20, 64), (40, 32),
    # PixArt 4608 (+ pad at 5120)
    (36, 128), (48, 96), (64, 72), (40, 128),
    # SANA 2240 (+ pads at 2304 / 2560)
    (40, 56), (28, 80), (20, 112), (16, 140), (36, 64), (40, 64),
    # SANA FFN pads (5632 / 5760)
    (44, 128), (36, 160),
    # FLUX 3072 / 12288 — Paley-I atoms p48/p24: the composed 48 = 12 (x) 4 and
    # 24 = 12 (x) 2 re-associate with Sylvester co-factors to the SAME 12 (x) 2^k
    # matrix, so only distinct atoms give distinct candidates at these widths.
    ("p48", 64), ("p24", 128), (12, 256), ("p48", 256), ("p24", 512), (12, 1024),
]

_FULL_EXACT_MAX = 2560   # full n^3 fp64 GEMM up to here; sampled+structural above
_SAMPLE_ROWS = 256


def _assert_pm1(H: torch.Tensor, n: int):
    assert H.shape == (n, n)
    assert H.abs().eq(1).all(), f"order {n}: entries not +-1"


def _assert_orthogonal_exact(H: torch.Tensor, n: int):
    """Full H @ H^T == n I, exact (fp64 GEMM of +-1 entries is exact here)."""
    H64 = H.to(torch.float64)
    expected = n * torch.eye(n, dtype=torch.float64)
    assert torch.equal(H64 @ H64.t(), expected), f"order {n}: H H^T != {n} I"


def _assert_orthogonal_sampled(H: torch.Tensor, n: int, seed: int = 0):
    """Seeded row-subset exact check (each asserted equality is exact)."""
    g = torch.Generator().manual_seed(seed)
    rows = torch.randperm(n, generator=g)[:_SAMPLE_ROWS]
    H64 = H.to(torch.float64)
    prod = H64[rows] @ H64.t()
    expected = torch.zeros(len(rows), n, dtype=torch.float64)
    expected[torch.arange(len(rows)), rows] = float(n)
    assert torch.equal(prod, expected), f"order {n}: sampled rows of H H^T != {n} I"


@pytest.mark.parametrize("n", VENDORED_ORDERS)
def test_vendored_orders_exact(n):
    H = build_hadamard(n)
    _assert_pm1(H, n)
    _assert_orthogonal_exact(H, n)


@pytest.mark.parametrize("q,n", [(11, 12), (19, 20), (23, 24), (43, 44), (47, 48)])
def test_paley1_builder_exact(q, n):
    # 12/20 cross-validate the construction against known-existing orders (same
    # order, different equivalence class than the vendored literals); 44 is the
    # deployed product (5632 = 44 x 128); 24/48 are the FLUX variant atoms.
    H = paley1_hadamard(q)
    _assert_pm1(H, n)
    _assert_orthogonal_exact(H, n)


@pytest.mark.parametrize("token", sorted(CONSTRUCTED_VARIANTS))
def test_variant_atoms_exact_and_distinct(token):
    n = order_of(token)
    H = build_hadamard(token)
    _assert_pm1(H, n)
    _assert_orthogonal_exact(H, n)
    # The variant is a genuinely different matrix than the composed default —
    # the whole reason it exists (re-association collapse at 12*2^k widths).
    assert not torch.equal(H, build_hadamard(n))


def test_flux_default_grid_degeneracy_on_record():
    # Documents the collapse the variants fix: composed 48 (x) 64 == 24 (x) 128
    # == 12 (x) 256 exactly (Kronecker re-association; Sylvester (x) Sylvester
    # is Sylvester).
    ref = build_kron(12, 256)
    assert torch.equal(build_kron(48, 64), ref)
    assert torch.equal(build_kron(24, 128), ref)
    assert not torch.equal(build_kron("p48", 64), ref)
    assert not torch.equal(build_kron("p24", 128), ref)


def test_paley_rejects_bad_q():
    with pytest.raises(ValueError):
        paley1_hadamard(17)   # 17 = 1 mod 4 -> Paley II territory
    with pytest.raises(ValueError):
        paley1_hadamard(15)   # not prime
    with pytest.raises(ValueError):
        paley2_hadamard(43)   # 43 = 3 mod 4 -> Paley I territory


def test_h36_matches_fwht_paley2():
    # Pins the core/ re-implementation to the in-repo Paley-II the kronecker
    # scheme deploys -- the (36, 32) factored spec must reproduce that basis.
    assert torch.equal(build_hadamard(36), dense_hadamard_m(36, dtype=torch.int64))


@pytest.mark.parametrize("n", [2, 8, 32, 64, 128, 256])
def test_sylvester_matches_fwht_dense_hadamard(n):
    # Same doubling layout as rods_a.fwht.dense_hadamard -- required for the
    # factored (36, 32) matrix to be bit-equal to the kron scheme's.
    assert torch.equal(build_hadamard(n),
                       dense_hadamard(n, dtype=torch.float32).to(torch.int64))


@pytest.mark.parametrize("n", sorted(COMPOSED_ORDERS))
def test_composed_table_exact(n):
    H = build_hadamard(n)
    _assert_pm1(H, n)
    _assert_orthogonal_exact(H, n)
    # The pinned tree is what was actually built.
    p, q = COMPOSED_ORDERS[n]
    assert torch.equal(H, torch.kron(build_hadamard(p), build_hadamard(q)))


def test_constructed_orders_registered():
    assert set(CONSTRUCTED_ORDERS) == {36, 44}
    for n in CONSTRUCTED_ORDERS:
        _assert_pm1(build_hadamard(n), n)


def test_unsupported_order_raises():
    with pytest.raises(ValueError):
        build_hadamard(100)   # not vendored/constructed/composed/pow2
    with pytest.raises(ValueError):
        composition_tree(100)


@pytest.mark.parametrize("p,q", GRID_PAIRS)
def test_grid_kron_exact(p, q):
    n = order_of(p) * order_of(q)
    H = build_kron(p, q)
    _assert_pm1(H, n)
    if n <= _FULL_EXACT_MAX:
        _assert_orthogonal_exact(H, n)
    else:
        # Structural pin (exactness is compositional) + sampled exact rows.
        assert torch.equal(H, torch.kron(build_hadamard(p), build_hadamard(q)))
        _assert_orthogonal_sampled(H, n)
    del H


def test_composition_tree_labels():
    assert composition_tree(36) == "36[paley2(q=17)]"
    assert composition_tree(140) == "had140"
    assert composition_tree(1024) == "2^10"
    assert "had28" in composition_tree(56)
    assert composition_tree("p24") == "p24[paley1(q=23)]"
    assert order_of("p48") == 48 and order_of(140) == 140
    with pytest.raises(ValueError):
        order_of("q24")
