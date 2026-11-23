"""Hadamard matrix factory for Kronecker-factorized rotations.

Resolves an order ``n`` to an exact +/-1 Hadamard matrix from one of four
sources, in this precedence:

1. vendored literals (orders 12, 20, 28, 40, 140 -- see ``vendored.py``),
2. Paley constructions (36 via type II q=17, 44 via type I q=43),
3. a pinned composition table (Kronecker products of smaller orders),
4. Sylvester doubling for powers of two.

The composition table is explicit rather than greedy so each derived order has
ONE canonical matrix (e.g. 80 = 20 (x) 4, not 40 (x) 2). Convention for derived
orders: the non-power-of-two factor sits on the slow (left) axis, matching the
existing ``H_m (x) H_{2^k}`` layout in ``rods_a.fwht``.

All non-power-of-two orders are verified integer-exact at first build (entries
+/-1 and ``H @ H^T == n I`` in int64); power-of-two orders are exact by the
Sylvester construction and covered by tests. This package must not import
``rods_a`` (equality of the order-36 matrix with ``rods_a.fwht.dense_hadamard_m``
is pinned by tests/test_hadamard_matrices.py instead).
"""

from __future__ import annotations

import torch
from torch import Tensor

from . import vendored
from .paley import paley1_hadamard, paley2_hadamard

__all__ = [
    "VENDORED_ORDERS",
    "CONSTRUCTED_ORDERS",
    "CONSTRUCTED_VARIANTS",
    "COMPOSED_ORDERS",
    "build_hadamard",
    "build_kron",
    "composition_tree",
    "order_of",
    "supported_orders",
]

VENDORED_ORDERS = (12, 20, 28, 40, 140)

# order -> (construction, q)
CONSTRUCTED_ORDERS = {36: ("paley2", 17), 44: ("paley1", 43)}

# Variant atoms, keyed by string token. These give a Hadamard matrix of an order
# that ALSO has a default (composed/vendored) construction, in a different
# equivalence class — needed because Kronecker products of Sylvester factors
# re-associate: with the default 24 = 12 (x) 2 and 48 = 12 (x) 4, any top-level
# split of a 12*2^k width collapses to the same 12 (x) 2^k matrix, so candidates
# at those widths are only distinct through genuinely different atoms.
CONSTRUCTED_VARIANTS = {"p24": ("paley1", 23), "p48": ("paley1", 47)}

# order -> (p, q) pinned composition: H_n = H_p (x) H_q
COMPOSED_ORDERS = {
    24: (12, 2),
    48: (12, 4),
    56: (28, 2),
    72: (36, 2),
    80: (20, 4),
    96: (12, 8),
    112: (28, 4),
    160: (40, 4),
}

_BUILD_CACHE: dict = {}


def _is_pow2(n: int) -> bool:
    return n >= 1 and (n & (n - 1)) == 0


def _sylvester(n: int) -> Tensor:
    """Sylvester-Hadamard H_n (int64, symmetric) by doubling; n a power of two.
    Same recursive [[H, H], [H, -H]] layout as ``rods_a.fwht.dense_hadamard``."""
    H = torch.ones(1, 1, dtype=torch.int64)
    while H.shape[0] < n:
        top = torch.cat([H, H], dim=1)
        bot = torch.cat([H, -H], dim=1)
        H = torch.cat([top, bot], dim=0)
    return H


def _assert_exact(H: Tensor, n: int) -> None:
    assert H.shape == (n, n), f"H_{n} shape {tuple(H.shape)}"
    assert H.abs().eq(1).all(), f"H_{n} entries not +/-1"
    eye = torch.eye(n, dtype=torch.int64)
    assert torch.equal(H @ H.t(), n * eye), f"H_{n} H^T != {n} I"


def supported_orders() -> tuple:
    """All non-power-of-two orders this factory can build directly."""
    return tuple(sorted(set(VENDORED_ORDERS) | set(CONSTRUCTED_ORDERS) | set(COMPOSED_ORDERS)))


def order_of(f) -> int:
    """Matrix order of a factor spec: an int order, or a variant token like
    "p24" (whose order is the trailing integer)."""
    if isinstance(f, int):
        return f
    if isinstance(f, str) and f in CONSTRUCTED_VARIANTS:
        return int(f[1:])
    raise ValueError(
        f"Unknown Hadamard factor {f!r}: expected an int order or a variant "
        f"token in {sorted(CONSTRUCTED_VARIANTS)}."
    )


def build_hadamard(n) -> Tensor:
    """Exact +/-1 Hadamard matrix for an int order ``n`` or a variant token
    (e.g. "p24" = Paley-I order 24). int64, cached; non-power-of-two orders are
    integer-exactness-checked at first build. Raises ValueError outside the
    vendored/constructed/variant/composed/Sylvester families.
    """
    H = _BUILD_CACHE.get(n)
    if H is not None:
        return H
    if isinstance(n, str):
        if n not in CONSTRUCTED_VARIANTS:
            raise ValueError(f"Unknown Hadamard variant token {n!r}; known: "
                             f"{sorted(CONSTRUCTED_VARIANTS)}")
        kind, q = CONSTRUCTED_VARIANTS[n]
        H = paley1_hadamard(q) if kind == "paley1" else paley2_hadamard(q)
        _assert_exact(H, order_of(n))
    elif n in VENDORED_ORDERS:
        H = getattr(vendored, f"get_hadamard_{n}")()
        _assert_exact(H, n)
    elif n in CONSTRUCTED_ORDERS:
        kind, q = CONSTRUCTED_ORDERS[n]
        H = paley1_hadamard(q) if kind == "paley1" else paley2_hadamard(q)
        _assert_exact(H, n)
    elif n in COMPOSED_ORDERS:
        p, q = COMPOSED_ORDERS[n]
        H = torch.kron(build_hadamard(p), build_hadamard(q))
        _assert_exact(H, n)
    elif _is_pow2(n):
        H = _sylvester(n)  # exact by construction; pinned by tests up to 256
    else:
        raise ValueError(
            f"No Hadamard construction for order {n}. Supported non-pow2 orders: "
            f"{supported_orders()} (plus any power of two)."
        )
    _BUILD_CACHE[n] = H
    return H


def build_kron(p, q, *, dtype: torch.dtype = torch.int64) -> Tensor:
    """Kronecker product ``H_p (x) H_q`` (order order_of(p)*order_of(q), entries
    +/-1). Factors are int orders or variant tokens.

    Exactness is compositional: both factors are exact Hadamard matrices, so
    the product is one (asserted factor-wise in `build_hadamard`; the composed
    product is additionally pinned by tests).
    """
    H = torch.kron(build_hadamard(p), build_hadamard(q))
    return H if dtype == torch.int64 else H.to(dtype)


def composition_tree(n) -> str:
    """Human-readable construction tree for an order / variant token."""
    if isinstance(n, str):
        if n not in CONSTRUCTED_VARIANTS:
            raise ValueError(f"No composition tree for variant token {n!r}.")
        kind, q = CONSTRUCTED_VARIANTS[n]
        return f"{n}[{kind}(q={q})]"
    if n in VENDORED_ORDERS:
        return f"had{n}"
    if n in CONSTRUCTED_ORDERS:
        kind, q = CONSTRUCTED_ORDERS[n]
        return f"{n}[{kind}(q={q})]"
    if n in COMPOSED_ORDERS:
        p, q = COMPOSED_ORDERS[n]
        return f"{n}=({composition_tree(p)})x({composition_tree(q)})"
    if _is_pow2(n):
        return f"2^{n.bit_length() - 1}"
    raise ValueError(f"No composition tree for order {n}.")
