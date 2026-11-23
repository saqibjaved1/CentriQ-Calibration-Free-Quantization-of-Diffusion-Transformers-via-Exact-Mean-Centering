"""Paley constructions for Hadamard matrices (public-domain mathematics).

Both constructions build the Jacobsthal matrix from the quadratic character
chi over GF(q) (q prime) and lift it to a +/-1 Hadamard matrix:

* Paley type I  (q = 3 mod 4): order q+1,   H = I + S with skew S, H H^T = (q+1) I.
* Paley type II (q = 1 mod 4): order 2(q+1), symmetric 2x2 block lift,
  H H^T = 2(q+1) I. Same construction as ``rods_a.fwht._paley2_hadamard``
  (re-implemented here so ``core`` does not import ``rods_a``; equality for
  q=17 is pinned by tests/test_hadamard_matrices.py).

Ref: R.E.A.C. Paley, "On orthogonal matrices", J. Math. Phys. 12 (1933) 311-320.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["paley1_hadamard", "paley2_hadamard"]


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    d = 2
    while d * d <= n:
        if n % d == 0:
            return False
        d += 1
    return True


def _chi_table(q: int) -> list:
    """Quadratic character chi(a) over GF(q): 0 at 0, +1 on residues, -1 else."""
    qr = {(x * x) % q for x in range(1, q)}
    return [0 if a == 0 else (1 if a in qr else -1) for a in range(q)]


def paley1_hadamard(q: int) -> Tensor:
    """Order-(q+1) Hadamard via Paley type I (q prime, q = 3 mod 4).

    Jacobsthal ``Q_ij = chi(j - i)`` (skew for q = 3 mod 4); border it into the
    skew matrix ``S = [[0, 1^T], [-1, Q]]`` and return ``H = I + S`` (entries
    +/-1, ``H H^T = (q+1) I``). Returns int64; correctness is asserted by the
    caller (`build_hadamard`).
    """
    if not _is_prime(q) or q % 4 != 3:
        raise ValueError(f"Paley I needs a prime q = 3 (mod 4), got q={q}")
    chi = _chi_table(q)
    n = q + 1
    H = torch.zeros(n, n, dtype=torch.int64)
    for j in range(1, n):
        H[0, j] = 1                                     # top border of S
        H[j, 0] = -1                                    # left border of S
    for i in range(1, n):
        for j in range(1, n):
            H[i, j] = chi[(j - i) % q]                  # Jacobsthal chi(j - i)
    H += torch.eye(n, dtype=torch.int64)                # H = I + S (Q diag is 0)
    return H


def paley2_hadamard(q: int) -> Tensor:
    """Symmetric order-2(q+1) Hadamard via Paley type II (q prime, q = 1 mod 4).

    Builds the (q+1)-dim symmetric conference matrix from the Jacobsthal matrix,
    then the 2x2 block lift with A = [[1,-1],[-1,-1]], B = [[1,1],[1,-1]].
    Returns int64; correctness is asserted by the caller (`build_hadamard`).
    """
    if not _is_prime(q) or q % 4 != 1:
        raise ValueError(f"Paley II needs a prime q = 1 (mod 4), got q={q}")
    chi = _chi_table(q)
    n = q + 1
    C = torch.zeros(n, n, dtype=torch.int64)
    for j in range(1, n):
        C[0, j] = 1
        C[j, 0] = 1
    for i in range(1, n):
        for j in range(1, n):
            C[i, j] = chi[((j - 1) - (i - 1)) % q]      # Jacobsthal chi(j - i)
    A = torch.tensor([[1, -1], [-1, -1]], dtype=torch.int64)
    B = torch.tensor([[1, 1], [1, -1]], dtype=torch.int64)
    H = torch.zeros(2 * n, 2 * n, dtype=torch.int64)
    for i in range(n):
        for j in range(n):
            blk = A if i == j else (int(C[i, j].item()) * B)
            H[2 * i:2 * i + 2, 2 * j:2 * j + 2] = blk
    return H
