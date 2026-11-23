"""
Fast Walsh-Hadamard transform and the orthonormal rotation used by RoDS-A.

Conventions
-----------
`fwht(X)` applies the *unnormalized* Walsh-Hadamard transform along the last
axis, batched over all leading dims. For the natural (Sylvester) ordering this
equals ``X @ H`` where ``H`` is the symmetric +/-1 Sylvester-Hadamard matrix
(``H @ H = C * I``). Hence:

    fwht(X) / sqrt(C)            == X @ (H / sqrt(C))          # orthonormal rotation
    fwht(fwht(X)) / C            == X                          # involution

`rotate(X)` is the orthonormal rotation ``R = H / sqrt(C)`` used on the
contraction axis, with **pad-to-next-power-of-two** fallback for non-power-of-two
channel counts. ``R`` is symmetric and ``R @ R = I``, so `unrotate` is the same
transform followed by a truncation back to the original width.

For deployment the pad fallback inflates the contraction width K (it caps the
INT4 speedup), so non-power-of-two ``C = m * 2^k`` instead uses the **Kronecker**
rotation ``R = (H_m (x) H_{2^k}) / sqrt(C)`` (`kron_rotate`), which keeps the
native width ``K = C`` (no padding). `rotate_for` / `unrotate_for` dispatch
between the ``"pad"`` and ``"kronecker"`` schemes; the pad path is bit-untouched.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from rods_a import profiling as _profiling


def next_pow2(n: int) -> int:
    """Smallest power of two >= n."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return 1 << (n - 1).bit_length()


def is_pow2(n: int) -> bool:
    return n >= 1 and (n & (n - 1)) == 0


def pad_to_pow2(X: Tensor) -> Tensor:
    """Zero-pad the last axis up to the next power of two (no-op if already)."""
    n = X.shape[-1]
    p = next_pow2(n)
    if p == n:
        return X
    return torch.nn.functional.pad(X, (0, p - n))


def fwht(X: Tensor) -> Tensor:
    """Unnormalized Walsh-Hadamard transform along the last axis (batched).

    Requires the last dim to be a power of two; callers needing arbitrary widths
    should `pad_to_pow2` first (see `rotate`). Returns a new tensor; `X` is not
    modified.
    """
    n = X.shape[-1]
    if not is_pow2(n):
        raise ValueError(
            f"fwht requires a power-of-two last dim, got {n}. "
            f"Pad with pad_to_pow2 / use rotate() for arbitrary widths."
        )
    lead = X.shape[:-1]
    y = X.reshape(-1, n).clone()
    h = 1
    while h < n:
        # View each length-2h block as [half(2), within(h)] and butterfly.
        y = y.view(-1, n // (2 * h), 2, h)
        a = y[:, :, 0, :]
        b = y[:, :, 1, :]
        y = torch.stack((a + b, a - b), dim=2)
        y = y.reshape(-1, n)
        h *= 2
    return y.reshape(*lead, n)


def dense_hadamard(n: int, dtype=torch.float32, device=None) -> Tensor:
    """Dense symmetric Sylvester-Hadamard matrix H (n x n, entries +/-1).

    For tests only: ``fwht(X) == X @ dense_hadamard(C)`` for power-of-two C.
    """
    if not is_pow2(n):
        raise ValueError(f"dense_hadamard needs power-of-two n, got {n}")
    H = torch.ones(1, 1, dtype=dtype, device=device)
    while H.shape[0] < n:
        top = torch.cat([H, H], dim=1)
        bot = torch.cat([H, -H], dim=1)
        H = torch.cat([top, bot], dim=0)
    return H


def rotate(X: Tensor) -> Tensor:
    """Orthonormal Hadamard rotation R = H / sqrt(C_padded) on the last axis,
    with pad-to-pow2 fallback. Output width is next_pow2(C)."""
    Xp = pad_to_pow2(X)
    return fwht(Xp) / math.sqrt(Xp.shape[-1])


def unrotate(Xr: Tensor, orig_dim: int) -> Tensor:
    """Inverse of `rotate`: apply R again (R is an involution) and truncate back
    to `orig_dim`."""
    X = fwht(Xr) / math.sqrt(Xr.shape[-1])
    return X[..., :orig_dim]


# --------------------------------------------------------------------------- #
# Dense-matmul rotation (fast path for GPU)
#
# `rotate`/`unrotate` above run an 11-stage Python butterfly: O(C log C) FLOPs
# but ~60 tiny CUDA launches + fresh allocations PER CALL, which dominates the
# per-layer cost on GPU. Since the rotation is just `X @ (H/√C)`, we instead
# cache the dense orthonormal Hadamard `H/√C` once per (C, device, dtype) and do
# a single GEMM (tensor cores, fp32 accumulation). It is numerically identical to
# `rotate` up to fp rounding (H is the same Sylvester matrix as `dense_hadamard`,
# i.e. `fwht(X) == X @ dense_hadamard(C)`), so it satisfies the same invariance.
# O(C^2) FLOPs, but the GEMM crushes the Python loop in wall-clock at these sizes.
# --------------------------------------------------------------------------- #

_HN_CACHE: dict = {}


def _orthonormal_hadamard(n: int, device, dtype) -> Tensor:
    """Cached H/√n (n x n, orthonormal). Built once per (n, device, dtype)."""
    key = (n, str(device), dtype)
    H = _HN_CACHE.get(key)
    if H is None:
        H = dense_hadamard(n, dtype=torch.float32, device=device) / math.sqrt(n)
        H = H.to(dtype)
        _HN_CACHE[key] = H
    return H


def dense_rotate(X: Tensor) -> Tensor:
    """Orthonormal Hadamard rotation via a single cached GEMM (pad-to-pow2).
    Drop-in equivalent of `rotate` for the hot per-forward path. Output dtype
    follows X (GEMM accumulates in fp32 internally on tensor cores)."""
    # events-only sub-bracket: the pad share of the caller's `rotate` stage
    # (a nested synchronize would corrupt the enclosing sync section).
    with _profiling.section("rotate_pad", events_only=True):
        Xp = pad_to_pow2(X)
    H = _orthonormal_hadamard(Xp.shape[-1], Xp.device, Xp.dtype)
    return Xp @ H


def dense_unrotate(Xr: Tensor, orig_dim: int) -> Tensor:
    """Inverse of `dense_rotate` (R is an involution): one GEMM, then truncate."""
    H = _orthonormal_hadamard(Xr.shape[-1], Xr.device, Xr.dtype)
    return (Xr @ H)[..., :orig_dim]


# --------------------------------------------------------------------------- #
# Kronecker (deployment-K) rotation:  R = (H_m (x) H_{2^k}) / sqrt(Cin)
#
# The pad-to-pow2 `rotate` above inflates the contraction width K to the next
# power of two (1152->2048, 4608->8192), which inflates every INT4 GEMM and caps
# the speedup at ~2.25x. For Cin = m * 2^k with a vendored order-m Hadamard H_m,
# the Kronecker rotation R = (H_m (x) H_{2^k})/sqrt(Cin) is an orthonormal map on
# the *native* width Cin (no padding, K = Cin). Both factors are symmetric +/-1
# Hadamard matrices, so R is a symmetric involution (R = R^T, R R = I): `rotate`
# and `unrotate` are the same transform and no truncation is needed.
#
# Vendored m factors (provenance below); raise for anything else.
#   m = 36 : PixArt-Sigma (1152 = 36*32, 4608 = 36*128). The order-36 Hadamard is
#            built constructively via Paley type II from the prime q = 17
#            (q = 1 mod 4): the (q+1)=18 symmetric conference matrix C
#            (C C^T = 17 I, zero diag) lifts to H_36 = I_18 (x) A + C (x) B with
#            A = [[1,-1],[-1,-1]], B = [[1,1],[1,-1]], giving H_36 H_36^T = 36 I
#            (verified integer-exact at build). This is a public-domain
#            mathematical construction -- no third-party code is vendored.
#            Ref: R.E.A.C. Paley, "On orthogonal matrices", J. Math. Phys. 12
#            (1933) 311-320.
#   m = 12 : FLUX (3072 = 12*256) is noted but intentionally NOT implemented here.
# --------------------------------------------------------------------------- #

_KRON_M_SUPPORTED = (36,)


def _paley2_hadamard(q: int) -> Tensor:
    """Symmetric order-2(q+1) Hadamard via Paley type II (q prime, q = 1 mod 4).

    Builds the Jacobsthal/conference matrix over GF(q) then the 2x2 block lift.
    Returns an int64 +/-1 matrix; correctness is asserted by the caller.
    """
    QR = {(x * x) % q for x in range(1, q)}            # nonzero quadratic residues

    def chi(a: int) -> int:
        a %= q
        return 0 if a == 0 else (1 if a in QR else -1)

    n = q + 1
    C = torch.zeros(n, n, dtype=torch.int64)
    for j in range(1, n):
        C[0, j] = 1
        C[j, 0] = 1
    for i in range(1, n):
        for j in range(1, n):
            C[i, j] = chi((j - 1) - (i - 1))           # Jacobsthal chi(j - i)
    A = torch.tensor([[1, -1], [-1, -1]], dtype=torch.int64)
    B = torch.tensor([[1, 1], [1, -1]], dtype=torch.int64)
    H = torch.zeros(2 * n, 2 * n, dtype=torch.int64)
    for i in range(n):
        for j in range(n):
            blk = A if i == j else (int(C[i, j].item()) * B)
            H[2 * i:2 * i + 2, 2 * j:2 * j + 2] = blk
    return H


_HM_CACHE: dict = {}


def dense_hadamard_m(m: int, dtype=torch.float32, device=None) -> Tensor:
    """Dense order-m Hadamard (+/-1) for a vendored non-pow2 m. Built once and
    verified integer-exact (entries +/-1, symmetric, H H^T = m I)."""
    if m not in _KRON_M_SUPPORTED:
        raise ValueError(
            f"No vendored order-{m} Hadamard. Supported m: {_KRON_M_SUPPORTED}. "
            f"(FLUX m=12 for 3072=12*256 is noted but not implemented.)"
        )
    H = _HM_CACHE.get(m)
    if H is None:
        if m == 36:
            H = _paley2_hadamard(17)
        else:  # pragma: no cover - guarded by _KRON_M_SUPPORTED
            raise ValueError(f"unreachable m={m}")
        eye = torch.eye(m, dtype=torch.int64)
        assert set(H.unique().tolist()) <= {-1, 1}, f"H_{m} entries not +/-1"
        assert torch.equal(H, H.t()), f"H_{m} not symmetric"
        assert torch.equal(H @ H.t(), m * eye), f"H_{m} H^T != {m} I"
        _HM_CACHE[m] = H
    return H.to(dtype=dtype, device=device)


def factor_kron(Cin: int) -> tuple[int, int]:
    """Factor ``Cin = m * 2^k`` with a vendored order-m Hadamard. Returns
    ``(m, 2^k)``; raises if no supported m divides Cin into a power of two."""
    for m in _KRON_M_SUPPORTED:
        if Cin % m == 0 and is_pow2(Cin // m):
            return m, Cin // m
    raise ValueError(
        f"Cin={Cin} is not m*2^k for any vendored m in {_KRON_M_SUPPORTED} "
        f"(need Cin/m a power of two)."
    )


def kron_rotate(X: Tensor) -> Tensor:
    """Orthonormal Kronecker-Hadamard rotation ``R=(H_m (x) H_{2^k})/sqrt(Cin)``
    on the last axis. No padding: output width == Cin. Equals ``X @ R`` exactly
    (both Hadamard factors are symmetric)."""
    Cin = X.shape[-1]
    m, two_k = factor_kron(Cin)
    lead = X.shape[:-1]
    Y = X.reshape(*lead, m, two_k)
    Y = fwht(Y)                                        # H_{2^k} on the fast axis
    Hm = dense_hadamard_m(m, dtype=Y.dtype, device=Y.device)
    Y = torch.einsum("ab,...bk->...ak", Hm, Y)         # H_m on the slow axis
    return Y.reshape(*lead, Cin) / math.sqrt(Cin)


def kron_unrotate(Xr: Tensor, orig_dim: Optional[int] = None) -> Tensor:
    """Inverse of `kron_rotate` (R is a symmetric involution): the same transform.
    `orig_dim` is accepted for dispatch-signature parity and, if given, must equal
    the (unchanged) width."""
    if orig_dim is not None and orig_dim != Xr.shape[-1]:
        raise ValueError(
            f"kron_unrotate width {Xr.shape[-1]} != orig_dim {orig_dim} "
            f"(Kronecker rotation preserves width; no truncation)."
        )
    return kron_rotate(Xr)


_HKRON_CACHE: dict = {}


def _orthonormal_kron(Cin: int, device, dtype) -> Tensor:
    """Cached dense ``(H_m (x) H_{2^k})/sqrt(Cin)`` (Cin x Cin, orthonormal)."""
    key = (Cin, str(device), dtype)
    H = _HKRON_CACHE.get(key)
    if H is None:
        m, two_k = factor_kron(Cin)
        Hm = dense_hadamard_m(m, dtype=torch.float32)
        H2 = dense_hadamard(two_k, dtype=torch.float32)
        H = (torch.kron(Hm, H2) / math.sqrt(Cin)).to(dtype)
        if device is not None:
            H = H.to(device)
        _HKRON_CACHE[key] = H
    return H


def dense_kron_rotate(X: Tensor) -> Tensor:
    """Orthonormal Kronecker-Hadamard rotation via a single cached GEMM (no pad).
    Drop-in equivalent of `kron_rotate` for the hot per-forward path."""
    H = _orthonormal_kron(X.shape[-1], X.device, X.dtype)
    return X @ H


def dense_kron_unrotate(Xr: Tensor, orig_dim: Optional[int] = None) -> Tensor:
    """Inverse of `dense_kron_rotate` (R is a symmetric involution): one GEMM."""
    if orig_dim is not None and orig_dim != Xr.shape[-1]:
        raise ValueError(
            f"dense_kron_unrotate width {Xr.shape[-1]} != orig_dim {orig_dim}."
        )
    H = _orthonormal_kron(Xr.shape[-1], Xr.device, Xr.dtype)
    return Xr @ H


# --------------------------------------------------------------------------- #
# "kronecker_d" (U4.B): randomized-Hadamard-transform proper, R = D·(H_kron/√C)
# with a SEEDED Rademacher ±1 diagonal D applied to BOTH sides (x̃ = (x∘d)Ĥ,
# W̃ = (W∘d)Ĥ, same d), restoring the incoherence guarantee the deterministic
# Kronecker basis lacks (U3: structured-mixing deficit). R is orthogonal but NOT
# an involution: x = (x̃ Ĥ)∘d. d derives deterministically from the seed (CPU
# generator, cached per (n, seed, device, dtype)) — never stored as a loose
# tensor, so any consumer reproduces it from (rotation, rotation_seed) alone.
# --------------------------------------------------------------------------- #

_RADEMACHER_CACHE: dict = {}


def rademacher_diag(n: int, seed: int, device=None, dtype=torch.float32) -> Tensor:
    """Seeded ±1 diagonal, device-independent (built on CPU, then moved)."""
    key = (n, int(seed), str(device), dtype)
    d = _RADEMACHER_CACHE.get(key)
    if d is None:
        g = torch.Generator().manual_seed(int(seed))
        d = (torch.randint(0, 2, (n,), generator=g, dtype=torch.int64) * 2 - 1).to(dtype)
        if device is not None:
            d = d.to(device)
        _RADEMACHER_CACHE[key] = d
    return d


def kron_d_rotate(X: Tensor, seed: int = 0) -> Tensor:
    """x̃ = (x ∘ d) Ĥ_kron — sign flip then the deterministic Kronecker rotate."""
    d = rademacher_diag(X.shape[-1], seed, X.device, X.dtype)
    return kron_rotate(X * d)


def kron_d_unrotate(Xr: Tensor, orig_dim: Optional[int] = None, seed: int = 0) -> Tensor:
    """Inverse of kron_d_rotate: x = (x̃ Ĥ_kron) ∘ d (R is not an involution)."""
    out = kron_unrotate(Xr, orig_dim)
    d = rademacher_diag(out.shape[-1], seed, out.device, out.dtype)
    return out * d


def dense_kron_d_rotate(X: Tensor, seed: int = 0) -> Tensor:
    d = rademacher_diag(X.shape[-1], seed, X.device, X.dtype)
    return dense_kron_rotate(X * d)


def dense_kron_d_unrotate(Xr: Tensor, orig_dim: Optional[int] = None, seed: int = 0) -> Tensor:
    out = dense_kron_unrotate(Xr, orig_dim)
    d = rademacher_diag(out.shape[-1], seed, out.device, out.dtype)
    return out * d


# --------------------------------------------------------------------------- #
# "dense_haar" (R1.3): a dense Haar random-orthogonal rotation R = Q on the native
# width (Q Qᵀ = I, no padding, K = Cin). This is the DIAGNOSTIC native-width
# incoherence CEILING for the R2 gate: the densest possible d=1152 mixing, used to
# test whether pad's A4 SQNR edge is width-intrinsic or mixing-intrinsic. It is NOT
# kernel-realizable (a full dense GEMM has no INT4 packing) — fake-quant measurement
# only. Q is orthogonal but NOT symmetric, so it is not an involution: x = x̃ Qᵀ. Q
# derives deterministically from the seed (CPU generator; QR of a seeded Gaussian
# with the sign(diag(R)) correction for a true Haar draw over O(n)), cached per
# (n, seed) — never stored as a loose tensor, so any consumer reproduces it from
# (rotation, rotation_seed) alone (the SAME Q drives both the activation rotate and
# the weight refit, so X̃ W̃ᵀ == X Wᵀ holds).
# --------------------------------------------------------------------------- #

_HAAR_CACHE: dict = {}


def haar_orthogonal(n: int, seed: int, device=None, dtype=torch.float32) -> Tensor:
    """Seeded dense Haar-distributed orthogonal matrix (n x n), device-independent
    (built once on CPU in fp64 for a clean QR, stored fp32, then moved/cast)."""
    key = (n, int(seed))
    Q = _HAAR_CACHE.get(key)
    if Q is None:
        g = torch.Generator().manual_seed(int(seed))
        A = torch.randn(n, n, generator=g, dtype=torch.float64)
        Q64, R = torch.linalg.qr(A)
        # Sign-correct columns by sign(diag(R)) so the draw is uniform over O(n).
        s = torch.sign(torch.diagonal(R))
        s[s == 0] = 1.0
        Q = (Q64 * s.unsqueeze(0)).to(torch.float32)
        _HAAR_CACHE[key] = Q
    out = Q.to(dtype=dtype)
    if device is not None:
        out = out.to(device)
    return out


def haar_rotate(X: Tensor, seed: int = 0) -> Tensor:
    """x̃ = x Q — dense Haar rotation on the last axis (no padding, width preserved)."""
    Q = haar_orthogonal(X.shape[-1], seed, X.device, X.dtype)
    return X @ Q


def haar_unrotate(Xr: Tensor, orig_dim: Optional[int] = None, seed: int = 0) -> Tensor:
    """Inverse of `haar_rotate`: x = x̃ Qᵀ (Q is orthogonal, not an involution).
    `orig_dim` is accepted for dispatch-signature parity and, if given, must equal
    the (unchanged) width."""
    if orig_dim is not None and orig_dim != Xr.shape[-1]:
        raise ValueError(
            f"haar_unrotate width {Xr.shape[-1]} != orig_dim {orig_dim} "
            f"(Haar rotation preserves width; no truncation)."
        )
    Q = haar_orthogonal(Xr.shape[-1], seed, Xr.device, Xr.dtype)
    return Xr @ Q.t()


# --------------------------------------------------------------------------- #
# Scheme dispatch: "pad" (zero-pad to pow2) | "kronecker" (deployment K = Cin)
# | "kronecker_d" (seeded RHT, U4.B) | "dense_haar" (R1.3 diagnostic ceiling).
# Orthogonal to the `impl` ("fwht"|"dense") performance axis; "dense_haar" is
# inherently a dense GEMM so both impls resolve to the same callable. The pad and
# kronecker branches are bit-untouched and ignore `seed`, so every existing caller
# is unchanged.
# --------------------------------------------------------------------------- #

ROTATION_SCHEMES = ("pad", "kronecker", "kronecker_d", "dense_haar")


def rotate_for(scheme: str, impl: str = "fwht", *, seed: int = 0):
    """Return the rotate callable ``f(X) -> X_rot`` for (scheme, impl)."""
    if scheme == "pad":
        return dense_rotate if impl == "dense" else rotate
    if scheme == "kronecker":
        return dense_kron_rotate if impl == "dense" else kron_rotate
    if scheme == "kronecker_d":
        fn = dense_kron_d_rotate if impl == "dense" else kron_d_rotate
        return lambda X: fn(X, seed=seed)
    if scheme == "dense_haar":
        return lambda X: haar_rotate(X, seed=seed)
    raise ValueError(f"unknown rotation scheme {scheme!r}; choose from {ROTATION_SCHEMES}")


def unrotate_for(scheme: str, impl: str = "fwht", *, seed: int = 0):
    """Return the unrotate callable ``f(Xr, orig_dim) -> X`` for (scheme, impl)."""
    if scheme == "pad":
        return dense_unrotate if impl == "dense" else unrotate
    if scheme == "kronecker":
        return dense_kron_unrotate if impl == "dense" else kron_unrotate
    if scheme == "kronecker_d":
        fn = dense_kron_d_unrotate if impl == "dense" else kron_d_unrotate
        return lambda Xr, orig_dim=None: fn(Xr, orig_dim, seed=seed)
    if scheme == "dense_haar":
        return lambda Xr, orig_dim=None: haar_unrotate(Xr, orig_dim, seed=seed)
    raise ValueError(f"unknown rotation scheme {scheme!r}; choose from {ROTATION_SCHEMES}")


# --------------------------------------------------------------------------- #
# R1.4: fixed-shape order-2^k factor-discrepancy sweep (ADD-only, diagnostic).
#
# D(H) = max_j |sum_i H_ij|  -- the worst-case absolute COLUMN sum. This is NOT
# the L1 column norm sum_i |H_ij| (which is a constant = n for any +/-1 matrix and
# carries no signal). Under the Kronecker product column sums multiply, so
#   D(H_m (x) H_k) = D(H_m) * D(H_k)
# i.e. D is a per-factor, multiplicative worst-case-mixing budget. For the DC
# (all-ones) input, R.(1/sqrt(C)).1 concentrates a fraction D(H_kron)/C on its
# peak coordinate, so smaller D == a more even (lower-discrepancy) DC spread.
#
# STATED LIMITATION (explicit, not implied): a *regular* Hadamard of order 32 does
# NOT exist -- a regular Hadamard requires n = 4u^2 and 32 is not a perfect square
# (its D would be sqrt(32) ~ 5.66, unreachable). The variants below are LOWER-D
# REALIZABLE order-2^k variants (row sign-flips + permutations of the Sylvester
# baseline), NEVER a regular Hadamard. Reports and comments must say "lower-D
# variant," never "regular Hadamard."
#
# Only ROW sign-flips move D: negating row i shifts every column sum by -2 H_ij.
# Row permutations preserve each column sum; column permutations reorder the
# column-sum multiset (max unchanged); output-column sign-flips flip a column
# sum's sign (|.| unchanged). So permutation variants are same-D LOCALITY
# CONTROLS, and the D-spread for the regression comes from the row-sign-flip
# family. Row-flipped Sylvester is generally NOT symmetric and NOT butterfly
# (FWHT) compatible -> variants run via a dense GEMM (fine: host-only diagnostic).
# --------------------------------------------------------------------------- #


def hadamard_discrepancy(H: Tensor) -> float:
    """D(H) = max_j |sum_i H_ij| -- worst-case absolute column sum (fp64 reduce).

    Multiplicative under the Kronecker product: D(A (x) B) = D(A) * D(B). See the
    module note: this is deliberately NOT the L1 column norm (constant for a +/-1
    matrix). Lower D == a lower-discrepancy (more even) mixing of a DC input.
    """
    return float(H.to(torch.float64).sum(dim=0).abs().max())


def h32_row_sign_variant(seed: int, n: int = 32) -> Tensor:
    """Row-sign-flipped Sylvester-Hadamard ``diag(+/-1) @ H_n`` (fp64, +/-1).

    Still orthogonal (H H^T = n I) but generally neither symmetric nor
    Sylvester-butterfly compatible (applied via dense GEMM). Row flips are the
    ONLY lever that moves D(H) -- this is the D-spread family for the sweep."""
    H = dense_hadamard(n, dtype=torch.float64)
    g = torch.Generator().manual_seed(int(seed))
    d = torch.randint(0, 2, (n,), generator=g, dtype=torch.float64) * 2 - 1
    return d.unsqueeze(1) * H                              # scale each row by +/-1


def h32_perm_variant(seed: int, n: int = 32) -> Tensor:
    """Row+column permuted Sylvester-Hadamard ``P H P^T`` (fp64, +/-1). Symmetric
    and orthogonal, with the SAME D as the baseline -- a same-D LOCALITY CONTROL
    (permutations do not change the max column sum)."""
    H = dense_hadamard(n, dtype=torch.float64)
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=g)
    return H[perm][:, perm]


def enumerate_h32_variants(
    n: int = 32,
    n_rowflip: int = 32,
    n_perm: int = 4,
    rowflip_seed0: int = 1000,
    perm_seed0: int = 9000,
) -> List[Dict]:
    """Order-``n`` Hadamard-equivalent variants for the R1.4 discrepancy sweep.

    Returns a list of dicts ``{variant_id, family, seed, H (n x n fp64), D,
    preserves_fast_transform}``. Families: ``sylvester`` (baseline, D=n, preserves
    FWHT), ``rowflip`` (the D-spread family), ``perm`` (same-D locality controls).
    Every variant is asserted +/-1 and orthogonal (H H^T = n I)."""
    variants: List[Dict] = [{
        "variant_id": "sylvester", "family": "sylvester", "seed": None,
        "H": dense_hadamard(n, dtype=torch.float64),
        "preserves_fast_transform": True}]
    for i in range(n_rowflip):
        s = rowflip_seed0 + i
        variants.append({"variant_id": f"rowflip_s{s}", "family": "rowflip",
                         "seed": s, "H": h32_row_sign_variant(s, n),
                         "preserves_fast_transform": False})
    for i in range(n_perm):
        s = perm_seed0 + i
        variants.append({"variant_id": f"perm_s{s}", "family": "perm",
                         "seed": s, "H": h32_perm_variant(s, n),
                         "preserves_fast_transform": False})
    eye = torch.eye(n, dtype=torch.float64)
    for v in variants:
        H = v["H"]
        assert set(torch.unique(H).tolist()) <= {-1.0, 1.0}, \
            f"{v['variant_id']}: entries not +/-1"
        assert torch.allclose(H @ H.t(), n * eye), \
            f"{v['variant_id']}: not orthogonal (H H^T != {n} I)"
        v["D"] = hadamard_discrepancy(H)
    return variants


def kron_matrix_from_h32(H32: Tensor, m: int = 36) -> Tensor:
    """Dense orthonormal ``(H_m (x) H32) / sqrt(m * two_k)`` for a custom order-
    ``two_k`` factor H32 (m defaults to the vendored PixArt order 36). fp32."""
    two_k = H32.shape[0]
    C = m * two_k
    Hm = dense_hadamard_m(m, dtype=torch.float64)
    Hk = torch.kron(Hm, H32.to(torch.float64)) / math.sqrt(C)
    return Hk.to(torch.float32)


def kron_variant_rotator(
    H32: Tensor, m: int = 36
) -> Tuple[Callable[[Tensor], Tensor], Callable[..., Tensor]]:
    """Build ``(rot, unrot)`` callables for the custom-H32 Kronecker basis, around
    a dense ``Hk = (H_m (x) H32)/sqrt(C)`` built once. ``rot: X -> X @ Hk``;
    ``unrot: Xr, orig_dim=None -> Xr @ Hk^T`` (row-flip variants are not symmetric,
    so this is not an involution -- mirrors `haar_unrotate`). Width is preserved.
    Drop-in for `quantize_act(rotate_override=..., unrotate_override=...)`."""
    Hk = kron_matrix_from_h32(H32, m)
    HkT = Hk.t().contiguous()
    C = Hk.shape[0]

    def rot(X: Tensor) -> Tensor:
        return X @ Hk.to(device=X.device, dtype=X.dtype)

    def unrot(Xr: Tensor, orig_dim: Optional[int] = None) -> Tensor:
        if orig_dim is not None and orig_dim != C:
            raise ValueError(
                f"kron variant preserves width {C}, got orig_dim {orig_dim}")
        return Xr @ HkT.to(device=Xr.device, dtype=Xr.dtype)

    return rot, unrot
