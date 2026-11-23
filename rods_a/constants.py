"""
Phase 0 constants for RoDS-A: the data-free optimal-clip constant gamma* and a
few statistical helpers.

gamma* = argmin_{gamma>0} [ clip_err(gamma) + round_err(gamma) ] for a symmetric
uniform b-bit quantizer on a unit-variance distribution; derivation in doc 3.1
(RoDS_activation_proposal1.md).

LEVEL-COUNT DECISION: round_err uses level count 2^b, NOT the doc's literal
`q_levels = 2^b - 1`; only 2^b reproduces the doc's Gaussian and ACIQ Laplace targets.

UNITS
-----
gamma* is stored in sigma (unit-variance) units for BOTH families. The activation
scale estimators (rods_a.quantize_act) produce a std-equivalent sigma_hat, so
the clip is alpha = gamma*_sigma * sigma_hat. Laplace values in the doc's "x b_L"
convention are the sigma-unit values times sqrt(2).
"""

from __future__ import annotations

import math

# Bit-widths and families we tabulate.
# b=6 added for the Phase-3 activation-bits ladder (A6); same golden-section optimizer,
# reference values verified in GAMMA_STAR_REFERENCE below.
_BITS = (2, 3, 4, 6, 8)
_FAMILIES = ("gaussian", "laplace")

# Laplace scale for unit variance: Var = 2 b_L^2 = 1 => b_L = 1/sqrt(2).
_LAPLACE_BL = 1.0 / math.sqrt(2.0)


# --------------------------------------------------------------------------- #
# Standard-normal helpers (stdlib only; this module must import without torch).
# --------------------------------------------------------------------------- #
def _phi(x: float) -> float:
    """Standard normal pdf."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _Phi(x: float) -> float:
    """Standard normal cdf via erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def phi_inv(p: float) -> float:
    """Inverse standard-normal CDF (quantile function).

    Acklam's rational approximation, |abs error| < ~1.15e-9, refined with one
    Halley step. Used by the percentile scale estimator (Gaussian-consistent
    half-normal quantile) and to derive PHI_INV_999.
    """
    if not (0.0 < p < 1.0):
        raise ValueError(f"phi_inv requires 0 < p < 1, got {p}")

    # Coefficients (Acklam).
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]

    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
            ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    elif p <= phigh:
        q = p - 0.5
        r = q * q
        x = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
            (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
             ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)

    # One Halley refinement step.
    e = _Phi(x) - p
    u = e * math.sqrt(2.0 * math.pi) * math.exp(0.5 * x * x)
    x = x - u / (1.0 + 0.5 * x * u)
    return x


def beta_c(c: float) -> float:
    """Huber scale consistency constant beta = E[psi_c(Z)^2], Z ~ N(0,1),
    psi_c(u) = clip(u, -c, c). Makes the Huber scale estimator unbiased at the
    Gaussian. Closed form:

        beta = (2 Phi(c) - 1) - 2 c phi(c) + 2 c^2 (1 - Phi(c))
    """
    return (2.0 * _Phi(c) - 1.0) - 2.0 * c * _phi(c) + 2.0 * c * c * (1.0 - _Phi(c))


# --------------------------------------------------------------------------- #
# gamma* optimization
# --------------------------------------------------------------------------- #
def _clip_err(gamma: float, family: str) -> float:
    if family == "gaussian":
        return 2.0 * ((1.0 + gamma * gamma) * (1.0 - _Phi(gamma)) - gamma * _phi(gamma))
    if family == "laplace":
        return 2.0 * _LAPLACE_BL * _LAPLACE_BL * math.exp(-gamma / _LAPLACE_BL)
    raise ValueError(f"Unknown family '{family}'. Choose from {_FAMILIES}.")


def _round_err(gamma: float, b: int) -> float:
    # Level count 2^b (see module docstring), NOT 2^b - 1.
    return (2.0 * gamma / (2 ** b)) ** 2 / 12.0


def _total_err(gamma: float, b: int, family: str) -> float:
    return _clip_err(gamma, family) + _round_err(gamma, b)


def compute_gamma_star(b: int, family: str = "gaussian") -> float:
    """Optimal clip constant gamma* (in sigma units) for b-bit symmetric uniform
    quantization of a `family` distribution. Golden-section search on the
    unimodal clip+round objective (doc 3.1). Dependency-free (stdlib math only).
    """
    if b <= 0:
        raise ValueError(f"b must be a positive int, got {b}")
    if family not in _FAMILIES:
        raise ValueError(f"Unknown family '{family}'. Choose from {_FAMILIES}.")

    lo, hi = 0.05, 14.0
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0  # golden ratio conjugate
    c = hi - inv_phi * (hi - lo)
    d = lo + inv_phi * (hi - lo)
    fc = _total_err(c, b, family)
    fd = _total_err(d, b, family)
    for _ in range(200):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - inv_phi * (hi - lo)
            fc = _total_err(c, b, family)
        else:
            lo, c, fc = c, d, fd
            d = lo + inv_phi * (hi - lo)
            fd = _total_err(d, b, family)
        if hi - lo < 1e-10:
            break
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- #
# Precomputed table (computed at import; verified against the reference below).
# --------------------------------------------------------------------------- #
GAMMA_STAR = {
    family: {b: compute_gamma_star(b, family) for b in _BITS}
    for family in _FAMILIES
}

# Reference values (sigma units), checked into source per the task. These are the
# golden-section solutions of the clip+round objective with level count 2^b.
#   Gaussian: b4 = 2.5591 (~2.55), b8 = 3.9240 (~3.92).
#   Laplace : sigma-units; x sqrt(2) -> doc's b_L column {2.83, 3.89, 5.03}.
GAMMA_STAR_REFERENCE = {
    "gaussian": {2: 1.7106, 3: 2.1516, 4: 2.5591, 6: 3.2869, 8: 3.9240},
    "laplace":  {2: 2.0016, 3: 2.7558, 4: 3.5558, 6: 5.2419, 8: 6.9981},
}

# Half-normal Gaussian-consistency constant for the percentile estimator at the
# default p = 0.999: Phi^{-1}((1 + 0.999) / 2) = Phi^{-1}(0.9995).
PHI_INV_999 = phi_inv(0.9995)


def gamma_star(b: int, family: str = "gaussian") -> float:
    """Table lookup with fallback to on-the-fly computation."""
    fam = GAMMA_STAR.get(family)
    if fam is not None and b in fam:
        return fam[b]
    return compute_gamma_star(b, family)
