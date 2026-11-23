"""Auto-generated folder names that encode the quantization configuration.

Pattern:  {Method}_{nbits}b_gs{group_size}[_<slug>...]

A slug is appended for every recognised hyperparameter whose value differs
from its documented default. Two runs with the same method/bits/group_size
that share defaults produce the same folder; once any knob is overridden,
its short-key slug is added so runs don't collide.

Short-key legend (also documented in outputs/quant_compare/README.md):

    p     lp_norm / hqq_p     (default 0.7)
    hd    huber_delta         (default 1.0)
    px    prox_op             (default lp)  # rods inner prox operator
    so    sinkhorn_order      (default 16)
    bm    beta_mult           (default 1.0)
    qs    quant_start         (default 0)
    ko    K_outer             (default 3)   # rods outer loop
    ki    K_irls              (default 5)   # rods IRLS inner loop
    kw    K_warm              (default 1)   # rods Sinkhorn warm-start
    r     rank                (default 32)  # svdquant low-rank branch rank
    to    tol_outer           (default 1e-5) # rods outer rel-improvement threshold
    ti    tol_irls            (default 1e-5) # rods IRLS rel-change threshold
    km    K_min               (default 3)   # rods min outer iters before early stop
    CHtoL mixed precision     (e.g. C8to4 when nbits_high=8, nbits=4)

Floats are formatted to drop trailing zeros so `huber_delta=0.5` → `hd0.5`
(not `hd0.50`).
"""

from typing import Any, Mapping

DEFAULTS: dict[str, Any] = {
    "lp_norm": 0.7,
    "hqq_p": 0.7,
    "huber_delta": 0.0,
    "sinkhorn_order": 16,
    "beta_mult": 1.0,
    "quant_start": 0,
    "K_outer": 3,
    "K_irls": 5,
    "K_warm": 1,
    "tol_outer": 1e-3,
    "tol_irls": 1e-3,
    "K_min": 3,
    "prox_op": "lp",           # rods inner prox operator
    "tau_mode": "fixed_one",   # BPDQ per-layer Bregman prox-weight mode
    "rank": 32,                # svdquant low-rank branch rank
}

SHORT_KEYS: dict[str, str] = {
    "lp_norm": "p",
    "hqq_p": "p",
    "huber_delta": "hd",
    "sinkhorn_order": "so",
    "beta_mult": "bm",
    "quant_start": "qs",
    "K_outer": "ko",
    "K_irls": "ki",
    "K_warm": "kw",
    "tol_outer": "to",
    "tol_irls": "ti",
    "K_min": "km",
    "prox_op": "px",
    "tau_mode": "tau",
    "rank": "r",
}

# Slug-ordering: stable across runs. Anything not listed sorts alphabetically.
_ORDER = ["p", "hd", "px", "so", "bm", "qs", "ko", "ki", "kw", "to", "ti", "km", "tau", "r"]


def _fmt(val: Any) -> str:
    if isinstance(val, float):
        s = f"{val:g}"
        return s if "." in s or "e" in s else f"{s}.0"
    return str(val)


def folder_name(
    method_label: str,
    nbits: int,
    group_size: int,
    params: Mapping[str, Any] | None = None,
    nbits_high: int | None = None,
) -> str:
    parts = [method_label, f"{nbits}b", f"gs{group_size}"]
    if nbits_high is not None:
        parts.append(f"C{nbits_high}to{nbits}")

    seen: set[str] = set()
    slugs: list[tuple[int, str, str]] = []
    for key, val in (params or {}).items():
        if key not in SHORT_KEYS:
            continue
        short = SHORT_KEYS[key]
        if short in seen:
            continue
        if val == DEFAULTS.get(key):
            continue
        seen.add(short)
        order_idx = _ORDER.index(short) if short in _ORDER else len(_ORDER)
        slugs.append((order_idx, short, f"{short}{_fmt(val)}"))

    slugs.sort(key=lambda t: (t[0], t[1]))
    parts.extend(s for _, _, s in slugs)
    return "_".join(parts)
