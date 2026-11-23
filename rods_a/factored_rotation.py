"""Width-keyed Kronecker-factorized rotation for the composed W4A4 path.

Builds the orthonormal rotation ``R = (H_p (x) H_q) / sqrt(d')`` for an explicit
factorization ``(p, q)`` of the (possibly padded) width ``d' = p * q``, instead
of the width-derived ``factor_kron`` choice hardwired in ``rods_a.fwht``. The
resulting ``(rotate, unrotate)`` callables enter the locked quant path through
the existing ``rotate_override`` / ``unrotate_override`` hooks of
``quantize_act`` and a ``rotate_fn`` injection in ``prepare_weight`` —
``rods_a.fwht`` itself is untouched.

Unlike the pad/kronecker schemes, the composed matrix is NOT symmetric in
general (the vendored non-Sylvester factors are asymmetric), so the inverse is
``X @ R^T`` (the ``dense_haar`` non-involution convention), not a re-apply.

The matrix is built once in fp32 from the exact int64 factors and cached per
(spec, device, dtype) — the factorization is part of the key so two candidates
for the same width never collide within one process.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor

from core.hadamard import build_hadamard, order_of

from rods_a import profiling as _profiling

# Direct-function imports: `from . import fwht` would bind the FWHT *function*
# re-exported by rods_a/__init__, not the submodule.
from rods_a.fwht import dense_rotate as _pad_dense_rotate
from rods_a.fwht import dense_unrotate as _pad_dense_unrotate
from rods_a.fwht import rotate as _pad_fwht_rotate
from rods_a.fwht import unrotate as _pad_fwht_unrotate

__all__ = [
    "FactoredSpec",
    "resolve_factored",
    "factored_rotators",
    "pad_rotators",
]


@dataclass(frozen=True)
class FactoredSpec:
    """One layer-width's factored rotation: rotate at width ``pad_to = p * q``.

    A factor is an int order or a variant token (e.g. "p24" = Paley-I order 24,
    a different equivalence class than the composed 24 = 12 (x) 2 — needed where
    Kronecker re-association would otherwise collapse candidates)."""

    width: int                 # real Cin of the layer (d)
    factors: Tuple           # (p, q): H_p on the slow axis, H_q on the fast axis
    pad_to: int                # rotated width d' = p * q (>= width; == width if native)


def resolve_factored(
    width: int,
    factor_map: Optional[Mapping[int, Sequence[int]]],
    pad_to_map: Optional[Mapping[int, int]] = None,
) -> Optional[FactoredSpec]:
    """Resolve a layer width against the width-keyed factorization config.

    Returns None when `width` has no entry in `factor_map` — the caller falls
    back to the pad scheme for that layer. When both maps carry the width, the
    explicit ``pad_to`` must equal ``p * q`` (it is redundant but documents the
    padding in the config).
    """
    if factor_map is None:
        return None
    factors = factor_map.get(width)
    if factors is None:
        return None
    if len(factors) != 2:
        raise ValueError(
            f"rotation_factorization[{width}] must be a (p, q) pair, got {list(factors)!r}"
        )
    # Normalize: numeric strings (a YAML artifact) become ints; variant tokens
    # like "p24" stay strings and are validated by order_of.
    p, q = (int(f) if isinstance(f, str) and f.isdigit() else f for f in factors)
    d_rot = order_of(p) * order_of(q)   # validates int orders / variant tokens
    if d_rot < width:
        raise ValueError(
            f"rotation_factorization[{width}] = ({p}, {q}) rotates at {d_rot} < width {width}"
        )
    if pad_to_map is not None and width in pad_to_map:
        pad_to = int(pad_to_map[width])
        if pad_to != d_rot:
            raise ValueError(
                f"rotation_pad_to[{width}] = {pad_to} != p*q = {d_rot} for factors ({p}, {q})"
            )
    return FactoredSpec(width=width, factors=(p, q), pad_to=d_rot)


_FACTORED_CACHE: dict = {}


def _orthonormal_factored(spec: FactoredSpec, device, dtype, transpose: bool) -> Tensor:
    """Cached dense ``(H_p (x) H_q)/sqrt(d')`` (or its transpose), d' x d'.

    Built once in fp32 from the exact int64 factors (same float op order as
    ``fwht._orthonormal_kron``, so the (36, 32) spec reproduces the kronecker
    scheme's matrix bit-for-bit), then cast/moved per (device, dtype).
    """
    key = (spec, str(device), dtype, transpose)
    H = _FACTORED_CACHE.get(key)
    if H is None:
        base_key = (spec, "cpu-fp32-base")
        base = _FACTORED_CACHE.get(base_key)
        if base is None:
            p, q = spec.factors
            Hp = build_hadamard(p).to(torch.float32)
            Hq = build_hadamard(q).to(torch.float32)
            base = torch.kron(Hp, Hq) / math.sqrt(spec.pad_to)
            _FACTORED_CACHE[base_key] = base
        H = base.t().contiguous() if transpose else base
        H = H.to(dtype)
        if device is not None:
            H = H.to(device)
        _FACTORED_CACHE[key] = H
    return H


def factored_rotators(spec: FactoredSpec) -> Tuple[Callable, Callable]:
    """(rotate, unrotate) callables for `spec`, cache-backed dense GEMMs.

    rotate(X):  zero-pad the last axis width -> pad_to, then ``X @ R``.
    unrotate(Xr, orig_dim=None): ``Xr @ R^T``, truncate back to `spec.width`
      (`orig_dim`, if given, must equal it — dispatch-signature parity with
      the fwht unrotates).
    """
    width, pad_to = spec.width, spec.pad_to

    def rot(X: Tensor) -> Tensor:
        if X.shape[-1] != width:
            raise ValueError(f"factored rotate expects width {width}, got {X.shape[-1]}")
        if pad_to != width:
            # events-only sub-bracket: the pad share of the caller's `rotate`
            # stage (same name as the fwht pad bracket -> one of-which column).
            with _profiling.section("rotate_pad", events_only=True):
                X = torch.nn.functional.pad(X, (0, pad_to - width))
        H = _orthonormal_factored(spec, X.device, X.dtype, transpose=False)
        return X @ H

    def unrot(Xr: Tensor, orig_dim: Optional[int] = None) -> Tensor:
        if orig_dim is not None and orig_dim != width:
            raise ValueError(
                f"factored unrotate orig_dim {orig_dim} != spec width {width}"
            )
        if Xr.shape[-1] != pad_to:
            raise ValueError(f"factored unrotate expects width {pad_to}, got {Xr.shape[-1]}")
        Ht = _orthonormal_factored(spec, Xr.device, Xr.dtype, transpose=True)
        return (Xr @ Ht)[..., :width]

    return rot, unrot


def pad_rotators(impl: str = "dense") -> Tuple[Callable, Callable]:
    """The pad-scheme (rotate, unrotate) pair — the unmapped-width fallback.

    Bitwise the existing pad scheme: "dense" is the hot-path GEMM pair, "fwht"
    the butterfly pair used by the weight side in `prepare_weight`.
    """
    if impl == "dense":
        return _pad_dense_rotate, _pad_dense_unrotate
    if impl == "fwht":
        return _pad_fwht_rotate, _pad_fwht_unrotate
    raise ValueError(f"impl must be 'dense' or 'fwht', got {impl!r}")
