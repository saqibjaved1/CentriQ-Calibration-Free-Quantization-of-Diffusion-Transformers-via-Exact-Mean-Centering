"""
Loader for the published Kronecker W4 weight fits (RoDS-A Step 5).

`scripts/refit_kron_w4.py` (Task K / K2) writes `kron_w4_fits.pt` + a sidecar
`metadata.json` carrying the sha256 of the `.pt`. Both the Step-2 fixture packer
(`scripts/pack_fixtures.py`) and the Step-5 int4 install path consume EXACTLY those
fits (re-fitting on a different device/order would perturb the BCD rounding), so the
reconstruction + sha verification live here once and are imported by both.

The fit record is the field-by-field content of a `PreparedWeight`; `prepared_from_fit`
rebuilds the dataclass so `to_kernel_operands` / `pack_weights` / `dequantize_prepared`
operate on it unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, Optional

import torch

from rods_a.prepare_weight import PreparedWeight


def prepared_from_fit(rec: dict) -> PreparedWeight:
    """Reconstruct a PreparedWeight from a K2 saved fit record (kronecker mode)."""
    return PreparedWeight(
        W_q=rec["W_q"], meta=rec["meta"], Cin_orig=int(rec["Cin_orig"]),
        Cin_pad=int(rec["Cin_pad"]), rotated=bool(rec["rotated"]),
        out_features=int(rec["out_features"]), rho_median=float(rec["rho_median"]),
        incoherence_W=float(rec.get("incoherence_W", 0.0)),
        incoherence_Wt=float(rec.get("incoherence_Wt", 0.0)),
        bias=rec.get("bias"), colsum=rec.get("colsum"),
    )


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_kron_fits(
    fits_path: str,
    *,
    expected_sha: Optional[str] = None,
    meta_path: Optional[str] = None,
    expected_rotation: str = "kronecker",
) -> Dict[str, dict]:
    """Load + verify a pinned W4 fits artifact, returning {layer_name: fit_record}.

    Verifies (hard): the `.pt` sha256 against `expected_sha` (and, when present, the
    sidecar metadata.json's recorded sha256), and that the fits' rotation tag matches
    `expected_rotation` (default "kronecker" -- existing callers unchanged; the
    V2 pad-int4 deployment passes "pad" with the pad artifact). Mismatch raises --
    the pinned artifact is part of the deployment contract.
    """
    if not os.path.exists(fits_path):
        raise FileNotFoundError(
            f"W4 fits not found at {fits_path}. Run scripts/refit_kron_w4.py "
            f"(kron) or scripts/make_pad_w4_fits.py (pad) first."
        )
    actual_sha = _sha256_file(fits_path)
    if meta_path is None:
        meta_path = os.path.join(os.path.dirname(fits_path), "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        recorded = meta.get("sha256")
        if recorded is not None and recorded != actual_sha:
            raise RuntimeError(
                f"kron fits sha256 mismatch vs sidecar metadata: file {actual_sha} "
                f"!= metadata {recorded} ({meta_path})."
            )
        if str(meta.get("rotation")) != expected_rotation:
            raise RuntimeError(
                f"fits metadata rotation={meta.get('rotation')!r} != "
                f"{expected_rotation!r}."
            )
    if expected_sha is not None and actual_sha != expected_sha:
        raise RuntimeError(
            f"kron fits sha256 {actual_sha} != pinned {expected_sha}. Refusing to "
            f"proceed -- this is the audited weight artifact."
        )

    blob = torch.load(fits_path, map_location="cpu")
    if str(blob.get("rotation")) != expected_rotation:
        raise RuntimeError(
            f"fits blob rotation={blob.get('rotation')!r} != {expected_rotation!r}."
        )
    fits = blob["fits"]
    return fits
