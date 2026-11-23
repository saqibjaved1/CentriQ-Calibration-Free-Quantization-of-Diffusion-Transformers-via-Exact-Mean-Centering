"""V2 glue — load_kron_fits expected_rotation generalization (CPU).

The pinned-kron default must stay bit-identical (kron blobs load with NO new
arguments); a pad artifact loads ONLY when expected_rotation='pad'; any
basis/artifact mismatch raises.
"""

import hashlib
import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rods_a.kron_fits import load_kron_fits


def _write_blob(tmp_path, rotation, with_meta=True):
    blob = {"fits": {"layer0": {"W_q": torch.zeros(2, 2)}}, "rotation": rotation}
    p = os.path.join(str(tmp_path), f"{rotation}_fits.pt")
    torch.save(blob, p)
    sha = hashlib.sha256(open(p, "rb").read()).hexdigest()
    if with_meta:
        with open(os.path.join(str(tmp_path), "metadata.json"), "w") as f:
            json.dump({"sha256": sha, "rotation": rotation}, f)
    return p, sha


def test_kron_default_unchanged(tmp_path):
    p, sha = _write_blob(tmp_path, "kronecker")
    fits = load_kron_fits(p, expected_sha=sha)          # no new args: old call shape
    assert "layer0" in fits


def test_pad_blob_loads_with_expected_rotation(tmp_path):
    p, sha = _write_blob(tmp_path, "pad")
    fits = load_kron_fits(p, expected_sha=sha, expected_rotation="pad")
    assert "layer0" in fits


def test_pad_blob_rejected_by_kron_default(tmp_path):
    p, sha = _write_blob(tmp_path, "pad")
    with pytest.raises(RuntimeError, match="rotation"):
        load_kron_fits(p, expected_sha=sha)


def test_kron_blob_rejected_when_pad_expected(tmp_path):
    p, sha = _write_blob(tmp_path, "kronecker")
    with pytest.raises(RuntimeError, match="rotation"):
        load_kron_fits(p, expected_sha=sha, expected_rotation="pad")


def test_sha_mismatch_still_raises(tmp_path):
    p, _ = _write_blob(tmp_path, "pad", with_meta=False)
    with pytest.raises(RuntimeError, match="sha256"):
        load_kron_fits(p, expected_sha="0" * 64, expected_rotation="pad")
