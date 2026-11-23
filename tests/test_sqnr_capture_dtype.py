"""
SQNR capture-dtype tests (the FLUX composed-SQNR enabler).

The per-layer SQNR capture keeps a resident reference copy of the FP weight;
fp32 (default) is the legacy bit-identical path, fp16 halves the copy. Pins:

(a) default capture is fp32 and sqnr_db() behaves as before;
(b) fp16 capture agrees with fp32 within 0.2 dB on a synthetic layer (the
    estimator-bias bound the sweep relies on);
(c) the install-time amax guard rejects fp16 capture of weights outside fp16
    range (the bf16->fp16 exactness precondition).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from rods_a.prepare_weight import prepare_weight
from rods_a.wrapper import ActQuantConfig, RoDSAW4A4Linear


def _make(capture_dtype, W, pack):
    return RoDSAW4A4Linear(pack, ActQuantConfig(), fp_weight=W,
                           matmul_dtype=torch.float32,
                           capture_dtype=capture_dtype)


def test_capture_dtype_default_fp32():
    torch.manual_seed(0)
    out_f, Cin = 32, 128
    W = torch.randn(out_f, Cin, dtype=torch.float32) * 0.05
    pack = prepare_weight(W, bits=4, group_size=64, device="cpu",
                          compute_dtype=torch.float32, bitpack=False,
                          rotation="pad", strict_rho=False)
    lyr = RoDSAW4A4Linear(pack, ActQuantConfig(), fp_weight=W,
                          matmul_dtype=torch.float32)
    assert lyr.fp_weight.dtype == torch.float32
    assert lyr.sqnr_db() is None                     # no forward yet
    lyr(torch.randn(64, Cin, dtype=torch.float32))
    v = lyr.sqnr_db()
    assert v is not None and torch.isfinite(torch.tensor(v))


def test_capture_fp16_close_to_fp32():
    torch.manual_seed(1)
    out_f, Cin = 32, 128
    W = torch.randn(out_f, Cin, dtype=torch.float32) * 0.05
    pack = prepare_weight(W, bits=4, group_size=64, device="cpu",
                          compute_dtype=torch.float32, bitpack=False,
                          rotation="pad", strict_rho=False)
    x = torch.randn(256, Cin, dtype=torch.float32)
    lyr32 = _make(torch.float32, W, pack)
    lyr16 = _make(torch.float16, W, pack)
    assert lyr16.fp_weight.dtype == torch.float16
    lyr32(x)
    lyr16(x)
    assert abs(lyr32.sqnr_db() - lyr16.sqnr_db()) < 0.2


def test_capture_fp16_amax_guard():
    torch.manual_seed(2)
    out_f, Cin = 32, 128
    W = torch.randn(out_f, Cin, dtype=torch.float32) * 0.05
    W_hot = W.clone()
    W_hot[0, 0] = 70000.0                            # outside fp16 range
    pack = prepare_weight(W_hot, bits=4, group_size=64, device="cpu",
                          compute_dtype=torch.float32, bitpack=False,
                          rotation="pad", strict_rho=False)
    with pytest.raises(ValueError):
        _make(torch.float16, W_hot, pack)
    # fp32 capture of the same weight is fine (guard is fp16-only).
    _make(torch.float32, W_hot, pack)
