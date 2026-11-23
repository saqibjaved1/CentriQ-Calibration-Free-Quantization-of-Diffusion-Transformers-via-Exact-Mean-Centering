"""
Unit checks for the plain NF4 weight-only fake-quant baseline
(scripts/fid_eval.quantize_nf4) added to freeze the D1 spine.

NF4 here is the standard QLoRA 16-level NormalFloat codebook applied per-group
with an absmax scale (FP scales, no Sinkhorn, no double-quant). These tests are
pure-PyTorch and CPU-only; they run via RunAI (launch/test_nf4_quant.sh) since the
front node has no project venv.

Verifies:
  1. Codebook is the 16-value NF4 set: sorted/monotonic, endpoints +/-1, contains 0.
  2. The absmax->nearest-codebook mapping is monotonic (x up => index non-decreasing).
  3. Round-trip SQNR on a Gaussian tile beats RTN INT4 at the same group_size
     (NF4 spends levels where Gaussian weight mass is, so it should win).
  4. quantize_nf4() honors ignore_layer_names and leaves dtype/shape intact.
"""

import sys
import os

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from core.sinq.dual_shift import NF4_CODEBOOK
from baselines.baseline_methods import quantize_nf4, quantize_rtn


def sqnr_db(ref: torch.Tensor, approx: torch.Tensor) -> float:
    """Signal-to-quantization-noise ratio in dB (higher = better)."""
    sig = (ref ** 2).mean()
    noise = ((ref - approx) ** 2).mean().clamp(min=1e-30)
    return (10.0 * torch.log10(sig / noise)).item()


# ---------------------------------------------------------------------- #
# 1. Codebook properties
# ---------------------------------------------------------------------- #

def test_codebook_shape_and_endpoints():
    cb = NF4_CODEBOOK
    assert cb.numel() == 16, "NF4 codebook must have exactly 16 levels"
    assert torch.isclose(cb.min(), torch.tensor(-1.0))
    assert torch.isclose(cb.max(), torch.tensor(1.0))
    # contains an exact zero level
    assert torch.any(cb == 0.0), "NF4 codebook must contain 0.0"


def test_codebook_sorted_strictly_monotonic():
    cb = NF4_CODEBOOK
    diffs = cb[1:] - cb[:-1]
    assert torch.all(diffs > 0), "NF4 codebook must be strictly increasing"


# ---------------------------------------------------------------------- #
# 2. Monotonic mapping
# ---------------------------------------------------------------------- #

def test_mapping_monotonic():
    cb = NF4_CODEBOOK
    # Normalised inputs spanning [-1, 1]; nearest-codebook index must be
    # non-decreasing as the input increases.
    x = torch.linspace(-1.0, 1.0, 4096)
    idx = (x.unsqueeze(-1) - cb).abs().argmin(dim=-1)
    assert torch.all(idx[1:] - idx[:-1] >= 0), "mapping must be monotonic in x"
    # endpoints snap to the extreme codes
    assert idx[0].item() == 0
    assert idx[-1].item() == cb.numel() - 1


# ---------------------------------------------------------------------- #
# 3. Round-trip SQNR vs RTN INT4 on a Gaussian tile
# ---------------------------------------------------------------------- #

@pytest.mark.parametrize("group_size", [64, 128])
def test_nf4_beats_rtn_int4_sqnr_gaussian(group_size):
    torch.manual_seed(0)
    out_features, in_features = 256, 1024
    W = torch.randn(out_features, in_features)  # standard normal weights

    lin_nf4 = nn.Linear(in_features, out_features, bias=False)
    lin_rtn = nn.Linear(in_features, out_features, bias=False)
    lin_nf4.weight.data.copy_(W)
    lin_rtn.weight.data.copy_(W)

    m_nf4 = nn.Sequential(lin_nf4)
    m_rtn = nn.Sequential(lin_rtn)

    n_nf4 = quantize_nf4(m_nf4, group_size=group_size)
    n_rtn = quantize_rtn(m_rtn, nbits=4, group_size=group_size)
    assert n_nf4 == 1 and n_rtn == 1

    s_nf4 = sqnr_db(W, lin_nf4.weight.data.float())
    s_rtn = sqnr_db(W, lin_rtn.weight.data.float())
    print(f"\n[gs={group_size}] SQNR  NF4={s_nf4:.2f} dB  RTN-INT4={s_rtn:.2f} dB  "
          f"(delta={s_nf4 - s_rtn:+.2f} dB)")
    assert s_nf4 > s_rtn, "NF4 should beat RTN INT4 on Gaussian weights"


# ---------------------------------------------------------------------- #
# 4. ignore_layer_names + dtype/shape preservation
# ---------------------------------------------------------------------- #

def test_ignore_layers_and_invariants():
    torch.manual_seed(1)
    model = nn.Module()
    model.keep = nn.Linear(512, 128, bias=False)   # leaf name "keep"
    model.proj_out = nn.Linear(512, 128, bias=False)
    W_keep_before = model.keep.weight.data.clone()
    W_proj_before = model.proj_out.weight.data.clone()
    orig_dtype = model.proj_out.weight.dtype

    n = quantize_nf4(model, group_size=64, ignore_layer_names=["keep"])
    assert n == 1, "only the non-ignored Linear should be quantized"
    # ignored layer untouched
    assert torch.equal(model.keep.weight.data, W_keep_before)
    # quantized layer changed but same shape/dtype
    assert not torch.equal(model.proj_out.weight.data, W_proj_before)
    assert model.proj_out.weight.shape == W_proj_before.shape
    assert model.proj_out.weight.dtype == orig_dtype


def test_nf4_reduces_error_vs_unquantized():
    # Dequantized weights must stay close to the originals (sanity floor).
    torch.manual_seed(2)
    lin = nn.Linear(1024, 256, bias=False)
    W = lin.weight.data.clone()
    quantize_nf4(nn.Sequential(lin), group_size=64)
    assert sqnr_db(W, lin.weight.data.float()) > 15.0  # comfortable INT4-class floor
