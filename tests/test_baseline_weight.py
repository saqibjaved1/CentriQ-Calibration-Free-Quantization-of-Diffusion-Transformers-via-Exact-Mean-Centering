"""
Unit checks for the RTN/NF4 baseline-cell plumbing (cross-family baselines task).

Covers, CPU-only (run via RunAI, launch/test_baseline_weight.sh — the front node
has no project venv):
  1. Dotted-suffix ignore matching in quantize_rtn/quantize_nf4: "attn2.to_k"
     skips ONLY the cross-attn twin; bare leaf entries behave exactly as before.
  2. target_names: quantizes exactly the named set, everything else bit-identical.
  3. RTN W3: per-group level count <= 8 (signed INT3 grid).
  4. stats_out rho = sigma_FP/Delta lands in the healthy INT4 band on a Gaussian
     tile (report-only elsewhere; the band assert here validates the formula).
  5. apply_baseline_weight_quant: selects the act-target set on a PixArt-shaped
     toy (cross-attn K/V + registry-ignored leaves untouched), enforces the
     expected count, and rejects NF4 at 3 bits (no W3 NF4 cell exists).
  6. Naive-W4A4 composition: in-place RTN weights + install_w4a4 A4-only wrapper
     (rotate off, absmax estimator, centering off) == per-token symmetric absmax
     INT4 fakequant matmul against the RTN weight, to fp32-matmul tolerance.
"""

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from baselines.baseline_methods import quantize_nf4, quantize_rtn
from rods_a.baseline_weight import apply_baseline_weight_quant


# ---------------------------------------------------------------------- #
# Toy PixArt-shaped transformer (names mirror the diffusers layout so
# select_act_targets / classify_role resolve exactly as on the real model)
# ---------------------------------------------------------------------- #

D = 128  # divisible by group_size=64


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(D, D, bias=False)
        self.to_k = nn.Linear(D, D, bias=False)
        self.to_v = nn.Linear(D, D, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(D, D, bias=False)])


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1 = _Attn()
        self.attn2 = _Attn()


class _ToyPixArt(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_Block()])
        self.proj_out = nn.Linear(D, D, bias=False)  # registry-ignored leaf


# Act-target set on the toy: attn1.{to_q,to_k,to_v,to_out.0} + attn2.{to_q,to_out.0}
TOY_TARGETS = 6


def _snapshot(model):
    return {n: m.weight.data.clone() for n, m in model.named_modules()
            if isinstance(m, nn.Linear)}


# ---------------------------------------------------------------------- #
# 1. Dotted-suffix ignore matching
# ---------------------------------------------------------------------- #

@pytest.mark.parametrize("quantize", [quantize_rtn, quantize_nf4])
def test_suffix_ignore_cross_attn_only(quantize):
    torch.manual_seed(0)
    model = _ToyPixArt()
    before = _snapshot(model)
    n = quantize(model, group_size=64,
                 ignore_layer_names=["proj_out", "attn2.to_k", "attn2.to_v"])
    assert n == 6
    after = _snapshot(model)
    for name in before:
        untouched = torch.equal(after[name], before[name])
        if name.endswith("attn2.to_k") or name.endswith("attn2.to_v") or name == "proj_out":
            assert untouched, f"{name} should be skipped"
        else:
            assert not untouched, f"{name} should be quantized (suffix must not over-match attn1)"


def test_leaf_ignore_legacy_unchanged():
    # Bare leaf entries keep the original semantics: "to_k" skips BOTH attn1
    # and attn2 K projections.
    torch.manual_seed(0)
    model = _ToyPixArt()
    before = _snapshot(model)
    n = quantize_rtn(model, nbits=4, group_size=64, ignore_layer_names=["to_k"])
    after = _snapshot(model)
    assert n == 7  # 9 Linears - 2 to_k
    for name in before:
        if name.endswith(".to_k"):
            assert torch.equal(after[name], before[name])


# ---------------------------------------------------------------------- #
# 2. target_names restriction
# ---------------------------------------------------------------------- #

@pytest.mark.parametrize("quantize", [quantize_rtn, quantize_nf4])
def test_target_names_restriction(quantize):
    torch.manual_seed(1)
    model = _ToyPixArt()
    before = _snapshot(model)
    only = "transformer_blocks.0.attn1.to_q"
    n = quantize(model, group_size=64, target_names={only},
                 ignore_layer_names=["to_q"])  # must be ignored when target_names given
    assert n == 1
    after = _snapshot(model)
    for name in before:
        if name == only:
            assert not torch.equal(after[name], before[name])
        else:
            assert torch.equal(after[name], before[name]), f"{name} must be untouched"


# ---------------------------------------------------------------------- #
# 3. RTN W3 grid
# ---------------------------------------------------------------------- #

def test_rtn_w3_levels():
    torch.manual_seed(2)
    lin = nn.Linear(64, 32, bias=False)  # one group per output row
    quantize_rtn(nn.Sequential(lin), nbits=3, group_size=64)
    W = lin.weight.data
    for r in range(W.shape[0]):
        # signed INT3 grid: at most 8 distinct levels per group
        assert torch.unique(W[r]).numel() <= 8


# ---------------------------------------------------------------------- #
# 4. rho statistic (sigma_FP / Delta)
# ---------------------------------------------------------------------- #

def test_rho_stats_band_int4():
    torch.manual_seed(3)
    lin = nn.Linear(1024, 256, bias=False)
    lin.weight.data.normal_()
    stats = {}
    quantize_rtn(nn.Sequential(lin), nbits=4, group_size=64, stats_out=stats)
    (layer_stats,) = stats.values()
    rho = layer_stats["rho_median"]
    # Gaussian gs=64: absmax ~ 2.4-2.8 sigma -> rho = qmax*sigma/absmax ~ 2.5-3
    assert 2.0 <= rho <= 4.0, f"rho_median={rho} outside the healthy INT4 band"
    assert layer_stats["n_groups"] == 256 * (1024 // 64)

    stats_nf4 = {}
    quantize_nf4(nn.Sequential(nn.Linear(1024, 256, bias=False)), group_size=64,
                 stats_out=stats_nf4)
    (s,) = stats_nf4.values()
    assert s["sigma_over_absmax_median"] > 0.0


# ---------------------------------------------------------------------- #
# 5. apply_baseline_weight_quant (act-target selection + guards)
# ---------------------------------------------------------------------- #

def test_apply_baseline_act_target_set():
    torch.manual_seed(4)
    model = _ToyPixArt()
    before = _snapshot(model)
    diag = apply_baseline_weight_quant(model, method="rtn", nbits=4, group_size=64,
                                       model_type="pixart", expected_n=TOY_TARGETS)
    assert diag["n_quantized"] == TOY_TARGETS
    assert len(diag["per_layer"]) == TOY_TARGETS
    after = _snapshot(model)
    for name in before:
        skipped = (name.endswith("attn2.to_k") or name.endswith("attn2.to_v")
                   or name == "proj_out")
        assert torch.equal(after[name], before[name]) == skipped, name


def test_apply_baseline_expected_n_guard():
    model = _ToyPixArt()
    with pytest.raises(RuntimeError, match="act-target parity"):
        apply_baseline_weight_quant(model, method="rtn", nbits=4, group_size=64,
                                    model_type="pixart", expected_n=999)


def test_nf4_rejects_w3():
    model = _ToyPixArt()
    before = _snapshot(model)
    with pytest.raises(ValueError, match="fixed 4-bit"):
        apply_baseline_weight_quant(model, method="nf4", nbits=3, group_size=64,
                                    model_type="pixart", expected_n=TOY_TARGETS)
    after = _snapshot(model)
    for name in before:  # must raise BEFORE touching any weight
        assert torch.equal(after[name], before[name])


def test_apply_baseline_rejects_unknown_method():
    with pytest.raises(ValueError, match="rtn"):
        apply_baseline_weight_quant(_ToyPixArt(), method="hqq", nbits=4,
                                    group_size=64, model_type="pixart")


# ---------------------------------------------------------------------- #
# 6. Naive-W4A4 composition micro-check
# ---------------------------------------------------------------------- #

def test_naive_w4a4_composition():
    from rods_a.install_w4a4 import install_w4a4
    from rods_a.wrapper import ActQuantConfig, RoDSAW4A4Linear

    torch.manual_seed(5)
    model = _ToyPixArt()
    apply_baseline_weight_quant(model, method="rtn", nbits=4, group_size=64,
                                model_type="pixart", expected_n=TOY_TARGETS)
    name = "transformer_blocks.0.attn1.to_q"
    W_rtn = dict(model.named_modules())[name].weight.data.clone().float()

    cfg = ActQuantConfig(act_bits=4, scale_estimator="absmax",
                         rotate_act=False, center_act_mean=False)
    install_w4a4(model, cfg, model_type="pixart", weight_bits=4, group_size=64,
                 device="cpu", compute_dtype=torch.float32,
                 matmul_dtype=torch.float32, strict_rho=False,
                 weight_quant_enabled=False, verbose=False)
    wrapper = dict(model.named_modules())[name]
    assert isinstance(wrapper, RoDSAW4A4Linear)

    X = torch.randn(4, D)
    with torch.no_grad():
        y = wrapper(X)

    # Manual naive per-token symmetric absmax INT4 fakequant + matmul.
    qmax = 7.0
    a = X.float().abs().amax(dim=-1, keepdim=True) / qmax
    Xq = (X.float() / a).round().clamp(-qmax, qmax)
    y_ref = (a * Xq) @ W_rtn.t()

    assert torch.allclose(y.float(), y_ref, rtol=1e-4, atol=1e-5), (
        f"max abs err {(y.float() - y_ref).abs().max().item():.3e}"
    )
