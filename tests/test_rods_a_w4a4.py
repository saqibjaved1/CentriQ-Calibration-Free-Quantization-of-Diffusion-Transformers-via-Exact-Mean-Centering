"""
RoDS-A Phase 2 unit tests: composed W4A4 fake-quant path.

(i)   folded == unfolded: the dense matmul x̃_hat @ W̃_hatᵀ (RoDSAW4A4Linear)
      equals the grouped/epilogue form a_i · Σ_g s_{j,g} Σ_{k∈g} Q^X_ik (Q^W-z)_jk
      — verifies the placement of a_i (per-token), s_{j,g} / z_{j,g}
      (per-(output,group)), with t absorbed and no online scale on axis k.
(ii)  rotation invariance: X Wᵀ == rotate(X) rotate(W)ᵀ pre-quant, < 1e-4.
(iii) re-fit not transferred: RoDS codes fit on W̃ differ from the codes fit on W.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from rods_a.constants import gamma_star
from rods_a.fwht import rotate, rotate_for
from rods_a.prepare_weight import prepare_weight, dequantize_prepared
from rods_a.quantize_act import quantize_act
from rods_a.wrapper import ActQuantConfig, RoDSAW4A4Linear
from rods.quantizer import quantize_rods


def _codes_sz(pack):
    """Unpack (Q^W, s, z) of a bitpack=False prepared weight to (out, Cin_pad)
    integer codes and (out, n_tpr) scales/zeros."""
    out_f = pack.out_features
    gs = int(pack.meta["group_size"])
    n_tpr = pack.Cin_pad // gs
    Q = pack.W_q.float().reshape(out_f, n_tpr, gs).reshape(out_f, pack.Cin_pad)
    s = pack.meta["scale"].float().reshape(out_f, n_tpr)
    z = pack.meta["zero"].float().reshape(out_f, n_tpr)
    return Q, s, z, gs, n_tpr


def test_folded_equals_unfolded():
    torch.manual_seed(0)
    out_f, Cin = 96, 256  # pow2 Cin -> no rotation padding (clean indexing)
    cfg = ActQuantConfig(act_bits=4, scale_estimator="percentile", rotate_act=True)
    W = torch.randn(out_f, Cin, dtype=torch.float32) * 0.05
    bias = torch.randn(out_f, dtype=torch.float32)

    pack = prepare_weight(
        W, bias=bias, bits=4, group_size=64, device="cpu",
        compute_dtype=torch.float32, bitpack=False, strict_rho=False,
    )
    layer = RoDSAW4A4Linear(pack, cfg, matmul_dtype=torch.float32)

    x = torch.randn(40, Cin, dtype=torch.float32)
    y_unfolded = layer(x)  # dense matmul path

    # Folded/grouped reconstruction from primitives. Mirror the wrapper's rotation
    # (dense, fp32) so the integer codes match exactly.
    x_hat_rot, diag = quantize_act(
        x, b=4, gamma_star=cfg.gamma_star, estimator="percentile",
        rotate_act=True, percentile_p=cfg.percentile_p, return_diag=True,
        return_rotated=True, rotate_impl="dense", rotate_dtype=torch.float32,
    )
    a = diag["a"]               # (T, 1)
    Xq = diag["Xq_codes"]       # (T, Cin_pad) integer codes
    Q, s, z, gs, n_tpr = _codes_sz(pack)

    T = x.shape[0]
    y_folded = torch.zeros(T, out_f, dtype=torch.float32)
    for g in range(n_tpr):
        cols = slice(g * gs, (g + 1) * gs)
        Xqg = Xq[:, cols]                          # (T, gs)
        Qg = Q[:, cols]                            # (out_f, gs)
        # Σ_{k∈g} Q^X_ik (Q^W_jk - z_jg) = Xqg @ Qg^T - z_jg * Σ_k Xqg
        acc = Xqg @ Qg.t()                         # (T, out_f)
        acc = acc - Xqg.sum(dim=1, keepdim=True) * z[:, g].unsqueeze(0)
        y_folded = y_folded + acc * s[:, g].unsqueeze(0)  # per-(out,group) scale
    y_folded = a * y_folded + bias                  # per-token scale + bias

    err = (y_unfolded - y_folded).abs().max().item()
    assert err < 1e-3, f"folded vs unfolded mismatch: max|Δ|={err:.2e}"


def test_rotation_invariance_prequant():
    torch.manual_seed(1)
    out_f, Cin = 64, 1152  # non-pow2 -> exercises pad-to-2048
    X = torch.randn(33, Cin, dtype=torch.float32)
    W = torch.randn(out_f, Cin, dtype=torch.float32)

    y = X @ W.t()
    y_rot = rotate(X) @ rotate(W).t()  # padded width cancels: X̃ W̃ᵀ == X Wᵀ
    err = (y - y_rot).abs().max().item()
    assert err < 1e-4, f"rotation not invariant pre-quant: max|Δ|={err:.2e}"


def test_refit_not_transferred():
    torch.manual_seed(2)
    out_f, Cin = 96, 256  # pow2 so W and W̃ fits share shape for code comparison
    W = torch.randn(out_f, Cin, dtype=torch.float32) * 0.05
    Wt = rotate(W)
    assert (W - Wt).abs().max().item() > 1e-3, "rotation must change the weight"

    kw = dict(nbits=4, group_size=64, device="cpu",
              compute_dtype=torch.float32, bitpack=False, fit_t=False)
    Wq_W, _ = quantize_rods(W, **kw)
    Wq_Wt, _ = quantize_rods(Wt, **kw)

    # Codes fit on W̃ must differ from codes fit on W (a transferred fit would
    # reuse the W codes). They share shape here, so compare directly.
    assert Wq_W.shape == Wq_Wt.shape
    frac_diff = (Wq_W != Wq_Wt).float().mean().item()
    assert frac_diff > 0.1, (
        f"W̃ fit looks transferred from W fit: only {frac_diff:.1%} codes differ"
    )


def test_center_off_is_regression():
    """Centering OFF: the wrapper output is exactly the un-corrected matmul path
    (x_hat @ W̃_hatᵀ + bias), so existing configs stay bit-identical."""
    torch.manual_seed(10)
    out_f, Cin = 96, 256
    cfg = ActQuantConfig(act_bits=4, scale_estimator="percentile", rotate_act=True,
                         center_act_mean=False)
    W = torch.randn(out_f, Cin, dtype=torch.float32) * 0.05
    bias = torch.randn(out_f, dtype=torch.float32)
    pack = prepare_weight(W, bias=bias, bits=4, group_size=64, device="cpu",
                          compute_dtype=torch.float32, bitpack=False, strict_rho=False)
    layer = RoDSAW4A4Linear(pack, cfg, center_mean=False, matmul_dtype=torch.float32)
    assert not hasattr(layer, "colsum"), "center off must not register a colsum buffer"

    x = torch.randn(40, Cin, dtype=torch.float32)
    y = layer(x)

    # Reference: the pre-centering path -- no x_bar, no correction term. Match the
    # wrapper's rotation (dense, fp32) so the comparison is bit-identical.
    x_hat, _ = quantize_act(x, b=4, gamma_star=cfg.gamma_star, estimator="percentile",
                            rotate_act=True, percentile_p=cfg.percentile_p,
                            return_diag=False, return_rotated=True, center_mean=False,
                            rotate_impl="dense", rotate_dtype=torch.float32)
    W_hat = dequantize_prepared(pack)
    y_ref = x_hat @ W_hat.t() + bias
    err = (y - y_ref).abs().max().item()
    assert err == 0.0, f"center-off path not bit-identical: max|Δ|={err:.2e}"


def test_decomposition_exact_pure_fp():
    """Decision-independent: rotate(x-x̄)@rotate(W)ᵀ + x̄*colsum(W) == x@Wᵀ.
    Non-pow2 Cin exercises the zero-padding (which must not enter x̄ or colsum)."""
    torch.manual_seed(11)
    out_f, Cin = 64, 1152  # non-pow2 -> pad to 2048
    x = torch.randn(33, Cin, dtype=torch.float32)
    W = torch.randn(out_f, Cin, dtype=torch.float32)

    x_bar = x.mean(dim=-1, keepdim=True)          # (T,1) over real Cin
    colsum = W.sum(dim=1)                          # (out_f,) over real Cin
    y = x @ W.t()
    y_dec = rotate(x - x_bar) @ rotate(W).t() + x_bar * colsum
    err = (y - y_dec).abs().max().item()
    assert err < 1e-4, f"mean-centering decomposition not exact: max|Δ|={err:.2e}"


def test_centered_rotated_dc_is_zero():
    """After per-token centering the rotated DC coordinate (index 0) is ~0 -- the
    spike that inflated the per-token scale is gone."""
    torch.manual_seed(12)
    Cin = 1152
    x = torch.randn(50, Cin, dtype=torch.float32) + 3.7  # strong non-zero mean
    xc = x - x.mean(dim=-1, keepdim=True)
    dc = rotate(xc)[:, 0].abs().max().item()
    raw_dc = rotate(x)[:, 0].abs().max().item()
    assert dc < 1e-3, f"centered DC coordinate not ~0: {dc:.2e}"
    assert raw_dc > 10.0 * (dc + 1e-9), "control: uncentered DC should be large"


def test_centered_rotated_dc_is_zero_pad_and_kron():
    """R2.5: centering kills the DC component in BOTH the pad and kronecker bases.
    Basis-agnostic: project the rotated activation onto the rotated all-ones direction
    (= the DC component, since rotation is orthonormal). H_36 has no all-ones row, so
    the projection form -- not 'index 0' -- is the correct test for the kron basis."""
    torch.manual_seed(14)
    Cin = 1152                                   # 36*32: valid for kronecker
    for scheme in ("pad", "kronecker"):
        rot = rotate_for(scheme, "fwht")
        x = torch.randn(50, Cin, dtype=torch.float32) + 3.7
        xc = x - x.mean(dim=-1, keepdim=True)
        u = rot(torch.ones(1, Cin, dtype=torch.float32))   # rotated all-ones over real Cin
        u = u / u.norm()
        dc = (rot(xc) * u).sum(dim=-1).abs().max().item()  # DC projection per token
        raw = (rot(x) * u).sum(dim=-1).abs().max().item()
        assert dc < 1e-3, f"{scheme}: centered DC projection not ~0: {dc:.2e}"
        assert raw > 10.0 * (dc + 1e-9), f"{scheme}: control uncentered DC should be large"


def test_d1_rotation_centering_invariance():
    """R4.1 PERMANENT invariance gate (rotation ON, quant OFF), centering ON and OFF,
    pad AND kron: the composed decomposition is exact in FP. With centering,
        rotate(x-x̄) @ rotate(W)ᵀ + x̄⊗colsum(W_fp) == x Wᵀ;
    without, rotate(x) @ rotate(W)ᵀ == x Wᵀ. colsum comes from prepare_weight (the real
    operand the kernel path uses), not an ad-hoc sum -- this closes the
    consistency-vs-invariance gap the validation chain had. Random + one real cell."""
    torch.manual_seed(15)
    out_f, Cin = 64, 1152                        # 36*32: valid for both bases
    cells = [(torch.randn(33, Cin) + 1.9, torch.randn(out_f, Cin) * 0.05)]
    real = _maybe_real_cell(Cin)
    if real is not None:
        cells.append(real)

    for scheme in ("pad", "kronecker"):
        rot = rotate_for(scheme, "fwht")
        for x, W in cells:
            # prepare_weight gives the exact colsum operand the path consumes (FP W).
            pack = prepare_weight(W, bits=4, group_size=64, device="cpu",
                                  compute_dtype=torch.float32, rotation=scheme,
                                  strict_rho=False)
            colsum = pack.colsum
            y = x @ W.t()
            den = y.norm().clamp_min(1e-12)
            for center in (True, False):
                if center:
                    xb = x.mean(dim=-1, keepdim=True)
                    y_dec = rot(x - xb) @ rot(W).t() + xb * colsum
                else:
                    y_dec = rot(x) @ rot(W).t()
                rel = ((y_dec - y).norm() / den).item()
                assert rel < 1e-4, (
                    f"{scheme} center={center}: D1 invariance broken, rel={rel:.2e}"
                )


def _maybe_real_cell(Cin):
    """One real (X, W) cell from the captured raw_acts dump, if present (CPU-safe;
    returns None so the test still runs in a bare checkout)."""
    import json
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "outputs", "rods_a_oracle", "raw_acts")
    mpath = os.path.join(base, "dump_manifest.json")
    if not os.path.exists(mpath):
        return None
    try:
        cells = json.load(open(mpath))["cells"]
        c = next(c for c in cells if int(c["shape"][-1]) == Cin)
        X = torch.load(os.path.join(base, c["act_file"]), map_location="cpu").to(torch.float32)
        wb = torch.load(os.path.join(base, c["weight_file"]), map_location="cpu")
        W = wb["weight"].to(torch.float32)
        return (X[:16], W)                       # 16 tokens keeps the test fast
    except Exception:
        return None


def test_folded_equals_unfolded_centered():
    """folded == unfolded WITH centering: the dense corrected matmul equals the
    grouped epilogue a_i·Σ_g s_jg Σ_k Q^X(Q^W-z) plus the x̄·colsum correction."""
    torch.manual_seed(13)
    out_f, Cin = 96, 256  # pow2 Cin -> clean group indexing
    cfg = ActQuantConfig(act_bits=4, scale_estimator="percentile", rotate_act=True,
                         center_act_mean=True)
    W = torch.randn(out_f, Cin, dtype=torch.float32) * 0.05
    bias = torch.randn(out_f, dtype=torch.float32)
    pack = prepare_weight(W, bias=bias, bits=4, group_size=64, device="cpu",
                          compute_dtype=torch.float32, bitpack=False, strict_rho=False)
    layer = RoDSAW4A4Linear(pack, cfg, center_mean=True, matmul_dtype=torch.float32)

    x = torch.randn(40, Cin, dtype=torch.float32) + 1.3  # non-zero mean
    y_unfolded = layer(x)

    x_hat_rot, diag = quantize_act(
        x, b=4, gamma_star=cfg.gamma_star, estimator="percentile",
        rotate_act=True, percentile_p=cfg.percentile_p, return_diag=True,
        return_rotated=True, center_mean=True,
        rotate_impl="dense", rotate_dtype=torch.float32,
    )
    a = diag["a"]
    Xq = diag["Xq_codes"]
    x_bar = diag["x_bar"]
    Q, s, z, gs, n_tpr = _codes_sz(pack)

    T = x.shape[0]
    y_folded = torch.zeros(T, out_f, dtype=torch.float32)
    for g in range(n_tpr):
        cols = slice(g * gs, (g + 1) * gs)
        Xqg = Xq[:, cols]
        Qg = Q[:, cols]
        acc = Xqg @ Qg.t()
        acc = acc - Xqg.sum(dim=1, keepdim=True) * z[:, g].unsqueeze(0)
        y_folded = y_folded + acc * s[:, g].unsqueeze(0)
    y_folded = a * y_folded + x_bar * pack.colsum + bias  # + mean correction

    err = (y_unfolded - y_folded).abs().max().item()
    assert err < 1e-3, f"folded vs unfolded (centered) mismatch: max|Δ|={err:.2e}"


def test_fit_t_false_drops_scale2():
    torch.manual_seed(3)
    W = torch.randn(64, 128, dtype=torch.float32) * 0.05
    _, meta_t = quantize_rods(W, nbits=4, group_size=64, device="cpu",
                                        compute_dtype=torch.float32, fit_t=True)
    _, meta_no = quantize_rods(W, nbits=4, group_size=64, device="cpu",
                                         compute_dtype=torch.float32, fit_t=False)
    assert meta_t["scale2"] is not None, "fit_t=True must keep per-column t"
    assert meta_no["scale2"] is None, "fit_t=False must drop t (scale2=None)"


def test_prepare_weight_free_t_applies_scale2():
    """A12 free-t wiring: prepare_weight(fit_t=True) keeps the per-column t and the
    cached fake-quant weight W̃_hat (dequantize_prepared) APPLIES it multiplicatively;
    fit_t=False (default) is the unchanged absorbed-t path."""
    import copy
    from core.sinq.quantizer import Quantizer
    torch.manual_seed(5)
    W = torch.randn(96, 256, dtype=torch.float32) * 0.05

    # regression: default fit_t=False -> scale2 None, t absorbed
    pack0 = prepare_weight(W, bits=4, group_size=64, device="cpu",
                           compute_dtype=torch.float32, bitpack=False, strict_rho=False)
    assert pack0.meta.get("scale2") is None

    # free-t: scale2 populated and applied in the cached W̃_hat
    packt = prepare_weight(W, bits=4, group_size=64, device="cpu",
                           compute_dtype=torch.float32, bitpack=False, strict_rho=False,
                           fit_t=True)
    s2 = packt.meta.get("scale2")
    assert s2 is not None, "fit_t=True must keep per-column t"
    assert s2.reshape(-1).numel() == packt.Cin_pad

    W_hat = dequantize_prepared(packt)
    meta_no_s2 = copy.copy(packt.meta)
    meta_no_s2["scale2"] = None
    W_hat_no_s2 = Quantizer.dequantize(packt.W_q, meta_no_s2).float()
    # dequant applies s2: W_hat == W_hat_no_s2 * s2 (broadcast over rows)
    assert torch.allclose(W_hat, W_hat_no_s2 * s2.reshape(1, -1).float(),
                          atol=1e-4, rtol=1e-3)
    # and the free-t reconstruction is genuinely different from the absorbed-t one
    assert not torch.allclose(W_hat, dequantize_prepared(pack0), atol=1e-5)

    # the fake-quant layer forward consumes the s2-applied W̃_hat without error
    cfg = ActQuantConfig(act_bits=4, scale_estimator="percentile", rotate_act=True)
    layer = RoDSAW4A4Linear(packt, cfg, matmul_dtype=torch.float32)
    y = layer(torch.randn(8, 256, dtype=torch.float32))
    assert y.shape == (8, 96) and torch.isfinite(y).all()


def test_dense_rotate_matches_fwht():
    """The fast cached-Hadamard GEMM rotate == the Python-butterfly rotate (fp32),
    and preserves the X̃ W̃ᵀ == X Wᵀ invariance — incl. the non-pow2 pad path."""
    from rods_a.fwht import rotate as fwht_rotate, dense_rotate
    torch.manual_seed(7)
    for Cin in (256, 1152):                       # pow2 and non-pow2 (pads to 2048)
        X = torch.randn(20, Cin, dtype=torch.float32)
        W = torch.randn(48, Cin, dtype=torch.float32)
        err = (dense_rotate(X) - fwht_rotate(X)).abs().max().item()
        assert err < 1e-4, f"dense vs fwht rotate mismatch (Cin={Cin}): {err:.2e}"
        inv = (dense_rotate(X) @ fwht_rotate(W).t() - X @ W.t()).abs().max().item()
        assert inv < 1e-4, f"dense-path invariance broken (Cin={Cin}): {inv:.2e}"


def test_u22_act_passthrough():
    """U2.2 knob 1 (W4A16-on-W̃): passthrough=True keeps centering+rotation but no
    quantization — X̂t equals the centered+rotated X exactly, a=1, Xq_codes=None;
    and the default (passthrough=False) path is bit-untouched."""
    torch.manual_seed(3)
    Cin = 1152
    X = torch.randn(32, Cin, dtype=torch.float32) + 0.7   # nonzero token mean
    g = gamma_star(4, "gaussian")
    for scheme in ("pad", "kronecker"):
        rot = rotate_for(scheme, "fwht")
        X_hat, diag = quantize_act(
            X, b=4, gamma_star=g, rotate_act=True, rotation=scheme,
            center_mean=True, return_rotated=True, return_diag=True,
            passthrough=True)
        x_bar = X.mean(dim=-1, keepdim=True)
        expect = rot(X - x_bar)
        assert torch.allclose(X_hat, expect, atol=1e-5), f"{scheme}: not exact FP"
        assert diag["Xq_codes"] is None
        assert torch.all(diag["a"] == 1.0)
    # default path bit-unchanged by the new kwarg's existence
    a1, _ = quantize_act(X, b=4, gamma_star=g, rotate_act=True, rotation="pad")
    a2, _ = quantize_act(X, b=4, gamma_star=g, rotate_act=True, rotation="pad",
                         passthrough=False)
    assert torch.equal(a1, a2)


def test_u22_weight_passthrough_w_hat_override():
    """U2.2 knob 2 (A4-only): w_hat_override deploys the rotated FP weight; the
    module output equals quantize_act(X)@rotate(W)ᵀ (+ exact mean rank-1)."""
    torch.manual_seed(4)
    Cin, out_f, T = 1152, 64, 16
    W = torch.randn(out_f, Cin, dtype=torch.float32)
    X = torch.randn(T, Cin, dtype=torch.float32) + 0.3
    from rods_a.prepare_weight import PreparedWeight
    for scheme in ("pad", "kronecker"):
        rot = rotate_for(scheme, "fwht")
        Wt = rot(W)
        pack = PreparedWeight(
            W_q=None, meta={"group_size": 64}, Cin_orig=Cin, Cin_pad=Wt.shape[-1],
            rotated=True, out_features=out_f, rho_median=float("nan"),
            incoherence_W=1.0, incoherence_Wt=1.0, bias=None, colsum=W.sum(dim=1))
        cfg = ActQuantConfig(rotation=scheme, center_act_mean=True)
        lin = RoDSAW4A4Linear(pack, cfg, center_mean=True,
                              matmul_dtype=torch.float32, w_hat_override=Wt)
        y = lin(X)
        x_hat, diag = quantize_act(
            X, b=4, gamma_star=cfg.gamma_star, rotate_act=True, rotation=scheme,
            center_mean=True, return_rotated=True, return_diag=True,
            rotate_impl="dense", rotate_dtype=torch.float32)
        y_ref = x_hat @ Wt.t() + diag["x_bar"] * W.sum(dim=1)
        rel = ((y - y_ref).norm() / y_ref.norm().clamp_min(1e-12)).item()
        assert rel < 1e-4, f"{scheme}: A4-only forward mismatch rel={rel:.2e}"
    # guard: codes-domain backends must reject the override
    try:
        RoDSAW4A4Linear(pack, ActQuantConfig(rotation="kronecker", semantics="kernel"),
                        matmul_dtype=torch.float32, w_hat_override=Wt)
        raise AssertionError("w_hat_override accepted under kernel semantics")
    except ValueError:
        pass


if __name__ == "__main__":
    test_folded_equals_unfolded()
    test_rotation_invariance_prequant()
    test_refit_not_transferred()
    test_center_off_is_regression()
    test_decomposition_exact_pure_fp()
    test_centered_rotated_dc_is_zero()
    test_folded_equals_unfolded_centered()
    test_fit_t_false_drops_scale2()
    test_prepare_weight_free_t_applies_scale2()
    test_dense_rotate_matches_fwht()
    test_u22_act_passthrough()
    test_u22_weight_passthrough_w_hat_override()
    print("all RoDS-A W4A4 unit tests passed")
