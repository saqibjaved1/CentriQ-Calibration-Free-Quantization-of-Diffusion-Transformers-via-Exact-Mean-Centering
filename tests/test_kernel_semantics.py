"""
Kernel-semantics oracle unit tests (RoDS-A Step 1, plan O1.4).

The oracle (rods_a.kernel_semantics) is the pure-PyTorch reference for the forked
Nunchaku INT4 W4A4 kernel. These tests pin the contract (plan O0):

(O1.2) round-trip identity: the signed remap (Q_s = Q-8, z' = z-8) reconstructs the
       SAME weight as the deployed minus-z unsigned dequant and SINQ's
       Quantizer.dequantize.
(a)    t-structure: oracle with t_g == 1 equals the pipeline with the ×t_g stage
       removed (exact).
(b)    ground-truth identity: oracle vs an fp64 matmul of the IDENTICALLY
       quantized/dequantized tensors (same codes, same scales). Integer part is
       exact; only the fp32 epilogue + fp16 output cast differ. The empirical
       tolerance is printed -- it seeds the O2.4 parity criteria.
(c)    z-correction: a real (nonzero-z) layer and a zero-z control each match their
       fp64 reference, and the two differ (z matters).
(d)    grouped-vs-flat: with s,z constant across groups the grouped form equals the
       flat form (the only regime where the proposal's pulled-out reduction holds),
       and Σ_g P[i,g] == Σ_k Xq[i,k].

CPU only, deterministic seeds.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from rods_a.kernel_semantics import (
    QZERO_4BIT,
    KernelOperands,
    dequant_weight_signed,
    oracle_matmul,
    oracle_linear,
    to_kernel_operands,
)
from rods_a.fwht import rotate_for, unrotate_for
from rods_a.pack_operands import pack_weights
from rods_a.prepare_weight import prepare_weight, dequantize_prepared
from rods_a.quantize_act import quantize_act
from rods_a.wrapper import ActQuantConfig


def _make_pack(out_f=96, Cin=256, *, bitpack=True, seed=0, scale=0.05):
    """A RoDS PreparedWeight on a pow2 Cin (no rotation padding -> clean indexing)."""
    torch.manual_seed(seed)
    W = torch.randn(out_f, Cin, dtype=torch.float32) * scale
    bias = torch.randn(out_f, dtype=torch.float32)
    return prepare_weight(
        W, bias=bias, bits=4, group_size=64, device="cpu",
        compute_dtype=torch.float32, bitpack=bitpack, strict_rho=False,
    )


def _quant_act(x, cfg, rotated):
    _, diag = quantize_act(
        x, b=cfg.act_bits, gamma_star=cfg.gamma_star, estimator=cfg.scale_estimator,
        rotate_act=rotated, percentile_p=cfg.percentile_p, return_diag=True,
        return_rotated=True, rotate_impl="dense", rotate_dtype=torch.float32,
    )
    return diag["a"], diag["Xq_codes"]


def _fp64_truth(Xq, a, ops):
    """Genuine fp64 ground truth: reconstruct the weight from the signed operands
    entirely in fp64 (note dequant_weight_signed reconstructs in fp32, to match
    Quantizer.dequantize, so it is NOT a valid fp64 reference). Y = (a⊙Xq) @ Ŵᵀ."""
    out_f, n_tpr, gs = ops.Q_s.shape
    sg = ops.s.double() * ops.t_g.double()                       # (out, G)
    z = ops.z_signed.double()                                    # (out, G)
    Qs = ops.Q_s.double()                                        # (out, G, gs)
    W = (sg.unsqueeze(-1) * (Qs - z.unsqueeze(-1))).reshape(out_f, n_tpr * gs)
    return (a.double() * Xq.double()) @ W.t()


# --------------------------------------------------------------------------- #
# O1.2 -- signed-remap round-trip identity
# --------------------------------------------------------------------------- #
def test_signed_remap_roundtrip_identity():
    for bitpack in (False, True):
        pack = _make_pack(bitpack=bitpack, seed=1)
        ops = to_kernel_operands(pack)

        # signed-domain reconstruction
        W_signed = dequant_weight_signed(ops)

        # deployed minus-z unsigned reconstruction (same object, plus-8 shifted back)
        Q = ops.Q_s + QZERO_4BIT                      # back to {0..15}
        z = ops.z_signed + QZERO_4BIT                 # back to deployed z
        out_f, n_tpr, gs = ops.Q_s.shape
        sg = (ops.s * ops.t_g).reshape(out_f, n_tpr, 1)
        W_unsigned = (sg * (Q.to(torch.float32) - z.reshape(out_f, n_tpr, 1))).reshape(out_f, n_tpr * gs)

        assert torch.equal(W_signed, W_unsigned), "signed remap != minus-z unsigned dequant"

        # and both equal SINQ's Quantizer.dequantize
        W_ref = dequantize_prepared(pack).float()
        err = (W_signed - W_ref).abs().max().item()
        assert err < 1e-5, f"signed dequant vs Quantizer.dequantize mismatch: {err:.2e}"


# --------------------------------------------------------------------------- #
# (a) -- t-structure: t_g == 1 equals the t-removed pipeline (exact)
# --------------------------------------------------------------------------- #
def test_t_structure_identity_when_ones():
    pack = _make_pack(seed=2)
    cfg = ActQuantConfig(act_bits=4, scale_estimator="percentile", rotate_act=pack.rotated)
    x = torch.randn(40, pack.Cin_orig, dtype=torch.float32)
    a, Xq = _quant_act(x, cfg, pack.rotated)
    ops = to_kernel_operands(pack)                    # t_g defaults to ones

    y_t = oracle_matmul(Xq, a, ops, apply_t=True, out_dtype=torch.float32)
    y_no_t = oracle_matmul(Xq, a, ops, apply_t=False, out_dtype=torch.float32)
    assert torch.equal(y_t, y_no_t), "t_g==1 must be identical to dropping the ×t_g stage"

    # And a non-unit t_g actually changes the result (the slot is live for grouped-t).
    out_f, n_tpr = ops.out_features, ops.n_tpr
    t_g = 1.0 + 0.1 * torch.randn(out_f, n_tpr)
    ops_t = to_kernel_operands(pack, t_g=t_g)
    y_scaled = oracle_matmul(Xq, a, ops_t, apply_t=True, out_dtype=torch.float32)
    assert not torch.allclose(y_scaled, y_t), "non-unit t_g should change the output"


# --------------------------------------------------------------------------- #
# (b) -- ground-truth identity vs fp64 matmul of identically quantized tensors
# --------------------------------------------------------------------------- #
def test_ground_truth_identity_fp64():
    pack = _make_pack(out_f=128, Cin=256, seed=3)
    cfg = ActQuantConfig(act_bits=4, scale_estimator="percentile", rotate_act=pack.rotated)
    x = torch.randn(64, pack.Cin_orig, dtype=torch.float32)
    a, Xq = _quant_act(x, cfg, pack.rotated)
    ops = to_kernel_operands(pack)

    # fp64 truth: same codes, same scales, exact accumulation.
    y_true = _fp64_truth(Xq, a, ops)                             # (T, out)

    # oracle in the contract dtypes: fp32 accumulate, fp16 output.
    y_fp16 = oracle_matmul(Xq, a, ops, out_dtype=torch.float16, accum_dtype=torch.float32)
    # and an fp32-output / fp64-accumulate variant to isolate the integer exactness.
    y_fp64 = oracle_matmul(Xq, a, ops, out_dtype=torch.float64, accum_dtype=torch.float64)

    abs_true = y_true.abs().clamp_min(1e-12)
    rel_fp16 = ((y_fp16.to(torch.float64) - y_true).abs() / abs_true)
    abs_fp16 = (y_fp16.to(torch.float64) - y_true).abs()
    abs_fp64 = (y_fp64 - y_true).abs().max().item()

    print(
        f"\n[O1.4b parity envelope] fp64-accum: max|Δ|={abs_fp64:.2e} "
        f"(integer part exact -> ~0); "
        f"fp32-accum/fp16-out: max|Δ|={abs_fp16.max().item():.3e}, "
        f"max rel={rel_fp16.max().item():.3e}, median rel={rel_fp16.median().item():.3e}"
    )

    # integer reduction is exact when accumulated in fp64.
    assert abs_fp64 < 1e-9, f"fp64-accum oracle must equal fp64 truth, got {abs_fp64:.2e}"
    # fp32-accum + fp16 cast: within fp16 relative envelope.
    assert rel_fp16.median().item() < 5e-3, f"median rel too large: {rel_fp16.median().item():.2e}"
    assert abs_fp16.max().item() < 5e-2, f"max abs too large: {abs_fp16.max().item():.2e}"


# --------------------------------------------------------------------------- #
# (c) -- z-correction: real-z and zero-z control each match fp64; they differ
# --------------------------------------------------------------------------- #
def test_z_correction_against_fp64():
    pack = _make_pack(out_f=96, Cin=256, seed=4)
    cfg = ActQuantConfig(act_bits=4, scale_estimator="percentile", rotate_act=pack.rotated)
    x = torch.randn(48, pack.Cin_orig, dtype=torch.float32)
    a, Xq = _quant_act(x, cfg, pack.rotated)

    ops = to_kernel_operands(pack)
    assert ops.z_signed.add(QZERO_4BIT).abs().max() > 1e-6, "expected a nonzero deployed z"

    # zero-z control: deployed z == 0  ->  z' = -8
    ops_zero = KernelOperands(
        Q_s=ops.Q_s, s=ops.s,
        z_signed=torch.full_like(ops.z_signed, -float(QZERO_4BIT)),
        t_g=ops.t_g, group_size=ops.group_size,
        out_features=ops.out_features, Cin_pad=ops.Cin_pad,
    )

    for label, o in (("real-z", ops), ("zero-z", ops_zero)):
        y_true = _fp64_truth(Xq, a, o)
        y = oracle_matmul(Xq, a, o, out_dtype=torch.float64, accum_dtype=torch.float64)
        err = (y - y_true).abs().max().item()
        assert err < 1e-9, f"{label}: oracle != fp64 truth ({err:.2e})"

    y_real = oracle_matmul(Xq, a, ops, out_dtype=torch.float32)
    y_zero = oracle_matmul(Xq, a, ops_zero, out_dtype=torch.float32)
    assert not torch.allclose(y_real, y_zero), "z-correction must change the output"


# --------------------------------------------------------------------------- #
# (d) -- grouped-vs-flat reduction (constant s,z across groups) + P identity
# --------------------------------------------------------------------------- #
def test_grouped_equals_flat_when_sz_constant():
    torch.manual_seed(5)
    out_f, n_tpr, gs = 32, 4, 64
    Cin = n_tpr * gs
    T = 20

    Q_s = torch.randint(-8, 8, (out_f, n_tpr, gs))               # {-8..7}
    s_col = torch.rand(out_f, 1) * 0.02 + 0.001
    z_col = torch.randn(out_f, 1) - float(QZERO_4BIT)            # signed z'
    s = s_col.expand(out_f, n_tpr).contiguous()                 # constant across groups
    z_signed = z_col.expand(out_f, n_tpr).contiguous()
    ops = KernelOperands(
        Q_s=Q_s, s=s, z_signed=z_signed, t_g=torch.ones(out_f, n_tpr),
        group_size=gs, out_features=out_f, Cin_pad=Cin,
    )

    Xq = torch.randint(-7, 8, (T, Cin)).to(torch.float64)        # {-7..7}
    a = torch.rand(T, 1).to(torch.float64) + 0.1

    y_grouped = oracle_matmul(Xq, a, ops, out_dtype=torch.float64, accum_dtype=torch.float64)

    # flat form (valid only because s,z are constant across groups):
    #   a · s · ( Σ_k Xq·Q_s − z' · Σ_k Xq )
    Q_s_flat = Q_s.reshape(out_f, Cin).to(torch.float64)
    I_flat = Xq @ Q_s_flat.t()                                   # (T, out)
    P_flat = Xq.sum(dim=1, keepdim=True)                         # (T, 1)
    y_flat = a * (s_col.t().to(torch.float64) * (I_flat - z_col.t().to(torch.float64) * P_flat))

    err = (y_grouped - y_flat).abs().max().item()
    assert err < 1e-7, f"grouped != flat when s,z constant: {err:.2e}"

    # P identity: Σ_g P[i,g] == Σ_k Xq[i,k]
    P_g = Xq.reshape(T, n_tpr, gs).sum(dim=-1).sum(dim=-1)       # (T,)
    assert torch.equal(P_g, Xq.sum(dim=1)), "Σ_g P[i,g] must equal Σ_k Xq[i,k]"


# --------------------------------------------------------------------------- #
# oracle_linear convenience smoke (end-to-end, bitpacked pack)
# --------------------------------------------------------------------------- #
def test_oracle_linear_smoke_bitpacked():
    pack = _make_pack(out_f=64, Cin=256, bitpack=True, seed=6)
    cfg = ActQuantConfig(act_bits=4, scale_estimator="percentile", rotate_act=pack.rotated)
    x = torch.randn(8, 3, pack.Cin_orig, dtype=torch.float32)   # batched shape preserved
    y = oracle_linear(x, pack, cfg, out_dtype=torch.float32)
    assert y.shape == (8, 3, pack.out_features)
    assert torch.isfinite(y).all()


# --------------------------------------------------------------------------- #
# Per-token centering on the kernel path (Contract v3, R2.3 / R2.4)
# --------------------------------------------------------------------------- #
def _median_rel(y, truth):
    y, truth = y.detach().double(), truth.detach().double()
    return float(((y - truth).abs() / (truth.abs() + 1e-6)).median())


def test_centered_oracle_matches_reference_fakequant():
    """R2.3: the CENTERED kernel oracle (oracle_linear, center on) equals the centered
    reference fake-quant (x̃_hat @ W̃_hatᵀ + x̄⊗colsum) to the same fp32 envelope as the
    uncentered pair -- the exact rank-1 is added consistently in both paths, so the
    residual is just int-reduction-vs-dense-matmul ordering. Strong per-token mean."""
    torch.manual_seed(20)
    out_f, Cin = 96, 256
    cfg = ActQuantConfig(act_bits=4, scale_estimator="percentile", rotate_act=True,
                         center_act_mean=True)
    W = torch.randn(out_f, Cin, dtype=torch.float32) * 0.05
    bias = torch.randn(out_f, dtype=torch.float32)
    pack = prepare_weight(W, bias=bias, bits=4, group_size=64, device="cpu",
                          compute_dtype=torch.float32, bitpack=True, strict_rho=False)
    x = torch.randn(40, Cin, dtype=torch.float32) + 1.7         # strong per-token DC

    # centered kernel oracle (fp32 accum -> single cast)
    y_oracle = oracle_linear(x, pack, cfg, out_dtype=torch.float32,
                             rotate_impl="dense", rotate_dtype=torch.float32)
    # centered reference fake-quant: same codes, dense dequant matmul + exact rank-1
    x_hat, diag = quantize_act(
        x, b=4, gamma_star=cfg.gamma_star, estimator="percentile", rotate_act=True,
        percentile_p=cfg.percentile_p, return_diag=True, return_rotated=True,
        center_mean=True, rotate_impl="dense", rotate_dtype=torch.float32,
    )
    W_hat = dequantize_prepared(pack)
    y_ref = x_hat @ W_hat.t() + diag["x_bar"] * pack.colsum + bias
    rel = _median_rel(y_oracle, y_ref)
    assert rel < 1e-3, f"centered oracle vs centered fakequant: median_rel={rel:.2e}"

    # control: centering actually changed the output (vs the uncentered oracle).
    cfg_off = ActQuantConfig(act_bits=4, scale_estimator="percentile", rotate_act=True,
                             center_act_mean=False)
    y_off = oracle_linear(x, pack, cfg_off, out_dtype=torch.float32,
                          rotate_impl="dense", rotate_dtype=torch.float32)
    assert _median_rel(y_oracle, y_off) > 1e-2, "centering should change the output"


def test_colsum_operand_is_fp_weight_not_w_hat():
    """R2.4 anti-regression (MANDATORY): the centering colsum operand is the ORIGINAL
    FP-weight column sum (DECISION 1), NOT a dequantized-Ŵ row sum. This test fails if
    anyone 'simplifies' the rank-1 to Ŵ row-sums later. Constructed so colsum(W_fp) and
    colsum(Ŵ) differ measurably."""
    torch.manual_seed(21)
    out_f, Cin = 128, 256        # out_f % mem_n(128) == 0 so the Nunchaku packer accepts it
    W = torch.randn(out_f, Cin, dtype=torch.float32) * 0.05
    pack = prepare_weight(W, bits=4, group_size=64, device="cpu",
                          compute_dtype=torch.float32, bitpack=True, strict_rho=False)

    colsum_fp = W.sum(dim=1)                                     # the correct operand
    # (1) prepare_weight + pack_weights carry EXACTLY the FP colsum.
    assert torch.allclose(pack.colsum, colsum_fp, atol=1e-5), "PreparedWeight.colsum != FP"
    pw = pack_weights(pack, c_dtype=torch.float16, scheme="zero-pad")
    assert pw.colsum is not None and torch.allclose(pw.colsum, colsum_fp, atol=1e-5), \
        "PackedWeights.colsum must be the FP-weight column sum"

    # (2) the trap value: colsum of the dequantized weight, brought back to the original
    # basis, differs measurably from the FP colsum (quantization error in the DC channel).
    W_hat_rot = dequantize_prepared(pack)                        # (out_f, Cin_pad) rotated
    W_hat = unrotate_for("pad", "fwht")(W_hat_rot, Cin)          # back to (out_f, Cin)
    colsum_what = W_hat.sum(dim=1)
    drift = (colsum_what - colsum_fp).abs().max().item()
    assert drift > 1e-4, f"test ill-posed: Ŵ colsum too close to FP ({drift:.2e})"
    assert not torch.allclose(pw.colsum, colsum_what, atol=drift / 4), \
        "operand colsum coincides with the Ŵ row-sum trap value"
