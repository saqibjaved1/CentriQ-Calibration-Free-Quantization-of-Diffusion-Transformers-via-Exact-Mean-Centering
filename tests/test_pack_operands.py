"""
Operand-packing + correction-precompute tests (RoDS-A Step 2, plan P1.2/P2.2/P3).

Validates rods_a.pack_operands against the Step-1 oracle (rods_a.kernel_semantics).
Host-only (torch, CPU), deterministic seeds. No CUDA.

P1.2  weight round-trip: pack -> unpack codes bit-identical to the remapped Q_s;
      wmscale round-trip lossless; C matches s·t·z'. (Byte-identical vs the real
      Nunchaku packer is checked against the A100 fixture when present.)
P2.2  activation round-trip: unpack(Xq_packed) bit-identical to the oracle codes;
      P_ig == Σ_{k∈g} Xq_ik. On synthetic seeds and the 18 captured real cells.
P3.1  EPILOGUE IDENTITY (the gate): the restructured form
      [Σ_g wmscale·intred − a·(P Cᵀ)] from packed-then-unpacked buffers equals the
      oracle (scale_dtype=fp16): integer parts bit-exact; FP within the O2.4 envelope
      (median rel <= 3e-4). Covers every deployed (Cin, Cout, scheme) class.
P3.3  side-GEMM precision: C stored fp16 vs fp32 both stay inside the envelope.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from rods_a.kernel_semantics import oracle_matmul, to_kernel_operands
from rods_a.pack_operands import (
    pack_weights,
    pack_activations_ref,
    restructured_epilogue,
    unpack_weight_codes,
    unpack_scale,
    unpack_int4_codes,
)
from rods_a.prepare_weight import prepare_weight
from rods_a.quantize_act import quantize_act
from rods_a.wrapper import ActQuantConfig

# Two distinct envelopes:
#  IDENTITY -- the P3.1 GATE: the restructured epilogue vs the oracle (both fp16
#    scales, P0.3). This is the O2.4 "median rel <= 3e-4 vs the oracle" criterion;
#    it is algebraically exact, observed 0.0 (FP summation order is identical).
#  FIDELITY -- INFORMATIONAL (P0.3 amendment): the fp16-scale path vs fp64 truth.
#    FP16-rounded wmscale+a_i add rounding beyond the fp32-scale oracle's 1.66e-4
#    (Step-1 O2.3); re-measured here (worst 5.4e-4, typical ~2.4e-4). The 1e-3 bound
#    is Step-1's documented preliminary envelope -- a regression floor, not the gate.
IDENTITY_MEDIAN_REL = 3e-4
FIDELITY_MEDIAN_REL = 1e-3

# Real raw-activation cells (18) dumped on A100; tests degrade gracefully if absent.
_DUMP_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs", "rods_a_oracle", "raw_acts",
)
# Byte-identical Nunchaku packer fixture (produced by scripts/pack_byteref.py on A100).
_BYTEREF = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs", "pack_byteref", "byteref.pt",
)

# Shape classes: (out_f, Cin) with out_f % 128 == 0; Cin_pad = next_pow2 % 128 == 0.
# Includes a pow2 Cin (no padding) and a non-pow2 Cin (zero-pad fallback, P0.4).
_SYNTH_SHAPES = [(128, 256), (256, 384), (256, 1152)]


def _make_pack(out_f, Cin, *, seed=0, scale=0.05):
    torch.manual_seed(seed)
    W = torch.randn(out_f, Cin, dtype=torch.float32) * scale
    bias = torch.randn(out_f, dtype=torch.float32)
    return prepare_weight(
        W, bias=bias, bits=4, group_size=64, device="cpu",
        compute_dtype=torch.float32, bitpack=True, strict_rho=False,
    )


def _cfg(pack):
    return ActQuantConfig(act_bits=4, scale_estimator="percentile", rotate_act=pack.rotated)


def _quant(x, cfg, rotated):
    _, d = quantize_act(
        x, b=cfg.act_bits, gamma_star=cfg.gamma_star, estimator=cfg.scale_estimator,
        rotate_act=rotated, percentile_p=cfg.percentile_p, return_diag=True,
        return_rotated=True, rotate_impl="dense", rotate_dtype=torch.float32,
    )
    return d["a"], d["Xq_codes"]


def _fp64_truth(Xq, a, ops):
    out_f, n_tpr, gs = ops.Q_s.shape
    sg = ops.s.double() * ops.t_g.double()
    z = ops.z_signed.double()
    Qs = ops.Q_s.double()
    W = (sg.unsqueeze(-1) * (Qs - z.unsqueeze(-1))).reshape(out_f, n_tpr * gs)
    return (a.double() * Xq.double()) @ W.t()


def _median_rel(y, truth):
    y = y.detach().double()
    truth = truth.detach().double()
    scale = truth.abs().median().clamp_min(1e-6)
    return float(((y - truth).abs() / scale).median())


# --------------------------------------------------------------------------- #
# P1.2 -- weight-side round-trip
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("out_f,Cin", _SYNTH_SHAPES)
def test_weight_code_roundtrip(out_f, Cin):
    pack = _make_pack(out_f, Cin, seed=1)
    ops = to_kernel_operands(pack)
    pw = pack_weights(pack)
    codes = unpack_weight_codes(pw.Qpacked, ops.out_features, ops.Cin_pad)
    expect = ops.Q_s.reshape(ops.out_features, ops.Cin_pad).to(torch.int64)
    assert torch.equal(codes, expect), "pack->unpack weight codes not bit-identical"
    # codes live on the signed int4 grid {-8..7}.
    assert int(codes.min()) >= -8 and int(codes.max()) <= 7


@pytest.mark.parametrize("out_f,Cin", _SYNTH_SHAPES)
def test_wmscale_and_C_roundtrip(out_f, Cin):
    pack = _make_pack(out_f, Cin, seed=2)
    ops = to_kernel_operands(pack)
    pw = pack_weights(pack)
    # wmscale round-trip is lossless (fp16 values only permuted).
    wm = unpack_scale(pw.wmscales, ops.out_features, ops.n_tpr)
    assert torch.equal(wm, pw.wmscale_dense), "wmscale pack->unpack not bit-identical"
    assert wm.dtype == torch.float16
    # C == s·t·z' (fp32 compute).
    C_ref = (ops.s.float() * ops.t_g.float() * ops.z_signed.float())
    assert torch.allclose(pw.C.float(), C_ref, atol=0, rtol=0), "C != s·t·z'"
    # wmscale_dense == (s·t) rounded to fp16.
    assert torch.equal(pw.wmscale_dense, (ops.s * ops.t_g).to(torch.float16))


def test_weight_pack_byteidentical_fixture():
    """Vendored packer == real Nunchaku packer, byte-for-byte (plan P1.2).

    Compares the vendored pack_weight/pack_scale output to the real installed
    Nunchaku packer output captured on A100 by scripts/pack_byteref.py. Skipped
    until that fixture exists; this is the standard that justifies the vendoring.
    """
    if not os.path.exists(_BYTEREF):
        pytest.skip(f"byte-ref fixture not found at {_BYTEREF} (run scripts/pack_byteref.py on A100)")
    ref = torch.load(_BYTEREF, map_location="cpu")
    from rods_a._nunchaku_pack import NunchakuWeightPacker
    packer = NunchakuWeightPacker(bits=4, warp_n=int(ref.get("warp_n", 128)))
    for case in ref["cases"]:
        q = case["weight_int32"]
        got_w = packer.pack_weight(q.to(torch.int32))
        assert torch.equal(got_w, case["packed_weight"]), (
            f"vendored pack_weight != Nunchaku on shape {tuple(q.shape)}"
        )
        s = case["scale_fp16"]
        got_s = packer.pack_scale(s.to(torch.float16).clone(), group_size=int(case["group_size"]))
        assert torch.equal(got_s, case["packed_scale"]), (
            f"vendored pack_scale != Nunchaku on shape {tuple(s.shape)}"
        )


# --------------------------------------------------------------------------- #
# P2.2 -- activation-side round-trip
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("out_f,Cin", _SYNTH_SHAPES)
def test_activation_code_roundtrip(out_f, Cin):
    pack = _make_pack(out_f, Cin, seed=3)
    cfg = _cfg(pack)
    torch.manual_seed(100)
    X = torch.randn(40, pack.Cin_orig, dtype=torch.float32)
    pa = pack_activations_ref(X, cfg, group_size=64, rotate_impl="dense", rotate_dtype=torch.float32)
    codes = unpack_int4_codes(pa.Xq_packed, pa.meta["Cin_pad"])
    assert torch.equal(codes, pa.Xq_codes.to(torch.int64)), "activation pack->unpack mismatch"
    # ±7 grid (never -8).
    assert int(codes.min()) >= -7 and int(codes.max()) <= 7
    # P identity: P_ig == Σ_{k∈g} Xq_ik.
    n_tpr, gs = pa.meta["n_tpr"], pa.meta["group_size"]
    P_ref = pa.Xq_codes.reshape(-1, n_tpr, gs).sum(dim=-1).to(torch.int16)
    assert torch.equal(pa.P, P_ref), "P != group sum of codes"
    assert pa.P.dtype == torch.int16 and pa.a.dtype == torch.float16


# --------------------------------------------------------------------------- #
# P3.1 -- restructured-epilogue identity (THE GATE), synthetic coverage
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("out_f,Cin", _SYNTH_SHAPES)
def test_epilogue_identity_synthetic(out_f, Cin):
    pack = _make_pack(out_f, Cin, seed=4)
    cfg = _cfg(pack)
    ops = to_kernel_operands(pack)
    pw = pack_weights(pack, c_dtype=torch.float32)

    torch.manual_seed(200)
    X = torch.randn(48, pack.Cin_orig, dtype=torch.float32)
    pa = pack_activations_ref(X, cfg, group_size=64, rotate_impl="dense", rotate_dtype=torch.float32)

    # Buffers Step 3 consumes, recovered by pack->unpack (the identity is computed
    # from these, not the in-memory fp32 fit).
    Qs = unpack_weight_codes(pw.Qpacked, ops.out_features, ops.Cin_pad).reshape(
        ops.out_features, ops.n_tpr, ops.group_size)
    wm = unpack_scale(pw.wmscales, ops.out_features, ops.n_tpr)
    Xq = unpack_int4_codes(pa.Xq_packed, pa.meta["Cin_pad"]).to(torch.float64)

    # (1) integer parts bit-exact: int reductions from unpacked == from oracle ops.
    assert torch.equal(Qs, ops.Q_s.to(torch.int64)), "unpacked weight codes drifted"
    assert torch.equal(Xq.to(torch.int64), pa.Xq_codes.to(torch.int64)), "unpacked act codes drifted"

    # (2) restructured epilogue from packed buffers vs the oracle (fp16 scales, P0.3).
    y_oracle = oracle_matmul(pa.Xq_codes.to(torch.float64), pa.a, ops,
                             out_dtype=torch.float16, accum_dtype=torch.float32,
                             scale_dtype=torch.float16)
    y_restr = restructured_epilogue(
        Xq, pa.a, Qs, wm, pw.C, pa.P, group_size=ops.group_size,
        out_dtype=torch.float16, accum_dtype=torch.float32, corr_dtype=torch.float32,
    )
    rel_vs_oracle = _median_rel(y_restr, y_oracle)
    # (3) fp16-scale fidelity vs fp64 truth (informational, P0.3 envelope).
    y_true = _fp64_truth(pa.Xq_codes.to(torch.float64), pa.a, ops)
    rel_vs_truth = _median_rel(y_restr, y_true)
    print(f"\n[P3.1 {out_f}x{Cin}] restr-vs-oracle med_rel={rel_vs_oracle:.2e} (identity gate) "
          f"restr-vs-truth med_rel={rel_vs_truth:.2e} (fp16-scale fidelity)")
    # GATE: the restructured epilogue == the oracle (fp16 scales) -> exact.
    assert rel_vs_oracle <= IDENTITY_MEDIAN_REL, (
        f"restructured vs oracle median rel {rel_vs_oracle:.2e} > {IDENTITY_MEDIAN_REL:.0e}")
    # Sanity: fp16-scale path vs truth stays inside the P0.3 envelope.
    assert rel_vs_truth <= FIDELITY_MEDIAN_REL, (
        f"restructured vs truth median rel {rel_vs_truth:.2e} > {FIDELITY_MEDIAN_REL:.0e}")


# --------------------------------------------------------------------------- #
# P3.3 -- side-GEMM precision: C stored fp16 vs fp32 both inside the envelope
# --------------------------------------------------------------------------- #
def test_side_gemm_precision_choice():
    pack = _make_pack(256, 1152, seed=5)
    cfg = _cfg(pack)
    ops = to_kernel_operands(pack)
    pw = pack_weights(pack, c_dtype=torch.float32)
    torch.manual_seed(300)
    X = torch.randn(64, pack.Cin_orig, dtype=torch.float32)
    pa = pack_activations_ref(X, cfg, group_size=64, rotate_impl="dense", rotate_dtype=torch.float32)

    Qs = unpack_weight_codes(pw.Qpacked, ops.out_features, ops.Cin_pad).reshape(
        ops.out_features, ops.n_tpr, ops.group_size)
    wm = unpack_scale(pw.wmscales, ops.out_features, ops.n_tpr)
    Xq = unpack_int4_codes(pa.Xq_packed, pa.meta["Cin_pad"]).to(torch.float64)
    y_true = _fp64_truth(pa.Xq_codes.to(torch.float64), pa.a, ops)

    rels = {}
    for cdt in (torch.float32, torch.float16):
        y = restructured_epilogue(Xq, pa.a, Qs, wm, pw.C, pa.P, group_size=ops.group_size,
                                  out_dtype=torch.float16, corr_dtype=cdt)
        rels[str(cdt)] = _median_rel(y, y_true)
    print(f"\n[P3.3 side-GEMM C dtype vs truth] {rels}  "
          f"(fp16 adds {rels['torch.float16'] - rels['torch.float32']:+.1e})")
    # fp16-stored C is the cheaper choice and stays inside the P0.3 fidelity envelope.
    for k, v in rels.items():
        assert v <= FIDELITY_MEDIAN_REL, f"C={k} median rel {v:.2e} > {FIDELITY_MEDIAN_REL:.0e}"


# --------------------------------------------------------------------------- #
# P2.2 + P3.1 -- REAL captured cells (18); skipped if artifacts absent
# --------------------------------------------------------------------------- #
def _load_cells():
    mpath = os.path.join(_DUMP_DIR, "dump_manifest.json")
    if not os.path.exists(mpath):
        return None
    with open(mpath) as f:
        return json.load(f)["cells"]


def test_real_cells_roundtrip_and_identity():
    cells = _load_cells()
    if not cells:
        pytest.skip(f"raw_acts not found at {_DUMP_DIR} (A100 dump_raw artifacts)")
    cfg = ActQuantConfig(act_bits=4, scale_estimator="percentile", rotate_act=True)
    worst_identity = 0.0     # restructured vs fp16-scale oracle (the P3.1 gate)
    worst_fidelity = 0.0     # fp16-scale path vs fp64 truth (P0.3 envelope, informational)
    # Cache one prepared weight per layer (the fit is layer-, not cell-, specific).
    packs = {}
    for c in cells:
        layer = c["layer"]
        if layer not in packs:
            wb = torch.load(os.path.join(_DUMP_DIR, c["weight_file"]), map_location="cpu")
            packs[layer] = prepare_weight(
                wb["weight"], bias=wb["bias"], bits=4, group_size=64, device="cpu",
                compute_dtype=torch.float32, rotate_weight=True, strict_rho=False,
            )
        pack = packs[layer]
        ops = to_kernel_operands(pack)
        pw = pack_weights(pack, c_dtype=torch.float32)

        X = torch.load(os.path.join(_DUMP_DIR, c["act_file"]), map_location="cpu").to(torch.float32)
        pa = pack_activations_ref(X, cfg, group_size=64, rotate_impl="dense", rotate_dtype=torch.float32)

        # round-trip (P2.2)
        codes = unpack_int4_codes(pa.Xq_packed, pa.meta["Cin_pad"])
        assert torch.equal(codes, pa.Xq_codes.to(torch.int64)), f"{layer}: act round-trip"
        wcodes = unpack_weight_codes(pw.Qpacked, ops.out_features, ops.Cin_pad)
        assert torch.equal(wcodes, ops.Q_s.reshape(ops.out_features, ops.Cin_pad).to(torch.int64)), \
            f"{layer}: weight round-trip"

        # epilogue identity (P3.1): restructured (packed buffers) vs fp16-scale oracle.
        Qs = wcodes.reshape(ops.out_features, ops.n_tpr, ops.group_size)
        wm = unpack_scale(pw.wmscales, ops.out_features, ops.n_tpr)
        Xq = unpack_int4_codes(pa.Xq_packed, pa.meta["Cin_pad"]).to(torch.float64)
        y_restr = restructured_epilogue(Xq, pa.a, Qs, wm, pw.C, pa.P,
                                        group_size=ops.group_size, out_dtype=torch.float16)
        y_oracle = oracle_matmul(pa.Xq_codes.to(torch.float64), pa.a, ops,
                                 out_dtype=torch.float16, accum_dtype=torch.float32,
                                 scale_dtype=torch.float16)
        y_true = _fp64_truth(pa.Xq_codes.to(torch.float64), pa.a, ops)
        r_id = _median_rel(y_restr, y_oracle)        # GATE: must be ~0
        r_fid = _median_rel(y_restr, y_true)         # informational (P0.3 envelope)
        worst_identity = max(worst_identity, r_id)
        worst_fidelity = max(worst_fidelity, r_fid)
        print(f"  {layer:46} shape={c['shape']} Cin_pad={ops.Cin_pad} "
              f"identity(vs oracle)={r_id:.2e} fidelity(vs truth)={r_fid:.2e}")
    print(f"\n[P3.1 real cells] worst IDENTITY (vs oracle) = {worst_identity:.3e} "
          f"(gate {IDENTITY_MEDIAN_REL:.0e}) | worst FIDELITY (fp16-scale vs truth) = "
          f"{worst_fidelity:.3e} (P0.3 envelope, floor {FIDELITY_MEDIAN_REL:.0e})")
    assert worst_identity <= IDENTITY_MEDIAN_REL, (
        f"real-cell epilogue identity {worst_identity:.2e} > {IDENTITY_MEDIAN_REL:.0e}")
    assert worst_fidelity <= FIDELITY_MEDIAN_REL, (
        f"real-cell fp16-scale fidelity {worst_fidelity:.2e} > {FIDELITY_MEDIAN_REL:.0e}")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
