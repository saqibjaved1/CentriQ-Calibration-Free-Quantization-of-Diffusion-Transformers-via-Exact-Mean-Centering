"""
RoDS-A Step 3 / G5.2 — unit tests for rods_a.int4_gemm.

CPU tests validate the swizzle-independent index/scatter math of
`pack_activations_kernel` (round-trip against an inverse built from the same store
formula, using a random bijection in place of the recovered perm). The real
device-recovered perm + GEMM parity run on the A100 via scripts/int4_gemm_parity.py
(gated below with skipif when CUDA / the artifact are unavailable).
"""

import os

import pytest
import torch

from rods_a import int4_gemm as ig


def _inverse_unpack(act_u8, M, K, perm):
    """Invert pack_activations_kernel using the SAME store index + within-tile perm,
    so a round-trip exercises the index math independent of the real swizzle."""
    Mpad = act_u8.shape[0]
    buf = act_u8.reshape(-1).to(torch.int64)
    inv = torch.empty(ig._INSN_M * ig._INSN_K, dtype=torch.int64)
    flat = perm.reshape(-1)
    inv[flat] = torch.arange(flat.numel())  # nibble -> (m*INSN_K + k)
    codes = torch.zeros(Mpad, K, dtype=torch.int64)
    for i in range(Mpad):
        bm, rem = i // ig._BLOCK_M, i % ig._BLOCK_M
        warpId, wrem = rem // ig._WARP_M, rem % ig._WARP_M
        tileId, m = wrem // ig._INSN_M, wrem % ig._INSN_M
        for k in range(K):
            bk, kk = k // ig._WARP_K, k % ig._WARP_K
            nu = int(perm[m, kk])
            base = ((((bm * (K // ig._WARP_K) + bk) * ig._NUM_WARPS + warpId)
                     * ig._WARP_M_TILES + tileId) * ig._WARP_SIZE)
            gnib = base * ig._NIB_PER_UINT4 + nu
            byte, half = gnib // 2, gnib % 2
            nib = (int(buf[byte]) >> (4 * half)) & 0xF
            codes[i, k] = nib - 16 if nib >= 8 else nib
    return codes[:M]


def _random_perm(seed=0):
    g = torch.Generator().manual_seed(seed)
    flat = torch.randperm(ig._INSN_M * ig._INSN_K, generator=g)
    return flat.reshape(ig._INSN_M, ig._INSN_K)


def test_pack_shape_and_dtype():
    perm = _random_perm()
    codes = torch.randint(-7, 8, (200, 128), dtype=torch.int64)
    out = ig.pack_activations_kernel(codes, perm=perm)
    assert out.dtype == torch.uint8
    assert out.shape == (256, 64)  # M padded 200->256, K//2


@pytest.mark.parametrize("M,K", [(256, 128), (300, 64), (16, 192)])
def test_pack_roundtrip(M, K):
    perm = _random_perm(seed=M + K)
    codes = torch.randint(-7, 8, (M, K), dtype=torch.int64)
    out = ig.pack_activations_kernel(codes, perm=perm)
    rec = _inverse_unpack(out, M, K, perm)
    assert torch.equal(rec, codes), "pack/unpack round-trip mismatch (index/scatter bug)"


def test_pack_negative_nibble_twos_complement():
    perm = _random_perm()
    codes = torch.full((256, 128), -1, dtype=torch.int64)
    out = ig.pack_activations_kernel(codes, perm=perm)
    assert (out == 0xFF).all(), "code -1 must store as nibble 0xF (two's complement)"


_HAS_ARTIFACT = os.path.exists(ig._ACT_SWIZZLE_PATH)


@pytest.mark.skipif(not torch.cuda.is_available() or not _HAS_ARTIFACT,
                    reason="GEMM parity needs CUDA + recovered act_swizzle.pt")
def test_g21_synthetic_parity():
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from int4_gemm_parity import g2_1_synthetic
    dev = torch.device("cuda")
    _, gemm = ig.load_nunchaku_ops()
    quantize = ig.load_nunchaku_quantize()
    perm = ig.load_act_swizzle()
    assert g2_1_synthetic(dev, gemm, quantize, perm), "G2.1 synthetic exact parity failed"
