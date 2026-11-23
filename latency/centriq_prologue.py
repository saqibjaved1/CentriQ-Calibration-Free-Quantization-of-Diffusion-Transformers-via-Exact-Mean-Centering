"""CentriQ prologue (per-token centering + factorized Hadamard H_C (x) H_16 (x) H_16) as one Triton kernel.
Shared by the latency scripts. Exact to bf16 rounding vs the reference rotation (rel. err 0.003-0.004)."""
import math, os, sys
import torch
from torch import nn
import triton
import triton.language as tl
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dev, dt = "cuda", torch.bfloat16


def sylvester(n):
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)


def h12():
    from core.hadamard import build_hadamard
    return build_hadamard(12).to(torch.float32) / math.sqrt(12)



@triton.jit
def _fact_prologue_kernel(x_ptr, out_ptr, h16_ptr, hc_ptr, T, K: tl.constexpr, C: tl.constexpr, CPAD: tl.constexpr,
                          BT: tl.constexpr, KIN: tl.constexpr, GELU: tl.constexpr = False,
                          side_ptr=None, RS: tl.constexpr = 0, SIDE: tl.constexpr = False):
    """Fused centering + (H_C (x) H_16 (x) H_16) rotation. Per token, x is viewed as (C, 16, 16); the three axes are
    mixed by three small tensor-core dots, so no large Hadamard matrix is ever loaded (16x16 + CPADxCPAD only)."""
    pid = tl.program_id(0)
    t = pid * BT + tl.arange(0, BT)
    c = tl.arange(0, CPAD)
    a = tl.arange(0, 16)
    b = tl.arange(0, 16)
    off = t[:, None, None, None] * K + c[None, :, None, None] * 256 + a[None, None, :, None] * 16 + b[None, None, None, :]
    msk = (t[:, None, None, None] < T) & (c[None, :, None, None] < C)
    col = c[None, :, None, None] * 256 + a[None, None, :, None] * 16 + b[None, None, None, :]
    imsk = msk & (col < KIN)                                                         # input zero-padded KIN -> K
    x = tl.load(x_ptr + t[:, None, None, None] * KIN + col, mask=imsk, other=0.0).to(tl.float32)
    if GELU:   # tanh-GELU of the previous layer, applied on load: x * sigmoid(2*sqrt(2/pi)*(x + 0.044715 x^3))
        x = x * tl.sigmoid(1.5957691216057308 * (x + 0.044715 * x * x * x))
    mean = tl.sum(tl.reshape(x, (BT, CPAD * 256)), axis=1) / KIN                     # (BT,) mean over real channels
    x = tl.where(imsk, x - mean[:, None, None, None], 0.0)
    h16 = tl.load(h16_ptr + a[:, None] * 16 + b[None, :])                             # (16, 16) bf16, symmetric
    # inner axis b
    y = tl.dot(tl.reshape(x.to(tl.bfloat16), (BT * CPAD * 16, 16)), h16)              # rows (t,c,a), cols b'
    # middle axis a
    y = tl.permute(tl.reshape(y.to(tl.bfloat16), (BT * CPAD, 16, 16)), (0, 2, 1))     # (tc, b', a)
    y = tl.dot(tl.reshape(y, (BT * CPAD * 16, 16)), h16)                              # rows (t,c,b'), cols a'
    # chunk axis c
    y = tl.permute(tl.reshape(y.to(tl.bfloat16), (BT, CPAD, 256)), (0, 2, 1))        # (t, b'a', c)
    cc = tl.arange(0, CPAD)
    hcm = tl.load(hc_ptr + cc[:, None] * C + cc[None, :], mask=(cc[:, None] < C) & (cc[None, :] < C), other=0.0)
    z = tl.dot(tl.reshape(y, (BT * 256, CPAD)), hcm.to(tl.bfloat16))                 # z[., c'] = sum_c y[., c] H_C[c, c']
    z = tl.permute(tl.reshape(z, (BT, 16, 16, CPAD)), (0, 3, 2, 1))                  # (t, c', a', b')
    tl.store(out_ptr + off, z.to(tl.bfloat16), mask=msk)
    if SIDE:   # CentriQ side vector per token: 64-group sums of the rotated centered activation, then the mean
        rs = tl.sum(z, axis=3)                                                          # (BT, CPAD, 16): sum over b'
        for j in tl.static_range(4):                                                    # group j = a' in [4j, 4j+4)
            sel = (a >= 4 * j) & (a < 4 * j + 4)
            gj = tl.sum(tl.where(sel[None, None, :], rs, 0.0), axis=2)                  # (BT, CPAD)
            tl.store(side_ptr + t[:, None] * RS + c[None, :] * 4 + j, gj,
                     mask=(t[:, None] < T) & (c[None, :] < C))
        tl.store(side_ptr + t * RS + C * 4, mean, mask=t < T)


class FactPrologue(nn.Module):
    """Same math as FusedPrologue (H_12 (x) H_q with H_q Sylvester, q = K/12), via H_256 = H_16 (x) H_16."""
    def __init__(self, K, BT=None):
        super().__init__()
        self.KIN = K
        if K == 15360:                    # single-block proj_out: zero-pad to 16384, Sylvester H (rods_a "pad" scheme)
            K = 16384
        Q = 256; C = K // Q; assert C * Q == K and C in (12, 48, 64)
        hc = h12() if C == 12 else (torch.kron(h12(), sylvester(4)) if C == 48 else sylvester(64))
        self.K, self.C = K, C
        self.CPAD = 16 if C == 12 else 64
        self.BT = 1   # one token per program: exact and fastest (multi-token tiles are not supported)
        self.register_buffer("h16", sylvester(16).to(dev, dt).contiguous())
        self.register_buffer("hc", hc.to(dev, torch.float32).contiguous())
    def forward(self, x2, gelu=False, side=None):
        T = x2.shape[0]; out = torch.empty(T, self.K, device=x2.device, dtype=x2.dtype)
        _fact_prologue_kernel[(triton.cdiv(T, self.BT),)](x2, out, self.h16, self.hc, T, self.K, self.C, self.CPAD,
                                                          self.BT, self.KIN, GELU=gelu,
                                                          side_ptr=side, RS=0 if side is None else side.shape[1],
                                                          SIDE=side is not None, num_warps=8)
        return out




@triton.jit
def _fact_prologue_q_kernel(x_ptr, act_ptr, asc_ptr, side_ptr, perm_ptr, h16_ptr, hc_ptr, T, K: tl.constexpr,
                            C: tl.constexpr, CPAD: tl.constexpr, KIN: tl.constexpr, RS: tl.constexpr,
                            GELU: tl.constexpr = False):
    """Fused CentriQ activation path for Nunchaku's W4A4 GEMM, one token per program:
    centering + (H_C (x) H_16 (x) H_16) rotation + per-64-group INT4 quantization, written directly in Nunchaku's
    packed MMA layout (act) and scale layout (ascales), plus CentriQ's side vector (64-group sums + mean) as the
    GEMM's low-rank input. Replaces both the BF16 prologue output and Nunchaku's quantize kernel."""
    t = tl.program_id(0)
    c = tl.arange(0, CPAD)
    a = tl.arange(0, 16)
    b = tl.arange(0, 16)
    col = c[:, None, None] * 256 + a[None, :, None] * 16 + b[None, None, :]
    imsk = (c[:, None, None] < C) & (col < KIN)
    x = tl.load(x_ptr + t * KIN + col, mask=imsk, other=0.0).to(tl.float32)        # (CPAD, 16, 16)
    if GELU:
        x = x * tl.sigmoid(1.5957691216057308 * (x + 0.044715 * x * x * x))
    mean = tl.sum(tl.reshape(x, (CPAD * 256,)), axis=0) / KIN
    x = tl.where(imsk, x - mean, 0.0)
    h16 = tl.load(h16_ptr + a[:, None] * 16 + b[None, :])
    y = tl.dot(tl.reshape(x.to(tl.bfloat16), (CPAD * 16, 16)), h16)                  # rows (c,a), cols b'
    y = tl.permute(tl.reshape(y.to(tl.bfloat16), (CPAD, 16, 16)), (0, 2, 1))          # (c, b', a)
    y = tl.dot(tl.reshape(y, (CPAD * 16, 16)), h16)                                   # rows (c,b'), cols a'
    y = tl.permute(tl.reshape(y.to(tl.bfloat16), (CPAD, 256)), (1, 0))               # (b'a', c)
    cc = tl.arange(0, CPAD)
    hcm = tl.load(hc_ptr + cc[:, None] * C + cc[None, :], mask=(cc[:, None] < C) & (cc[None, :] < C), other=0.0)
    z = tl.dot(y, hcm.to(tl.bfloat16))                                                # (256 = b'a', CPAD = c')
    zr = tl.reshape(z, (16, 16, CPAD))                                                # (b', a', c')
    # per-(c', 64-group j) absmax -> scale; group j = a' in [4j, 4j+4), all b'
    am = tl.max(tl.abs(zr), axis=0)                                                   # (a', c')
    rs = tl.sum(zr, axis=0)                                                           # (a', c')
    rem = t % 256
    bm = t // 256
    pos = 16 * (rem // 16) + 2 * (rem % 8) + (rem % 16) // 8
    sc = tl.zeros((16, CPAD), dtype=tl.float32)
    for j in tl.static_range(4):
        sel = (a >= 4 * j) & (a < 4 * j + 4)
        sj = tl.max(tl.where(sel[:, None], am, 0.0), axis=0) / 7.0                    # (CPAD,)
        sj = tl.where(sj > 0, sj, 1e-8)
        sc = tl.where(sel[:, None], sj[None, :], sc)
        g = c * 4 + j
        tl.store(asc_ptr + bm * (K // 64) * 256 + g * 256 + pos, sj.to(tl.bfloat16), mask=c < C)
        gs = tl.sum(tl.where(sel[:, None], rs, 0.0), axis=0)
        tl.store(side_ptr + t * RS + g, gs, mask=c < C)
    tl.store(side_ptr + t * RS + C * 4, mean)
    q = tl.floor(zr / sc[None, :, :] + 0.5)
    q = tl.minimum(tl.maximum(q, -7.0), 7.0).to(tl.int32)                             # (b', a', c')
    # pack 8 consecutive k (b' = 8r + i) of one token into one 32-bit word: nibble i at bits 4i (little-endian)
    q4 = tl.reshape(q, (2, 8, 16, CPAD))                                              # (r, i, a', c')
    ii = tl.arange(0, 8)
    v = (q4 & 15).to(tl.uint32) << (4 * ii[None, :, None, None]).to(tl.uint32)
    word = tl.sum(v, axis=1)                                                          # (r, a', c'), disjoint bits
    r = tl.arange(0, 2)
    m = rem % 16
    warp = rem // 32
    tile = (rem % 32) // 16
    kk = (a[None, :, None] % 4) * 16 + 8 * r[:, None, None]                          # (r, a', 1) first k of the run
    nu = tl.load(perm_ptr + m * 64 + kk)                                              # nibble index, multiple of 8
    g3 = c[None, None, :] * 4 + a[None, :, None] // 4                                 # (1, a', c')
    base = (((bm * (K // 64) + g3) * 8 + warp) * 2 + tile) * 32
    tl.store(act_ptr + (base * 32 + nu) // 8, word, mask=(c[None, None, :] < C) & (r[:, None, None] >= 0))

class FactPrologueQ(nn.Module):
    """Fused CentriQ prologue + INT4 quantize + pack for Nunchaku's W4A4 GEMM (see kernel docstring)."""
    def __init__(self, K, perm):
        super().__init__()
        assert K in (3072, 12288), K
        C = K // 256
        hc = h12() if C == 12 else torch.kron(h12(), sylvester(4))
        self.K, self.C, self.CPAD = K, C, (16 if C == 12 else 64)
        self.register_buffer("h16", sylvester(16).to(dev, dt).contiguous())
        self.register_buffer("hc", hc.to(dev, torch.float32).contiguous())
        self.register_buffer("perm", perm.to(dev, torch.int32).contiguous())
    def forward(self, x2, R, gelu=False, pad_size=256):
        T = x2.shape[0]; Mpad = (T + pad_size - 1) // pad_size * pad_size
        act = torch.zeros(Mpad, self.K // 2, device=x2.device, dtype=torch.uint8) if Mpad != T else \
            torch.empty(Mpad, self.K // 2, device=x2.device, dtype=torch.uint8)
        asc = torch.zeros(self.K // 64, Mpad, device=x2.device, dtype=torch.bfloat16)
        side = torch.zeros(Mpad, R, device=x2.device, dtype=torch.float32)
        _fact_prologue_q_kernel[(T,)](x2, act.view(torch.int32), asc, side, self.perm, self.h16, self.hc, T, self.K, self.C, self.CPAD,
                                      self.K, R, GELU=gelu, num_warps=8)
        return act, asc, side
