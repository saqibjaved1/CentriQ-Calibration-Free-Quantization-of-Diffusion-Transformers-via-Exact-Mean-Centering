"""[v2: --share, --prologue fwht] CentriQ deployed on Blackwell (RTX PRO 6000): latency only (weights random; images not meaningful).

Per quantized linear (CentriQ's FLUX layer set, select_act_targets):
  center   : x - mean(x)                                   (torch; optionally torch.compile-fused)
  rotate   : factorized Hadamard (H_12 (x) H_2^k), contiguous cuBLAS GEMMs, no permute copies
  W4A4     : Nunchaku NVFP4 quantize + GEMM, side branch at CentriQ's rank (zero-point + rank-1 mean, App. B.2)
Unfused prologue => an UPPER BOUND on CentriQ's cost; the paper's kernel fuses center+rotate into quantize.

Modes:  --mode bf16 | centriq      end-to-end FLUX.1-schnell 1024^2, 4 steps (CUDA events breakdown)
        --mode layers              per-component breakdown per denoising step (count-weighted, real shapes)
"""
import argparse, json, math, statistics as st, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import nn
from nunchaku.models.linear import SVDQW4A4Linear

ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True)
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--warmup", type=int, default=3)
ap.add_argument("--bs", type=int, default=1)
ap.add_argument("--compile", action="store_true")
ap.add_argument("--fused", action="store_true")
ap.add_argument("--json", default=None)
ap.add_argument("--precision", default="nvfp4", choices=["nvfp4", "int4"], help="Nunchaku 4-bit format (A100: int4)")
ap.add_argument("--share", action="store_true", help="one prologue/quantize/GEMM per shared input (fused QKV[+MLP]), for every method")
ap.add_argument("--prologue", default="triton", choices=["triton", "fwht", "fact"])
a = ap.parse_args()
dev, dt = "cuda", torch.bfloat16


def sylvester(n):
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)


def h12():
    from core.hadamard import build_hadamard
    return build_hadamard(12).to(torch.float32) / math.sqrt(12)


class Rot(nn.Module):
    """y = x (H_12 (x) H_q) for K = 12*q, q in {256, 1024}; contiguous GEMMs only."""
    def __init__(self, K):
        super().__init__()
        assert K % 12 == 0
        self.K, q = K, K // 12
        self.register_buffer("H12", h12().to(dev, dt))
        self.q = q
        if q <= 256:
            self.register_buffer("Hq", sylvester(q).to(dev, dt)); self.split = None
        else:
            s = int(math.isqrt(q)); assert s * s == q
            self.register_buffer("Hs", sylvester(s).to(dev, dt)); self.split = s

    def forward(self, x):                      # x: (T, K)
        T = x.shape[0]
        if self.split is None:
            y = (x.reshape(T * 12, self.q) @ self.Hq).view(T, 12, self.q)
        else:
            s = self.split
            y = (x.reshape(-1, s) @ self.Hs).view(T * 12, s, s)          # inner axis
            y = torch.matmul(self.Hs.t(), y).view(T, 12, self.q)        # middle axis (Hs symmetric)
        return torch.matmul(self.H12.t(), y).reshape(T, self.K)        # chunk axis


import triton
import triton.language as tl


@triton.jit
def _center_rotate_kernel(x_ptr, out_ptr, hq_ptr, hc_ptr, T, K: tl.constexpr, C: tl.constexpr,
                          CPAD: tl.constexpr, BT: tl.constexpr, Q: tl.constexpr, NB: tl.constexpr):
    """Fused CentriQ prologue: per-token mean removal + (H_C (x) H_Q) rotation, one read + one write.
    Rows of the tile are (token, chunk) pairs: BT tokens x CPAD chunks (chunks >= C are zero padding)."""
    pid = tl.program_id(0)
    R: tl.constexpr = BT * CPAD
    r = tl.arange(0, R)
    t = pid * BT + r // CPAD
    c = r % CPAD
    rv = (c < C) & (t < T)
    j = tl.arange(0, Q)
    x = tl.load(x_ptr + t[:, None] * K + c[:, None] * Q + j[None, :], mask=rv[:, None], other=0.0).to(tl.float32)
    rs = tl.reshape(tl.sum(x, axis=1), (BT, CPAD))
    mean = tl.sum(rs, axis=1) / K                                    # (BT,)
    mrow = tl.reshape(tl.broadcast_to(mean[:, None], (BT, CPAD)), (R,))
    xc = tl.where(rv[:, None], x - mrow[:, None], 0.0).to(tl.bfloat16)
    # block-diagonal chunk mixer BD[r, s] = H_C[s%CPAD, r%CPAD] within the same token (= H_C^T per token)
    s = tl.arange(0, R)
    same = (r[:, None] // CPAD) == (s[None, :] // CPAD)
    cr = r[:, None] % CPAD
    cs = s[None, :] % CPAD
    bd = tl.load(hc_ptr + cs * C + cr, mask=same & (cr < C) & (cs < C), other=0.0).to(tl.bfloat16)
    nn_ = tl.arange(0, NB)
    for nb in tl.static_range(Q // NB):
        h = tl.load(hq_ptr + j[:, None] * Q + nb * NB + nn_[None, :])          # (Q, NB) bf16
        y = tl.dot(xc, h).to(tl.bfloat16)                                    # inner Hadamard
        z = tl.dot(bd, y)                                                    # chunk mixing
        tl.store(out_ptr + t[:, None] * K + c[:, None] * Q + nb * NB + nn_[None, :], z.to(tl.bfloat16),
                 mask=rv[:, None])


class FusedPrologue(nn.Module):
    """center + rotate in one Triton kernel. K = C*256 with C in {12, 48}: H_12 (x) H_1024 == (H_12 (x) H_4) (x) H_256."""
    def __init__(self, K):
        super().__init__()
        Q = 256; C = K // Q; assert C * Q == K and C in (12, 48)
        hc = h12() if C == 12 else torch.kron(h12(), sylvester(4))
        self.K, self.C, self.Q = K, C, Q
        self.CPAD = 16 if C == 12 else 64
        self.BT = 64 // self.CPAD
        self.register_buffer("hq", sylvester(Q).to(dev, dt).contiguous())
        self.register_buffer("hc", hc.to(dev, torch.float32).contiguous())

    def forward(self, x2):
        T = x2.shape[0]; out = torch.empty_like(x2)
        _center_rotate_kernel[(triton.cdiv(T, self.BT),)](x2, out, self.hq, self.hc, T, self.K, self.C, self.CPAD,
                                                          self.BT, self.Q, 64, num_warps=4)
        return out



@triton.jit
def _fact_prologue_kernel(x_ptr, out_ptr, h16_ptr, hc_ptr, T, K: tl.constexpr, C: tl.constexpr, CPAD: tl.constexpr,
                          BT: tl.constexpr, KIN: tl.constexpr):
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
    def forward(self, x2):
        T = x2.shape[0]; out = torch.empty(T, self.K, device=x2.device, dtype=x2.dtype)
        _fact_prologue_kernel[(triton.cdiv(T, self.BT),)](x2, out, self.h16, self.hc, T, self.K, self.C, self.CPAD,
                                                          self.BT, self.KIN, num_warps=8)
        return out


class FWHTPrologue(nn.Module):
    """center + rotate via the fast Walsh-Hadamard CUDA kernel (O(K log K)), exact:
    H(x - m1) = Hx - m*H1, and H1 is nonzero on only 12 coordinates, so centering is a 12-entry fix-up after the
    transform; m itself is read off those coordinates (<x,1> = <Hx, H1>)."""
    def __init__(self, K):
        super().__init__()
        from fast_hadamard_transform import hadamard_transform_12N
        self.f, self.K = hadamard_transform_12N, K
        self.scale = 1.0 / math.sqrt(K)
        v = self.f(torch.ones(1, K, device=dev, dtype=torch.float32), self.scale)[0]
        idx = torch.nonzero(v.abs() > 1e-3).flatten()
        assert idx.numel() == 12, idx.numel()
        self.register_buffer("idx", idx)
        self.register_buffer("v", v[idx].to(torch.float32))           # H1 restricted to its support
    def forward(self, x2):
        z = self.f(x2, self.scale)                                      # (T, K) bf16
        zs = z[:, self.idx].float()                                     # (T, 12)
        m = (zs @ self.v) / self.K                                      # per-token mean of x
        z[:, self.idx] = (zs - m[:, None] * self.v[None, :]).to(z.dtype)
        return z

PROLOGUES = {"triton": lambda K: FusedPrologue(K), "fwht": lambda K: FWHTPrologue(K), "fact": lambda K: FactPrologue(K)}
FUSED = bool(a.fused)


class CentriQLinear(nn.Module):
    def __init__(self, K, N, bias=True):
        super().__init__()
        self.K, self.N = K, N
        KG = 16384 if K == 15360 else K       # GEMM input width after zero-padding
        self.KG = KG
        self.rot = Rot(K) if K != 15360 else None
        self.fused = PROLOGUES[a.prologue](K) if FUSED else None
        rank = int(math.ceil((KG / 64 + 2) / 16) * 16)
        self.w4 = SVDQW4A4Linear(KG, N, rank=rank, bias=bias, precision=a.precision, torch_dtype=dt, device=dev)
        with torch.no_grad():
            for p in list(self.w4.parameters()) + list(self.w4.buffers()):
                if p.dtype in (torch.bfloat16, torch.float16, torch.float32): p.normal_(0, 0.02)
                elif not p.dtype.is_floating_point: p.random_(0, 15)

    def prologue(self, x2):
        if self.fused is not None:
            return self.fused(x2.contiguous())
        xc = x2 - x2.mean(-1, keepdim=True)
        return self.rot(xc)

    def forward(self, x):
        shp = x.shape
        z = self.prologue(x.reshape(-1, self.K))
        return self.w4(z.view(1, -1, self.KG)).view(*shp[:-1], self.N)


def timeit(fn, reps=200):
    for _ in range(10): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return float(torch.tensor(ts).median())


if a.mode == "check":
    for K in (3072, 12288):
        R = Rot(K).float(); x = torch.randn(8, K, device=dev)
        D = torch.kron(h12(), sylvester(K // 12)).to(dev)
        print(K, "max|rot - dense kron| =", float((R(x) - x @ D).abs().max()), " norm ratio", float(R(x).norm() / x.norm()))
        y = CentriQLinear(K, 3072)(torch.randn(1, 4096, K, device=dev, dtype=dt)); print(K, "layer ok", tuple(y.shape))
        xb = (torch.randn(4100, K, device=dev) + 0.7 * torch.randn(4100, 1, device=dev)).to(dt)   # tokens with a mean
        ref = Rot(K)((xb.float() - xb.float().mean(-1, keepdim=True)).to(dt)).float()
        fz = FusedPrologue(K)(xb).float()
        print(K, "fused vs reference: rel err", float((fz - ref).norm() / ref.norm()), " max", float((fz - ref).abs().max()))
        FC = FactPrologue(K); fc = FC(xb).float()
        print(K, "fact vs reference: rel err", float((fc - ref).norm() / ref.norm()), " max", float((fc - ref).abs().max()))
        for bt in (1,):
            try:
                F2 = FactPrologue(K, BT=bt); F0 = FusedPrologue(K); e2 = F2(xb).float(); print(K, f"BT={bt} rel err", float((e2 - ref).norm() / ref.norm()), "err in tokens 0..3:", [round(float((e2[i]-ref[i]).norm()/ref[i].norm()),3) for i in range(4)]); print(K, f"ms fact BT={bt}:", round(timeit(lambda: F2(xb)), 4), "| triton fused:", round(timeit(lambda: F0(xb)), 4))
            except Exception as e:
                print(K, "BT", bt, "failed:", str(e)[:120])
        try:
            FW = FWHTPrologue(K); zz = FW(xb).float(); xc = (xb.float() - xb.float().mean(-1, keepdim=True))
            D = FW.f(torch.eye(K, device=dev, dtype=torch.float32), FW.scale)   # explicit matrix of the transform
            ref2 = xc @ D
            print(K, "fwht vs exact centered transform: rel err", float((zz - ref2).norm() / ref2.norm()),
                  " orthogonal:", float((D @ D.t() - torch.eye(K, device=dev)).abs().max()))
            for _ in range(3): FW(xb)
            F0 = FusedPrologue(K); print(K, "ms: triton fused", timeit(lambda: F0(xb)), " fwht", timeit(lambda: FW(xb)))
        except ImportError as e:
            print("fwht unavailable:", e)
    sys.exit(0)

class SVDQSwap(nn.Module):
    def __init__(self, K, N, bias=True):
        super().__init__(); self.N = N
        self.w4 = SVDQW4A4Linear(K, N, rank=32, bias=bias, precision=a.precision, torch_dtype=dt, device=dev)
        with torch.no_grad():
            for p in list(self.w4.parameters()) + list(self.w4.buffers()):
                if p.dtype in (torch.bfloat16, torch.float16, torch.float32): p.normal_(0, 0.02)
                elif not p.dtype.is_floating_point: p.random_(0, 15)
    def forward(self, x):
        shp = x.shape
        return self.w4(x.reshape(1, -1, shp[-1])).view(*shp[:-1], self.N)


if a.mode == "layers":
    SHAPES = [(3072, 3072, 4096, 76), (3072, 3072, 512, 76), (3072, 12288, 4096, 19), (12288, 3072, 4096, 19),
              (3072, 12288, 512, 19), (12288, 3072, 512, 19), (3072, 3072, 4608, 114), (3072, 12288, 4608, 38)]
    if a.share:   # same layers, grouped by shared input (fused QKV / QKV+MLP)
        SHAPES = [(3072, 9216, 4096, 19), (3072, 3072, 4096, 19), (3072, 9216, 512, 19), (3072, 3072, 512, 19),
                  (3072, 12288, 4096, 19), (12288, 3072, 4096, 19), (3072, 12288, 512, 19), (12288, 3072, 512, 19),
                  (3072, 21504, 4608, 38)]
    tot = {}
    for K, N, T, cnt in SHAPES:
        x = torch.randn(T, K, device=dev, dtype=dt); W = torch.randn(N, K, device=dev, dtype=dt) * 0.02
        L = CentriQLinear(K, N); xc = x - x.mean(-1, keepdim=True); z = L.rot(xc)
        q = L.w4.quantize(z); out = torch.empty(T, N, device=dev, dtype=dt)
        center = (lambda: x - x.mean(-1, keepdim=True))
        if a.compile: center = torch.compile(center)
        r = {"bf16_gemm": timeit(lambda: x @ W.t()), "center": timeit(center), "rotate": timeit(lambda: L.rot(xc)),
             "quantize": timeit(lambda: L.w4.quantize(z)), "gemm_epilogue": timeit(lambda: L.w4.forward_quant(*q, output=out)),
             "centriq_layer_total": timeit(lambda: L(x))}
        FP = PROLOGUES[a.prologue](K)
        r["fused_center_rotate"] = timeit(lambda: FP(x))
        SQ = SVDQSwap(K, N); r["svdq_total"] = timeit(lambda: SQ(x.view(1, T, K)))
        qs = SQ.w4.quantize(x); outs = torch.empty(T, N, device=dev, dtype=dt)
        r["svdq_quantize"] = timeit(lambda: SQ.w4.quantize(x))
        r["svdq_gemm_epilogue"] = timeit(lambda: SQ.w4.forward_quant(*qs, output=outs))
        r["centriq_fused_total"] = timeit(lambda: L.w4(FP(x).view(1, -1, K)))
        for k, v in r.items(): tot[k] = tot.get(k, 0.0) + v * cnt
        print(K, N, T, {k: round(v, 3) for k, v in r.items()}, flush=True)
        del L; torch.cuda.empty_cache()
    print(json.dumps({k: round(v, 1) for k, v in tot.items()}))
    if a.json: json.dump({"gpu": torch.cuda.get_device_name(0), "ms_per_step": tot}, open(a.json, "w"), indent=1)
    sys.exit(0)

# ---------------- end to end ----------------
from diffusers import FluxPipeline
if os.environ.get("FLUX_RANDOM") == "1":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from flux_random import build_random_pipe
    pipe = build_random_pipe()
else:
    pipe = FluxPipeline.from_pretrained(os.environ.get("FLUX_PATH", "black-forest-labs/FLUX.1-schnell"), torch_dtype=dt).to(dev)
pipe.set_progress_bar_config(disable=True)
class BF16Rot(nn.Module):
    """Fusion control: CentriQ's fused prologue (center + rotate) + rank-1 epilogue, but a BF16 GEMM on the rotated weight."""
    def __init__(self, K, N, bias=True):
        super().__init__(); self.K, self.N = K, N
        self.pro = PROLOGUES[a.prologue](K)
        self.W = nn.Parameter(torch.randn(N, K, device=dev, dtype=dt) * 0.02, requires_grad=False)
        self.w1 = nn.Parameter(torch.randn(N, device=dev, dtype=dt) * 0.02, requires_grad=False)
        self.b = nn.Parameter(torch.zeros(N, device=dev, dtype=dt), requires_grad=False) if bias else None
    def forward(self, x):
        shp = x.shape; x2 = x.reshape(-1, self.K)
        xbar = x2.float().mean(-1, keepdim=True).to(dt)
        y = torch.addmm(self.b, self.pro(x2.contiguous()), self.W.t()) if self.b is not None else self.pro(x2.contiguous()) @ self.W.t()
        y = y + xbar * self.w1
        return y.view(*shp[:-1], self.N)


class W4Only(nn.Module):
    """4-bit GEMM + CentriQ-rank side branch, NO prologue (upper bound for the 4-bit path)."""
    def __init__(self, K, N, bias=True):
        super().__init__(); self.N = N
        rank = int(math.ceil((K / 64 + 2) / 16) * 16)
        self.w4 = SVDQW4A4Linear(K, N, rank=rank, bias=bias, precision=a.precision, torch_dtype=dt, device=dev)
        with torch.no_grad():
            for p in list(self.w4.parameters()) + list(self.w4.buffers()):
                if p.dtype in (torch.bfloat16, torch.float16, torch.float32): p.normal_(0, 0.02)
                elif not p.dtype.is_floating_point: p.random_(0, 15)
    def forward(self, x):
        shp = x.shape
        return self.w4(x.reshape(1, -1, shp[-1])).view(*shp[:-1], self.N)





import re
def group_shared(tg):
    """Linears that read the same tensor: double blocks q/k/v (+ context q/k/v), single blocks q/k/v + proj_mlp."""
    groups = {}
    for name in tg:
        m = re.match(r"(single_transformer_blocks\.\d+)\.(attn\.to_[qkv]|proj_mlp)$", name)
        key = (m.group(1) + ".in") if m else None
        if key is None:
            m = re.match(r"(.*\.attn)\.(to_[qkv])$", name); key = (m.group(1) + ".qkv") if m else None
        if key is None:
            m = re.match(r"(.*\.attn)\.add_[qkv]_proj$", name); key = (m.group(1) + ".addqkv") if m else None
        groups.setdefault(key or name, []).append(name)
    return groups

STATS = {"compute": 0, "reuse": 0}
class _Group:
    def __init__(self, fused): self.fused, self.x, self.y, self.n, self.used = fused, None, None, 0, 0
class SharedMember(nn.Module):
    def __init__(self, g, off, n):
        super().__init__(); self.g, self.off, self.n = g, off, n
        self.add_module("fused_" + str(off), g.fused) if off == 0 else None
    def forward(self, x):
        g = self.g
        if g.x is not x:
            g.x, g.y, g.used = x, g.fused(x), 0; STATS["compute"] += 1
        else:
            STATS["reuse"] += 1
        out = g.y[..., self.off:self.off + self.n]
        g.used += 1
        if g.used == g.n:          # last member read it: release, so no activation outlives its layer
            g.x = g.y = None
        return out
def install_group(root, names, mods, fused):
    g = _Group(fused); g.n = len(names); off = 0
    for n_, m in zip(names, mods):
        _set_submodule(root, n_, SharedMember(g, off, m.out_features)); off += m.out_features

class BF16Fused(nn.Module):
    def __init__(self, mods):
        super().__init__()
        self.W = nn.Parameter(torch.cat([m.weight for m in mods], 0).detach(), requires_grad=False)
        self.b = nn.Parameter(torch.cat([m.bias for m in mods], 0).detach(), requires_grad=False) if mods[0].bias is not None else None
    def forward(self, x):
        return nn.functional.linear(x, self.W, self.b)

if a.mode == "bf16" and a.share:
    from rods_a.wrapper import select_act_targets, _set_submodule
    tg = select_act_targets(pipe.transformer, model_type="flux", group_size=64)
    for key, names in group_shared(tg).items():
        if len(names) > 1:
            mods = [tg[n] for n in names]; install_group(pipe.transformer, names, mods, BF16Fused(mods))
    del tg; import gc; gc.collect(); torch.cuda.empty_cache()

if a.mode in ("centriq", "svdqswap", "bf16rot", "w4only"):
    from rods_a.wrapper import select_act_targets, _set_submodule
    tg = select_act_targets(pipe.transformer, model_type="flux", group_size=64)
    cls = {"centriq": CentriQLinear, "svdqswap": SVDQSwap, "bf16rot": BF16Rot, "w4only": W4Only}[a.mode]
    if a.share:
        groups = group_shared(tg)
        for key, names in groups.items():
            mods = [tg[n] for n in names]
            fused = cls(mods[0].in_features, sum(m.out_features for m in mods), bias=mods[0].bias is not None)
            install_group(pipe.transformer, names, mods, fused)
        print("[share] groups:", {k: len(v) for k, v in list(groups.items())[:4]}, "...", len(groups), "groups", flush=True)
    else:
        for name, m in tg.items():
            _set_submodule(pipe.transformer, name, cls(m.in_features, m.out_features, bias=m.bias is not None))
    n_rep = len(tg); del tg
    import gc; gc.collect(); torch.cuda.empty_cache()
    print(f"[centriq] replaced {n_rep} linears", flush=True)

ev = {}
def wrap(mod, name):
    f = mod.forward
    def g(*args, **kw):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); out = f(*args, **kw); e.record(); ev.setdefault(name, []).append((s, e)); return out
    mod.forward = g
wrap(pipe.transformer, "transformer")
prompt = "a photo of an astronaut riding a horse on the moon, highly detailed"
def run(seed):
    ev.clear(); torch.cuda.synchronize(); t0 = time.perf_counter()
    pipe([prompt] * a.bs, num_inference_steps=4, guidance_scale=0.0, height=1024, width=1024,
         generator=torch.Generator("cuda").manual_seed(seed))
    torch.cuda.synchronize(); return time.perf_counter() - t0, sum(s.elapsed_time(e) for s, e in ev["transformer"]) / 1000
for i in range(a.warmup): run(i)
torch.cuda.reset_peak_memory_stats()
W, Tr = zip(*[run(100 + i) for i in range(a.n)])
res = {"mode": a.mode, "bs": a.bs, "gpu": torch.cuda.get_device_name(0), "n": a.n,
       "end_to_end_s": (st.mean(W), st.stdev(W)), "transformer_s": (st.mean(Tr), st.stdev(Tr)),
       "imgs_per_s": a.bs / st.mean(W), "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
       "share": a.share, "prologue": a.prologue, "precision": a.precision, "single_proj_out": os.environ.get("CENTRIQ_QUANT_SINGLE_PROJ_OUT") == "1", "fused": a.fused, "share_stats": STATS}
print(json.dumps(res, indent=1))
if a.json: json.dump(res, open(a.json, "w"), indent=1)
