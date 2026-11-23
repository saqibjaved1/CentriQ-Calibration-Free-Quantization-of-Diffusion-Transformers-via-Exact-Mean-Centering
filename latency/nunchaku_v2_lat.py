"""FLUX.1-schnell latency inside Nunchaku's own engine (NunchakuFluxTransformer2DModelV2, INT4, A100).
Identical engine for both methods: same fused QKV+RMSNorm+RoPE GEMMs, same attention, same quantized layer set
(incl. adaLN-free single-block proj_out split). Only the quantizer differs:
  --mode svdq     : SVDQuant as released (smoothing + rank-32 low-rank branch; fc2 input quantized in fc1's epilogue)
  --mode centriq  : CentriQ: per-token centering + factorized Hadamard (one fused Triton kernel) before every
                    quantize call, CentriQ's side-branch rank (zero-point group sums + rank-1 mean: ceil((K/64+2)/16)*16),
                    signed activations; MLP = fc1 -> GELU -> CentriQ prologue -> fc2 (no next-layer fusion possible)
  --mode bf16     : diffusers BF16 transformer (same pipeline)
Weights are random (latency only). Usage: python latency/nunchaku_v2_lat.py --mode centriq --n 20 --json out.json
"""
import argparse, json, math, os, statistics as st, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, choices=["svdq", "centriq", "bf16"])
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--warmup", type=int, default=3)
ap.add_argument("--bs", type=int, default=1)
ap.add_argument("--no_gelu_fuse", action="store_true", help="CentriQ MLP: separate GELU pass instead of GELU-on-load in the prologue")
ap.add_argument("--rank32", action="store_true", help="ablation: SVDQuant-size side branch")
ap.add_argument("--no_prologue", action="store_true", help="ablation: skip centering+rotation")
ap.add_argument("--side_prologue", action="store_true", help="CentriQ side branch (group sums + mean) emitted by the prologue kernel; quantize kernel runs a rank-16 pass only")
ap.add_argument("--paper_layers", action="store_true", help="CentriQ paper layer set: adaLN linears and single-block proj_out (attn.to_out, mlp_fc2) in BF16")
ap.add_argument("--share_quant", action="store_true", help="reuse the whole quantized activation (codes, scales, low-rank input) across layers reading the same input: CentriQ has no per-layer smoothing and a fixed down-projection")
ap.add_argument("--profile", default=None, help="write a kernel-level breakdown of the transformer (JSON) to this path")
ap.add_argument("--compile", action="store_true", help="BF16 control: torch.compile the transformer (fuses norm/RoPE/elementwise like an optimized engine)")
ap.add_argument("--no_share", action="store_true", help="CentriQ: recompute the prologue for every layer")
ap.add_argument("--json", default=None)
a = ap.parse_args()

from flux_random import build_random_pipe as _build_random_pipe
from diffusers import FluxPipeline
def build_random_pipe(transformer=None):
    # FLUX_RANDOM=1: architecture from configs with random weights (no gated checkpoint needed; latency only).
    # Otherwise: the real checkpoint (FLUX_PATH or the HF id), with the given quantized transformer swapped in.
    if os.environ.get("FLUX_RANDOM") == "1":
        return _build_random_pipe(transformer=transformer)
    mid = os.environ.get("FLUX_PATH", "black-forest-labs/FLUX.1-schnell")
    kw = {} if transformer is None else {"transformer": transformer}
    return FluxPipeline.from_pretrained(mid, torch_dtype=torch.bfloat16, **kw).to("cuda")
STATS = {"prologue_calls": 0, "prologue_reuse": 0}

if a.mode == "bf16":
    pipe = build_random_pipe()
else:
    from nunchaku import NunchakuFluxTransformer2DModelV2
    from nunchaku.models.linear import SVDQW4A4Linear
    tr = NunchakuFluxTransformer2DModelV2.from_pretrained(
        "nunchaku-tech/nunchaku-flux.1-schnell/svdq-int4_r32-flux.1-schnell.safetensors")
    if a.paper_layers:
        from nunchaku.models.linear import AWQW4A16Linear
        def _bf16(mod):
            lin = torch.nn.Linear(mod.in_features, mod.out_features, bias=getattr(mod, "bias", None) is not None,
                                  device="cuda", dtype=torch.bfloat16)
            with torch.no_grad(): lin.weight.normal_(0, 0.02)
            return lin
        n_bf = 0
        for name, m in list(tr.named_modules()):
            for cn, ch in list(m.named_children()):
                full = f"{name}.{cn}" if name else cn
                single_po = full.startswith("single_transformer_blocks.") and (full.endswith("attn.to_out") or full.endswith("mlp_fc2"))
                if isinstance(ch, AWQW4A16Linear) or (single_po and isinstance(ch, SVDQW4A4Linear)):
                    setattr(m, cn, _bf16(ch)); n_bf += 1
        print(f"[paper_layers] {n_bf} layers -> BF16", flush=True)
        import nunchaku.models.transformers.transformer_flux_v2 as _tf2
        _orig_mlp = _tf2.fused_gelu_mlp
        def _mlp(x, fc1, fc2, pad_size=256):
            if isinstance(fc2, SVDQW4A4Linear):
                return _orig_mlp(x, fc1, fc2, pad_size)
            return fc2(F.gelu(fc1(x), approximate="tanh"))
        _tf2.fused_gelu_mlp = _mlp
    if a.mode == "centriq":
        from centriq_prologue import FactPrologue
        import nunchaku.models.transformers.transformer_flux_v2 as tf2
        PRO = {}
        cache = {"x": None, "y": None, "k": None, "side": None}
        from nunchaku.ops.quantize import svdq_quantize_w4a4_act_fuse_lora_cuda as _q

        def key(x): return (x.data_ptr(), x.numel(), x.shape[-1], x.stride(-1), x._version)   # same storage, any 2D/3D view
        def prologue(x, gelu=False, R=None, pad_size=256):
            # same storage/shape/stride/version as the previous input (we hold a reference, so the storage cannot be
            # reused by another tensor) => identical content => reuse the centered + rotated activation (+ side vector)
            if not gelu and not a.no_share and cache["x"] is not None and cache["k"] == key(x) and \
                    (R is None or (cache["side"] is not None and cache["side"].shape[1] == R)):
                STATS["prologue_reuse"] += 1; return cache["y"], cache["side"]
            K = x.shape[-1]
            if K not in PRO: PRO[K] = FactPrologue(K)
            x2 = x.reshape(-1, K).contiguous(); M = x2.shape[0]
            side = None if R is None else torch.zeros((M + pad_size - 1) // pad_size * pad_size, R, device=x.device, dtype=torch.float32)
            y = PRO[K](x2, gelu=gelu, side=side).view(*x.shape[:-1], PRO[K].K)
            if not gelu:
                cache["x"], cache["y"], cache["k"], cache["side"] = x, y, key(x), side
            STATS["prologue_calls"] += 1
            return y, side

        def quant_with_side(self, xin, gelu=False, pad_size=256):
            y, side = prologue(xin, gelu=gelu, R=self.rank, pad_size=pad_size)
            qx, asc, _ = _q(y.reshape(-1, y.shape[-1]), lora_down=self._pd16, smooth=self.smooth_factor, fp4=False, pad_size=pad_size)
            return qx, asc, side

        orig_quantize = SVDQW4A4Linear.quantize
        qcache = {"k": None, "v": None, "x": None}
        def cq_quantize(self, x, pad_size=256):
            if a.share_quant and not a.no_prologue:
                k = (key(x), self.in_features, self.rank)
                if qcache["k"] == k:
                    STATS["quant_reuse"] = STATS.get("quant_reuse", 0) + 1; return qcache["v"]
                v = orig_quantize(self, prologue(x)[0], pad_size=pad_size)
                qcache["k"], qcache["v"], qcache["x"] = k, v, x
                return v
            if a.side_prologue and not a.no_prologue:
                return quant_with_side(self, x, pad_size=pad_size)
            return orig_quantize(self, x if a.no_prologue else prologue(x)[0], pad_size=pad_size)

        PD = {}
        n_lin = 0
        for name, m in tr.named_modules():
            if isinstance(m, SVDQW4A4Linear):
                K, N = m.in_features, m.out_features
                R = 32 if a.rank32 else int(math.ceil((K / 64 + 2) / 16) * 16)   # CentriQ side branch (zero-point + rank-1)
                if a.share_quant:   # CentriQ's down-projection is the fixed group-sum/mean matrix: one per (K, R)
                    PD.setdefault((K, R), torch.nn.Parameter(torch.randn(K, R, device="cuda", dtype=m.proj_down.dtype) * 0.02, requires_grad=False))
                    m.proj_down = PD[(K, R)]
                else:
                    m.proj_down = torch.nn.Parameter(torch.randn(K, R, device="cuda", dtype=m.proj_down.dtype) * 0.02, requires_grad=False)
                m.proj_up = torch.nn.Parameter(torch.randn(N, R, device="cuda", dtype=m.proj_up.dtype) * 0.02, requires_grad=False)
                m.rank = R
                if a.side_prologue:   # quantize kernel only needs a minimal rank-16 pass; side vector comes from the prologue
                    m._pd16 = torch.randn(K, 16, device="cuda", dtype=m.proj_down.dtype) * 0.02
                with torch.no_grad(): m.smooth_factor.fill_(1.0)    # no smoothing in CentriQ (kernel still multiplies)
                m.act_unsigned = False                              # centered activations are signed
                m.quantize = cq_quantize.__get__(m, SVDQW4A4Linear)
                n_lin += 1

        def cq_gelu_mlp(x, fc1, fc2, pad_size=256):
            if not isinstance(fc2, SVDQW4A4Linear):          # paper layer set: single-block fc2 stays BF16
                return fc2(F.gelu(fc1(x), approximate="tanh"))
            if a.no_gelu_fuse or a.no_prologue:
                return fc2(F.gelu(fc1(x), approximate="tanh"))
            h = fc1(x)                                               # pre-activation, BF16
            B, S, _ = h.shape
            if a.side_prologue:
                q = quant_with_side(fc2, h.reshape(B * S, -1), gelu=True, pad_size=pad_size)
            else:
                q = orig_quantize(fc2, prologue(h.reshape(B * S, -1), gelu=True)[0], pad_size=pad_size)
            out = torch.empty(q[0].shape[0], fc2.out_features, dtype=h.dtype, device=h.device)
            out = fc2.forward_quant(*q, out)
            return out[: B * S].reshape(B, S, -1)
        tf2.fused_gelu_mlp = cq_gelu_mlp
        print(f"[centriq] patched {n_lin} SVDQW4A4Linear layers", flush=True)
    pipe = build_random_pipe(transformer=tr)

pipe.set_progress_bar_config(disable=True)
if a.compile:
    pipe.transformer.forward = torch.compile(pipe.transformer.forward)
ev = {}
f0 = pipe.transformer.forward
def g(*args, **kw):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record(); out = f0(*args, **kw); e.record(); ev.setdefault("tr", []).append((s, e)); return out
pipe.transformer.forward = g
CALLS = []
def g_cap(*args, **kw):
    if len(CALLS) < 4: CALLS.append((args, kw))
    return g(*args, **kw)
pipe.transformer.forward = g_cap
prompt = "a photo of an astronaut riding a horse on the moon, highly detailed"
def run(seed):
    ev.clear(); torch.cuda.synchronize(); t0 = time.perf_counter()
    pipe([prompt] * a.bs, num_inference_steps=4, guidance_scale=0.0, height=1024, width=1024,
         generator=torch.Generator("cuda").manual_seed(seed))
    torch.cuda.synchronize(); return time.perf_counter() - t0, sum(s.elapsed_time(e) for s, e in ev["tr"]) / 1000
for i in range(a.warmup): run(i)
torch.cuda.reset_peak_memory_stats()
W, Tr = zip(*[run(100 + i) for i in range(a.n)])
res = {"mode": a.mode, "engine": "nunchaku_v2" if a.mode != "bf16" else "diffusers", "bs": a.bs, "n": a.n,
       "gpu": torch.cuda.get_device_name(0), "end_to_end_s": (st.mean(W), st.stdev(W)),
       "transformer_s": (st.mean(Tr), st.stdev(Tr)), "imgs_per_s": a.bs / st.mean(W),
       "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9, "compile": a.compile, "share": not a.no_share, "gelu_fused": not a.no_gelu_fuse, "rank32": a.rank32, "share_quant": a.share_quant, "paper_layers": a.paper_layers, "side_prologue": a.side_prologue, "no_prologue": a.no_prologue, "stats": STATS}
print(json.dumps(res, indent=1))
if a.json: json.dump(res, open(a.json, "w"), indent=1)

if a.profile:
    # Kernel-level decomposition of the transformer: replay the 4 captured denoising-step calls (same inputs the
    # pipeline used), 3 repetitions, under torch.profiler; device time summed per kernel, grouped by category.
    import re
    from torch.profiler import profile, ProfilerActivity
    REPS = 3
    def tr_pass():
        for args, kw in CALLS: f0(*args, **kw)
    for _ in range(2): tr_pass()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(REPS): tr_pass()
        torch.cuda.synchronize()
    cats = [("CentriQ prologue (center + rotate)", r"fact_prologue|center_rotate"),
            ("4-bit activation quantize", r"quantize"),
            ("4-bit GEMM (+ fused epilogue)", r"gemm_w4a4|w4a4.*gemm|gemm.*w4a4"),
            ("4-bit weight-only GEMM (W4A16)", r"w4a16|awq|gemv"),
            ("Attention", r"flash|fmha|attention|attn|sdpa"),
            ("BF16 GEMM (cuBLAS / CUTLASS)", r"gemm|cutlass|xmma|sm80_|ampere_|cublas|s16816"),
            ("Norm / RoPE / GELU", r"norm|rope|rotary|gelu"),
            ("Elementwise / copy / other", r".*")]
    agg, kern = {c: 0.0 for c, _ in cats}, {}
    for e in prof.key_averages():
        t = getattr(e, "device_time_total", None)
        if t is None: t = getattr(e, "cuda_time_total", 0)
        if not t or e.key.startswith("ProfilerStep"): continue
        if getattr(e, "device_type", None) is not None and "CUDA" not in str(e.device_type): continue
        name = e.key
        for c, rx in cats:
            if re.search(rx, name, re.I):
                agg[c] += t / 1e3 / REPS; kern[name] = (c, round(t / 1e3 / REPS, 2)); break
    out = {"mode": a.mode, "compile": a.compile, "paper_layers": a.paper_layers, "share_quant": a.share_quant, "ms_per_image": {k: round(v, 2) for k, v in agg.items()},
           "kernel_sum_ms": round(sum(agg.values()), 2), "timed_transformer_ms": round(st.mean(Tr) * 1e3, 2),
           "top_kernels": sorted(kern.items(), key=lambda kv: -kv[1][1])[:40]}
    print(json.dumps(out, indent=1))
    json.dump(out, open(a.profile, "w"), indent=1)
