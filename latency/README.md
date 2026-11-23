# Latency: CentriQ in Nunchaku's fused 4-bit FLUX engine

These scripts time CentriQ and SVDQuant inside the same Nunchaku engine
(`NunchakuFluxTransformer2DModelV2`), so the two methods differ only in the quantizer.
They reproduce the latency table of the paper (FLUX.1-schnell, 1024², 4 steps, batch 1,
CUDA graphs off).

| File | What it does |
|---|---|
| `nunchaku_v2_lat.py` | Matched latency, `--mode bf16 \| svdq \| centriq` |
| `centriq_prologue.py` | Fused Triton kernel: per-token centering + factorized Hadamard rotation (`FactPrologue`) |
| `centriq_blackwell2.py` | Kernel correctness check (`--mode check`) and per-layer timing (`--mode layers`) |
| `flux_random.py` | FLUX built from configs with random weights (only when `FLUX_RANDOM=1`) |

## Extra requirements

Nunchaku (`nunchaku==1.3.0.dev20260306`, built for your torch/CUDA), and a C compiler for Triton
(set `CC` if none is on the path). INT4 GEMMs need an Ampere/Ada/Hopper GPU (we used an A100);
Blackwell GPUs use Nunchaku's NVFP4 GEMMs instead.

## 1. Check the kernel is exact (1 min)

```bash
python latency/centriq_blackwell2.py --mode check --precision int4
```

Expected relative error against the reference rotation `H_12 ⊗ H_q`: about 0.003 (K=3072)
and 0.004 (K=12288), i.e. bf16 rounding.

## 2. Matched latency

```bash
python latency/nunchaku_v2_lat.py --mode bf16 --n 20 --json bf16.json
python latency/nunchaku_v2_lat.py --mode bf16 --compile --n 20 --json bf16_compile.json
python latency/nunchaku_v2_lat.py --mode svdq --paper_layers --n 20 --json svdq.json
python latency/nunchaku_v2_lat.py --mode centriq --paper_layers --share_quant --n 20 --json centriq.json
```

Each prints `transformer_s` and `end_to_end_s` as (mean, std) after 3 warm-up images.
Add `--n 3 --profile out.json` for a per-kernel breakdown with `torch.profiler`.

## What `--mode centriq` changes in the engine

1. Every `SVDQW4A4Linear.quantize(x)` first runs CentriQ's fused prologue (centering + rotation).
2. The 16-bit side branch carries CentriQ's rank-1 mean term and zero-point correction;
   its rank is `ceil((K/64 + 2)/16)*16` (64 for K=3072, 208 for K=12288).
3. Smoothing is disabled and activations are signed.
4. `--share_quant` quantizes each activation once for all layers that read it
   (exact for CentriQ, which has no per-layer smoothing).
5. `--paper_layers` keeps adaLN and the single-block `proj_out` in BF16 for both 4-bit methods.

These are timing harnesses: the side-branch weights are random, because only their shapes
affect latency. Image quality is measured with the simulated path in `scripts/`.
