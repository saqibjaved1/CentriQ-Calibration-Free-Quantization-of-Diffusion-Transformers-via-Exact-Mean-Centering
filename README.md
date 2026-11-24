# CentriQ: Calibration-Free Quantization of Diffusion Transformers via Exact Mean Centering

**Nataša Jovanović<sup>1,2</sup>, Mathieu Salzmann<sup>2</sup>, Saqib Javed<sup>2</sup>**

<sup>1</sup>Tenstorrent · <sup>2</sup>EPFL

[![Website](https://img.shields.io/badge/Project-Website-87CEEB)](https://saqibjaved1.github.io/CentriQ-Calibration-Free-Quantization-of-Diffusion-Transformers-via-Exact-Mean-Centering/)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-FFF933)](https://arxiv.org/abs/2610.06260) [![Code](https://img.shields.io/badge/GitHub-Code-181717?logo=github&logoColor=white)](https://github.com/saqibjaved1/CentriQ-Calibration-Free-Quantization-of-Diffusion-Transformers-via-Exact-Mean-Centering)

CentriQ quantizes the weights and activations of diffusion transformers to 4 bits **without any
calibration data**. Adaptive LayerNorm adds a per-token mean to DiT activations, and a Hadamard
rotation piles that mean onto one coordinate, which then sets the 4-bit range of the whole token.
CentriQ removes the mean before the rotation and adds it back exactly through a full-precision
rank-1 term:

```
y_t = Ŵ_R · Q_a( H P (x_t − x̄_t·1) )  +  x̄_t · W1  +  b
```

- **Activations:** per-token centering, a fixed factorized Hadamard rotation `H = H_m ⊗ H_2^k`
  (zero-padded when the width needs it), and a closed-form per-token scale.
- **Weights:** grouped INT4 (g = 64) fitted offline under a robust ℓp objective (p = 0.7), which
  we call RoDS.
- Nothing is fitted to data. A new checkpoint converts in minutes.

At W4A4, CentriQ matches calibrated SVDQuant on PixArt-Σ, SANA-1.6B and FLUX.1-schnell. On
FLUX.1-schnell it is the first calibration-free method to retain usable image quality at 2-bit
activations (W4A2), and it outperforms the strongest reported calibration-free method at 2-bit
weights (W2A4).

## Install

```bash
git clone https://github.com/saqibjaved1/CentriQ-Calibration-Free-Quantization-of-Diffusion-Transformers-via-Exact-Mean-Centering.git centriq && cd centriq
conda create -n centriq python=3.11 -y && conda activate centriq
pip install -r requirements.txt
```

Tested with Python 3.11, PyTorch 2.10 / CUDA 12.8 and diffusers 0.36. A GPU with 80 GB is
needed for FLUX.1-schnell; PixArt-Σ and SANA-1.6B fit in less. FLUX.1-schnell is gated on the
Hugging Face Hub, so log in with `huggingface-cli login` first.

## Quickstart: one image, no dataset

```bash
python scripts/quickstart.py --model flux   --prompt "a lighthouse at dusk, oil painting"
python scripts/quickstart.py --model pixart --act_bits 2     # W4A2
python scripts/quickstart.py --model flux   --bf16            # 16-bit reference
```

From Python:

```python
from centriq import load_and_quantize
pipe = load_and_quantize("flux")             # pixart | flux | sana, paper configuration
image = pipe("a red fox in the snow", num_inference_steps=4, guidance_scale=0.0).images[0]
```

Quantization here is *simulated*: weights and activations are rounded to the 4-bit grid and the
matrix products run in 16 bits, which is how all image-quality numbers in the paper are
measured. The fused INT4 kernels used for latency are in [`latency/`](latency/README.md).

## Reproduce the paper

The evaluation uses the MJHQ-5k split of SVDQuant. It is downloaded automatically
(about 11.7 GB) into `$CENTRIQ_DATA/mjhq5k` (default `./data`). Outputs go to
`$CENTRIQ_OUTPUTS` (default `./outputs`).

| Script | Paper result |
|---|---|
| `bash reproduce/table1_w4a4.sh` | Main W4A4 table, MJHQ rows: FID, ImageReward, PSNR, LPIPS for the 16-bit model and CentriQ on PixArt-Σ, FLUX.1-schnell, SANA-1.6B |
| `bash reproduce/table_ablation.sh` | Component ablation on FLUX.1-schnell at W4A4, W4A2, W2A4 |
| `bash reproduce/table_lowbit.sh` | CentriQ at W4A2 and W2A4 on FLUX.1-schnell |
| `bash reproduce/smoke_test.sh` | 2 prompts per model, to check the setup end to end |
| `python scripts/collect_results.py outputs/` | Prints a table of every finished run |

Set `N=1000` for a faster, noisier run (the paper uses `N=5000`). Single runs can also be
launched directly, for example:

```bash
python scripts/rods_a_w4a4_eval.py --config-name rods_a_w4a4_eval_flux \
  quant=rods_a_w4a4_fact_flux3 num_samples=5000 compute_fid=true compute_ir=true \
  capture_sqnr=false ablation_tag=centriq
```

| Model | Eval config | Quant config | Rotation |
|---|---|---|---|
| PixArt-Σ | `rods_a_w4a4_eval` | `rods_a_w4a4_pad` | zero-pad to 2048 / 8192, Sylvester |
| FLUX.1-schnell | `rods_a_w4a4_eval_flux` | `rods_a_w4a4_fact_flux3` | native `H_12 ⊗ H_256`, `H_12 ⊗ H_1024` |
| SANA-1.6B | `rods_a_w4a4_eval_sana` | `rods_a_w4a4_fact_sana5` | 2240 → 2304 = `H_36 ⊗ H_64` |

Bit-widths are command-line overrides: `quant.nbits=2` for 2-bit weights,
`quant.act_bits=2` for 2-bit activations. The clip percentile (0.9995), group size (64) and
ℓp exponent (0.7) are shared by all models and layers.

FID uses [clean-fid](https://github.com/GaParmar/clean-fid) against the MJHQ-5k reference
images, ImageReward uses the official model, and PSNR / LPIPS (AlexNet, 256 px) compare each
image with the 16-bit image from the same prompt and seed. FID at 5k samples has a noise floor
of a few tenths; expect small differences from the paper on other GPUs or library versions.

## Repository layout

```
centriq.py              public API: load_and_quantize, quantize_transformer
rods_a/                 activation path: centering, Hadamard rotation, per-token quantizer, W4A4 layer
rods/                   RoDS weight quantizer (robust ℓp grouped INT4)
core/                   model registry, pipeline loading, MJHQ data, metrics, Hadamard matrices
core/sinq/              quantizer internals vendored from SINQ (Apache-2.0)
baselines/              RTN / NF4 weight baselines
scripts/                rods_a_w4a4_eval.py (generation + metrics), paired_image_metrics.py, quickstart.py
configs/                Hydra configs (eval, model, paths, quant)
reproduce/              one script per paper table
latency/                fused INT4 engine for the latency table (needs Nunchaku)
tests/                  unit tests (pytest tests/)
docs/                   project page (index.html, assets/)
tools/page/             scripts used to build the project page figures and comparisons
```

Module names (`rods_a`, `rods`) follow our internal naming: RoDS is the weight quantizer and
`rods_a` is its activation-side companion, which together form CentriQ.

## Tests

```bash
pytest -q tests
```

## Citation

```bibtex
@article{jovanovic2026centriq,
  title   = {CentriQ: Calibration-Free Quantization of Diffusion Transformers via Exact Mean Centering},
  author  = {Jovanovi{\'c}, Nata{\v{s}}a and Salzmann, Mathieu and Javed, Saqib},
  journal = {arXiv preprint arXiv:2610.06260},
  year    = {2026}
}
```

## Acknowledgements and license

`core/sinq/` contains code from [SINQ](https://github.com/huawei-csl/SINQ) and
`core/hadamard/vendored.py` contains Hadamard matrices from
[DeepCompressor](https://github.com/mit-han-lab/deepcompressor), both under the Apache-2.0
license; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The latency harness builds on
[Nunchaku](https://github.com/nunchaku-tech/nunchaku). We thank the authors of SVDQuant for
releasing their MJHQ-5k evaluation split.

License: see [LICENSE](LICENSE).
