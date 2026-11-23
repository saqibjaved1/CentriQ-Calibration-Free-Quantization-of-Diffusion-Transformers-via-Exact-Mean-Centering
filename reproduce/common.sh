#!/usr/bin/env bash
# Shared helpers. Every run writes $OUT/<model>/<arm>/{images/,results.json}.
#   N=5000 (paper) | N=16 (smoke test)     DEVICE=cuda:0
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
N=${N:-5000}
DEVICE=${DEVICE:-cuda:0}
OUT=${CENTRIQ_OUTPUTS:-$ROOT/outputs}
export WANDB_MODE=${WANDB_MODE:-disabled}

eval_cfg() { case $1 in pixart) echo rods_a_w4a4_eval;; flux) echo rods_a_w4a4_eval_flux;; sana) echo rods_a_w4a4_eval_sana;; esac; }
quant_cfg() { case $1 in pixart) echo rods_a_w4a4_pad;; flux) echo rods_a_w4a4_fact_flux3;; sana) echo rods_a_w4a4_fact_sana5;; esac; }

# run_arm <model> <arm-name> [extra hydra overrides...]
run_arm() {
  local model=$1 arm=$2; shift 2
  python "$ROOT/scripts/rods_a_w4a4_eval.py" --config-name "$(eval_cfg $model)" \
    quant="$(quant_cfg $model)" output_dir="$OUT/$model" ablation_tag="$arm" \
    num_samples=$N device=$DEVICE compute_fid=true compute_ir=true \
    capture_sqnr=false contact_sheet=false use_wandb=false "$@"
}

# paired <model> <arm>: PSNR / LPIPS of <arm> against the 16-bit arm (same prompt and seed)
paired() {
  local model=$1 arm=$2
  python "$ROOT/scripts/paired_image_metrics.py" --dir_a "$OUT/$model/$arm/images" \
    --dir_b "$OUT/$model/fp16/images" --n $N --device $DEVICE --out "$OUT/$model/$arm/paired_vs_fp16.json"
}
