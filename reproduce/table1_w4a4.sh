#!/usr/bin/env bash
# Table 1 (MJHQ-5k rows): 16-bit reference and CentriQ W4A4 on PixArt-Sigma, FLUX.1-schnell, SANA-1.6B.
# Usage: bash reproduce/table1_w4a4.sh [pixart flux sana]
source "$(dirname "$0")/common.sh"
for m in ${@:-pixart flux sana}; do
  run_arm $m fp16 quant_enabled=false
  run_arm $m centriq
  paired  $m centriq
done
python "$ROOT/scripts/collect_results.py" "$OUT"
