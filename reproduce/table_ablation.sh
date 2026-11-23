#!/usr/bin/env bash
# Component ablation on FLUX.1-schnell (MJHQ) at W4A4, W4A2 and W2A4.
# Rows: no transform / + rotation / + centering / + percentile clip (= CentriQ).
# NOTE: the "without percentile clip" rows use the per-token absmax scale; confirm with the authors.
source "$(dirname "$0")/common.sh"
m=flux
run_arm $m fp16 quant_enabled=false
for bits in "w4a4 quant.nbits=4 quant.act_bits=4" "w4a2 quant.nbits=4 quant.act_bits=2" "w2a4 quant.nbits=2 quant.act_bits=4"; do
  set -- $bits; tag=$1; shift
  run_arm $m ${tag}_none     "$@" quant.rotate_act=false quant.rotation=pad quant.center_act_mean=false quant.scale_estimator=absmax
  run_arm $m ${tag}_rot      "$@" quant.center_act_mean=false quant.scale_estimator=absmax
  run_arm $m ${tag}_rot_cen  "$@" quant.scale_estimator=absmax
  run_arm $m ${tag}_centriq  "$@"
done
python "$ROOT/scripts/collect_results.py" "$OUT"
