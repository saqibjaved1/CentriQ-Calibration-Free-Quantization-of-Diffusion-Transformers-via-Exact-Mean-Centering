#!/usr/bin/env bash
# Sub-4-bit CentriQ on FLUX.1-schnell (simulated quantization): W4A2 and W2A4.
source "$(dirname "$0")/common.sh"
m=flux
run_arm $m fp16 quant_enabled=false
run_arm $m centriq_w4a2 quant.act_bits=2
run_arm $m centriq_w2a4 quant.nbits=2
paired  $m centriq_w4a2
paired  $m centriq_w2a4
python "$ROOT/scripts/collect_results.py" "$OUT"
