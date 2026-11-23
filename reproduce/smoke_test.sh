#!/usr/bin/env bash
# Fast end-to-end check: 2 prompts per model, no FID (needs the MJHQ-5k prompt file, ~11.7 GB download).
# For a check without any dataset, use scripts/quickstart.py instead.
export N=${N:-2}
source "$(dirname "$0")/common.sh"
for m in ${@:-pixart flux}; do
  run_arm $m fp16 quant_enabled=false compute_fid=false compute_ir=false
  run_arm $m centriq compute_fid=false compute_ir=false
  paired  $m centriq
done
