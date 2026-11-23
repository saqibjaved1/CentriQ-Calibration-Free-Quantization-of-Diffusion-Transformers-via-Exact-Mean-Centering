"""
Parity-probe step-tagging tests (Contract v3, R4.2).

Regression for the Step-5 incident: the live run tagged all 8960 cells "step 0"
(advancing the step from a post-step sampler callback that never propagated, so
_CURRENT_STEP stayed 0 while every step recorded). The fix advances the step from a
transformer forward_pre_hook -- once per denoise step, BEFORE that step's forwards --
so tags are the TRUE step index and honor enabled_steps. These tests pin both the
probe contract and the hook mechanism, on CPU.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn

from rods_a import parity_probe


def test_recorded_steps_match_enabled_steps():
    """Honor enabled_steps with correct per-step tags: recording only the enabled
    steps, each cell tagged with its true step index (no off-by-one, no stuck step)."""
    enabled = [0, 9, 19]
    parity_probe.reset(enabled_steps=enabled, capture_violations=False, gate=1.5e-3)
    y = torch.randn(4, 8)
    for step in range(20):
        parity_probe.set_step(step)
        for layer in ("attn.to_q", "ff.net.2"):       # two layers per step
            parity_probe.record(layer, "role", y, y.clone())
    recs = parity_probe.get_records()
    parity_probe.disarm()

    assert sorted(set(r["step"] for r in recs)) == enabled, "recorded steps != config"
    assert len(recs) == len(enabled) * 2, "wrong cell count (subset not honored)"
    # regression signature of the step-0 pinning bug: NOT everything pinned to step 0.
    assert any(r["step"] != 0 for r in recs), "all cells tagged step 0"


def test_step_pre_hook_advances_true_index():
    """The forward_pre_hook advances the step once per transformer call, producing the
    true 0..N-1 indices across a denoise loop, reset per prompt by begin_run_steps()."""
    parity_probe.reset(enabled_steps=None, capture_violations=False, gate=1.5e-3)
    transformer = nn.Identity()
    seen = []

    def _spy(mod, args):
        seen.append(parity_probe.current_step())   # step BEFORE the forward runs

    h_advance = transformer.register_forward_pre_hook(parity_probe.make_step_pre_hook())
    h_spy = transformer.register_forward_pre_hook(_spy)
    try:
        for _ in range(2):                          # two prompts
            parity_probe.begin_run_steps()
            base = len(seen)
            for _ in range(20):                     # 20 denoise steps
                transformer(torch.zeros(1))
            assert seen[base:] == list(range(20)), "steps not tagged 0..19 in order"
    finally:
        h_advance.remove()
        h_spy.remove()
        parity_probe.disarm()
