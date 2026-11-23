"""
Shadow-mode parity probe (RoDS-A Step 5, I2).

A process-global coordinator so a denoising driver can tag the *current* timestep and
gate which steps the int4-vs-oracle shadow comparison records, while each wrapped
`RoDSAW4A4Linear` (which has no view of the sampler loop) writes its layer-local
metrics. Layer-local by construction: each record is one (layer, step) cell measured
on the SAME live input -- the oracle output is never fed forward (no error
propagation between the two arithmetic paths).

Usage (driver):
    parity_probe.reset(enabled_steps={0, mid, last})   # None -> record every step
    for i, t in enumerate(timesteps):
        parity_probe.set_step(i)                         # via callback_on_step_end
        ...
    records = parity_probe.get_records()
Usage (wrapper, in forward, only when shadow): if parity_probe.recording():
    parity_probe.record(name, y_int4, y_oracle)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

import torch

_CURRENT_STEP: Optional[int] = None
_ENABLED_STEPS: Optional[Set[int]] = None   # None => all steps enabled
_RECORDS: List[dict] = []
_ACTIVE: bool = False
# Violation capture (I2.3): when armed, a cell whose median_rel
# exceeds `_GATE` has its self-contained reproducer (fixture cell) stashed.
_CAPTURE: bool = False
_GATE: float = 1.5e-3
_VIOLATIONS: List[dict] = []


def reset(enabled_steps: Optional[Set[int]] = None, *,
          capture_violations: bool = False, gate: float = 1.5e-3) -> None:
    """Arm the probe for a run. `enabled_steps=None` records every step.
    `capture_violations` stashes a fixture-cell reproducer for any cell over `gate`."""
    global _CURRENT_STEP, _ENABLED_STEPS, _RECORDS, _ACTIVE, _CAPTURE, _GATE, _VIOLATIONS
    _CURRENT_STEP = None
    _ENABLED_STEPS = None if enabled_steps is None else set(int(s) for s in enabled_steps)
    _RECORDS = []
    _ACTIVE = True
    _CAPTURE = bool(capture_violations)
    _GATE = float(gate)
    _VIOLATIONS = []


def capture_armed() -> bool:
    return _CAPTURE


def gate() -> float:
    return _GATE


def stash_violation(cell: dict) -> None:
    _VIOLATIONS.append(cell)


def get_violations() -> List[dict]:
    return list(_VIOLATIONS)


def disarm() -> None:
    global _ACTIVE
    _ACTIVE = False


def set_step(i: int) -> None:
    global _CURRENT_STEP
    _CURRENT_STEP = int(i)


def current_step() -> Optional[int]:
    return _CURRENT_STEP


# --- denoise-step counter (robust step tagging, Contract v3 / R4.2) -----------
# A previous live run tagged every cell "step 0" because it advanced the step from a
# post-step sampler callback that never propagated -> _CURRENT_STEP stayed 0 while all
# steps recorded (0 ∈ enabled_steps). The robust mechanism: advance the step from a
# forward_pre_hook on the TOP-LEVEL transformer, which fires exactly once per denoise
# step BEFORE that step's layer forwards (PixArt-Σ batches CFG into one transformer
# call per step). Tags are then the TRUE step index, never dependent on the sampler API.
_STEP_COUNTER: int = -1


def begin_run_steps() -> None:
    """Reset the per-prompt denoise-step counter. Call before each pipe() invocation."""
    global _STEP_COUNTER
    _STEP_COUNTER = -1


def advance_step() -> int:
    """Increment the denoise-step counter and set it as the current step; return it.
    Intended for the transformer forward_pre_hook (fires once per step, pre-forwards)."""
    global _STEP_COUNTER
    _STEP_COUNTER += 1
    set_step(_STEP_COUNTER)
    return _STEP_COUNTER


def make_step_pre_hook():
    """A torch forward_pre_hook for the top-level diffusion transformer that advances
    the step once per call. Register on `pipe.transformer`; call begin_run_steps()
    before each pipe() run. Robust to sampler/callback API differences."""
    def _hook(module, args):
        advance_step()
        return None
    return _hook


def recording() -> bool:
    """True iff the probe is armed and the current step is in the enabled set."""
    if not _ACTIVE or _CURRENT_STEP is None:
        return False
    return _ENABLED_STEPS is None or _CURRENT_STEP in _ENABLED_STEPS


def _rel_stats(y: torch.Tensor, ref: torch.Tensor) -> Dict[str, float]:
    """median/max of |y-ref|/(|ref|+eps) -- the O2.4-v2 / Step-3 parity metric."""
    num = (y.float() - ref.float()).abs()
    den = ref.float().abs() + 1e-6
    rel = num / den
    return {
        "median_rel": float(rel.median()),
        "max_rel": float(rel.max()),
        "max_abs": float(num.max()),
    }


def record(layer: str, role: str, y_int4: torch.Tensor,
           y_oracle: torch.Tensor, Cin_pad: Optional[int] = None) -> Optional[dict]:
    """Record one (layer, current_step) int4-vs-oracle cell, returning its stats (or
    None if not recording). Layer-local: y_oracle is never propagated. `Cin_pad`
    (the contraction width K) feeds the per-K acceptance gate (contract S11): the
    kernel's fp16-fpsum epilogue floor grows ~sqrt(K), so the bound is K-indexed."""
    if not recording():
        return None
    stats = _rel_stats(y_int4, y_oracle)
    stats.update({"layer": layer, "role": role, "step": int(_CURRENT_STEP),
                  "T": int(y_int4.shape[0]), "out_features": int(y_int4.shape[1])})
    if Cin_pad is not None:
        stats["Cin_pad"] = int(Cin_pad)
    _RECORDS.append(stats)
    return stats


def get_records() -> List[dict]:
    return list(_RECORDS)
