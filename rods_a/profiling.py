"""
Lightweight per-stage latency profiler for the RoDS-A W4A4 forward.

Two modes, selected at enable() time:

* "sync" (legacy default): wall-clock with torch.cuda.synchronize() at both
  section boundaries. CUDA work is synchronized so the times are correctly
  attributed (this serializes the GPU, so ABSOLUTE numbers are inflated vs a
  normal run, but the RELATIVE breakdown -- the question being asked -- is
  faithful). Consumed by scripts/rods_a_w4a4_profile.py via reset()/summary().

* "events": CUDA-event pairs recorded at section boundaries -- NO
  synchronization is added inside the bracketed region or the denoising loop.
  elapsed_time is read only in harvest(), which the caller invokes AFTER its
  own torch.cuda.synchronize() at the end of the timed unit (one image). Event
  objects are pooled and recycled across images. Consumed by
  scripts/latency_profile.py.

Disabled by default: when `enable()` has not been called the context manager
does essentially nothing (one bool check), so the production forward is
unaffected.

Step attribution (events mode): begin_image() resets the step index; the
driver bumps it via next_step() from a transformer forward_pre_hook, which
fires once per denoise step (PixArt/SANA/FLUX batch CFG into one transformer
call per step). Sections opened before the first next_step() carry step -1.

Sections constructed with events_only=True are no-ops in sync mode as well as
when disabled: they nest INSIDE existing sync sections (the pad share inside
`rotate`, the int4 pack/GEMM/epilogue split inside the `int4_gemm` lump), and
a nested synchronize would inflate the enclosing sync section's numbers.

Memory bound (events mode): worst case PixArt is ~224 layers x ~7 sections x
20 steps ~ 31k records (62k events) per image; the pool is populated during
the first (warmup) image and recycled by harvest(), so the steady state
allocates nothing.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch

_ENABLED = False
_MODE = "sync"                       # "sync" | "events"
_SECONDS: Dict[str, float] = defaultdict(float)
_COUNTS: Dict[str, int] = defaultdict(int)
_STEP = -1                           # events mode: current denoise-step index
_RECORDS: List[tuple] = []           # (name, step, ev_start, ev_end)
_POOL: List["torch.cuda.Event"] = [] # recycled timing events


def enable(flag: bool = True, mode: str = "sync") -> None:
    """Turn profiling on/off. mode="events" requires CUDA (event pairs); the
    legacy callers (`enable(True)`) keep the sync behavior unchanged."""
    global _ENABLED, _MODE
    if mode not in ("sync", "events"):
        raise ValueError(f"mode must be 'sync' or 'events', got {mode!r}")
    if flag and mode == "events" and not torch.cuda.is_available():
        raise RuntimeError("profiling mode='events' requires CUDA")
    _ENABLED = flag
    _MODE = mode


def is_enabled() -> bool:
    return _ENABLED


def mode() -> str:
    return _MODE


def reset() -> None:
    _SECONDS.clear()
    _COUNTS.clear()
    _RECORDS.clear()


def summary() -> Tuple[Dict[str, float], Dict[str, int]]:
    """Return (seconds_by_stage, calls_by_stage) as plain dicts (sync mode)."""
    return dict(_SECONDS), dict(_COUNTS)


# --------------------------------------------------------------- events mode
def begin_image() -> None:
    """Reset the step index at the start of one timed image. Asserts the
    previous image's records were harvested (unharvested event pairs would
    otherwise leak across images and corrupt the pool accounting)."""
    global _STEP
    if _RECORDS:
        raise RuntimeError(
            f"{len(_RECORDS)} unharvested profiling records; call harvest() first"
        )
    _STEP = -1


def next_step() -> None:
    """Bump the denoise-step index (driver: transformer forward_pre_hook)."""
    global _STEP
    _STEP += 1


def current_step() -> int:
    return _STEP


def _take_event() -> "torch.cuda.Event":
    if _POOL:
        return _POOL.pop()
    return torch.cuda.Event(enable_timing=True)


def open_section(name: str) -> Optional[tuple]:
    """Record the start event of a hook-driven bracket (step_total, vae_decode
    -- regions whose boundaries live in two different callbacks). Returns a
    handle for close_section, or None when events profiling is off."""
    if not _ENABLED or _MODE != "events":
        return None
    ev0 = _take_event()
    ev0.record()
    return (name, _STEP, ev0)


def close_section(handle: Optional[tuple]) -> None:
    if handle is None:
        return
    name, step, ev0 = handle
    ev1 = _take_event()
    ev1.record()
    _RECORDS.append((name, step, ev0, ev1))


def harvest() -> List[dict]:
    """Read every recorded event pair -> [{"name", "step", "ms"}], recycling
    the events into the pool. The caller MUST torch.cuda.synchronize() first
    (the single per-image sync); elapsed_time on an incomplete event raises."""
    out = []
    for name, step, ev0, ev1 in _RECORDS:
        out.append({"name": name, "step": step, "ms": float(ev0.elapsed_time(ev1))})
        _POOL.append(ev0)
        _POOL.append(ev1)
    _RECORDS.clear()
    return out


class section:
    """Context manager timing a named stage (no-op unless enable() was called).

    events_only=True additionally makes the bracket a no-op in sync mode: such
    sections nest inside existing sync sections, where the nested synchronize
    would corrupt the enclosing section's legacy numbers."""

    __slots__ = ("name", "_t0", "_sync", "_events_only", "_handle")

    def __init__(self, name: str, events_only: bool = False):
        self.name = name
        self._events_only = events_only

    def __enter__(self):
        if not _ENABLED:
            return self
        if _MODE == "events":
            self._handle = open_section(self.name)
            return self
        if self._events_only:
            return self
        self._sync = torch.cuda.is_available()
        if self._sync:
            torch.cuda.synchronize()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if not _ENABLED:
            return False
        if _MODE == "events":
            close_section(getattr(self, "_handle", None))
            self._handle = None
            return False
        if self._events_only:
            return False
        if self._sync:
            torch.cuda.synchronize()
        _SECONDS[self.name] += time.perf_counter() - self._t0
        _COUNTS[self.name] += 1
        return False
