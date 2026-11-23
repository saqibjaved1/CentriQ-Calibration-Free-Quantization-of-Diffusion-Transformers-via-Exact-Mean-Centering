"""core — shared quantization primitives.

Modules used by BOTH the weight-only (rods) path and the baseline path: model
registry/config, quant-run naming, evaluation I/O discovery, diffusers pipeline
loading, the method-string registry, and the layer-type predicate. Imported
directly (``from core.X import ...``); this package intentionally re-exports nothing.
"""
