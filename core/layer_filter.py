"""Shared layer-type predicate for quantization target discovery.

The single owner of the module-type test that was duplicated as
``sinq_diffusion._is_quant_target`` and the verbatim mirror in
``scripts/count_ignored_params._is_quant_target``. This owns ONLY the type test
(Linear, or 1x1 groups=1 Conv2d); every call site keeps its own skip-set and
iteration logic and passes its own ignore sets.
"""
from __future__ import annotations

from torch import nn


def is_quant_target(mod: nn.Module, include_conv2d: bool = True) -> bool:
    """Predicate for modules eligible for weight quantization.

    Linear: any nn.Linear.
    Conv2d: only 1x1, groups=1 (when ``include_conv2d``). Depthwise (groups != 1)
    is skipped because the per-output-channel weight count (kH*kW) is too small to
    group-quantize; larger kernels would need a different quant strategy.
    """
    if type(mod) is nn.Linear:
        return True
    if (include_conv2d and type(mod) is nn.Conv2d
            and mod.kernel_size == (1, 1) and mod.groups == 1):
        return True
    return False
