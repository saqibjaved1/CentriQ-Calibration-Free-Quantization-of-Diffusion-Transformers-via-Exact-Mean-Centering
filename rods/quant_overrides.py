"""Per-method override merge for the list-of-methods quant config.

`quant_compare.yaml` keeps the methods as a YAML list, which Hydra cannot
easily override positionally. The `quant_overrides:` block keys overrides by
method `name`, and `apply_quant_overrides` merges them into the list at run
time:

    quant_overrides:
      HQQ-Joint:
        huber_delta: 0.5
      SINQ:
        sinkhorn_order: 64

CLI form:
    +quant_overrides.HQQ-Joint.huber_delta=0.5
"""

from typing import Iterable


def apply_quant_overrides(methods: Iterable[dict], overrides: dict | None) -> list[dict]:
    out = [dict(m) for m in methods]
    if not overrides:
        return out
    by_name = {m["name"]: m for m in out}
    for name, kv in overrides.items():
        if name not in by_name:
            raise KeyError(
                f"quant_overrides references unknown method '{name}'. "
                f"Known: {list(by_name)}"
            )
        by_name[name].update(dict(kv))
    return out
