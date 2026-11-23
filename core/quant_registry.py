"""Single source of truth for quantization method strings.

SINQ dispatches on substring matches over the method string (see
`sinq/dual_shift.py:quantize_dual_scale_shift`), so a typo like
`hqq_jiont` would route to plain HQQ silently. This registry lists the
recognised method strings that the cfq-diffusion entrypoints accept, and
provides `validate_method()` to catch typos at config-load time.

Keep this in sync with the dispatch branches in SINQ. Methods marked
"dispatched in cfq-diffusion" are intercepted by scripts/quantize_weights.py
before reaching SINQ and are implemented in this repo (not in the SINQ
library).
"""

KNOWN_METHODS: set[str] = {
    # Plain RTN baseline (no proximal optimisation).
    "rtn",
    # Plain NF4 baseline — per-group absmax mapping onto the fixed 16-level NF4
    # quantile codebook, weight-only, FP scales, no Sinkhorn, no double-quant.
    # Dispatched in cfq-diffusion via fid_eval.quantize_nf4 (NOT the SINQ
    # `sinq_nf4` path, which adds Sinkhorn normalisation).
    "nf4",
    # SINQ family (Sinkhorn-normalised, dual-scale).
    "sinq",
    "sinq_nf4",
    "sinq_nf3",
    "asinq",
    # HQQ family (single-scale, proximal-legacy step).
    "hqq",
    "hqq_l2sq",
    "hqq_huber",
    # HQQ-Joint family (mean-update for z, closed-form LS for s).
    "hqq_joint",
    "hqq_joint_l2sq",
    "hqq_joint_huber",
    # Dual-Scale HQQ — dispatched in cfq-diffusion (utils/rods.py).
    "rods",
    # SVDQuant weight-only (W4A16) — dispatched in cfq-diffusion
    # (utils/svdquant_weight_only.py). Low-rank bf16 branch + low-bit residual;
    # activations stay bf16 (no nunchaku W4A4 kernel).
    "svdquant",
    # SVDQuant *calibrated* weight-only (W4A16 / W3A16) — reconstructed from a
    # deepcompressor W4A4/W3A4 checkpoint as W_eff=(R_dq+LR)/s via
    # utils/svdquant_calibrated.py (mode="weight_only").
    "svdquant_cal",
    # SVDQuant *calibrated* native (W4A4 / W3A4) — same artifacts, but each linear
    # is wrapped with SVDQuantCalLinear which fake-quantizes activations to INT4
    # (mode="w4a4").
    "svdquant_w4a4",
    # RoDS-A — activation-side companion to RoDS (rods). Inference-time
    # per-token activation fake-quant with RHT rotation + data-free optimal clip
    # (rods_a/). Registered for naming consistency only; it is an activation
    # wrapper applied at inference, NOT dispatched in quantize_weights.py.
    "rods_a",
}


def validate_method(method: str) -> str:
    """Lowercase + dash→underscore, then check against the registry."""
    canonical = method.lower().replace("-", "_")
    if canonical not in KNOWN_METHODS:
        raise ValueError(
            f"Unknown quantization method '{method}' "
            f"(canonical form '{canonical}'). "
            f"Known: {sorted(KNOWN_METHODS)}"
        )
    return canonical
