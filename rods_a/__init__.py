"""
RoDS-A — activation-side companion to the RoDS weight quantizer.

Phase 0 + Phase 1 (CPU scaffolding). Applies a fixed normalized Walsh-Hadamard
rotation on the contraction (input-channel) axis and quantizes activations
per-token with a closed-form, data-free optimal clip whose scale is a
bounded-influence estimator (the activation mirror of RoDS's Huber weight
objective). Spec: RoDS_activation_proposal1.md, sections 1-3 and 5.

This package is fake-quant only this phase: weights stay FP16, no weight
rotation / RoDS re-fit, no int4 bit-packing.
"""

from rods_a.constants import (
    GAMMA_STAR,
    compute_gamma_star,
    gamma_star,
    phi_inv,
    beta_c,
    PHI_INV_999,
)
from rods_a.fwht import fwht, rotate, unrotate, next_pow2, pad_to_pow2, dense_hadamard
from rods_a.quantize_act import quantize_act, estimate_scale, huber_scale_fixedpoint
from rods_a.wrapper import RoDSActLinear, select_act_targets, wrap_transformer

__all__ = [
    "GAMMA_STAR",
    "compute_gamma_star",
    "gamma_star",
    "phi_inv",
    "beta_c",
    "PHI_INV_999",
    "fwht",
    "rotate",
    "unrotate",
    "next_pow2",
    "pad_to_pow2",
    "dense_hadamard",
    "quantize_act",
    "estimate_scale",
    "huber_scale_fixedpoint",
    "RoDSActLinear",
    "select_act_targets",
    "wrap_transformer",
]
