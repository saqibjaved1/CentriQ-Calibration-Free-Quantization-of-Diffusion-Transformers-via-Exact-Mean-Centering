"""
Install the composed W4A4 fake-quant path (RoDS-A Phase 2) into a diffusion
transformer.

For each activation-quant target Linear (same filter as Phase 1's
`select_act_targets`, so per-layer SQNR stays comparable), this:
  1. runs PREPARE_WEIGHT (rotate + grouped RoDS re-fit on W̃, t absorbed), and
  2. replaces the Linear with `RoDSAW4A4Linear` (per-token rotated activation
     fake-quant + matmul against W̃_hat).

adaLN-linear and cross-attn K/V stay FP16 (handled by `select_act_targets`'s
filter). Per-role toggles allow the image-metric ablation (turn a role's W4A4
off → that role stays full-precision FP16).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import nn

from rods_a.prepare_weight import prepare_weight
from rods_a.wrapper import (
    ActQuantConfig,
    RoDSAW4A4Linear,
    _set_submodule,
    select_act_targets,
)

# Pinned Kronecker W4 weight-fit artifact (Task K / K2). The int4 backend deploys
# EXACTLY these audited fits (sha-verified) rather than re-fitting on install.
KRON_FITS_SHA = "45bd1f4fd833e7866137cd663506c69cf4f63c1f2200e930512c9e3726085401"
# Pinned PAD W4 weight-fit artifact (V-task V2.1): produced at the HEADLINE install
# lineage (fp16 compute, tol 1e-3, fit_t=False) and verified cross-run bit-identical
# across two independent installs -- the int4 pad deployment IS the 19.13-headline
# quantization decisions. outputs/rods_a_pad_fits/pad_w4_fits.pt.
PAD_FITS_SHA = "9ca25bbc6f87905dd487ee8154a9e1dadaf97d4250e9755d8955ad4f77dfe597"
EXPECTED_TARGETS = 224  # recipe-(i) 224-layer set (select_act_targets)


# Role classification by leaf-suffix, used for the per-role precision ablation.
# Order matters: first match wins.
#
# PixArt-Sigma / SANA share these suffix conventions (attn1=self, attn2=cross, ff.net).
_ROLE_SUFFIXES: List[tuple[str, tuple[str, ...]]] = [
    ("self_attn_qkv", (".attn1.to_q", ".attn1.to_k", ".attn1.to_v")),
    ("self_attn_proj", (".attn1.to_out.0",)),
    ("cross_attn_q", (".attn2.to_q",)),
    ("cross_attn_proj", (".attn2.to_out.0",)),
    ("fc1", (".ff.net.0.proj",)),
    ("fc2", (".ff.net.2",)),
]


def _classify_role_flux(name: str) -> str:
    """FLUX MM-DiT role taxonomy (FL1). Disjoint from the PixArt suffixes (`.attn.to_q`
    vs `.attn1.to_q`), so this is purely additive. The image-stream attn/ff suffixes
    (`.attn.to_q/.to_out.0`, `.ff.net.*`) are shared between double-stream blocks and
    single-stream blocks, so the block path disambiguates. Kept identical to the FL0
    census tagger (`scripts/rods_a_flux_census.py:_flux_role`) so role strings match.
    Per FL0.3, the context stream (add_q/k/v, to_add_out, ff_context) IS quantized —
    these roles exist precisely to track it in the ablation / by-role stats."""
    single = "single_transformer_blocks." in name
    # context (text) sub-stream — double-stream only
    if name.endswith((".attn.add_k_proj", ".attn.add_v_proj")):
        return "dbl_ctx_attn_kv"          # MM-DiT analog of cross-attn K/V (NOT skipped)
    if name.endswith(".attn.add_q_proj"):
        return "dbl_ctx_attn_q"
    if name.endswith(".attn.to_add_out"):
        return "dbl_ctx_proj"
    if name.endswith(".ff_context.net.0.proj"):
        return "dbl_ctx_fc1"
    if name.endswith(".ff_context.net.2"):
        return "dbl_ctx_fc2"
    # single-stream
    if name.endswith(".proj_mlp"):
        return "single_mlp_in"
    if single and name.endswith((".attn.to_q", ".attn.to_k", ".attn.to_v")):
        return "single_attn_qkv"
    # double-stream image
    if name.endswith((".attn.to_q", ".attn.to_k", ".attn.to_v")):
        return "dbl_img_attn_qkv"
    if name.endswith(".attn.to_out.0"):
        return "dbl_img_proj"
    if name.endswith(".ff.net.0.proj"):
        return "dbl_img_fc1"
    if name.endswith(".ff.net.2"):
        return "dbl_img_fc2"
    return "other"


def block_type_flux(name: str) -> str:
    """FLUX block-type tag (double_img / double_ctx / single / toplevel) for by-block
    aggregation in the FLUX weight audit + stats. Mirrors the FL0 census tagger."""
    if "single_transformer_blocks." in name:
        return "single"
    if "transformer_blocks." in name:
        if (".add_q_proj" in name or ".add_k_proj" in name or ".add_v_proj" in name
                or ".to_add_out" in name or ".ff_context." in name
                or ".norm1_context." in name):
            return "double_ctx"
        return "double_img"
    return "toplevel"


def classify_role(name: str, model_type: str = "pixart") -> str:
    """Role tag for the per-role ablation / by-role stats. `model_type` selects the
    suffix taxonomy; defaults to "pixart" (shared by SANA) so existing callers are
    unchanged. "flux" uses the MM-DiT taxonomy."""
    if str(model_type) == "flux":
        return _classify_role_flux(name)
    for role, suffixes in _ROLE_SUFFIXES:
        if any(name.endswith(s) for s in suffixes):
            return role
    return "other"


def assert_deploy_instrument_off(cfg: ActQuantConfig) -> None:
    """Silent-regression guard (opt-B/B1): any deployed / measured / imaged run must
    resolve `instrument=off`. The §6 analysis floats are 4 DtoH host-syncs/layer
    (~5.8 s/img regression) and block CUDA-graph capture; a stray `instrument=true` would
    silently slow the path without changing any operand. Asserted from the RESOLVED
    config so the regression cannot slip through unnoticed."""
    if bool(getattr(cfg, "instrument", False)):
        raise ValueError(
            "instrument=true in a deployed/measured run -- the §6 analysis floats add "
            "4 DtoH host-syncs/layer (~5.8 s/img silent regression) and block CUDA-graph "
            "capture. Set quant.instrument=false for any measured/imaged run."
        )


def install_w4a4(
    transformer: nn.Module,
    cfg: ActQuantConfig,
    *,
    model_type: str = "pixart",
    weight_bits: int = 4,
    group_size: int = 64,
    roles_enabled: Optional[Dict[str, bool]] = None,
    center_act_mean: bool = False,
    center_roles_enabled: Optional[Dict[str, bool]] = None,
    device: str = "cpu",
    compute_dtype: torch.dtype = torch.float16,
    matmul_dtype: torch.dtype = torch.float16,
    strict_rho: bool = True,
    rho_band: tuple[float, float] = (2.0, 4.0),
    capture_sqnr: bool = False,
    backend: str = "fakequant",
    fit_t: bool = False,
    tol_outer: float = 1e-3,
    tol_irls: float = 1e-3,
    fits_path: Optional[str] = None,
    expected_sha: Optional[str] = KRON_FITS_SHA,
    shadow: bool = False,
    verbose: bool = True,
    weight_quant_enabled: bool = True,
    capture_dtype: torch.dtype = torch.float32,
) -> Dict[str, dict]:
    """Replace target Linears with RoDSAW4A4Linear in place. Returns a per-layer
    diagnostics dict {name: {role, rho_median, incoherence_W, incoherence_Wt,
    Cin_orig, Cin_pad, center_mean}}.

    `roles_enabled`: map role->bool; a role set False keeps its layers FP16
    (the per-role ablation). Default: all enabled (uniform W4A4).
    `center_act_mean`: global toggle for per-token mean-centering (DC-blowup fix);
    OFF by default. `center_roles_enabled`: optional per-role override consulted
    only when `center_act_mean` is True; unlisted roles default to True (so the
    default is: center every quantized role).
    `cfg.rotate_act` drives BOTH the weight rotation and the activation rotation
    (kept consistent so X̃ W̃ᵀ == X Wᵀ).

    `backend`: "fakequant" (re-fit weights via prepare_weight; host fp matmul or, with
    cfg.semantics="kernel", the oracle) | "int4" (the real Nunchaku INT4 W4A4 GEMM).
    The int4 path LOADS the pinned Kronecker fit artifact (`fits_path`, sha-verified
    against `expected_sha`) and deploys those audited weights -- it does NOT re-fit --
    and asserts the recipe-(i) 224-layer set. `shadow` (int4 only) keeps the kernel
    operands resident so each layer also computes the oracle output for I2 parity.
    """
    if getattr(transformer, "rods_a_w4a4_installed", False):
        print("[rods_a] Transformer already W4A4-installed, skipping.")
        return {}

    roles_enabled = roles_enabled or {}
    center_roles_enabled = center_roles_enabled or {}
    targets = select_act_targets(transformer, model_type=model_type, group_size=group_size)

    use_int4 = str(backend) == "int4"

    # Factored rotation (explicit per-width H_p (x) H_q): resolved here per layer
    # and injected through rotate_fn / rotate_override, so neither prepare_weight
    # nor quantize_act ever sees "factored" as a scheme string. Widths absent from
    # cfg.rotation_factorization fall back bitwise to the pad scheme.
    factored = str(getattr(cfg, "rotation", "pad")) == "factored"
    if factored:
        if bool(getattr(cfg, "rotation_factor_order_swap", False)):
            raise ValueError(
                "rotation_factor_order_swap is reserved for a later ordering sweep; "
                "it must stay false.")
        if (use_int4 or fits_path is not None or not weight_quant_enabled
                or str(getattr(cfg, "semantics", "reference")) != "reference"):
            raise ValueError(
                "rotation='factored' is fakequant+reference with a fresh refit only "
                "(no int4 backend, no pinned fits, no A4-only arm).")
        if not cfg.rotate_act:
            raise ValueError("rotation='factored' requires rotate_act=true.")
        from rods_a.factored_rotation import (factored_rotators, pad_rotators,
                                              resolve_factored)
        from rods_a.fwht import next_pow2 as _next_pow2
        from rods_a.fwht import rotate as _pad_rotate_fwht

    if fit_t and (use_int4 or fits_path is not None
                  or str(getattr(cfg, "semantics", "reference")) != "reference"):
        raise ValueError(
            "fit_t=True (free-t, A12) is FAKEQUANT+reference only: the per-column "
            "scale2 is applied by Quantizer.dequantize but not consumed by the int4 / "
            "kron-fits / kernel-semantics epilogue.")
    # Deploy the pinned audited fits whenever fits_path is given -- for BOTH the int4
    # backend and the kernel-semantics fake-quant, so they share identical quantization
    # decisions (variant (b) is exactly variant (a)'s oracle). int4 always requires it.
    use_fits = fits_path is not None
    if use_int4 and not use_fits:
        raise ValueError("backend='int4' requires fits_path (the pinned kron fits).")
    # U2.2 A4-only arm: keep the weight FP (rotated, NO refit) while the activation
    # is quantized. fakequant+reference only; mutually exclusive with pinned fits.
    if not weight_quant_enabled and (
            use_int4 or use_fits
            or str(getattr(cfg, "semantics", "reference")) != "reference"):
        raise ValueError("weight_quant_enabled=False (A4-only arm) requires "
                         "backend='fakequant' + semantics='reference' and no fits_path.")
    kron_fits = None
    if use_fits:
        from rods_a.kron_fits import load_kron_fits, prepared_from_fit
        # The artifact's rotation tag must MATCH the configured basis (V2 pad-int4
        # generalization): kron deploys kron_w4_fits.pt, pad deploys pad_w4_fits.pt.
        # A basis/artifact mismatch raises inside load_kron_fits.
        kron_fits = load_kron_fits(fits_path, expected_sha=expected_sha,
                                   expected_rotation=str(cfg.rotation))
        if len(targets) != EXPECTED_TARGETS:
            raise RuntimeError(
                f"fits install: select_act_targets returned {len(targets)} layers, "
                f"expected the recipe-(i) {EXPECTED_TARGETS}-layer set."
            )

    diags: Dict[str, dict] = {}
    n_quant, n_skip_role, n_center = 0, 0, 0
    for name, mod in targets.items():
        role = classify_role(name, model_type)
        if not roles_enabled.get(role, True):
            n_skip_role += 1
            continue  # keep FP16 for this role (ablation)
        center = bool(center_act_mean) and bool(center_roles_enabled.get(role, True))
        w_hat_override = None
        rot_w = act_rot = act_unrot = None
        fact_spec = None
        rot_desc = None
        if factored:
            fact_spec = resolve_factored(mod.in_features,
                                         getattr(cfg, "rotation_factorization", None),
                                         getattr(cfg, "rotation_pad_to", None))
            if fact_spec is not None:
                if fact_spec.pad_to % group_size != 0:
                    raise ValueError(
                        f"factored rotation for {name}: pad_to={fact_spec.pad_to} is "
                        f"not divisible by group_size={group_size}.")
                act_rot, act_unrot = factored_rotators(fact_spec)
                rot_w = act_rot                   # same orthonormal map on both sides
                rot_desc = (f"{fact_spec.factors[0]}x{fact_spec.factors[1]}"
                            f"@{fact_spec.pad_to}")
            else:
                # Unmapped width: bitwise the pad scheme. Weight side uses the
                # fwht-impl pad rotate (what prepare_weight's pad branch runs);
                # activation side the dense pad pair (what the hot path runs).
                act_rot, act_unrot = pad_rotators("dense")
                rot_w = _pad_rotate_fwht
                rot_desc = f"pad@{_next_pow2(mod.in_features)}"
        if use_fits:
            if name not in kron_fits:
                raise RuntimeError(f"fits install: layer {name} missing from kron fits.")
            pack = prepared_from_fit(kron_fits[name])
            # Centering needs the FP-weight colsum (DECISION 1). Pinned fits predating
            # Contract v3 may not carry it; backfill from the ORIGINAL unrotated weight
            # here (real Cin, fp32). This touches NO W4 codes/scales -> sha unchanged.
            if center and pack.colsum is None:
                pack.colsum = mod.weight.detach().to(torch.float32).sum(dim=1)
        elif not weight_quant_enabled:
            # U2.2 A4-only arm: rotated FP weight, NO refit, no W4 codes. The pack
            # carries shapes/colsum only; the actual operand is w_hat_override.
            from rods_a.fwht import rotate_for
            from rods_a.prepare_weight import PreparedWeight, max_colnorm_ratio
            W = mod.weight.detach().to(device=device, dtype=torch.float32)
            Wt = (rotate_for(cfg.rotation, "fwht",
                             seed=int(getattr(cfg, "rotation_seed", 0)))(W)
                  if cfg.rotate_act else W)
            pack = PreparedWeight(
                W_q=None, meta={"group_size": group_size},
                Cin_orig=W.shape[-1], Cin_pad=Wt.shape[-1],
                rotated=bool(cfg.rotate_act), out_features=W.shape[0],
                rho_median=float("nan"),
                incoherence_W=max_colnorm_ratio(W),
                incoherence_Wt=max_colnorm_ratio(Wt),
                bias=None if mod.bias is None else mod.bias.detach(),
                colsum=W.sum(dim=1),
            )
            w_hat_override = Wt
        else:
            pack = prepare_weight(
                mod.weight,
                bias=mod.bias,
                bits=weight_bits,
                group_size=group_size,
                device=device,
                compute_dtype=compute_dtype,
                rotate_weight=cfg.rotate_act,
                rotation=cfg.rotation,
                rho_band=rho_band,
                strict_rho=strict_rho,
                fit_t=fit_t,
                tol_outer=tol_outer,
                tol_irls=tol_irls,
                rotation_seed=int(getattr(cfg, "rotation_seed", 0)),
                rotate_fn=rot_w,
            )
        fp_w = mod.weight if capture_sqnr else None
        _set_submodule(transformer, name,
                       RoDSAW4A4Linear(pack, cfg, fp_weight=fp_w, center_mean=center,
                                       matmul_dtype=matmul_dtype,
                                       backend=backend, shadow=shadow,
                                       layer_name=name, role=role, device=device,
                                       w_hat_override=w_hat_override,
                                       rotate_override=act_rot,
                                       unrotate_override=act_unrot,
                                       capture_dtype=capture_dtype))
        diags[name] = {
            "role": role,
            "rho_median": pack.rho_median,
            "incoherence_W": pack.incoherence_W,
            "incoherence_Wt": pack.incoherence_Wt,
            "Cin_orig": pack.Cin_orig,
            "Cin_pad": pack.Cin_pad,
            "center_mean": center,
        }
        if factored:
            # Conditional keys: default-scheme runs keep the legacy diag dict shape.
            diags[name]["rotation_desc"] = rot_desc
            if fact_spec is not None:
                diags[name]["factors"] = list(fact_spec.factors)
                diags[name]["pad_to"] = fact_spec.pad_to
        n_quant += 1
        n_center += int(center)

    if use_fits and not roles_enabled and n_quant != EXPECTED_TARGETS:
        raise RuntimeError(
            f"fits install: wrapped {n_quant} layers, expected {EXPECTED_TARGETS} "
            f"(uniform recipe-(i)). Role toggles off -> count must match."
        )

    transformer.rods_a_w4a4_installed = True
    if verbose:
        rhos = [d["rho_median"] for d in diags.values()]
        rho_med = float(torch.tensor(rhos).median()) if rhos else float("nan")
        print(
            f"[rods_a] W4A{cfg.act_bits} installed on {n_quant} Linears "
            f"({n_skip_role} skipped by role toggle), backend={backend}"
            f"{'/shadow' if (use_int4 and shadow) else ''}, model_type={model_type}, "
            f"weight {weight_bits}b gs={group_size}, rotate={cfg.rotate_act}, "
            f"est={cfg.scale_estimator}, center_mean={n_center}/{n_quant}. "
            f"rho_median(all layers)={rho_med:.3f}."
        )
        # Deploy knobs (the ~5.8 s/img silent-regression guard surface): every measured /
        # imaged run prints the resolved instrument/act_path/graphs/epilogue so the
        # configuration is visible in the logs and `instrument` is auditable post-hoc.
        print(
            f"[rods_a] deploy knobs: act_path={cfg.act_path}, "
            f"instrument={'on' if cfg.instrument else 'off'}, "
            f"graphs={'on' if cfg.graphs else 'off'}, "
            f"epilogue={getattr(cfg, 'epilogue', 'reference')}."
        )
        if factored:
            # One line per width-class so the resolved basis is auditable in logs.
            by_class: Dict[tuple, int] = {}
            for d in diags.values():
                key = (d["Cin_orig"], d["rotation_desc"])
                by_class[key] = by_class.get(key, 0) + 1
            parts = [f"{w}->{desc} ({n} layers)"
                     for (w, desc), n in sorted(by_class.items())]
            print(f"[rods_a] factored rotation: " + "; ".join(parts))
    return diags
