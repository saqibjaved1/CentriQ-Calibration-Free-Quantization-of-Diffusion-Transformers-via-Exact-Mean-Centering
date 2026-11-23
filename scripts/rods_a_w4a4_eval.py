"""
Generation + evaluation driver for the composed W4A4 path (RoDS-A Phase 2).

Reuses the MJHQ-5k (svdquant) prompt/reference loaders and the diffusers pipeline
helpers; installs the RoDS-A W4A4 path (`rods_a.install_w4a4`). Modes via config:
SANITY (gate 3: num_samples<=16, contact_sheet=true, per-layer real-activation
SQNR) and FULL (gate 4: num_samples=5000, compute_fid=true, compute_ir=true).
Per-role ablation (quant.roles_enabled) and the rotation isolator (quant.rotate_act)
are driven by the quant config; `ablation_tag` namespaces outputs. dev_slice=true is
the Phase-3 (AB1) mode (firewall: screening runs on the disjoint dev slice, NEVER
the svdquant headline set; slice built by scripts/build_dev_slice.py; FP16 baseline
arm = quant_enabled=false). Run via RunAI (never on the front node).
"""

import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from core.model_config import get_model_config, load_pipeline
from core.generate_core import generate_into_dir
from rods_a.wrapper import ActQuantConfig, RoDSAW4A4Linear
from rods_a.install_w4a4 import install_w4a4
from core.mjhq_data import (  # repointed P1 c3 (was scripts.fid_eval shim + c2 line)
    download_mjhq_svdquant,
    load_prompts_svdquant,
    load_prompts_dev,
)

try:
    import wandb
except ImportError:
    wandb = None


def _make_args(cfg, model_config):
    steps = cfg.num_inference_steps or model_config.default_steps
    gs = cfg.guidance_scale if cfg.guidance_scale is not None else model_config.default_guidance_scale
    return SimpleNamespace(
        num_inference_steps=int(steps), guidance_scale=float(gs),
        height=int(cfg.height), width=int(cfg.width),
        device=str(cfg.device), seed=int(cfg.seed),
        max_sequence_length=int(cfg.get("max_sequence_length", 300)),
    )


def _generate(pipe, model_config, entries, gen_dir, args, overwrite=False,
              paired_seed=False):
    """Generation.

    Default (paired_seed=False): a single advancing global generator seeded once
    with args.seed, matching fid_eval.py's SVDQuant / deepcompressor convention
    (one global seed per generation run, not per-image) so image i is noise-matched
    with the rods (RoDS) fid_eval path. NOTE: skipping already-generated
    images does NOT advance the generator, so partial-resume runs will not
    reproduce a from-scratch run bit-for-bit.

    Paired mode (paired_seed=True, AB1.1 dev-slice ablation): the generator is
    RESEEDED per image from entry["seed"] (the persisted per-prompt seed map). This
    makes the prompt->noise mapping identical across ALL arms and FP16 regardless of
    arm ordering or resume-skips -- the precondition for the paired per-prompt
    IR/CLIP significance tests."""
    r = generate_into_dir(pipe, model_config, entries, gen_dir, args,
                          overwrite=overwrite, paired_seed=paired_seed)
    print(f"[gen] {r.n_generated} generated, {r.n_skipped} resumed/skipped "
          f"({r.elapsed:.1f}s, {r.elapsed/max(r.n_generated,1):.2f}s/img) -> {gen_dir}")
    return r.paths, r.elapsed


def _contact_sheet(paths, out_path, cols=4, thumb=256):
    n = len(paths)
    if n == 0:
        return
    cols = min(cols, n)
    rows = (n + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, rows * thumb), (20, 20, 20))
    for i, p in enumerate(paths):
        im = Image.open(p).convert("RGB").resize((thumb, thumb))
        sheet.paste(im, ((i % cols) * thumb, (i // cols) * thumb))
    sheet.save(out_path)
    print(f"[contact] wrote {out_path} ({n} imgs, {rows}x{cols})")


def _collect_layer_sqnr(transformer):
    try:
        from rods_a.conv_wrapper import RoDSAW4A4Conv1x1
    except ImportError:  # conv extension absent -> Linear-only collection
        RoDSAW4A4Conv1x1 = None
    rows = {}
    conv_parents = set()
    for name, m in transformer.named_modules():
        if RoDSAW4A4Conv1x1 is not None and isinstance(m, RoDSAW4A4Conv1x1):
            # Key the conv wrapper by ITS name (matches install diags), and skip
            # its inner `.lin` below so the per-role join sees one row per conv.
            conv_parents.add(name)
            v = m.sqnr_db()
            if v is not None:
                rows[name] = v
        elif isinstance(m, RoDSAW4A4Linear):
            if name.rsplit(".", 1)[0] in conv_parents:
                continue  # inner .lin of a RoDSAW4A4Conv1x1, already collected
            v = m.sqnr_db()
            if v is not None:
                rows[name] = v
    return rows


@hydra.main(version_base=None, config_path="../configs", config_name="rods_a_w4a4_eval")
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    model_config = get_model_config(str(cfg.model_type))
    model_id = cfg.model_id or model_config.default_model_id
    args = _make_args(cfg, model_config)
    actcfg = ActQuantConfig.from_mapping(cfg.quant)
    roles = OmegaConf.to_container(cfg.quant.get("roles_enabled", {}), resolve=True) or {}

    tag = str(cfg.get("ablation_tag", "uniform"))
    out_root = os.path.join(str(cfg.output_dir), tag)
    gen_dir = os.path.join(out_root, "images")
    os.makedirs(out_root, exist_ok=True)

    run = None
    if wandb is not None and bool(cfg.get("use_wandb", True)):
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "rods-a-w4a4"),
            name=tag,
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=os.environ.get("WANDB_MODE", "offline"),
        )

    # ---- load + (optionally) install W4A4 ----
    # cpu_offload defaults True (PixArt/SANA use pipe.to(device) anyway); FLUX W4A4 sets
    # it False so weights are on `device` before install_w4a4 (offload + in-place module
    # replacement otherwise leave operands on mixed devices -> prepare_weight crash).
    pipe = load_pipeline(model_config, model_id, args.device,
                         cpu_offload=bool(cfg.get("cpu_offload", True)))
    quant_enabled = bool(cfg.get("quant_enabled", True))   # false => FP16 baseline arm

    # ---- optional in-place RTN/NF4 baseline weights (naive-W4A4 cells) ----
    # Quantizes the act-target layer set BEFORE install_w4a4; the A4-only branch
    # (weight_quant_enabled=false) then snapshots the already-quantized weight as
    # w_hat_override. Mutually exclusive with the RoDS weight refit and with fits.
    bw_method = cfg.get("baseline_weight_method", None)
    bw_diag = None
    if bw_method:
        if quant_enabled and bool(cfg.quant.get("weight_quant_enabled", True)):
            raise ValueError("baseline_weight_method with the RoDS weight refit would "
                             "double-quantize: set +quant.weight_quant_enabled=false "
                             "(W4A4 cells) or quant_enabled=false (W*A16 cells).")
        if cfg.get("fits_path", None):
            raise ValueError("baseline_weight_method is incompatible with fits_path "
                             "(pinned RoDS fits would overwrite the baseline weights).")
        from rods_a.baseline_weight import apply_baseline_weight_quant
        bw_diag = apply_baseline_weight_quant(
            pipe.transformer,
            method=str(bw_method),
            nbits=int(cfg.get("baseline_weight_bits", 4)),
            group_size=int(cfg.quant.get("group_size", 64)),
            model_type=str(cfg.model_type),
        )

    if quant_enabled:
        # Firewall/established guards: no instrumentation host-syncs and a fakequant
        # backend during generation (decisions identical to the kernel path). Assert
        # from the resolved config before anything runs.
        if bool(getattr(actcfg, "instrument", False)) and not bool(cfg.get("allow_instrument", False)):
            raise ValueError("instrument must be false for generation runs (host syncs). "
                             "Set allow_instrument=true ONLY for a dedicated dev stats run "
                             "(do NOT use its images for metrics).")
        backend = str(getattr(actcfg, "backend", "fakequant"))
        allow_int4 = bool(cfg.get("allow_int4_backend", False))
        if backend != "fakequant" and not (backend == "int4" and allow_int4):
            raise ValueError(f"backend must be 'fakequant' for the quality track, "
                             f"got {backend!r}. The U1 attribution arm may set "
                             f"quant.backend=int4 with allow_int4_backend=true "
                             f"(pinned kron fits required).")
        fits_path = cfg.get("fits_path", None)
        if fits_path is not None and not cfg.get("expected_sha", None):
            raise ValueError("fits_path requires expected_sha (pinned fits are "
                             "sha-verified; never deploy an unverified artifact).")
        if backend == "int4":
            from rods_a.install_w4a4 import assert_deploy_instrument_off
            assert_deploy_instrument_off(actcfg)
            if not fits_path:
                raise ValueError("backend=int4 requires fits_path + expected_sha "
                                 "(the pinned, sha-verified kron fits).")
        center_roles = OmegaConf.to_container(cfg.quant.get("center_roles_enabled", {}), resolve=True) or {}
        # fp16 matmul by default (faster + matches deployment); fp32 only to isolate
        # W4A4 quant error from fp16 rounding.
        matmul_dtype = torch.float32 if bool(cfg.get("matmul_fp32", False)) else torch.float16
        # SQNR-capture reference-copy dtype: fp32 (legacy, bit-identical) or fp16
        # (halves the resident fp_weight memory -- the FLUX enabler; exact cast of
        # bf16 weights, amax-guarded at install).
        capture_dtype = (torch.float16 if str(cfg.get("capture_dtype", "float32")) == "float16"
                         else torch.float32)
        diags = install_w4a4(
            pipe.transformer, actcfg, model_type=str(cfg.model_type),
            weight_bits=int(cfg.quant.get("nbits", 4)), group_size=int(cfg.quant.get("group_size", 64)),
            roles_enabled=roles, center_act_mean=actcfg.center_act_mean,
            center_roles_enabled=center_roles,
            device=args.device, compute_dtype=model_config.default_dtype,
            matmul_dtype=matmul_dtype,
            strict_rho=bool(cfg.get("strict_rho", False)),
            # capture_sqnr keeps an FP-reference matmul per layer -- fakequant-only
            # (the int4 path has no fp_weight shadow in deployment).
            capture_sqnr=bool(cfg.get("capture_sqnr", True)) and backend == "fakequant",
            backend=backend,
            fits_path=None if fits_path is None else str(fits_path),
            expected_sha=None if fits_path is None else str(cfg.get("expected_sha")),
            fit_t=bool(cfg.quant.get("fit_t", False)),  # A12 free-t arm (fakequant only)
            # U2.2 A4-only arm: FP rotated weight, no refit (fakequant+reference only).
            weight_quant_enabled=bool(cfg.quant.get("weight_quant_enabled", True)),
            tol_outer=float(cfg.quant.get("tol_outer", 1e-3)),  # BCD outer early-stop tol
            tol_irls=float(cfg.quant.get("tol_irls", 1e-3)),    # IRLS inner early-stop tol
            capture_dtype=capture_dtype,
        )
        # Optional SANA GLUMBConv 1x1 pointwise-conv quantization (diagnostic runs
        # only; default off keeps the deployed Linear-only set untouched).
        if bool(cfg.get("install_ffn_conv", False)):
            from rods_a.conv_wrapper import install_ffn_conv_w4a4
            conv_diags = install_ffn_conv_w4a4(
                pipe.transformer, actcfg,
                weight_bits=int(cfg.quant.get("nbits", 4)),
                group_size=int(cfg.quant.get("group_size", 64)),
                rotation=actcfg.rotation,
                device=args.device, compute_dtype=model_config.default_dtype,
                matmul_dtype=matmul_dtype,
                capture_sqnr=bool(cfg.get("capture_sqnr", True)) and backend == "fakequant",
                center_act_mean=actcfg.center_act_mean,
                capture_dtype=capture_dtype,
            )
            diags.update(conv_diags)
        rho_med = float(torch.tensor([d["rho_median"] for d in diags.values()]).median()) if diags else float("nan")
    else:
        print("[fp16] quant_enabled=false -> pure FP16 baseline (no install_w4a4).")
        diags = {}
        rho_med = float("nan")
        backend, fits_path = None, None

    # ---- prompts: dev slice (firewall) or svdquant headline ----
    dev_mode = bool(cfg.get("dev_slice", False))
    if dev_mode:
        dev_path = cfg.get("dev_prompts_path")
        if not dev_path:
            raise ValueError("dev_slice=true needs dev_prompts_path (a JSON of held-out MJHQ prompts)")
        dev_path = str(dev_path)
        dev = load_prompts_dev(dev_path, int(cfg.num_samples))
        entries = dev["images"]
        images_ref_dir = dev["ref_dir"]
        print(f"[dev] dev slice: {len(entries)} prompts, FID ref={images_ref_dir}")
    else:
        images_ref_dir, prompts_path = download_mjhq_svdquant(str(cfg.mjhq_dir))
        entries = load_prompts_svdquant(prompts_path, int(cfg.num_samples), int(cfg.seed))["images"]

    # ---- generate (resumable: skips valid existing images unless overwrite=true) ----
    paired = bool(cfg.get("paired_seed", False))
    paths, gen_s = _generate(pipe, model_config, entries, gen_dir, args,
                             overwrite=bool(cfg.get("overwrite", False)),
                             paired_seed=paired)

    if bool(cfg.get("contact_sheet", False)):
        _contact_sheet(paths, os.path.join(out_root, "contact_sheet.png"))

    # ---- per-layer real-activation SQNR ----
    layer_sqnr = _collect_layer_sqnr(pipe.transformer)
    sqnr_median = float(torch.tensor(list(layer_sqnr.values())).median()) if layer_sqnr else float("nan")
    print(f"[sqnr] composed-W4A4 vs FP16 per-layer SQNR: median={sqnr_median:.2f} dB "
          f"over {len(layer_sqnr)} layers")

    results = {
        "tag": tag, "model_id": model_id, "num_samples": len(paths),
        "seed": int(cfg.seed),
        "dev_slice": dev_mode, "paired_seed": paired, "quant_enabled": quant_enabled,
        # Baseline-weight provenance (None unless a bl_* cell): so no run with
        # in-place RTN/NF4 weights can be mistaken for FP16 or RoDS.
        "baseline_weight_method": None if bw_diag is None else bw_diag["method"],
        "baseline_weight_bits": None if bw_diag is None else bw_diag["nbits"],
        "baseline_n_layers": None if bw_diag is None else bw_diag["n_quantized"],
        "baseline_rho_stat": None if bw_diag is None else bw_diag["rho_stat"],
        "baseline_rho_median": None if bw_diag is None else bw_diag["rho_median_global"],
        "baseline_per_layer_rho": None if bw_diag is None else bw_diag["per_layer"],
        "rho_median": rho_med, "sqnr_median_db": sqnr_median,
        "layer_sqnr_db": layer_sqnr, "install_diags": diags,
        "gen_seconds": gen_s, "roles_enabled": roles,
        "rotate_act": actcfg.rotate_act, "center_act_mean": actcfg.center_act_mean,
        "act_bits": actcfg.act_bits,
        "weight_bits": int(cfg.quant.get("nbits", 4)),
        "group_size": int(cfg.quant.get("group_size", 64)),
        # U1 attribution provenance: which arithmetic path + which weight fits.
        "backend": backend,
        "rotation": actcfg.rotation,
        # Factored-rotation provenance (None unless rotation="factored"): the
        # width-keyed maps plus the capture/conv knobs, so every sweep arm is
        # reconstructible from its results.json alone.
        "rotation_factorization": (None if actcfg.rotation_factorization is None
                                   else {str(k): list(v) for k, v
                                         in actcfg.rotation_factorization.items()}),
        "rotation_pad_to": (None if actcfg.rotation_pad_to is None
                            else {str(k): int(v) for k, v
                                  in actcfg.rotation_pad_to.items()}),
        "capture_dtype": str(cfg.get("capture_dtype", "float32")),
        "install_ffn_conv": bool(cfg.get("install_ffn_conv", False)),
        "percentile_p": actcfg.percentile_p,
        "fits_path": None if fits_path is None else str(fits_path),
        "fits_sha": None if fits_path is None else str(cfg.get("expected_sha")),
        "act_path": actcfg.act_path, "epilogue": actcfg.epilogue,
        "instrument": actcfg.instrument, "graphs": actcfg.graphs,
    }

    # ---- FID (dir-vs-dir clean-FID; dev ref or svdquant ref) ----
    if bool(cfg.get("compute_fid", False)):
        from cleanfid import fid as cleanfid
        print(f"[fid] computing clean-FID: ref={images_ref_dir} gen={gen_dir}")
        results["fid"] = float(cleanfid.compute_fid(images_ref_dir, gen_dir, mode="clean"))
        results["fid_n"] = len(paths)
        print(f"[fid] FID = {results['fid']:.3f} (n={len(paths)})")

    # ---- ImageReward (per-prompt scores persisted for paired tests) ----
    if bool(cfg.get("compute_ir", False)):
        import ImageReward as RM
        ir_model = RM.load("ImageReward-v1.0", device=args.device)
        scores = [float(ir_model.score(entries[i]["prompt"], paths[i]))
                  for i in range(len(paths))]
        st = torch.tensor(scores)
        results["imagereward_mean"] = float(st.mean())
        results["imagereward_std"] = float(st.std())
        results["ir_scores"] = scores
        print(f"[ir] ImageReward mean={results['imagereward_mean']:.4f} "
              f"std={results['imagereward_std']:.4f}")

    # ---- CLIP (per-prompt scores persisted for paired tests) ----
    if bool(cfg.get("compute_clip", False)):
        from scripts.clip_eval import load_clip, score_method, truncate_prompts_inplace
        clip_id = str(cfg.get("clip_model_id", "openai/clip-vit-large-patch14"))
        metric = load_clip(clip_id, args.device)
        clip_entries = [dict(e) for e in entries]   # truncation must not mutate IR prompts
        truncate_prompts_inplace(clip_entries, metric)
        clip = score_method(metric, clip_entries, lambda i, e: paths[i], args.device)
        results["clip_scores"] = clip["scores"]
        results["clip_mean"] = clip["mean"]
        results["clip_std"] = clip["std"]
        results["clip_model_id"] = clip_id
        print(f"[clip] CLIP mean={clip['mean']:.4f} std={clip['std']:.4f} "
              f"(n={clip['n']}, skipped={clip['skipped']})")

    with open(os.path.join(out_root, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[done] wrote {os.path.join(out_root, 'results.json')}")

    if run is not None:
        log = {k: v for k, v in results.items()
               if isinstance(v, (int, float)) and not isinstance(v, bool)}
        run.log(log)
        art = wandb.Artifact(f"rods_a_w4a4_{tag}", type="eval_result")
        art.add_file(os.path.join(out_root, "results.json"))
        run.log_artifact(art)
        run.finish()


if __name__ == "__main__":
    main()
