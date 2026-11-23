"""Dual-Scale HQQ weight quantization.

Minimises || W - s*(Q - z)*t ||_p^p (SINQ storage convention; z_task = -z) by
alternating updates: Sinkhorn warm-start of (s, t); SINQ's HQQ inner loop on the
preconditioned tile W/s/t giving (Q, z, s_q); K_outer rounds of lp-IRLS sweeps on
t then s, each with an HQQ re-run; s_q absorbed into s. K_warm = K_outer = 0 is
bit-exact vanilla SINQ HQQ (modulo sinkhorn_order = 0).

`prox_op` controls the inner HQQ step only: "lp" (default) uses SINQ's
lp shrinkage; "huber" uses the Huber prox with threshold `huber_delta`;
"l2sq" uses the plain L2-squared prox. The outer IRLS sweeps for s and t
remain lp-IRLS regardless. With prox_op="huber", `huber_delta` becomes
active; otherwise it is ignored.

Output is packed into the (W_q, meta) tuple format produced by
sinq.quantizer.Quantizer.quantize, so the existing artifact builder and
Quantizer.dequantize work unchanged.

This module is dispatched from scripts/quantize_weights.py:quantize_layer
when method == "rods".
"""

from __future__ import annotations

from typing import Optional

import torch

from core.sinq.dual_shift import hqq_rtn, optimize_weights_proximal_legacy_step
from core.sinq.quantizer import Quantizer
from core.sinq.sinkhorn import sinkhorn_log

try:
    import wandb as _wandb
except ImportError:
    _wandb = None


def _hqq_inner(
    matrix: torch.Tensor,
    min_max: list[int],
    p: float,
    huber_delta: float,
    beta_mult: float,
    prox_op: str = "lp",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Verbatim re-implementation of the HQQ branch from
    SINQ/sinq/dual_shift.py:238-257, parameterised on lp_norm = p.

    Inputs:
        matrix:       (m, n) float tile
        min_max:      [q_min, q_max]
        p:            lp norm exponent (used when prox_op='lp')
        huber_delta:  Huber threshold (used when prox_op='huber')
        beta_mult:    multiplicative annealing of beta per inner iteration
        prox_op:      'lp' (default), 'huber', or 'l2sq'; selects the inner prox operator

    Returns:
        (q, z, scales)  -- SINQ storage convention
        q:      (m, n) int codes in [q_min, q_max]
        z:      (m, 1) zero-point
        scales: (m, 1) per-row scale; reconstruction is (q - z) * scales
    """
    q, scales, z, _ = hqq_rtn(matrix, min_max)
    best_z = torch.zeros_like(z)
    best_scales = torch.ones_like(scales)
    best_error = torch.tensor(float("inf"), dtype=torch.float32, device=matrix.device)
    W_q = q
    beta = 1e1
    for _ in range(20):
        W_r, W_q, z, scales = optimize_weights_proximal_legacy_step(
            matrix,
            scales.clip(1e-5, 1e5),
            z,
            min_max,
            beta=beta,
            lp_norm=p,
            prox_op=prox_op,
            huber_delta=huber_delta,
        )
        current_error = torch.abs(matrix - W_r).mean().float()
        take = current_error < best_error
        best_error = torch.where(take, current_error, best_error)
        best_z = torch.where(take[..., None], z, best_z)
        best_scales = torch.where(take[..., None], scales, best_scales)
        beta *= beta_mult
    # Match SINQ behaviour: use last iteration's q with the best (z, scales).
    q = W_q
    scales = 1.0 / best_scales
    z = best_z
    return q, z, scales


def _objective_lp(
    matrix: torch.Tensor,
    s: torch.Tensor,
    t: torch.Tensor,
    Q: torch.Tensor,
    z: torch.Tensor,
    s_q: torch.Tensor,
    p: float,
) -> float:
    """|| W - s * (Q - z) * s_q * t ||_p^p (SINQ sign convention)."""
    W_hat = s * (Q.to(matrix.dtype) - z) * s_q * t
    return matrix.sub(W_hat).abs().pow(p).sum().item()


def _rods_tile(
    matrix: torch.Tensor,
    min_max: list[int],
    p: float,
    K_outer: int,
    K_irls: int,
    K_warm: int,
    epsilon: float,
    huber_delta: float,
    beta_mult: float,
    tol_outer: float = 1e-3,
    tol_irls: float = 1e-3,
    K_min: int = 1,
    eps_div: float = 1e-12,
    return_history: bool = False,
    prox_op: str = "lp",
    fit_t: bool = True,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[list[float]],
    dict,
]:
    """Process a single (m, n) tile. Returns (Q, z, s_combined, t, history, diag).

    `fit_t=False` disables the per-input-column scale `t`: it stays the all-ones
    vector throughout (no Sinkhorn warm-start of `t`, no t-IRLS sweep), so the
    tile reduces to a grouped robust-BCD affine quantizer (per-(output,group)
    `s, z` only). This is the W4A4-realizable RoDS-A variant (RoDS-A Phase 2,
    doc 3.4): a per-input-GROUP `t` would be redundant with the per-(output,
    group) `s`, and a per-COLUMN `t` cannot fold into a grouped INT4 kernel, so
    the Sinkhorn dual-scale is dropped for the composed W4A4 path.

    `history` (only populated when return_history=True) keeps the legacy shape
    of length K_outer+1 (initial objective + one entry per outer iter actually
    run) so existing summed-across-tiles callers stay compatible. The per-tile
    `objective_per_iter` exposed via `diag` is the in-loop curve only.
    """
    m, n = matrix.shape
    device, dtype = matrix.device, matrix.dtype

    if K_warm > 0 and fit_t:
        _, mu1, mu2 = sinkhorn_log(matrix, order=K_warm)
        # sinkhorn_log returns mu1 as 1D (n,); reshape so the K_outer=0 path
        # (which skips the IRLS sweeps that would otherwise re-rank t) still
        # produces a 2D tile that stacks correctly downstream.
        s = mu2.to(dtype).reshape(m, 1)
        t = mu1.to(dtype).reshape(1, n)
    else:
        # fit_t=False: t is pinned to ones (Sinkhorn jointly normalises s and t,
        # so there is no clean "s-only" warm-start; fall back to the s=t=1 init
        # and let the HQQ inner + s-IRLS carry the grouped affine fit).
        s = torch.ones((m, 1), device=device, dtype=dtype)
        t = torch.ones((1, n), device=device, dtype=dtype)

    Q, z, s_q = _hqq_inner(matrix / s / t, min_max, p, huber_delta, beta_mult, prox_op)

    history: Optional[list[float]] = None
    init_obj = _objective_lp(matrix, s, t, Q, z, s_q, p)
    if return_history:
        history = [init_obj]

    # theta_best snapshots the (Q, z, s, t, s_q) tuple at the iteration with
    # lowest in-loop objective seen so far. Initialise with the pre-iter state
    # and L_best = init_obj so K_outer=0 still returns sensible parameters.
    L_prev = float("inf")
    L_best = init_obj
    theta_best = (Q.clone(), z.clone(), s.clone(), t.clone(), s_q.clone())
    objective_per_iter: list[float] = []
    total_irls_iters = 0
    k_stop = 0

    for k in range(1, K_outer + 1):
        A = s_q * (Q.to(dtype) - z)

        if fit_t:
            c_mat = s * A
            for _ in range(K_irls):
                t_prev = t.clone()
                R = matrix - c_mat * t
                Wt = (epsilon + R * R) ** ((p - 2.0) / 2.0)
                num = (Wt * c_mat * matrix).sum(dim=0, keepdim=True)
                den = (Wt * c_mat * c_mat).sum(dim=0, keepdim=True)
                t = num / (den + epsilon)
                total_irls_iters += 1
                rel_change = (t - t_prev).abs().max().item() / (
                    t_prev.abs().max().item() + eps_div
                )
                if rel_change < tol_irls:
                    print(f"Outer iter {k}, t converged after {_+1} IRLS iterations.")
                    break

        c_mat = A * t
        for _ in range(K_irls):
            s_prev = s.clone()
            R = matrix - s * c_mat
            Wt = (epsilon + R * R) ** ((p - 2.0) / 2.0)
            num = (Wt * c_mat * matrix).sum(dim=1, keepdim=True)
            den = (Wt * c_mat * c_mat).sum(dim=1, keepdim=True)
            s = num / (den + epsilon)
            total_irls_iters += 1
            rel_change = (s - s_prev).abs().max().item() / (
                s_prev.abs().max().item() + eps_div
            )
            if rel_change < tol_irls:
                print(f"Outer iter {k}, s converged after {_+1} IRLS iterations.")
                break

        Q, z, s_q = _hqq_inner(matrix / s / t, min_max, p, huber_delta, beta_mult, prox_op)

        L_curr = _objective_lp(matrix, s, t, Q, z, s_q, p)
        objective_per_iter.append(L_curr)
        if return_history:
            history.append(L_curr)

        if L_curr < L_best:
            L_best = L_curr
            theta_best = (Q.clone(), z.clone(), s.clone(), t.clone(), s_q.clone())

        k_stop = k
        rel_improvement = (L_prev - L_curr) / (abs(L_prev) + eps_div)
        if k >= K_min and rel_improvement < tol_outer:
            print(f"Outer iter {k}, converged after {k_stop} iterations.")
            break
        L_prev = L_curr

    Q, z, s, t, s_q = theta_best
    if objective_per_iter:
        best_is_final = L_best == objective_per_iter[-1]
    else:
        best_is_final = True
    s_combined = s_q * s

    diag = {
        "objective_per_iter": objective_per_iter,
        "k_stop": k_stop,
        "best_is_final": bool(best_is_final),
        "total_irls_iters": total_irls_iters,
    }
    return Q, z, s_combined, t, history, diag


def _log_tile_diag(layer_name: Optional[str], tile_idx: int, diag: dict) -> None:
    """Emit one wandb row per (layer, tile) with the per-iter convergence curve."""
    if _wandb is None or _wandb.run is None or layer_name is None:
        return
    _wandb.log(
        {
            "layer": layer_name,
            "tile": tile_idx,
            "rods/objective_per_iter": diag["objective_per_iter"],
            "rods/k_stop": diag["k_stop"],
            "rods/best_is_final": diag["best_is_final"],
        }
    )


def quantize_rods(
    W: torch.Tensor,
    *,
    nbits: int = 4,
    group_size: int = 64,
    p: float = 0.7,
    K_outer: int = 3,
    K_irls: int = 5,
    K_warm: int = 1,
    epsilon: float = 1e-6,
    huber_delta: float = 1.0,
    beta_mult: float = 1.0,
    tol_outer: float = 1e-3,
    tol_irls: float = 1e-3,
    K_min: int = 1,
    eps_div: float = 1e-12,
    prox_op: str = "lp",
    layer_name: Optional[str] = None,
    device: str = "cpu",
    compute_dtype: torch.dtype = torch.float32,
    bitpack: bool = True,
    return_history: bool = False,
    return_diags: bool = False,
    fit_t: bool = True,
) -> tuple:
    """Quantize a 2D weight tensor with Dual-Scale HQQ.

    Returns (W_q, meta) matching sinq.quantizer.Quantizer.quantize's output:
    meta["scale"] has shape (out_f * n_tpr, 1), meta["scale2"] has shape
    (1, in_f), meta["zero"] has shape (out_f * n_tpr, 1). The artifact
    builder reshapes scale/zero to (out_f, n_tpr).

    Extra returns:
      return_history=True  -> appends `merged_history` (summed across tiles).
      return_diags=True    -> appends `tile_diags` (per-tile dicts).
    When both are true, the order is (W_q, meta, merged_history, tile_diags).

    `layer_name` is forwarded to the per-tile wandb logger so each row can be
    keyed by (layer, tile).
    """
    assert prox_op in {"lp", "huber", "l2sq"}, (
        f"prox_op must be 'lp', 'huber', or 'l2sq', got {prox_op!r}"
    )
    W = W.to(device).to(torch.float32)
    out_f, in_f = W.shape
    assert in_f % group_size == 0, (
        f"in_f={in_f} must be divisible by group_size={group_size}"
    )
    n_tpr = in_f // group_size
    min_max = [0, 2 ** nbits - 1]

    tiles_Q: list[torch.Tensor] = []
    tiles_s: list[torch.Tensor] = []
    tiles_z: list[torch.Tensor] = []
    tiles_t: list[torch.Tensor] = []
    tile_histories: list[list[float]] = []
    tile_diags: list[dict] = []

    for j in range(n_tpr):
        tile = W[:, j * group_size : (j + 1) * group_size]
        Q_j, z_j, s_j, t_j, hist_j, diag_j = _rods_tile(
            tile,
            min_max,
            p=p,
            K_outer=K_outer,
            K_irls=K_irls,
            K_warm=K_warm,
            epsilon=epsilon,
            huber_delta=huber_delta,
            beta_mult=beta_mult,
            tol_outer=tol_outer,
            tol_irls=tol_irls,
            K_min=K_min,
            eps_div=eps_div,
            return_history=return_history,
            prox_op=prox_op,
            fit_t=fit_t,
        )
        tiles_Q.append(Q_j)
        tiles_z.append(z_j)
        tiles_s.append(s_j)
        tiles_t.append(t_j)
        tile_diags.append(diag_j)
        if hist_j is not None:
            tile_histories.append(hist_j)
        _log_tile_diag(layer_name, j, diag_j)

    Q_stack = torch.stack(tiles_Q, dim=0)
    Q_packed = Q_stack.permute(1, 0, 2).reshape(-1, group_size)
    s_packed = torch.stack(tiles_s, dim=0).permute(1, 0, 2).reshape(-1, 1)
    z_packed = torch.stack(tiles_z, dim=0).permute(1, 0, 2).reshape(-1, 1)
    t_packed = torch.stack(tiles_t, dim=0).permute(1, 0, 2).reshape(1, -1)

    merged_history: Optional[list[float]] = None
    if return_history and tile_histories:
        # Outer-loop early stopping can give different per-tile history lengths.
        # Pad shorter tiles with their final (= best-so-far) objective so the
        # summed-across-tiles curve has well-defined length max(per-tile len).
        L = max(len(h) for h in tile_histories)
        merged_history = []
        for i in range(L):
            merged_history.append(
                sum(h[i] if i < len(h) else h[-1] for h in tile_histories)
            )

    packing = Quantizer.bit_to_packing[nbits]
    if bitpack:
        Q_int = Q_packed.to(torch.uint8)
        W_q = Quantizer.pack[packing](Q_int)
    else:
        W_q = Q_packed.to(compute_dtype)

    meta = {
        "nbits": nbits,
        "group_size": group_size,
        "shape": (out_f, in_f),
        "scale": s_packed.to(compute_dtype),
        # fit_t=False -> t is pinned to ones; store None so Quantizer.dequantize
        # takes the s2_eff=1 path (no per-column scale) and the artifact records
        # that this is a grouped-affine-only (W4A4-realizable) fit.
        "scale2": t_packed.to(compute_dtype) if fit_t else None,
        "awq_scale": None,
        "zero": z_packed.to(compute_dtype),
        "axis": 1,
        "packing": packing if bitpack else None,
        "method": "rods",
        "prox_op": prox_op,
        "compute_dtype": compute_dtype,
        "unpack_view_dtype": Quantizer.unpack_view_dtype[packing] if bitpack else None,
        "view_as_float": False,
    }

    out: tuple = (W_q, meta)
    if return_history:
        out = out + (merged_history,)
    if return_diags:
        out = out + (tile_diags,)
    return out


def quantize_rods_inplace(
    transformer,
    *,
    nbits: int = 4,
    group_size: int = 64,
    p: float = 0.7,
    K_outer: int = 3,
    K_irls: int = 5,
    K_warm: int = 1,
    epsilon: float = 1e-6,
    huber_delta: float = 1.0,
    beta_mult: float = 1.0,
    tol_outer: float = 1e-3,
    tol_irls: float = 1e-3,
    K_min: int = 1,
    eps_div: float = 1e-12,
    prox_op: str = "lp",
    device: str = "cuda",
    ignore_layer_names: Optional[list] = None,
) -> int:
    """Fake-quantize all nn.Linear weights via Dual-Scale HQQ in place.

    Mirrors the `quantize_rtn(...)` pattern in scripts/fid_eval.py: quantize
    each weight, dequantize, write fp16/bf16 reconstruction back into the
    nn.Linear so generation can proceed without a SINQLinear wrapper.

    Layers whose in_features is not divisible by `group_size` are skipped.
    Returns the number of layers quantized.
    """
    import torch.nn as nn

    ignore_layer_names = ignore_layer_names or []
    n_quantized = 0
    n_skipped = 0

    for name, module in transformer.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name.split(".")[-1] in ignore_layer_names:
            continue

        W = module.weight.data
        if W.shape[1] % group_size != 0:
            n_skipped += 1
            continue

        W_q, meta = quantize_rods(
            W,
            nbits=nbits,
            group_size=group_size,
            p=p,
            K_outer=K_outer,
            K_irls=K_irls,
            K_warm=K_warm,
            epsilon=epsilon,
            huber_delta=huber_delta,
            beta_mult=beta_mult,
            tol_outer=tol_outer,
            tol_irls=tol_irls,
            K_min=K_min,
            eps_div=eps_div,
            prox_op=prox_op,
            layer_name=name,
            device=device,
            compute_dtype=torch.float32,
            bitpack=True,
        )

        W_dq = Quantizer.dequantize(W_q, meta)
        module.weight.data = W_dq.to(W.dtype).to(W.device)
        n_quantized += 1

    if n_skipped > 0:
        print(
            f"[rods] Skipped {n_skipped} layer(s) whose in_features "
            f"is not divisible by group_size={group_size}."
        )
    return n_quantized
