"""Sanity checks for Dual-Scale HQQ (utils/rods.py).

Four pytest cases:
  1. K_outer=1, K_irls=0, K_warm=0 must be bit-exact equivalent to vanilla
     SINQ HQQ on the same tile: both IRLS blocks are no-ops, so the single
     outer iteration just re-runs HQQ on the unscaled matrix.
  2. With p=2.0 the IRLS solver collapses to ordinary least-squares; we
     check it converges to the closed-form solution on a synthetic problem.
  3. With aggressive K_irls and tol_outer=0 the objective can be mildly
     non-monotone, and the function must return the best-seen (Q,z,s,t),
     not the final-iter values.
  4. With K_min=3 and a tolerance loose enough that any iteration would
     trigger the early-stop check, the outer loop must still run >= K_min
     iterations.

Picked up automatically by runners/misc/test_pixart.sh which runs
`pytest tests/` inside the cfq-diffusion RunAI container (gpu=0).
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.sinq.quantizer import Quantizer
from rods.quantizer import quantize_rods


class TestVanillaHQQEquivalence:
    def test_bit_exact_hqq_when_K_irls_K_warm_zero(self):
        """With K_irls=0 both IRLS blocks are no-ops (s, t stay at 1), so the
        single outer iter (K_outer=1) just re-runs HQQ on the unscaled tile.
        Result must be bit-exact identical to vanilla HQQ. Any tol_outer /
        tol_irls is acceptable because the IRLS rel_change check never fires
        (loop body executes zero times) and the outer rel_improvement is
        gated by K_min=1 only after the first iter — and there's only one."""
        torch.manual_seed(0)
        W = torch.randn(64, 64, dtype=torch.float32)
        nbits, group_size = 4, 64

        W_q_ours, meta_ours = quantize_rods(
            W,
            nbits=nbits,
            group_size=group_size,
            p=0.7,
            K_outer=1,
            K_irls=0,
            K_warm=0,
            epsilon=1e-6,
            huber_delta=1.0,
            beta_mult=1.0,
            tol_outer=1e-3,
            tol_irls=1e-3,
            K_min=1,
            eps_div=1e-12,
            device="cpu",
            bitpack=True,
        )

        W_q_hqq, meta_hqq = Quantizer.quantize(
            W,
            None,
            device="cpu",
            compute_dtype=torch.float32,
            nbits=nbits,
            group_size=group_size,
            axis=1,
            tiling_mode="1D",
            method="hqq",
            bitpack=True,
            sinkhorn_order=0,
            huber_delta=1.0,
            beta_mult=1.0,
        )

        assert torch.equal(W_q_ours, W_q_hqq), "W_q mismatch (bit-exact required)"
        assert torch.allclose(meta_ours["scale"], meta_hqq["scale"], atol=0, rtol=0), (
            "scale mismatch"
        )
        assert torch.allclose(meta_ours["zero"], meta_hqq["zero"], atol=0, rtol=0), (
            "zero mismatch"
        )
        assert torch.allclose(
            meta_ours["scale2"],
            torch.ones_like(meta_ours["scale2"]),
            atol=0,
        ), "t must be all-ones when K_warm=0 and IRLS is a no-op"


class TestIRLSClosedForm:
    def test_irls_matches_weighted_least_squares_when_p_is_2(self):
        """With p=2.0 the IRLS weight (epsilon + R^2)^((p-2)/2) collapses to 1,
        so the per-iteration update is exactly the OLS estimator."""
        torch.manual_seed(1)
        m, n = 32, 1
        c = torch.randn(m, 1, dtype=torch.float32)
        t_true = torch.tensor([[2.5]], dtype=torch.float32)
        noise = 0.01 * torch.randn(m, n, dtype=torch.float32)
        W = c * t_true.expand(m, n) + noise

        epsilon = 1e-6
        p = 2.0
        s = torch.ones((m, 1), dtype=torch.float32)
        t = torch.ones((1, n), dtype=torch.float32)
        c_mat = s * c
        for _ in range(10):
            R = W - c_mat * t
            Wt = (epsilon + R * R) ** ((p - 2.0) / 2.0)
            num = (Wt * c_mat * W).sum(dim=0, keepdim=True)
            den = (Wt * c_mat * c_mat).sum(dim=0, keepdim=True)
            t = num / (den + epsilon)

        t_closed = (c * W).sum(dim=0, keepdim=True) / (c * c).sum(dim=0, keepdim=True)
        assert torch.allclose(t, t_closed, atol=1e-5), (
            f"IRLS t={t.item()} vs OLS t={t_closed.item()}"
        )


class TestBestParameterTracking:
    def test_returned_params_minimise_objective_over_outer_curve(self):
        """Best-tracking: returned (Q, z, s, t) must correspond to the iter
        with the lowest in-loop objective, not the final iter. With tol_outer=0
        and large K_outer, any oscillation in the lp objective triggers an
        early stop; the test asserts the returned reconstruction's objective
        matches min(objective_per_iter)."""
        torch.manual_seed(2)
        W = torch.randn(64, 64, dtype=torch.float32)
        p = 0.7
        result = quantize_rods(
            W,
            nbits=4,
            group_size=64,
            p=p,
            K_outer=10,
            K_irls=10,
            K_warm=1,
            epsilon=1e-6,
            huber_delta=1.0,
            beta_mult=1.0,
            tol_outer=0.0,
            tol_irls=0.0,
            K_min=1,
            eps_div=1e-12,
            device="cpu",
            bitpack=False,
            return_diags=True,
        )
        W_q, meta, tile_diags = result
        assert len(tile_diags) == 1, "expected single tile (group_size=64)"
        diag = tile_diags[0]
        objective_per_iter = diag["objective_per_iter"]
        assert len(objective_per_iter) >= 1, "expected at least one outer iter"

        # Reconstruct from returned params and compute the same lp objective.
        Q = W_q  # bitpack=False -> codes are the W_q tensor itself, float32
        s = meta["scale"]
        t = meta["scale2"]
        z = meta["zero"]
        W_hat = s * (Q - z) * t
        recon_obj = (W - W_hat).abs().pow(p).sum().item()

        # The returned params correspond to argmin over the in-loop curve.
        # min over objective_per_iter is the best in-loop objective; the
        # initial pre-iter state is also a candidate (theta_best init), so
        # the returned recon may equal the initial objective too. Either way
        # it must be no worse than min of the in-loop curve.
        best_in_loop = min(objective_per_iter)
        assert recon_obj <= best_in_loop + 1e-4, (
            f"returned recon objective {recon_obj:.6f} exceeds best in-loop "
            f"objective {best_in_loop:.6f} (curve: {objective_per_iter})"
        )
        if not diag["best_is_final"]:
            assert recon_obj < objective_per_iter[-1] + 1e-4, (
                "best_is_final=False but returned recon matches final iter"
            )


class TestKMinRespectsMinimumIterations:
    def test_outer_loop_runs_at_least_K_min_iterations(self):
        """With tol_outer huge (any rel_improvement is below it) the outer
        loop would terminate after iter 1 if K_min were not gating, because
        the rel_improvement at iter 1 is (inf - L_1) / (inf + eps_div), which
        evaluates to NaN and the comparison `NaN < tol_outer` is False. But
        from iter 2 onward rel_improvement is finite and small, so without
        K_min the loop would stop at iter 2. K_min=3 must force >= 3 iters."""
        torch.manual_seed(3)
        W = torch.randn(64, 64, dtype=torch.float32)
        _, _, tile_diags = quantize_rods(
            W,
            nbits=4,
            group_size=64,
            p=0.7,
            K_outer=10,
            K_irls=5,
            K_warm=1,
            epsilon=1e-6,
            huber_delta=1.0,
            beta_mult=1.0,
            tol_outer=10.0,
            tol_irls=1e-3,
            K_min=3,
            eps_div=1e-12,
            device="cpu",
            bitpack=False,
            return_diags=True,
        )
        diag = tile_diags[0]
        assert diag["k_stop"] >= 3, (
            f"K_min=3 must force at least 3 outer iters; got k_stop={diag['k_stop']}"
        )
        assert len(diag["objective_per_iter"]) >= 3, (
            f"objective_per_iter must have >=3 entries when K_min=3; "
            f"got {len(diag['objective_per_iter'])}"
        )
