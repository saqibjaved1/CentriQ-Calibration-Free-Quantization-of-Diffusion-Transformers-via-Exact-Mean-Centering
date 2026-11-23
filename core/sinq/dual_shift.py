# Vendored verbatim from sinq/dual_shift.py @ commit 91dc9fb
# (https://github.com/huawei-csl/SINQ). Do not edit in place;
# changes go upstream then re-vendor.

'''
written by SINQ authors
'''

import torch
import torch.nn as nn
import numpy as np
from torch import vmap
import torch.nn.functional as F
from .sinkhorn import *

NF4_CODEBOOK = torch.tensor([
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
     0.0,
     0.07958029955625534,
     0.16093020141124725,
     0.24611230194568634,
     0.33791524171829224,
     0.44070982933044434,
     0.5626170039176941,
     0.7229568362236023,
     1.0
], dtype=torch.float32)

# NF3 codebook (~[-1, 1], 8 non-uniform levels from the paper "Asymmetric Floating Point Quantization for LLMs")
NF3_CODEBOOK = torch.tensor([
    -1.0,
    -0.5350227355957031,
    -0.2469314038753510,
     0.0,
     0.1833375245332718,
     0.3819939494132996,
     0.6229856610298157,
     1.0
], dtype=torch.float32)

def quantize_rtn(
    matrix: torch.Tensor,
    min_max=[],
    niter=None,
    mode: str = "uniform",        # "uniform" | "nf4" | "nf3"
    nf_use_shift: bool = True,    # applies to both NF4 and NF3
    nf_shift_kind: str = "mean"   # "minmax" | "mean"
):
    w = matrix
    orig_dtype = w.dtype
    dev = w.device
    w = w.to(torch.float32)

    if mode.lower() in ("nf4", "nf3"):
        cb = NF4_CODEBOOK if mode.lower() == "nf4" else NF3_CODEBOOK

        if not nf_use_shift:
            scales = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-4)
            zeros  = torch.zeros_like(scales)
            norm   = w / scales
        else:
            if nf_shift_kind == "minmax":
                w_max = w.amax(dim=1, keepdim=True)
                w_min = w.amin(dim=1, keepdim=True)
                denom  = (w_max - w_min).clamp_min(1e-4)
                scales = denom / 2.0
                zeros  = - (w_max + w_min) / denom
                norm   = w / scales + zeros
            elif nf_shift_kind == "mean":
                mu     = w.mean(dim=1, keepdim=True)
                x      = w - mu
                scales = x.abs().amax(dim=1, keepdim=True).clamp_min(1e-4)
                zeros  = (-mu / scales)
                norm   = w / scales + zeros
            else:
                raise ValueError("nf_shift_kind must be 'minmax' or 'mean'")

        cb = cb.to(dev, dtype=norm.dtype)
        q  = (norm.unsqueeze(-1) - cb.view(1,1,-1)).abs().argmin(dim=-1).to(torch.int8)
        return q.contiguous(), scales.to(orig_dtype), zeros.to(orig_dtype), orig_dtype

    max_val = w.amax(dim=1, keepdim=True)
    min_val = w.amin(dim=1, keepdim=True)
    max_int = min_max[1]
    min_int = min_max[0]
    scales  = (max_val - min_val).clamp(min=1e-4) / max_int
    zeros   = -torch.round(min_val / scales)
    q       = torch.clamp(torch.round(w / scales + zeros), min_int, max_int).to(torch.int8)
    return q.contiguous(), scales.to(orig_dtype), zeros.to(orig_dtype), orig_dtype


def shrink_lp_op(x, beta, lp_norm):
    if lp_norm == 1:
        out = torch.abs(x)
        out.sub_(1.0 / beta).clamp_min_(0.0)
        out.mul_(torch.sign(x))
        return out
    else:
        out = torch.abs(x).clip(1e-5, 1e5)
        out.sub_((1.0 / beta) * out.pow(lp_norm - 1)).clamp_min_(0.0)
        out.mul_(torch.sign(x))
        return out

def prox_l2sq(v, beta):
    # prox of (1/beta) * x^2/2, i.e. lambda=1/beta: v/(1+lambda) = v*beta/(beta+1)
    return v * (beta / (beta + 1))

def prox_huber(v, beta, delta):
    # prox of (1/beta) * Huber_delta(x) where Huber_delta(x) = x^2/2 if |x|<=delta, else delta*(|x|-delta/2)
    # inner region |v| <= delta*(beta+1)/beta: v*beta/(beta+1)  [l2 shrinkage]
    # outer region |v| >  delta*(beta+1)/beta: sign(v)*(|v| - delta/beta)  [soft-threshold by delta/beta]
    threshold = delta * (beta + 1) / beta
    abs_v = v.abs()
    l2_part = v * (beta / (beta + 1))
    soft_part = v.sign() * (abs_v - delta / beta).clamp_min(0.0)
    return torch.where(abs_v <= threshold, l2_part, soft_part)

def _dispatch_prox(residual, beta, lp_norm, prox_op, huber_delta):
    if prox_op == 'l2sq':
        return prox_l2sq(residual, beta)
    elif prox_op == 'huber':
        return prox_huber(residual, beta, huber_delta)
    else:
        return shrink_lp_op(residual, beta, lp_norm)

def optimize_weights_proximal_legacy_step(W_f, scale, zero, min_max, beta=1e1, lp_norm=.7, axis=1, prox_op='lp', huber_delta=1.0):
    W_q = torch.round(W_f * scale + zero).clamp_(min_max[0], min_max[1])
    W_r = (W_q - zero) / scale
    W_e = _dispatch_prox(W_f - W_r, beta, lp_norm, prox_op, huber_delta)
    zero = torch.mean(W_q - (W_f - W_e) * scale, axis=axis, keepdim=True)
    return W_r, W_q, zero, scale

    
def optimize_weights_joint_step(W_f, s, z, min_max, beta=1e1, lp_norm=0.7, prox_op='lp', huber_delta=1.0):
    """Single HQQ-Joint step: mean-update for z and closed-form LS for scale s, applied jointly."""
    W_q = torch.round(W_f / s + z).clamp_(min_max[0], min_max[1])
    W_r = (W_q - z) * s
    W_e = _dispatch_prox(W_f - W_r, beta, lp_norm, prox_op, huber_delta)
    z = torch.mean(W_q - W_f / s, dim=1, keepdim=True)
    u = (W_q - z).float()
    numer = ((W_f - W_e).float() * u).sum(dim=1, keepdim=True)
    denom = (u * u).sum(dim=1, keepdim=True).clamp(min=1e-8)
    s_new = (numer / denom).clamp(1e-5, 1e5).to(W_f.dtype)

    return W_r, W_q, z, s_new

def hqq_rtn(W, min_max=[]):
    dtype = W.dtype
    max_v = min_max[1]
    min_v = min_max[0]
    _max = W.max(axis=1, keepdim=True)[0]
    _min = W.min(axis=1, keepdim=True)[0]
    # Note: here we work with the inverse of the scale to avoid division and quantize instead via W*scale + zero, the scale is inverted later on.
    denom = (_max - _min)
    scale = (max_v / denom)
    scale = torch.where(denom.abs() <= 1e-4, torch.full_like(scale, 1.0), scale) #Avoid small denom values
    scale = scale.clamp(max=2e4) # clamp to avoid half-precision problems
    zero = -_min * scale
    W_q = (W * scale + zero).round_().clamp_(min_max[0], min_max[1])
    return W_q, scale, zero, dtype


def quantize_symmetric_rtn(matrix, min_max, niter=None):
    w = matrix
    dtype = w.dtype
    max_val = w.abs().amax(dim=1, keepdim=True)
    max_int = min_max[1]//2 - 1
    min_int = -min_max[1]//2
    scales = (max_val).clamp(min=1e-5) / max_int
    q = torch.round(w/scales).clip(min_int, max_int)
    return q, scales, dtype

def dequantize_symmetric_rtn(q, scales, dtype):
    return (q*scales).to(dtype)

def quantize_dual_scale_shift(matrix, min_max, method='sinq', awq_scale=None, col_target=None, sinkhorn_order=16, huber_delta=1.0, beta_mult=1.0):
    dtype = matrix.dtype
    dev = matrix.device
    matrix = matrix.float()

    # normalize the matrix with sinkhorn inspired std.dev. scaling
    # matrix, mu1, mu2 = min_kurt_vectors_vmap(matrix,32) # for kurtosis exp.
    matrix, mu1, mu2 = sinkhorn_log(matrix, sinkhorn_order, col_target=col_target)

    if not ('sinq' in method):
        matrix = matrix * mu1 * mu2
        mu1 = torch.ones_like(mu1)
        mu2 = torch.ones_like(mu2)

    if 'awq' in method and awq_scale is not None:
        matrix = matrix * awq_scale
        mu1 = mu1 / awq_scale.float()

    if 'l2sq' in method:
        prox_op = 'l2sq'
    elif 'huber' in method:
        prox_op = 'huber'
    else:
        prox_op = 'lp'

    if not ('hqq' in method):
        if 'noz' in method:
            q, scales, _ = quantize_symmetric_rtn(matrix, min_max)
            q = q + min_max[1]//2
            z = torch.tensor(min_max[1] // 2) # note that this is a single constant
        else:
            if "nf4" in method.lower():
                q, scales, z, _ = quantize_rtn(matrix, min_max, mode="nf4")
            elif "nf3" in method.lower():
                q, scales, z, _ = quantize_rtn(matrix, min_max, mode="nf3")
            else:
                q, scales, z, _ = quantize_rtn(matrix, min_max, mode="uniform")

    if 'hqq_joint' in method:
        assert not ('noz' in method), 'noz incompatible with hqq_joint'
        q, scale_inv, z, _ = hqq_rtn(matrix, min_max)
        z = torch.round(z)                               # initialise z as integer zero-point
        s = (1.0 / scale_inv.clamp(1e-7, 1e7))          # convert to actual scale domain
        best_error = torch.inf
        best_s = s.clone()
        best_z = torch.zeros_like(z)
        beta = 1e1
        for i in range(20):
            W_r, W_q, z, s = optimize_weights_joint_step(matrix, s, z, min_max, beta=beta, prox_op=prox_op, huber_delta=huber_delta)
            current_error = torch.abs(matrix - W_r).mean().float()
            take = current_error < best_error
            best_error = torch.where(take, current_error, best_error)
            best_s     = torch.where(take[..., None], s, best_s)
            best_z     = torch.where(take[..., None], z, best_z)
            beta *= beta_mult
        scales = best_s
        z = best_z
        q = W_q

    elif 'hqq' in method:
        assert not ('noz' in method), 'noz incompatible with hqq'
        # alternative approach, use hqq after normalization
        q, scales, z, _ = hqq_rtn(matrix, min_max)
        best_error = torch.inf
        best_z = torch.zeros_like(z)
        best_scales = torch.ones_like(scales)
        beta = 1e1
        for i in range(20):
            W_r, W_q, z, scales = optimize_weights_proximal_legacy_step(matrix, scales.clip(1e-5,1e5), z, min_max, beta=beta, prox_op=prox_op, huber_delta=huber_delta)
            current_error = torch.abs(matrix - W_r).mean().float()
            take = current_error < best_error
            best_error  = torch.where(take, current_error, best_error)
            best_z      = torch.where(take[..., None],           z, best_z)
            best_scales = torch.where(take[..., None],      scales, best_scales)
            beta *= beta_mult
        scales = best_scales
        z = best_z
        q = W_q
        scales = 1/scales

    scales2 = torch.ones(1,matrix.shape[1]).to(dev).to(mu1.dtype) * mu1
    scales = scales*mu2

    q = q.to(dtype).to(dev)
    s1 = (scales).to(dtype) 
    s2 = (scales2).to(dtype) 
    z = (z).to(dtype).to(dev)

    return q, s1.to(dev), s2.to(dev), z
 

def tiled_quant_rectangle(M, min_max, block, method='dual', awq_scale=None, col_target=None, sinkhorn_order=16, huber_delta=1.0, beta_mult=1.0):
    q = quantize_dual_scale_shift
    mshape = M.shape
    H,W = M.shape
    block = int(block)
    orig = block

    while block >= 16 and (W % block) != 0:
        block //= 2

    assert (W % block) == 0 and block >= 16, f"block must divide W (W={W}, block={block})"
    if block != orig:
        print(f"[SINQ] Adjusted tile {orig} -> {block} for W={W}", flush=True)
    n_w = W//block

    M = M.view(H, W//block, block)
    M_batched = M.permute(1,0,2).contiguous().view(n_w, H, block)

    if awq_scale is not None and col_target is not None:
        awq_scale = awq_scale.view(n_w, 1, block)
        col_target_t = col_target.to(M_batched.device).view(n_w, block)
        def process_block(mat, aws, ct):
            return q(mat, min_max, method=method, awq_scale=aws, col_target=ct, sinkhorn_order=sinkhorn_order, huber_delta=huber_delta, beta_mult=beta_mult)
        Q, s1, s2, z = vmap(process_block, randomness='different')(M_batched, awq_scale, col_target_t)
    elif awq_scale is not None:
        awq_scale = awq_scale.view(n_w, 1, block)
        def process_block(mat, aws):
            return q(mat, min_max, method=method, awq_scale=aws, sinkhorn_order=sinkhorn_order, huber_delta=huber_delta, beta_mult=beta_mult)
        Q, s1, s2, z = vmap(process_block, randomness='different')(M_batched, awq_scale)
    elif col_target is not None:
        col_target_t = col_target.to(M_batched.device).view(n_w, block)
        def process_block(mat, ct):
            return q(mat, min_max, method=method, col_target=ct, sinkhorn_order=sinkhorn_order, huber_delta=huber_delta, beta_mult=beta_mult)
        Q, s1, s2, z = vmap(process_block, randomness='different')(M_batched, col_target_t)
    else:
        def process_block(mat):
            return q(mat, min_max, method=method, sinkhorn_order=sinkhorn_order, huber_delta=huber_delta, beta_mult=beta_mult)
        Q, s1, s2, z = vmap(process_block, randomness='different')(M_batched)

    if 'noz' in method:
        z = z.view(-1)[0]
    else:
        z = z.permute(1,0,2).reshape(-1,1)

    Q = Q.permute(1,0,2).reshape(-1, block)
    s1 = s1.permute(1,0,2).reshape(-1,1)
    s2 = s2.permute(1,0,2).view(1, -1)
    return Q, s1, s2, z


def tiled_quant_square(M, min_max, block, method='dual', awq_scale=None, sinkhorn_order=16, huber_delta=1.0, beta_mult=1.0):
    q = quantize_dual_scale_shift
    H,W = M.shape
    assert H%block==0 and W%block==0, 'block must divie H and W'
    n_w = W//block
    n_h = H//block

    M = M.view(H//block, block, W//block, block)
    M_batched = M.permute(0, 2, 1, 3).contiguous().view(-1, block, block)

    if not (awq_scale is None):
        awq_scale = awq_scale.view(1, 1, n_w, block).repeat(n_h, 1, 1, 1).permute(0,2,1,3).contiguous().view(-1, 1, block)
        def process_block(mat, awq_scale):
            return q(mat, min_max, method=method, awq_scale=awq_scale, sinkhorn_order=sinkhorn_order, huber_delta=huber_delta, beta_mult=beta_mult)
        Q, s1, s2, z = vmap(process_block, randomness='different')(M_batched, awq_scale)
    else:
        def process_block(mat):
            return q(mat, min_max, method=method, sinkhorn_order=sinkhorn_order, huber_delta=huber_delta, beta_mult=beta_mult)
        Q, s1, s2, z = vmap(process_block, randomness='different')(M_batched)


    if 'noz' in method:
        z = z.view(-1)[0]
    else:
        z = z.view(-1,block,1)

    s1 = s1.view(-1,block,1)
    s2 = s2.view(-1,1,block)
    Q = Q.view(n_h, n_w, block, block).permute(0, 2, 1, 3).contiguous()
    Q = Q.view(-1, block)
    return Q, s1, s2, z
