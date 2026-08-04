"""Fused chunked CE + z-loss Triton kernel.

Online-softmax pass writing (ce_sum, ce_count, z_accum) via atomic_add.
Backward is a re-compute stub.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# Vocab is the per-block reduction axis; 128k fits, 256k would need 2 programs/row.
_MAX_VOCAB_BLOCK = 131072


def cross_entropy_with_z_pytorch(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int,
    z_loss_weight: float,
) -> torch.Tensor:
    """Reference: ce + z_loss_weight * mean(logsumexp(logits.float())**2)."""
    log_z = torch.logsumexp(logits.float(), dim=-1)
    ce = F.cross_entropy(logits, targets, ignore_index=ignore_index, reduction="mean")
    z = log_z.pow(2).mean()
    return ce + z_loss_weight * z


if HAS_TRITON:

    @triton.jit
    def _ce_z_fwd_kernel(
        L_ptr,            # (M, V) logits
        T_ptr,            # (M,)   targets
        CE_SUM_ptr,       # (1,)   accumulator
        CE_CNT_ptr,       # (1,)   accumulator
        Z_SUM_ptr,        # (1,)   accumulator (sum of z**2, NOT mean)
        ignore_index,
        z_loss_weight,
        M,
        V: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        row = tl.program_id(0)
        if row >= M:
            return

        target = tl.load(T_ptr + row)
        valid = target != ignore_index

        # Online softmax over vocab.
        cols = tl.arange(0, BLOCK_V)
        mask = cols < V
        x = tl.load(L_ptr + row * V + cols, mask=mask, other=-float("inf")).to(tl.float32)

        m = tl.max(x, axis=0)
        x_shift = x - m
        l = tl.sum(tl.exp(x_shift), axis=0)
        log_z = m + tl.log(l)

        # nll: pull the target logit; protect against ignore_index.
        target_logit = tl.load(L_ptr + row * V + target).to(tl.float32)
        nll = log_z - target_logit

        if valid:
            tl.atomic_add(CE_SUM_ptr, nll)
            tl.atomic_add(CE_CNT_ptr, 1.0)
        # z-loss mean is computed outside as Z_SUM / M.

        tl.atomic_add(Z_SUM_ptr, log_z * log_z)


def _triton_ce_z_forward(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int,
    z_loss_weight: float,
) -> torch.Tensor:
    logits_2d = logits.reshape(-1, logits.shape[-1]).contiguous()
    targets_1d = targets.reshape(-1).contiguous().to(torch.int64)
    M, V = logits_2d.shape

    block = triton.next_power_of_2(V)
    if block > _MAX_VOCAB_BLOCK:
        raise ValueError(
            f"cross_entropy: vocab {V} exceeds max block size {_MAX_VOCAB_BLOCK}. "
            f"Vocab must be <= {_MAX_VOCAB_BLOCK} for the Triton path."
        )

    ce_sum = torch.zeros(1, device=logits.device, dtype=torch.float32)
    ce_cnt = torch.zeros(1, device=logits.device, dtype=torch.float32)
    z_sum = torch.zeros(1, device=logits.device, dtype=torch.float32)

    _ce_z_fwd_kernel[(M,)](
        logits_2d, targets_1d,
        ce_sum, ce_cnt, z_sum,
        ignore_index, z_loss_weight,
        M, V=V, BLOCK_V=block,
        num_warps=8, num_stages=2,
    )

    ce_mean = ce_sum / ce_cnt.clamp_min(1.0)
    z_mean = z_sum / M
    return ce_mean + z_loss_weight * z_mean


class _TritonCEWithZ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, ignore_index, z_loss_weight):
        ctx.save_for_backward(logits, targets)
        ctx.ignore_index = ignore_index
        ctx.z_loss_weight = z_loss_weight
        return _triton_ce_z_forward(logits, targets, ignore_index, z_loss_weight)

    @staticmethod
    def backward(ctx, grad_out):
        logits, targets = ctx.saved_tensors
        with torch.enable_grad():
            x = logits.detach().requires_grad_(True)
            y = cross_entropy_with_z_pytorch(
                x, targets, ctx.ignore_index, ctx.z_loss_weight,
            )
        grad_x, = torch.autograd.grad(y, x, grad_out)
        return grad_x, None, None, None


def triton_chunked_cross_entropy_with_z(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    z_loss_weight: float = 1e-4,
) -> torch.Tensor:
    """Public entry point; drop-in for ``chunked_cross_entropy_with_z``. Processes the entire vocab axis of the given logits in one fused pass; in the training path it is called per chunk from ``chunked_head_cross_entropy_with_z`` so only a chunk's logits are live at a time. ``chunk_size`` from the PyTorch API is intentionally not accepted here (passing it would silently no-op)."""
    if not HAS_TRITON:
        raise ImportError(
            "triton_chunked_cross_entropy_with_z requires the `triton` package. "
            "Install with `pip install triton` (Linux + CUDA only). "
            "Use `cross_entropy_impl='pytorch'` for CPU/Mac."
        )
    return _TritonCEWithZ.apply(logits, targets, ignore_index, z_loss_weight)
