"""Fused RMSNorm Triton kernel + pure-PyTorch reference.

Drop-in replacement for ``nn.RMSNorm`` in the LLaMA-3-Lite hot path. The
PyTorch eager chain ``pow().mean() + rsqrt + multiply`` is 4 separate
launches; this fuses them into a single row-wise Triton program.

The kernel reads ``x`` once, computes the row mean-square, normalises, and
applies the learnable weight in a single pass. d_model=1024 (LLaMA-3-Lite
default) maps to BLOCK_SIZE=1024 — well inside the 256-cap register
budget for a single program.

Backward is a v1 reference stub: forward saves ``x`` and ``weight``;
backward re-runs the PyTorch autograd through the reference. Slower than
a re-compute backward, but correct and small. v2 plan in the kernel
docstring.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

_MAX_BLOCK_SIZE = 8192


def rmsnorm_pytorch(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Reference RMSNorm in raw PyTorch. Numerically the same as ``nn.RMSNorm``."""
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


if HAS_TRITON:

    @triton.jit
    def _rmsnorm_fwd_kernel(
        X_ptr,
        W_ptr,
        Y_ptr,
        stride_x_row,
        stride_y_row,
        N: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N

        x_ptr = X_ptr + row * stride_x_row + cols
        y_ptr = Y_ptr + row * stride_y_row + cols

        x = tl.load(x_ptr, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / N
        rstd = 1.0 / tl.sqrt(var + eps)

        w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x * rstd) * w
        tl.store(y_ptr, y.to(Y_ptr.dtype.element_ty), mask=mask)


def _triton_rmsnorm_forward(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
    M, N = x_2d.shape

    block = triton.next_power_of_2(N)
    if block > _MAX_BLOCK_SIZE:
        raise ValueError(
            f"rmsnorm: last dim {N} exceeds max block size {_MAX_BLOCK_SIZE}. "
            f"d_model must be <= {_MAX_BLOCK_SIZE} for the Triton path."
        )

    y = torch.empty_like(x_2d)
    _rmsnorm_fwd_kernel[(M,)](
        x_2d, weight, y,
        x_2d.stride(0), y.stride(0),
        N=N, eps=eps, BLOCK_SIZE=block,
        num_warps=4, num_stages=1,
    )
    return y.reshape(orig_shape)


class _TritonRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        return _triton_rmsnorm_forward(x, weight, eps)

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        with torch.enable_grad():
            x_ref = x.detach().requires_grad_(True)
            w_ref = weight.detach().requires_grad_(True)
            y = rmsnorm_pytorch(x_ref, w_ref, ctx.eps)
        grad_x, grad_w = torch.autograd.grad(y, [x_ref, w_ref], grad_out)
        return grad_x, grad_w, None


def triton_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Public entry point. Raises ``ImportError`` when triton is missing."""
    if not HAS_TRITON:
        raise ImportError(
            "triton_rmsnorm requires the `triton` package. "
            "Install with `pip install triton` (Linux + CUDA only). "
            "Use `rmsnorm_impl='pytorch'` for CPU/Mac."
        )
    return _TritonRMSNorm.apply(x, weight, eps)
