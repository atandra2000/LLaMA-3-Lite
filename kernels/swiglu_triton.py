"""Fused SwiGLU activation Triton kernel + pure-PyTorch reference.

Fuses ``silu(gate) * up`` (two elementwise launches in eager) into one
row-wise program. ``gate_up`` is the fused 2*d_ff projection output.
Backward is a PyTorch autograd re-compute stub.
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


_MAX_BLOCK_SIZE = 8192


def swiglu_pytorch(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Reference: silu(gate) * up. Gate and up are split halves of the fused projection."""
    return F.silu(gate) * up


if HAS_TRITON:

    @triton.jit
    def _swiglu_fwd_kernel(
        GU_ptr,        # (..., 2*D) fused gate||up
        Y_ptr,         # (..., D) output
        stride_row,
        D: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < D

        g = tl.load(GU_ptr + row * stride_row + cols, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(GU_ptr + row * stride_row + D + cols, mask=mask, other=0.0).to(tl.float32)

        silu_g = g * tl.sigmoid(g)
        y = silu_g * u
        tl.store(Y_ptr + row * stride_row + cols, y.to(Y_ptr.dtype.element_ty), mask=mask)


def _triton_swiglu_forward(gate_up: torch.Tensor, d_ff: int) -> torch.Tensor:
    orig_shape = gate_up.shape
    last = orig_shape[-1]
    if last != 2 * d_ff:
        raise ValueError(
            f"swiglu: last dim {last} != 2 * d_ff ({2 * d_ff}). "
            f"Pass the fused gate_up projection output of width 2*d_ff."
        )
    gu_2d = gate_up.reshape(-1, last).contiguous()
    M = gu_2d.shape[0]

    block = triton.next_power_of_2(d_ff)
    if block > _MAX_BLOCK_SIZE:
        raise ValueError(
            f"swiglu: d_ff {d_ff} exceeds max block size {_MAX_BLOCK_SIZE}."
        )

    y = torch.empty((M, d_ff), device=gate_up.device, dtype=gate_up.dtype)
    _swiglu_fwd_kernel[(M,)](
        gu_2d, y,
        gu_2d.stride(0),
        D=d_ff, BLOCK_SIZE=block,
        num_warps=8, num_stages=2,
    )
    return y.reshape(*orig_shape[:-1], d_ff)


class _TritonSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate_up, d_ff):
        ctx.save_for_backward(gate_up)
        ctx.d_ff = d_ff
        return _triton_swiglu_forward(gate_up, d_ff)

    @staticmethod
    def backward(ctx, grad_out):
        (gate_up,) = ctx.saved_tensors
        d_ff = ctx.d_ff
        with torch.enable_grad():
            gu = gate_up.detach().requires_grad_(True)
            gate, up = gu.chunk(2, dim=-1)
            y = swiglu_pytorch(gate, up)
        grad_gu, = torch.autograd.grad(y, gu, grad_out)
        return grad_gu, None


def triton_swiglu(gate_up: torch.Tensor, d_ff: int) -> torch.Tensor:
    """Public entry point; raises ``ImportError`` when triton is missing; ``gate_up`` is the fused ``gate_proj + up_proj`` output of width ``2*d_ff`` and the kernel returns the ``d_ff``-wide activation."""
    if not HAS_TRITON:
        raise ImportError(
            "triton_swiglu requires the `triton` package. "
            "Install with `pip install triton` (Linux + CUDA only). "
            "Use `swiglu_impl='pytorch'` for CPU/Mac."
        )
    return _TritonSwiGLU.apply(gate_up, d_ff)
