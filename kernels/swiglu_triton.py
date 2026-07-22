"""Fused SwiGLU activation Triton kernel + pure-PyTorch reference.

The LLaMA-3 FFN path: ``down_proj(silu(gate) * up)``. The
``silu(gate) * up`` step is two separate elementwise launches in PyTorch
eager (silu, then multiply). The kernel fuses them, reading the already-
fused ``gate_up`` projection output (2*d_ff wide) and writing the
``d_ff`` activation in a single pass.

For d_ff=4096 (LLaMA-3-Lite default), BLOCK_SIZE=4096 fits A100 register
budget. Backward is a v1 reference stub: forward saves ``gate_up``;
backward re-runs PyTorch autograd on the reference. v2 plan in the kernel
docstring.
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
    """Public entry point. Raises ``ImportError`` when triton is missing.

    ``gate_up`` is the fused ``gate_proj + up_proj`` output of width
    ``2 * d_ff``. The kernel splits internally and returns the
    ``d_ff``-wide activation.
    """
    if not HAS_TRITON:
        raise ImportError(
            "triton_swiglu requires the `triton` package. "
            "Install with `pip install triton` (Linux + CUDA only). "
            "Use `swiglu_impl='pytorch'` for CPU/Mac."
        )
    return _TritonSwiGLU.apply(gate_up, d_ff)
