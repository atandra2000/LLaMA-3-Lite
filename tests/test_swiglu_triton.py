"""Tests for ``kernels/swiglu_triton.py``.

CPU-only. The fused reference is checked against the ``SwiGLUFFN`` pytorch
path, so the suite runs without the `triton` package (AGENTS.md rule 8).
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from kernels.swiglu_triton import HAS_TRITON, swiglu_pytorch, triton_swiglu
from model import SwiGLUFFN


class TestSwiGLUReference:
    @pytest.mark.numeric
    def test_is_silu_gate_times_up(self, device):
        gate = torch.randn(4, 16, device=device)
        up = torch.randn(4, 16, device=device)
        assert torch.allclose(swiglu_pytorch(gate, up), F.silu(gate) * up)

    @pytest.mark.numeric
    def test_matches_ffn_pytorch_path(self, device):
        torch.manual_seed(0)
        ffn = SwiGLUFFN(d_model=16, d_ff=32).to(device)
        x = torch.randn(2, 5, 16, device=device)
        gate, up = ffn.gate_up_proj(x).chunk(2, dim=-1)
        expected = ffn.down_proj(swiglu_pytorch(gate, up))
        assert torch.allclose(expected, ffn(x), atol=1e-6)

    def test_halves_the_fused_width(self, device):
        gate_up = torch.randn(3, 12, device=device)
        gate, up = gate_up.chunk(2, dim=-1)
        assert swiglu_pytorch(gate, up).shape == (3, 6)


@pytest.mark.skipif(HAS_TRITON, reason="triton installed; ImportError path unreachable")
def test_triton_entry_raises_without_triton():
    with pytest.raises(ImportError):
        triton_swiglu(torch.randn(2, 8), d_ff=4)
