"""Tests for ``kernels/rmsnorm_triton.py``.

CPU-only. The pure-PyTorch reference and the triton guard are exercised here,
so the suite runs without the `triton` package (AGENTS.md rule 8).
"""
from __future__ import annotations

import pytest
import torch

from kernels.rmsnorm_triton import HAS_TRITON, rmsnorm_pytorch, triton_rmsnorm
from model import RMSNorm


class TestRMSNormReference:
    @pytest.mark.numeric
    def test_matches_model_rmsnorm(self, device):
        torch.manual_seed(0)
        norm = RMSNorm(d_model=32, eps=1e-5).to(device)
        with torch.no_grad():
            norm.weight.copy_(torch.randn(32, device=device))
        x = torch.randn(4, 16, 32, device=device)
        ref = rmsnorm_pytorch(x, norm.weight, norm.eps)
        assert torch.allclose(ref, norm(x), atol=1e-6)

    def test_preserves_shape(self, device):
        x = torch.randn(3, 5, 8, device=device)
        w = torch.ones(8, device=device)
        assert rmsnorm_pytorch(x, w, 1e-5).shape == x.shape

    def test_zero_input_maps_to_zero(self, device):
        x = torch.zeros(2, 8, device=device)
        w = torch.randn(8, device=device)
        assert torch.allclose(rmsnorm_pytorch(x, w, 1e-5), torch.zeros_like(x))

    def test_unit_weight_gives_unit_row_rms(self, device):
        x = torch.randn(6, 32, device=device)
        out = rmsnorm_pytorch(x, torch.ones(32, device=device), 1e-5)
        rms = out.pow(2).mean(dim=-1)
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)


@pytest.mark.skipif(HAS_TRITON, reason="triton installed; ImportError path unreachable")
def test_triton_entry_raises_without_triton():
    with pytest.raises(ImportError):
        triton_rmsnorm(torch.randn(2, 8), torch.ones(8), 1e-5)
