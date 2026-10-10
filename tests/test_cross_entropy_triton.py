"""Tests for ``kernels/cross_entropy_triton.py``.

CPU-only. The reference is checked against
``model.chunked_cross_entropy_with_z``, so the suite runs without the
`triton` package (AGENTS.md rule 8).
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from kernels.cross_entropy_triton import (
    HAS_TRITON,
    cross_entropy_with_z_pytorch,
    triton_chunked_cross_entropy_with_z,
)
from model import chunked_cross_entropy_with_z


def _logits_targets(device, rows: int = 64, vocab: int = 128):
    torch.manual_seed(0)
    return (
        torch.randn(rows, vocab, device=device),
        torch.randint(0, vocab, (rows,), device=device),
    )


class TestCrossEntropyReference:
    @pytest.mark.numeric
    def test_matches_model_chunked_loss(self, device):
        logits, targets = _logits_targets(device)
        ref = cross_entropy_with_z_pytorch(logits, targets, -100, 1e-4)
        assert torch.allclose(ref, chunked_cross_entropy_with_z(logits, targets), atol=1e-5)

    @pytest.mark.numeric
    def test_zero_z_weight_is_plain_cross_entropy(self, device):
        logits, targets = _logits_targets(device)
        ref = cross_entropy_with_z_pytorch(logits, targets, -100, 0.0)
        assert torch.allclose(ref, F.cross_entropy(logits, targets, ignore_index=-100), atol=1e-5)

    def test_z_loss_raises_the_loss(self, device):
        logits, targets = _logits_targets(device)
        assert (cross_entropy_with_z_pytorch(logits, targets, -100, 1e-4)
                > cross_entropy_with_z_pytorch(logits, targets, -100, 0.0))

    def test_z_term_averages_over_every_row(self, device):
        """The reference does not mask ignored rows out of the z term, unlike
        ``model.chunked_cross_entropy_with_z``. The training path never passes
        -100, so the two agree in practice; this pins the kernel's own rule."""
        logits, targets = _logits_targets(device)
        targets = targets.clone()
        targets[:16] = -100
        ref = cross_entropy_with_z_pytorch(logits, targets, -100, 1e-4)
        expected = (
            F.cross_entropy(logits, targets, ignore_index=-100)
            + 1e-4 * torch.logsumexp(logits.float(), dim=-1).pow(2).mean()
        )
        assert torch.allclose(ref, expected, atol=1e-5)


@pytest.mark.skipif(HAS_TRITON, reason="triton installed; ImportError path unreachable")
def test_triton_entry_raises_without_triton():
    with pytest.raises(ImportError):
        triton_chunked_cross_entropy_with_z(
            torch.randn(4, 16), torch.zeros(4, dtype=torch.long)
        )
