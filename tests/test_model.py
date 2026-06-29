"""Tests for ``model.py``."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from model import (
    GroupedQueryAttention,
    InputEmbedding,
    RMSNorm,
    RoPE,
    SwiGLUFFN,
    Transformer,
    build_transformer,
    chunked_cross_entropy,
)


class TestRMSNorm:
    def test_output_shape(self, device):
        norm = RMSNorm(d_model=16).to(device)
        x = torch.randn(2, 5, 16, device=device)
        assert norm(x).shape == x.shape

    def test_zero_input_yields_weight(self, device):
        norm = RMSNorm(d_model=8).to(device)
        x = torch.zeros(1, 3, 8, device=device)
        out = norm(x)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)

    def test_matches_reference(self, device):
        torch.manual_seed(0)
        d = 32
        norm = RMSNorm(d_model=d, eps=1e-5).to(device)
        x = torch.randn(4, 7, d, device=device, dtype=torch.float64)
        norm64 = RMSNorm(d_model=d, eps=1e-5).to(device).double()
        out = norm64(x)
        ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
        ref = norm64.weight * ref
        assert torch.allclose(out, ref, atol=1e-10)

    def test_scale_invariance(self, device):
        norm = RMSNorm(d_model=16).to(device).eval()
        x = torch.randn(1, 5, 16, device=device)
        out1 = norm(x)
        out2 = norm(x * 3.0)
        assert torch.allclose(out1, out2, atol=1e-5)

    def test_weight_is_learnable(self):
        norm = RMSNorm(d_model=8)
        assert isinstance(norm.weight, torch.nn.Parameter)
        assert torch.allclose(norm.weight, torch.ones(8))


class TestRoPE:
    def test_buffer_shapes(self, device):
        rope = RoPE(head_dim=16, max_seq_len=64, theta=10000.0).to(device)
        assert rope.cos_cached.shape == (1, 1, 64, 8)
        assert rope.sin_cached.shape == (1, 1, 64, 8)
        assert rope.inv_freq.shape == (8,)

    def test_inv_freq_monotonic(self, device):
        rope = RoPE(head_dim=16, max_seq_len=32, theta=10000.0).to(device)
        assert torch.all(rope.inv_freq[:-1] > rope.inv_freq[1:])

    def test_rotation_is_orthogonal(self, device):
        """A RoPE-rotated vector should preserve its norm (it's a rotation)."""
        head_dim = 16
        rope = RoPE(head_dim, max_seq_len=8, theta=10000.0).to(device)
        x = torch.randn(1, 1, 4, head_dim, device=device)
        out = rope(x, seq_len=4)
        assert out.shape == x.shape
        n_in = x.norm(dim=-1)
        n_out = out.norm(dim=-1)
        assert torch.allclose(n_in, n_out, atol=1e-5), (
            n_in, n_out, "RoPE must preserve L2 norm"
        )

    def test_position_zero_is_identity(self, device):
        """RoPE at position 0 must be the identity (cos=1, sin=0)."""
        head_dim = 16
        rope = RoPE(head_dim, max_seq_len=4, theta=10000.0).to(device)
        x = torch.randn(1, 1, 1, head_dim, device=device)
        out = rope(x, seq_len=1)
        assert torch.allclose(out, x, atol=1e-6), (out, x)

    def test_relative_position_property(self, device):
        """Inner product q_i . k_j should depend only on (i-j) under RoPE."""
        head_dim = 8
        rope = RoPE(head_dim, max_seq_len=32, theta=10000.0).to(device)
        q = torch.zeros(1, 1, 1, head_dim, device=device)
        q[..., 0] = 1.0
        k = torch.zeros(1, 1, 1, head_dim, device=device)
        k[..., 1] = 1.0
        q_seq = torch.zeros(1, 1, 6, head_dim, device=device)
        k_seq = torch.zeros(1, 1, 6, head_dim, device=device)
        q_seq[..., 0, :] = q
        k_seq[..., 0, :] = k
        q_seq[..., 5, :] = q
        k_seq[..., 5, :] = k
        q_rot = rope(q_seq, seq_len=6)
        k_rot = rope(k_seq, seq_len=6)
        attn_0 = (q_rot[..., 0, :] * k_rot[..., 0, :]).sum()
        attn_5 = (q_rot[..., 5, :] * k_rot[..., 5, :]).sum()
        assert torch.allclose(attn_0, attn_5, atol=1e-5), (
            attn_0, attn_5, "RoPE should be translation-equivariant"
        )


class TestGroupedQueryAttention:
    def test_output_shape(self, device, dtype):
        attn = GroupedQueryAttention(
            d_model=64, n_heads=4, n_kv_heads=2, head_dim=16,
            max_seq_len=32, rope_theta=10000.0,
        ).to(device=device, dtype=dtype)
        x = torch.randn(2, 10, 64, device=device, dtype=dtype)
        out = attn(x)
        assert out.shape == (2, 10, 64)

    def test_causality(self, device):
        """Later tokens must not affect earlier outputs (causal mask)."""
        torch.manual_seed(0)
        attn = GroupedQueryAttention(
            d_model=32, n_heads=4, n_kv_heads=2, head_dim=8,
            max_seq_len=16, rope_theta=10000.0,
        ).to(device).eval()
        x = torch.randn(1, 6, 32, device=device)
        out1 = attn(x)
        x2 = x.clone()
        x2[:, -1, :] += torch.randn_like(x[:, -1, :]) * 10.0
        out2 = attn(x2)
        assert torch.allclose(out1[:, :3, :], out2[:, :3, :], atol=1e-5), (
            "Future-token perturbation leaked into past outputs"
        )

    def test_n_rep_consistency(self, device):
        for n_heads, n_kv in [(4, 2), (8, 4), (4, 4), (2, 1)]:
            attn = GroupedQueryAttention(
                d_model=32, n_heads=n_heads, n_kv_heads=n_kv, head_dim=8,
                max_seq_len=16, rope_theta=10000.0,
            ).to(device)
            assert attn.n_rep == n_heads // n_kv
            x = torch.randn(1, 8, 32, device=device)
            assert attn(x).shape == (1, 8, 32)

    def test_invalid_n_kv_heads_raises(self, device):
        attn = GroupedQueryAttention(
            d_model=32, n_heads=8, n_kv_heads=3, head_dim=8,
            max_seq_len=16, rope_theta=10000.0,
        ).to(device)
        assert attn.n_rep == 2
        x = torch.randn(1, 8, 32, device=device)
        with pytest.raises(RuntimeError):
            attn(x)


class TestSwiGLUFFN:
    def test_output_shape(self, device):
        ffn = SwiGLUFFN(d_model=64, d_ff=128).to(device)
        x = torch.randn(2, 8, 64, device=device)
        assert ffn(x).shape == (2, 8, 64)

    def test_fused_equals_unfused_reference(self, device):
        """Fused gate+up projection must equal two separate projections."""
        torch.manual_seed(0)
        d_model, d_ff = 32, 64
        ffn = SwiGLUFFN(d_model, d_ff).to(device)
        gate_up_w = ffn.gate_up_proj.weight.data
        gate_w, up_w = torch.split(gate_up_w, d_ff, dim=0)
        down_w = ffn.down_proj.weight.data

        x = torch.randn(3, 5, d_model, device=device)
        gate = F.linear(x, gate_w)
        up = F.linear(x, up_w)
        ref = F.linear(F.silu(gate) * up, down_w)
        out = ffn(x)
        assert torch.allclose(out, ref, atol=1e-6), (out - ref)

    def test_gate_up_proj_has_2x_d_ff_rows(self, device):
        ffn = SwiGLUFFN(d_model=16, d_ff=32).to(device)
        assert ffn.gate_up_proj.weight.shape == (64, 16)
        assert ffn.down_proj.weight.shape == (16, 32)


class TestChunkedCrossEntropy:
    @pytest.mark.numeric
    def test_equals_pytorch_cross_entropy(self, device):
        """Chunked CE must match F.cross_entropy to <1e-5."""
        torch.manual_seed(0)
        B, S, V = 4, 32, 100
        logits = torch.randn(B * S, V, device=device, requires_grad=True)
        targets = torch.randint(0, V, (B * S,), device=device)

        ref = F.cross_entropy(logits, targets, reduction="mean")
        chk = chunked_cross_entropy(logits.detach().clone().requires_grad_(True),
                                    targets, chunk_size=8192)
        diff = (ref - chk).abs().item()
        assert diff < 1e-5, f"chunked CE differs from reference by {diff}"
        assert 3.0 < chk.item() < 6.0

    @pytest.mark.numeric
    def test_chunk_size_does_not_change_result(self, device):
        """Different chunk sizes must all give the same loss."""
        torch.manual_seed(1)
        N, V = 1000, 50
        logits = torch.randn(N, V, device=device)
        targets = torch.randint(0, V, (N,), device=device)
        losses = [
            chunked_cross_entropy(logits.clone(), targets, chunk_size=c).item()
            for c in (1, 7, 50, 256, 100_000)
        ]
        assert max(losses) - min(losses) < 1e-5, losses

    @pytest.mark.numeric
    def test_ignore_index_excluded_from_loss(self, device):
        """Targets marked ignore_index must not contribute to the mean."""
        torch.manual_seed(2)
        N, V = 20, 10
        logits = torch.randn(N, V, device=device)
        targets = torch.randint(0, V, (N,), device=device)
        targets[:10] = -100

        ref = F.cross_entropy(logits, targets, ignore_index=-100,
                               reduction="mean")
        chk = chunked_cross_entropy(logits.clone(), targets, chunk_size=5,
                                    ignore_index=-100)
        assert torch.allclose(ref, chk, atol=1e-6), (ref, chk)

    @pytest.mark.numeric
    def test_all_ignored_returns_zero(self, device):
        N, V = 16, 8
        logits = torch.randn(N, V, device=device, requires_grad=True)
        targets = torch.full((N,), -100, device=device, dtype=torch.long)
        loss = chunked_cross_entropy(logits, targets, ignore_index=-100)
        assert loss.item() == 0.0

    @pytest.mark.numeric
    def test_gradients_flow(self, device):
        """The chunked loss must produce gradients."""
        torch.manual_seed(3)
        logits = torch.randn(64, 20, device=device, requires_grad=True)
        targets = torch.randint(0, 20, (64,), device=device)
        loss = chunked_cross_entropy(logits, targets, chunk_size=16)
        loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()
        assert logits.grad.abs().sum().item() > 0


class TestTransformerParamCount:
    def test_full_model_total_params(self, full_config):
        """README advertises ~515M total params; assert within 1%."""
        model = build_transformer(
            vocab_size=full_config["vocab_size"],
            d_model=full_config["d_model"],
            n_layers=full_config["n_layers"],
            n_heads=full_config["n_heads"],
            n_kv_heads=full_config["n_kv_heads"],
            head_dim=full_config["head_dim"],
            d_ff=full_config["d_ff"],
            max_seq_len=full_config["seq_len"],
            rope_theta=full_config["rope_theta"],
            rms_norm_eps=full_config["rms_norm_eps"],
            gradient_checkpointing=False,
        )
        total = sum(p.numel() for p in model.parameters())
        advertised = 514_891_808
        assert abs(total - advertised) / advertised < 0.01, (
            f"total={total:,} vs advertised={advertised:,}"
        )

    def test_get_num_params_definition_mismatch(self, full_config):
        """Flag a metric-definitions drift between get_num_params and README."""
        model = build_transformer(
            vocab_size=full_config["vocab_size"],
            d_model=full_config["d_model"],
            n_layers=full_config["n_layers"],
            n_heads=full_config["n_heads"],
            n_kv_heads=full_config["n_kv_heads"],
            head_dim=full_config["head_dim"],
            d_ff=full_config["d_ff"],
            max_seq_len=full_config["seq_len"],
            rope_theta=full_config["rope_theta"],
            rms_norm_eps=full_config["rms_norm_eps"],
        )
        total = sum(p.numel() for p in model.parameters())
        in_emb = model.input_embedding.embedding.weight.numel()
        out_emb = model.output_proj.weight.numel()
        readme_non_embed = total - in_emb - out_emb
        model_non_embed = model.get_num_params(non_embedding=True)
        advertised = 251_684_896

        assert abs(readme_non_embed - advertised) / advertised < 0.01, (
            f"README non-embed={readme_non_embed:,} vs advertised={advertised:,}"
        )
        assert abs(model_non_embed - advertised) / advertised < 0.01, (
            f"model.get_num_params(non_embedding=True)={model_non_embed:,} "
            f"does not match the README's non-embedding definition "
            f"({advertised:,}). Discrepancy: {model_non_embed - readme_non_embed:,}"
        )


class TestTransformerForward:
    def test_forward_output_shape(self, tiny_model, tiny_config, device):
        B, S = 2, tiny_config["seq_len"]
        ids = torch.randint(0, tiny_config["vocab_size"], (B, S),
                            device=device, dtype=torch.long)
        logits = tiny_model(ids)
        assert logits.shape == (B, S, tiny_config["vocab_size"])

    def test_backward_produces_grads(self, tiny_model, tiny_config, device):
        B, S = 2, tiny_config["seq_len"]
        ids = torch.randint(0, tiny_config["vocab_size"], (B, S),
                            device=device, dtype=torch.long)
        targets = torch.randint(0, tiny_config["vocab_size"], (B, S),
                                 device=device, dtype=torch.long)
        logits = tiny_model(ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                               targets.view(-1))
        loss.backward()
        for name, p in tiny_model.named_parameters():
            assert p.grad is not None, f"no grad for {name}"
            assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"

    def test_gradient_checkpointing_matches_normal(self, tiny_config, device,
                                                    seed_everything):
        """With gradient checkpointing the forward output must be identical to the non-checkpointed path."""
        seed_everything(42)
        model_a = build_transformer(
            vocab_size=tiny_config["vocab_size"],
            d_model=tiny_config["d_model"],
            n_layers=tiny_config["n_layers"],
            n_heads=tiny_config["n_heads"],
            n_kv_heads=tiny_config["n_kv_heads"],
            head_dim=tiny_config["head_dim"],
            d_ff=tiny_config["d_ff"],
            max_seq_len=tiny_config["seq_len"],
            rope_theta=tiny_config["rope_theta"],
            rms_norm_eps=tiny_config["rms_norm_eps"],
            gradient_checkpointing=False,
        ).to(device)

        model_b = build_transformer(
            vocab_size=tiny_config["vocab_size"],
            d_model=tiny_config["d_model"],
            n_layers=tiny_config["n_layers"],
            n_heads=tiny_config["n_heads"],
            n_kv_heads=tiny_config["n_kv_heads"],
            head_dim=tiny_config["head_dim"],
            d_ff=tiny_config["d_ff"],
            max_seq_len=tiny_config["seq_len"],
            rope_theta=tiny_config["rope_theta"],
            rms_norm_eps=tiny_config["rms_norm_eps"],
            gradient_checkpointing=True,
        ).to(device)
        model_b.load_state_dict(model_a.state_dict())

        ids = torch.randint(0, tiny_config["vocab_size"],
                            (2, tiny_config["seq_len"]),
                            device=device, dtype=torch.long)
        model_a.eval(); model_b.eval()
        with torch.no_grad():
            out_a = model_a(ids)
            out_b = model_b(ids)
        assert torch.allclose(out_a, out_b, atol=1e-6), (out_a - out_b).abs().max()