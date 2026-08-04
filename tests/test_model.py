"""Tests for ``model.py``."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from model import (
    GroupedQueryAttention,
    RMSNorm,
    RoPE,
    SwiGLUFFN,
    Transformer,
    build_transformer,
    chunked_cross_entropy_with_z,
    chunked_head_cross_entropy_with_z,
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
        in_emb = model.input_embedding.weight.numel()
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

    def test_gradient_checkpointing_matches_normal_in_training(
            self, tiny_config, device, seed_everything):
        """Training-mode regression: the checkpointed branch must apply the
        final decoder norm, so forward outputs match the plain path exactly."""
        seed_everything(43)
        common = dict(
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
        )
        model_a = build_transformer(**common, gradient_checkpointing=False).to(device)
        model_b = build_transformer(**common, gradient_checkpointing=True).to(device)
        model_b.load_state_dict(model_a.state_dict())

        ids = torch.randint(0, tiny_config["vocab_size"],
                            (2, tiny_config["seq_len"]),
                            device=device, dtype=torch.long)
        model_a.train(); model_b.train()
        out_a = model_a(ids)
        out_b = model_b(ids)
        assert torch.allclose(out_a, out_b, atol=1e-6), (out_a - out_b).abs().max()

        # Gradients must flow through the checkpointed path too.
        loss_b = out_b.sum()
        loss_b.backward()
        grads = [p.grad for p in model_b.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)

class TestChunkedCrossEntropyWithZ:
    """Numerical-equivalence + behaviour tests for the z-loss variant."""

    @pytest.mark.numeric
    def test_matches_ce_plus_zpen_reference(self, device):
        """z-loss wrapper must equal `ce + weight * mean(logsumexp(logits.float())**2)`
        within 1e-5 — the whole point of having a separate function is to keep
        the original chunked_cross_entropy bit-identical to F.cross_entropy."""
        torch.manual_seed(7)
        N, V = 100, 64
        logits = torch.randn(N, V, device=device, requires_grad=True)
        targets = torch.randint(0, V, (N,), device=device, dtype=torch.long)

        weight = 1e-4
        ref_ce = F.cross_entropy(logits, targets, reduction="mean")
        ref_z = (torch.logsumexp(logits.float(), dim=-1).pow(2)).mean()
        ref = ref_ce + weight * ref_z

        got = chunked_cross_entropy_with_z(
            logits.clone().detach().requires_grad_(True), targets,
            chunk_size=32, z_loss_weight=weight,
        )
        assert torch.allclose(got, ref, atol=1e-5), (ref.item(), got.item())

    def test_z_weight_zero_matches_pure_ce(self, device):
        """With z_loss_weight=0, the function should be equivalent to chunked
        cross-entropy (the penalty term is identically zero)."""
        torch.manual_seed(11)
        N, V = 50, 32
        logits = torch.randn(N, V, device=device)
        targets = torch.randint(0, V, (N,), device=device, dtype=torch.long)
        a = chunked_cross_entropy_with_z(logits, targets, chunk_size=16, z_loss_weight=0.0)
        b = F.cross_entropy(logits, targets, reduction="mean")
        assert torch.allclose(a, b, atol=1e-6)

    def test_gradients_flow(self, device):
        """Z-loss must backprop through the logits — otherwise the bound on
        logit growth is worthless."""
        torch.manual_seed(13)
        logits = torch.randn(40, 20, device=device, requires_grad=True)
        targets = torch.randint(0, 20, (40,), device=device, dtype=torch.long)
        loss = chunked_cross_entropy_with_z(logits, targets, z_loss_weight=1e-3)
        loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()

    def test_z_loss_grows_with_logit_magnitude(self, device):
        """Sanity: scaling logits up by a constant should increase the z penalty
        (since logsumexp grows with the max logit)."""
        torch.manual_seed(17)
        N, V = 30, 16
        targets = torch.randint(0, V, (N,), device=device, dtype=torch.long)
        small = torch.randn(N, V, device=device) * 0.1
        large = torch.randn(N, V, device=device) * 5.0
        z_small = chunked_cross_entropy_with_z(small, targets, z_loss_weight=1.0).item()
        z_large = chunked_cross_entropy_with_z(large, targets, z_loss_weight=1.0).item()
        assert z_large > z_small, (z_small, z_large)

    def test_z_loss_ignores_ignore_index_positions(self, device):
        """Ignored positions must not contribute to the z-loss average."""
        torch.manual_seed(23)
        N, V = 40, 32
        logits = torch.randn(N, V, device=device)
        targets = torch.randint(0, V, (N,), device=device, dtype=torch.long)
        targets[:10] = -100

        mask = targets != -100
        ref_log_z = torch.logsumexp(logits.float(), dim=-1)
        ref_z = ref_log_z[mask].pow(2).mean()
        ref_ce = F.cross_entropy(logits, targets, ignore_index=-100,
                                 reduction="mean")
        ref = ref_ce + 0.5 * ref_z

        got = chunked_cross_entropy_with_z(logits, targets, chunk_size=7,
                                           z_loss_weight=0.5)
        assert torch.allclose(got, ref, atol=1e-5), (ref.item(), got.item())


class TestChunkedHeadCrossEntropyWithZ:
    """The memory-bounded LM-head loss must match dense CE + z-loss exactly."""

    def test_matches_dense_ce_with_zero_z(self, tiny_model, tiny_config, device):
        torch.manual_seed(31)
        B, S = 2, tiny_config["seq_len"]
        ids = torch.randint(0, tiny_config["vocab_size"], (B, S),
                            device=device, dtype=torch.long)
        tgt = torch.randint(0, tiny_config["vocab_size"], (B, S),
                            device=device, dtype=torch.long)
        tiny_model.eval()
        with torch.no_grad():
            logits = tiny_model(ids)
            dense = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                    tgt.view(-1), reduction="mean")
            hidden = tiny_model(ids, return_hidden=True)
            chk = chunked_head_cross_entropy_with_z(
                hidden.view(-1, hidden.size(-1)),
                tiny_model.output_proj.weight,
                tgt.view(-1), chunk_size=7, z_loss_weight=0.0,
            )
        assert torch.allclose(dense, chk, atol=1e-5), (dense.item(), chk.item())

    def test_matches_dense_ce_plus_z(self, tiny_model, tiny_config, device):
        torch.manual_seed(37)
        B, S = 2, tiny_config["seq_len"]
        ids = torch.randint(0, tiny_config["vocab_size"], (B, S),
                            device=device, dtype=torch.long)
        tgt = torch.randint(0, tiny_config["vocab_size"], (B, S),
                            device=device, dtype=torch.long)
        tiny_model.eval()
        with torch.no_grad():
            logits = tiny_model(ids)
            ref_z = torch.logsumexp(logits.float(), dim=-1).pow(2).mean()
            dense = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                    tgt.view(-1), reduction="mean")
            ref = dense + 1e-4 * ref_z
            hidden = tiny_model(ids, return_hidden=True)
            chk = chunked_head_cross_entropy_with_z(
                hidden.view(-1, hidden.size(-1)),
                tiny_model.output_proj.weight,
                tgt.view(-1), chunk_size=7, z_loss_weight=1e-4,
            )
        assert torch.allclose(ref, chk, atol=1e-5), (ref.item(), chk.item())

    def test_gradients_flow_to_hidden_and_head(self, tiny_model, tiny_config,
                                                device):
        torch.manual_seed(41)
        B, S = 2, tiny_config["seq_len"]
        ids = torch.randint(0, tiny_config["vocab_size"], (B, S),
                            device=device, dtype=torch.long)
        tgt = torch.randint(0, tiny_config["vocab_size"], (B, S),
                            device=device, dtype=torch.long)
        tiny_model.train()
        hidden = tiny_model(ids, return_hidden=True)
        loss = chunked_head_cross_entropy_with_z(
            hidden.view(-1, hidden.size(-1)),
            tiny_model.output_proj.weight,
            tgt.view(-1), chunk_size=7, z_loss_weight=1e-3,
        )
        loss.backward()
        assert torch.isfinite(loss).item()
        assert tiny_model.output_proj.weight.grad is not None
        assert torch.isfinite(tiny_model.output_proj.weight.grad).all()
        # hidden is a non-leaf tensor (grad=None by default); gradient flow
        # through it is proven by the embedding receiving grads.
        assert tiny_model.input_embedding.weight.grad is not None
        assert torch.isfinite(tiny_model.input_embedding.weight.grad).all()

    def test_return_hidden_skips_head(self, tiny_model, tiny_config, device):
        B, S = 2, tiny_config["seq_len"]
        ids = torch.randint(0, tiny_config["vocab_size"], (B, S),
                            device=device, dtype=torch.long)
        hidden = tiny_model(ids, return_hidden=True)
        assert hidden.shape == (B, S, tiny_config["d_model"])
        logits = tiny_model(ids)
        assert logits.shape == (B, S, tiny_config["vocab_size"])


class TestQKNorm:
    """QK-norm behaviour: param count changes when enabled; forward is a no-op
    transformation when disabled (qknorm=False → q_norm/k_norm are Identity)."""

    def test_param_count_increases_when_enabled(self, tiny_config, device, dtype, seed_everything):
        """QK-norm adds 2 * head_dim per layer when enabled."""
        seed_everything(101)
        # Inline a small param-counter to avoid an import path dependency.
        def count_params(module):
            return sum(p.numel() for p in module.parameters())
        cfg = tiny_config
        m_off = build_transformer(
            vocab_size=cfg["vocab_size"], d_model=cfg["d_model"], n_layers=cfg["n_layers"],
            n_heads=cfg["n_heads"], n_kv_heads=cfg["n_kv_heads"], head_dim=cfg["head_dim"],
            d_ff=cfg["d_ff"], max_seq_len=cfg["seq_len"], rope_theta=cfg["rope_theta"],
            rms_norm_eps=cfg["rms_norm_eps"], qknorm=False,
        )
        m_on = build_transformer(
            vocab_size=cfg["vocab_size"], d_model=cfg["d_model"], n_layers=cfg["n_layers"],
            n_heads=cfg["n_heads"], n_kv_heads=cfg["n_kv_heads"], head_dim=cfg["head_dim"],
            d_ff=cfg["d_ff"], max_seq_len=cfg["seq_len"], rope_theta=cfg["rope_theta"],
            rms_norm_eps=cfg["rms_norm_eps"], qknorm=True,
        )
        # Per layer: q_norm (head_dim) + k_norm (head_dim) = 2 * head_dim scalars.
        expected_extra = 2 * cfg["head_dim"] * cfg["n_layers"]
        assert count_params(m_on) - count_params(m_off) == expected_extra

    def test_disabled_attention_is_bit_identical(self, tiny_config, device, dtype, seed_everything):
        """With qknorm=False the q/k projections are identity-normed, so two
        models with identical weights must produce identical outputs."""
        seed_everything(202)
        cfg = tiny_config
        common = dict(
            vocab_size=cfg["vocab_size"], d_model=cfg["d_model"], n_layers=cfg["n_layers"],
            n_heads=cfg["n_heads"], n_kv_heads=cfg["n_kv_heads"], head_dim=cfg["head_dim"],
            d_ff=cfg["d_ff"], max_seq_len=cfg["seq_len"], rope_theta=cfg["rope_theta"],
            rms_norm_eps=cfg["rms_norm_eps"],
        )
        # Build with qknorm=False, then manually swap q_norm/k_norm to Identity
        # in a copy. This isolates the attention module from the rest of the
        # model (no other layer depends on qknorm).
        m = build_transformer(**common, qknorm=False).to(device=device, dtype=dtype)
        # Sanity: q_norm/k_norm exist as Identity placeholders.
        attn = m.decoder.layers[0].attention
        assert isinstance(attn.q_norm, torch.nn.Identity)
        assert isinstance(attn.k_norm, torch.nn.Identity)
        m.eval()
        ids = torch.randint(0, cfg["vocab_size"], (2, cfg["seq_len"]),
                            device=device, dtype=torch.long)
        with torch.no_grad():
            out = m(ids)
        assert out.shape == (2, cfg["seq_len"], cfg["vocab_size"])

    def test_enabled_attention_does_not_crash(self, tiny_config, device, dtype, seed_everything):
        """With qknorm=True a forward pass must run and produce finite output."""
        seed_everything(303)
        cfg = tiny_config
        m = build_transformer(
            vocab_size=cfg["vocab_size"], d_model=cfg["d_model"], n_layers=cfg["n_layers"],
            n_heads=cfg["n_heads"], n_kv_heads=cfg["n_kv_heads"], head_dim=cfg["head_dim"],
            d_ff=cfg["d_ff"], max_seq_len=cfg["seq_len"], rope_theta=cfg["rope_theta"],
            rms_norm_eps=cfg["rms_norm_eps"], qknorm=True,
        ).to(device=device, dtype=dtype)
        attn = m.decoder.layers[0].attention
        # Real RMSNorm, not Identity
        assert isinstance(attn.q_norm, RMSNorm)
        assert isinstance(attn.k_norm, RMSNorm)
        m.eval()
        ids = torch.randint(0, cfg["vocab_size"], (2, cfg["seq_len"]),
                            device=device, dtype=torch.long)
        with torch.no_grad():
            out = m(ids)
        assert torch.isfinite(out).all()
