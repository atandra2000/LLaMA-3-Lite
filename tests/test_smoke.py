"""End-to-end smoke tests for the training pipeline.

These run a *tiny* number of training steps on synthetic in-memory data
(no tokenizer download, no W&B) and assert that loss decreases and the model
state actually changes. They are the cheapest way to catch integration
regressions (e.g. config keys going missing, dataloader/sampler mismatches).
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from conftest import make_token_stream

import dataset as ds
import train as train_mod


@pytest.fixture
def tiny_dataloaders(tiny_config):
    """Build train/val dataloaders from synthetic tokens (no tokenizer)."""
    seq_len = tiny_config["seq_len"]
    vocab = tiny_config["vocab_size"]
    eos, bos = 0, 1
    # Enough tokens for several chunks in both splits.
    n_tokens = (seq_len + 1) * 32 + 10
    data = make_token_stream(n_tokens, vocab, seq_len, eos_id=eos, bos_id=bos,
                             seed=0)
    # Chunk-align the split.
    chunk = seq_len + 1
    split = (int(len(data) * (1.0 - tiny_config["val_split"])) // chunk) * chunk
    train_ds = ds.PackedDataset(data[:split], seq_len, eos)
    val_ds = ds.PackedDataset(data[split:], seq_len, eos)
    sampler = ds.ShuffledRangeSampler(train_ds.n_chunks, seed=42, offset=0)
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=tiny_config["batch_size"], sampler=sampler,
        collate_fn=ds.collate_fn, drop_last=True)
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=tiny_config["batch_size"], shuffle=False,
        collate_fn=ds.collate_fn)
    return train_dl, val_dl


class TestEndToEndSmoke:
    def test_one_forward_backward_step(self, tiny_model, tiny_config,
                                         tiny_dataloaders, device):
        """Single optimizer step on synthetic data; loss is finite and grads flow."""
        train_dl, _ = tiny_dataloaders
        opt = torch.optim.AdamW(tiny_model.parameters(), lr=3e-4)

        tiny_model.train()
        batch = next(iter(train_dl))
        ids = batch["input"].to(device)
        tgt = batch["target"].to(device)
        logits = tiny_model(ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1))
        loss.backward()
        # Check grads BEFORE zero_grad (PyTorch defaults to set_to_none=True,
        # so grads would be None after the step otherwise).
        grad_snapshot = {name: (p.grad is not None,
                                 p.grad.isfinite().all().item() if p.grad is not None else False)
                         for name, p in tiny_model.named_parameters()}
        opt.step()
        opt.zero_grad()

        assert torch.isfinite(loss).item()
        assert loss.item() > 0
        # Every parameter must have received a finite grad from backward.
        for name, (has_grad, is_finite) in grad_snapshot.items():
            assert has_grad, f"no grad for {name}"
            assert is_finite, f"non-finite grad for {name}"

    def test_loss_decreases_over_few_steps(self, tiny_config, device,
                                            seed_everything):
        """Sanity check: a tiny model can overfit a single batch.

        If loss does not go down on a single repeated batch, something is
        fundamentally broken (init scale, gradient flow, optimizer wiring).
        This is *not* a generalization claim — it's a wiring smoke test.
        """
        from model import build_transformer
        seed_everything(0)
        model = build_transformer(
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
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)

        torch.manual_seed(1)
        ids = torch.randint(0, tiny_config["vocab_size"],
                            (4, tiny_config["seq_len"]), device=device)
        tgt = torch.randint(0, tiny_config["vocab_size"],
                            (4, tiny_config["seq_len"]), device=device)

        first_loss = None
        last_loss = None
        model.train()
        for step in range(30):
            logits = model(ids)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   tgt.view(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step == 0:
                first_loss = loss.item()
            last_loss = loss.item()
        assert last_loss < first_loss, (first_loss, last_loss)

    def test_chunked_ce_matches_full_ce_in_training(self, tiny_model,
                                                      tiny_config, device):
        """In a real forward pass the chunked loss must match the dense one."""
        ids = torch.randint(0, tiny_config["vocab_size"],
                            (4, tiny_config["seq_len"]), device=device)
        tgt = torch.randint(0, tiny_config["vocab_size"],
                            (4, tiny_config["seq_len"]), device=device)
        tiny_model.eval()
        with torch.no_grad():
            logits = tiny_model(ids)
            full = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   tgt.view(-1), reduction="mean")
            from model import chunked_cross_entropy
            chk = chunked_cross_entropy(logits.view(-1, logits.size(-1)),
                                        tgt.view(-1), chunk_size=7)
        assert torch.allclose(full, chk, atol=1e-5), (full, chk)

    def test_validate_runs_and_returns_finite_loss(self, tiny_model, tiny_config,
                                                    tiny_dataloaders, device,
                                                    monkeypatch):
        """``validate`` calls wandb.log; stub it so the test is offline."""
        train_dl, val_dl = tiny_dataloaders
        # Stub out wandb.log used inside validate.
        import wandb
        calls = []
        monkeypatch.setattr(wandb, "log", lambda *a, **k: calls.append((a, k)),
                            raising=False)
        pad_id = 0
        loss = train_mod.validate(tiny_model, val_dl, pad_id, device,
                                  step=0, config=tiny_config)
        assert np.isfinite(loss)
        assert loss > 0
        # validate logs val/loss and val/perplexity.
        assert len(calls) == 1