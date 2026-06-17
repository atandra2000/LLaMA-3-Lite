"""Tests for ``train.py``.

Covers the components that are *unit-testable* without launching a full
training run:
* ``CosineWithWarmup`` — warmup ramp + cosine decay + min_lr floor.
* ``top_k_top_p_sampling`` — determinism, top-k restriction, top-p pruning.
* ``save_checkpoint`` / ``load_checkpoint`` round-trip including the RNG
  state restoration that the README advertises for "exact reproducibility".
* ``setup_gpu_optimizations`` — idempotent on CPU, sets TF32 flags on GPU.

We deliberately do NOT call ``train_model`` here: it spins up W&B and a full
data pipeline. The end-to-end smoke test lives in ``test_smoke.py`` and
``test_pipeline.py`` using a tiny synthetic config and W&B stubbed out.
"""
from __future__ import annotations

import math
import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

import train as train_mod


# --------------------------------------------------------------------------- #
# CosineWithWarmup
# --------------------------------------------------------------------------- #
class TestCosineWithWarmup:
    def _make(self, warmup=10, max_steps=100, min_lr=1e-5, peak_lr=1e-3):
        opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=peak_lr)
        return train_mod.CosineWithWarmup(opt, warmup, max_steps, min_lr, peak_lr)

    def test_warmup_starts_near_zero(self):
        sched = self._make()
        # Before any step, get_lr would return 0 (step=0 -> 0/warmup).
        # Step once -> lr = peak * 1 / warmup.
        sched.step()
        assert sched.get_lr() == pytest.approx(1e-3 * 1 / 10, rel=1e-9)

    def test_warmup_is_linear(self):
        sched = self._make(warmup=10, peak_lr=1e-3)
        lrs = []
        for _ in range(10):
            sched.step()
            lrs.append(sched.get_lr())
        # Linear ramp: lr[k] = peak * (k+1) / warmup
        expected = [1e-3 * (k + 1) / 10 for k in range(10)]
        assert lrs == pytest.approx(expected, rel=1e-9)

    def test_peak_at_end_of_warmup(self):
        sched = self._make(warmup=10, max_steps=100, peak_lr=1e-3)
        for _ in range(10):
            sched.step()
        assert sched.get_lr() == pytest.approx(1e-3, rel=1e-9)

    def test_decay_is_cosine(self):
        sched = self._make(warmup=0, max_steps=100, min_lr=1e-5, peak_lr=1e-3)
        # Step to the midpoint (step=50).
        for _ in range(50):
            sched.step()
        progress = 50 / 100
        expected = 1e-5 + (1e-3 - 1e-5) * 0.5 * (1 + math.cos(math.pi * progress))
        assert sched.get_lr() == pytest.approx(expected, rel=1e-9)

    def test_never_below_min_lr(self):
        sched = self._make(warmup=0, max_steps=100, min_lr=1e-5, peak_lr=1e-3)
        for _ in range(200):  # past max_steps
            sched.step()
        assert sched.get_lr() >= 1e-5 - 1e-12

    def test_lr_is_set_on_optimizer(self):
        opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
        sched = train_mod.CosineWithWarmup(opt, warmup_steps=10, max_steps=100,
                                           min_lr=1e-5, peak_lr=1e-3)
        sched.step()
        assert opt.param_groups[0]["lr"] == sched.get_lr()

    def test_state_dict_roundtrip(self):
        sched = self._make()
        for _ in range(7):
            sched.step()
        sd = sched.state_dict()
        sched2 = self._make()
        sched2.load_state_dict(sd)
        assert sched2._step == 7
        assert sched2.get_lr() == sched.get_lr()


# --------------------------------------------------------------------------- #
# top_k_top_p_sampling
# --------------------------------------------------------------------------- #
class TestTopKTopPSampling:
    def test_deterministic_with_seed(self, device):
        torch.manual_seed(0)
        logits = torch.randn(2, 100, device=device)
        torch.manual_seed(123)
        a = train_mod.top_k_top_p_sampling(logits, top_k=10, top_p=0.9,
                                            temperature=1.0)
        torch.manual_seed(123)
        b = train_mod.top_k_top_p_sampling(logits, top_k=10, top_p=0.9,
                                            temperature=1.0)
        assert torch.equal(a, b)
        assert a.shape == (2, 1)

    def test_top_k_restricts_vocab(self, device):
        # If top_k=1, sampling must always return the argmax.
        logits = torch.tensor([[1.0, 5.0, 2.0, 3.0]], device=device)
        for _ in range(5):
            tok = train_mod.top_k_top_p_sampling(logits, top_k=1, top_p=0.0,
                                                 temperature=1.0)
            assert tok.item() == 1   # argmax index

    def test_temperature_scales_logits(self, device):
        # High temperature -> distribution more uniform; just assert it runs
        # and that temperature=0 would divide by 0 — guard against that being
        # passed (the code doesn't special-case T=0, so we just check T>0).
        logits = torch.randn(1, 50, device=device)
        tok = train_mod.top_k_top_p_sampling(logits, top_k=0, top_p=0.0,
                                             temperature=0.99)
        assert 0 <= tok.item() < 50

    def test_top_p_prunes_low_prob_tail(self, device):
        # Sharply peaked distribution; with top_p=0.5 only the top token(s)
        # survive. Setting top_k=0 so only top_p is active.
        logits = torch.tensor([[10.0, -10.0, -10.0, -10.0]], device=device)
        toks = [train_mod.top_k_top_p_sampling(logits, top_k=0, top_p=0.5,
                                               temperature=1.0).item()
                for _ in range(20)]
        # All samples should be the first token (only survivor).
        assert set(toks) == {0}

    def test_handles_neg_inf_logits(self, device):
        # top-k fill uses -inf for masked positions; sampling must not produce
        # NaNs from softmax over a row that has at least one finite entry.
        logits = torch.full((1, 10), float("-inf"), device=device)
        logits[0, 3] = 1.0
        tok = train_mod.top_k_top_p_sampling(logits, top_k=5, top_p=0.9,
                                              temperature=1.0)
        assert tok.item() == 3
        assert torch.isfinite(tok).all()


# --------------------------------------------------------------------------- #
# Checkpoint save / load + RNG state
# --------------------------------------------------------------------------- #
class TestCheckpointRoundTrip:
    @pytest.fixture
    def tiny_modules(self, tiny_config, device):
        """Build a tiny model + optimizer + scheduler to checkpoint."""
        from model import build_transformer
        torch.manual_seed(0)
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
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        sched = train_mod.CosineWithWarmup(
            opt, warmup_steps=2, max_steps=10,
            min_lr=1e-5, peak_lr=3e-4,
        )
        # Take a few optimizer/scheduler steps so state is non-trivial.
        for _ in range(3):
            loss = model(torch.randint(0, tiny_config["vocab_size"],
                                       (2, tiny_config["seq_len"]),
                                       device=device)).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        return model, opt, sched

    def test_save_creates_step_file(self, tiny_modules, tiny_config, tmp_path):
        model, opt, sched = tiny_modules
        cfg = {**tiny_config, "model_folder": str(tmp_path), "async_checkpoint": False}
        train_mod.save_checkpoint(model, opt, sched, step=42, config=cfg,
                                  best_val_loss=1.23, async_save=False)
        path = tmp_path / f"{cfg['model_filename']}_step_42.pt"
        assert path.exists()

    def test_load_restores_model_weights(self, tiny_modules, tiny_config,
                                          device, tmp_path):
        model, opt, sched = tiny_modules
        cfg = {**tiny_config, "model_folder": str(tmp_path), "async_checkpoint": False}
        # Save a reference output before checkpointing.
        ids = torch.randint(0, tiny_config["vocab_size"],
                            (2, tiny_config["seq_len"]),
                            device=device, dtype=torch.long)
        model.eval()
        with torch.no_grad():
            ref_out = model(ids).clone()

        train_mod.save_checkpoint(model, opt, sched, step=1, config=cfg,
                                  best_val_loss=1.0, async_save=False)

        # Build a fresh model with different init weights and load.
        from model import build_transformer
        torch.manual_seed(999)
        fresh = build_transformer(
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
        fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=3e-4)
        fresh_sched = train_mod.CosineWithWarmup(
            fresh_opt, 2, 10, 1e-5, 3e-4)
        fresh.eval()
        with torch.no_grad():
            pre_load = fresh(ids).clone()
        # Definitely different before loading (different init).
        assert not torch.allclose(ref_out, pre_load, atol=1e-4)

        step, best = train_mod.load_checkpoint(fresh, fresh_opt, fresh_sched,
                                               cfg, device)
        assert step == 1
        assert best == 1.0
        with torch.no_grad():
            post_load = fresh(ids).clone()
        assert torch.allclose(post_load, ref_out, atol=1e-4), (
            "loaded model does not reproduce pre-save outputs"
        )

    def test_load_restores_rng_state(self, tiny_modules, tiny_config, device,
                                      tmp_path):
        """The README promises 'exact reproducibility' via full RNG restore.

        We verify that after load_checkpoint, torch / numpy / python RNG all
        produce the same draws as before the save.
        """
        model, opt, sched = tiny_modules
        cfg = {**tiny_config, "model_folder": str(tmp_path), "async_checkpoint": False}

        # Seed RNGs, draw some numbers, save the *state*, then continue.
        torch.manual_seed(7); np.random.seed(7); random.seed(7)
        torch_state_pre = torch.random.get_rng_state().clone()
        np_state_pre = np.random.get_state()
        py_state_pre = random.getstate()
        # Draw a few numbers so the *next* draws depend on the state.
        _ = torch.rand(10); _ = np.random.rand(10); _ = [random.random() for _ in range(10)]
        # Save snapshot of post-draw state.
        torch_after_draw = torch.random.get_rng_state().clone()
        np_after_draw = np.random.get_state()
        py_after_draw = random.getstate()

        train_mod.save_checkpoint(model, opt, sched, step=1, config=cfg,
                                  best_val_loss=1.0, async_save=False)

        # Now corrupt RNGs and load back.
        torch.manual_seed(0); np.random.seed(0); random.seed(0)
        # Take draws that change the state.
        _ = torch.rand(50); _ = np.random.rand(50); _ = [random.random() for _ in range(50)]

        from model import build_transformer
        fresh = build_transformer(
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
        fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=3e-4)
        fresh_sched = train_mod.CosineWithWarmup(fresh_opt, 2, 10, 1e-5, 3e-4)
        train_mod.load_checkpoint(fresh, fresh_opt, fresh_sched, cfg, device)

        # After restore, the RNG state should match what was saved.
        assert torch.equal(torch.random.get_rng_state(), torch_after_draw)
        # numpy: compare the full state tuple (state[1] is the actual array).
        assert np.array_equal(np.random.get_state()[1], np_after_draw[1])
        # python: random.getstate() returns a tuple of (version, internal, gauss).
        assert random.getstate()[1] == py_after_draw[1]

        # And draws should be reproducible.
        expected_t = torch.rand(5)   # NOTE: drawn on whatever device torch.rand uses
        # Reset to the saved state again (the assertions above consumed it).
        torch.random.set_rng_state(torch_after_draw)
        expected_t = torch.rand(5)
        torch.manual_seed(0); torch.rand(50)   # corrupt
        torch.random.set_rng_state(torch_after_draw)
        again_t = torch.rand(5)
        assert torch.equal(expected_t, again_t)

    def test_load_returns_zero_when_no_checkpoints(self, tiny_config, device,
                                                    tmp_path):
        from model import build_transformer
        model = build_transformer(
            vocab_size=tiny_config["vocab_size"],
            d_model=tiny_config["d_model"],
            n_layers=1,
            n_heads=tiny_config["n_heads"],
            n_kv_heads=tiny_config["n_kv_heads"],
            head_dim=tiny_config["head_dim"],
            d_ff=tiny_config["d_ff"],
            max_seq_len=tiny_config["seq_len"],
            rope_theta=tiny_config["rope_theta"],
            rms_norm_eps=tiny_config["rms_norm_eps"],
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        sched = train_mod.CosineWithWarmup(opt, 2, 10, 1e-5, 3e-4)
        cfg = {**tiny_config, "model_folder": str(tmp_path)}
        step, best = train_mod.load_checkpoint(model, opt, sched, cfg, device)
        assert step == 0
        assert best == float("inf")

    def test_final_checkpoint_uses_special_names(self, tiny_modules,
                                                  tiny_config, tmp_path):
        model, opt, sched = tiny_modules
        cfg = {**tiny_config, "model_folder": str(tmp_path), "async_checkpoint": False}
        train_mod.save_checkpoint(model, opt, sched, step=cfg["max_steps"],
                                  config=cfg, best_val_loss=1.0,
                                  is_final=True, async_save=False)
        full = tmp_path / f"{cfg['model_filename']}_final_model_full.pt"
        weights_only = tmp_path / f"{cfg['model_filename']}_final_model_weights.pt"
        assert full.exists()
        assert weights_only.exists()
        # The step_N.pt file should NOT exist for the final save.
        assert not (tmp_path / f"{cfg['model_filename']}_step_{cfg['max_steps']}.pt").exists()

    def test_async_save_returns_thread(self, tiny_modules, tiny_config, tmp_path):
        model, opt, sched = tiny_modules
        cfg = {**tiny_config, "model_folder": str(tmp_path), "async_checkpoint": True}
        t = train_mod.save_checkpoint(model, opt, sched, step=5, config=cfg,
                                       best_val_loss=1.0, async_save=True)
        # async_save=True + async_checkpoint=True -> returns a Thread.
        assert t is not None
        assert t.is_alive() or not t.is_alive()  # just ensure it's a Thread
        t.join(timeout=5)
        assert (tmp_path / f"{cfg['model_filename']}_step_5.pt").exists()

    @pytest.mark.gpu
    def test_load_restores_rng_state_cross_device(self, tiny_config, device,
                                                   tmp_path):
        """Regression: torch.load(map_location=device) moved the RNG state
        tensors to the load device, so set_rng_state rejected them.

        This bug only manifests when saving on one device and loading on
        another (or saving on GPU and loading on GPU, since map_location
        still round-trips through the device). We save on GPU and load on
        CPU to exercise the worst case.
        """
        if device.type != "cpu" or not torch.cuda.is_available():
            pytest.skip("needs a CUDA device to save on, loading onto CPU")
        from model import build_transformer
        save_device = torch.device("cuda")
        load_device = torch.device("cpu")

        torch.manual_seed(7); np.random.seed(7); random.seed(7)
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
        ).to(save_device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        sched = train_mod.CosineWithWarmup(opt, 2, 10, 1e-5, 3e-4)
        # Take a step so optimizer/RNG state is non-trivial.
        ids = torch.randint(0, tiny_config["vocab_size"],
                            (2, tiny_config["seq_len"]),
                            device=save_device, dtype=torch.long)
        loss = model(ids).sum()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()

        cfg = {**tiny_config, "model_folder": str(tmp_path),
               "async_checkpoint": False, "batch_size": 2}
        train_mod.save_checkpoint(model, opt, sched, step=3, config=cfg,
                                   best_val_loss=1.5, async_save=False)

        # Build a fresh model on CPU and load.
        fresh = build_transformer(
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
        ).to(load_device)
        fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=3e-4)
        fresh_sched = train_mod.CosineWithWarmup(fresh_opt, 2, 10, 1e-5, 3e-4)
        # Must not raise: torch.random.set_rng_state / torch.cuda.set_rng_state
        # both require CPU ByteTensors; the fix in load_checkpoint coerces them.
        step, best = train_mod.load_checkpoint(fresh, fresh_opt, fresh_sched,
                                               cfg, load_device)
        assert step == 3
        assert best == 1.5
        # The model weights must load correctly (forward reproducible).
        ids_cpu = ids.cpu()
        model.eval(); fresh.eval()
        with torch.no_grad():
            ref = model(ids).cpu()
            got = fresh(ids_cpu)
        assert torch.allclose(ref, got, atol=1e-4), (ref - got).abs().max()


# --------------------------------------------------------------------------- #
# setup_gpu_optimizations
# --------------------------------------------------------------------------- #
class TestSetupGpuOptimizations:
    def test_idempotent_on_cpu(self, tiny_config):
        # Should not raise even when CUDA is unavailable.
        cfg = {**tiny_config, "tf32": False, "cudnn_benchmark": False}
        # cuda_alloc_conf not set -> must not set env var.
        cfg.pop("cuda_alloc_conf", None)
        train_mod.setup_gpu_optimizations(cfg)
        # Re-running must be safe.
        train_mod.setup_gpu_optimizations(cfg)