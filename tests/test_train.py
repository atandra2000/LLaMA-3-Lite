"""Tests for ``train.py`` (unit-testable components only, not full train_model)."""
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


class TestCosineWithWarmup:
    def _make(self, warmup=10, max_steps=100, min_lr=1e-5, peak_lr=1e-3):
        opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=peak_lr)
        return train_mod.CosineWithWarmup(opt, warmup, max_steps, min_lr, peak_lr)

    def test_warmup_starts_near_zero(self):
        sched = self._make()
        sched.step()
        assert sched.get_lr() == pytest.approx(1e-3 * 1 / 10, rel=1e-9)

    def test_warmup_is_linear(self):
        sched = self._make(warmup=10, peak_lr=1e-3)
        lrs = []
        for _ in range(10):
            sched.step()
            lrs.append(sched.get_lr())
        expected = [1e-3 * (k + 1) / 10 for k in range(10)]
        assert lrs == pytest.approx(expected, rel=1e-9)

    def test_peak_at_end_of_warmup(self):
        sched = self._make(warmup=10, max_steps=100, peak_lr=1e-3)
        for _ in range(10):
            sched.step()
        assert sched.get_lr() == pytest.approx(1e-3, rel=1e-9)

    def test_decay_is_cosine(self):
        sched = self._make(warmup=0, max_steps=100, min_lr=1e-5, peak_lr=1e-3)
        for _ in range(50):
            sched.step()
        progress = 50 / 100
        expected = 1e-5 + (1e-3 - 1e-5) * 0.5 * (1 + math.cos(math.pi * progress))
        assert sched.get_lr() == pytest.approx(expected, rel=1e-9)

    def test_never_below_min_lr(self):
        sched = self._make(warmup=0, max_steps=100, min_lr=1e-5, peak_lr=1e-3)
        for _ in range(200):
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
        logits = torch.tensor([[1.0, 5.0, 2.0, 3.0]], device=device)
        for _ in range(5):
            tok = train_mod.top_k_top_p_sampling(logits, top_k=1, top_p=0.0,
                                                 temperature=1.0)
            assert tok.item() == 1

    def test_temperature_scales_logits(self, device):
        logits = torch.randn(1, 50, device=device)
        tok = train_mod.top_k_top_p_sampling(logits, top_k=0, top_p=0.0,
                                             temperature=0.99)
        assert 0 <= tok.item() < 50

    def test_top_p_prunes_low_prob_tail(self, device):
        logits = torch.tensor([[10.0, -10.0, -10.0, -10.0]], device=device)
        toks = [train_mod.top_k_top_p_sampling(logits, top_k=0, top_p=0.5,
                                               temperature=1.0).item()
                for _ in range(20)]
        assert set(toks) == {0}

    def test_handles_neg_inf_logits(self, device):
        logits = torch.full((1, 10), float("-inf"), device=device)
        logits[0, 3] = 1.0
        tok = train_mod.top_k_top_p_sampling(logits, top_k=5, top_p=0.9,
                                              temperature=1.0)
        assert tok.item() == 3
        assert torch.isfinite(tok).all()


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
        ids = torch.randint(0, tiny_config["vocab_size"],
                            (2, tiny_config["seq_len"]),
                            device=device, dtype=torch.long)
        model.eval()
        with torch.no_grad():
            ref_out = model(ids).clone()

        train_mod.save_checkpoint(model, opt, sched, step=1, config=cfg,
                                  best_val_loss=1.0, async_save=False)

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
        """The README promises 'exact reproducibility' via full RNG restore."""
        model, opt, sched = tiny_modules
        cfg = {**tiny_config, "model_folder": str(tmp_path), "async_checkpoint": False}

        torch.manual_seed(7); np.random.seed(7); random.seed(7)
        _ = torch.rand(10); _ = np.random.rand(10); _ = [random.random() for _ in range(10)]
        torch_after_draw = torch.random.get_rng_state().clone()
        np_after_draw = np.random.get_state()
        py_after_draw = random.getstate()

        train_mod.save_checkpoint(model, opt, sched, step=1, config=cfg,
                                  best_val_loss=1.0, async_save=False)

        torch.manual_seed(0); np.random.seed(0); random.seed(0)
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

        assert torch.equal(torch.random.get_rng_state(), torch_after_draw)
        assert np.array_equal(np.random.get_state()[1], np_after_draw[1])
        assert random.getstate()[1] == py_after_draw[1]

        expected_t = torch.rand(5)
        torch.random.set_rng_state(torch_after_draw)
        expected_t = torch.rand(5)
        torch.manual_seed(0); torch.rand(50)
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
        assert not (tmp_path / f"{cfg['model_filename']}_step_{cfg['max_steps']}.pt").exists()

    def test_async_save_returns_thread(self, tiny_modules, tiny_config, tmp_path):
        model, opt, sched = tiny_modules
        cfg = {**tiny_config, "model_folder": str(tmp_path), "async_checkpoint": True}
        t = train_mod.save_checkpoint(model, opt, sched, step=5, config=cfg,
                                       best_val_loss=1.0, async_save=True)
        assert t is not None
        assert t.is_alive() or not t.is_alive()
        t.join(timeout=5)
        assert (tmp_path / f"{cfg['model_filename']}_step_5.pt").exists()

    @pytest.mark.gpu
    def test_load_restores_rng_state_cross_device(self, tiny_config, device,
                                                   tmp_path):
        """Regression: torch.load(map_location=device) moved RNG state tensors to the load device."""
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
        ids = torch.randint(0, tiny_config["vocab_size"],
                            (2, tiny_config["seq_len"]),
                            device=save_device, dtype=torch.long)
        loss = model(ids).sum()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()

        cfg = {**tiny_config, "model_folder": str(tmp_path),
               "async_checkpoint": False, "batch_size": 2}
        train_mod.save_checkpoint(model, opt, sched, step=3, config=cfg,
                                   best_val_loss=1.5, async_save=False)

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
        step, best = train_mod.load_checkpoint(fresh, fresh_opt, fresh_sched,
                                               cfg, load_device)
        assert step == 3
        assert best == 1.5
        ids_cpu = ids.cpu()
        model.eval(); fresh.eval()
        with torch.no_grad():
            ref = model(ids).cpu()
            got = fresh(ids_cpu)
        assert torch.allclose(ref, got, atol=1e-4), (ref - got).abs().max()


class TestSetupGpuOptimizations:
    def test_idempotent_on_cpu(self, tiny_config):
        cfg = {**tiny_config, "tf32": False, "cudnn_benchmark": False}
        cfg.pop("cuda_alloc_conf", None)
        train_mod.setup_gpu_optimizations(cfg)
        train_mod.setup_gpu_optimizations(cfg)