"""
================================================================================
FILE: tests/test_integration.py
PROJECT: Transformer Language Model — 45H Final Phase
PURPOSE: Integration tests — modules working together across the full pipeline
================================================================================

Run:  pytest tests/test_integration.py -v --tb=short
================================================================================
"""

import sys
import os
import json
import math
import shutil
import tempfile
import pytest
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from decoder        import Decoder
from encoder        import Encoder
from model          import (LanguageModel, AdamW, get_lr,
                             _extract_model_state, _load_model_state,
                             _atomic_save, _load_checkpoint_safe,
                             CheckpointCorruptError, CheckpointNotFoundError)
from config_loader  import load_config, ConfigError
from logger         import setup_logging, TrainingDashboard


# ──────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmpdir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)

@pytest.fixture
def small_lm(tmpdir):
    """Minimal LanguageModel backed by tiny.yaml, output routed to tmpdir."""
    cfg_path = Path(__file__).parent.parent / "config" / "tiny.yaml"
    overrides = [
        f"paths.output_dir={tmpdir}",
        f"paths.log_dir={tmpdir}/logs",
        f"paths.checkpoint_dir={tmpdir}/ckpts",
        "logging.log_to_file=false",
        "logging.log_to_console=false",
    ]
    return LanguageModel(str(cfg_path), overrides=overrides)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG SYSTEM INTEGRATION
# ══════════════════════════════════════════════════════════════════════════════

class TestConfigIntegration:
    def test_tiny_preset_loads(self):
        cfg = load_config("config/tiny.yaml")
        assert cfg.model.d_model == 128
        assert cfg.model.num_layers == 4

    def test_cli_override_applied(self):
        cfg = load_config("config/tiny.yaml", overrides=["train.batch_size=999"])
        assert cfg.train.batch_size == 999

    def test_invalid_config_raises(self):
        with pytest.raises(ConfigError):
            load_config(overrides=["model.d_model=7", "model.num_heads=8"])

    def test_seq_len_exceeds_max_seq_len_raises(self):
        with pytest.raises(ConfigError):
            load_config(overrides=[
                "model.max_seq_len=64",
                "train.seq_len=128",   # > max_seq_len
                "model.d_model=128",
                "model.num_heads=4",
                "model.num_kv_heads=2",
                "model.vocab_size=1000",
                "model.num_layers=2",
                "model.dropout_rate=0.1",
                "model.ffn_type=swiglu",
                "model.tie_weights=true",
            ])

    def test_nonexistent_config_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("config/nonexistent_preset.yaml")

    def test_deep_merge_does_not_bleed(self):
        """CLI override of one field must not reset sibling fields."""
        cfg = load_config("config/tiny.yaml", overrides=["model.num_layers=6"])
        assert cfg.model.d_model == 128   # unchanged by overriding num_layers


# ══════════════════════════════════════════════════════════════════════════════
# LR SCHEDULER
# ══════════════════════════════════════════════════════════════════════════════

class TestLRScheduler:
    def test_warmup_increases_monotonically(self):
        lrs = [get_lr(s, 1e-3, 1e-4, warmup_steps=100, max_steps=1000) for s in range(100)]
        for i in range(1, len(lrs)):
            assert lrs[i] >= lrs[i-1], f"LR decreased during warmup at step {i}"

    def test_peak_reached_after_warmup(self):
        peak = get_lr(100, 1e-3, 1e-4, warmup_steps=100, max_steps=1000)
        assert abs(peak - 1e-3) < 1e-6

    def test_cosine_decays_from_peak(self):
        lrs = [get_lr(s, 1e-3, 1e-4, warmup_steps=100, max_steps=1000)
               for s in range(100, 1001, 100)]
        for i in range(1, len(lrs)):
            assert lrs[i] <= lrs[i-1], f"Cosine LR increased at decay step {i}"

    def test_floor_at_min_lr(self):
        floor = get_lr(9999, 1e-3, 1e-4, warmup_steps=100, max_steps=1000)
        assert abs(floor - 1e-4) < 1e-7

    def test_linear_schedule(self):
        lr = get_lr(500, 1e-3, 1e-4, warmup_steps=100, max_steps=1000, schedule="linear")
        # Halfway through decay → roughly halfway between min and max
        assert 1e-4 <= lr <= 1e-3

    def test_constant_schedule(self):
        lr = get_lr(500, 1e-3, 1e-4, warmup_steps=100, max_steps=1000, schedule="constant")
        assert abs(lr - 1e-3) < 1e-7


# ══════════════════════════════════════════════════════════════════════════════
# ADAMW OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════════

class TestAdamW:
    def test_loss_decreases(self):
        """Simple parabolic loss: f(x) = x² — optimizer should reach x≈0."""
        rng = np.random.default_rng(0)
        params = {"x": rng.standard_normal((10,)).astype(np.float32)}
        opt = AdamW(params, lr=0.01, weight_decay=0.0, grad_clip=0.0)

        for _ in range(500):
            grads = {"x": 2.0 * params["x"]}   # df/dx = 2x
            opt.step(grads)

        assert np.abs(params["x"]).max() < 0.1, "AdamW failed to minimize x²"

    def test_weight_decay_shrinks_weights(self):
        params = {"w": np.ones(10, dtype=np.float32)}
        opt = AdamW(params, lr=0.1, weight_decay=0.5, grad_clip=0.0)
        for _ in range(100):
            opt.step({"w": np.zeros(10, dtype=np.float32)})  # zero gradient
        assert np.abs(params["w"]).mean() < 0.5

    def test_grad_clip(self):
        params = {"w": np.ones(10, dtype=np.float32)}
        opt = AdamW(params, lr=0.001, weight_decay=0.0, grad_clip=1.0)
        huge_grad = {"w": np.full(10, 1e6, dtype=np.float32)}
        norm = opt.step(huge_grad)
        # After clipping, params should not explode
        assert np.isfinite(params["w"]).all()

    def test_set_lr(self):
        params = {"w": np.zeros(4, dtype=np.float32)}
        opt = AdamW(params, lr=1e-3)
        opt.set_lr(1e-2)
        assert opt.lr == 1e-2


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

class TestCheckpointSystem:
    def test_save_and_load_roundtrip(self, tmpdir):
        dec = Decoder(vocab_size=64, d_model=32, num_layers=2, num_heads=2,
                      num_kv_heads=1, max_seq_len=32, seed=0)
        state_before = {k: v.copy() for k, v in _extract_model_state(dec).items()}

        # Save
        ckpt = tmpdir / "test.npy"
        data = {"step": 42, "loss": 2.5, "config": {}, "model_state": state_before}
        _atomic_save(data, ckpt)
        assert ckpt.exists()

        # Load
        loaded = _load_checkpoint_safe(ckpt)
        assert loaded["step"] == 42
        assert abs(loaded["loss"] - 2.5) < 1e-6

        # Verify weights match
        for k, v in state_before.items():
            np.testing.assert_array_equal(v, loaded["model_state"][k])

    def test_corrupt_checkpoint_raises(self, tmpdir):
        bad = tmpdir / "corrupt.npy"
        bad.write_bytes(b"this is not valid numpy data!!!")
        with pytest.raises(CheckpointCorruptError):
            _load_checkpoint_safe(bad)

    def test_missing_checkpoint_raises(self, tmpdir):
        with pytest.raises(CheckpointNotFoundError):
            _load_checkpoint_safe(tmpdir / "nonexistent.npy")

    def test_atomic_write_no_partial(self, tmpdir):
        """Verify temp file is cleaned up and final file exists."""
        data = {"step": 1, "loss": 0.5, "config": {}, "model_state": {}}
        path = tmpdir / "atomic.npy"
        _atomic_save(data, path)
        tmp = path.with_suffix(".tmp")
        assert path.exists()
        assert not tmp.exists()

    def test_missing_keys_raises(self, tmpdir):
        """Checkpoint missing 'model_state' key should raise."""
        data = {"step": 1, "loss": 0.5}   # missing config + model_state
        path = tmpdir / "incomplete.npy"
        np.save(str(path), data, allow_pickle=True)
        with pytest.raises(CheckpointCorruptError):
            _load_checkpoint_safe(path)

    def test_language_model_save_load(self, small_lm, tmpdir):
        """LanguageModel.save() then .load() must restore identical weights."""
        path = tmpdir / "test_ckpt.npy"
        small_lm.save(path, step=100, loss=1.5)

        # Extract weights before load
        state_before = _extract_model_state(small_lm._model)
        vals_before  = {k: v.copy() for k, v in state_before.items()}

        # Overwrite weights with noise, then reload
        for v in state_before.values():
            v[:] = 0.0
        _load_model_state(small_lm._model, vals_before)

        step = small_lm.load(path)
        assert step == 100

        state_after = _extract_model_state(small_lm._model)
        for k in vals_before:
            np.testing.assert_array_equal(vals_before[k], state_after[k])


# ══════════════════════════════════════════════════════════════════════════════
# LANGUAGE MODEL API
# ══════════════════════════════════════════════════════════════════════════════

class TestLanguageModelAPI:
    def test_generate_returns_string(self, small_lm):
        result = small_lm.generate("Hello world", max_new_tokens=10, temperature=0.0)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_includes_prompt(self, small_lm):
        prompt = "Test prompt"
        result = small_lm.generate(prompt, max_new_tokens=5, temperature=0.0)
        # Decoded output starts with the prompt (character-level tokenizer)
        assert result.startswith(prompt)

    def test_empty_prompt_raises(self, small_lm):
        with pytest.raises(ValueError, match="empty"):
            small_lm.generate("")

    def test_whitespace_only_prompt_raises(self, small_lm):
        with pytest.raises(ValueError):
            small_lm.generate("   ")

    def test_num_parameters_positive(self, small_lm):
        assert small_lm.num_parameters > 0

    def test_repr_contains_key_info(self, small_lm):
        r = repr(small_lm)
        assert "LanguageModel" in r
        assert "params=" in r

    def test_evaluate_returns_dict(self, small_lm, tmpdir):
        # Create a tiny eval text file
        eval_file = tmpdir / "eval.txt"
        eval_file.write_text("Hello world " * 500, encoding="utf-8")
        results = small_lm.evaluate(str(eval_file), max_batches=5)
        assert "loss" in results
        assert "perplexity" in results
        assert results["loss"] > 0
        assert results["perplexity"] > 1.0

    def test_evaluate_missing_file_raises(self, small_lm):
        with pytest.raises(FileNotFoundError):
            small_lm.evaluate("/nonexistent/path/data.txt")

    def test_train_tiny(self, tmpdir):
        """Quick 10-step training run — loss must be finite."""
        # Write minimal training data
        train_file = tmpdir / "train.txt"
        val_file   = tmpdir / "val.txt"
        train_file.write_text("The quick brown fox jumps over the lazy dog. " * 200)
        val_file.write_text("Hello world how are you doing today? " * 50)

        lm = LanguageModel("config/tiny.yaml", overrides=[
            f"train.dataset_path={train_file}",
            f"train.val_dataset_path={val_file}",
            f"paths.checkpoint_dir={tmpdir}/ckpts",
            f"paths.log_dir={tmpdir}/logs",
            "train.max_steps=10",
            "train.eval_every=5",
            "train.checkpoint_every=10",
            "train.batch_size=2",
            "train.seq_len=32",
            "logging.log_to_file=false",
            "logging.log_to_console=false",
        ])
        results = lm.train()
        assert results["final_loss"] is not None
        assert math.isfinite(results["final_loss"])
        assert results["total_steps"] == 10


# ══════════════════════════════════════════════════════════════════════════════
# ENCODER ↔ DECODER PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class TestEncoderDecoderPipeline:
    def test_seq2seq_forward_pass(self):
        rng = np.random.default_rng(42)
        enc = Encoder(vocab_size=128, d_model=64, num_layers=2, num_heads=4, num_kv_heads=2, max_seq_len=64)
        dec = Decoder(vocab_size=128, d_model=64, num_layers=2, num_heads=4, num_kv_heads=2, max_seq_len=64, use_cross_attn=True)

        src = rng.integers(1, 128, size=(1, 12)).astype(np.int64)
        tgt = rng.integers(1, 128, size=(1, 8)).astype(np.int64)

        mem = enc.forward(src, training=False)
        logits, _ = dec.forward(tgt, enc_memory=mem, training=False)

        assert mem.shape    == (1, 12, 64)
        assert logits.shape == (1, 8,  128)
        assert np.isfinite(logits).all()

    def test_gpt_pipeline(self):
        rng = np.random.default_rng(99)
        dec = Decoder(vocab_size=128, d_model=64, num_layers=2, num_heads=4,
                      num_kv_heads=2, max_seq_len=64, use_cross_attn=False, seed=0)
        prompt = np.array([[1, 10, 20]], dtype=np.int64)
        gen = dec.generate(
            prompt, max_new_tokens=15,
            temperature=0.8, top_k=30, top_p=0.92, repetition_penalty=1.1,
        )
        assert gen.shape == (1, 18)
        assert np.all(gen >= 0) and np.all(gen < 128)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
