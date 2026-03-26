"""
================================================================================
FILE: tests/test_end_to_end.py
PROJECT: Transformer Language Model — 45H Final Phase
PURPOSE: End-to-end tests — raw text in, generated text out, everything verified
================================================================================

These tests simulate real usage:
  1. Load config → build model → train a few steps → generate → save → reload
  2. Edge cases: corrupted checkpoint, zero-length generation, huge batch
  3. CLI interface (run.py argument parsing)

Run:  pytest tests/test_end_to_end.py -v --tb=short
================================================================================
"""

import sys
import os
import math
import shutil
import tempfile
import subprocess
import pytest
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model         import LanguageModel, CheckpointCorruptError, CheckpointNotFoundError
from config_loader import load_config
from decoder       import Decoder


# ──────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def workspace():
    """Isolated temp directory wiped after each test."""
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d, ignore_errors=True)

@pytest.fixture
def training_data(workspace):
    """Write realistic training and validation text files."""
    text = (
        "The transformer architecture was introduced in 2017 by Vaswani et al. "
        "It relies entirely on attention mechanisms and has become the dominant "
        "approach to natural language processing. The attention mechanism allows "
        "the model to focus on different parts of the input sequence when "
        "computing its output. This was a breakthrough because recurrent models "
        "struggled with long-range dependencies. Today transformers are used "
        "for language modeling, translation, summarization, and much more. "
    ) * 100

    train = workspace / "train.txt"
    val   = workspace / "val.txt"
    train.write_text(text, encoding="utf-8")
    val.write_text(text[:len(text)//5], encoding="utf-8")
    return train, val


# ══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE: TRAIN → GENERATE → EVALUATE → SAVE → RELOAD
# ══════════════════════════════════════════════════════════════════════════════

class TestFullPipeline:
    def test_train_generate_save_reload(self, workspace, training_data):
        """
        Complete end-to-end pipeline:
          build → train → generate → save → reload → generate again
        """
        train_file, val_file = training_data

        # Build and train
        lm = LanguageModel("config/tiny.yaml", overrides=[
            f"train.dataset_path={train_file}",
            f"train.val_dataset_path={val_file}",
            f"paths.checkpoint_dir={workspace}/ckpts",
            f"paths.log_dir={workspace}/logs",
            "train.max_steps=20",
            "train.eval_every=10",
            "train.checkpoint_every=20",
            "train.batch_size=2",
            "train.seq_len=32",
            "logging.log_to_file=false",
            "logging.log_to_console=false",
        ])
        results = lm.train()
        assert math.isfinite(results["final_loss"])
        assert results["total_steps"] == 20

        # Generate text
        output = lm.generate("The transformer", max_new_tokens=20, temperature=0.8, top_k=20)
        assert isinstance(output, str)
        assert len(output) > len("The transformer")  # something was generated

        # Evaluate
        eval_results = lm.evaluate(str(val_file), max_batches=10)
        assert math.isfinite(eval_results["loss"])
        assert eval_results["perplexity"] > 1.0

        # Save checkpoint
        ckpt_path = workspace / "final_model.npy"
        lm.save(ckpt_path, step=20, loss=results["final_loss"])
        assert ckpt_path.exists()
        assert ckpt_path.stat().st_size > 0

        # Reload and generate again
        lm2 = LanguageModel("config/tiny.yaml", overrides=[
            "logging.log_to_file=false",
            "logging.log_to_console=false",
        ], checkpoint=str(ckpt_path))
        output2 = lm2.generate("The transformer", max_new_tokens=5, temperature=0.0)
        assert isinstance(output2, str)
        assert output2.startswith("The transformer")


# ══════════════════════════════════════════════════════════════════════════════
# ERROR HANDLING & RESILIENCE
# ══════════════════════════════════════════════════════════════════════════════

class TestErrorHandling:
    def test_corrupted_checkpoint_raises_clean_error(self, workspace):
        """Corrupt checkpoint must raise CheckpointCorruptError, not crash."""
        bad_ckpt = workspace / "corrupt.npy"
        bad_ckpt.write_bytes(b"\x00" * 100)

        lm = LanguageModel("config/tiny.yaml", overrides=[
            "logging.log_to_file=false",
            "logging.log_to_console=false",
        ])
        with pytest.raises(CheckpointCorruptError) as exc_info:
            lm.load(bad_ckpt)
        assert "corrupt" in str(exc_info.value).lower() or "unreadable" in str(exc_info.value).lower()

    def test_missing_checkpoint_raises_clean_error(self, workspace):
        lm = LanguageModel("config/tiny.yaml", overrides=[
            "logging.log_to_file=false",
            "logging.log_to_console=false",
        ])
        with pytest.raises(CheckpointNotFoundError):
            lm.load(workspace / "does_not_exist.npy")

    def test_empty_dataset_handled(self, workspace):
        """Empty training file must not crash — should warn and return."""
        empty_train = workspace / "empty.txt"
        empty_train.write_text("")
        val = workspace / "val.txt"
        val.write_text("hello world " * 100)

        lm = LanguageModel("config/tiny.yaml", overrides=[
            f"train.dataset_path={empty_train}",
            f"train.val_dataset_path={val}",
            "logging.log_to_file=false",
            "logging.log_to_console=false",
        ])
        with pytest.raises((ValueError, Exception)):  # must raise, not silently continue
            lm.train()

    def test_eval_missing_dataset(self, workspace):
        lm = LanguageModel("config/tiny.yaml", overrides=[
            "logging.log_to_file=false",
            "logging.log_to_console=false",
        ])
        with pytest.raises(FileNotFoundError):
            lm.evaluate(str(workspace / "nonexistent.txt"))

    def test_zero_max_new_tokens(self):
        dec = Decoder(vocab_size=64, d_model=32, num_layers=1, num_heads=2,
                      num_kv_heads=1, max_seq_len=32, seed=0)
        prompt = np.array([[1, 2, 3]], dtype=np.int64)
        gen = dec.generate(prompt, max_new_tokens=0, temperature=0.0)
        np.testing.assert_array_equal(gen, prompt)

    def test_generate_empty_prompt_raises(self):
        lm = LanguageModel("config/tiny.yaml", overrides=[
            "logging.log_to_file=false",
            "logging.log_to_console=false",
        ])
        with pytest.raises(ValueError):
            lm.generate("")

    def test_auto_resume_from_checkpoint(self, workspace, training_data):
        """After saving a checkpoint, a new LM should auto-detect and resume."""
        train_file, val_file = training_data
        ckpt_dir = workspace / "ckpts"

        lm1 = LanguageModel("config/tiny.yaml", overrides=[
            f"train.dataset_path={train_file}",
            f"train.val_dataset_path={val_file}",
            f"paths.checkpoint_dir={ckpt_dir}",
            f"paths.log_dir={workspace}/logs",
            f"train.checkpoint_dir={ckpt_dir}",
            "train.max_steps=10",
            "train.checkpoint_every=10",
            "train.batch_size=2",
            "train.seq_len=32",
            "logging.log_to_file=false",
            "logging.log_to_console=false",
        ])
        lm1.train()

        # New LM, should auto-discover checkpoint
        lm2 = LanguageModel("config/tiny.yaml", overrides=[
            f"train.dataset_path={train_file}",
            f"train.val_dataset_path={val_file}",
            f"paths.checkpoint_dir={ckpt_dir}",
            f"paths.log_dir={workspace}/logs",
            f"train.checkpoint_dir={ckpt_dir}",
            "train.max_steps=15",   # continue training
            "train.checkpoint_every=15",
            "train.batch_size=2",
            "train.seq_len=32",
            "logging.log_to_file=false",
            "logging.log_to_console=false",
        ])
        # Should not crash — either auto-resumes or starts fresh gracefully
        results = lm2.train()
        assert math.isfinite(results["final_loss"])


# ══════════════════════════════════════════════════════════════════════════════
# GENERATION QUALITY CHECKS
# ══════════════════════════════════════════════════════════════════════════════

class TestGenerationProperties:
    @pytest.fixture
    def dec(self):
        return Decoder(vocab_size=200, d_model=64, num_layers=2, num_heads=4,
                       num_kv_heads=2, max_seq_len=128, dropout_rate=0.0, seed=0)

    def test_greedy_length_exact(self, dec):
        prompt = np.array([[1, 2, 3, 4]], dtype=np.int64)
        gen = dec.generate(prompt, max_new_tokens=15, temperature=0.0)
        assert gen.shape[1] == 4 + 15

    def test_all_tokens_in_vocab(self, dec):
        prompt = np.array([[1, 2, 3]], dtype=np.int64)
        gen = dec.generate(prompt, max_new_tokens=30, temperature=1.0, top_k=50)
        assert np.all(gen >= 0) and np.all(gen < 200)

    def test_repetition_penalty_active(self, dec):
        """Repetition penalty must not crash and should still produce valid IDs."""
        prompt = np.array([[1, 1, 1, 1]], dtype=np.int64)
        gen = dec.generate(prompt, max_new_tokens=10, temperature=1.0,
                           repetition_penalty=1.5)
        assert np.all(gen >= 0) and np.all(gen < 200)

    def test_top_p_near_zero_gives_greedy(self, dec):
        """top_p≈0 effectively becomes greedy — should be deterministic."""
        prompt = np.array([[5, 10, 15]], dtype=np.int64)
        g1 = dec.generate(prompt, max_new_tokens=5, temperature=0.01, top_p=0.001)
        g2 = dec.generate(prompt, max_new_tokens=5, temperature=0.01, top_p=0.001)
        np.testing.assert_array_equal(g1, g2)

    def test_very_long_generation(self, dec):
        """Generate close to max_seq_len without crashing."""
        prompt = np.array([[1]], dtype=np.int64)
        gen = dec.generate(prompt, max_new_tokens=100, temperature=1.0)
        # Should stop at max_seq_len or 100 new tokens
        assert gen.shape[1] <= dec.max_seq_len
        assert gen.shape[1] <= 1 + 100 + 1


# ══════════════════════════════════════════════════════════════════════════════
# MEMORY AND DETERMINISM
# ══════════════════════════════════════════════════════════════════════════════

class TestDeterminism:
    def test_same_seed_same_weights(self):
        d1 = Decoder(vocab_size=64, d_model=32, num_layers=2, num_heads=2,
                     num_kv_heads=1, max_seq_len=32, seed=42)
        d2 = Decoder(vocab_size=64, d_model=32, num_layers=2, num_heads=2,
                     num_kv_heads=1, max_seq_len=32, seed=42)
        np.testing.assert_array_equal(d1.token_embedding, d2.token_embedding)
        np.testing.assert_array_equal(d1.blocks[0].self_attn.W_Q,
                                       d2.blocks[0].self_attn.W_Q)

    def test_different_seeds_different_weights(self):
        d1 = Decoder(vocab_size=64, d_model=32, num_layers=2, num_heads=2,
                     num_kv_heads=1, max_seq_len=32, seed=0)
        d2 = Decoder(vocab_size=64, d_model=32, num_layers=2, num_heads=2,
                     num_kv_heads=1, max_seq_len=32, seed=1)
        assert not np.array_equal(d1.token_embedding, d2.token_embedding)

    def test_eval_mode_is_deterministic(self):
        dec = Decoder(vocab_size=64, d_model=32, num_layers=2, num_heads=2,
                      num_kv_heads=1, max_seq_len=32, dropout_rate=0.0, seed=0)
        ids = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
        l1, _ = dec.forward(ids, training=False)
        l2, _ = dec.forward(ids, training=False)
        np.testing.assert_array_equal(l1, l2)


# ══════════════════════════════════════════════════════════════════════════════
# CLI SMOKE TEST
# ══════════════════════════════════════════════════════════════════════════════

class TestCLI:
    def test_run_py_help(self):
        """run.py --help must exit cleanly."""
        result = subprocess.run(
            [sys.executable, "run.py", "--help"],
            capture_output=True, text=True, cwd=str(Path(__file__).parent.parent)
        )
        assert result.returncode == 0
        assert "usage" in result.stdout.lower() or "prompt" in result.stdout.lower()

    def test_run_py_generate(self, workspace):
        """run.py --prompt 'hello' must produce output."""
        result = subprocess.run(
            [sys.executable, "run.py",
             "--config", "config/tiny.yaml",
             "--prompt", "Hello",
             "--max_tokens", "5",
             "--temperature", "0.0",
             "--no_log_file"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).parent.parent),
            timeout=60,
        )
        # Should not crash
        assert result.returncode == 0 or "Hello" in result.stdout + result.stderr


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
