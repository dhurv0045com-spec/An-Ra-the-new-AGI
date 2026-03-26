"""
45G / test_45G.py — Full End-to-End Test Suite
================================================
Validates every module in 45G independently and then
verifies they integrate correctly.

Run:
    python test_45G.py
    python test_45G.py -v          # verbose
    python test_45G.py TestGreedy  # single class
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import List

import torch
import torch.nn as nn


# ─── Shared test fixtures ────────────────────────────────────────────────────

class _TinyLM(nn.Module):
    """Minimal GRU-based LM for offline tests (no GPU required)."""
    VOCAB = 64

    def __init__(self):
        super().__init__()
        self.emb  = nn.Embedding(self.VOCAB, 32)
        self.rnn  = nn.GRU(32, 64, batch_first=True)
        self.proj = nn.Linear(64, self.VOCAB)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.emb(x)
        h, _ = self.rnn(h)
        return self.proj(h)


class _TinyTok:
    vocab_size = 64

    def encode(self, text: str) -> List[int]:
        return [b % self.vocab_size for b in text.encode()]

    def decode(self, ids: List[int]) -> str:
        return bytes([i + 32 for i in ids]).decode(errors="replace")

    def to_dict(self):
        return {"vocab": [chr(i + 32) for i in range(self.vocab_size)]}


def _make_model():
    torch.manual_seed(0)
    return _TinyLM()


def _make_tok():
    return _TinyTok()


# ─── Step 27: greedy.py ──────────────────────────────────────────────────────

class TestGreedy(unittest.TestCase):

    def test_greedy_step_picks_argmax(self):
        from greedy import greedy_step
        logits = torch.tensor([0.1, 9.9, 0.3, 2.1])
        self.assertEqual(greedy_step(logits), 1)

    def test_greedy_step_negative_logits(self):
        from greedy import greedy_step
        logits = torch.tensor([-5.0, -1.0, -3.0])
        self.assertEqual(greedy_step(logits), 1)

    def test_greedy_decode_shape(self):
        from greedy import greedy_decode
        model = _make_model()
        ids   = torch.tensor([[1, 2, 3]])
        out   = greedy_decode(model, ids, max_new_tokens=10, device="cpu")
        self.assertEqual(out.shape[0], 1)
        self.assertEqual(out.shape[1], 13)        # 3 prompt + 10 new

    def test_greedy_decode_deterministic(self):
        from greedy import greedy_decode
        model = _make_model()
        ids   = torch.tensor([[5, 10, 15]])
        out1  = greedy_decode(model, ids, max_new_tokens=5, device="cpu")
        out2  = greedy_decode(model, ids, max_new_tokens=5, device="cpu")
        self.assertTrue(torch.equal(out1, out2), "Greedy must be deterministic")

    def test_greedy_decode_eos_stops_early(self):
        from greedy import greedy_decode
        model = _make_model()
        ids   = torch.tensor([[0]])
        # inject a bias so the model always predicts token 7 = EOS
        model.proj.bias.data.fill_(float("-inf"))
        model.proj.bias.data[7] = 10.0
        out = greedy_decode(model, ids, max_new_tokens=50,
                            eos_token_id=7, device="cpu")
        # Should stop after emitting one EOS token
        self.assertLessEqual(out.shape[1], 10)

    def test_greedy_decoder_class(self):
        from greedy import GreedyDecoder
        model = _make_model()
        tok   = _make_tok()
        dec   = GreedyDecoder(model, tok, device="cpu")
        out   = dec.generate("Hi", max_new_tokens=15)
        self.assertIsInstance(out, str)
        self.assertGreater(len(out), 0)


# ─── Steps 28-30: sampling.py ────────────────────────────────────────────────

class TestSampling(unittest.TestCase):

    def _logits(self, n=1000):
        torch.manual_seed(1)
        return torch.randn(n)

    def test_temperature_sharpening(self):
        from sampling import apply_temperature
        import torch.nn.functional as F
        logits = self._logits()
        p_cold = F.softmax(apply_temperature(logits, 0.1), dim=-1).max().item()
        p_hot  = F.softmax(apply_temperature(logits, 2.0), dim=-1).max().item()
        self.assertGreater(p_cold, p_hot)

    def test_temperature_zero_raises(self):
        from sampling import apply_temperature
        with self.assertRaises(ValueError):
            apply_temperature(torch.randn(10), 0.0)

    def test_top_k_count(self):
        from sampling import apply_top_k
        k = 10
        filtered = apply_top_k(self._logits(), k)
        n_valid  = (filtered > float("-inf")).sum().item()
        self.assertEqual(n_valid, k)

    def test_top_k_zero_is_noop(self):
        from sampling import apply_top_k
        logits   = self._logits()
        filtered = apply_top_k(logits, 0)
        self.assertTrue(torch.equal(logits, filtered))

    def test_top_p_mass_coverage(self):
        from sampling import apply_top_p
        import torch.nn.functional as F
        p = 0.9
        filtered = apply_top_p(self._logits(), p)
        probs    = F.softmax(filtered, dim=-1)
        mass     = probs[probs > 0].sum().item()
        self.assertGreaterEqual(mass, p - 1e-4)

    def test_top_p_one_is_noop(self):
        from sampling import apply_top_p
        logits   = self._logits()
        filtered = apply_top_p(logits, 1.0)
        self.assertTrue(torch.equal(logits, filtered))

    def test_sample_token_in_range(self):
        from sampling import sample_token
        V = 1000
        logits = torch.randn(V)
        for _ in range(200):
            tid = sample_token(logits, temperature=0.8, top_k=50, top_p=0.9)
            self.assertGreaterEqual(tid, 0)
            self.assertLess(tid, V)

    def test_sampling_decode_shape(self):
        from sampling import sampling_decode
        model = _make_model()
        ids   = torch.tensor([[1, 2, 3]])
        out   = sampling_decode(model, ids, max_new_tokens=12,
                                temperature=0.9, device="cpu")
        self.assertEqual(out.shape[1], 15)

    def test_sampling_decode_stochastic(self):
        """Two runs with temperature>0 should (almost certainly) differ."""
        from sampling import sampling_decode
        torch.manual_seed(42)
        model = _make_model()
        ids   = torch.tensor([[5, 10, 15]])
        torch.manual_seed(1)
        out1  = sampling_decode(model, ids, max_new_tokens=20,
                                temperature=1.5, device="cpu")
        torch.manual_seed(99)
        out2  = sampling_decode(model, ids, max_new_tokens=20,
                                temperature=1.5, device="cpu")
        # With high temperature, outputs should differ
        self.assertFalse(torch.equal(out1, out2),
                         "High-temperature runs should not be identical")


# ─── Step 31: inference.py ───────────────────────────────────────────────────

class TestInference(unittest.TestCase):

    def _pipe(self):
        from inference import InferencePipeline
        return InferencePipeline(_make_model(), _make_tok(), device="cpu")

    def test_generate_returns_string(self):
        pipe = self._pipe()
        out  = pipe.generate("Hello", max_new_tokens=10)
        self.assertIsInstance(out, str)

    def test_generate_greedy(self):
        from inference import GenerationConfig
        pipe = self._pipe()
        cfg  = GenerationConfig(strategy="greedy", max_new_tokens=15)
        out  = pipe.generate("test", config=cfg)
        self.assertGreater(len(out), 0)

    def test_generate_temperature(self):
        from inference import GenerationConfig
        pipe = self._pipe()
        cfg  = GenerationConfig(strategy="temperature", temperature=1.0, max_new_tokens=15)
        out  = pipe.generate("test", config=cfg)
        self.assertGreater(len(out), 0)

    def test_generate_top_k(self):
        from inference import GenerationConfig
        pipe = self._pipe()
        cfg  = GenerationConfig(strategy="top_k", top_k=10, max_new_tokens=15)
        out  = pipe.generate("test", config=cfg)
        self.assertGreater(len(out), 0)

    def test_generate_top_p(self):
        from inference import GenerationConfig
        pipe = self._pipe()
        cfg  = GenerationConfig(strategy="top_p", top_p=0.9, max_new_tokens=15)
        out  = pipe.generate("test", config=cfg)
        self.assertGreater(len(out), 0)

    def test_stream_yields_strings(self):
        pipe   = self._pipe()
        pieces = list(pipe.stream("Once", max_new_tokens=10))
        self.assertGreater(len(pieces), 0)
        for p in pieces:
            self.assertIsInstance(p, str)

    def test_stop_token(self):
        from inference import GenerationConfig
        pipe = self._pipe()
        # Inject strong bias so model will produce '!' quickly
        pipe.model.proj.bias.data[1] = 100.0   # token 1 → chr(33) = '!'
        cfg = GenerationConfig(
            strategy="greedy", max_new_tokens=100,
            stop_tokens=["!"]
        )
        out = pipe.generate("A", config=cfg)
        # Output should stop at or shortly after '!'
        self.assertLessEqual(len(out), 50)

    def test_batch_generate(self):
        pipe    = self._pipe()
        results = pipe.batch_generate(["A", "B", "C"], max_new_tokens=5)
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertIsInstance(r, str)

    def test_kwargs_forwarding(self):
        pipe = self._pipe()
        out  = pipe.generate("Hi", strategy="greedy", max_new_tokens=8)
        self.assertIsInstance(out, str)

    def test_empty_prompt(self):
        pipe = self._pipe()
        out  = pipe.generate("", max_new_tokens=5)
        self.assertIsInstance(out, str)


# ─── Step 32: model_io.py ────────────────────────────────────────────────────

class TestModelIO(unittest.TestCase):

    def _model(self):
        return _make_model()

    def test_save_and_load_weights_match(self):
        from model_io import save_checkpoint, load_checkpoint, CheckpointMetadata
        m1   = self._model()
        meta = CheckpointMetadata(epoch=3, val_loss=0.77)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "test.pt"
            save_checkpoint(m1, path, metadata=meta)
            m2   = self._model()
            lmeta = load_checkpoint(m2, path, device="cpu")
            # All parameters identical
            for (n1, p1), (n2, p2) in zip(m1.named_parameters(),
                                           m2.named_parameters()):
                self.assertTrue(torch.allclose(p1, p2), f"Mismatch: {n1}")
            self.assertEqual(lmeta.epoch, 3)
            self.assertAlmostEqual(lmeta.val_loss, 0.77, places=4)

    def test_load_nonexistent_raises(self):
        from model_io import load_checkpoint
        m = self._model()
        with self.assertRaises(FileNotFoundError):
            load_checkpoint(m, "/nonexistent/path.pt")

    def test_metadata_roundtrip(self):
        from model_io import CheckpointMetadata
        meta = CheckpointMetadata(
            model_class="TestModel", d_model=128, n_layers=4,
            epoch=10, train_loss=0.23, val_loss=0.31,
            dataset="wikitext-2", notes="unit test"
        )
        d    = meta.to_dict()
        meta2 = CheckpointMetadata.from_dict(d)
        self.assertEqual(meta.epoch,    meta2.epoch)
        self.assertEqual(meta.dataset,  meta2.dataset)
        self.assertEqual(meta.d_model,  meta2.d_model)

    def test_list_checkpoints(self):
        from model_io import save_checkpoint, list_checkpoints, CheckpointMetadata
        m = self._model()
        with tempfile.TemporaryDirectory() as d:
            for ep, vl in [(1, 0.9), (2, 0.7), (3, 0.5)]:
                save_checkpoint(m, Path(d) / f"ck_{ep}.pt",
                                metadata=CheckpointMetadata(epoch=ep, val_loss=vl))
            rows = list_checkpoints(d)
            self.assertEqual(len(rows), 3)
            # Should be sorted by val_loss ascending
            self.assertLess(rows[0]["val_loss"], rows[-1]["val_loss"])

    def test_tokenizer_persisted(self):
        from model_io import save_checkpoint
        m   = self._model()
        tok = _make_tok()
        with tempfile.TemporaryDirectory() as d:
            p = save_checkpoint(m, Path(d) / "with_tok.pt", tokenizer=tok)
            payload = torch.load(p, map_location="cpu", weights_only=False)
            self.assertIn("tokenizer", payload)
            self.assertIn("vocab", payload["tokenizer"])


# ─── Step 33: evaluate.py ────────────────────────────────────────────────────

class TestEvaluate(unittest.TestCase):

    def test_perplexity_positive_finite(self):
        from evaluate import perplexity
        model  = _make_model()
        tok    = _make_tok()
        corpus = "hello world " * 20
        ids    = tok.encode(corpus)
        ppl    = perplexity(model, ids, context_len=16, stride=8, device="cpu")
        self.assertGreater(ppl, 1.0)
        self.assertTrue(math.isfinite(ppl))

    def test_perplexity_short_sequence(self):
        from evaluate import perplexity
        model = _make_model()
        tok   = _make_tok()
        ids   = tok.encode("hi")
        ppl   = perplexity(model, ids, context_len=8, stride=4, device="cpu")
        self.assertGreater(ppl, 1.0)

    def test_speed_benchmark_returns_results(self):
        from evaluate import speed_benchmark
        model   = _make_model()
        results = speed_benchmark(model, vocab_size=64,
                                  batch_sizes=[1], seq_lens=[16],
                                  n_trials=3, device="cpu")
        self.assertEqual(len(results), 1)
        self.assertGreater(results[0].tokens_per_second, 0)

    def test_memory_profile(self):
        from evaluate import memory_profile
        model = _make_model()
        mem   = memory_profile(model, seq_len=16, vocab_size=64, device="cpu")
        self.assertGreater(mem.n_params, 0)
        self.assertGreater(mem.param_mb, 0)

    def test_score_generation(self):
        from evaluate import score_generation
        tok = _make_tok()
        q   = score_generation("hello", "world is good and nice and world", tok)
        self.assertEqual(q.n_tokens, len(tok.encode("world is good and nice and world")))
        self.assertGreaterEqual(q.type_token_ratio, 0)
        self.assertLessEqual(q.type_token_ratio, 1)

    def test_compare_checkpoints(self):
        from evaluate import compare_checkpoints
        ra = {"perplexity": 80.0, "tokens_per_second": 1000.0,
              "peak_mb": 50.0, "type_token_ratio": 0.6, "repetition_score": 0.1}
        rb = {"perplexity": 60.0, "tokens_per_second": 1200.0,
              "peak_mb": 48.0, "type_token_ratio": 0.7, "repetition_score": 0.08}
        table = compare_checkpoints(ra, rb, "v1", "v2")
        self.assertIn("Perplexity", table)
        self.assertIn("v1", table)
        self.assertIn("v2", table)

    def test_run_eval_suite_writes_report(self):
        from evaluate import run_eval_suite
        model  = _make_model()
        tok    = _make_tok()
        corpus = "the quick brown fox " * 30
        with tempfile.TemporaryDirectory() as d:
            results = run_eval_suite(model, tok, corpus,
                                     output_dir=d, label="test",
                                     device="cpu")
            self.assertIn("perplexity", results)
            self.assertIn("tokens_per_second", results)
            reports = list(Path(d).glob("*.json"))
            self.assertEqual(len(reports), 1)
            data = json.loads(reports[0].read_text())
            self.assertEqual(data["label"], "test")


# ─── Integration: run.py end-to-end ──────────────────────────────────────────

class TestRunPy(unittest.TestCase):

    def test_untrained_model_generates(self):
        """run.py with no checkpoint should still produce output."""
        from run import main
        with tempfile.TemporaryDirectory():
            ret = main(["--prompt", "The quick brown fox",
                        "--max_tokens", "20",
                        "--strategy", "greedy"])
            self.assertEqual(ret, 0)

    def test_save_then_load_then_generate(self):
        """Full pipeline: train a tiny step → save → load via run.py."""
        from run import main, TransformerLM, CharTokenizer
        from model_io import save_checkpoint, CheckpointMetadata
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            tok   = CharTokenizer()
            model = TransformerLM(vocab_size=tok.vocab_size)
            meta  = CheckpointMetadata(
                vocab_size=tok.vocab_size, d_model=128, n_heads=4,
                n_layers=2, d_ff=512, max_seq_len=512,
                epoch=1, val_loss=3.5
            )
            ckpt = Path(d) / "model.pt"
            save_checkpoint(model, ckpt, metadata=meta, tokenizer=tok)

            ret = main([
                "--prompt",      "Hello world",
                "--checkpoint",  str(ckpt),
                "--strategy",    "top_p",
                "--temperature", "0.8",
                "--max_tokens",  "15",
            ])
            self.assertEqual(ret, 0)

    def test_batch_mode(self):
        from run import main
        with tempfile.TemporaryDirectory() as d:
            batch_file = Path(d) / "prompts.json"
            batch_file.write_text(json.dumps(["Hello", "World", "Test"]))
            os.chdir(d)
            ret = main(["--batch_file", str(batch_file), "--max_tokens", "10"])
            self.assertEqual(ret, 0)
            self.assertTrue(Path(d, "batch_output.json").exists())

    def test_list_checkpoints_mode(self):
        from run import main, TransformerLM, CharTokenizer
        from model_io import save_checkpoint
        with tempfile.TemporaryDirectory() as d:
            model = TransformerLM()
            save_checkpoint(model, Path(d) / "ck.pt")
            ret = main(["--list_checkpoints", d])
            self.assertEqual(ret, 0)

    def test_eval_mode(self):
        from run import main
        with tempfile.TemporaryDirectory() as d:
            corpus_file = Path(d) / "corpus.txt"
            corpus_file.write_text("the quick brown fox " * 50)
            os.chdir(d)
            ret = main([
                "--eval",
                "--eval_corpus", str(corpus_file),
                "--max_tokens", "5",
            ])
            self.assertEqual(ret, 0)


# ─────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────

if __name__ == "__main__":
    loader  = unittest.TestLoader()
    suite   = unittest.TestSuite()

    test_classes = [
        TestGreedy,
        TestSampling,
        TestInference,
        TestModelIO,
        TestEvaluate,
        TestRunPy,
    ]

    # Allow running a single class: python test_45G.py TestGreedy
    target = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else None
    for cls in test_classes:
        if target is None or cls.__name__ == target:
            suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
