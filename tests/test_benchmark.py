import pytest


def _tiny_model():
    pytest.importorskip("torch")
    from anra_brain import CausalTransformerV2

    return CausalTransformerV2(
        vocab_size=4096,
        n_embd=64,
        n_head=2,
        n_layer=2,
        block_size=64,
        n_kv_head=2,
        mod_layers=set(),
    )


class _FakeTok:
    special_ids = {"<eos>": 3}

    def encode(self, text):
        return [1, 2, 3, 4, 5][: len(text.split()) + 1]

    def decode(self, ids):
        return " ".join(str(i) for i in ids)


def test_suite_val_perplexity_runs():
    from training.benchmark import BenchmarkSuite

    model = _tiny_model()
    suite = BenchmarkSuite(model, _FakeTok(), holdout_texts=["hello world"] * 5)
    ppl = suite.val_perplexity()
    assert isinstance(ppl, float) and ppl > 0


def test_suite_civ_no_guard():
    from training.benchmark import BenchmarkSuite

    model = _tiny_model()
    suite = BenchmarkSuite(model, _FakeTok())
    assert suite.civ_similarity() == 1.0


def test_suite_run_all_returns_result():
    from training.benchmark import BenchmarkResult, BenchmarkSuite

    model = _tiny_model()
    suite = BenchmarkSuite(model, _FakeTok(), holdout_texts=["x"] * 3)
    result = suite.run_all()
    assert isinstance(result, BenchmarkResult)
    assert hasattr(result, "val_perplexity")
    assert hasattr(result, "civ_score")


def test_original_run_benchmark_unchanged():
    from training.benchmark import BenchmarkResult, run_benchmark

    result = run_benchmark(val_loss=2.5, rlvr_rewards=[1.0, 0.0, 1.0])
    assert isinstance(result, BenchmarkResult)
    assert result.val_perplexity > 0
