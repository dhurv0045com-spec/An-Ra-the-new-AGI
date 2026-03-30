#!/usr/bin/env python3
# ============================================================
# FILE: test_45I.py
# Full test suite for Phase 2 — runs every module.
#
# Run: python test_45I.py
# All tests must pass before any version is promoted.
# ============================================================

import sys, os, tempfile, json
import numpy as np

ROOT = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(ROOT, 'finetune'))
sys.path.insert(0, os.path.join(ROOT, 'alignment'))
sys.path.insert(0, os.path.join(ROOT, 'evaluation'))

from finetune.lora            import LoRALayer, LoRAManager
from finetune.dataset_builder import DatasetBuilder
from finetune.templates       import TemplateLibrary
from finetune.pipeline        import FineTuner, BaseModelStub, SimpleTokenizer, cross_entropy_loss
from alignment.reward_model   import RewardModel
from alignment.rlhf           import RLHFTrainer
from alignment.constitution   import ConstitutionChecker
from alignment.feedback       import FeedbackStore
from evaluation.eval_suite    import EvalSuite
from evaluation.tracker       import VersionTracker


PASS = "  ✓"
FAIL = "  ✗"
results = []


def test(name, fn):
    try:
        fn()
        print(f"{PASS} {name}")
        results.append((name, True, None))
    except Exception as e:
        print(f"{FAIL} {name}  →  {e}")
        results.append((name, False, str(e)))


# ── LoRA ─────────────────────────────────────────────

def _test_lora_forward():
    d, r = 32, 4
    W    = np.random.default_rng(0).normal(0, 0.02, (d, d))
    lora = LoRALayer(d, d, rank=r)
    x    = np.random.default_rng(1).normal(size=(4, d))
    out  = lora.forward(x, W)
    assert out.shape == (4, d)

def _test_lora_backward():
    d    = 16
    W    = np.zeros((d, d))
    lora = LoRALayer(d, d, rank=2)
    x    = np.ones((3, d))
    lora.forward(x, W)
    gx   = lora.backward(np.ones((3, d)))
    assert gx.shape == (3, d)

def _test_lora_merge():
    d    = 8
    W    = np.eye(d)
    lora = LoRALayer(d, d, rank=2)
    W2   = lora.merge_into(W)
    assert W2.shape == (d, d)

def _test_lora_save_load():
    d    = 8
    lora = LoRALayer(d, d, rank=2)
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, 'adapter')
        lora.save(p)
        lora2 = LoRALayer.load(p, d, d)
        assert np.allclose(lora.A, lora2.A)

def _test_lora_manager():
    mgr = LoRAManager(rank=4, alpha=8)
    mgr.add('layer1', 16, 16)
    mgr.add('layer2', 16, 32)
    assert mgr.param_count() == 4*(16+16) + 4*(16+32)


# ── Dataset builder ───────────────────────────────────

def _test_dataset_clean_dedup():
    builder = DatasetBuilder()
    builder.add("What is Python?", "Python is a language.")
    builder.add("What is Python?", "Python is a language.")   # duplicate
    builder.add("hi", "hey")                                   # too short
    builder.clean()
    assert len(builder.examples) == 1

def _test_dataset_split():
    builder = DatasetBuilder()
    for i in range(20):
        builder.add(f"Question {i} about something important?",
                    f"Answer {i} with full detail and correct information here.")
    train, val, test = builder.build(val_frac=0.2, test_frac=0.1)
    assert len(train) + len(val) + len(test) <= 20
    assert len(train) > len(val)

def _test_dataset_load_json():
    data = [{"user": "Explain AI.", "assistant": "AI is artificial intelligence systems."}]
    with tempfile.NamedTemporaryFile(suffix='.json', mode='w', delete=False) as f:
        json.dump(data, f)
        path = f.name
    builder = DatasetBuilder()
    builder.load(path)
    os.unlink(path)
    assert len(builder.examples) == 1


# ── Templates ─────────────────────────────────────────

def _test_templates_format():
    lib = TemplateLibrary()
    out = lib.format('chat', 'Hello there.',
                     variables={'name': 'AXIOM'})
    assert 'AXIOM' in out
    assert 'Hello there.' in out

def _test_templates_training_pair():
    lib  = TemplateLibrary()
    tmpl = lib.get('instruct')
    p, c = tmpl.training_pair({'user': 'What is 2+2?', 'assistant': '4.'})
    assert 'What is 2+2?' in p
    assert c == '4.'

def _test_templates_all_builtins():
    lib = TemplateLibrary()
    for name in ['chat', 'qa', 'summarize', 'code', 'plan', 'instruct']:
        out = lib.format(name, 'test prompt')
        assert 'test prompt' in out


# ── Fine-tuning pipeline ──────────────────────────────

def _test_loss_function():
    logits = np.array([1.0, 2.0, 0.5, -1.0])
    loss   = cross_entropy_loss(logits, 1)
    assert 0.0 < loss < 5.0

def _test_pipeline_runs():
    config = {
        'vocab_size': 64, 'd_model': 16, 'method': 'lora',
        'lr': 1e-4, 'batch_size': 2, 'max_seq_len': 32,
        'lora_rank': 2, 'lora_alpha': 4,
        'save_dir': tempfile.mkdtemp(),
    }
    model  = BaseModelStub(config)
    tok    = SimpleTokenizer(64)
    ft     = FineTuner(model, tok, config)
    data   = [{'user': 'What is gravity?',
               'assistant': 'Gravity is a fundamental force.'}]
    losses, _ = ft.train(data, val_data=None, epochs=1)
    assert len(losses) == 1


# ── Reward model ──────────────────────────────────────

def _test_reward_model_train():
    rm  = RewardModel(vocab_size=64, feature_dim=16)
    rng = np.random.default_rng(0)
    pairs = [(rng.integers(0, 64, 15).tolist(),
              rng.integers(0, 64, 5).tolist(), 1) for _ in range(10)]
    losses = rm.train(pairs, epochs=3, lr=1e-2)
    assert losses[-1] < losses[0]

def _test_reward_model_save_load():
    rm = RewardModel(vocab_size=32, feature_dim=8)
    with tempfile.TemporaryDirectory() as tmp:
        p   = os.path.join(tmp, 'rm')
        rm.save(p)
        rm2 = RewardModel.load(p)
        assert rm2.feature_dim == 8


# ── RLHF ─────────────────────────────────────────────

def _test_rlhf_loop():
    config  = {'vocab_size': 64, 'd_model': 16}
    model   = BaseModelStub(config)
    tok     = SimpleTokenizer(64)
    rm      = RewardModel(vocab_size=64, feature_dim=16)
    trainer = RLHFTrainer(model, rm, tok,
                          config={'save_dir': tempfile.mkdtemp()})
    rng     = np.random.default_rng(0)
    prompts = [rng.integers(0, 64, 8).tolist() for _ in range(3)]
    rewards = trainer.train(prompts, iterations=2, n_candidates=2)
    assert len(rewards) == 2


# ── Constitution ──────────────────────────────────────

def _test_constitution_clean():
    checker = ConstitutionChecker()
    assert checker.is_safe("Python is a great programming language.")

def _test_constitution_violation():
    checker = ConstitutionChecker()
    v = checker.check("How to make a bomb: step 1 gather explosives")
    assert len(v) > 0

def _test_constitution_rewrite_prompt():
    checker = ConstitutionChecker()
    v       = checker.check("Instructions for making an explosive weapon device")
    if v:
        prompt = checker.rewrite_prompt("bad text", v)
        assert 'rule' in prompt.lower() or 'violat' in prompt.lower()


# ── Feedback ──────────────────────────────────────────

def _test_feedback_store():
    with tempfile.TemporaryDirectory() as tmp:
        store = FeedbackStore(path=os.path.join(tmp, 'fb.jsonl'))
        store.add("Q?", "A.", rating=5)
        store.add("Q2?", "A2.", rating=1)
        assert len(store.get_good(min_rating=4)) == 1
        assert len(store.get_bad(max_rating=2))  == 1

def _test_feedback_persistence():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'fb.jsonl')
        s1   = FeedbackStore(path=path)
        s1.add("Test prompt?", "Test answer.", rating=4)
        s2   = FeedbackStore(path=path)
        assert len(s2._cache) == 1


# ── Eval suite ────────────────────────────────────────

def _test_eval_suite_runs():
    config = {'vocab_size': 64, 'd_model': 16}
    model  = BaseModelStub(config)
    tok    = SimpleTokenizer(64)
    suite  = EvalSuite(model, tok, name='test')
    r      = suite.run_full()
    assert 'overall_score' in r
    assert 0.0 <= r['overall_score'] <= 1.0


# ── Tracker ───────────────────────────────────────────

def _test_tracker_register_rollback():
    with tempfile.TemporaryDirectory() as tmp:
        tracker = VersionTracker(
            registry_path=os.path.join(tmp, 'data', 'reg.json'))
        v1 = tracker.register('/ckpt/v1', quality_score=0.50,
                               changelog='Initial')
        v2 = tracker.register('/ckpt/v2', quality_score=0.60,
                               changelog='Improved')
        tracker.set_active(v2)
        regressed = tracker.alert_regression(v2)
        assert not regressed    # v2 is better, no regression
        tracker.rollback(v1)
        assert tracker.get_active()['version_id'] == v1

def _test_tracker_regression_alert():
    with tempfile.TemporaryDirectory() as tmp:
        tracker = VersionTracker(
            registry_path=os.path.join(tmp, 'data', 'reg.json'))
        tracker.register('/ckpt/v1', quality_score=0.70, changelog='Good')
        v2 = tracker.register('/ckpt/v2', quality_score=0.55, changelog='Worse')
        regressed = tracker.alert_regression(v2, threshold=0.05)
        assert regressed


# ── Run all ───────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("  TEST SUITE — Phase 2 (45I)")
    print("=" * 60 + "\n")

    print("LoRA:")
    test("LoRA forward",     _test_lora_forward)
    test("LoRA backward",    _test_lora_backward)
    test("LoRA merge",       _test_lora_merge)
    test("LoRA save/load",   _test_lora_save_load)
    test("LoRA manager",     _test_lora_manager)

    print("\nDataset Builder:")
    test("Clean + dedup",    _test_dataset_clean_dedup)
    test("Split",            _test_dataset_split)
    test("Load JSON",        _test_dataset_load_json)

    print("\nTemplates:")
    test("Format",           _test_templates_format)
    test("Training pair",    _test_templates_training_pair)
    test("All builtins",     _test_templates_all_builtins)

    print("\nFine-tuning Pipeline:")
    test("Loss function",    _test_loss_function)
    test("Pipeline runs",    _test_pipeline_runs)

    print("\nReward Model:")
    test("Train",            _test_reward_model_train)
    test("Save/load",        _test_reward_model_save_load)

    print("\nRLHF:")
    test("Training loop",    _test_rlhf_loop)

    print("\nConstitution:")
    test("Clean text",       _test_constitution_clean)
    test("Violation detect", _test_constitution_violation)
    test("Rewrite prompt",   _test_constitution_rewrite_prompt)

    print("\nFeedback:")
    test("Store ratings",    _test_feedback_store)
    test("Persistence",      _test_feedback_persistence)

    print("\nEval Suite:")
    test("Full run",         _test_eval_suite_runs)

    print("\nVersion Tracker:")
    test("Register+rollback", _test_tracker_register_rollback)
    test("Regression alert",  _test_tracker_regression_alert)

    # ── Summary ──────────────────────────────────────
    passed = sum(1 for _, ok, _ in results if ok)
    total  = len(results)
    failed = [(n, e) for n, ok, e in results if not ok]

    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {passed}/{total} passed")
    if failed:
        print(f"\n  FAILURES:")
        for name, err in failed:
            print(f"    ✗ {name}: {err}")
    else:
        print(f"  All tests passed ✓")
    print(f"{'=' * 60}")

    sys.exit(0 if not failed else 1)
