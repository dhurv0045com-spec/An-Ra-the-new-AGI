#!/usr/bin/env python3
# ============================================================
# FILE: finetune.py  (master entry point)
# Phase 2 — Intelligence Layer
#
# Commands:
#
#   Fine-tune:
#     python finetune.py --data my_data.json \
#                        --base best_model.pt \
#                        --method lora \
#                        --epochs 3
#
#   Collect feedback:
#     python finetune.py --mode feedback \
#                        --model finetuned_v1.pt
#
#   Evaluate:
#     python finetune.py --mode eval \
#                        --model finetuned_v1.pt \
#                        --benchmark full
#
#   Compare two versions:
#     python finetune.py --mode compare \
#                        --model-a base_model.pt \
#                        --model-b finetuned_v1.pt
#
#   Rollback:
#     python finetune.py --mode rollback --version v2
# ============================================================

import argparse, sys, os, json

# Add sub-packages to path
ROOT = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(ROOT, 'finetune'))
sys.path.insert(0, os.path.join(ROOT, 'alignment'))
sys.path.insert(0, os.path.join(ROOT, 'evaluation'))

from finetune.pipeline         import FineTuner, BaseModelStub, SimpleTokenizer
from finetune.dataset_builder  import DatasetBuilder
from finetune.lora             import LoRAManager
from alignment.feedback        import FeedbackStore, FeedbackLoop
from alignment.constitution    import ConstitutionChecker
from evaluation.eval_suite     import EvalSuite
from evaluation.human_eval     import HumanEvaluator, compare_models
from evaluation.tracker        import VersionTracker


# ── Model loading stub ────────────────────────────────
def load_model(path, config=None):
    """
    Load a model from checkpoint.
    Replace BaseModelStub with real Phase 1 Transformer when available.
    """
    cfg = config or {'vocab_size': 512, 'd_model': 128}
    if path and os.path.exists(str(path) + '.npz'):
        return BaseModelStub.load(path, cfg)
    print(f"[Warning] Checkpoint '{path}' not found — using fresh model stub.")
    return BaseModelStub(cfg)


def load_tokenizer(vocab_size=512):
    """Load tokenizer — replace with Phase 1 BPE tokenizer."""
    return SimpleTokenizer(vocab_size)


# ── Modes ─────────────────────────────────────────────

def mode_finetune(args):
    """Supervised fine-tuning from a data file."""
    print(f"\n[Phase 2] Fine-tuning mode")
    print(f"  Data   : {args.data}")
    print(f"  Base   : {args.base}")
    print(f"  Method : {args.method}")
    print(f"  Epochs : {args.epochs}\n")

    # Load data
    builder = DatasetBuilder()
    builder.load(args.data)
    train, val, test = builder.build()

    # Load model + tokenizer
    model     = load_model(args.base)
    tokenizer = load_tokenizer()

    # Fine-tune
    config = {
        'vocab_size':  512,
        'd_model':     128,
        'method':      args.method,
        'lr':          1e-4,
        'batch_size':  4,
        'max_seq_len': 256,
        'lora_rank':   8,
        'lora_alpha':  16,
        'save_dir':    os.path.join(ROOT, 'checkpoints', 'finetuned'),
    }
    finetuner = FineTuner(model, tokenizer, config)
    train_losses, val_losses = finetuner.train(train, val, epochs=args.epochs)

    # Register version
    tracker   = VersionTracker()
    version_id = tracker.register(
        checkpoint_dir = config['save_dir'],
        method         = args.method,
        train_loss     = train_losses[-1] if train_losses else None,
        val_loss       = val_losses[-1]   if val_losses   else None,
        changelog      = f"Fine-tuned on {args.data}, method={args.method}, epochs={args.epochs}",
    )
    tracker.set_active(version_id)
    print(f"\n[Phase 2] ✓ Fine-tuning complete. Version: {version_id}")


def mode_feedback(args):
    """Interactive feedback session — rate outputs, improve model."""
    print(f"\n[Phase 2] Feedback mode — model: {args.model}\n")

    model     = load_model(args.model)
    tokenizer = load_tokenizer()

    def model_fn(prompt):
        tokens = tokenizer.encode(prompt)
        out    = model.forward(tokens)
        return f"[Response to: {prompt[:50]}]"   # stub

    store = FeedbackStore(path=os.path.join(ROOT, 'data', 'feedback.jsonl'))
    loop  = FeedbackLoop(store, model)

    # Build pairs for rating
    pairs = [
        {'prompt': p, 'response': model_fn(p)}
        for p in [
            "Explain the concept of overfitting.",
            "What is the difference between supervised and unsupervised learning?",
            "How does attention mechanism work?",
        ]
    ]
    loop.rate_batch(pairs)
    store.stats()


def mode_eval(args):
    """Run the full evaluation suite."""
    print(f"\n[Phase 2] Evaluation mode — model: {args.model}\n")

    model     = load_model(args.model)
    tokenizer = load_tokenizer()

    # Load test data if available
    test_path = os.path.join(ROOT, 'data', 'examples', 'sample_training_data.json')
    test_data = []
    if os.path.exists(test_path):
        with open(test_path) as f:
            test_data = json.load(f)

    suite   = EvalSuite(model, tokenizer, name=str(args.model))
    results = suite.run_full(test_examples=test_data)
    suite.print_report(results)

    # Update tracker
    tracker = VersionTracker()
    active  = tracker.get_active()
    if active:
        tracker.update_quality(active['version_id'], results['overall_score'])
        tracker.alert_regression(active['version_id'])

    save_path = os.path.join(ROOT, 'data', f"eval_results_{args.model}.json")
    suite.save_results(results, save_path)


def mode_compare(args):
    """A/B comparison between two model versions."""
    print(f"\n[Phase 2] Compare mode")
    print(f"  Model A: {args.model_a}")
    print(f"  Model B: {args.model_b}\n")

    model_a = load_model(args.model_a)
    model_b = load_model(args.model_b)

    def fn_a(prompt): return f"[A] Response to: {prompt[:60]}"
    def fn_b(prompt): return f"[B] Response to: {prompt[:60]}"

    compare_models(fn_a, fn_b,
                   name_a=str(args.model_a),
                   name_b=str(args.model_b),
                   n_prompts=5,
                   save_dir=os.path.join(ROOT, 'data', 'human_eval'))


def mode_rollback(args):
    """Roll back to a previous model version."""
    tracker = VersionTracker()
    tracker.changelog()
    tracker.rollback(args.version)


# ── CLI ───────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Phase 2 — Intelligence Layer')
    parser.add_argument('--mode',    default='finetune',
                        choices=['finetune', 'feedback', 'eval', 'compare', 'rollback'])
    parser.add_argument('--data',    default='data/examples/sample_training_data.json')
    parser.add_argument('--base',    default=None,        help='Base model checkpoint path')
    parser.add_argument('--model',   default='model',     help='Model checkpoint for eval/feedback')
    parser.add_argument('--model-a', default='base',      dest='model_a')
    parser.add_argument('--model-b', default='finetuned', dest='model_b')
    parser.add_argument('--method',  default='lora',      choices=['lora', 'full'])
    parser.add_argument('--epochs',  default=3,           type=int)
    parser.add_argument('--benchmark', default='full',    choices=['full', 'quick'])
    parser.add_argument('--version', default=None,        help='Version ID for rollback')

    args = parser.parse_args()

    dispatch = {
        'finetune': mode_finetune,
        'feedback': mode_feedback,
        'eval':     mode_eval,
        'compare':  mode_compare,
        'rollback': mode_rollback,
    }
    dispatch[args.mode](args)


if __name__ == '__main__':
    main()
