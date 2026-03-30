# ============================================================
# FILE: evaluation/human_eval.py
# Manual rating interface.
#
# Shows you 20 model outputs, collects ratings 1-5,
# computes a quality score, and saves the session.
#
# Run:  python human_eval.py --model finetuned_v1.pt
# ============================================================

import json, os, sys, argparse
import numpy as np
from datetime import datetime


EVAL_PROMPTS = [
    "What is the difference between a list and a tuple in Python?",
    "Explain quantum entanglement simply.",
    "What are three principles of good UI design?",
    "How does a transformer model work?",
    "Give me a recipe for a simple pasta dish.",
    "What is the Socratic method?",
    "How do I improve my sleep quality?",
    "Explain compound interest with an example.",
    "What are the main causes of World War I?",
    "How does GPS work?",
    "Write a short motivational message.",
    "What is the difference between TCP and UDP?",
    "How do vaccines work?",
    "Explain blockchain in 3 sentences.",
    "What is recursion in programming?",
    "Give advice for staying focused while working from home.",
    "What causes inflation?",
    "How does the human immune system fight viruses?",
    "What is the difference between machine learning and deep learning?",
    "Explain the concept of entropy.",
]


class HumanEvaluator:
    """
    Interactive session: rate N model outputs, receive quality score.
    """

    def __init__(self, model_fn, model_name='model', n_prompts=20):
        """
        Args:
            model_fn   (callable): fn(prompt) → response string.
            model_name (str):      Label for this model version.
            n_prompts  (int):      How many outputs to rate (max 20).
        """
        self.model_fn   = model_fn
        self.model_name = model_name
        self.n_prompts  = min(n_prompts, len(EVAL_PROMPTS))
        self.ratings    = []
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    def run(self, save_dir='data/human_eval'):
        """
        Interactive rating session.

        Returns:
            dict: Session results with score.
        """
        os.makedirs(save_dir, exist_ok=True)
        prompts = EVAL_PROMPTS[:self.n_prompts]

        print("\n" + "=" * 65)
        print(f"  HUMAN EVALUATION — {self.model_name}")
        print(f"  Rate {self.n_prompts} responses on a scale of 1 to 5")
        print("  1 = poor   2 = below avg   3 = ok   4 = good   5 = excellent")
        print("  Type 's' to skip any response.")
        print("=" * 65)

        for i, prompt in enumerate(prompts, 1):
            print(f"\n[{i}/{self.n_prompts}] {'─' * 55}")
            print(f"PROMPT: {prompt}\n")

            # Generate response
            response = self.model_fn(prompt)
            print(f"RESPONSE:\n{response}\n")

            while True:
                raw = input("Your rating (1-5, or 's' to skip): ").strip()
                if raw.lower() == 's':
                    break
                if raw.isdigit() and 1 <= int(raw) <= 5:
                    comment = input("Comment (optional, press Enter to skip): ").strip()
                    self.ratings.append({
                        'prompt':   prompt,
                        'response': response,
                        'rating':   int(raw),
                        'comment':  comment,
                    })
                    break
                print("Please enter 1, 2, 3, 4, or 5.")

        return self._finish(save_dir)

    def _finish(self, save_dir):
        if not self.ratings:
            print("[HumanEval] No ratings collected.")
            return {}

        scores  = [r['rating'] for r in self.ratings]
        quality = float(np.mean(scores))
        dist    = {i: scores.count(i) for i in range(1, 6)}

        results = {
            'model':        self.model_name,
            'session_id':   self.session_id,
            'n_rated':      len(self.ratings),
            'quality_score': quality,
            'score_pct':    round((quality - 1) / 4 * 100, 1),  # 0-100 scale
            'distribution': dist,
            'ratings':      self.ratings,
        }

        path = os.path.join(save_dir, f"eval_{self.model_name}_{self.session_id}.json")
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)

        self._print_summary(results, path)
        return results

    def _print_summary(self, results, saved_path):
        print("\n" + "=" * 65)
        print(f"  EVALUATION COMPLETE — {results['model']}")
        print("=" * 65)
        print(f"  Responses rated   : {results['n_rated']}/{self.n_prompts}")
        print(f"  Mean quality score: {results['quality_score']:.2f} / 5.00")
        print(f"  Quality %         : {results['score_pct']}%")
        print("\n  Rating distribution:")
        for star, count in sorted(results['distribution'].items()):
            bar = "█" * count
            print(f"    {'★' * star:<5} ({star}) : {bar} {count}")
        print(f"\n  Results saved → {saved_path}")
        print("=" * 65)


def compare_models(fn_a, fn_b, name_a='model_a', name_b='model_b',
                   n_prompts=10, save_dir='data/human_eval'):
    """
    Side-by-side A/B comparison.

    For each prompt: show both responses, ask which is better.
    Returns preference counts.
    """
    os.makedirs(save_dir, exist_ok=True)
    prompts = EVAL_PROMPTS[:n_prompts]
    prefs   = {'A': 0, 'B': 0, 'tie': 0}
    records = []

    print("\n" + "=" * 65)
    print(f"  A/B COMPARISON: {name_a}  vs  {name_b}")
    print("  Type 'a', 'b', or 't' (tie) for each pair.")
    print("=" * 65)

    for i, prompt in enumerate(prompts, 1):
        resp_a = fn_a(prompt)
        resp_b = fn_b(prompt)
        print(f"\n[{i}/{n_prompts}] PROMPT: {prompt}\n")
        print(f"── Model A ({name_a}) ──\n{resp_a}\n")
        print(f"── Model B ({name_b}) ──\n{resp_b}\n")

        while True:
            raw = input("Better response? (a / b / t): ").strip().lower()
            if raw in ('a', 'b', 't', 'tie'):
                key = 'tie' if raw in ('t', 'tie') else raw.upper()
                prefs[key] += 1
                records.append({
                    'prompt': prompt, 'resp_a': resp_a,
                    'resp_b': resp_b, 'winner': key,
                })
                break
            print("Type 'a', 'b', or 't'.")

    # Summary
    total = sum(prefs.values())
    print(f"\n  {name_a} preferred : {prefs['A']}/{total} ({prefs['A']/total:.0%})")
    print(f"  {name_b} preferred : {prefs['B']}/{total} ({prefs['B']/total:.0%})")
    print(f"  Ties              : {prefs['tie']}/{total}")

    session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(save_dir, f"ab_{name_a}_vs_{name_b}_{session_id}.json")
    with open(path, 'w') as f:
        json.dump({'prefs': prefs, 'records': records,
                   'name_a': name_a, 'name_b': name_b}, f, indent=2)
    print(f"  A/B results saved → {path}")
    return prefs


# ============================================================
# TEST (non-interactive simulation)
# ============================================================

if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'finetune'))
    from pipeline import SimpleTokenizer

    print("=" * 60)
    print("  HUMAN EVAL SELF TEST (simulated ratings)")
    print("=" * 60)

    def dummy_model(prompt):
        return f"Response to: {prompt[:40]}. This is a simulated answer."

    evaluator = HumanEvaluator(dummy_model, model_name='test-v0', n_prompts=3)

    # Simulate by monkey-patching input
    ratings_iter = iter(['4', '', '3', '', '5', ''])
    original_input = __builtins__.__dict__.get('input', input)

    import builtins
    call_count = [0]
    def fake_input(prompt=''):
        val = next(ratings_iter, '3')
        print(f"[simulated input] {val}")
        return val
    builtins.input = fake_input

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        results = evaluator.run(save_dir=tmp)

    builtins.input = original_input

    assert 'quality_score' in results
    print(f"\nSimulated quality score: {results['quality_score']:.2f}")
    print("\n✓ Human eval all checks passed.")
    print("=" * 60)
