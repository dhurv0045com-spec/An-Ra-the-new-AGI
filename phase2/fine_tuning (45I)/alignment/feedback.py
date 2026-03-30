# ============================================================
# FILE: alignment/feedback.py
# Feedback collection, storage, and retraining trigger.
#
# Every interaction can be rated. Good outputs become training
# data. Bad outputs are flagged. When enough feedback
# accumulates, retraining is triggered automatically.
# ============================================================

import json, os, uuid
from datetime import datetime
from collections import defaultdict


FEEDBACK_STORE_PATH = 'data/feedback.jsonl'
RETRAIN_THRESHOLD   = 50    # trigger retraining after N new rated examples
MIN_GOOD_RATING     = 4     # rating >= this → add to training data (scale 1-5)
MAX_BAD_RATING      = 2     # rating <= this → flag for review


class FeedbackStore:
    """
    Persistent store for interaction ratings.

    Storage format: one JSON object per line (JSONL).
    Each record:
      {
        "id":           UUID,
        "timestamp":    ISO string,
        "session_id":   str,
        "prompt":       str,
        "response":     str,
        "rating":       int (1-5),
        "comment":      str,
        "used_training": bool
      }
    """

    def __init__(self, path=FEEDBACK_STORE_PATH):
        self.path    = path
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        self._cache  = []           # in-memory copy
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            return
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self._cache.append(json.loads(line))

    def add(self, prompt, response, rating, session_id='',
            comment=''):
        """
        Record one rated interaction.

        Args:
            prompt     (str): The user's input.
            response   (str): The model's output.
            rating     (int): 1 (terrible) to 5 (excellent).
            session_id (str): Current session UUID.
            comment    (str): Optional free-text note.

        Returns:
            dict: The stored record.
        """
        record = {
            'id':            str(uuid.uuid4()),
            'timestamp':     datetime.now().isoformat(),
            'session_id':    session_id,
            'prompt':        prompt,
            'response':      response,
            'rating':        int(rating),
            'comment':       comment,
            'used_training': False,
        }
        self._cache.append(record)
        with open(self.path, 'a') as f:
            f.write(json.dumps(record) + '\n')
        return record

    def get_good(self, min_rating=MIN_GOOD_RATING, unused_only=True):
        """
        Return high-rated examples suitable for training data.

        Args:
            min_rating  (int):  Minimum rating to include.
            unused_only (bool): Only return examples not yet used for training.
        """
        return [
            r for r in self._cache
            if r['rating'] >= min_rating
            and (not unused_only or not r['used_training'])
        ]

    def get_bad(self, max_rating=MAX_BAD_RATING):
        """Return low-rated examples flagged for review."""
        return [r for r in self._cache if r['rating'] <= max_rating]

    def mark_used(self, record_ids):
        """Mark records as incorporated into training."""
        id_set = set(record_ids)
        for r in self._cache:
            if r['id'] in id_set:
                r['used_training'] = True
        # Rewrite the file
        with open(self.path, 'w') as f:
            for r in self._cache:
                f.write(json.dumps(r) + '\n')

    def should_retrain(self, threshold=RETRAIN_THRESHOLD):
        """True if enough new rated examples have accumulated."""
        unused_good = self.get_good(unused_only=True)
        return len(unused_good) >= threshold

    def to_training_examples(self, min_rating=MIN_GOOD_RATING):
        """
        Convert good feedback into instruction fine-tuning examples.

        Returns:
            list[dict]: [{"user": …, "assistant": …, "source": "feedback"}, …]
        """
        examples = []
        for r in self.get_good(min_rating):
            examples.append({
                'user':      r['prompt'],
                'assistant': r['response'],
                'system':    '',
                'source':    'feedback',
                '_rating':   r['rating'],
            })
        return examples

    def stats(self):
        if not self._cache:
            print("[Feedback] No records yet.")
            return

        ratings = [r['rating'] for r in self._cache]
        by_r    = defaultdict(int)
        for rt in ratings:
            by_r[rt] += 1

        print(f"\n[Feedback] Total records : {len(self._cache)}")
        print(f"  Rating distribution  : "
              f"{dict(sorted(by_r.items()))}")
        avg = sum(ratings) / len(ratings)
        print(f"  Average rating       : {avg:.2f}")
        print(f"  Unused good examples : {len(self.get_good(unused_only=True))}")
        print(f"  Bad (flagged)        : {len(self.get_bad())}")
        print(f"  Retrain threshold    : {RETRAIN_THRESHOLD}  "
              f"{'⚡ RETRAIN NOW' if self.should_retrain() else '(not yet)'}\n")

    def preference_pairs(self):
        """
        Build (A, B, label) pairs for reward model training.
        Pairs: good response vs bad response for similar prompts.
        Simplified: any 4-5 star response vs any 1-2 star response.
        """
        good = self.get_good(min_rating=4, unused_only=False)
        bad  = self.get_bad(max_rating=2)
        pairs = []
        for g, b in zip(good, bad):
            pairs.append({
                'prompt_a':    g['prompt'],
                'response_a':  g['response'],
                'prompt_b':    b['prompt'],
                'response_b':  b['response'],
                'label':       1,  # A is better
            })
        return pairs


class FeedbackLoop:
    """
    Interactive feedback session.

    Shows the user model outputs and collects ratings.
    Optionally triggers retraining when threshold is hit.
    """

    def __init__(self, store, model=None, retrain_fn=None):
        """
        Args:
            store      (FeedbackStore): Where to persist ratings.
            model      : Model for generating responses (optional).
            retrain_fn (callable):      fn(examples) → triggers retraining.
        """
        self.store      = store
        self.model      = model
        self.retrain_fn = retrain_fn

    def rate_batch(self, pairs):
        """
        Present (prompt, response) pairs and collect ratings.

        Args:
            pairs (list[dict]): [{"prompt": …, "response": …}, …]

        Returns:
            list[dict]: Stored records.
        """
        records = []
        for i, pair in enumerate(pairs, 1):
            print(f"\n[{i}/{len(pairs)}] ─────────────────────────────────")
            print(f"  Prompt  : {pair['prompt'][:120]}")
            print(f"  Response: {pair['response'][:200]}")
            print()

            while True:
                raw = input("  Rate 1-5 (or 's' to skip): ").strip()
                if raw.lower() == 's':
                    break
                if raw.isdigit() and 1 <= int(raw) <= 5:
                    comment = input("  Comment (optional): ").strip()
                    record = self.store.add(
                        pair['prompt'], pair['response'], int(raw),
                        session_id=pair.get('session_id', ''),
                        comment=comment,
                    )
                    records.append(record)
                    break
                print("  Enter a number 1-5.")

        print(f"\n[Feedback] Recorded {len(records)} ratings.")
        self._maybe_retrain()
        return records

    def _maybe_retrain(self):
        """Check threshold and trigger retraining if needed."""
        if self.store.should_retrain() and self.retrain_fn:
            print("[Feedback] ⚡ Retrain threshold reached — starting retraining…")
            examples = self.store.to_training_examples()
            self.retrain_fn(examples)
            self.store.mark_used([e.get('id') for e in examples if 'id' in e])


# ============================================================
# TEST (non-interactive)
# ============================================================

if __name__ == '__main__':
    import tempfile

    print("=" * 55)
    print("  FEEDBACK STORE SELF TEST")
    print("=" * 55)

    with tempfile.TemporaryDirectory() as tmp:
        store = FeedbackStore(path=os.path.join(tmp, 'feedback.jsonl'))

        # Simulate ratings
        samples = [
            ("What is Python?", "Python is a high-level programming language.", 5),
            ("Explain AI.",     "AI stands for artificial intelligence.",       4),
            ("Tell me a joke.", "Why did the chicken cross the road?",          2),
            ("What is 2+2?",   "I don't know.",                                1),
        ]
        for prompt, response, rating in samples:
            store.add(prompt, response, rating, session_id='test-session')

        store.stats()

        good = store.get_good()
        bad  = store.get_bad()
        assert len(good) == 2, f"Expected 2 good, got {len(good)}"
        assert len(bad)  == 2, f"Expected 2 bad,  got {len(bad)}"

        training_data = store.to_training_examples()
        assert len(training_data) == 2
        print(f"Training examples built: {len(training_data)}")

        pairs = store.preference_pairs()
        print(f"Preference pairs built:  {len(pairs)}")

        # Test persistence: reload from disk
        store2 = FeedbackStore(path=os.path.join(tmp, 'feedback.jsonl'))
        assert len(store2._cache) == 4, "Persistence failed"
        print("Persistence: ✓")

    print("\n✓ Feedback store all checks passed.")
    print("=" * 55)
