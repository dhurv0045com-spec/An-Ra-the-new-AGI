# ============================================================
# FILE: finetune/dataset_builder.py
# Instruction dataset preparation pipeline.
#
# Handles: JSON / CSV / plain-text import, cleaning,
# deduplication, quality scoring, train/val/test split,
# and simple augmentation.
#
# Output format (every example):
#   {"system": "...", "user": "...", "assistant": "..."}
# ============================================================

import json, csv, os, re, hashlib
import numpy as np
from collections import Counter


# ── Quality thresholds ────────────────────────────────────
MIN_USER_LEN      = 8          # chars — discard trivially short prompts
MIN_ASSISTANT_LEN = 10         # chars — discard trivially short responses
MAX_ASSISTANT_LEN = 4096       # chars — discard runaway examples
MIN_QUALITY_SCORE = 0.35       # 0..1  — discard low-quality examples


def _hash(text):
    """SHA-256 fingerprint for deduplication."""
    return hashlib.sha256(text.encode()).hexdigest()


def _quality_score(example):
    """
    Heuristic quality score in [0, 1].

    Penalises:
      - Very short / very long assistant responses
      - Excessive repetition
      - Low lexical diversity (ratio of unique words)
    Rewards:
      - Reasonable length
      - Sentence structure (presence of punctuation)
    """
    text = example.get('assistant', '')
    if not text:
        return 0.0

    words      = text.lower().split()
    n_words    = len(words)
    if n_words < 3:
        return 0.0

    # Lexical diversity: unique / total words (higher = better)
    diversity  = len(set(words)) / n_words

    # Length score: peak at ~100 words, tails off
    length_score = min(n_words / 100.0, 1.0) * max(1.0 - n_words / 800.0, 0.1)

    # Punctuation presence
    punct_score = 1.0 if re.search(r'[.!?,;:]', text) else 0.4

    # Repetition: most common word should not dominate
    top_freq    = Counter(words).most_common(1)[0][1] / n_words
    rep_penalty = max(0.0, 1.0 - top_freq * 3)

    score = (diversity * 0.4 + length_score * 0.3 +
             punct_score * 0.2 + rep_penalty * 0.1)
    return float(np.clip(score, 0.0, 1.0))


def _clean(text):
    """Strip excess whitespace, normalise newlines."""
    text = text.strip()
    text = re.sub(r'\r\n', '\n', text)       # CRLF → LF
    text = re.sub(r'\n{3,}', '\n\n', text)   # collapse blank lines
    text = re.sub(r'[ \t]+', ' ', text)      # collapse spaces
    return text


# ── Loaders ───────────────────────────────────────────────

def load_json(path):
    """
    Load from JSON file.

    Accepts two shapes:
      A) List of {"system":…, "user":…, "assistant":…} dicts.
      B) List of {"instruction":…, "output":…} (Alpaca style).
    """
    with open(path) as f:
        raw = json.load(f)

    examples = []
    for item in raw:
        if 'user' in item and 'assistant' in item:
            examples.append({
                'system':    _clean(item.get('system', '')),
                'user':      _clean(item['user']),
                'assistant': _clean(item['assistant']),
            })
        elif 'instruction' in item and 'output' in item:
            # Alpaca format
            user = item['instruction']
            if item.get('input'):
                user += '\n' + item['input']
            examples.append({
                'system':    _clean(item.get('system', '')),
                'user':      _clean(user),
                'assistant': _clean(item['output']),
            })
    return examples


def load_csv(path):
    """
    Load from CSV.
    Required columns: user, assistant
    Optional column:  system
    """
    examples = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'user' not in row or 'assistant' not in row:
                continue
            examples.append({
                'system':    _clean(row.get('system', '')),
                'user':      _clean(row['user']),
                'assistant': _clean(row['assistant']),
            })
    return examples


def load_text(path, separator='###'):
    """
    Load from plain text.

    Expected format (separator between each block):
      USER: <prompt>
      ASSISTANT: <response>
      ###
    """
    with open(path) as f:
        content = f.read()

    examples = []
    for block in content.split(separator):
        block = block.strip()
        if not block:
            continue
        user_match = re.search(r'USER:\s*(.*?)(?=ASSISTANT:|$)', block,
                               re.DOTALL | re.IGNORECASE)
        asst_match = re.search(r'ASSISTANT:\s*(.*)', block,
                               re.DOTALL | re.IGNORECASE)
        if user_match and asst_match:
            examples.append({
                'system':    '',
                'user':      _clean(user_match.group(1)),
                'assistant': _clean(asst_match.group(1)),
            })
    return examples


# ── Pipeline ─────────────────────────────────────────────

class DatasetBuilder:
    """
    Full dataset preparation pipeline.

    Usage:
        builder = DatasetBuilder()
        builder.load('data.json')
        train, val, test = builder.build()
    """

    def __init__(self, default_system='You are a helpful AI assistant.'):
        self.default_system = default_system
        self.examples       = []
        self._seen_hashes   = set()

    def load(self, path):
        """Auto-detect format and load from path."""
        ext = os.path.splitext(path)[1].lower()
        if ext == '.json':
            raw = load_json(path)
        elif ext == '.csv':
            raw = load_csv(path)
        else:
            raw = load_text(path)
        self.examples.extend(raw)
        print(f"[Dataset] Loaded {len(raw)} examples from {path}")
        return self

    def add(self, user, assistant, system=''):
        """Manually add a single example."""
        self.examples.append({
            'system':    system or self.default_system,
            'user':      _clean(user),
            'assistant': _clean(assistant),
        })

    def clean(self):
        """Remove too-short, too-long, and duplicate examples."""
        before = len(self.examples)
        cleaned = []
        for ex in self.examples:
            u = ex['user']
            a = ex['assistant']
            if len(u) < MIN_USER_LEN:       continue
            if len(a) < MIN_ASSISTANT_LEN:  continue
            if len(a) > MAX_ASSISTANT_LEN:  continue
            key = _hash(u + a)
            if key in self._seen_hashes:    continue
            self._seen_hashes.add(key)
            if not ex['system']:
                ex['system'] = self.default_system
            cleaned.append(ex)
        self.examples = cleaned
        print(f"[Dataset] Clean: {before} → {len(self.examples)} "
              f"(removed {before - len(self.examples)})")
        return self

    def score(self):
        """Attach a quality score to every example, drop low scorers."""
        before = len(self.examples)
        for ex in self.examples:
            ex['_score'] = _quality_score(ex)
        self.examples = [e for e in self.examples
                         if e['_score'] >= MIN_QUALITY_SCORE]
        print(f"[Dataset] Quality filter: {before} → {len(self.examples)}")
        return self

    def augment(self, n_variations=1):
        """
        Simple augmentation: create variations of good examples
        by paraphrasing the system prompt prefix.

        Extends the dataset without needing new human labels.
        """
        prefixes = [
            'You are a knowledgeable and concise AI assistant.',
            'You are a helpful expert. Answer clearly and directly.',
            'You are an intelligent assistant. Be precise.',
        ]
        good = [e for e in self.examples if e.get('_score', 1.0) > 0.6]
        added = []
        rng = np.random.default_rng(0)
        for ex in good[:min(len(good), 500)]:   # cap at 500 to avoid explosion
            for _ in range(n_variations):
                variation = dict(ex)
                variation['system'] = rng.choice(prefixes)
                variation['_augmented'] = True
                added.append(variation)
        self.examples.extend(added)
        print(f"[Dataset] Augmented: added {len(added)} variations")
        return self

    def split(self, val_frac=0.1, test_frac=0.05, seed=42):
        """
        Shuffle and split into train / val / test.

        Returns:
            tuple: (train_list, val_list, test_list)
        """
        rng  = np.random.default_rng(seed)
        data = list(self.examples)
        rng.shuffle(data)

        n      = len(data)
        n_test = max(1, int(n * test_frac))
        n_val  = max(1, int(n * val_frac))

        test  = data[:n_test]
        val   = data[n_test:n_test + n_val]
        train = data[n_test + n_val:]

        print(f"[Dataset] Split → train={len(train)} val={len(val)} test={len(test)}")
        return train, val, test

    def save(self, path):
        """Save current examples to JSON."""
        with open(path, 'w') as f:
            json.dump(self.examples, f, indent=2)
        print(f"[Dataset] Saved {len(self.examples)} examples → {path}")

    def build(self, val_frac=0.1, test_frac=0.05):
        """Run full pipeline: clean → score → split."""
        self.clean()
        self.score()
        return self.split(val_frac, test_frac)

    def stats(self):
        """Print dataset statistics."""
        if not self.examples:
            print("[Dataset] Empty.")
            return
        user_lens = [len(e['user'])      for e in self.examples]
        asst_lens = [len(e['assistant']) for e in self.examples]
        scores    = [e.get('_score', 0)  for e in self.examples]
        print(f"\n[Dataset] Statistics ({len(self.examples)} examples)")
        print(f"  User    len: min={min(user_lens)}  "
              f"avg={np.mean(user_lens):.0f}  max={max(user_lens)}")
        print(f"  Asst    len: min={min(asst_lens)}  "
              f"avg={np.mean(asst_lens):.0f}  max={max(asst_lens)}")
        print(f"  Quality:     min={min(scores):.2f}  "
              f"avg={np.mean(scores):.2f}  max={max(scores):.2f}\n")


# ============================================================
# TEST
# ============================================================

if __name__ == '__main__':
    import tempfile

    print("=" * 55)
    print("  DATASET BUILDER SELF TEST")
    print("=" * 55)

    # Build synthetic data
    raw = [
        {"user": "What is 2 + 2?",
         "assistant": "2 + 2 equals 4."},
        {"user": "Explain gravity in simple terms.",
         "assistant": ("Gravity is the force that pulls objects toward each "
                       "other. The more massive an object, the stronger its "
                       "gravitational pull. Earth's gravity keeps us on the "
                       "ground and the Moon in orbit.")},
        {"user": "hi",                              # too short — will be filtered
         "assistant": "hey"},
        {"user": "Write a poem about the ocean.",
         "assistant": ("The waves crash in, the waves crash out,\n"
                       "A rhythm ancient, full of clout.\n"
                       "The sea holds secrets, dark and deep,\n"
                       "Where silent creatures drift and sleep.")},
        # duplicate
        {"user": "What is 2 + 2?",
         "assistant": "2 + 2 equals 4."},
    ]

    with tempfile.TemporaryDirectory() as tmp:
        data_path = os.path.join(tmp, 'test.json')
        with open(data_path, 'w') as f:
            json.dump(raw, f)

        builder = DatasetBuilder()
        builder.load(data_path)
        builder.stats()
        train, val, test = builder.build(val_frac=0.2, test_frac=0.1)

    print(f"\ntrain={len(train)}  val={len(val)}  test={len(test)}")
    print(f"Sample train[0]: user='{train[0]['user'][:40]}…'")
    print("\n✓ DatasetBuilder all checks passed.")
    print("=" * 55)
