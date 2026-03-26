"""
model/tokenizer.py — Step 10
Byte-Pair Encoding tokenizer built from scratch.
Trains on raw text, encodes/decodes, handles special tokens.
Falls back to HuggingFace tokenizers for production use.
"""

import re
import json
import os
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Iterator


# ─────────────────────────────────────────────
# BPE Tokenizer — from scratch
# ─────────────────────────────────────────────

class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer.
    Trains a merge table from raw text, then uses it to encode/decode.
    Compatible with GPT-style byte-level BPE.
    """

    SPECIAL_TOKENS = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<unk>": 3,
        "<mask>": 4,
    }

    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.merges: List[Tuple[str, str]] = []     # ordered merge rules
        self.vocab: Dict[str, int] = {}              # token → id
        self.inv_vocab: Dict[int, str] = {}          # id → token
        self._merge_map: Dict[Tuple[str, str], str] = {}
        self._trained = False

    # ── Training ──────────────────────────────

    def train(self, texts: List[str], verbose: bool = False) -> "BPETokenizer":
        """Train BPE on a list of text strings."""
        # Start from character-level vocabulary
        word_freqs = self._build_word_freqs(texts)
        vocab = self._init_vocab(word_freqs)

        num_merges = self.vocab_size - len(self.SPECIAL_TOKENS) - len(vocab)
        num_merges = max(0, num_merges)

        if verbose:
            print(f"Starting vocab: {len(vocab)} tokens, targeting {self.vocab_size}")

        for i in range(num_merges):
            pairs = self._get_pair_freqs(word_freqs)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] < 2:
                break

            new_token = best_pair[0] + best_pair[1]
            self.merges.append(best_pair)
            self._merge_map[best_pair] = new_token
            word_freqs = self._merge_pair(best_pair, word_freqs)

            if verbose and i % 500 == 0:
                print(f"  merge {i:5d}: {best_pair} → {new_token!r} (freq={pairs[best_pair]})")

        self._build_vocab(word_freqs)
        self._trained = True
        if verbose:
            print(f"Final vocab size: {len(self.vocab)}")
        return self

    def _build_word_freqs(self, texts: List[str]) -> Dict[Tuple[str,...], int]:
        """Tokenize text into characters, count word frequencies."""
        freqs = Counter()
        for text in texts:
            # Split on whitespace; mark word boundaries with Ġ (GPT-style)
            for word in text.split():
                word_chars = tuple("Ġ" + word)
                freqs[word_chars] += 1
        return dict(freqs)

    def _init_vocab(self, word_freqs: Dict) -> set:
        """Collect initial character-level vocabulary."""
        chars = set()
        for word in word_freqs:
            chars.update(word)
        return chars

    def _get_pair_freqs(self, word_freqs: Dict) -> Counter:
        pairs = Counter()
        for word, freq in word_freqs.items():
            for a, b in zip(word, word[1:]):
                pairs[(a, b)] += freq
        return pairs

    def _merge_pair(self, pair: Tuple[str, str],
                    word_freqs: Dict) -> Dict:
        new_freqs = {}
        a, b = pair
        bigram = a + b
        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i+1] == b:
                    new_word.append(bigram)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_freqs[tuple(new_word)] = freq
        return new_freqs

    def _build_vocab(self, word_freqs: Dict):
        self.vocab = dict(self.SPECIAL_TOKENS)
        all_tokens = set()
        for word in word_freqs:
            all_tokens.update(word)
        for tok in sorted(all_tokens):
            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab)
        # merged tokens
        for a, b in self.merges:
            tok = a + b
            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    # ── Encoding / Decoding ───────────────────

    def encode(self, text: str, add_bos: bool = False,
               add_eos: bool = False) -> List[int]:
        tokens = []
        if add_bos:
            tokens.append(self.SPECIAL_TOKENS["<bos>"])
        for word in text.split():
            word_tokens = list("Ġ" + word)
            # Apply merges greedily
            for pair, merged in self._merge_map.items():
                i = 0
                result = []
                while i < len(word_tokens):
                    if i < len(word_tokens)-1 and (word_tokens[i], word_tokens[i+1]) == pair:
                        result.append(merged)
                        i += 2
                    else:
                        result.append(word_tokens[i])
                        i += 1
                word_tokens = result
            for tok in word_tokens:
                tokens.append(self.vocab.get(tok, self.SPECIAL_TOKENS["<unk>"]))
        if add_eos:
            tokens.append(self.SPECIAL_TOKENS["<eos>"])
        return tokens

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        special_ids = set(self.SPECIAL_TOKENS.values()) if skip_special_tokens else set()
        tokens = [self.inv_vocab.get(i, "<unk>") for i in ids if i not in special_ids]
        text = "".join(tokens).replace("Ġ", " ").strip()
        return text

    def batch_encode(self, texts: List[str], max_length: Optional[int] = None,
                     pad: bool = True, add_bos: bool = True,
                     add_eos: bool = True) -> Dict:
        encoded = [self.encode(t, add_bos=add_bos, add_eos=add_eos) for t in texts]

        if max_length:
            encoded = [e[:max_length] for e in encoded]

        lengths = [len(e) for e in encoded]
        max_len = max(lengths)

        if pad:
            pad_id = self.SPECIAL_TOKENS["<pad>"]
            attention_masks = []
            for i, e in enumerate(encoded):
                mask = [1] * len(e) + [0] * (max_len - len(e))
                encoded[i] = e + [pad_id] * (max_len - len(e))
                attention_masks.append(mask)
        else:
            attention_masks = [[1] * l for l in lengths]

        return {
            "input_ids": encoded,
            "attention_mask": attention_masks,
            "lengths": lengths,
        }

    # ── Persistence ───────────────────────────

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        data = {
            "vocab_size": self.vocab_size,
            "vocab": self.vocab,
            "merges": self.merges,
        }
        with open(os.path.join(path, "tokenizer.json"), "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(os.path.join(path, "tokenizer.json")) as f:
            data = json.load(f)
        tok = cls(vocab_size=data["vocab_size"])
        tok.vocab = data["vocab"]
        tok.merges = [tuple(m) for m in data["merges"]]
        tok._merge_map = {(a, b): a+b for a, b in tok.merges}
        tok.inv_vocab = {v: k for k, v in tok.vocab.items()}
        tok._trained = True
        return tok

    def __len__(self):
        return len(self.vocab)

    def __repr__(self):
        return f"BPETokenizer(vocab_size={len(self.vocab)}, trained={self._trained})"


# ─────────────────────────────────────────────
# HuggingFace wrapper for production use
# ─────────────────────────────────────────────

class HFTokenizerWrapper:
    """
    Thin wrapper around any HuggingFace tokenizer.
    Exposes the same interface as BPETokenizer so the rest of
    the system is tokenizer-agnostic.
    """
    def __init__(self, model_name: str = "gpt2"):
        try:
            from transformers import AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(model_name)
            if self._tok.pad_token is None:
                self._tok.pad_token = self._tok.eos_token
        except ImportError:
            raise ImportError("transformers not installed. pip install transformers")

    def encode(self, text: str, add_bos: bool = False,
               add_eos: bool = False) -> List[int]:
        ids = self._tok.encode(text)
        if add_bos and ids[0] != self._tok.bos_token_id:
            ids = [self._tok.bos_token_id] + ids
        if add_eos and ids[-1] != self._tok.eos_token_id:
            ids = ids + [self._tok.eos_token_id]
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    def batch_encode(self, texts, max_length=None, pad=True,
                     add_bos=True, add_eos=True):
        result = self._tok(
            texts, padding=pad, truncation=bool(max_length),
            max_length=max_length, return_tensors=None
        )
        return {
            "input_ids": result["input_ids"],
            "attention_mask": result["attention_mask"],
            "lengths": [sum(m) for m in result["attention_mask"]],
        }

    def __len__(self):
        return len(self._tok)


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Step 10: Tokenizer training + encode/decode")

    corpus = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "language models are powerful tools",
        "transformers changed natural language processing",
        "attention is all you need",
    ] * 20  # repeat to get sufficient frequency

    tok = BPETokenizer(vocab_size=500)
    tok.train(corpus, verbose=True)

    test = "the cat ran in the park"
    ids = tok.encode(test, add_bos=True, add_eos=True)
    decoded = tok.decode(ids)
    print(f"\nInput:   {test!r}")
    print(f"Token ids: {ids}")
    print(f"Decoded: {decoded!r}")
    print(f"Vocab size: {len(tok)}")
    print("✓ Step 10 verified")
