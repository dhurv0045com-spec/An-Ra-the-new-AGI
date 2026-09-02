"""Unit tests for senora.tokenizer."""

from __future__ import annotations

import unittest

from senora.tokenizer import (
    EXPECTED_VOCABULARY_SIZE,
    SPECIAL_TOKENS,
    SenoraTokenizer,
    TokenizerValidationError,
    load_verified_tokenizer,
)


class TestSenoraTokenizer(unittest.TestCase):
    def test_default_tokenizer_spec_conformance(self) -> None:
        tok = SenoraTokenizer()
        self.assertEqual(tok.vocabulary_size, EXPECTED_VOCABULARY_SIZE)
        self.assertEqual(tok.special_tokens["pad"], 0)
        self.assertEqual(tok.special_tokens["unk"], 1)
        self.assertEqual(tok.special_tokens["bos"], 2)
        self.assertEqual(tok.special_tokens["eos"], 3)

    def test_invalid_vocabulary_size_fails(self) -> None:
        with self.assertRaises(TokenizerValidationError):
            SenoraTokenizer(vocabulary_size=32000)

    def test_byte_fallback_roundtrip_zero_unknowns(self) -> None:
        tok = SenoraTokenizer()
        sample_texts = [
            "Hello, world!",
            "def test_fn(x: int) -> int:\n    return x + 42",
            "An-Ra AGI research: 15% cognition mixture.",
            "Unicode test: \u03c0 \u2248 3.14159 \u2192 \u221e",
        ]
        for text in sample_texts:
            tokens = tok.encode(text, add_bos=True, add_eos=True)
            self.assertEqual(tokens[0], SPECIAL_TOKENS["bos"])
            self.assertEqual(tokens[-1], SPECIAL_TOKENS["eos"])
            self.assertNotIn(SPECIAL_TOKENS["unk"], tokens)

            # Round trip
            decoded = tok.decode(tokens)
            self.assertEqual(decoded, text)
            self.assertTrue(tok.verify_roundtrip(text))


if __name__ == "__main__":
    unittest.main()