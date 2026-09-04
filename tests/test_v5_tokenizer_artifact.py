from __future__ import annotations

import unittest
from pathlib import Path

try:
    import tokenizers  # noqa: F401
    HAS_TOKENIZERS = True
except ImportError:
    HAS_TOKENIZERS = False

from v5_tokenizer.artifact import load_frozen, load_verified_tokenizer

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/e1/local_tournament/tokenizer-24576.json.gz"
ARTIFACT_SHA256 = "97e12db63b343312e5e4abc37df9ef4b01fcb1faba792a6420a4c1b15d0a7fbc"
CORPUS_SHA256 = "eb1f0dbac64524ff4dc589c0292af6dc4c3803f48f8fe0af0a77684fea26fc67"
TRAINER_RECORD = ROOT / "v5_tokenizer/legacy_24k_trainer_record.json"


def _trainer_sha() -> str:
    import hashlib

    return hashlib.sha256(TRAINER_RECORD.read_bytes()).hexdigest()


@unittest.skipIf(not HAS_TOKENIZERS, "tokenizers package is not installed")
class RealArtifactTests(unittest.TestCase):
    def test_loads_verified_artifact_with_exact_identities(self) -> None:
        backend, identity = load_verified_tokenizer(
            ARTIFACT,
            expected_sha256=ARTIFACT_SHA256,
            vocabulary_size=24576,
            trainer_config_sha256=_trainer_sha(),
            corpus_manifest_sha256=CORPUS_SHA256,
        )
        self.assertEqual(identity.vocabulary_size, 24576)
        self.assertEqual(backend.get_vocab_size(), 24576)

    def test_wrong_sha_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            load_verified_tokenizer(
                ARTIFACT,
                expected_sha256="0" * 64,
                vocabulary_size=24576,
                trainer_config_sha256=_trainer_sha(),
                corpus_manifest_sha256=CORPUS_SHA256,
            )

    def test_real_audit_has_zero_unknowns_and_round_trip(self) -> None:
        tokenizer = load_frozen(
            ARTIFACT,
            expected_sha256=ARTIFACT_SHA256,
            vocabulary_size=24576,
            trainer_config_sha256=_trainer_sha(),
            corpus_manifest_sha256=CORPUS_SHA256,
        )
        probes = [
            "hello world",
            "The quick brown fox jumps over 13 lazy dogs.",
            "def train_step(state, batch): return state.advance(batch)",
            "import torch; x = torch.randn(4, 4) @ torch.eye(4)",
            "SELECT * FROM runs WHERE tokens > 1000000;",
            "BSD 3-Clause License: redistribution and use permitted.",
            "x = 0.02 / math.sqrt(2 * 26)",
            "Answer:",
        ]
        report = tokenizer.audit(probes)
        self.assertTrue(report["identity_roundtrip_passed"])
        self.assertEqual(report["unknown_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
