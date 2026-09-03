from __future__ import annotations

import unittest

from v5_contracts.lineage import ArtifactIdentity
from v5_tokenizer.adapter import SPECIAL_TOKEN_IDS, FrozenTokenizer, TokenizerIdentity


def _identity(**overrides) -> TokenizerIdentity:
    fields: dict[str, object] = {
        "schema": "anra-v5-tokenizer-identity/v1",
        "vocabulary_size": 24576,
        "special_token_ids": dict(SPECIAL_TOKEN_IDS),
        "artifact_sha256": "a" * 64,
        "trainer_config_sha256": "b" * 64,
        "corpus_manifest_sha256": "c" * 64,
    }
    fields.update(overrides)
    return TokenizerIdentity(**fields)  # type: ignore[arg-type]


class _Backend:
    def __init__(self, vocab: dict[str, int]) -> None:
        self._vocab = vocab
        self._inverse = {value: key for key, value in vocab.items()}
        self._inverse.setdefault(1, "<unk>")

    def encode(self, text: str):
        return [self._vocab.get(word, 1) for word in text.split(" ")]

    def decode(self, ids):
        return " ".join(self._inverse[i] for i in ids)


class TokenizerTests(unittest.TestCase):
    def test_identity_freezes_to_valid_receipt(self) -> None:
        identity = _identity()
        artifact = ArtifactIdentity(artifact_id="tok-24576", sha256="a" * 64, byte_size=1024)
        receipt = identity.freeze(
            artifact=artifact, identity_roundtrip_passed=True, unknown_rate=0.0
        )
        self.assertEqual(receipt.vocabulary_size, 24576)

    def test_freeze_rejects_provenance_mismatch_and_shortcuts(self) -> None:
        identity = _identity()
        artifact = ArtifactIdentity(artifact_id="tok-24576", sha256="f" * 64, byte_size=1024)
        with self.assertRaises(ValueError):
            identity.freeze(
                artifact=artifact, identity_roundtrip_passed=True, unknown_rate=0.0
            )
        good = ArtifactIdentity(artifact_id="tok-24576", sha256="a" * 64, byte_size=1024)
        with self.assertRaises(ValueError):
            identity.freeze(
                artifact=good, identity_roundtrip_passed=False, unknown_rate=0.0
            )
        with self.assertRaises(ValueError):
            identity.freeze(
                artifact=good, identity_roundtrip_passed=True, unknown_rate=0.01
            )

    def test_identity_rejects_wrong_specials_and_sizes(self) -> None:
        with self.assertRaises(ValueError):
            _identity(vocabulary_size=256).assert_valid()
        with self.assertRaises(ValueError):
            _identity(special_token_ids={"pad": 0, "unk": 1, "bos": 2, "eos": 2}).assert_valid()

    def test_adapter_round_trip_audit_and_segment(self) -> None:
        backend = _Backend({"hello": 10, "world": 11})
        tokenizer = FrozenTokenizer(identity=_identity(), backend=backend)
        self.assertEqual(tokenizer.encode("hello world"), [10, 11])
        self.assertEqual(tokenizer.decode([10, 11]), "hello world")
        report = tokenizer.audit(["hello world", "hello"])
        self.assertTrue(report["identity_roundtrip_passed"])
        self.assertEqual(report["unknown_rate"], 0.0)
        self.assertEqual(tokenizer.segment([10, 11]), [2, 10, 11, 3])
        with self.assertRaises(ValueError):
            tokenizer.segment([10, 2, 11])

    def test_adapter_counts_unknowns(self) -> None:
        backend = _Backend({"hello": 10})
        tokenizer = FrozenTokenizer(identity=_identity(), backend=backend)
        report = tokenizer.audit(["hello", "mystery word"])
        self.assertEqual(report["unknowns"], 2)
        self.assertGreater(float(report["unknown_rate"]), 0.0)


if __name__ == "__main__":
    unittest.main()
