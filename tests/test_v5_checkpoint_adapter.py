"""Checkpoint-backed adapter and gold-firewall tests."""

from __future__ import annotations

import hashlib
import io
import unittest

import torch

from v5_contracts.model_spec import ModelSpec
from v5_evaluation.checkpoint_adapter import CheckpointBackedV5Adapter
from v5_evaluation.firewall import (
    CommittedOutput,
    EvaluatorTruth,
    VisibleTask,
    build_evaluator_truth,
    build_visible_tasks,
    score_committed,
)
from v5_model.core import initialize
from v5_tokenizer.adapter import FrozenTokenizer, TokenizerIdentity


TINY_SPEC = ModelSpec(
    schema="anra-v5-model-spec/v1",
    family="dense-decoder-transformer",
    vocabulary_size=512,
    width=64,
    layers=2,
    query_heads=4,
    kv_heads=2,
    head_dimension=16,
    ffn_width=128,
    context_length=64,
    rope_base=10_000.0,
    norm_epsilon=1e-5,
    tied_embeddings=True,
    qk_norm=True,
    qk_norm_affine=True,
    linear_bias=False,
    dropout=0.0,
)


def _identity() -> TokenizerIdentity:
    return TokenizerIdentity(
        schema="anra-v5-tokenizer-identity/v1",
        vocabulary_size=512,
        special_token_ids={"pad": 0, "unk": 1, "bos": 2, "eos": 3},
        artifact_sha256=hashlib.sha256(b"tiny-tokenizer").hexdigest(),
        trainer_config_sha256="b" * 64,
        corpus_manifest_sha256="c" * 64,
    )


class _ByteTokenizer:
    """Deterministic byte fallback tokenizer for a 512-entry vocabulary."""

    def encode(self, text: str) -> list[int]:
        return [4 + byte for byte in text.encode("utf-8")]

    def decode(self, ids: list[int]) -> str:
        return bytes(max(0, token - 4) for token in ids).decode("utf-8", errors="replace")


def _adapter() -> CheckpointBackedV5Adapter:
    torch.manual_seed(0)
    model = initialize(TINY_SPEC, seed=5)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return CheckpointBackedV5Adapter(
        checkpoint_sha256="a" * 64,
        model_payload=buffer.getvalue(),
        model_spec=TINY_SPEC,
        tokenizer=FrozenTokenizer(identity=_identity(), backend=_ByteTokenizer()),
    )


class CheckpointAdapterTest(unittest.TestCase):
    def test_scores_candidate_suffixes_only(self) -> None:
        adapter = _adapter()
        scores = adapter.score_candidates("the capital of", " france is", [" paris", " london"])
        self.assertEqual(len(scores), 2)
        self.assertTrue(all(isinstance(score, float) for score in scores))
        # identical candidate under two contexts: shared prefix, no prompt leakage
        direct = adapter.score_candidates("", "", ["hello"])
        self.assertEqual(len(direct), 1)

    def test_score_is_deterministic(self) -> None:
        adapter = _adapter()
        first = adapter.score_candidates("ctx", " q", [" a", " b"])
        second = adapter.score_candidates("ctx", " q", [" a", " b"])
        self.assertEqual(first, second)

    def test_generation_is_greedy_and_capped(self) -> None:
        adapter = _adapter()
        output = adapter.generate_free("hello", max_new_tokens=8)
        self.assertIsInstance(output, str)
        again = adapter.generate_free("hello", max_new_tokens=8)
        self.assertEqual(output, again)

    def test_constrained_generation_returns_a_candidate(self) -> None:
        adapter = _adapter()
        choice = adapter.generate_constrained("pick one:", [" alpha", " beta"])
        self.assertIn(choice, [" alpha", " beta"])

    def test_adapter_never_mutates_weights(self) -> None:
        adapter = _adapter()
        before = {
            name: parameter.detach().clone()
            for name, parameter in adapter.model.named_parameters()
        }
        adapter.generate_free("hello", max_new_tokens=4)
        adapter.score_candidates("ctx", " q", [" a"])
        for name, parameter in adapter.model.named_parameters():
            self.assertTrue(torch.equal(before[name], parameter.detach()))

    def test_identity_binds_checkpoint_spec_and_tokenizer(self) -> None:
        adapter = _adapter()
        self.assertEqual(adapter.identity.checkpoint_sha256, "a" * 64)
        self.assertEqual(adapter.identity.model_spec_sha256, TINY_SPEC.sha256())
        self.assertNotEqual(adapter.identity.sha256(), adapter.identity.sha256() + "x")
        self.assertIn("suffix", adapter.identity.scoring_rule)


class FirewallTest(unittest.TestCase):
    def _records(self) -> list[dict[str, object]]:
        return [
            {
                "task_id": "t1",
                "cluster_id": "c1",
                "family": "query_binding",
                "split": "fresh",
                "difficulty": "easy",
                "prompt": "The scarf is green. What color is the scarf?",
                "candidates": ("green", "red"),
                "gold": "green",
            }
        ]

    def test_projection_drops_truth(self) -> None:
        visible = build_visible_tasks(self._records())[0]
        self.assertFalse(hasattr(visible, "gold"))
        truth = build_evaluator_truth(self._records())[0]
        self.assertEqual(truth.gold, "green")

    def test_prompt_embedded_truth_is_rejected(self) -> None:
        records = self._records()
        records[0]["prompt"] = "The scarf is green. answer: green. What color?"
        with self.assertRaises(ValueError):
            build_visible_tasks(records)

    def test_truth_joins_only_after_commit(self) -> None:
        records = self._records()
        visible = build_visible_tasks(records)[0]
        truth = build_evaluator_truth(records)[0]
        committed = CommittedOutput(task_id="t1", output="green", candidate_scores=(-1.5,))
        result = score_committed(committed, visible, truth)
        self.assertTrue(result.correct)
        self.assertEqual(result.gold, "green")
        self.assertEqual(result.raw_output, "green")
        with self.assertRaises(ValueError):
            score_committed(
                CommittedOutput(task_id="other", output="green", candidate_scores=None),
                visible,
                truth,
            )

    def test_visible_task_structurally_has_no_truth_field(self) -> None:
        with self.assertRaises(TypeError):
            VisibleTask(  # type: ignore[call-arg]
                task_id="t2",
                cluster_id="c2",
                family="f",
                split="fresh",
                difficulty="easy",
                prompt="p",
                candidates=(),
                gold="secret",  # type: ignore[misc]
            )


if __name__ == "__main__":
    unittest.main()
