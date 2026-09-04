"""Hostile red-team battery: every attack must fail loudly.

Each test plays an adversary: stale optimizers, tampered batches, wrong
identities, unchanged-parameter certification, unsigned promotion. Success
is a raised ValueError, never a silent pass.
"""

from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover - environment-dependent
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class BackendAttackTests(unittest.TestCase):
    def _model_opt(self):
        from v5_model.core import initialize
        from v5_contracts.model_spec import V5A_250M
        import dataclasses

        torch.manual_seed(0)
        spec = dataclasses.replace(
            V5A_250M, layers=1, width=32, query_heads=2, kv_heads=1,
            head_dimension=16, ffn_width=64, vocabulary_size=256, context_length=64,
        )
        model = initialize(spec, 0, torch_module=torch)
        from v5_training.optimizer import build_adamw_optimizer

        optimizer = build_adamw_optimizer(model, torch_module=torch)
        return model, optimizer

    def _batch(self, tokens, segments, counts):
        from v5_training.production_backend import PackedBatch
        from v5_training.state import CURSOR_SCHEMA, CursorState

        return PackedBatch(
            tokens=torch.tensor(tokens, dtype=torch.int64),
            segment_ids=torch.tensor(segments, dtype=torch.int64),
            tokens_by_source=counts,
            cursor=CursorState(CURSOR_SCHEMA, "b" * 64, 0, 1, 0),
            rng_state_sha256="c" * 64,
        )

    def test_stale_optimizer_after_parameter_replacement_fails(self) -> None:
        from v5_training.production_backend import assert_live_ownership

        model, optimizer = self._model_opt()
        assert_live_ownership(model, optimizer)
        model.embedding = torch.nn.Embedding(256, 32)
        with self.assertRaises(ValueError):
            assert_live_ownership(model, optimizer)

    def test_frozen_model_cannot_build_optimizer(self) -> None:
        from v5_model.core import initialize
        from v5_contracts.model_spec import V5A_250M
        import dataclasses
        from v5_training.optimizer import build_adamw_optimizer

        spec = dataclasses.replace(
            V5A_250M, layers=1, width=32, query_heads=2, kv_heads=1,
            head_dimension=16, ffn_width=64, vocabulary_size=256, context_length=64,
        )
        model = initialize(spec, 0, torch_module=torch)
        for parameter in model.parameters():
            parameter.requires_grad = False
        with self.assertRaises(ValueError):
            build_adamw_optimizer(model, torch_module=torch)

    def test_unchanged_evidence_refuses_certification(self) -> None:
        from v5_training.production_backend import capture_evidence, certify_real_update

        model, optimizer = self._model_opt()
        evidence = capture_evidence(model, optimizer, torch=torch)
        with self.assertRaises(ValueError):
            certify_real_update(
                model=model, optimizer=optimizer, before=evidence, after=evidence,
                expected_learning_rate=1e-3, supervised_tokens=8, loss=2.5,
                grad_norm_pre_clip=0.7, grad_norm_post_clip=0.7, torch=torch,
            )

    def test_empty_ledger_batch_rejected(self) -> None:
        from v5_model.core import initialize
        from v5_contracts.model_spec import V5A_250M
        import dataclasses
        from v5_training.production_backend import ProductionTrainingBackend
        from v5_training.optimizer import build_adamw_optimizer

        spec = dataclasses.replace(
            V5A_250M, layers=1, width=32, query_heads=2, kv_heads=1,
            head_dimension=16, ffn_width=64, vocabulary_size=256, context_length=64,
        )
        model = initialize(spec, 0, torch_module=torch)
        backend = ProductionTrainingBackend(
            model=model, optimizer=build_adamw_optimizer(model, torch_module=torch),
            bos_id=2, pad_id=0, device="cpu", torch_module=torch,
        )
        batch = self._batch([[2, 3]], [[0, 0]], {})
        with self.assertRaises(ValueError):
            backend._validate_batch(batch)


class IdentityAttackTests(unittest.TestCase):
    def test_wrong_source_commit_rejected(self) -> None:
        from v5_training.provenance import build_manifest
        from v5_training.state import CURSOR_SCHEMA, IDENTITY_SCHEMA, CursorState, IdentityBindings, TrainingState

        identities = IdentityBindings(
            IDENTITY_SCHEMA, "a" * 40, *["b" * 64] * 8,
        )
        state = TrainingState.initial(
            lineage_id="red", token_budget=8, tokens_per_update=4,
            cursor=CursorState(CURSOR_SCHEMA, identities.pack_manifest_sha256, 0, 0, 0),
            rng_state_sha256="c" * 64, curriculum_phase="u", identities=identities,
        )
        with self.assertRaises(ValueError):
            build_manifest(
                state, lineage_id="red", checkpoint_id="x",
                parent_checkpoint_sha256=None, source_commit="short",
                model_spec_sha256="b" * 64, tokenizer_sha256="b" * 64,
                data_manifest_sha256="b" * 64, parameter_sha256="b" * 64,
                rng_state_sha256="c" * 64, distributed_topology="t",
                precision="p",
            )

    def test_wrong_tokenizer_shape_rejected(self) -> None:
        from v5_tokenizer.adapter import FrozenTokenizer, TokenizerIdentity

        class Backend:
            def encode(self, text):
                return [999999]

            def decode(self, ids):
                return ""

        identity = TokenizerIdentity(
            schema="anra-v5-tokenizer-identity/v1", vocabulary_size=24576,
            special_token_ids={"pad": 0, "unk": 1, "bos": 2, "eos": 3},
            artifact_sha256="a" * 64, trainer_config_sha256="b" * 64,
            corpus_manifest_sha256="c" * 64,
        )
        tokenizer = FrozenTokenizer(identity=identity, backend=Backend())
        with self.assertRaises(ValueError):
            tokenizer.encode("hello")

    def test_unsigned_promotion_never_promotes(self) -> None:
        from v5_promotion.decide import PromotionDecision, decide

        decision = PromotionDecision(
            schema="anra-v5-promotion-decision/v2",
            checkpoint_sha256="a" * 64, evaluation_receipt_sha256="b" * 64,
            durability_receipt_sha256="c" * 64, gate_spec_sha256="d" * 64,
            passed_gates=tuple(f"g{i}" for i in range(10)), failed_gates=(),
            signer_id=None, detached_signature_sha256=None,
        )
        self.assertEqual(decide(decision, verifier=lambda _d, _s: True), "INCONCLUSIVE")

    def test_production_scoring_policy_stays_null(self) -> None:
        from v5_contracts.training_spec import build_training_spec

        spec = build_training_spec()
        self.assertIsNone(spec["evaluation"]["production_candidate_scoring_mode"])
        self.assertEqual(spec["objective"]["query_swap_lambda"], 0.0)


if __name__ == "__main__":
    unittest.main()
