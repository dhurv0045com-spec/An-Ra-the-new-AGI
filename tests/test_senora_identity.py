"""Unit tests for senora.experiment_identity."""

from __future__ import annotations

import unittest
from dataclasses import replace

from senora.experiment_identity import ExperimentIdentity, SCHEMA


class TestExperimentIdentity(unittest.TestCase):
    def setUp(self) -> None:
        self.sha = "a" * 64
        self.commit = "b" * 40
        self.identity = ExperimentIdentity(
            schema=SCHEMA,
            experiment_id="P35-CMS-1",
            source_commit_sha=self.commit,
            model_spec_sha256=self.sha,
            model_constructor_sha256=self.sha,
            tokenizer_artifact_sha256=self.sha,
            corpus_manifest_sha256=self.sha,
            data_manifest_sha256=self.sha,
            pack_manifest_sha256=self.sha,
            generator_version="e0-train/0.2.0",
            split_identities={"training": self.sha, "fresh": self.sha},
            optimizer_spec={"family": "AdamW", "lr": 3e-4, "weight_decay": 0.1},
            schedule_spec={"family": "WSD", "warmup_tokens": 1_000_000},
            precision="bf16-mixed-fp32-master",
            token_budget=50_000_000,
            tokens_per_update=131_072,
            random_seeds=(42, 43),
            evaluator_spec={"suite_version": "e0-eval/0.4.0"},
            scorer_firewall_status="BYPASS_CANDIDATE_LOGPROB_RAW_CORE_ONLY",
            statistical_protocol={"paired_sign_test": True, "resamples": 10_000},
            promotion_criteria={"min_ood_delta": 0.25},
            abort_criteria={
                "max_loss_regression_fraction": 0.03,
                "fail_on_nan_loss": True,
                "fail_on_gradient_explosion": True,
                "fail_on_stagnation": True,
            },
        )

    def test_identity_assert_valid_and_sha256(self) -> None:
        self.identity.assert_valid()
        digest = self.identity.sha256()
        self.assertEqual(len(digest), 64)

    def test_invalid_sha_rejected(self) -> None:
        with self.assertRaises(ValueError):
            replace(self.identity, model_spec_sha256="not_a_valid_sha").assert_valid()

    def test_missing_abort_key_rejected(self) -> None:
        with self.assertRaises(ValueError):
            replace(self.identity, abort_criteria={}).assert_valid()

    def test_is_run_authorized_fails_on_placeholder_hash(self) -> None:
        placeholder = replace(self.identity, corpus_manifest_sha256="0" * 64)
        authorized, blockers = placeholder.is_run_authorized()
        self.assertFalse(authorized)
        self.assertTrue(any("corpus_manifest_sha256 is a placeholder" in b for b in blockers))


if __name__ == "__main__":
    unittest.main()