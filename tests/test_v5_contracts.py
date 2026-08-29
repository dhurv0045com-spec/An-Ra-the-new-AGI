from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from e0_cognition.inference import paired_bootstrap_delta, paired_sign_test_pvalue
from e0_cognition.preregistration import PROTOCOL, protocol_sha256
from e0_cognition.sealed import build_commitment
from v5_contracts.certify import build_certificate
from v5_contracts.data_spec import (
    DataManifest,
    PackManifest,
    PackShard,
    SourceRecord,
    assert_source_disjoint,
)
from v5_contracts.import_boundaries import scan_repository
from v5_contracts.lineage import CheckpointManifest, PromotionDecision
from v5_contracts.model_spec import V5A_250M
from v5_contracts.run_spec import V5A_RUN_CENTER


class V5ContractTests(unittest.TestCase):
    def test_exact_250m_center_receipt(self) -> None:
        receipt = V5A_250M.parameter_receipt()
        self.assertEqual(receipt.total, 250_216_960)
        self.assertLess(abs(receipt.total - 250_000_000) / 250_000_000, 0.001)
        self.assertEqual(V5A_250M.query_heads // V5A_250M.kv_heads, 2)

    def test_run_center_scales_tokens_and_compute(self) -> None:
        receipt = V5A_RUN_CENTER.receipt(V5A_250M)
        self.assertEqual(receipt["token_budget"], 5_000_000_000)
        self.assertAlmostEqual(receipt["tokens_per_parameter"], 19.98265824986444)
        self.assertEqual(receipt["idealized_6nd_flops"], 7_506_508_800_000_000_000)
        self.assertEqual(sum(receipt["data_tokens"].values()), 5_000_000_000)
        self.assertEqual(
            receipt["checkpoint_storage_planning_bytes"]["full_resume_without_gradients"],
            3_503_037_440,
        )

    def test_configuration_certificate_is_scoped_and_passes(self) -> None:
        certificate = build_certificate()
        self.assertEqual(certificate["status"], "PASS")
        self.assertFalse(certificate["checks"]["main_training_authorized"])
        self.assertIn("no model or trainer", certificate["scope"])

    def test_repository_import_boundaries(self) -> None:
        self.assertEqual(scan_repository(Path.cwd()), [])

    def test_checkpoint_manifest_enforces_token_and_optimizer_identity(self) -> None:
        sha = "a" * 64
        manifest = CheckpointManifest(
            schema="anra-v5-checkpoint/v1",
            lineage_id="trial",
            checkpoint_id="step-10",
            parent_checkpoint_sha256=None,
            source_commit="deadbeef",
            model_spec_sha256=sha,
            tokenizer_sha256=sha,
            data_manifest_sha256=sha,
            global_update=10,
            cumulative_tokens=100,
            tokens_by_source={"natural": 70, "cognition": 30},
            curriculum_phase="uniform",
            sampler_cursor="shard=0,row=100",
            distributed_topology="1xCPU-test",
            precision="fp32-test",
            parameter_sha256=sha,
            optimizer_step_max=10,
            rng_state_sha256=sha,
        )
        manifest.assert_valid()
        with self.assertRaises(ValueError):
            replace(manifest, optimizer_step_max=9).assert_valid()

    def test_data_and_pack_manifests_reconcile_exact_tokens(self) -> None:
        sha = "a" * 64
        source = SourceRecord(
            source_id="source-1",
            authorization_category="public-domain",
            acquired_date="2026-08-29",
            raw_sha256="b" * 64,
            split="training",
            domain="technical",
        )
        data = DataManifest(
            schema="anra-v5-data/v1",
            manifest_id="data-1",
            tokenizer_sha256=sha,
            filter_version="f1",
            dedup_version="d1",
            contamination_scan_sha256="c" * 64,
            sources=(source,),
            tokens_by_family={"natural": 80, "cognition": 20},
            total_tokens=100,
        )
        data.assert_valid()
        pack = PackManifest(
            schema="anra-v5-pack/v1",
            tokenizer_sha256=sha,
            data_manifest_sha256="d" * 64,
            packer_version="p1",
            cursor_schema="cursor/v1",
            shards=(PackShard("s1", "e" * 64, 200, 4, 100),),
            total_tokens=100,
        )
        pack.assert_valid()
        with self.assertRaises(ValueError):
            replace(data, total_tokens=99).assert_valid()
        with self.assertRaises(ValueError):
            assert_source_disjoint(data, replace(data, manifest_id="fresh-natural"))

    def test_promotion_contract_forbids_failed_gate_promotion(self) -> None:
        with self.assertRaises(ValueError):
            PromotionDecision(
                schema="anra-v5-promotion/v1",
                checkpoint_sha256="a" * 64,
                evaluation_receipt_sha256="b" * 64,
                decision="promote",
                failed_gates=("state",),
                signed_by="independent-evaluator",
            ).assert_valid()

    def test_statistical_protocol_and_paired_methods(self) -> None:
        self.assertEqual(len(protocol_sha256()), 64)
        self.assertEqual(PROTOCOL["binary_accuracy"]["alpha"], 0.05)
        self.assertLess(paired_sign_test_pvalue([True] * 10, [False] * 10), 0.01)
        delta = paired_bootstrap_delta([2.0, 3.0, 4.0], [1.0, 1.0, 1.0], seed=7, resamples=1_000)
        self.assertGreater(delta.lower_95, 0.0)

    def test_sealed_commitment_refuses_repository_fixture(self) -> None:
        repository = Path.cwd()
        fixture = repository / "not-a-real-sealed-fixture.json"
        with self.assertRaises(ValueError):
            build_commitment(fixture=fixture, repository=repository, custody_id="test")

    def test_sealed_commitment_contains_hash_not_seed_or_cases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "sealed.json"
            payload = {
                "schema": "esoes-e0-suite/v1",
                "split": "sealed",
                "generator_version": "test-only",
                "seed": 123,
                "cases": [{"answer": "SECRET"}],
            }
            raw = json.dumps(payload).encode()
            fixture.write_bytes(raw)
            commitment = build_commitment(
                fixture=fixture, repository=Path.cwd(), custody_id="external-test-custodian"
            )
            self.assertEqual(commitment["fixture_sha256"], hashlib.sha256(raw).hexdigest())
            self.assertNotIn("seed", commitment)
            self.assertNotIn("cases", commitment)


if __name__ == "__main__":
    unittest.main()
