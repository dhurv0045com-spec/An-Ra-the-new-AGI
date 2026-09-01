from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from v5_contracts.training_spec import build_receipt, build_training_spec, validate_training_spec


class V5TrainingSpecTests(unittest.TestCase):
    def test_spec_has_no_silent_numeric_defaults_and_is_fail_closed(self) -> None:
        spec = build_training_spec()
        self.assertTrue(all(validate_training_spec(spec).values()))
        self.assertFalse(spec["main_training_authorized"])
        self.assertEqual(spec["objective"]["launch_objective"], "causal_cross_entropy_only")
        self.assertEqual(spec["objective"]["query_swap_lambda"], 0.0)
        self.assertIsNone(spec["evaluation"]["production_candidate_scoring_mode"])
        self.assertEqual(spec["core"]["qk_norm_epsilon"], 1e-6)
        self.assertEqual(spec["optimization"]["epsilon"], 1e-8)

    def test_scale_family_and_topology_are_consistent_two_to_one_gqa(self) -> None:
        spec = build_training_spec()
        for stage in spec["scale_ladder"].values():
            if isinstance(stage, dict):
                self.assertEqual(stage["query_heads"], 2 * stage["kv_heads"])
        topology = spec["target_topology"]
        self.assertEqual(
            topology["replicas"]
            * topology["real_tokens_per_replica_microstep"]
            * topology["gradient_accumulation_microsteps"],
            spec["optimization"]["global_tokens_per_update"],
        )

    def test_committed_receipt_is_reproducible_and_source_bound(self) -> None:
        root = Path(__file__).resolve().parents[1]
        committed = json.loads(
            (root / "artifacts/v5/training_spec_v1.json").read_text(encoding="utf-8")
        )
        self.assertEqual(committed, build_receipt())
        normalized = (root / "v5_contracts/training_spec.py").read_text(encoding="utf-8").replace("\r\n", "\n")
        self.assertEqual(
            committed["spec"]["implementation_sha256"],
            hashlib.sha256(normalized.encode()).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
