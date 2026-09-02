"""Unit tests for senora.data_pipeline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from senora.data_pipeline import (
    compute_exact_budget_schedule,
    create_binary_pack_shard,
    BASE_CODE_PARTS,
    BASE_NATURAL_PARTS,
    ContaminationScanner,
    ContaminationViolationError,
    CursorState,
    DataPipeline,
    MissingCorpusArtifactError,
    MixtureRecipe,
    MIXTURE_COGNITION_05,
    MIXTURE_COGNITION_15,
    MIXTURE_COGNITION_30,
    MIXTURE_CONTROL_SUBSTRATE,
)
from v5_training.state import CURSOR_SCHEMA


class TestDataPipeline(unittest.TestCase):
    def test_65_20_ratio_invariance(self) -> None:
        expected_ratio = BASE_NATURAL_PARTS / BASE_CODE_PARTS  # 3.25
        for cognition in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30):
            recipe = MixtureRecipe.from_cognition_fraction(cognition)
            self.assertAlmostEqual(recipe.cognition_fraction, cognition, places=5)
            self.assertAlmostEqual(recipe.natural_fraction / recipe.code_fraction, expected_ratio, places=5)
            self.assertAlmostEqual(
                recipe.natural_fraction + recipe.code_fraction + recipe.cognition_fraction,
                1.0,
                places=5,
            )

    def test_mixture_recipe_token_allocation(self) -> None:
        recipe = MIXTURE_COGNITION_15  # 0.65, 0.20, 0.15
        alloc = recipe.token_allocation(1_000_000)
        self.assertEqual(alloc["natural"], 650_000)
        self.assertEqual(alloc["code"], 200_000)
        self.assertEqual(alloc["cognition"], 150_000)
        self.assertEqual(sum(alloc.values()), 1_000_000)

    def test_control_substrate_preserves_65_20_ratio(self) -> None:
        recipe = MIXTURE_CONTROL_SUBSTRATE
        self.assertEqual(recipe.cognition_fraction, 0.0)
        self.assertAlmostEqual(recipe.natural_fraction, 65.0 / 85.0, places=5)
        self.assertAlmostEqual(recipe.code_fraction, 20.0 / 85.0, places=5)

    def test_fail_closed_without_pack_manifest(self) -> None:
        with self.assertRaises(MissingCorpusArtifactError):
            DataPipeline(pack_manifest=None, recipe=MIXTURE_COGNITION_15, allow_synthetic_mock=False)

    def test_3_level_contamination_detection(self) -> None:
        scanner = ContaminationScanner()

        # Level 1: Template ID collisions
        scanner.level_1_template_disjointness(["train.causal.facts"], {"eval.case.1"})
        with self.assertRaises(ContaminationViolationError):
            scanner.level_1_template_disjointness(["unprefixed.template"], {"eval.case.1"})
        with self.assertRaises(ContaminationViolationError):
            scanner.level_1_template_disjointness(["train.causal.collision"], {"train.causal.collision"})

        # Level 2: Substring / n-gram overlap
        train_text = "the quick brown fox jumps over the lazy dog in the middle of summer"
        eval_safe = "an elephant walks across the savannah during the hot rainy afternoon"
        eval_contaminated = "we saw the quick brown fox jumps over the lazy dog in the park"
        scanner.level_2_ngram_overlap(train_text, [eval_safe], n=6)
        with self.assertRaises(ContaminationViolationError):
            scanner.level_2_ngram_overlap(train_text, [eval_contaminated], n=6)

        # Level 3: Structural signature overlap
        scanner.level_3_structural_signature_overlap("graph_topo_A", {"graph_topo_B", "graph_topo_C"})
        with self.assertRaises(ContaminationViolationError):
            scanner.level_3_structural_signature_overlap("graph_topo_A", {"graph_topo_A", "graph_topo_B"})

    def test_real_binary_shard_reader(self) -> None:
        import hashlib
        import struct

        pipeline = DataPipeline(pack_manifest=None, recipe=MIXTURE_COGNITION_15, allow_synthetic_mock=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            shard_path = Path(temp_dir) / "test_shard.bin"
            token_ids = [101, 202, 303, 404]
            raw_bytes = struct.pack(f"<{len(token_ids)}H", *token_ids)
            shard_path.write_bytes(raw_bytes)
            expected_sha = hashlib.sha256(raw_bytes).hexdigest()

            read_tokens = pipeline.read_real_binary_shard(shard_path, expected_sha)
            self.assertEqual(read_tokens, token_ids)

            # Checksum mismatch
            with self.assertRaises(ValueError):
                pipeline.read_real_binary_shard(shard_path, "0" * 64)


if __name__ == "__main__":
    unittest.main()