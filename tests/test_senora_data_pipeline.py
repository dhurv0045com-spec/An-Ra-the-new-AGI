"""Unit tests for senora.data_pipeline."""

from __future__ import annotations

import unittest

from v5_training.state import CURSOR_SCHEMA
from senora.data_pipeline import (
    ContaminationViolationError,
    CursorState,
    DataPipeline,
    MissingCorpusArtifactError,
    MixtureRecipe,
    MIXTURE_COGNITION_15,
)


class TestDataPipeline(unittest.TestCase):
    def test_mixture_recipe_token_allocation(self) -> None:
        recipe = MIXTURE_COGNITION_15  # 0.65, 0.20, 0.15
        alloc = recipe.token_allocation(1_000_000)
        self.assertEqual(alloc["natural"], 650_000)
        self.assertEqual(alloc["code"], 200_000)
        self.assertEqual(alloc["cognition"], 150_000)
        self.assertEqual(sum(alloc.values()), 1_000_000)

    def test_invalid_mixture_recipe_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MixtureRecipe("bad-mixture", 0.5, 0.5, 0.5).assert_valid()

    def test_fail_closed_without_pack_manifest(self) -> None:
        with self.assertRaises(MissingCorpusArtifactError):
            DataPipeline(pack_manifest=None, recipe=MIXTURE_COGNITION_15, allow_synthetic_mock=False)

    def test_contamination_detection(self) -> None:
        pipeline = DataPipeline(pack_manifest=None, recipe=MIXTURE_COGNITION_15, allow_synthetic_mock=True)

        # 1. Valid disjoint templates pass
        pipeline.assert_no_contamination(
            training_template_ids=["train.causal.registry", "train.causal.revision"],
            evaluation_template_ids={"dev.eval.binding", "fresh.eval.state"},
        )

        # 2. Collision with evaluation suite triggers error
        with self.assertRaises(ContaminationViolationError):
            pipeline.assert_no_contamination(
                training_template_ids=["train.causal.registry", "dev.eval.binding"],
                evaluation_template_ids={"dev.eval.binding"},
            )

        # 3. Training template escaping reserved namespace triggers error
        with self.assertRaises(ContaminationViolationError):
            pipeline.assert_no_contamination(
                training_template_ids=["unprefixed.template"],
                evaluation_template_ids={"dev.eval.binding"},
            )

    def test_mock_stream_cursor_progression(self) -> None:
        pipeline = DataPipeline(
            pack_manifest=None,
            recipe=MIXTURE_COGNITION_15,
            sequence_length=128,
            batch_size=4,
            allow_synthetic_mock=True,
        )
        init_cursor = CursorState(
            schema=CURSOR_SCHEMA,
            pack_manifest_sha256="c" * 64,
            shard_ordinal=0,
            sequence_ordinal=0,
            token_offset=0,
        )
        stream = pipeline.mock_stream(initial_cursor=init_cursor, total_batches=3)
        batches = list(stream)
        self.assertEqual(len(batches), 3)
        self.assertEqual(batches[0].batch_token_count, 4 * 128)
        self.assertEqual(batches[2].new_cursor.token_offset, 3 * 4 * 128)


if __name__ == "__main__":
    unittest.main()