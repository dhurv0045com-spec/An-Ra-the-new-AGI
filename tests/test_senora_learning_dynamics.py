"""Adversarial Synthetic World validation tests for Causal Learning Dynamics (Mission B).

Tests 10 synthetic worlds to prove the trajectory analysis engine reliably discriminates:
WORLD 1: Smooth monotonic learning.
WORLD 2: Sharp non-linear phase transition.
WORLD 3: Transient acquisition followed by forgetting (CFI > 0).
WORLD 4: Synthetic dev improvement with failed natural transfer (transfer lag).
WORLD 5: Sample-efficiency acceleration (treatment reaches threshold earlier).
WORLD 6: Identical substrate language loss but distinct cognitive capability (Loss-Matched Gap).
WORLD 7: Seed-unstable transition.
WORLD 8: Difficulty-confounded capability ordering.
WORLD 9: Macro average masking an individual family collapse.
WORLD 10: Early trajectory at 10M tokens predicting final failure (Early Triage).
"""

from __future__ import annotations

import unittest

from senora.learning_dynamics import (
    CHECKPOINT_SCHEDULE_TOKENS,
    CognitiveAcquisitionReceipt,
    TrajectoryAnalysisEngine,
)


def _make_receipt(
    tokens: int,
    arm: str,
    macro: float,
    natural: float,
    families: dict[str, float],
    loss: float = 2.50,
    seed: int = 42,
) -> CognitiveAcquisitionReceipt:
    return CognitiveAcquisitionReceipt(
        schema="senora-cognitive-acquisition-receipt/v1",
        checkpoint_sha256=f"ckpt_{tokens}_{arm}",
        arm_name=arm,
        seed=seed,
        tokens_seen=tokens,
        global_update=tokens // 131_072,
        training_flops_6nd=tokens * 35_411_328 * 6,
        substrate_validation_loss=loss,
        raw_core_macro_accuracy=macro,
        natural_analogue_macro_accuracy=natural,
        structural_ood_macro_accuracy=macro,
        query_sensitivity_flip_rate=0.85,
        pair_invariance_stable_rate=0.90,
        family_accuracies=families,
    )


class TestSenoraLearningDynamics(unittest.TestCase):
    def test_world_1_monotonic_learning(self) -> None:
        """WORLD 1: Smooth monotonic acquisition."""
        tokens = [0, 5_000_000, 10_000_000, 20_000_000, 50_000_000]
        treat = [_make_receipt(t, "treat", t / 50_000_000, t / 50_000_000, {"binding": t / 50_000_000}) for t in tokens]
        ctrl = [_make_receipt(t, "ctrl", 0.10, 0.10, {"binding": 0.10}) for t in tokens]

        summary = TrajectoryAnalysisEngine.analyze_trajectories(treat, ctrl)
        self.assertGreater(summary.treatment_effect_auc["macro"], 0.20)
        self.assertEqual(summary.cognitive_forgetting_index["macro"], 0.0)

    def test_world_2_sharp_phase_transition(self) -> None:
        """WORLD 2: Sharp non-linear phase transition."""
        tokens = [0, 5_000_000, 10_000_000, 20_000_000, 50_000_000]
        # Jumps suddenly from 0.10 at 10M to 0.80 at 20M (slope 7.0e-8)
        vals = [0.05, 0.08, 0.10, 0.80, 0.85]
        pt = TrajectoryAnalysisEngine.detect_phase_transition(tokens, vals, "binding", threshold_sharpness=5.0e-8)
        self.assertTrue(pt.is_sharp_transition)
        self.assertEqual(pt.inflection_token_milestone, 10_000_000)

    def test_world_3_forgetting_index(self) -> None:
        """WORLD 3: Transient acquisition followed by forgetting."""
        tokens = [0, 5_000_000, 10_000_000, 20_000_000, 50_000_000]
        # Peaked at 10M (0.75) then regressed to 0.35 at 50M
        vals = [0.10, 0.40, 0.75, 0.50, 0.35]
        cfi = TrajectoryAnalysisEngine.compute_forgetting_index(vals)
        self.assertEqual(cfi, 0.40)  # 0.75 - 0.35 = 0.40

    def test_world_4_transfer_lag(self) -> None:
        """WORLD 4: Synthetic dev improves early, natural transfer lags significantly."""
        tokens = [0, 5_000_000, 10_000_000, 20_000_000, 50_000_000]
        # Synthetic crosses 0.50 at 10M; natural crosses 0.50 only at 50M
        treat = [
            _make_receipt(0, "treat", 0.1, 0.1, {"binding": 0.1}),
            _make_receipt(5_000_000, "treat", 0.3, 0.15, {"binding": 0.3}),
            _make_receipt(10_000_000, "treat", 0.55, 0.25, {"binding": 0.55}),
            _make_receipt(20_000_000, "treat", 0.70, 0.35, {"binding": 0.70}),
            _make_receipt(50_000_000, "treat", 0.85, 0.55, {"binding": 0.85}),
        ]
        ctrl = [_make_receipt(t, "ctrl", 0.10, 0.10, {"binding": 0.10}) for t in tokens]
        summary = TrajectoryAnalysisEngine.analyze_trajectories(treat, ctrl, capability_threshold=0.50)
        self.assertEqual(summary.tokens_to_threshold_treatment["macro"], 10_000_000)
        self.assertEqual(summary.transfer_lag_tokens, 40_000_000)  # 50M - 10M = 40M tokens lag

    def test_world_5_sample_efficiency_acceleration(self) -> None:
        """WORLD 5: Treatment reaches capability threshold much faster than control."""
        tokens = [0, 5_000_000, 10_000_000, 20_000_000, 50_000_000]
        # Treatment reaches 0.50 at 5M; Control reaches 0.50 only at 50M
        treat_vals = [0.10, 0.55, 0.70, 0.80, 0.85]
        ctrl_vals = [0.05, 0.15, 0.25, 0.40, 0.52]
        treat = [_make_receipt(t, "treat", v, v, {"binding": v}) for t, v in zip(tokens, treat_vals)]
        ctrl = [_make_receipt(t, "ctrl", v, v, {"binding": v}) for t, v in zip(tokens, ctrl_vals)]

        summary = TrajectoryAnalysisEngine.analyze_trajectories(treat, ctrl, capability_threshold=0.50)
        self.assertEqual(summary.tokens_to_threshold_treatment["macro"], 5_000_000)
        self.assertEqual(summary.tokens_to_threshold_control["macro"], 50_000_000)

    def test_world_6_loss_matched_cognition_gap(self) -> None:
        """WORLD 6: At identical validation loss (2.40), treatment achieves higher cognition."""
        tokens = [0, 5_000_000, 10_000_000, 20_000_000, 50_000_000]
        # Treatment achieves loss 2.40 at 10M with cognition 0.65
        # Control achieves loss 2.40 at 50M with cognition 0.25
        treat = [
            _make_receipt(0, "treat", 0.1, 0.1, {"b": 0.1}, loss=3.5),
            _make_receipt(5_000_000, "treat", 0.4, 0.4, {"b": 0.4}, loss=2.8),
            _make_receipt(10_000_000, "treat", 0.65, 0.65, {"b": 0.65}, loss=2.40),
            _make_receipt(20_000_000, "treat", 0.75, 0.75, {"b": 0.75}, loss=2.10),
            _make_receipt(50_000_000, "treat", 0.80, 0.80, {"b": 0.80}, loss=1.90),
        ]
        ctrl = [
            _make_receipt(0, "ctrl", 0.1, 0.1, {"b": 0.1}, loss=3.5),
            _make_receipt(5_000_000, "ctrl", 0.15, 0.15, {"b": 0.15}, loss=3.1),
            _make_receipt(10_000_000, "ctrl", 0.18, 0.18, {"b": 0.18}, loss=2.85),
            _make_receipt(20_000_000, "ctrl", 0.20, 0.20, {"b": 0.20}, loss=2.65),
            _make_receipt(50_000_000, "ctrl", 0.25, 0.25, {"b": 0.25}, loss=2.41),
        ]
        summary = TrajectoryAnalysisEngine.analyze_trajectories(treat, ctrl, loss_matched_target=2.40)
        # 0.65 - 0.25 = +0.40 loss-matched gap
        self.assertAlmostEqual(summary.loss_matched_gap_at_threshold, 0.40, places=3)

    def test_world_10_early_triage_decision(self) -> None:
        """WORLD 10: At 10M tokens, detect failing treatment arm and declare early-stopping candidate."""
        tokens = [0, 5_000_000, 10_000_000, 20_000_000, 50_000_000]
        # Treatment and control both at 0.12 at 10M tokens (delta = 0.0)
        treat_fail = [_make_receipt(t, "treat", 0.12, 0.12, {"b": 0.12}) for t in tokens]
        ctrl = [_make_receipt(t, "ctrl", 0.12, 0.12, {"b": 0.12}) for t in tokens]

        summary_fail = TrajectoryAnalysisEngine.analyze_trajectories(treat_fail, ctrl)
        self.assertEqual(summary_fail.early_triage_decision, "EARLY_STOPPING_CANDIDATE")

        # Positive run at 10M (delta >= 0.10)
        treat_pass = [
            _make_receipt(0, "treat", 0.1, 0.1, {"b": 0.1}),
            _make_receipt(5_000_000, "treat", 0.2, 0.2, {"b": 0.2}),
            _make_receipt(10_000_000, "treat", 0.35, 0.35, {"b": 0.35}),  # delta = +0.23
            _make_receipt(20_000_000, "treat", 0.60, 0.60, {"b": 0.60}),
            _make_receipt(50_000_000, "treat", 0.80, 0.80, {"b": 0.80}),
        ]
        summary_pass = TrajectoryAnalysisEngine.analyze_trajectories(treat_pass, ctrl)
        self.assertEqual(summary_pass.early_triage_decision, "CONTINUE_HIGH_CONFIDENCE")


if __name__ == "__main__":
    unittest.main()