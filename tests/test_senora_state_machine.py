"""Unit tests for senora.state_machine."""

from __future__ import annotations

import unittest

from senora.state_machine import (
    ExperimentLifecycle,
    ExperimentPhase,
    FreshLeakageViolationError,
    IllegalPhaseTransitionError,
)


class TestSenoraStateMachine(unittest.TestCase):
    def test_valid_lifecycle_progression(self) -> None:
        lifecycle = ExperimentLifecycle()
        self.assertEqual(lifecycle.current_phase, ExperimentPhase.DRAFT)

        lifecycle.transition_to(ExperimentPhase.PREREGISTERED)
        lifecycle.transition_to(ExperimentPhase.IDENTITIES_BOUND)
        lifecycle.transition_to(ExperimentPhase.REMOTE_CANARY_REQUIRED)
        lifecycle.transition_to(ExperimentPhase.REMOTE_CANARY_PASS)
        lifecycle.transition_to(ExperimentPhase.DEVELOPMENT_RUN)
        lifecycle.transition_to(ExperimentPhase.DEVELOPMENT_COMPLETE)

        # Freeze winning recipe
        lifecycle.freeze_winning_recipe({"arm": "cognition-mixture-15-ce", "lr": 3e-4})
        self.assertEqual(lifecycle.current_phase, ExperimentPhase.RECIPE_FROZEN)

        lifecycle.transition_to(ExperimentPhase.FRESH_COMMITMENT)
        lifecycle.transition_to(ExperimentPhase.FRESH_RUN)
        lifecycle.transition_to(ExperimentPhase.REPLICATED)
        lifecycle.transition_to(ExperimentPhase.M102_ELIGIBLE)
        self.assertEqual(lifecycle.current_phase, ExperimentPhase.M102_ELIGIBLE)

    def test_illegal_phase_skipping_prevented(self) -> None:
        lifecycle = ExperimentLifecycle()
        with self.assertRaises(IllegalPhaseTransitionError):
            lifecycle.transition_to(ExperimentPhase.FRESH_RUN)

        with self.assertRaises(IllegalPhaseTransitionError):
            lifecycle.transition_to(ExperimentPhase.M102_ELIGIBLE)

    def test_fresh_leakage_prevented_before_recipe_frozen(self) -> None:
        lifecycle = ExperimentLifecycle()
        lifecycle.transition_to(ExperimentPhase.PREREGISTERED)
        lifecycle.transition_to(ExperimentPhase.IDENTITIES_BOUND)
        lifecycle.transition_to(ExperimentPhase.REMOTE_CANARY_REQUIRED)
        lifecycle.transition_to(ExperimentPhase.REMOTE_CANARY_PASS)
        lifecycle.transition_to(ExperimentPhase.DEVELOPMENT_RUN)

        # Development access is allowed
        lifecycle.verify_suite_access("development")

        # FRESH / SEALED access is strictly blocked
        with self.assertRaises(FreshLeakageViolationError):
            lifecycle.verify_suite_access("fresh")

        with self.assertRaises(FreshLeakageViolationError):
            lifecycle.verify_suite_access("sealed")


if __name__ == "__main__":
    unittest.main()