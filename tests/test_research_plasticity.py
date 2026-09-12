"""B06 focused tests: plasticity-controller state machine (incl. B2.1
evidence-aligned refinements: qualification-gated protection, collapse
confirmation, sustained recovery).

All tests are deterministic state transitions over crafted traces; no model,
no data and no optimizer are involved.
"""
import unittest

from bramastra_lab.research.config import ControllerSection
from bramastra_lab.research.learning.plasticity import (
    ControllerDecision,
    ControllerInputError,
    ControllerMetrics,
    ControllerState,
    PoolViolationError,
    transition,
)


def controller_config(**overrides) -> ControllerSection:
    raw = {"mode": "evidence_driven", "controller_pool_id": "controller-pool-a",
           "formation_threshold": 0.9, "reacquire_threshold": 0.6,
           "stabilize_window": 3, "cooldown_updates": 10, "max_transitions": 8,
           "max_metrics_age_updates": 8, "stabilize_lr_multiplier": 0.25,
           "expand_lr_multiplier": 1.0, "reacquire_lr_multiplier": 1.0,
           "collapse_confirmation_evaluations": 2, "recovery_confirmation_evaluations": 2}
    raw.update(overrides)
    return ControllerSection.from_dict(raw)


def metrics(family_scores, at_update=100, displacement=0.5, pool="controller-pool-a",
            improvements=None, relative=None):
    return ControllerMetrics(pool_id=pool, at_update=at_update,
                             family_scores=dict(family_scores),
                             parameter_displacement=displacement,
                             relative_parameter_displacement=relative,
                             score_improvements=improvements or {})


class PoolFirewallTests(unittest.TestCase):
    def test_sealed_and_measurement_pools_reject(self) -> None:
        for pool in ("sealed", "development", "measurement", "strategy_validation",
                     "confirmation"):
            with self.assertRaises(PoolViolationError):
                metrics({"f1": 0.9}, pool=pool)

    def test_wrong_declared_pool_is_rejected_at_use(self) -> None:
        state = ControllerState()
        with self.assertRaises(PoolViolationError):
            transition(state, metrics({"f1": 0.9}, pool="controller-pool-b"),
                       controller_config(), current_update=100)

    def test_invalid_metrics_reject_or_hold(self) -> None:
        with self.assertRaises(ControllerInputError):
            metrics({"f1": 1.4})
        with self.assertRaises(ControllerInputError):
            ControllerMetrics(pool_id="p", at_update=-1, family_scores={"f": 0.5})
        state = ControllerState()
        bad = ControllerMetrics(pool_id="controller-pool-a", at_update=100,
                                family_scores={"f1": 0.5}, nonfinite_observed=True)
        next_state, decision = transition(state, bad, controller_config(), current_update=100)
        self.assertEqual(next_state.state, "HOLD")
        self.assertTrue(decision.pause_updates)
        self.assertEqual(decision.reason, "invalid_metrics")


class DisabledAndFixedModesTests(unittest.TestCase):
    def test_disabled_mode_never_transitions(self) -> None:
        config = ControllerSection.from_dict({"mode": "disabled"})
        state = ControllerState()
        for _ in range(3):
            state, decision = transition(state, metrics({"f1": 0.1}), config,
                                         current_update=10)
            self.assertEqual(decision.state, "FORM")
            self.assertEqual(decision.lr_multiplier, 1.0)
            self.assertEqual(decision.reason, "controller_disabled")
            self.assertEqual(state.transitions, 0)

    def test_fixed_schedule_control_available(self) -> None:
        config = ControllerSection.from_dict({"mode": "fixed_schedule"})
        state, decision = transition(ControllerState(), metrics({"f1": 0.95}), config,
                                     current_update=10)
        self.assertEqual(decision.state, "FIXED_SCHEDULE")
        self.assertEqual(decision.lr_multiplier, 1.0)


class FormationPathTests(unittest.TestCase):
    def test_expand_tracks_qualifying_family_without_protection(self) -> None:
        """Protection is earned at qualification, not at introduction."""
        state = ControllerState()
        next_state, decision = transition(
            state, metrics({"f_new": 0.2, "f_old": 0.95}), controller_config(),
            current_update=100, family_request="f_new")
        self.assertEqual(next_state.state, "EXPAND")
        self.assertEqual(next_state.acquiring_family, "f_new")
        self.assertNotIn("f_new", next_state.protected_families)
        self.assertEqual(next_state.protected_families, ())
        self.assertEqual(decision.lr_multiplier, 1.0)
        self.assertTrue(decision.checkpoint_requested)
        self.assertEqual(decision.replay_mixture,
                         {"acquisition": 0.75, "protected_replay": 0.25})

    def test_qualification_promotes_acquiring_family_to_protection(self) -> None:
        config = controller_config()
        state, _ = transition(ControllerState(),
                              metrics({"f1": 0.2}, at_update=0, displacement=0.1),
                              config, current_update=0, family_request="f1")
        for update in (10, 11, 12):
            state, _ = transition(
                state, metrics({"f1": 0.95}, at_update=update, displacement=0.1),
                config, current_update=update)
        self.assertEqual(state.state, "STABILIZE")
        self.assertIn("f1", state.protected_families)  # earned at qualification
        self.assertNotIn("f1", state.qualifying_families)
        # Introducing f2 keeps f1 protected and f2 unqualified.
        state, _ = transition(
            state, metrics({"f1": 0.95, "f2": 0.2}, at_update=100, displacement=0.1),
            config, current_update=100, family_request="f2")
        self.assertEqual(state.acquiring_family, "f2")
        self.assertIn("f1", state.protected_families)
        self.assertNotIn("f2", state.protected_families)
        # A missing protected metric holds; a missing qualifying metric does not.
        held, decision = transition(
            state, metrics({"f2": 0.9}, at_update=101, displacement=0.1),
            config, current_update=101)
        self.assertEqual(held.state, "HOLD")
        self.assertIn("missing_protected_metric", decision.reason)
        released, decision = transition(
            held, metrics({"f1": 0.95, "f2": 0.2}, at_update=102, displacement=0.1),
            config, current_update=102)
        self.assertNotEqual(released.state, "HOLD")
        self.assertIn("hold_released", decision.reason)
        no_pass, _ = transition(
            released, metrics({"f1": 0.95}, at_update=103, displacement=0.1),
            config, current_update=103)
        self.assertEqual(no_pass.state, "EXPAND")
        self.assertEqual(no_pass.consecutive_passes, 0)  # missing acquiring: no pass, no hold

    def test_formation_requires_window_and_plasticity_evidence(self) -> None:
        config = controller_config()
        # Introduced with an already-high score and no subsequent learning:
        # stable scores alone are not acquired improvement.
        state, _ = transition(ControllerState(),
                              metrics({"f_new": 0.95, "f_old": 0.95}), config,
                              current_update=100, family_request="f_new")
        self.assertEqual(state.window_start_score, 0.95)
        state, decision = transition(
            state, metrics({"f_new": 0.95, "f_old": 0.95}, at_update=110,
                           displacement=0.0),
            config, current_update=110)
        self.assertEqual(state.state, "EXPAND")
        self.assertEqual(state.consecutive_passes, 0)
        self.assertEqual(decision.reason, "acquiring_formation_in_progress")

        # With displacement evidence the window counts up.
        for update in (111, 112):
            state, _ = transition(
                state, metrics({"f_new": 0.95, "f_old": 0.95}, at_update=update,
                               displacement=0.1),
                config, current_update=update)
        self.assertEqual(state.consecutive_passes, 2)
        state, decision = transition(
            state, metrics({"f_new": 0.95, "f_old": 0.95}, at_update=113,
                           displacement=0.1),
            config, current_update=113)
        self.assertEqual(state.state, "STABILIZE")
        self.assertAlmostEqual(decision.lr_multiplier, 0.25)

    def test_relative_displacement_counts_as_plasticity_evidence(self) -> None:
        config = controller_config()
        state, _ = transition(ControllerState(),
                              metrics({"f_new": 0.95, "f_old": 0.95}, at_update=0,
                                      displacement=0.0, relative=0.0),
                              config, current_update=0, family_request="f_new")
        state, _ = transition(
            state, metrics({"f_new": 0.95, "f_old": 0.95}, at_update=10,
                           displacement=0.0, relative=0.05),
            config, current_update=10)
        self.assertEqual(state.consecutive_passes, 1)

    def test_threshold_equality_counts_as_pass(self) -> None:
        config = controller_config()
        state, _ = transition(ControllerState(),
                              metrics({"f_new": 0.2, "f_old": 0.95}), config,
                              current_update=100, family_request="f_new")
        state, _ = transition(
            state, metrics({"f_new": 0.9, "f_old": 0.95}, at_update=110, displacement=0.1),
            config, current_update=110)
        self.assertEqual(state.consecutive_passes, 1)

    def test_score_improvement_counts_as_plasticity_evidence(self) -> None:
        config = controller_config()
        state, _ = transition(ControllerState(),
                              metrics({"f_new": 0.2, "f_old": 0.95}), config,
                              current_update=100, family_request="f_new")
        self.assertEqual(state.window_start_score, 0.2)
        state, _ = transition(
            state, metrics({"f_new": 0.92, "f_old": 0.95}, at_update=110,
                           displacement=0.0),
            config, current_update=110)
        self.assertEqual(state.consecutive_passes, 1)


class PreservationTests(unittest.TestCase):
    def qualified_two_family_state(self, config) -> tuple[ControllerState, int]:
        """f1 qualified (protected), f2 introduced (acquiring, unqualified)."""
        state, _ = transition(ControllerState(),
                              metrics({"f1": 0.2}, at_update=0, displacement=0.1),
                              config, current_update=0, family_request="f1")
        update = 10
        for index in range(3):
            state, _ = transition(
                state, metrics({"f1": 0.95}, at_update=update, displacement=0.1),
                config, current_update=update)
            update += 1
        self.assertEqual(state.state, "STABILIZE")
        self.assertIn("f1", state.protected_families)
        update = 100
        state, _ = transition(
            state, metrics({"f1": 0.95, "f2": 0.2}, at_update=update, displacement=0.1),
            config, current_update=update, family_request="f2")
        self.assertEqual(state.state, "EXPAND")
        self.assertEqual(state.acquiring_family, "f2")
        return state, update

    def test_confirmed_protected_collapse_triggers_reacquire_without_lowering_lr(self) -> None:
        config = controller_config()
        state, update = self.qualified_two_family_state(config)
        # First dipping evaluation arms the pending collapse only; the
        # boundary then continues normal EXPAND formation handling.
        state, decision = transition(
            state, metrics({"f1": 0.3, "f2": 0.95}, at_update=update + 1,
                           displacement=0.1),
            config, current_update=update + 1)
        self.assertEqual(state.state, "EXPAND")
        self.assertEqual(state.pending_collapse.get("f1"), 1)
        self.assertEqual(decision.reason, "acquiring_formation_in_progress")
        # Second consecutive dipping evaluation confirms the collapse.
        state, decision = transition(
            state, metrics({"f1": 0.3, "f2": 0.95}, at_update=update + 2,
                           displacement=0.1),
            config, current_update=update + 2)
        self.assertEqual(state.state, "REACQUIRE")
        # ARK-010: a collapse must never automatically force a lower LR.
        self.assertAlmostEqual(decision.lr_multiplier, 1.0)
        self.assertIn("protected_family_below_threshold", decision.reason)

    def test_single_dipping_evaluation_does_not_hand_over(self) -> None:
        config = controller_config()
        state, update = self.qualified_two_family_state(config)
        state, _ = transition(
            state, metrics({"f1": 0.3, "f2": 0.95}, at_update=update + 1,
                           displacement=0.1),
            config, current_update=update + 1)
        self.assertEqual(state.state, "EXPAND")
        # Recovery above the exit threshold resets the pending collapse.
        state, _ = transition(
            state, metrics({"f1": 0.95, "f2": 0.95}, at_update=update + 2,
                           displacement=0.1),
            config, current_update=update + 2)
        self.assertEqual(state.state, "EXPAND")
        self.assertEqual(state.pending_collapse, {})

    def test_reacquire_exit_requires_sustained_recovery(self) -> None:
        config = controller_config()
        state, update = self.qualified_two_family_state(config)
        state, _ = transition(
            state, metrics({"f1": 0.3, "f2": 0.95}, at_update=update + 1,
                           displacement=0.1),
            config, current_update=update + 1)
        state, _ = transition(
            state, metrics({"f1": 0.3, "f2": 0.95}, at_update=update + 2,
                           displacement=0.1),
            config, current_update=update + 2)
        self.assertEqual(state.state, "REACQUIRE")
        # One recovered evaluation is not enough.
        state, decision = transition(
            state, metrics({"f1": 0.95, "f2": 0.95}, at_update=update + 3,
                           displacement=0.1),
            config, current_update=update + 3)
        self.assertEqual(state.state, "REACQUIRE")
        self.assertEqual(state.consecutive_recovery_passes, 1)
        # Second consecutive recovered evaluation confirms sustained recovery.
        state, decision = transition(
            state, metrics({"f1": 0.95, "f2": 0.95}, at_update=update + 4,
                           displacement=0.1),
            config, current_update=update + 4)
        self.assertEqual(state.state, "STABILIZE")
        self.assertIn("protected_families_recovered", decision.reason)

    def test_missing_protected_metric_holds(self) -> None:
        config = controller_config()
        state, update = self.qualified_two_family_state(config)
        state, decision = transition(
            state, metrics({"f2": 0.95}, at_update=update + 1, displacement=0.1),
            config, current_update=update + 1)
        self.assertEqual(state.state, "HOLD")
        self.assertTrue(decision.pause_updates)
        self.assertIn("missing_protected_metric", decision.reason)

    def test_overall_mean_cannot_override_protected_failure(self) -> None:
        config = controller_config()
        state, update = self.qualified_two_family_state(config)
        # Mean across families is high, but the protected family collapsed.
        state, decision = transition(
            state, metrics({"f1": 0.1, "f2": 0.99, "filler": 0.99},
                           at_update=update + 1, displacement=0.1),
            config, current_update=update + 1)
        state, decision = transition(
            state, metrics({"f1": 0.1, "f2": 0.99, "filler": 0.99},
                           at_update=update + 2, displacement=0.1),
            config, current_update=update + 2)
        self.assertEqual(state.state, "REACQUIRE")

    def test_unmeasured_families_are_not_protected(self) -> None:
        """A family the controller never qualified cannot trigger REACQUIRE."""
        config = controller_config()
        state, update = self.qualified_two_family_state(config)
        state, decision = transition(
            state, metrics({"f1": 0.95, "f2": 0.95, "stranger": 0.0},
                           at_update=update + 1, displacement=0.1),
            config, current_update=update + 1)
        self.assertEqual(state.state, "EXPAND")
        self.assertNotIn("stranger", state.protected_families)
        self.assertEqual(state.pending_collapse, {})


class StalenessAndCooldownTests(unittest.TestCase):
    def test_stale_metrics_cannot_count_as_passes(self) -> None:
        config = controller_config()
        state, _ = transition(ControllerState(),
                              metrics({"f_new": 0.2, "f_old": 0.95}), config,
                              current_update=100, family_request="f_new")
        self.assertEqual(state.last_metrics_update, 100)
        # Metrics observed at update 50 are far older than the freshness bound.
        state, decision = transition(
            state, metrics({"f_new": 0.95, "f_old": 0.95}, at_update=50,
                           displacement=0.1),
            config, current_update=110)
        self.assertEqual(decision.reason, "stale_metrics_ignored")
        self.assertEqual(state.consecutive_passes, 0)
        self.assertEqual(state.last_metrics_update, 100)

    def test_cooldown_blocks_transitions_then_allows(self) -> None:
        config = controller_config(cooldown_updates=50)
        state, _ = transition(ControllerState(),
                              metrics({"f1": 0.2, "f_old": 0.95}), config,
                              current_update=0, family_request="f1")
        self.assertEqual(state.state, "EXPAND")
        self.assertEqual(state.at_update, 0)
        for index in range(3):
            state, _ = transition(
                state, metrics({"f1": 0.95, "f_old": 0.95}, at_update=1 + index,
                               displacement=0.1),
                config, current_update=1 + index)
        self.assertEqual(state.consecutive_passes, 3)
        # Window met at update 3 but cooldown (50) has not elapsed.
        state, decision = transition(
            state, metrics({"f1": 0.95, "f_old": 0.95}, at_update=4, displacement=0.1),
            config, current_update=4)
        self.assertEqual(state.state, "EXPAND")
        self.assertEqual(decision.reason, "cooldown_stabilize")
        state, decision = transition(
            state, metrics({"f1": 0.95, "f_old": 0.95}, at_update=60, displacement=0.1),
            config, current_update=60)
        self.assertEqual(state.state, "STABILIZE")

    def test_resume_mid_window_keeps_counts(self) -> None:
        config = controller_config()
        state, _ = transition(ControllerState(),
                              metrics({"f_new": 0.2, "f_old": 0.95}), config,
                              current_update=100, family_request="f_new")
        state, _ = transition(
            state, metrics({"f_new": 0.95, "f_old": 0.95}, at_update=110,
                           displacement=0.1),
            config, current_update=110)
        self.assertEqual(state.consecutive_passes, 1)
        restored = ControllerState.from_dict(state.to_dict())
        self.assertEqual(restored.consecutive_passes, 1)
        restored, _ = transition(
            restored, metrics({"f_new": 0.95, "f_old": 0.95}, at_update=111,
                              displacement=0.1),
            config, current_update=111)
        self.assertEqual(restored.consecutive_passes, 2)


class TransitionBudgetTests(unittest.TestCase):
    def test_exhausted_budget_records_proposal_without_application(self) -> None:
        config = controller_config(max_transitions=1)
        state, _ = transition(ControllerState(),
                              metrics({"f1": 0.2, "f_old": 0.95}), config,
                              current_update=0, family_request="f1")
        self.assertEqual(state.state, "EXPAND")
        self.assertEqual(state.transitions, 1)
        for index in range(3):
            state, _ = transition(
                state, metrics({"f1": 0.95, "f_old": 0.95}, at_update=100 + index,
                               displacement=0.1),
                config, current_update=100 + index)
        state, decision = transition(
            state, metrics({"f1": 0.95, "f_old": 0.95}, at_update=300, displacement=0.1),
            config, current_update=300)
        self.assertEqual(state.state, "EXPAND")  # unchanged
        self.assertTrue(decision.proposed_only)
        self.assertEqual(decision.reason, "proposed_stabilize")
        # The proposal is visible in history: a no-event regime is never silent.
        self.assertEqual(state.history[-1].status, "proposed")
        self.assertEqual(state.history[-1].reason, "proposed_stabilize")

    def test_history_persists_and_is_bounded(self) -> None:
        config = controller_config()
        state = ControllerState()
        for index in range(6):
            request = f"f{index}"
            state, _ = transition(state,
                                  metrics({request: 0.2, "f_old": 0.95}), config,
                                  current_update=index * 100, family_request=request)
        self.assertGreater(len(state.history), 0)
        serialized = state.to_dict()
        self.assertLessEqual(len(serialized["history"]), 64)
        restored = ControllerState.from_dict(serialized)
        self.assertEqual(restored.history[-1].reason, state.history[-1].reason)


class HoldResumeTests(unittest.TestCase):
    def test_hold_released_by_valid_metrics(self) -> None:
        config = controller_config()
        state = ControllerState()
        bad = ControllerMetrics(pool_id="controller-pool-a", at_update=10,
                                family_scores={"f": 0.5}, nonfinite_observed=True)
        state, decision = transition(state, bad, config, current_update=10)
        self.assertEqual(state.state, "HOLD")
        # While metrics are missing, the pause persists.
        state, decision = transition(state, None, config, current_update=11)
        self.assertEqual(state.state, "HOLD")
        self.assertTrue(decision.pause_updates)
        state, decision = transition(
            state, metrics({"f1": 0.9}, at_update=12, displacement=0.1), config,
            current_update=12)
        self.assertEqual(state.state, "FORM")
        self.assertFalse(decision.pause_updates)
        self.assertIn("hold_released", decision.reason)


if __name__ == "__main__":
    unittest.main()
