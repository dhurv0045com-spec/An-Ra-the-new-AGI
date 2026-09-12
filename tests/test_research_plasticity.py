"""B06 focused tests: plasticity-controller state machine.

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
           "expand_lr_multiplier": 1.0, "reacquire_lr_multiplier": 1.0}
    raw.update(overrides)
    return ControllerSection.from_dict(raw)


def metrics(family_scores, at_update=100, displacement=0.5, pool="controller-pool-a",
            improvements=None):
    return ControllerMetrics(pool_id=pool, at_update=at_update,
                             family_scores=dict(family_scores),
                             parameter_displacement=displacement,
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
    def test_expand_sets_acquiring_family_and_protects_existing(self) -> None:
        state = ControllerState()
        next_state, decision = transition(
            state, metrics({"f_new": 0.2, "f_old": 0.95}), controller_config(),
            current_update=100, family_request="f_new")
        self.assertEqual(next_state.state, "EXPAND")
        self.assertEqual(next_state.acquiring_family, "f_new")
        self.assertIn("f_old", next_state.protected_families)
        self.assertEqual(decision.lr_multiplier, 1.0)
        self.assertTrue(decision.checkpoint_requested)
        self.assertEqual(decision.replay_mixture,
                         {"acquisition": 0.75, "protected_replay": 0.25})

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
    def trained_state(self, config) -> tuple[ControllerState, int]:
        state, _ = transition(ControllerState(),
                              metrics({"f1": 0.2, "f_old": 0.95}), config,
                              current_update=0, family_request="f1")
        update = 20
        for index in range(3):
            state, _ = transition(
                state, metrics({"f1": 0.95, "f_old": 0.95}, at_update=update,
                               displacement=0.1),
                config, current_update=update)
            update += 20
        self.assertEqual(state.state, "STABILIZE")
        return state, update

    def test_protected_collapse_triggers_reacquire_without_lowering_lr(self) -> None:
        config = controller_config()
        state, update = self.trained_state(config)
        state, decision = transition(
            state, metrics({"f1": 0.95, "f_old": 0.3}, at_update=update + 20,
                           displacement=0.1),
            config, current_update=update + 20)
        self.assertEqual(state.state, "REACQUIRE")
        # ARK-010: a collapse must never automatically force a lower LR.
        self.assertAlmostEqual(decision.lr_multiplier, 1.0)
        self.assertIn("protected_family_below_threshold", decision.reason)

    def test_reacquire_exits_with_hysteresis(self) -> None:
        config = controller_config()
        state, update = self.trained_state(config)
        state, _ = transition(state,
                              metrics({"f1": 0.95, "f_old": 0.3}, at_update=update + 20,
                                      displacement=0.1),
                              config, current_update=update + 20)
        self.assertEqual(state.state, "REACQUIRE")
        # Still below the exit threshold: remain.
        state, decision = transition(
            state, metrics({"f1": 0.95, "f_old": 0.5}, at_update=update + 40,
                           displacement=0.1),
            config, current_update=update + 40)
        self.assertEqual(state.state, "REACQUIRE")
        # At the exit threshold (= enter + hysteresis) it recovers.
        state, decision = transition(
            state, metrics({"f1": 0.95, "f_old": 0.65}, at_update=update + 60,
                           displacement=0.1),
            config, current_update=update + 60)
        self.assertEqual(state.state, "STABILIZE")
        self.assertIn("protected_families_recovered", decision.reason)

    def test_missing_protected_metric_holds(self) -> None:
        config = controller_config()
        state, update = self.trained_state(config)
        state, decision = transition(
            state, metrics({"f1": 0.95}, at_update=update + 20, displacement=0.1),
            config, current_update=update + 20)
        self.assertEqual(state.state, "HOLD")
        self.assertTrue(decision.pause_updates)
        self.assertIn("missing_protected_metric", decision.reason)

    def test_overall_mean_cannot_override_protected_failure(self) -> None:
        config = controller_config()
        state, update = self.trained_state(config)
        # Mean across families is high, but the protected family collapsed.
        state, decision = transition(
            state, metrics({"f1": 0.98, "f_old": 0.1, "filler": 0.99},
                           at_update=update + 20, displacement=0.1),
            config, current_update=update + 20)
        self.assertEqual(state.state, "REACQUIRE")


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
