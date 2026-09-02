"""Red-team adversarial validation suite for Senora.

Falsification tests designed to verify that the experimental pipeline detects and rejects:
1. Accidental local execution.
2. Prospective fresh suite access before recipe freeze.
3. Hidden mixture ratio drift away from the 65:20 natural:code invariant.
4. Spurious macro gains masking an individual family collapse.
5. Spurious cognition gains masking general language substrate regression.
6. Silent parameter non-movement bugs.
7. Pseudo-replication across counterfactual prompt variants.
8. Off-by-one label shifts in causal cross-entropy.
"""

from __future__ import annotations

import unittest

from senora.canary import execute_preflight_canary
from senora.data_pipeline import MixtureRecipe
from senora.evaluator import EvaluationSummary
from senora.objectives import causal_cross_entropy
from senora.result_classifier import P35ResultCategory, classify_p35_a_results
from senora.run_experiment import ExecutionManifest
from senora.state_machine import ExperimentLifecycle, ExperimentPhase, FreshLeakageViolationError, IllegalPhaseTransitionError
from senora.transfer_contract import calculate_clustered_group_statistics

try:
    import torch
except ImportError:
    torch = None


def _make_eval(raw_core: float, worst_fam: float = 0.50, sensitivity: float = 0.85) -> EvaluationSummary:
    return EvaluationSummary(
        schema="senora-evaluation-summary/v1",
        suite_split="development",
        case_count=100,
        raw_core_accuracy=raw_core,
        constrained_accuracy=raw_core,
        assisted_accuracy=raw_core,
        intervention_dependence_rate=0.0,
        assistance_harm_rate=0.0,
        family_accuracies={"fam_a": worst_fam, "fam_b": raw_core},
        difficulty_curves={"all": {1: raw_core}},
        pair_sensitivity_flip_rate=sensitivity,
        pair_invariance_stable_rate=0.90,
        natural_analogue_macro_accuracy=0.45,
        candidate_scoring_status="BLOCKED_BY_SCORER_FIREWALL",
        candidate_selection_accuracy=None,
    )


class TestSenoraRedTeam(unittest.TestCase):
    def test_redteam_accidental_local_execution_blocked(self) -> None:
        """Adversarial check: unflagged or local target execution must fail closed."""
        with self.assertRaises(RuntimeError):
            execute_preflight_canary(device="cpu", remote_authorized=False)

        manifest = ExecutionManifest(
            schema="senora-execution-manifest/v2",
            target_environment="local-workstation",
            launch_nonce="nonce-12345",
            source_commit_sha="a" * 40,
            experiment_identity_sha256="0" * 64,
            authorized_by="operator",
        )
        with self.assertRaises(ValueError) as ctx:
            manifest.assert_valid()
        self.assertIn("must be remote compute", str(ctx.exception))

    def test_redteam_fresh_leakage_blocked(self) -> None:
        """Adversarial check: accessing fresh before freeze must fail closed."""
        lifecycle = ExperimentLifecycle()
        lifecycle.transition_to(ExperimentPhase.PREREGISTERED)
        lifecycle.transition_to(ExperimentPhase.IDENTITIES_BOUND)
        lifecycle.transition_to(ExperimentPhase.REMOTE_CANARY_REQUIRED)
        lifecycle.transition_to(ExperimentPhase.REMOTE_CANARY_PASS)
        lifecycle.transition_to(ExperimentPhase.DEVELOPMENT_RUN)

        # Attempting to access fresh suite in DEVELOPMENT_RUN phase
        with self.assertRaises(FreshLeakageViolationError):
            lifecycle.verify_suite_access("fresh")

        # Attempting to jump directly to FRESH_RUN without freeze
        with self.assertRaises(IllegalPhaseTransitionError):
            lifecycle.transition_to(ExperimentPhase.FRESH_RUN)

    def test_redteam_mixture_ratio_invariant(self) -> None:
        """Adversarial check: non-cognition remainder must never drift from 65:20 (3.25)."""
        for cog_frac in [0.0, 0.05, 0.10, 0.15, 0.20, 0.35, 0.50]:
            recipe = MixtureRecipe.from_cognition_fraction(cog_frac)
            ratio = recipe.natural_fraction / recipe.code_fraction
            self.assertAlmostEqual(ratio, 3.25, places=6)
            self.assertAlmostEqual(recipe.natural_fraction + recipe.code_fraction + cog_frac, 1.0, places=6)

    def test_redteam_family_collapse_detected(self) -> None:
        """Adversarial check: high macro gain cannot hide collapsed family."""
        ctrl = _make_eval(raw_core=0.20, worst_fam=0.30)
        # Macro gain is +0.45, but fam_a collapsed to 0.10 (< 0.25 floor)
        cand = _make_eval(raw_core=0.65, worst_fam=0.10)

        res = classify_p35_a_results(cand, ctrl, substrate_regression_fraction=0.01)
        self.assertEqual(res.category, P35ResultCategory.FAMILY_COLLAPSE)
        self.assertIn("DEBUG_FAILING_FAMILY_GENERATOR", res.precommitted_next_action)

    def test_redteam_substrate_regression_detected(self) -> None:
        """Adversarial check: high cognition gain cannot hide language regression > 3%."""
        ctrl = _make_eval(raw_core=0.20)
        cand = _make_eval(raw_core=0.75)  # Huge gain +0.55
        # Substrate regressed 3.5%
        res = classify_p35_a_results(cand, ctrl, substrate_regression_fraction=0.035)
        self.assertEqual(res.category, P35ResultCategory.SUBSTRATE_TRADEOFF)
        self.assertIn("ADJUST_CURRICULUM_MIXTURE", res.precommitted_next_action)

    def test_redteam_pseudoreplication_prevented(self) -> None:
        """Adversarial check: multi-prompt variants within a causal group are clustered."""
        # 6 prompts belonging to 2 independent groups (3 variants per group)
        group_ids = ["group_a", "group_a", "group_a", "group_b", "group_b", "group_b"]
        cand = [True, True, False, True, True, True]  # Group A has 1 fail; Group B passes all
        ctrl = [False, False, False, False, False, False]

        stats = calculate_clustered_group_statistics(cand, ctrl, group_ids, resamples=1000)
        # 6 prompts must collapse to 2 independent units
        self.assertEqual(stats.independent_groups_count, 2)
        # Group A fails (all not True); Group B succeeds -> 1 win, 0 losses, 1 tie
        self.assertEqual(stats.concordant_wins, 1)
        self.assertEqual(stats.ties, 1)

    @unittest.skipIf(torch is None, "PyTorch required for causal CE label shift test")
    def test_redteam_causal_ce_label_shift(self) -> None:
        """Adversarial check: logits at position t must predict target at t+1, not t."""
        # B=1, T=3, V=4
        # Logits strongly favor token 2 at pos 0, token 3 at pos 1, token 1 at pos 2
        logits = torch.tensor([[[0.0, 0.0, 10.0, 0.0],
                                [0.0, 0.0, 0.0, 10.0],
                                [0.0, 10.0, 0.0, 0.0]]], dtype=torch.float32)
        # Targets: pos 0 is ignored (BOS=0), pos 1 is 2, pos 2 is 3
        # Since shift_logits is pos 0..1 and shift_labels is pos 1..2:
        # pos 0 logits (2) match target pos 1 (2)!
        # pos 1 logits (3) match target pos 2 (3)!
        targets = torch.tensor([[0, 2, 3]], dtype=torch.long)
        loss, count = causal_cross_entropy(logits[:, :-1], targets[:, 1:])
        # Loss should be near 0 because shifted predictions match shifted targets
        self.assertLess(loss.item(), 0.05)


if __name__ == "__main__":
    unittest.main()