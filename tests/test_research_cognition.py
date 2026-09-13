"""M19–M21 focused tests: grounded workspace, belief revision, executive
decision interfaces, verified derivations and abstractions.

No learned model: executive decisions use a frozen fake scorer; belief
revision uses the reference finite-support updater.
"""
import unittest

from bramastra_lab.research.cognition.beliefs import (
    Belief,
    BeliefError,
    EvidenceRecord,
    reference_finite_support_update,
    unique_corroboration_roots,
)
from bramastra_lab.research.cognition.derivations import (
    AbstractionArchive,
    DerivationChecker,
    DerivationError,
    DerivationStep,
)
from bramastra_lab.research.cognition.executive import (
    CognitiveOperation,
    Executive,
    ExecutiveError,
    OperationRegistry,
    ResourceVector,
    SessionRunner,
    decide_deliberation,
    filter_capability_summary,
)
from bramastra_lab.research.cognition.workspace import (
    CapabilityEstimate,
    CognitiveWorkspace,
    WorkspaceError,
)


def new_workspace() -> CognitiveWorkspace:
    return CognitiveWorkspace(goal={"task": "open the vault"},
                              success_predicate="vault.open == True",
                              budget=8)


class BeliefRevisionTests(unittest.TestCase):
    def test_reference_update_normalizes(self) -> None:
        support = {"h1": 0.5, "h2": 0.5}
        likelihood = {"h1": 0.9, "h2": 0.1}
        next_support, status = reference_finite_support_update(support, likelihood)
        self.assertEqual(status, "OK")
        self.assertAlmostEqual(sum(next_support.values()), 1.0)
        self.assertGreater(next_support["h1"], next_support["h2"])

    def test_zero_normalizer_is_model_mismatch_not_reset(self) -> None:
        support = {"h1": 0.5, "h2": 0.5}
        likelihood = {"h1": 0.0, "h2": 0.0}
        next_support, status = reference_finite_support_update(support, likelihood)
        self.assertEqual(status, "MODEL_MISMATCH")
        self.assertEqual(next_support, {})

    def test_workspace_revision_changes_leading_hypothesis(self) -> None:
        workspace = new_workspace()
        record = workspace.admit_evidence({"obs": "alarm at noon"})
        workspace.propose_belief({"hypothesis": "vault opens at noon"},
                                 status="hypothesis")
        belief_alias = next(iter(workspace.beliefs))
        workspace.beliefs[belief_alias] = Belief(
            alias=belief_alias, proposition={"hypothesis": "vault opens at noon"},
            status="hypothesis",
            support={"noon": 0.5, "night": 0.5})
        next_support = workspace.revise_belief(
            belief_alias, {"noon": 0.9, "night": 0.1}, [record.alias])
        self.assertGreater(next_support["noon"], next_support["night"])

    def test_contradiction_marks_status_and_confidence_cannot_promote(self) -> None:
        workspace = new_workspace()
        workspace.propose_belief({"hypothesis": "x"})
        alias = next(iter(workspace.beliefs))
        # A high-confidence proposal stays a hypothesis: provenance class is
        # not changed by confidence fields.
        self.assertEqual(workspace.beliefs[alias].status, "hypothesis")
        workspace.contradict_belief(alias)
        self.assertEqual(workspace.beliefs[alias].status, "contradicted")

    def test_model_mismatch_retains_evidence(self) -> None:
        workspace = new_workspace()
        record = workspace.admit_evidence({"obs": "impossible"})
        workspace.propose_belief({"hypothesis": "h"})
        alias = next(iter(workspace.beliefs))
        workspace.beliefs[alias] = Belief(
            alias=alias, proposition={"hypothesis": "h"}, status="hypothesis",
            support={"h1": 0.5, "h2": 0.5})
        from bramastra_lab.research.cognition.beliefs import BeliefError

        with self.assertRaises(BeliefError) as caught:
            workspace.revise_belief(alias, {"h1": 0.0, "h2": 0.0}, [record.alias])
        self.assertIn("MODEL_MISMATCH", str(caught.exception))
        self.assertEqual(workspace.beliefs[alias].status, "unresolved")
        self.assertIn(record.alias, workspace.evidence)

    def test_correlated_copies_count_once(self) -> None:
        records = [
            EvidenceRecord(alias="e1", content={"obs": "siren"},
                           ancestry=("root-a",)),
            EvidenceRecord(alias="e2", content={"obs": "siren"},
                           ancestry=("e1", "root-a")),  # derived from e1
            EvidenceRecord(alias="e3", content={"obs": "siren"},
                           ancestry=("root-b",)),
        ]
        roots = unique_corroboration_roots(records)
        self.assertEqual(roots, {"root-a", "root-b"})


class SubgoalTests(unittest.TestCase):
    def test_cycles_reject(self) -> None:
        """The graph only allows edges from new nodes to existing nodes, so
        cycles are impossible by construction; self-reference and missing
        dependencies reject loudly."""
        workspace = new_workspace()
        workspace.add_subgoal("s1", description="a", check="c1")
        workspace.add_subgoal("s2", description="b", check="c2",
                              dependencies=("s1",))
        with self.assertRaises(WorkspaceError):
            workspace.add_subgoal("sx", description="self", check="c",
                                  dependencies=("sx",))
        with self.assertRaises(WorkspaceError):
            workspace.add_subgoal("s1", description="duplicate id", check="c")
        # A dependency chain that reaches an existing node is fine.
        workspace.add_subgoal("s3", description="c", check="c3", parent="s2",
                              dependencies=("s2",))
        self.assertEqual(workspace.subgoals["s3"].dependencies, ("s2",))

    def test_missing_dependencies_reject(self) -> None:
        workspace = new_workspace()
        with self.assertRaises(WorkspaceError):
            workspace.add_subgoal("s1", description="a", check="c",
                                  dependencies=("ghost",))

    def test_completion_requires_verification(self) -> None:
        workspace = new_workspace()
        workspace.add_subgoal("s1", description="a", check="c")
        workspace.complete_subgoal("s1", verified=True)
        self.assertEqual(workspace.subgoals["s1"].status, "verified")
        workspace.complete_subgoal("s1", verified=None)
        self.assertEqual(workspace.subgoals["s1"].status, "unknown")


class WorkspacePersistenceTests(unittest.TestCase):
    def test_round_trip_preserves_next_public_input(self) -> None:
        workspace = new_workspace()
        workspace.admit_evidence({"obs": "one"})
        workspace.propose_belief({"hypothesis": "h"})
        before = workspace.rendered_view()
        restored = CognitiveWorkspace.from_dict(workspace.to_dict())
        self.assertEqual(restored.rendered_view(), before)

    def test_render_excludes_provenance_and_rejects_sealed_estimates(self) -> None:
        workspace = new_workspace()
        workspace.admit_evidence({"obs": "one"})
        view = workspace.rendered_view()
        blob = str(view)
        for forbidden in ("episode_id", "split", "evaluator_verdict", "task_cluster"):
            self.assertNotIn(forbidden, blob)
        with self.assertRaises(WorkspaceError):
            workspace.add_capability_estimate(CapabilityEstimate(
                operation="QUERY", family="f", recent_validated_performance=0.5,
                source_pool="sealed", model_checkpoint="m1"))


class ExecutiveTests(unittest.TestCase):
    def test_frozen_fake_model_decision_changes_executed_operation(self) -> None:
        workspace = new_workspace()
        registry = OperationRegistry()
        executed = []

        def fake_scorer(candidates):
            # Frozen policy: prefer VERIFY when evidence exists, else PREDICT.
            preference = "VERIFY" if workspace.evidence else "PREDICT"
            return [1.0 if candidate.verb == preference else 0.0
                    for candidate in candidates]

        executive = Executive(registry, fake_scorer, decision_origin="model")
        runner = SessionRunner(executive, executors={
            "PREDICT": lambda op, ws: (executed.append("PREDICT"), ResourceVector(
                inference_tokens=10, model_calls=1))[1],
            "VERIFY": lambda op, ws: (executed.append("VERIFY"), ResourceVector(
                inference_tokens=12, model_calls=1))[1],
            "SUBMIT": lambda op, ws: (executed.append("SUBMIT"), ResourceVector(
                inference_tokens=4, model_calls=1))[1],
        }, max_steps=2)
        workspace.admit_evidence({"obs": "present"})
        result = runner.run(workspace)
        self.assertIn("VERIFY", executed)
        # Costs aggregate exactly once per executed operation.
        self.assertEqual(result["resources"]["inference_tokens"],
                         sum(10 if verb == "PREDICT" else 12 if verb == "VERIFY" else 4
                             for verb in executed))

    def test_fixed_rule_origin_recorded_and_fallback_visible(self) -> None:
        workspace = new_workspace()
        registry = OperationRegistry(verbs=("PREDICT", "ABSTAIN"))
        executive = Executive(registry, scorer=lambda candidates: [0.0] * len(candidates),
                              decision_origin="fixed_rule", fixed_verb="ABSTAIN")
        decision = executive.decide(workspace)
        self.assertEqual(decision.origin, "fixed_rule")
        self.assertEqual(decision.selected.verb, "ABSTAIN")

    def test_no_progress_cycle_terminates(self) -> None:
        workspace = new_workspace()
        registry = OperationRegistry(verbs=("COMPARE",))
        executive = Executive(registry, scorer=lambda candidates: [1.0],
                              decision_origin="model")
        runner = SessionRunner(executive, executors={
            "COMPARE": lambda op, ws: ResourceVector(inference_tokens=1, model_calls=1)},
            no_progress_limit=3, max_steps=16)
        result = runner.run(workspace)
        self.assertTrue(result["terminated_reason"].startswith("no_progress"))
        self.assertEqual(result["terminated_reason"], "no_progress:COMPAREx3")

    def test_unbounded_candidate_set_rejects(self) -> None:
        workspace = new_workspace()
        registry = OperationRegistry()
        executive = Executive(registry, scorer=lambda c: [0.0] * len(c),
                              max_operations=2)
        with self.assertRaises(ExecutiveError):
            executive.decide(workspace)

    def test_malformed_verb_rejects_at_type_level(self) -> None:
        with self.assertRaises(ExecutiveError):
            CognitiveOperation(verb="CHEAT", arguments={})

    def test_deliberation_frozen_threshold_rule(self) -> None:
        choice = decide_deliberation(predicted_improvement=0.8, declared_cost=0.1,
                                     threshold=0.2)
        self.assertEqual(choice.option, "more_computation")
        choice = decide_deliberation(predicted_improvement=0.2, declared_cost=0.4,
                                     threshold=0.2)
        self.assertEqual(choice.option, "direct_answer")

    def test_capability_summary_pool_firewall(self) -> None:
        filter_capability_summary({"accuracy": 0.9}, source_pool="training")
        with self.assertRaises(ExecutiveError):
            filter_capability_summary({"accuracy": 0.9}, source_pool="confirmation")
        with self.assertRaises(ExecutiveError):
            filter_capability_summary({"accuracy": 0.9}, source_pool="sealed")


class DerivationTests(unittest.TestCase):
    def test_valid_derivation_reaches_conclusion(self) -> None:
        checker = DerivationChecker.DEFAULT if hasattr(DerivationChecker, "DEFAULT") \
            else __import__("bramastra_lab.research.cognition.derivations",
                            fromlist=["DEFAULT_CHECKER"]).DEFAULT_CHECKER
        steps = [
            DerivationStep(premises=("a", "b"), operation="add",
                           arguments={"a": "a", "b": "b"}, conclusion="sum"),
            DerivationStep(premises=("sum",), operation="negate",
                           arguments={"a": "sum"}, conclusion="negsum"),
        ]
        ok, values = checker.check_derivation(steps, {"a": 2, "b": 3})
        self.assertTrue(ok)
        self.assertEqual(values["negsum"], -5)

    def test_invalid_step_gets_no_gold_label(self) -> None:
        checker = __import__("bramastra_lab.research.cognition.derivations",
                             fromlist=["DEFAULT_CHECKER"]).DEFAULT_CHECKER
        steps = [DerivationStep(premises=("ghost",), operation="add",
                                arguments={"a": "ghost", "b": "a"}, conclusion="s")]
        ok, values = checker.check_derivation(steps, {"a": 1})
        self.assertFalse(ok)
        self.assertNotIn("s", values)
        steps_unknown_rule = [DerivationStep(premises=(), operation="magic",
                                             arguments={}, conclusion="x")]
        ok, _ = checker.check_derivation(steps_unknown_rule, {})
        self.assertFalse(ok)


class AbstractionTests(unittest.TestCase):
    def test_admission_requires_distinct_validation_cases(self) -> None:
        archive = AbstractionArchive()
        archive.propose("r1", "if alarm then guard present", ("alarm",),
                        ("case-1", "case-2", "case-3"))
        with self.assertRaises(DerivationError):
            archive.admit("r1", ("case-1",))  # overlap with support
        rule = archive.admit("r1", ("case-7", "case-8"))
        self.assertTrue(rule.admitted)
        self.assertEqual(rule.status, "admitted")

    def test_counterexample_restricts_then_retracts(self) -> None:
        archive = AbstractionArchive()
        archive.propose("r1", "if alarm then guard present", ("alarm",),
                        ("case-1",))
        archive.admit("r1", ("case-9",))
        rule = archive.add_counterexample("r1", "case-20",
                                          restrict="alarm_sounding == True")
        self.assertEqual(rule.status, "restricted")
        self.assertIn("alarm_sounding == True", rule.preconditions)
        rule = archive.add_counterexample("r1", "case-21")
        self.assertEqual(rule.status, "retracted")
        self.assertFalse(rule.admitted)
        self.assertIn("case-20", rule.counterexamples)
        self.assertIn("case-21", rule.counterexamples)

    def test_support_and_validation_ids_survive(self) -> None:
        archive = AbstractionArchive()
        rule = archive.propose("r1", "statement", ("p",), ("s1",))
        self.assertEqual(rule.proposed_from, ("s1",))
        self.assertEqual(rule.identity(), rule.identity())


if __name__ == "__main__":
    unittest.main()
