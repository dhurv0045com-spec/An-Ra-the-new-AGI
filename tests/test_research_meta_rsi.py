"""M22–M24 focused tests: meta-episodes, curves, comparisons, typed method
language, origin validation and the fixture generation chain.

No training, no learned execution: dispatch uses deterministic callbacks and
the generation chain uses the fixture registry.
"""
import unittest

from bramastra_lab.research.metalearning.episodes import (
    AdaptationCurve,
    Attribution,
    MetaError,
    MetaEpisode,
    compare_methods,
    dispatch_meta_episode,
)
from bramastra_lab.research.metalearning.method_language import (
    GradientTransform,
    MethodError,
    MethodProgram,
    ObjectiveCoefficients,
    ReplayWeights,
    ScheduleExpression,
    SchedulePoint,
    compile_method,
)
from bramastra_lab.research.metalearning.generations import (
    GenerationRegistry,
    GenerationReceipt,
    OriginError,
    classify_external_submission,
    parse_model_output,
    run_fixture_generation,
    validate_origin,
)


def episode(**overrides):
    values = dict(episode_id="m-1", family="arithmetic",
                  mechanism_cluster="carry-v1",
                  support_case_ids=("s1", "s2"), query_case_ids=("q1", "q2"),
                  retention_case_ids=("r1",), adaptation_allowance=100)
    values.update(overrides)
    return MetaEpisode(**values)


class MetaEpisodeTests(unittest.TestCase):
    def test_support_query_leakage_rejects(self) -> None:
        with self.assertRaises(MetaError):
            episode(query_case_ids=("s1", "q1"))

    def test_identity_and_grid(self) -> None:
        self.assertEqual(episode().identity(), episode().identity())
        with self.assertRaises(MetaError):
            episode(adaptation_allowance=0)


class CurveTests(unittest.TestCase):
    def test_grid_and_missing_points_are_conservative(self) -> None:
        curve = AdaptationCurve(grid=(10, 20, 40))
        curve.record(10, 0.5)
        curve.record(20, 0.9)
        # Missing late measurement scores 0.0, never omitted favorably.
        self.assertEqual(curve.filled(), {10: 0.5, 20: 0.9, 40: 0.0})
        self.assertAlmostEqual(curve.auc(), (0.5 + 0.9 + 0.0) / 3)

    def test_off_grid_and_ordering_reject(self) -> None:
        curve = AdaptationCurve(grid=(10, 20))
        with self.assertRaises(MetaError):
            curve.record(15, 0.5)
        with self.assertRaises(MetaError):
            AdaptationCurve(grid=(20, 10))

    def test_dispatch_uses_deterministic_callback(self) -> None:
        seen = []

        def callback(ep, allowance):
            seen.append((ep.episode_id, allowance))
            curve = AdaptationCurve(grid=(10, 20))
            curve.record(10, 0.4)
            return curve

        curve = dispatch_meta_episode(episode(), callback)
        self.assertEqual(seen, [("m-1", 100)])
        # Grid (10, 20) with only 10 measured: the missing late point is
        # conservative 0.0 -> AUC (0.4 + 0.0) / 2.
        self.assertAlmostEqual(curve.auc(), 0.2)


class ComparisonTests(unittest.TestCase):
    def test_method_comparison_requires_identical_start(self) -> None:
        parent = AdaptationCurve(grid=(10, 20))
        parent.record(10, 0.2)
        parent.record(20, 0.3)
        candidate = AdaptationCurve(grid=(10, 20))
        candidate.record(10, 0.6)
        candidate.record(20, 0.8)
        with self.assertRaises(MetaError):
            compare_methods(parent, candidate, parent_method={"v": 1},
                            candidate_method={"v": 2},
                            starting_state_identity="child",
                            parent_starting_state_identity="parent")
        report = compare_methods(parent, candidate, parent_method={"v": 1},
                                 candidate_method={"v": 2},
                                 starting_state_identity="same",
                                 parent_starting_state_identity="same",
                                 failed_candidate_cost=3.0)
        self.assertAlmostEqual(report["auc_delta"], 0.45)
        self.assertTrue(report["total_cost_includes_failures"])

    def test_attribution_levels(self) -> None:
        Attribution(level="R0", proposal_origin="human", trainer_identity="t",
                    data_identity="d", checkpoint_identity="c")
        with self.assertRaises(MetaError):
            Attribution(level="R3", proposal_origin="external_agent",
                        trainer_identity="t", data_identity="d",
                        checkpoint_identity="c")
        Attribution(level="R3", proposal_origin="model", trainer_identity="t",
                    data_identity="d", checkpoint_identity="c")


class MethodLanguageTests(unittest.TestCase):
    def test_valid_program_compiles_deterministically(self) -> None:
        program = MethodProgram(
            schedule=ScheduleExpression(
                counter="optimizer_updates",
                points=(SchedulePoint(0, 1e-3), SchedulePoint(100, 1e-4))),
            replay_weights=ReplayWeights({"arithmetic": 1.0, "logic": 2.0}),
            objective_coefficients=ObjectiveCoefficients(token=1.0, world=0.5),
            gradient_transform=GradientTransform(kind="clip_norm", bound=1.0),
            comparison_protocol="equal-allowance/v1")
        one = compile_method(program, runtime_config={"profile": "tiny"})
        two = compile_method(program, runtime_config={"profile": "tiny"})
        self.assertEqual(one["identity"], two["identity"])
        other = compile_method(program, runtime_config={"profile": "development"})
        self.assertNotEqual(one["identity"], other["identity"])

    def test_prohibited_inputs_reject(self) -> None:
        with self.assertRaises(MethodError):
            ScheduleExpression(counter="sealed_accuracy",
                               points=(SchedulePoint(0, 1.0),))
        with self.assertRaises(MethodError):
            ScheduleExpression(counter="optimizer_updates",
                               points=(SchedulePoint(0, float("nan")),))
        with self.assertRaises(MethodError):
            GradientTransform(kind="grad_scale", bound=0.5)  # no state declared
        with self.assertRaises(MethodError):
            ObjectiveCoefficients(token=float("inf"))
        with self.assertRaises(MethodError):
            MethodProgram(expected_gain=0.1, comparison_protocol="")  # no component

    def test_no_change_representable(self) -> None:
        from bramastra_lab.research.metalearning.method_language import NO_CHANGE, \
            parse_no_change

        self.assertEqual(parse_no_change(), NO_CHANGE)

    def test_parse_model_output_round_trip(self) -> None:
        import json

        program = MethodProgram(
            schedule=ScheduleExpression(counter="optimizer_updates",
                                        points=(SchedulePoint(0, 1e-3),)),
            replay_weights=ReplayWeights({"arithmetic": 1.0}),
            comparison_protocol="equal-allowance/v1")
        raw = json.dumps({**program.to_dict(), "proposal": "candidate"})
        parsed = parse_model_output(raw)
        self.assertEqual(parsed.identity(), program.identity())


class OriginValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        import json

        self.program = MethodProgram(
            schedule=ScheduleExpression(counter="optimizer_updates",
                                        points=(SchedulePoint(0, 1e-3),)),
            comparison_protocol="equal-allowance/v1")
        self.raw = json.dumps({**self.program.to_dict(), "proposal": "candidate"})
        self.registry = {"ckpt-1": "model", "ckpt-2": "external_agent"}

    def test_valid_model_origin(self) -> None:
        capture = __import__("bramastra_lab.research.metalearning.generations",
                             fromlist=["ProposalCapture"]).ProposalCapture(
            checkpoint_payload_identity="ckpt-1", rendered_input="render",
            sampling={"temperature": 0.7}, raw_output=self.raw,
            parsed_program=self.program)
        checkpoint = validate_origin(capture, reparsed_program=self.program,
                                     checkpoint_registry=self.registry)
        self.assertEqual(checkpoint, "ckpt-1")

    def test_forged_checkpoint_reference_rejects(self) -> None:
        from bramastra_lab.research.metalearning.generations import ProposalCapture

        capture = ProposalCapture(
            checkpoint_payload_identity="ghost", rendered_input="render",
            sampling={"temperature": 0.7}, raw_output=self.raw,
            parsed_program=self.program)
        with self.assertRaises(OriginError):
            validate_origin(capture, reparsed_program=self.program,
                            checkpoint_registry=self.registry)

    def test_ast_mismatch_rejects(self) -> None:
        from bramastra_lab.research.metalearning.generations import ProposalCapture

        other = MethodProgram(
            schedule=ScheduleExpression(counter="optimizer_updates",
                                        points=(SchedulePoint(0, 2e-3),)),
            comparison_protocol="equal-allowance/v1")
        capture = ProposalCapture(
            checkpoint_payload_identity="ckpt-1", rendered_input="render",
            sampling={"temperature": 0.7}, raw_output=self.raw,
            parsed_program=other)
        with self.assertRaises(OriginError):
            validate_origin(capture, reparsed_program=self.program,
                            checkpoint_registry=self.registry)

    def test_transcript_hash_cannot_qualify_external_output(self) -> None:
        result = classify_external_submission(self.raw, checkpoint_registry=self.registry)
        self.assertEqual(result, "external_assisted")


class GenerationChainTests(unittest.TestCase):
    def program(self, value) -> MethodProgram:
        return MethodProgram(
            schedule=ScheduleExpression(counter="optimizer_updates",
                                        points=(SchedulePoint(0, value),)),
            comparison_protocol="equal-allowance/v1")

    def test_three_fixture_generations_chain_with_paths(self) -> None:
        registry = GenerationRegistry()
        # Generation 1: accepted.
        g1 = run_fixture_generation("gen-1", None, "proposer-1", {"base": 0},
                                    self.program(1e-3), comparison_identity="cmp-1",
                                    confirmed=True, registry=registry)
        self.assertEqual(g1.status, "accepted")
        self.assertEqual(registry.successor_proposer("gen-1"), "proposer-1")
        # Generation 2 first attempt crashes, retry is rejected, then a
        # no-change generation continues the chain.
        crashed = run_fixture_generation("gen-2a", g1.identity(), "proposer-1",
                                         {"base": 0}, self.program(2e-3),
                                         comparison_identity="cmp-2a", confirmed=False,
                                         crash=True, registry=registry)
        self.assertEqual(crashed.status, "crashed")
        rejected = run_fixture_generation("gen-2b", g1.identity(), "proposer-1",
                                          {"base": 0}, self.program(2e-3),
                                          comparison_identity="cmp-2b", confirmed=False,
                                          registry=registry)
        self.assertEqual(rejected.status, "rejected")
        no_change = run_fixture_generation("gen-2c", g1.identity(), "proposer-1",
                                           {"base": 0}, None,
                                           comparison_identity=None, confirmed=True,
                                           registry=registry)
        self.assertEqual(no_change.status, "accepted")
        self.assertEqual(registry.successor_proposer("gen-2c"), "proposer-1")
        # All attempts retained; each successful generation records both its
        # confirmed and accepted states (content-identity state transitions).
        self.assertEqual(len(registry.receipts), 6)

    def test_fixture_cannot_publish_learned_parent(self) -> None:
        registry = GenerationRegistry()
        receipt = GenerationReceipt(
            generation_id="gen-learned", predecessor_receipt_id=None,
            proposer_checkpoint="real-ckpt", proposer_origin="model",
            parent_method={"base": 0}, candidate_program=self.program(1e-3),
            compiled_identity=None, comparison_identity="cmp",
            fixture=False)
        receipt.status = "confirmed"
        with self.assertRaises(OriginError):
            registry.publish(receipt, chief_approval_hash=None,
                             successor_proposer="real-ckpt")

    def test_stale_approval_and_substituted_proposer(self) -> None:
        registry = GenerationRegistry()
        receipt = GenerationReceipt(
            generation_id="gen-x", predecessor_receipt_id=None,
            proposer_checkpoint="ckpt-a", proposer_origin="model",
            parent_method={}, candidate_program=self.program(1e-3),
            compiled_identity=None, comparison_identity="cmp", fixture=True)
        receipt.status = "confirmed"
        # A substituted successor is recorded and visible; the fixture chain
        # cannot update a learned parent.
        registry.publish(receipt, chief_approval_hash="fixture-approval",
                         successor_proposer="substituted-agent")
        self.assertEqual(registry.successor_proposer("gen-x"), "substituted-agent")
        self.assertFalse(registry.accepted_learned_parents)


if __name__ == "__main__":
    unittest.main()
