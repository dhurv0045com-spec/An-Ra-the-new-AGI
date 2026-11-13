"""Unit and mechanical tests for the Elendril Hitchhiker experiment suite."""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path

import pytest
import torch

from anra_v5.elendril.cognition import CognitiveState, OperationCandidate
from anra_v5.elendril.hitchhiker_data import (
    ExternalSlotCanonicalizer,
    HitchhikerSurface,
    apply_case_permutation,
    assert_all_stage_disjoint,
    build_counterfactual_suite,
    build_health_probes,
    entity_bijection,
    public_case_for,
    render_hitchhiker_public,
    score_health_probes,
    score_missing_bridge_probe,
    verify_counterfactual_suite,
)
from anra_v5.elendril.hitchhiker_diagnostics import (
    ANSWER_TOKEN_IDS,
    TOKEN_ABSTAIN,
    TOKEN_DELIBERATE,
    TOKEN_Q,
    TOKEN_R,
    AvailabilityOnlyMaskPolicy,
    ExternalSlotCanonicalizerPolicy,
    GrammarControlHitchhikerPolicy,
    RawHitchhikerPolicy,
    _apply_availability_mask,
    admit_frozen_checkpoint,
    classify_diagnostic_outcomes,
    inspect_first_byte_distribution,
    score_entity_equivariance,
)
from anra_v5.elendril.hitchhiker_evaluation import (
    build_experiment_manifest,
    build_mechanism_decision_table,
    evaluate_counterfactual_suite,
    evaluate_formation_gate,
    evaluate_hitchhiker,
    evaluate_health_probes,
    evaluate_screening_gate,
    holm_adjust,
    independent_rescore_hitchhiker,
    paired_world_bootstrap_contrast,
)
from anra_v5.elendril.hitchhiker_protocol import (
    ALL_ARMS,
    AUDITED_V717_COMMIT,
    PROTOCOL_SHA,
    protocol,
    resolve_scale_profile,
    stage_jobs,
    validate_job,
)
from anra_v5.elendril.hitchhiker_training import (
    aux_penalty_gradient_probe,
    create_hitchhiker_trainer,
    run_hitchhiker_updates,
)
from anra_v5.elendril.representation import SpecialToken, UTF8
from anra_v5.elendril.training import MODEL, configure_cpu
from anra_v5.elendril.v6_data import SYMBOLS, V6Dataset, make_dataset
from anra_v5.elendril.v6_evaluation import ControlPolicy
from tools.elendril_hitchhiker_runner import main as runner_main


def _tiny_utf8_spec():
    return replace(MODEL, vocabulary_size=UTF8.vocabulary_size, context_length=512)


def test_protocol_and_scale_registry():
    p = protocol()
    assert p["schema"] == "elendril-hitchhiker-protocol-v1"
    assert len(PROTOCOL_SHA) == 64
    assert len(stage_jobs("stage1_s3_screen")) == 15
    assert len(stage_jobs("stage0_checkpoint_audit")) == 9
    assert len(stage_jobs("stage2_followups")) == 12
    assert len(stage_jobs("stage3_scale_confirmation", winning_arm="AB")) == 12
    assert set(ALL_ARMS) >= {"B0", "A", "B", "AB", "C", "AB+C", "C+penalty", "D-external"}

    job = stage_jobs("stage1_s3_screen")[0]
    validate_job(job, "stage1_s3_screen")
    with pytest.raises(ValueError):
        validate_job({**job, "lr": 1e-3}, "stage1_s3_screen")
    with pytest.raises(ValueError):
        stage_jobs("stage3_scale_confirmation", winning_arm="B0")

    # Reject unregistered scale S9 without verified receipt.
    with pytest.raises(ValueError):
        resolve_scale_profile("S9")

    tiny = _tiny_utf8_spec()
    receipt = {
        "spec": tiny,
        "parameters": tiny.parameter_receipt().total,
        "model_sha256": tiny.sha256(),
    }
    resolved = resolve_scale_profile("S9", receipt)
    assert resolved.sha256() == tiny.sha256()
    with pytest.raises(ValueError):
        resolve_scale_profile("S9", {**receipt, "parameters": receipt["parameters"] + 1})
    with pytest.raises(ValueError):
        resolve_scale_profile("S9", {**receipt, "model_sha256": "0" * 64})


def test_entity_permutation_and_counterfactual_suite():
    suite = build_counterfactual_suite(seed=17071, worlds_per_hop=2, split="validation")
    check = verify_counterfactual_suite(suite)
    assert check["status"] == "PASS"

    # Reversibility of entity_bijection + apply_case_permutation
    sample_case = suite["cohorts"]["hop-2"][0]
    mapping, inverse = entity_bijection(("test", sample_case.case_id))
    permuted = apply_case_permutation(sample_case, mapping)
    restored = apply_case_permutation(permuted, inverse)
    assert restored.goal.subject == sample_case.goal.subject
    assert restored.goal.slots == sample_case.goal.slots
    assert restored.environment.initial == sample_case.environment.initial
    assert restored.truth.expected == sample_case.truth.expected

    # PublicCase excludes Truth & hidden environment channels/memory values
    pub = public_case_for(sample_case)
    assert not hasattr(pub, "truth")
    assert not hasattr(pub, "environment")
    assert pub.memory is None or type(pub.memory) is bool

    # Verify all 7 factor invariants explicitly
    by_pair = {}
    for cohort_name, cases in suite["cohorts"].items():
        if cohort_name.startswith("factor-"):
            grouped = {}
            for c in cases:
                grouped.setdefault(c.pair_id, []).append(c)
            by_pair[cohort_name] = list(grouped.values())

    for a, b in by_pair["factor-wording"]:
        assert a.environment.initial == b.environment.initial
        assert a.truth.expected == b.truth.expected
        assert a.goal.family != b.goal.family
    for a, b in by_pair["factor-timestamps"]:
        assert a.truth.expected == b.truth.expected
        assert b.environment.initial[0].tick - a.environment.initial[0].tick == 8000
    for trio in by_pair["factor-row-order"]:
        assert len({c.truth.expected for c in trio}) == 1
    for a, b in by_pair["factor-distractors"]:
        assert a.truth.expected == b.truth.expected
        assert len(b.environment.initial) - len(a.environment.initial) == 3
    for a, b in by_pair["factor-answer-edge-swap"]:
        assert a.truth.expected != b.truth.expected
    for a, b in by_pair["factor-answer-query-swap"]:
        assert a.environment.initial == b.environment.initial
        assert a.goal.subject != b.goal.subject
        assert a.truth.expected != b.truth.expected

    # Alias clause byte-length parity check
    for quad in by_pair["factor-alias-clause"]:
        panels = {c.panel: c for c in quad}
        s_alias = CognitiveState(panels["surface-alias-with-clause"].goal).observe(
            panels["surface-alias-with-clause"].environment.initial
        )
        s_neut = CognitiveState(panels["surface-neutral-length-matched"].goal).observe(
            panels["surface-neutral-length-matched"].environment.initial
        )
        p_alias = render_hitchhiker_public(s_alias, (), panels["surface-alias-with-clause"].surface, False)
        p_neut = render_hitchhiker_public(s_neut, (), panels["surface-neutral-length-matched"].surface, False)
        assert len(p_alias.encode("utf-8")) == len(p_neut.encode("utf-8"))

    # Public teacher & privileged oracle evaluations both reach 100% across all cohorts
    teacher_eval = evaluate_counterfactual_suite(suite, ControlPolicy("fixed"))
    assert all(m["accuracy"] == 1.0 for m in teacher_eval["cohort_metrics"].values())
    oracle_eval = evaluate_counterfactual_suite(suite, ControlPolicy("oracle"))
    assert all(m["accuracy"] == 1.0 for m in oracle_eval["cohort_metrics"].values())


def test_health_probes_and_missing_bridge_controls():
    probes = build_health_probes(seed=27173, worlds_per_probe=3, split="validation")
    teacher_res = evaluate_health_probes(probes, ControlPolicy("fixed"), stage_records=None)
    scores = teacher_res["scores"]
    assert scores["distractor_trap"]["accuracy"] == 1.0
    assert scores["belief_revision"]["both_correct_timestamp_pairs"] == 3
    assert scores["missing_bridge"]["combined_gate_pass"] is True
    assert scores["preservation"]["status"] == "NOT_QUALIFIED"

    class AlwaysAbstainFactory:
        origin = "fixed_baseline"
        decoder_schema = "fixed-abstain"

        def for_case(self, case, *, memory_available=None):
            class Actor:
                model_identity = None
                origin = "fixed_baseline"

                def propose(self, state, legal, channels, positions):
                    if "ABSTAIN" in legal:
                        return OperationCandidate("ABSTAIN", None, 1.0, 1, 1, ())
                    return None

            return Actor()

    class AlwaysAnswerFactory:
        origin = "fixed_baseline"
        decoder_schema = "fixed-answer"

        def for_case(self, case, *, memory_available=None):
            class Actor:
                model_identity = None
                origin = "fixed_baseline"

                def propose(self, state, legal, channels, positions):
                    if "ANSWER" in legal:
                        return OperationCandidate("ANSWER", "A", 1.0, 1, 1, ())
                    return None

            return Actor()

    abstain_rec = evaluate_hitchhiker(probes["missing-bridge"], AlwaysAbstainFactory())
    abstain_score = score_missing_bridge_probe(abstain_rec["episodes"])
    assert abstain_score["unsolvable_abstain_accuracy"] == 1.0
    assert abstain_score["solvable_answer_accuracy"] == 0.0
    assert abstain_score["combined_gate_pass"] is False

    answer_rec = evaluate_hitchhiker(probes["missing-bridge"], AlwaysAnswerFactory())
    answer_score = score_missing_bridge_probe(answer_rec["episodes"])
    assert answer_score["unsolvable_abstain_accuracy"] == 0.0
    assert answer_score["combined_gate_pass"] is False


def test_split_custody_catches_leaks():
    suite = build_counterfactual_suite(seed=17071, worlds_per_hop=2, split="validation")
    train_base = make_dataset("train", target_bytes=12_000, seed=606)
    res = assert_all_stage_disjoint(train_base, suite)
    assert res["status"] == "PASS"

    # Leaking a relabeled training case into validation must raise ValueError.
    leaked_case = train_base.cases[0]
    mapping, _ = entity_bijection(("leak", leaked_case.case_id))
    relabeled = apply_case_permutation(leaked_case, mapping)
    bad_val = V6Dataset(
        split="validation",
        cases=(relabeled,),
        examples=(),
        samples=(),
        manifest={"version": "leak-check"},
    )
    with pytest.raises(ValueError):
        assert_all_stage_disjoint(train_base, bad_val)

    # Duplicate world_id across splits raises ValueError.
    dup_val = V6Dataset(
        split="validation",
        cases=(leaked_case,),
        examples=(),
        samples=(),
        manifest={"version": "dup-check"},
    )
    with pytest.raises(ValueError):
        assert_all_stage_disjoint(train_base, dup_val)


def test_decoders_availability_mask_specificity_and_first_byte_diagnostics(tmp_path: Path):
    configure_cpu()
    tiny = _tiny_utf8_spec()
    job = stage_jobs("stage1_s3_screen")[0]
    suite = build_counterfactual_suite(seed=17071, worlds_per_hop=1, split="validation")
    train_base = make_dataset("train", target_bytes=8_000, seed=606)
    trainer, _ = create_hitchhiker_trainer(
        job,
        source_commit=AUDITED_V717_COMMIT,
        root=tmp_path,
        base_data=train_base,
        heldout_suites=(suite,),
        device="cpu",
        cpu_canary=True,
        spec_override=tiny,
        updates_override=1,
        batch_override=4,
    )
    backend = trainer.backend

    # Verify availability mask leaves every token except unavailable R/Q exact.
    raw_logits = torch.linspace(-2.0, 2.0, UTF8.vocabulary_size)
    raw_logits[TOKEN_R] = 9.0
    raw_logits[TOKEN_Q] = 8.0
    masked = _apply_availability_mask(raw_logits, memory_available=False, query_available=False)
    assert masked[TOKEN_R] == float("-inf")
    assert masked[TOKEN_Q] == float("-inf")
    for tid in range(UTF8.vocabulary_size):
        if tid not in (TOKEN_R, TOKEN_Q):
            assert masked[tid].item() == raw_logits[tid].item()
    for tid in (*ANSWER_TOKEN_IDS, TOKEN_ABSTAIN, TOKEN_DELIBERATE, UTF8.eos_id):
        assert masked[tid].item() == raw_logits[tid].item()

    # Verify AvailabilityOnlyMaskPolicy blocks R when Memory=0 while RawHitchhikerPolicy picks R.
    class FakeBackend:
        binding = backend.binding

        def __init__(self, step0_logits: torch.Tensor) -> None:
            self.step0 = step0_logits
            self.calls = 0

        def next_distribution(self, ids: list[int]) -> torch.Tensor:
            return self.logits_for(ids).softmax(-1)

        def logits_for(self, ids: list[int]) -> torch.Tensor:
            self.calls += 1
            if ids[-1] == UTF8.special(SpecialToken.ANSWER):
                return self.step0
            # Step 1+ finishes with 'A' / EOS cleanly
            nxt = torch.full((UTF8.vocabulary_size,), -10.0)
            nxt[UTF8.eos_id] = 5.0
            return nxt

        def logits(self, tokens: torch.Tensor, segments: torch.Tensor, activation_checkpointing: bool = False):
            batch, seq_len = tokens.shape
            out = torch.full((batch, seq_len, UTF8.vocabulary_size), -10.0)
            out[:, -1, :] = self.step0 if tokens[0, -1].item() == UTF8.special(SpecialToken.ANSWER) else -10.0
            if tokens[0, -1].item() != UTF8.special(SpecialToken.ANSWER):
                out[:, -1, UTF8.eos_id] = 5.0
            return out

        @property
        def model(self):
            return backend.model

        @property
        def device(self):
            return torch.device("cpu")

    fake_step0 = torch.full((UTF8.vocabulary_size,), -10.0)
    fake_step0[TOKEN_R] = 10.0
    fake_step0[TOKEN_Q] = 6.0
    fake_step0[ANSWER_TOKEN_IDS[0]] = 3.0
    fake_backend = FakeBackend(fake_step0)

    case_no_mem = suite["cohorts"]["hop-1"][0]
    state = CognitiveState(case_no_mem.goal).observe(case_no_mem.environment.initial)
    legal = frozenset({"ANSWER", "ABSTAIN", "DELIBERATE"})

    raw_actor = RawHitchhikerPolicy(fake_backend).for_case(case_no_mem, memory_available=False)
    raw_cand = raw_actor.propose(state, legal, (), 2048)
    assert raw_cand is not None and raw_cand.operation == "RETRIEVE"

    mask_actor = AvailabilityOnlyMaskPolicy(fake_backend).for_case(case_no_mem, memory_available=False)
    mask_cand = mask_actor.propose(state, legal, (), 2048)
    assert mask_cand is not None and mask_cand.operation == "ANSWER" and mask_cand.argument == "A"

    gram_actor = GrammarControlHitchhikerPolicy(fake_backend).for_case(case_no_mem, memory_available=False)
    gram_cand = gram_actor.propose(state, legal, (), 2048)
    assert gram_cand is not None and gram_cand.operation == "ANSWER"
    assert gram_actor.decoder_schema == "utf8-grammar-control"

    # First-byte 260-way diagnostics on real tiny backend
    insp = inspect_first_byte_distribution(backend, case_no_mem)
    assert len(insp["raw_probs_260"]) == UTF8.vocabulary_size
    assert math.isclose(sum(insp["raw_probs_260"]), 1.0, rel_tol=1e-4, abs_tol=1e-4)
    assert math.isclose(sum(insp["availability_masked_probs_260"]), 1.0, rel_tol=1e-4, abs_tol=1e-4)
    assert "logit_gap_R_minus_max_answer" in insp
    assert "total_answer_letter_mass_raw" in insp
    assert "illegal_action_mass" in insp
    assert insp["answer_only_forced_choice"]["label"] == "FORCED_CHOICE_DIAGNOSTIC_NOT_AUTONOMOUS_POLICY"
    assert insp["complete_action_score"]["label"] == "COMPLETE_ACTION_SCORE_SEPARATE_FROM_FIRST_BYTE"

    eq = score_entity_equivariance(backend, suite["cohorts"]["factor-entity-permutation"], suite["permutations"])
    assert eq["complete_pairs"] == len(suite["permutations"])
    diag_cls = classify_diagnostic_outcomes(
        {"accuracy": 0.1, "invalid_rate": 0.5},
        {"accuracy": 0.12, "invalid_rate": 0.0},
        factor_metrics={
            "factor-wording": {"accuracy": 0.9},
            "factor-timestamps": {"accuracy": 0.6},
            "factor-distractors": {"accuracy": 0.8, "accuracy_gap": 0.15},
            "factor-answer-edge-swap": {"accuracy": 0.85, "both_correct_rate": 0.85},
        },
        equivariance={"argmax_equivariance_rate": 0.5},
    )
    assert "CONTROLLER_LEGALITY_ONLY_MAPPING_UNRESOLVED" in diag_cls["findings"]
    assert "SURFACE_DISTRIBUTION_MISMATCH_PRIORITIZE_DATA" in diag_cls["findings"]
    assert "INVARIANCE_FAILURE" in diag_cls["findings"]
    assert "BEHAVIORAL_GROUNDING_SUPPORTED" in diag_cls["findings"]


def test_trainer_aux_gradient_probe_and_resume_parity(tmp_path: Path):
    configure_cpu()
    tiny = _tiny_utf8_spec()
    suite = build_counterfactual_suite(seed=17071, worlds_per_hop=1, split="validation")
    train_base = make_dataset("train", target_bytes=8_000, seed=606)

    # C+penalty aux gradient accumulation probe parity
    c_pen_job = [j for j in stage_jobs("stage2_followups") if j["arm"] == "C+penalty" and j["seed"] == 7][0]
    tr_pen, _ = create_hitchhiker_trainer(
        c_pen_job,
        source_commit=AUDITED_V717_COMMIT,
        root=tmp_path / "pen",
        base_data=train_base,
        heldout_suites=(suite,),
        device="cpu",
        cpu_canary=True,
        spec_override=tiny,
        updates_override=2,
        batch_override=4,
    )
    probe = aux_penalty_gradient_probe(tr_pen)
    assert probe["status"] == "PASS"
    assert probe["aux_illegal_weight"] == pytest.approx(0.1)
    run_pen = run_hitchhiker_updates(tr_pen)
    assert math.isfinite(run_pen["final_loss"])

    # Crash recovery / bit-for-bit checkpoint save-resume parity on arms A and AB
    for arm in ("A", "AB"):
        job = [j for j in stage_jobs("stage1_s3_screen") if j["arm"] == arm and j["seed"] == 7][0]
        full_tr, _ = create_hitchhiker_trainer(
            job,
            source_commit=AUDITED_V717_COMMIT,
            root=tmp_path / f"full_{arm}",
            base_data=train_base,
            heldout_suites=(suite,),
            device="cpu",
            cpu_canary=True,
            spec_override=tiny,
            updates_override=2,
            batch_override=4,
        )
        full_res = run_hitchhiker_updates(full_tr, save_every=1)
        full_sha = full_tr.store.latest_sha256()

        split_root = tmp_path / f"resumed_{arm}"
        part1, _ = create_hitchhiker_trainer(
            job,
            source_commit=AUDITED_V717_COMMIT,
            root=split_root,
            base_data=train_base,
            heldout_suites=(suite,),
            device="cpu",
            cpu_canary=True,
            spec_override=tiny,
            updates_override=2,
            batch_override=4,
        )
        # Execute only 1 update, save checkpoint, and resume in a fresh trainer instance
        step1 = part1.step()
        assert math.isfinite(step1["loss"])
        part1.save()

        part2, _ = create_hitchhiker_trainer(
            job,
            source_commit=AUDITED_V717_COMMIT,
            root=split_root,
            base_data=train_base,
            heldout_suites=(suite,),
            resume=True,
            device="cpu",
            cpu_canary=True,
            spec_override=tiny,
            updates_override=2,
            batch_override=4,
        )
        resumed_res = run_hitchhiker_updates(part2)
        resumed_sha = part2.save()
        assert resumed_res["updates_executed"] == 1
        assert resumed_res["final_loss"] == pytest.approx(full_res["final_loss"], rel=0.0, abs=0.0)
        assert resumed_sha == full_sha
        for k, tensor in full_tr.model.state_dict().items():
            assert torch.equal(tensor, part2.model.state_dict()[k])


def test_external_slot_canonicalizer_attribution_and_replay(tmp_path: Path):
    configure_cpu()
    tiny = _tiny_utf8_spec()
    suite = build_counterfactual_suite(seed=17071, worlds_per_hop=1, split="validation")
    train_base = make_dataset("train", target_bytes=8_000, seed=606)
    dext_job = [j for j in stage_jobs("stage2_followups") if j["arm"] == "D-external" and j["seed"] == 7][0]
    trainer, _ = create_hitchhiker_trainer(
        dext_job,
        source_commit=AUDITED_V717_COMMIT,
        root=tmp_path,
        base_data=train_base,
        heldout_suites=(suite,),
        device="cpu",
        cpu_canary=True,
        spec_override=tiny,
        updates_override=1,
        batch_override=4,
    )

    for assignment in ("first-mention", "random"):
        factory = ExternalSlotCanonicalizerPolicy(trainer.backend, assignment=assignment, slot_seed=17)
        rec = evaluate_hitchhiker(suite["cohorts"]["factor-entity-permutation"], factory, split="validation")
        assert rec["attribution_origin"] == "engineered_canonicalizer"
        assert rec["canonicalizer_config"]["assignment"] == assignment
        assert all(ep["overhead_receipts"] for ep in rec["episodes"])
        replayed = independent_rescore_hitchhiker(suite["cohorts"]["factor-entity-permutation"], rec)
        assert replayed == rec["metrics"]

    # First-mention canonicalizer produces identical slot-canonical prompts for isomorphic permutations
    canon = ExternalSlotCanonicalizer(assignment="first-mention", strip_aliases=True)
    orig, perm = suite["cohorts"]["factor-entity-permutation"][:2]
    m_o, inv_o = canon.build_mapping(CognitiveState(orig.goal).observe(orig.environment.initial), orig.surface)
    m_p, inv_p = canon.build_mapping(CognitiveState(perm.goal).observe(perm.environment.initial), perm.surface)
    s_o, surf_o, _ = canon.canonicalize_public(CognitiveState(orig.goal).observe(orig.environment.initial), orig.surface, m_o)
    s_p, surf_p, _ = canon.canonicalize_public(CognitiveState(perm.goal).observe(perm.environment.initial), perm.surface, m_p)
    assert render_hitchhiker_public(s_o, (), surf_o, False) == render_hitchhiker_public(s_p, (), surf_p, False)
    slot_for_orig_ans = m_o[orig.truth.expected]
    assert canon.dereference_output("ANSWER", slot_for_orig_ans, inv_p) == ("ANSWER", perm.truth.expected)


def test_admit_frozen_checkpoint_states(tmp_path: Path):
    configure_cpu()
    missing_dir = tmp_path / "does_not_exist"
    pending = admit_frozen_checkpoint(missing_dir, "S20-low-lr-i7-o7")
    assert pending["status"] == "CHECKPOINT_AUDIT_PENDING"
    assert pending["fabricate_permitted"] is False
    assert not missing_dir.exists()

    # Valid checkpoint creation + admission + mismatch checks
    tiny = _tiny_utf8_spec()
    job = stage_jobs("stage1_s3_screen")[0]
    train_base = make_dataset("train", target_bytes=8_000, seed=606)
    ckpt_root = tmp_path / "ckpts"
    trainer, _ = create_hitchhiker_trainer(
        job,
        source_commit=AUDITED_V717_COMMIT,
        root=ckpt_root,
        base_data=train_base,
        device="cpu",
        cpu_canary=True,
        spec_override=tiny,
        updates_override=1,
        batch_override=4,
    )
    trainer.step()
    sha = trainer.save()

    admitted = admit_frozen_checkpoint(ckpt_root, job["job_id"], expected_spec=tiny, expected_update=1)
    assert admitted["status"] == "ADMITTED"
    assert admitted["checkpoint_sha256"] == sha

    with pytest.raises(ValueError):
        admit_frozen_checkpoint(ckpt_root, job["job_id"], expected_spec=resolve_scale_profile("S20"))
    with pytest.raises(ValueError):
        admit_frozen_checkpoint(ckpt_root, job["job_id"], expected_update=99)


def test_statistics_gates_manifest_and_cli(tmp_path: Path):
    left = [{"world_id": f"w{i}", "outcome": {"correct": True}} for i in range(12)]
    right = [{"world_id": f"w{i}", "outcome": {"correct": i % 2 == 0}} for i in range(12)]
    contrast = paired_world_bootstrap_contrast(left, right, resamples=200)
    assert contrast["worlds"] == 12
    assert contrast["mean_diff"] > 0.0
    assert contrast["ci_95"][0] > 0.0

    adj = holm_adjust([0.005, 0.03, 0.2])
    assert adj == pytest.approx([0.015, 0.06, 0.2])

    # Screening gate: PASS_SCREENING vs FALSIFIED vs INCONCLUSIVE
    def mk_seed_report(acc: float, worlds: int = 8) -> dict:
        eps = [{"world_id": f"w{i}", "outcome": {"correct": i < int(acc * worlds)}} for i in range(worlds)]
        return {"records": {"factor-entity-permutation": {"metrics": {"accuracy": acc}, "episodes": eps}}}

    pass_gate = evaluate_screening_gate(
        {7: mk_seed_report(1.0), 19: mk_seed_report(1.0), 41: mk_seed_report(1.0)},
        {7: mk_seed_report(0.0), 19: mk_seed_report(0.0), 41: mk_seed_report(0.0)},
    )
    assert pass_gate["status"] == "PASS_SCREENING"

    fail_gate = evaluate_screening_gate(
        {7: mk_seed_report(0.0), 19: mk_seed_report(0.0), 41: mk_seed_report(0.0)},
        {7: mk_seed_report(1.0), 19: mk_seed_report(1.0), 41: mk_seed_report(1.0)},
    )
    assert fail_gate["status"] == "FALSIFIED"

    # Formation gate refuses to promote <512 world checks
    synth_metrics = {
        **{f"hop-{h}": {"accuracy": 0.99, "invalid_rate": 0.0} for h in (1, 2, 3, 4)},
        "factor-entity-permutation": {"accuracy": 0.99, "invalid_rate": 0.0},
        "factor-answer-edge-swap": {"accuracy": 0.99, "pair_accuracy": 0.99, "invalid_rate": 0.0},
        "factor-answer-query-swap": {"accuracy": 0.99, "pair_accuracy": 0.99, "invalid_rate": 0.0},
    }
    small_gate = evaluate_formation_gate(
        {"cohort_metrics": synth_metrics, "worlds_per_hop": 16},
        equivariance_report={"argmax_equivariance_rate": 0.99},
    )
    assert small_gate["status"] == "MECHANICAL_ONLY_NOT_FORMATION_QUALIFIED"

    large_gate = evaluate_formation_gate(
        {"cohort_metrics": synth_metrics, "worlds_per_hop": 512},
        equivariance_report={"argmax_equivariance_rate": 0.99},
    )
    assert large_gate["status"] == "FORMATION_QUALIFIED"

    with pytest.raises(ValueError):
        build_mechanism_decision_table(
            [{"mechanism": "A", "status": "PROVEN", "attribution": "model", "evidence_summary": "x", "next_action": "y"}]
        )
    manifest = build_experiment_manifest(
        source_commit=AUDITED_V717_COMMIT,
        stage="stage0_checkpoint_audit",
        stage0_status={"status": "CHECKPOINT_AUDIT_PENDING"},
        arm_runs=[],
        decision_table=[
            {"mechanism": "Stage 0", "status": "PENDING", "attribution": "model", "evidence_summary": "none", "next_action": "wait"}
        ],
        claim_status="PENDING",
    )
    assert manifest["capability_claim"] == "NO_COGNITION_CLAIM_FROM_MECHANICAL_OR_UNEXECUTED_CHECKS"

    # CLI verify-suite & stage0-audit without checkpoint
    verify_out = tmp_path / "verify_suite.json"
    assert runner_main(["--mode", "verify-suite", "--worlds-per-hop", "1", "--output", str(verify_out)]) == 0
    verify_payload = json.loads(verify_out.read_text(encoding="utf-8"))
    assert verify_payload["suite_verification"]["status"] == "PASS"

    stage0_out = tmp_path / "stage0_audit.json"
    assert runner_main(["--mode", "stage0-audit", "--worlds-per-hop", "1", "--output", str(stage0_out)]) == 0
    stage0_payload = json.loads(stage0_out.read_text(encoding="utf-8"))
    assert stage0_payload["stage0_checkpoint_audit"]["status"] == "CHECKPOINT_AUDIT_PENDING"
