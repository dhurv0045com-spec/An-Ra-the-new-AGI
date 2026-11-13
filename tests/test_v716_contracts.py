"""V7.16 contract tests: registration, custody, mechanisms, exposure locks.

Everything here runs on CPU with bounded fixtures. The suite verifies the
six falsification arms' code contracts and the preregistration custody chain;
it never claims a research result. GPU execution remains owner-run by design.
"""
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from anra_v5.elendril import custody_gate, v715_protocol, v716_protocol
from anra_v5.elendril.codec import canonical, digest
from anra_v5.elendril.foundation_training import StageTransition, TokenSample
from anra_v5.elendril.training import MODEL
from anra_v5.elendril.v6_data import _supervision, make_dataset
from anra_v5.elendril.v716_diagnostics import (LinearProbe, first_divergence,
                                               hop_accuracy, invalid_actions_total,
                                               proxy_mutual_information,
                                               row_reversal_delta, stale_memory_score,
                                               trace_diff_report)
from anra_v5.elendril.v716_mechanisms import (Ark014SurfaceAugmenter,
                                              Ark017SurpriseReplayBuffer,
                                              CanonicalRowNormalizer,
                                              CausalPointerTwoReadDecoder,
                                              CoAdaptedControllerMaskLoss,
                                              CounterfactualMemoryGenerator,
                                              OrderContrastiveLoss,
                                              OutcomeCoupledDeliberationTrainer,
                                              RolloutRecord, public_state_specs,
                                              rollout_sample)
from anra_v5.elendril.v716_training import _arm_rows, create_trainer
from anra_v5.elendril.v71_protocol import PROTOCOL_SHA as V71_SHA


CANARY = dict(device="cpu", cpu_canary=True)


def micro_spec():
    return replace(MODEL, vocabulary_size=260, width=16, layers=1, query_heads=2,
                   kv_heads=1, head_dimension=8, ffn_width=32, context_length=512)


def job(arm, seed=7, profile="S3", epochs=4):
    return {"job_id": f"{arm}-s{seed}" if profile == "S3" else f"{profile}-{arm}-s{seed}",
            "arm": arm, "profile": profile, "seed": seed, "epochs": epochs,
            "mechanism": v716_protocol.ARM_MECHANISMS[arm],
            "exposure": "blocked-ce-baseline"}


@pytest.fixture(scope="module")
def data():
    return make_dataset(worlds=48)


@pytest.fixture(scope="module")
def small():
    return make_dataset(worlds=8)


# ---------------------------------------------------------------------------
# Registration and preregistration custody
# ---------------------------------------------------------------------------

def test_parameter_counts_match_recorded_inventories():
    from anra_v5.elendril.v6_registry import MODEL_PROFILES
    for profile, expected in v716_protocol.PARAMETER_COUNTS.items():
        assert MODEL_PROFILES[profile].parameter_receipt().total == expected


def test_preregistered_matrix_is_exact_and_outcome_independent():
    mechanism = v716_protocol.jobs("mechanism")
    assert len(mechanism) == 7 * 3
    assert {entry["profile"] for entry in mechanism} == {"S3"}
    assert all(entry["epochs"] == v716_protocol.EXPOSURE["epochs"] for entry in mechanism)
    assert v716_protocol.jobs("mechanism") == mechanism  # deterministic rebuild
    assert len(v716_protocol.jobs("continual")) == 4 * 3
    with pytest.raises(ValueError):
        v716_protocol.jobs("unknown-phase")
    for phase in ("mechanism", "continual"):
        for entry in v716_protocol.jobs(phase):
            assert v716_protocol.validate(entry, phase)


def test_preregistration_json_is_the_committed_canonical_content():
    committed = Path("research/v7.16/PREREGISTRATION.json")
    if committed.is_file():  # generated artifact; byte equality when present
        payload = json.loads(committed.read_text(encoding="utf-8"))
        assert canonical(payload) == canonical(v716_protocol.preregistration())
        assert payload["protocol_sha256"] == v716_protocol.PROTOCOL_SHA
    else:  # until generated, the in-memory registration must already be canonical
        assert len(canonical(v716_protocol.preregistration())) > 100


def test_protocol_binds_parent_pins_and_claim_ceiling():
    protocol = v716_protocol.protocol()
    assert protocol["parent_branch_head"] == "29b1da646bd06c7f2deeb5e5f3c9f08fe24cabbb"
    assert protocol["essay_audit_pin"] == "9092acb1"
    assert protocol["v71_core_source_sha"] == "6104ffd7134cb4cf4b3f340bd060d24a1e2dd200"
    assert protocol["parent_full_sha256"] == (
        "1826c85b5a858d2a600661f7e5916f0fa812c55a8d53c37013c695df28a15d19")
    assert protocol["v715_s8_source_pin"] == "e5779922a0072a070f64356f38ea47090e543d6a"
    assert protocol["v715_s8_run_id"] == "3981e34a2ffe4b4b9528b4b4d588883b"
    assert protocol["v715_s20_source_pin"] == "4cc1e993c09d1475a3bc1e250fda58909d539407"
    assert protocol["claim_ceiling"] == "CONTROLLED_DEVELOPMENT_MECHANISM_ONLY"
    assert not any(protocol["authorization_flags"].values())
    assert protocol["v71_protocol_sha256"] == V71_SHA
    assert protocol["scientific_status"].startswith("NOT_RUN")
    assert protocol["exposure_lock"] == {"epochs": 4, "optimizer_updates": 2_368,
                                         "supervised_tokens": 75_776,
                                         "unpadded_input_positions": 8_914_404}


def test_execution_binding_template_is_unbound():
    binding = v716_protocol.binding_template()
    assert binding["status"] == "UNBOUND"
    assert binding["preregistration_sha256"] == digest(v716_protocol.preregistration())
    assert binding["protocol_sha256"] == v716_protocol.PROTOCOL_SHA
    assert binding["science_commit"] is None and binding["execution_commit"] is None
    assert binding["public_surface_sha256"] is None and binding["data_split_sha256"] is None
    assert binding["matched_seed_initialization"] == {"7": None, "19": None, "41": None}
    assert binding["custody"]["parent_full_sha256"] == v715_protocol.PARENT_FULL_SHA256
    assert binding["custody"]["parent_full_verified"] is None
    assert binding["custody"]["s8_analysis_sha256"] == (
        "08cca3edcec070b85fac752faf89f156b85fb7b7fc2fbd9b02c908461175c02b")


# ---------------------------------------------------------------------------
# Custody gate (Phase A)
# ---------------------------------------------------------------------------

def test_sha256_file_streams_and_matches(tmp_path):
    path, fingerprint, size = custody_gate.fixture_archive(tmp_path / "fixture.bin")
    assert fingerprint == hashlib.sha256(custody_gate.FIXTURE_PAYLOAD).hexdigest()
    assert size == len(custody_gate.FIXTURE_PAYLOAD)
    assert custody_gate.sha256_file(path) == (fingerprint, size)


def test_gate_missing_parent_archive_is_no_go(tmp_path):
    record = custody_gate.preexecution_gate(parent_archive=tmp_path / "absent.bin")
    assert record["CUSTODY_STATUS"] == "NO_GO_MISSING_PARENT_ARCHIVE"
    assert record["parent_archive"]["present"] is False
    assert record["failures"] == ["NO_GO_MISSING_PARENT_ARCHIVE"]
    assert record["pins"]["v716_protocol_sha256"] == v716_protocol.PROTOCOL_SHA


def test_gate_parent_sha_mismatch_is_no_go(tmp_path):
    path, _, _ = custody_gate.fixture_archive(tmp_path / "wrong.bin")
    record = custody_gate.preexecution_gate(parent_archive=path)
    assert record["CUSTODY_STATUS"] == "NO_GO_PARENT_ARCHIVE_SHA_MISMATCH"
    assert record["parent_archive"]["present"] is True
    assert record["parent_archive"]["match"] is False


def test_gate_reports_the_pinned_parent_and_refuses_fixture_bytes(tmp_path):
    path = tmp_path / "parent.bin"
    path.write_bytes(b"owner-run parent archive bytes")
    verified = custody_gate.verify_archive(path, custody_gate.sha256_file(path)[0])
    assert verified["present"] is True and verified["match"] is True
    # The strict gate still refuses: the fixture bytes are not the real parent.
    record = custody_gate.preexecution_gate(parent_archive=path)
    assert record["CUSTODY_STATUS"] == "NO_GO_PARENT_ARCHIVE_SHA_MISMATCH"
    assert record["pins"]["parent_full_sha256"] == v715_protocol.PARENT_FULL_SHA256


def test_gate_optional_s8_analysis_byte_size_check(tmp_path):
    path = tmp_path / "analysis.zip"
    path.write_bytes(b"x" * 100)
    record = custody_gate.preexecution_gate(parent_archive=tmp_path / "p.bin",
                                            s8_analysis_zip=path, require_s8_analysis=True)
    assert "NO_GO_MISSING_PARENT_ARCHIVE" in record["failures"]
    assert "NO_GO_S8_ANALYSIS_ARCHIVE_MISMATCH" in record["failures"]


def test_write_once_refuses_overwrite(tmp_path):
    path = custody_gate.write_once(tmp_path / "PREEXECUTION_GATE.json",
                                   {"CUSTODY_STATUS": "PASS"})
    assert path.is_file()
    with pytest.raises(FileExistsError):
        custody_gate.write_once(path, {"CUSTODY_STATUS": "PASS"})


# ---------------------------------------------------------------------------
# H1: co-adapted controller mask
# ---------------------------------------------------------------------------

def test_h1_teacher_targets_satisfy_the_public_grammar(small):
    specs = public_state_specs(small.cases)
    mask = CoAdaptedControllerMaskLoss()
    for case in small.cases:
        examples, _ = _supervision(case)
        for example in examples:
            sample = example.sample
            assert mask.span_violations(sample.tokens, sample.eligible,
                                        specs[example.task_id]) == []


def test_h1_mask_is_a_function_of_the_input_prefix_alone(small):
    """Flipping a target byte cannot change the mask (no target leakage)."""
    specs = public_state_specs(small.cases)
    mask = CoAdaptedControllerMaskLoss()
    example = next(e for c in small.cases for e in _supervision(c)[0])
    sample = example.sample
    first_target = sample.eligible.index(True)
    corrupted = list(sample.tokens)
    corrupted[first_target] = (corrupted[first_target] + 1) % 260
    assert mask.action_masks(sample.tokens, sample.eligible, specs[example.task_id]) \
        == mask.action_masks(tuple(corrupted), sample.eligible, specs[example.task_id])


def test_h1_masked_logits_are_finite_and_carry_the_fill(small):
    torch.manual_seed(0)
    specs = public_state_specs(small.cases)
    mask = CoAdaptedControllerMaskLoss()
    example = next(e for c in small.cases for e in _supervision(c)[0])
    sample = example.sample
    tokens = torch.tensor([sample.tokens]).repeat(2, 1)
    eligible = torch.tensor([sample.eligible]).repeat(2, 1)
    logits = torch.randn(2, len(sample.tokens), 260) * 0.1
    masked = mask(logits, tokens, eligible, [specs[example.task_id]] * 2)
    assert torch.isfinite(masked).all()
    first_target = int(torch.nonzero(eligible[0])[0])
    legal = mask.first_action_ids(specs[example.task_id])
    position = first_target - 1
    for token_id in range(260):
        if token_id in legal:
            assert masked[0, position, token_id].item() == pytest.approx(
                logits[0, position, token_id].item())
        else:
            assert masked[0, position, token_id].item() == pytest.approx(
                logits[0, position, token_id].item() + mask.MASKED_LOGIT)
    with pytest.raises(ValueError, match="margin"):
        mask(logits * 4000.0, tokens, eligible, [specs[example.task_id]] * 2)
    with pytest.raises(ValueError, match="nonfinite|finite"):
        mask(torch.full_like(logits, float("nan")), tokens, eligible,
             [specs[example.task_id]] * 2)


def test_h1_mask_policy_flags_are_typed():
    with pytest.raises(ValueError):
        CoAdaptedControllerMaskLoss(allow_query="yes")


# ---------------------------------------------------------------------------
# H2 / H2B: surface invariance and canonicalization
# ---------------------------------------------------------------------------

def test_h2_augmenter_parity_determinism_and_fallback_ledger(data):
    reserved = {case.world_id for case in data.cases}
    first = Ark014SurfaceAugmenter(data.cases, seed=31337, epochs=4,
                                   reserved_world_ids=reserved).audit()
    second = Ark014SurfaceAugmenter(data.cases, seed=31337, epochs=4,
                                    reserved_world_ids=reserved).audit()
    assert first == second
    assert first["exposure_parity"] is True
    assert first["epochs"]["canonical"]["supervised_tokens"] == \
        sum(sum(s.eligible) for s in data.samples)
    assert all(first["epochs"][plan]["examples"] == first["epochs"]["canonical"]["examples"]
               for plan in first["epochs"])
    assert all(first["epochs"][plan]["input_positions"] ==
               first["epochs"]["canonical"]["input_positions"] for plan in first["epochs"])
    assert set(first["fallbacks"]) == set(Ark014SurfaceAugmenter.EPOCH_PLAN)


def test_h2_audit_is_single_pass_and_ledger_pure(small):
    """Committed provenance must count exactly one evaluation per epoch.

    The audit previously re-evaluated every epoch three times (examples,
    case count and determinism digest), inflating the committed fallback
    counts on top of the training row build. A digit-ceiling tick forces a
    deterministic shifted-timestamp fallback to observe the ledger.
    """
    from anra_v5.elendril.v716_mechanisms import _rebuild
    source = small.cases[0]
    ceiling_rows = [(event.subject, event.source, 999, event.value)
                    for event in source.environment.initial]
    hidden_rows = [(event.subject, event.source, 999, event.value)
                   for values in source.environment.channels.values()
                   for event in values]
    remembered_rows = [(event.subject, event.source, 999, event.value)
                       for event in source.memory.events] \
        if source.memory is not None else []
    ceiling = _rebuild(source, initial_rows=ceiling_rows, hidden_rows=hidden_rows,
                       remembered_rows=remembered_rows, expected=source.truth.expected)
    augmenter = Ark014SurfaceAugmenter((ceiling,), seed=5, epochs=4)
    first = augmenter.audit()
    assert first["exposure_parity"] is True
    # No shift room at the digit ceiling: epoch 3 must fall back exactly once.
    assert first["fallbacks"]["shifted-timestamps"] == 1
    # The audit consumes nothing; the ledger still reads zero.
    assert augmenter.fallbacks == {plan: 0 for plan in augmenter.EPOCH_PLAN}
    assert augmenter.permutations_admitted == 0
    for epoch in range(1, 5):
        augmenter.epoch_cases(epoch)
    # One recorded consumption per epoch equals the single-pass audit counts.
    assert augmenter.fallbacks == first["fallbacks"]
    assert augmenter.permutations_admitted == first["permutations_admitted"]
    assert augmenter.audit() == first  # provenance is position-independent


def test_h2_epoch_plan_is_registered_and_shift_is_length_safe(small):
    augmenter = Ark014SurfaceAugmenter(small.cases, seed=5, epochs=4)
    assert augmenter.EPOCH_PLAN == ("canonical", "reversed-rows",
                                    "shifted-timestamps", "symbol-permutation")
    reversed_rows = augmenter.epoch_cases(2)
    assert all(case.surface.order == "reverse" for case in reversed_rows)
    # A digit-width-preserving shift keeps every rendered tick the same length.
    canonical_bytes = sum(len(e.prompt.encode("utf-8")) + len(e.target.encode("utf-8"))
                          for case in augmenter.epoch_cases(1) for e in _supervision(case)[0])
    shifted_bytes = sum(len(e.prompt.encode("utf-8")) + len(e.target.encode("utf-8"))
                        for case in augmenter.epoch_cases(3) for e in _supervision(case)[0])
    assert canonical_bytes == shifted_bytes
    with pytest.raises(ValueError):
        augmenter.epoch_cases(0)
    with pytest.raises(ValueError):
        augmenter.epoch_cases(5)


def test_h2_shifts_preserve_observation_state_keys(small):
    """Rank-preserving tick shifts must not move a world across the split."""
    from anra_v5.elendril.v6_data import _partition
    augmenter = Ark014SurfaceAugmenter(small.cases, seed=5, epochs=4)
    for case in augmenter.epoch_cases(3):
        _, keys = _supervision(case)
        assert all(_partition(key) == "train" for key in keys)


def test_h2b_normalizer_preserves_semantics_targets_and_state_keys(data):
    canonical_examples = tuple(e for case in data.cases
                               for e in CanonicalRowNormalizer.supervision(case))
    base_examples = tuple(e for case in data.cases for e in _supervision(case)[0])
    assert len(canonical_examples) == len(base_examples)
    assert all(a.task_id == b.task_id and a.target == b.target
               for a, b in zip(canonical_examples, base_examples))
    assert sum(sum(e.sample.eligible) for e in canonical_examples) == \
        sum(sum(s.eligible) for s in data.samples)
    for case in data.cases:
        assert _supervision(case)[1] == _supervision(replace(
            case, environment=replace(
                case.environment,
                initial=CanonicalRowNormalizer.sorted_events(case.environment.initial))))[1]


# ---------------------------------------------------------------------------
# H3 / H3B: counterfactual memory and contrastive pairs
# ---------------------------------------------------------------------------

def test_h3_variants_are_shape_matched_and_parity_safe(data):
    generator = CounterfactualMemoryGenerator(seed=4242)
    assert [case for case in data.cases if generator.eligible(case)], \
        "48 worlds must contain one-decision revision cases"
    audit = None
    for epoch, plan in enumerate(generator.VARIANT_PLAN, start=1):
        replacements, audit = generator.epoch_replacements(data.cases, epoch=epoch)
        cases = [replacements.get(case.case_id, case) for case in data.cases]
        examples = tuple(e for case in cases for e in _supervision(case)[0])
        assert sum(sum(e.sample.eligible) for e in examples) == \
            sum(sum(s.eligible) for s in data.samples)
        assert len(examples) == len(data.samples)
        for case_id, variant in replacements.items():
            base = next(case for case in data.cases if case.case_id == case_id)
            base_examples, variant_examples = _supervision(base)[0], _supervision(variant)[0]
            assert len(base_examples) == len(variant_examples)
            # Shape-matched means loss-exposure parity, not identical targets:
            # the unresolved-contradiction variant may legitimately flip an
            # answer to the safe ABSTAIN action once evidence contradicts.
            assert all(sum(a.sample.eligible) == sum(b.sample.eligible)
                       for a, b in zip(base_examples, variant_examples))
            if plan != "unresolved-contradiction":
                # Stale-overwrite and consistent-control render the same rows
                # (or one re-attributed row), so prompt lengths stay equal.
                assert all(len(a.sample.tokens) == len(b.sample.tokens)
                           for a, b in zip(base_examples, variant_examples))
            assert variant.panel.startswith("v716-")
    assert audit["replaced"] <= audit["selected"]


def test_h3_consistent_control_reattributes_every_stale_row(data):
    """The control must neutralize the value the stale-overwrite arm plants.

    With three subject rows the control previously re-attributed only the
    first stale row in row order, so later stale values stayed attributed
    to the subject and the oldest row — the one the treatment attaches to
    memory — could survive untouched. Every non-latest subject row must
    move, leaving a single-latest consistent subject history.
    """
    from anra_v5.elendril.v716_mechanisms import SYMBOLS, _rebuild
    generator = CounterfactualMemoryGenerator(seed=4242)
    candidates = [case for case in data.cases if generator.eligible(case)
                  and len(generator._subject_rows(case)) >= 2]
    assert candidates, "the fixture must contain multi-overwrite revision cases"
    base = candidates[0]
    subject_rows = generator._subject_rows(base)
    keeper = max(subject_rows, key=lambda row: row[2])
    ticks = [row[2] for row in subject_rows]
    shift = 2 if min(ticks) < 2 else 0
    rows = [(event.subject, event.source, event.tick + shift, event.value)
            for event in base.environment.initial]
    hidden = [(event.subject, event.source, event.tick + shift, event.value)
              for values in base.environment.channels.values() for event in values]
    remembered = [(event.subject, event.source, event.tick + shift, event.value)
                  for event in base.memory.events] if base.memory is not None else []
    taken = set(base.goal.slots) | {base.truth.expected} \
        | {row[3] for row in subject_rows if row[3]}
    stale_value = next(symbol for symbol in SYMBOLS if symbol not in taken)
    extra = (base.goal.subject, keeper[1], min(ticks) + shift - 1, stale_value)
    three = _rebuild(base, initial_rows=[extra, *rows], hidden_rows=hidden,
                     remembered_rows=remembered, expected=base.truth.expected)
    assert generator.eligible(three), "the three-overwrite history stays answerable"
    variant = generator.consistent_control(three)
    assert variant is not None
    survivors = generator._subject_rows(variant)
    assert len(survivors) == 1 and survivors[0][2] == max(ticks) + shift
    assert survivors[0][3] == base.truth.expected
    moved = [(event.subject, event.source, event.tick, event.value)
             for event in variant.environment.initial
             if event.subject != base.goal.subject]
    treatment = generator.stale_overwrite(three)
    assert treatment is not None
    planted = [event.value for event in treatment.memory.events]
    assert planted and all(value in {row[3] for row in moved} for value in planted)
    base_examples, variant_examples = _supervision(three)[0], _supervision(variant)[0]
    assert len(base_examples) == len(variant_examples) == 1
    assert sum(base_examples[0].sample.eligible) == sum(variant_examples[0].sample.eligible)
    assert len(base_examples[0].sample.tokens) == len(variant_examples[0].sample.tokens)


def test_h3_selection_is_deterministic_and_fraction_bounded(data):
    generator = CounterfactualMemoryGenerator(seed=7)
    replacements_a, audit_a = generator.epoch_replacements(data.cases, epoch=1)
    replacements_b, audit_b = CounterfactualMemoryGenerator(seed=7).epoch_replacements(
        data.cases, epoch=1)
    assert audit_a == audit_b and set(replacements_a) == set(replacements_b)
    eligible = sum(1 for case in data.cases if generator.eligible(case))
    assert audit_a["selected"] == min(eligible, int(0.25 * eligible))


def test_h3b_twin_pairs_reference_real_samples(data):
    rows, seams, ledger = _arm_rows(job("H3B-CONTRASTIVE"), data)
    pairs = seams.get("contrastive_pairs", ())
    assert pairs, "the 48-world fixture must yield contrastive twin pairs"
    assert ledger["contrastive_weight"] == 0.1 and ledger["contrastive_margin"] == 0.5
    sample_ids = {sample.task_id for epoch_rows in rows for sample in epoch_rows}
    for epoch, left, right in pairs:
        assert left in sample_ids and right in sample_ids
        assert left.endswith(f"/0@e{epoch}") and right.endswith(f"/0@e{epoch}")


def test_order_contrastive_loss_margin_math():
    loss = OrderContrastiveLoss(margin=0.5)
    identical = torch.ones(3, 4)
    orthogonal_a = torch.tensor([[1.0, 0.0, 1.0, 0.0]] * 3)
    orthogonal_b = torch.tensor([[0.0, 1.0, 0.0, 1.0]] * 3)
    assert loss(identical, identical).item() == pytest.approx(0.0)
    assert loss(orthogonal_a, orthogonal_b).item() == pytest.approx(0.5)  # cosine 0
    with pytest.raises(ValueError):
        loss(torch.ones(0, 4), torch.ones(0, 4))
    with pytest.raises(FloatingPointError):
        loss(torch.full((1, 4), float("nan")), torch.ones(1, 4))


# ---------------------------------------------------------------------------
# H4: outcome-coupled deliberation selection
# ---------------------------------------------------------------------------

def rollout(task, correct, queries, illegal=0):
    return RolloutRecord(task_id=task, trace_id=f"{task}-{correct}-{queries}",
                         actions="~A" if correct else "!", correct=correct,
                         knowable=True, verified_queries=queries,
                         deliberations=1, illegal=illegal, supervised_tokens=2)


def test_h4_utility_accounting_and_rloo_selection():
    trainer = OutcomeCoupledDeliberationTrainer(rollouts=2)
    assert rollout("w/0", True, 2, illegal=1).utility() == pytest.approx(1.0 - 0.2 - 0.05)
    group = {"w/0": [rollout("w/0", True, 0), rollout("w/0", False, 4)]}
    kept = trainer.select(group)
    assert set(kept) == {"w/0"}
    assert [record.trace_id for record in kept["w/0"]] == ["w/0-True-0"]
    utilities = (1.0, -0.4)
    assert trainer.rloo_baseline(utilities, 0) == pytest.approx(-0.4)
    assert trainer.rloo_baseline(utilities, 1) == pytest.approx(1.0)
    single = OutcomeCoupledDeliberationTrainer(rollouts=1)
    assert single.rloo_baseline((0.5,), 0) == 0.0  # K=1: baseline is zero
    with pytest.raises(ValueError):
        trainer.rloo_baseline((0.5,), 0)  # K=2 rejects the incomplete group
    assert trainer.select({"w/1": [rollout("w/1", False, 5), rollout("w/1", False, 6)]}) == {}
    with pytest.raises(ValueError):
        trainer.select({"w/2": [rollout("w/2", True, 0)]})  # incomplete group
    with pytest.raises(ValueError):
        OutcomeCoupledDeliberationTrainer(rollouts=3)
    audit = trainer.audit(group)
    assert audit["rollouts"] == 2 and audit["kept"] == 1


def test_h4_rollout_framing_is_bounded():
    sample = rollout_sample(rollout("w/0", True, 0), "Query=1\nAction:")
    assert isinstance(sample, TokenSample)
    assert sample.tokens[0] == 1 and sample.tokens[-1] == 2 and sample.eligible[-1]
    with pytest.raises(ValueError):
        rollout_sample(RolloutRecord("w/0", "t", "", False, True, 0, 0, 0, 0), "p")


# ---------------------------------------------------------------------------
# H5: pointer decoder contracts
# ---------------------------------------------------------------------------

def test_h5_parameter_receipt_within_ceiling():
    from anra_v5.elendril.v6_registry import MODEL_PROFILES
    spec = MODEL_PROFILES["S3"]
    assert spec.parameter_receipt().total == 3_084_800
    projected = spec.parameter_receipt().total + 2 * spec.width + 1
    assert projected <= CausalPointerTwoReadDecoder.PARAMETER_CEILING
    assert projected == 3_084_800 + 2 * spec.width + 1


def test_h5_forward_is_finite_and_copy_support_lies_in_the_prompt():
    from v5_model.core import initialize, packed_layout
    spec = micro_spec()
    core = initialize(spec, 7)
    decoder = CausalPointerTwoReadDecoder(core)
    assert decoder.parameter_receipt()["within_ceiling"] is True
    tokens = torch.tensor([[1, 65, 66, 67, 3, 68, 69, 2]])
    segments = torch.zeros_like(tokens)
    positions, mask = packed_layout(segments, torch_module=torch)
    output = decoder(tokens, positions, mask)
    assert output.shape == (1, tokens.shape[1], 260)
    assert torch.isfinite(output).all()
    copy = decoder.copy_distribution(tokens, positions, mask)
    observed = {int(token) for token in tokens[0]}
    outside = [token_id for token_id in range(260) if token_id not in observed]
    assert copy[0, :, outside].sum().item() == pytest.approx(0.0, abs=1e-6)
    assert copy[0].sum(-1).max().item() == pytest.approx(1.0, abs=1e-5)


def test_h5_lineage_is_fresh_only(data, tmp_path):
    args = dict(data=data, job=job("H5-POINTER-READ"), source='d' * 40,
                root=tmp_path / "h5", **CANARY, spec_override=micro_spec(),
                updates_override=4)
    trainer = create_trainer(**args)
    trainer.step(); trainer.save()
    with pytest.raises(ValueError, match="fresh-only"):
        create_trainer(**args, resume=True)
    with pytest.raises(ValueError, match="mechanism lineages are fresh"):
        create_trainer(**{**args, "root": tmp_path / "h5b"},
                       transition=StageTransition("0" * 64, "0" * 64, "0" * 64, "0" * 64),
                       parent_store=trainer.store)


# ---------------------------------------------------------------------------
# H6: sparse replay buffer contracts
# ---------------------------------------------------------------------------

def _pool(count, tokens=(1, 65, 2)):
    return tuple(TokenSample(f"t{index}", tokens, (False, True, True), f"w{index}")
                 for index in range(count))


def test_h6_swap_never_adds_updates_and_ledgers_exposure():
    buffer = Ark017SurpriseReplayBuffer(_pool(10), fraction=0.1)
    base = tuple(range(16, 26))  # a registered batch of stage-sample indices
    before = buffer.ledger()
    incoming, swapped_out, ledger = buffer.plan_update(base)
    assert ledger["updates_added"] == 0
    assert len(incoming) == 1 and len(swapped_out) == 1
    assert ledger["swaps"] == before["swaps"] + 1
    buffer.observe(incoming[0], 2.0)
    assert buffer.surprise[incoming[0]] == pytest.approx(0.3 * 2.0)
    with pytest.raises(ValueError):
        Ark017SurpriseReplayBuffer(_pool(10), fraction=1.0)
    with pytest.raises(ValueError):
        buffer.observe(99, 1.0)
    with pytest.raises(ValueError):
        buffer.plan_update(())


def test_h6_factory_guards(data, tmp_path):
    with pytest.raises(NotImplementedError):
        create_trainer(data, dict(v716_protocol.jobs("continual")[0]), source='d' * 40,
                       root=tmp_path, **CANARY, spec_override=micro_spec())
    ark = v716_protocol.jobs("continual")[11]  # seed 41, ark017-sparse
    with pytest.raises(ValueError, match="stage samples"):
        create_trainer(data, ark, source='d' * 40, root=tmp_path, **CANARY,
                       spec_override=micro_spec())
    with pytest.raises(ValueError, match="not yet checkpoint-qualified|fresh"):
        create_trainer(data, ark, source='d' * 40, root=tmp_path, **CANARY,
                       spec_override=micro_spec(), resume=True,
                       stage_samples=_pool(4), replay_samples=_pool(4),
                       parent_store=object(), transition="t")


# ---------------------------------------------------------------------------
# Trainer factory: exposure locks and qualification
# ---------------------------------------------------------------------------

def test_trainer_rejects_unregistered_and_misqualified_jobs(data, tmp_path):
    with pytest.raises(ValueError, match="exactly"):
        create_trainer(data, {"job_id": "x"}, source='d' * 40, root=tmp_path,
                       **CANARY, spec_override=micro_spec())
    with pytest.raises(ValueError, match="outside the preregistered"):
        create_trainer(data, job("H1-COADAPT-MASK", seed=99), source='d' * 40,
                       root=tmp_path, **CANARY, spec_override=micro_spec())
    with pytest.raises(ValueError, match="no CPU fallback"):
        create_trainer(data, job("H1-COADAPT-MASK"), source='d' * 40, root=tmp_path,
                       device="cpu", spec_override=micro_spec())
    with pytest.raises(ValueError, match="1..8"):
        create_trainer(data, job("H1-COADAPT-MASK"), source='d' * 40, root=tmp_path,
                       **CANARY, spec_override=micro_spec(), updates_override=9)
    with pytest.raises(ValueError, match="executed-rollout provider"):
        create_trainer(data, job("H4-OUTCOME-DELIB"), source='d' * 40, root=tmp_path,
                       **CANARY, spec_override=micro_spec(), updates_override=4)


def test_every_arm_canary_trains_with_locked_exposure(data, tmp_path):
    for arm in ("H1-COADAPT-MASK", "H2-ARK014-INVARIANCE", "H2B-NORM",
                "H3-CONTRAST-MEM", "H4-OUTCOME-DELIB"):
        args = dict(data=data, job=job(arm), source='d' * 40, root=tmp_path / arm,
                    **CANARY, spec_override=micro_spec(), updates_override=4,
                    rollout_provider=(lambda update, chosen: tuple(chosen))
                    if arm == "H4-OUTCOME-DELIB" else None)
        trainer = create_trainer(**args)
        result = trainer.step()
        assert result["supervised_tokens"] > 0
        supervision = trainer.metadata["provenance"]["v716_supervision"]
        assert supervision["exposure_parity"] == "exact"
        assert supervision["supervised_tokens_per_epoch"] == \
            supervision["baseline_supervised_tokens_per_epoch"]
        assert supervision["mechanism"] == v716_protocol.ARM_MECHANISMS[arm]
        trainer.save()


def test_cpu_canary_checkpoint_round_trip_is_bitwise(data, tmp_path):
    args = dict(data=data, job=job("H1-COADAPT-MASK"), source='d' * 40,
                root=tmp_path / "round", **CANARY, spec_override=micro_spec(),
                updates_override=4)
    trainer = create_trainer(**args)
    trainer.step(); trainer.step(); trainer.save()
    resumed = create_trainer(**args, resume=True)
    assert resumed.metadata["plan_sha256"] == trainer.metadata["plan_sha256"]
    for _ in range(2):
        assert trainer.step() == resumed.step()
    assert all(torch.equal(a, b) for a, b in zip(trainer.model.parameters(),
                                                 resumed.model.parameters()))
    assert resumed.state.complete


def test_h3b_canary_applies_the_contrastive_penalty_once_per_epoch(data, tmp_path):
    args = dict(data=data, job=job("H3B-CONTRASTIVE"), source='d' * 40,
                root=tmp_path / "h3b", **CANARY, spec_override=micro_spec(),
                updates_override=4)
    trainer = create_trainer(**args)
    first = trainer.step()
    assert "contrastive_penalty" in first
    second = trainer.step()
    assert "contrastive_penalty" not in second  # once per epoch, at its first update
    trainer.save()


def test_pointer_canary_publishes_and_refuses_resume(data, tmp_path):
    args = dict(data=data, job=job("H5-POINTER-READ"), source='d' * 40,
                root=tmp_path / "h5pub", **CANARY, spec_override=micro_spec(),
                updates_override=4)
    trainer = create_trainer(**args)
    receipt = trainer.model.parameter_receipt()
    assert receipt["within_ceiling"] is True
    assert receipt["base"] == micro_spec().parameter_receipt().total
    trainer.step(); trainer.step(); trainer.save()
    with pytest.raises(ValueError, match="fresh-only"):
        create_trainer(**args, resume=True)


def test_architecture_digest_binds_the_pointer_version():
    spec = micro_spec()
    architecture = digest([spec.sha256(), CausalPointerTwoReadDecoder.VERSION])
    assert len(architecture) == 64
    assert architecture != digest([spec.sha256(), "other-version"])


# ---------------------------------------------------------------------------
# Diagnostics instruments
# ---------------------------------------------------------------------------

def test_trace_diff_reports_first_divergence():
    report = trace_diff_report([("canonical", ("A", "B"), "reversed", ("A", "C")),
                                ("same", ("A",), "same", ("A",))])
    assert report[0]["divergence"] == {"step": 1, "a": "B", "b": "C"}
    assert report[1]["identical"] is True
    assert first_divergence((), ("A",)) == {"step": 0, "a": "<ended>", "b": "A"}
    assert first_divergence((), ()) is None


def test_linear_probe_is_deterministic_and_separable():
    features = [[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, -0.1]] * 8
    labels = ["yes" if row[0] > 0.0 else "no" for row in features]  # separable by sign
    probe_a = LinearProbe(classes=("yes", "no")).fit(features, labels)
    probe_b = LinearProbe(classes=("yes", "no")).fit(features, labels)
    assert probe_a.accuracy(features, labels) > 0.95
    assert probe_a.weights.tolist() == probe_b.weights.tolist()
    with pytest.raises(ValueError):
        LinearProbe(classes=()).fit(features, labels)


def test_proxy_mutual_information_is_labeled_as_proxy():
    features = [[1.0, 0.0] if index % 2 else [0.0, 1.0] for index in range(40)]
    labels = ["other" if index % 2 else "value" for index in range(40)]
    result = proxy_mutual_information(features, labels, classes=("value", "other"))
    assert result["status"] == "PROXY_ESTIMATE_NOT_EXACT_MI"
    assert result["heldout_accuracy"] >= 0.9
    assert result["mi_bits_proxy"] >= 0.0
    with pytest.raises(ValueError):
        proxy_mutual_information([], [], classes=("a",))
    small = proxy_mutual_information([[0.1], [0.2], [0.3], [0.4]], ["a", "a", "b", "b"],
                                     classes=("a", "b"), steps=50)
    assert "mi_bits_proxy" in small


def test_e_family_aggregation_helpers():
    reports = {"anti/hop-1": {"evaluation": {"metrics": {"episodes": 48, "correct": 30,
                                                         "accuracy": 30 / 48}}},
               "anti/hop-2": {"evaluation": {"metrics": {"episodes": 48, "correct": 20,
                                                         "accuracy": 20 / 48}}}}
    hops = hop_accuracy(reports)
    assert hops["hops"]["1"] == pytest.approx(30 / 48)
    assert hops["hops"]["3"] is None  # missing conditions are recorded, not dropped
    delta = row_reversal_delta({"accuracy": 0.8}, {"accuracy": 0.2})
    assert delta["falsified_row_reversal"] is True
    assert stale_memory_score({})["status"] == "NOT_RUN"
    metrics = {"memory/stale-corrected": {"evaluation": {"metrics": {
        "episodes": 48, "correct": 30, "accuracy": 30 / 48}}}}
    assert stale_memory_score(metrics)["no_regression"] is True
    invalid = invalid_actions_total({"c|raw": {"evaluation": {"metrics": {"invalid": 10}}},
                                     "c|grammar": {"evaluation": {"metrics": {"invalid": 0}}}})
    assert invalid["invalid"] == {"raw": 10, "grammar": 0}
    assert invalid["recorded_reference"].startswith("861 raw")


# ---------------------------------------------------------------------------
# Review iteration: harness seams verified against their reused helpers
# ---------------------------------------------------------------------------

def test_ark017_sparse_stage_migration_loads_parent_weights(data, tmp_path):
    """The continual factory must forward the transition to the engine.

    Without the forwarding, ark017-sparse would silently train from scratch
    instead of loading the verified parent through ``migrate_stage``.
    """
    from dataclasses import asdict
    from anra_v5.elendril.v716_training import (_continual_provenance,
                                                _continual_recipe)
    spec = micro_spec()
    parent = create_trainer(data, job("H1-COADAPT-MASK"), source='d' * 40,
                            root=tmp_path / "parent", **CANARY,
                            spec_override=spec, updates_override=1)
    parent.step()
    parent.save()
    parent_model = [parameter.detach().clone() for parameter in parent.model.parameters()]
    _, payloads = parent.store.restore(parent.parent)
    parent_recipe = digest(json.loads(payloads["scheduler.json"])["metadata"]["recipe"])
    ark = next(entry for entry in v716_protocol.jobs("continual")
               if entry["stage"] == "ark017-sparse" and entry["seed"] == 7)
    stage_samples = tuple(data.samples)[:64]
    replay_samples = tuple(TokenSample(f"pool-{index}", (1, 65, 2),
                                       (False, True, True), f"w{index}",
                                       category="memory") for index in range(8))
    _, _, base_recipe = _continual_recipe(ark, stage_samples)
    recipe = replace(base_recipe, total_updates=2)  # the bounded CPU canary plan
    provenance, _ = _continual_provenance(data, ark, stage_samples, replay_samples)
    # The engine computes data_sha from the V7-enriched provenance; the bare
    # mapping would never satisfy the transition check.
    from anra_v5.elendril.v7_training import sampling_provenance
    child_data = digest({"samples": [asdict(sample) for sample in stage_samples],
                         "provenance": sampling_provenance(provenance, "blocked")})
    transition = StageTransition(parent.parent, parent_recipe,
                                 digest(asdict(recipe)), child_data)
    child = create_trainer(data, ark, source='d' * 40, root=tmp_path / "child",
                           **CANARY, spec_override=spec, updates_override=2,
                           parent_store=parent.store, transition=transition,
                           stage_samples=stage_samples, replay_samples=replay_samples)
    assert child.metadata["stage_transition"] == asdict(transition)
    assert child.state.global_update == 0  # fresh child lineage, AdamW reset
    assert all(torch.equal(loaded, saved) for loaded, saved
               in zip(child.model.parameters(), parent_model))
    with pytest.raises(ValueError, match="not yet checkpoint-qualified|fresh lineages"):
        create_trainer(data, ark, source='d' * 40, root=tmp_path / "child",
                       **CANARY, spec_override=spec, updates_override=2,
                       parent_store=parent.store, transition=transition,
                       stage_samples=stage_samples, replay_samples=replay_samples,
                       resume=True)



def test_v16_extracted_parent_rejects_source_spoof_and_missing_checkpoint(tmp_path):
    """A V16-looking identity is never a historical V7.1 checkpoint."""
    from tools.elendril_v716_campaign import verify_parent_output
    root = tmp_path / "parent"
    (root / "analysis").mkdir(parents=True)
    execution_source = "f" * 40
    identity = {"source": execution_source, "protocol_sha256": V71_SHA,
                "phase": "core"}
    (root / "analysis" / "identity.json").write_bytes(canonical(identity))
    with pytest.raises(ValueError, match="parent core identity mismatch"):
        verify_parent_output(root, execution_source, required_seeds=(7,))
    identity["source"] = v716_protocol.V71_CORE_SOURCE_SHA
    (root / "analysis" / "identity.json").write_bytes(canonical(identity))
    with pytest.raises(ValueError, match="required parent epoch-4 report missing"):
        verify_parent_output(root, execution_source, required_seeds=(7,))
    with pytest.raises(ValueError, match="registered seed"):
        verify_parent_output(root, execution_source, required_seeds=())


def test_v16_parent_restore_rejects_bad_final_epoch_before_gpu_canary(tmp_path):
    from tools.elendril_v716_campaign import verify_parent_output
    root = tmp_path / "parent"
    (root / "analysis" / "jobs" / "S3-blocked-ce-s7").mkdir(parents=True)
    identity = {"source": v716_protocol.V71_CORE_SOURCE_SHA,
                "protocol_sha256": V71_SHA, "phase": "core"}
    (root / "analysis" / "identity.json").write_bytes(canonical(identity))
    report = {"grammar": True, "update": 2368, "checkpoint": "a" * 64}
    (root / "analysis" / "jobs" / "S3-blocked-ce-s7" /
     "epoch-4.json").write_bytes(canonical(report))
    with pytest.raises(ValueError, match="not qualified"):
        verify_parent_output(root, "f" * 40, required_seeds=(7,))

def test_selected_jobs_slices_are_predeclared_and_exact():
    from tools.elendril_v716_campaign import selected_jobs
    assert len(selected_jobs("mechanism")) == 21
    h4 = selected_jobs("mechanism", arm_subset="H4-OUTCOME-DELIB")
    assert len(h4) == 3 and {entry["arm"] for entry in h4} == {"H4-OUTCOME-DELIB"}
    s19 = selected_jobs("mechanism", seed_subset=19)
    assert len(s19) == 7 and {entry["seed"] for entry in s19} == {19}
    continual = selected_jobs("continual", stage_subset="ark017-sparse")
    assert len(continual) == 3
    assert {entry["stage"] for entry in continual} == {"ark017-sparse"}
    promo = selected_jobs("promotion", promotion_arm="H2-ARK014-INVARIANCE")
    assert len(promo) == 3 and {entry["profile"] for entry in promo} == {"S8"}
    with pytest.raises(ValueError, match="mechanism arm"):
        selected_jobs("mechanism", arm_subset="H9-UNREGISTERED")
    with pytest.raises(ValueError, match="registered seed"):
        selected_jobs("mechanism", seed_subset=3)
    with pytest.raises(ValueError, match="continual stage"):
        selected_jobs("continual", stage_subset="fresh-c")
    with pytest.raises(ValueError, match="winning arm"):
        selected_jobs("promotion")


def test_profile_conditions_profile_one_s3_shape_for_continual():
    from tools.elendril_v716_campaign import _profile_conditions
    # Continual: exactly one measured S3 mechanism shape (H1), never fourteen.
    assert _profile_conditions("continual", []) == [("S3", "coadapt-mask")]
    mechanism = _profile_conditions("mechanism", v716_protocol.jobs("mechanism"))
    assert mechanism == sorted({("S3", name) for name
                                in v716_protocol.ARM_MECHANISMS.values()})
    promotion = _profile_conditions("promotion", v716_protocol.jobs("promotion"))
    assert ("S8", "coadapt-mask") in promotion and len(promotion) == 7


def test_promotion_gate_admits_only_the_preregistered_thresholds(tmp_path):
    from tools.elendril_v716_campaign import promotion_gate
    evidence = tmp_path / "evidence.json"
    evidence.write_bytes(canonical({
        "schema": "elendril-v716-promotion-evidence-v1",
        "stale_memory_corrected": 30, "row_reversal_consistency": 0.6}))

    def mechanism_output(correct_by_seed, *, raw_invalid_per_seed=0,
                         grammar_invalid=0):
        root = tmp_path / ("mech-" + str(abs(hash((tuple(correct_by_seed.items()),
                                                   raw_invalid_per_seed,
                                                   grammar_invalid)))))
        (root / "analysis").mkdir(parents=True, exist_ok=True)
        rows = []
        for seed in correct_by_seed:
            job_id = f"H4-OUTCOME-DELIB-s{seed}"
            rows.append({"job_id": job_id, "arm": "H4-OUTCOME-DELIB", "mode": "free",
                         "epoch": 4, "cohort": "base", "episodes": 112, "invalid": 0})
            rows.append({"job_id": job_id, "arm": "H4-OUTCOME-DELIB", "mode": "free",
                         "epoch": 4, "cohort": "memory-manipulations", "episodes": 112,
                         "invalid": raw_invalid_per_seed})
            rows.append({"job_id": job_id, "arm": "H4-OUTCOME-DELIB", "mode": "grammar",
                         "epoch": None, "cohort": "base", "episodes": 112,
                         "invalid": grammar_invalid})
        summary = {"status": "AVAILABLE", "primary_decoder": "raw-neural",
                   "rows": rows,
                   "arm_aggregates": {"H4-OUTCOME-DELIB": {
            "jobs": {f"H4-OUTCOME-DELIB-s{seed}": {
                "base": {"correct": correct, "episodes": 112}}
                for seed, correct in correct_by_seed.items()}}}}
        (root / "analysis" / "summary.json").write_bytes(canonical(summary))
        return root

    passing = promotion_gate("H4-OUTCOME-DELIB",
                             mechanism_output({7: 94, 19: 93, 41: 92},
                                              raw_invalid_per_seed=10), evidence)
    assert passing["schema"] == "elendril-v716-promotion-gate-v1"
    assert passing["checks"]["admitted"] is True
    assert passing["checks"]["aggregate_pass"] is True
    assert passing["checks"]["per_seed_pass"] is True
    invalid = passing["checks"]["invalid_actions"]
    assert invalid["raw"] == 30 and invalid["records"] == 672
    assert invalid["ceiling"] == 672 * 861 // 6720  # rate-matched registered ceiling
    assert invalid["grammar"] == 0 and invalid["pass"] is True
    with pytest.raises(ValueError, match="thresholds not met"):  # aggregate floor
        promotion_gate("H4-OUTCOME-DELIB",
                       mechanism_output({7: 96, 19: 95, 41: 85}), evidence)
    with pytest.raises(ValueError, match="thresholds not met"):  # per-seed floor
        promotion_gate("H4-OUTCOME-DELIB",
                       mechanism_output({7: 105, 19: 100, 41: 74}), evidence)
    with pytest.raises(ValueError, match="invalid-action rule failed"):  # raw ceiling
        promotion_gate("H4-OUTCOME-DELIB",
                       mechanism_output({7: 94, 19: 93, 41: 92},
                                        raw_invalid_per_seed=40), evidence)
    with pytest.raises(ValueError, match="invalid-action rule failed"):  # grammar != 0
        promotion_gate("H4-OUTCOME-DELIB",
                       mechanism_output({7: 94, 19: 93, 41: 92}, grammar_invalid=1),
                       evidence)
    weak_evidence = tmp_path / "weak.json"
    weak_evidence.write_bytes(canonical({
        "schema": "elendril-v716-promotion-evidence-v1",
        "stale_memory_corrected": 27, "row_reversal_consistency": 0.9}))
    with pytest.raises(ValueError, match="thresholds not met"):  # stale-memory floor
        promotion_gate("H4-OUTCOME-DELIB",
                       mechanism_output({7: 94, 19: 93, 41: 92}), weak_evidence)
    with pytest.raises(ValueError, match="evidence not supplied"):
        promotion_gate("H4-OUTCOME-DELIB",
                       mechanism_output({7: 94, 19: 93, 41: 92}), None)
    with pytest.raises(ValueError, match="no final evaluation"):
        promotion_gate("H9-UNREGISTERED",
                       mechanism_output({7: 94, 19: 93, 41: 92}), evidence)


def test_v16_report_tracks_final_epoch_across_multiple_epoch_files(tmp_path):
    """Regression for the real T4 owner run: the final-epoch tracker compared
    int > (epoch, data) tuple, so every epoch file after the first raised
    TypeError, dropped its cohort rows, and left epoch-1 metrics published as
    the arm's final aggregate."""
    from tools.elendril_v716_report import write_summary

    job_id = "H2B-NORM-s7"
    directory = tmp_path / "analysis" / "jobs" / job_id
    directory.mkdir(parents=True)

    def epoch_report(epoch, correct, invalid=0):
        return canonical({"grammar": False, "update": 592 * epoch,
                          "checkpoint": f"checkpoint-{epoch}",
                          "evaluation": {"cause_status": "CAUSE_UNRESOLVED",
                                         "metrics": {"base": {"all": {
                                             "correct": correct, "episodes": 112,
                                             "invalid": invalid}}}}})

    (directory / "epoch-1.json").write_bytes(epoch_report(1, 42))
    (directory / "epoch-2.json").write_bytes(epoch_report(2, 52, invalid=2))
    (directory / "epoch-4.json").write_bytes(epoch_report(4, 60, invalid=2))
    grammar = canonical({"grammar": True, "update": 2368,
                         "checkpoint": "checkpoint-4",
                         "evaluation": {"cause_status": "CAUSE_UNRESOLVED",
                                        "metrics": {"base": {"all": {
                                            "correct": 61, "episodes": 112}}}}})
    (directory / "grammar-final.json").write_bytes(grammar)

    report = write_summary(tmp_path, campaign={"registered_job_ids": [job_id]})
    assert report["errors"] == [], report["errors"]
    free_epochs = sorted(row["epoch"] for row in report["rows"] if row["mode"] == "free")
    assert free_epochs == [1, 2, 4]
    aggregate = report["arm_aggregates"]["H2B-NORM"]["base"]
    assert aggregate == {"correct": 60, "episodes": 112, "accuracy": 60 / 112}
    by_epoch = {(row["epoch"], row["mode"]): row["correct"] for row in report["rows"]}
    assert by_epoch[(1, "free")] == 42 and by_epoch[(2, "free")] == 52
    assert by_epoch[(4, "free")] == 60
    assert by_epoch[(None, "grammar")] == 61


def test_v16_primary_report_uses_raw_not_grammar_assisted_scores(tmp_path):
    """Grammar-verified perfection cannot authorize a failed raw policy."""
    from tools.elendril_v716_report import write_summary

    job_id = "H4-OUTCOME-DELIB-s7"
    directory = tmp_path / "analysis" / "jobs" / job_id
    directory.mkdir(parents=True)
    raw = {"grammar": False, "update": 2368, "checkpoint": "raw-checkpoint",
           "evaluation": {"cause_status": "CAUSE_UNRESOLVED",
                          "metrics": {"base": {"all": {"correct": 2, "episodes": 112}}}}}
    grammar = {"grammar": True, "update": 2368, "checkpoint": "same-checkpoint",
               "evaluation": {"cause_status": "CAUSE_UNRESOLVED",
                              "metrics": {"base": {"all": {"correct": 112,
                                                           "episodes": 112}}}}}
    (directory / "epoch-4.json").write_bytes(canonical(raw))
    (directory / "grammar-final.json").write_bytes(canonical(grammar))
    report = write_summary(tmp_path, campaign={"registered_job_ids": [job_id]})
    assert report["status"] == "AVAILABLE"
    assert report["primary_decoder"] == "raw-neural"
    assert report["arm_aggregates"]["H4-OUTCOME-DELIB"]["base"] == {
        "correct": 2, "episodes": 112, "accuracy": 2 / 112}
    assert any(row["mode"] == "grammar" and row["correct"] == 112
               for row in report["rows"])
    assert any(row["mode"] == "free" and row["correct"] == 2
               for row in report["rows"])


def test_v16_promotion_rejects_legacy_or_incomplete_primary_summaries(tmp_path):
    from tools.elendril_v716_campaign import promotion_gate

    path = tmp_path / "analysis" / "summary.json"
    path.parent.mkdir(parents=True)
    jobs = {f"H4-OUTCOME-DELIB-s{seed}": {"base": {"correct": 100,
                                                    "episodes": 112}}
            for seed in (7, 19, 41)}
    base = {"status": "AVAILABLE", "primary_decoder": "raw-neural",
            "arm_aggregates": {"H4-OUTCOME-DELIB": {"jobs": jobs}}}
    path.write_bytes(canonical({**base, "primary_decoder": "grammar"}))
    with pytest.raises(ValueError, match="raw-neural"):
        promotion_gate("H4-OUTCOME-DELIB", tmp_path, None)
    path.write_bytes(canonical({**base, "status": "PARTIAL"}))
    with pytest.raises(ValueError, match="complete"):
        promotion_gate("H4-OUTCOME-DELIB", tmp_path, None)
    wrong = {name: dict(value) for name, value in jobs.items()}
    wrong["H4-OUTCOME-DELIB-s7"] = {"base": {"correct": 100, "episodes": 111}}
    path.write_bytes(canonical({**base, "arm_aggregates": {
        "H4-OUTCOME-DELIB": {"jobs": wrong}}}))
    with pytest.raises(ValueError, match="112 valid raw observations"):
        promotion_gate("H4-OUTCOME-DELIB", tmp_path, None)


def test_report_identity_separates_delegated_stages_and_falls_back_exactly(tmp_path):
    from tools.elendril_v716_report import _identity

    def jobdir(name, payload):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "0001-request.json").write_bytes(canonical(payload))
        return directory

    delegated = jobdir("S3-b-only-ce-s7", {"job": {
        "job_id": "S3-b-only-ce-s7", "profile": "S3", "method": "blocked",
        "seed": 7, "epochs": 2, "mechanism": "ce", "stage": "b-only"}})
    assert _identity(delegated) == {"arm": "H6-ARK017-CONTINUAL", "profile": "S3",
                                    "mechanism": "continual", "seed": 7}
    native = jobdir("S3-ark017-sparse-ce-s41", {"job": {
        "job_id": "S3-ark017-sparse-ce-s41", "arm": "H6-ARK017-CONTINUAL",
        "profile": "S3", "seed": 41, "epochs": 2, "mechanism": "continual",
        "stage": "ark017-sparse", "exposure": "stage-transition"}})
    assert _identity(native)["arm"] == "H6-ARK017-CONTINUAL"
    # Fallbacks apply when a request file is missing; registered shapes only.
    empty = tmp_path / "H4-OUTCOME-DELIB-s7"
    empty.mkdir()
    assert _identity(empty) == {"arm": "H4-OUTCOME-DELIB", "profile": "S3",
                                "mechanism": "outcome-deliberation", "seed": 7}
    promoted = tmp_path / "S8-H2-ARK014-INVARIANCE-s41"
    promoted.mkdir()
    assert _identity(promoted) == {"arm": "H2-ARK014-INVARIANCE", "profile": "S8",
                                   "mechanism": "ark014-invariance", "seed": 41}
    replayed = tmp_path / "S3-replay-ce-s19"
    replayed.mkdir()
    assert _identity(replayed) == {"arm": "H6-ARK017-CONTINUAL", "profile": "S3",
                                   "mechanism": "continual", "seed": 19}
    unknown = tmp_path / "canary0"
    unknown.mkdir()
    assert set(_identity(unknown).values()) == {None}


def test_v16_kaggle_disabled_path_fails_loudly_without_gpu(tmp_path, monkeypatch):
    """Kaggle must never report a successful experiment when RUN_STUDY=False."""
    import sys
    import types
    from tools.build_elendril_v716_notebook import build

    ipython = types.ModuleType("IPython")
    ipython.__path__ = []
    display_module = types.ModuleType("IPython.display")
    display_module.FileLink = lambda path: path
    display_module.display = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", display_module)
    notebook = build("a" * 40)
    launch = "".join(notebook["cells"][2]["source"])
    with pytest.raises(RuntimeError, match="V16 NOT STARTED"):
        exec(compile(launch, "<v16-disabled-notebook>", "exec"), {
            "SOURCE_SHA": "a" * 40,
            "RUN_STUDY": False,
            "PHASE": "mechanism",
            "ARM_SUBSET": None,
            "SEED_SUBSET": None,
            "MINUTES": 330,
            "OUTPUT": tmp_path / "none",
        })
    assert not list(tmp_path.iterdir())  # no training, checkout or results


def test_notebook_builder_structure_defaults_pin_and_parent_guard():
    from tools.build_elendril_v716_notebook import build
    source = "a" * 40
    notebook = build(source)
    assert notebook["nbformat"] == 4 and len(notebook["cells"]) == 3
    config = "".join(notebook["cells"][1]["source"])
    launch = "".join(notebook["cells"][2]["source"])
    compile(config, "config", "exec")  # the owner cells must always parse
    compile(launch, "launch", "exec")
    assert "RUN_STUDY = False" in config  # safe default-off
    assert f'SOURCE_SHA = "{source}"' in config
    assert "tools.elendril_v716_campaign" in launch
    # An extracted parent output replaces the archive flag; never both.
    assert "PARENT_ARCHIVE is not None and parent is None" in launch
    assert "--parent-archive" in launch and "--parent-output" in launch
    assert notebook["metadata"]["source_reference"] == source
    with pytest.raises(ValueError):
        build("0" * 39)
    with pytest.raises(ValueError):
        build("A" * 40)


def test_v16_owner_launch_profile_is_explicit_and_keeps_master_safe():
    from tools.build_elendril_v716_notebook import build, build_owner_launch
    source = "a" * 40
    safe = build(source)
    ready = build_owner_launch(source)
    assert len(safe["cells"]) == 3 and len(ready["cells"]) == 4
    assert "RUN_STUDY = False" in "".join(safe["cells"][1]["source"])
    cfg = "".join(ready["cells"][1]["source"])
    preflight = "".join(ready["cells"][2]["source"])
    launch = "".join(ready["cells"][3]["source"])
    assert "RUN_STUDY = True" in cfg
    assert 'ARM_SUBSET = "H2B-NORM"' in cfg
    assert "SEED_SUBSET = 7" in cfg
    assert "PARENT_ARCHIVE = None" in cfg
    assert ready["metadata"]["source_reference"] == source
    assert ready["metadata"]["owner_enabled"] is True
    assert "PINNED_PARENT_SHA256" in preflight
    assert "V16 INPUT NO-GO" in preflight
    assert "tools.elendril_v716_campaign" in launch
    assert "RUN_STUDY = True" not in "".join(safe["cells"][1]["source"])
    for index, cell in enumerate(ready["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"owner-cell-{index}", "exec")


def test_v16_owner_launch_detects_only_sha_matching_zip(tmp_path):
    """CPU fixture: refuse decoys, locate an authentic exact-hash input."""
    from tools.build_elendril_v716_notebook import OWNER_FIRST_RUN_PREFLIGHT
    source = OWNER_FIRST_RUN_PREFLIGHT.split("# EXECUTE OWNER PREFLIGHT")[0]
    namespace = {}
    exec(compile(source, "owner-preflight-definitions", "exec"), namespace)
    good = tmp_path / "z" / "random-name.zip"
    good.parent.mkdir(parents=True)
    good.write_bytes(b"fixture-full-archive-input")
    wrong = tmp_path / "a.zip"
    wrong.write_bytes(b"wrong-parent")
    expected = hashlib.sha256(good.read_bytes()).hexdigest()
    assert namespace["_find_parent_archive"](tmp_path, expected) == good
    with pytest.raises(RuntimeError, match="none of the 2 attached ZIP files matches"):
        namespace["_find_parent_archive"](tmp_path, "0" * 64)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="no ZIP files"):
        namespace["_find_parent_archive"](empty, expected)



def test_v16_owner_input_diagnostic_reports_extracted_only_dataset(tmp_path, capsys):
    """The actual user error: a previous Kaggle output, but no FULL ZIP."""
    from tools.build_elendril_v716_notebook import OWNER_FIRST_RUN_PREFLIGHT
    namespace = {}
    source = OWNER_FIRST_RUN_PREFLIGHT.split("# EXECUTE OWNER PREFLIGHT")[0]
    exec(compile(source, "owner-preflight-definitions", "exec"), namespace)
    root = tmp_path / "input"
    extracted = root / "ELENDRIL_V715_KAGGLE_T4X2" / "elendril-v715-e5779-scale-s8"
    extracted.mkdir(parents=True)
    (extracted / "analysis.json").write_text('{"phase":"scale-s8"}')
    (extracted / "checkpoint.bin").write_bytes(b"not-a-full-zip")
    diagnostic = tmp_path / "working" / "V16_INPUT_DIAGNOSTIC.json"
    with pytest.raises(RuntimeError, match="no ZIP files"):
        namespace["_find_parent_archive"](root, "0" * 64, diagnostic)
    record = json.loads(diagnostic.read_text())
    assert record["status"] == "NO_ZIP_ARCHIVES"
    assert record["file_count"] == 2
    assert record["attached_datasets"] == ["ELENDRIL_V715_KAGGLE_T4X2"]
    assert "V16 input example:" in capsys.readouterr().out


def test_v16_owner_input_diagnostic_rejects_wrong_zip_and_records_hash(tmp_path):
    from tools.build_elendril_v716_notebook import OWNER_FIRST_RUN_PREFLIGHT
    namespace = {}
    source = OWNER_FIRST_RUN_PREFLIGHT.split("# EXECUTE OWNER PREFLIGHT")[0]
    exec(compile(source, "owner-preflight-definitions", "exec"), namespace)
    root = tmp_path / "input"
    root.mkdir()
    original = root / "wrong-analysis.zip"
    original.write_bytes(b"fake analysis archive")
    diagnostic = tmp_path / "working" / "V16_INPUT_DIAGNOSTIC.json"
    with pytest.raises(RuntimeError, match="none of the 1 attached ZIP files matches"):
        namespace["_find_parent_archive"](root, "f" * 64, diagnostic)
    report = json.loads(diagnostic.read_text())
    assert report["status"] == "ZIP_PRESENT_BUT_HASH_MISMATCH"
    assert report["zip_archives"][0]["sha256"] == hashlib.sha256(original.read_bytes()).hexdigest()
    expected = hashlib.sha256(original.read_bytes()).hexdigest()
    selected = namespace["_find_parent_archive"](root, expected, diagnostic)
    assert selected == original
    verified = json.loads(diagnostic.read_text())
    assert verified["status"] == "VERIFIED_PARENT_FULL"
    assert verified["zip_archives"][0]["parent_full_match"] is True

def test_v16_owner_parent_is_original_v71_core_not_v715_s8():
    from tools.build_elendril_v716_notebook import OWNER_FIRST_RUN_PREFLIGHT, OWNER_FIRST_RUN_INTRO
    assert 'PARENT_CORE_RUN_ID = "20c80f66e76e4b589aed2053afbddffc"' in OWNER_FIRST_RUN_PREFLIGHT
    assert "PARENT_CORE_BYTES = 1958029910" in OWNER_FIRST_RUN_PREFLIGHT
    assert "V7.1 CORE" in OWNER_FIRST_RUN_INTRO
    assert "2,933,151,710 bytes" in OWNER_FIRST_RUN_INTRO


def test_v16_owner_diagnostic_flags_mistaken_scale_s8_export(tmp_path):
    from tools.build_elendril_v716_notebook import OWNER_FIRST_RUN_PREFLIGHT
    namespace = {}
    exec(compile(OWNER_FIRST_RUN_PREFLIGHT.split("# EXECUTE OWNER PREFLIGHT")[0],
                 "preflight-definitions", "exec"), namespace)
    wrong = tmp_path / "input" / "v715-scale-s8" / "exports"
    wrong.mkdir(parents=True)
    (wrong / "ELENDRIL_V71_FULL.zip").write_bytes(b"synthetic wrong stage")
    report = tmp_path / "V16_INPUT_DIAGNOSTIC.json"
    with pytest.raises(RuntimeError, match="earlier V7.1 CORE Kaggle output"):
        namespace["_find_parent_archive"](tmp_path / "input", "f" * 64, report)
    evidence = json.loads(report.read_text())
    assert evidence["wrong_v715_scale_s8_detected"] is True
    assert evidence["zip_archives"][0]["looks_like_wrong_s8_stage"] is True
    assert evidence["expected_parent_bytes"] == 1958029910


def test_v16_worker_modes_agree_on_deadline_signature():
    """Regression for observed real T4 failure: recover(req) got 2 args."""
    import inspect
    from tools import elendril_v716_worker as worker
    for handler in (worker.train, worker.profile, worker.recover):
        assert tuple(inspect.signature(handler).parameters) == ("req", "deadline")


@pytest.mark.parametrize("mode", ("profile", "recover", "train"))
def test_v16_worker_cli_dispatch_smoke_cpu_mock(monkeypatch, tmp_path, mode):
    """Exercise the real CLI dispatch without allocating a GPU or training."""
    from tools import elendril_v716_worker as worker
    import time
    target = tmp_path / mode
    target.mkdir()
    request_file = tmp_path / f"{mode}-request.json"
    request = {
        "source": "a" * 40,
        "protocol_sha": v716_protocol.PROTOCOL_SHA,
        "timecap_seconds": 40,
        "output": str(target),
        "checkpoint_root": str(tmp_path / f"{mode}-checkpoints"),
        "phase": "mechanism",
        "job": next(j for j in v716_protocol.jobs("mechanism")
                    if j["arm"] == "H2B-NORM" and j["seed"] == 7),
    }
    request_file.write_bytes(canonical(request))
    receipt_path = target / "receipt.json"
    called = []

    def handler(req, deadline):
        called.append((req, deadline))
        assert deadline > time.monotonic()
        return {"status": "PASS", "qualified": True}
    monkeypatch.setattr(worker, mode, handler)
    monkeypatch.setattr(worker, "check_source", lambda *args: None)
    monkeypatch.setattr(worker.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(worker.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(worker.torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(worker.torch.cuda, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(worker.signal, "signal", lambda *args: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "mocked-physical-T4")
    assert worker.main([mode, "--owner-enabled",
                        "--request", str(request_file),
                        "--receipt", str(receipt_path)]) == 0
    assert len(called) == 1
    assert called[0][0] == request
    receipt = json.loads(receipt_path.read_bytes())
    assert receipt["status"] == "PASS"
    assert receipt["mode"] == mode
    assert receipt["gpu_uuid"] == "mocked-physical-T4"


def test_v16_worker_recover_rejects_expired_deadline_before_io():
    from tools import elendril_v716_worker as worker
    import time
    with pytest.raises(TimeoutError, match="recover deadline"):
        worker.recover({}, time.monotonic() - 1)


def test_v16_auto_fetch_original_core_strict_hash_and_source(tmp_path, monkeypatch):
    """No input needed when KaggleHub can provide the exact saved CORE archive."""
    from tools.build_elendril_v716_notebook import OWNER_FIRST_RUN_PREFLIGHT
    import sys
    import types
    namespace = {}
    source = OWNER_FIRST_RUN_PREFLIGHT.split("# EXECUTE OWNER PREFLIGHT")[0]
    exec(compile(source, "<auto-fetch-definitions>", "exec"), namespace)
    archive = tmp_path / "ELENDRIL_V71_FULL.zip"
    archive.write_bytes(b"unique historical core fixture")
    namespace["PARENT_CORE_BYTES"] = archive.stat().st_size
    hub = types.ModuleType("kagglehub")
    calls = []
    def remote(handle, *, path):
        calls.append((handle, path))
        return str(archive)
    hub.notebook_output_download = remote
    monkeypatch.setitem(sys.modules, "kagglehub", hub)
    expected = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert namespace["_auto_fetch_original_core"](expected) == archive
    assert calls == [("ankitrajm7c/elendril-v71-kaggle-t4x2",
                      "elendril-v71-6104ffd7134c-core/exports/ELENDRIL_V71_FULL.zip")]
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        namespace["_auto_fetch_original_core"]("0" * 64)
    namespace["PARENT_CORE_BYTES"] = 3
    with pytest.raises(RuntimeError, match="size mismatched"):
        namespace["_auto_fetch_original_core"](expected)


def test_v16_auto_fetch_official_path_present_in_owner_notebook():
    import json
    from pathlib import Path
    from tools.build_elendril_v716_notebook import build_owner_launch
    source = "a" * 40
    rebuilt = build_owner_launch(source)
    assert source in "".join(rebuilt["cells"][1]["source"])
    assert source in "".join(rebuilt["cells"][0]["source"])
    assert "_auto_fetch_original_core" in "".join(rebuilt["cells"][2]["source"])
    assert "AUTO_FETCHED_PARENT_VERIFIED" in "".join(rebuilt["cells"][2]["source"])
    assert "RUN_STUDY = True" in "".join(rebuilt["cells"][1]["source"])
    safe = json.loads(Path("notebooks/ELENDRIL_V716_HYPOTHESIS_KAGGLE_T4X2.ipynb").read_text())
    assert "RUN_STUDY = False" in "".join(safe["cells"][1]["source"])


def test_v16_recover_replays_real_h2b_checkpoint_on_bounded_cpu(
        data, tmp_path, monkeypatch):
    """Real checkpoint serializer/resume + exact state comparison, no GPU."""
    from tools import elendril_v716_worker as worker
    import time

    source = "d" * 40
    base = tmp_path / "expected"
    work = dict(data=data, job=job("H2B-NORM"), source=source,
                root=base, **CANARY, spec_override=micro_spec(),
                updates_override=8)
    trainer = create_trainer(**work)
    heads = {}
    for update in range(1, 5):
        trainer.step()
        if update in (2, 4):
            trainer.save()
            heads[update] = trainer.parent
    assert heads[2] != heads[4]

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "CPU-test-isolated-lane")
    monkeypatch.setattr(worker, "read_dataset", lambda _: data)

    def bounded_create(dataset, chosen_job, *, source, root, resume=False,
                       transition=None, parent_store=None, stage_samples=None,
                       replay_samples=None, rollout_provider=None):
        return create_trainer(
            dataset, chosen_job, source=source, root=root, resume=resume,
            transition=transition, parent_store=parent_store,
            stage_samples=stage_samples, replay_samples=replay_samples,
            rollout_provider=rollout_provider, **CANARY,
            spec_override=micro_spec(), updates_override=8)
    monkeypatch.setattr(worker, "create", bounded_create)

    challenge_file = tmp_path / "challenge.json"
    challenge_file.write_bytes(canonical({
        "source": source,
        "protocol_sha256": v716_protocol.PROTOCOL_SHA,
        "gpu_uuid": "CPU-test-isolated-lane",
        "expected_root": str(base),
        "head2": heads[2], "head4": heads[4],
    }))
    request = {
        "job": job("H2B-NORM"), "source": source,
        "challenge": str(challenge_file), "data_file": "fixture-data",
        "checkpoint_root": str(tmp_path / "recovery")
    }
    result = worker.recover(request, time.monotonic() + 60)
    assert result["status"] == "PASS"
    assert result["global_update"] == 4
    assert result["comparison"] == "exact full payload comparison"
    assert result["checkpoint"] == heads[4]


def test_v16_owner_launch_doc_matches_notebook_source_pin():
    """Regression: the launch doc froze the known-broken pin 6614864 after the
    notebooks were re-pinned to the recovery fix 682c40e1, so the operational
    instructions contradicted the executable source. All launch surfaces must
    name the same frozen source."""
    import json
    import re
    from pathlib import Path

    def notebook_pin(name):
        notebook = json.loads(Path("notebooks", name).read_text(encoding="utf-8"))
        config = "".join(notebook["cells"][1]["source"])
        match = re.search(r'SOURCE_SHA = "([0-9a-f]{40})"', config)
        assert match, f"{name} config cell has no pinned SOURCE_SHA"
        return match.group(1)

    pins = {name: notebook_pin(name) for name in (
        "ELENDRIL_V716_OWNER_LAUNCH_KAGGLE_T4X2.ipynb",
        "ELENDRIL_V716_HYPOTHESIS_KAGGLE_T4X2.ipynb")}
    assert len(set(pins.values())) == 1, "notebooks disagree on the frozen source"
    pin = next(iter(pins.values()))

    doc = Path("research", "v7.16", "OWNER_KAGGLE_LAUNCH.md").read_text(encoding="utf-8")
    freeze = re.search(r"source is frozen to `([0-9a-f]{40})`", doc)
    assert freeze, "launch doc does not state the frozen source"
    assert freeze.group(1) == pin, (
        "launch doc freeze line contradicts the notebook source pin")
