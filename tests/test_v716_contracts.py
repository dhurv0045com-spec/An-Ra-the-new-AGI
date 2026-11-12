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
