"""CPU tests for V8 training, crash recovery, evaluation, pointer interventions, and conditional modules."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import pytest
import torch

from anra_v5.elendril.codec import digest
from anra_v5.elendril.foundation_training import Recipe, StageTransition
from anra_v5.elendril.v6_registry import MODEL_PROFILES
from anra_v5.elendril.v8.analysis import evaluate_advancement_gate, paired_world_comparison
from anra_v5.elendril.v8.contracts import build_v8_model_spec
from anra_v5.elendril.v8.data import build_d1_dev_suite, build_d1_training_dataset
from anra_v5.elendril.v8.evaluation import (
    diagnostic_probe,
    evaluate_a20_recovery_extension,
    evaluate_f03_reverse_extension,
    evaluate_worlds_by_mode,
    run_pointer_interventions,
    summarize_suite_episodes,
)
from anra_v5.elendril.v8.memory import evaluate_e16_memory_regimes
from anra_v5.elendril.v8.protocol import jobs
from anra_v5.elendril.v8.replay import (
    GuardianMonitorState,
    ReplayQualification,
    build_stratified_support_index,
    sample_fixed_replay_batch,
    verify_retention_prerequisites,
)
from anra_v5.elendril.v8.training import V8TypedTrainer, create_v8_trainer, prepare_v8_training_data
from anra_v5.elendril.v8.transfer import (
    evaluate_transfer_gain,
    right_censored_acquisition,
    verify_transfer_prerequisites,
)
from v5_training.checkpoint import InjectedCrash


def _tiny_base():
    return replace(
        MODEL_PROFILES["S20"],
        width=32,
        layers=1,
        query_heads=2,
        kv_heads=1,
        head_dimension=16,
        ffn_width=64,
        context_length=512,
    )


def test_v8_cpu_rejects_oversized_s20_and_supports_d0_d1_accum_and_migration(tmp_path: Path) -> None:
    d1 = prepare_v8_training_data("D1", d1_worlds_per_family=16, sample_limit_for_cpu=16)
    s20_p = build_v8_model_spec("P")
    with pytest.raises(ValueError, match="tiny fixture bounds"):
        create_v8_trainer(
            d1,
            jobs("reader-screen")[0],
            source="a" * 40,
            root=tmp_path / "oversized",
            device="cpu",
            cpu_canary=True,
            v8_spec_override=s20_p,
            updates_override=2,
        )

    tiny = _tiny_base()
    # B0 D0 & D1
    d0 = prepare_v8_training_data("D0", sample_limit_for_cpu=16)
    for j, d in [(jobs("data-control")[0], d0), (jobs("data-control")[3], d1)]:
        tr = create_v8_trainer(
            d,
            j,
            source="a" * 40,
            root=tmp_path / f"b0_{j['dataset']}",
            device="cpu",
            cpu_canary=True,
            v8_spec_override=build_v8_model_spec("B0", base_spec=tiny),
            updates_override=2,
        )
        tr.step()
        tr.step()
        tr.save()

    # Gradient accumulation parity: micro=1 vs micro=8 on P
    spec_p = build_v8_model_spec("P", base_spec=tiny, pointer_dim=16)
    r1 = Recipe(seed=7, total_updates=2, batch_size=8, micro_batch_size=1, lr=1e-4, schedule="cosine", precision="fp32")
    r8 = Recipe(seed=7, total_updates=2, batch_size=8, micro_batch_size=8, lr=1e-4, schedule="cosine", precision="fp32")
    t1 = V8TypedTrainer(
        d1.typed_samples, v8_spec=spec_p, recipe=r1, order_seed=7, source_commit="a" * 40,
        root=tmp_path / "acc1", lineage="acc1", data_provenance=d1.manifest, device="cpu", cpu_canary=True,
    )
    t8 = V8TypedTrainer(
        d1.typed_samples, v8_spec=spec_p, recipe=r8, order_seed=7, source_commit="a" * 40,
        root=tmp_path / "acc8", lineage="acc8", data_provenance=d1.manifest, device="cpu", cpu_canary=True,
    )
    t1.step()
    t8.step()
    for k in t1.model.state_dict():
        assert torch.allclose(t1.model.state_dict()[k], t8.model.state_dict()[k], atol=1e-6)


@pytest.mark.parametrize(
    "crash_point",
    [
        "after_component:model.bin",
        "after_manifest",
        "after_stage",
        "after_publish_before_pointer",
        "after_pointer_stage",
    ],
)
def test_v8_interrupted_restore_parity_at_every_crash_point(tmp_path: Path, crash_point: str) -> None:
    d1 = prepare_v8_training_data("D1", d1_worlds_per_family=16, sample_limit_for_cpu=16)
    spec = build_v8_model_spec("P", base_spec=_tiny_base(), pointer_dim=16)
    job = jobs("reader-screen")[0]
    src = "a" * 40

    t_cont = create_v8_trainer(
        d1, job, source=src, root=tmp_path / "cont", device="cpu", cpu_canary=True,
        v8_spec_override=spec, updates_override=4,
    )
    for _ in range(4):
        t_cont.step()

    res_root = tmp_path / f"res_{crash_point.replace(':', '_')}"
    t_a = create_v8_trainer(
        d1, job, source=src, root=res_root, device="cpu", cpu_canary=True,
        v8_spec_override=spec, updates_override=4,
    )
    t_a.step()
    t_a.step()
    sha2 = t_a.save()

    with pytest.raises(InjectedCrash):
        t_a.step()
        t_a.save(inject_crash_at=crash_point)

    t_b = create_v8_trainer(
        d1, job, source=src, root=res_root, resume=True, device="cpu", cpu_canary=True,
        v8_spec_override=spec, updates_override=4,
    )
    assert t_b.state.global_update == 2
    assert t_b.parent == sha2
    t_b.step()
    t_b.step()
    for k in t_cont.model.state_dict():
        assert torch.equal(t_cont.model.state_dict()[k], t_b.model.state_dict()[k])


def test_v8_stage_transition_and_mutated_checkpoint_rejection(tmp_path: Path) -> None:
    d1 = prepare_v8_training_data("D1", d1_worlds_per_family=16, sample_limit_for_cpu=16)
    spec = build_v8_model_spec("F", base_spec=_tiny_base())
    job = jobs("head-screen")[3]
    src = "a" * 40
    root = tmp_path / "parent"
    tr = create_v8_trainer(
        d1, job, source=src, root=root, device="cpu", cpu_canary=True,
        v8_spec_override=spec, updates_override=2,
    )
    tr.step()
    c1 = tr.save()

    child_rec = Recipe(seed=7, total_updates=2, batch_size=8, micro_batch_size=2, lr=1e-4, schedule="cosine", precision="fp32")
    probe_tr = V8TypedTrainer(
        d1.typed_samples, v8_spec=spec, recipe=child_rec, order_seed=7, source_commit=src,
        root=tmp_path / "probe", lineage="child", data_provenance=d1.manifest, device="cpu", cpu_canary=True,
    )
    parent_rec = json.loads(tr.store.restore(c1)[1]["scheduler.json"])["metadata"]["recipe"]
    trans = StageTransition(
        parent_checkpoint=c1,
        parent_recipe=digest(parent_rec),
        child_recipe=digest(asdict(child_rec)),
        child_data=probe_tr.data_sha,
    )
    child = V8TypedTrainer(
        d1.typed_samples, v8_spec=spec, recipe=child_rec, order_seed=7, source_commit=src,
        root=tmp_path / "child_real", lineage="child", data_provenance=d1.manifest,
        device="cpu", cpu_canary=True, transition=trans, parent_store=tr.store,
    )
    assert len(child.optimizer.state) == 0
    for k in child.model.state_dict():
        assert torch.equal(child.model.state_dict()[k], tr.model.state_dict()[k])

    # Mutate archived object byte and ensure restore rejects it
    obj_model = tr.store.objects / c1 / "model.bin"
    raw = bytearray(obj_model.read_bytes())
    raw[-1] ^= 0xFF
    obj_model.write_bytes(bytes(raw))
    with pytest.raises(ValueError):
        tr.store.restore(c1)


def test_v8_eval_pointer_interventions_and_conditional_w8_gates(tmp_path: Path) -> None:
    d1 = prepare_v8_training_data("D1", d1_worlds_per_family=16, sample_limit_for_cpu=16)
    spec_p = build_v8_model_spec("P", base_spec=_tiny_base(), pointer_dim=16)
    tr = create_v8_trainer(
        d1, jobs("reader-screen")[0], source="a" * 40, root=tmp_path / "p_eval",
        device="cpu", cpu_canary=True, v8_spec_override=spec_p, updates_override=2,
    )
    tr.step()

    cohorts = build_d1_dev_suite(worlds_per_cohort=2)
    worlds = [w for rows in cohorts.values() for w in rows][:15]
    eps = evaluate_worlds_by_mode(tr.model, spec_p, worlds, model_identity="P")
    summary = summarize_suite_episodes(eps, worlds)
    assert set(summary) == {"raw", "grammar", "availability-mask", "direct-answer-only"}
    d1_raw = build_d1_training_dataset(quota_per_family=16)
    diag = diagnostic_probe(tr.model, spec_p, d1_raw.steps, max_probe_steps=16)
    assert diag["architecture"] == "P"
    interv = run_pointer_interventions(tr.model, spec_p, d1_raw.steps[:32])
    assert set(["baseline_pointer", "no_copy_control", "cyclic_value_permutation", "sham_value_permutation"]).issubset(interv)
    assert evaluate_f03_reverse_extension(tr.model, spec_p, model_identity="P")["scenario"] == "F03"
    assert evaluate_a20_recovery_extension(tr.model, spec_p, worlds, model_identity="P")["scenario"] == "A20"

    comp = paired_world_comparison(eps["raw"], eps["grammar"], arm_a="P-raw", arm_b="P-gram", bootstrap_samples=50)
    assert comp["worlds"] == len(worlds)
    gate_fail = evaluate_advancement_gate(
        [{"paired_gain": 0.02, "candidate_raw_r09": 0.5, "candidate_relevant_cf_both_correct": 0.4, "candidate_irrelevant_cf_stability": 0.8}]
    )
    assert gate_fail["qualified"] is False

    # W8 gates & guardian state
    unqual = verify_retention_prerequisites(
        ReplayQualification("A", 0.70, 0.80, "a" * 40, "b" * 64),
        ReplayQualification("B", 0.85, 0.80, "a" * 40, "c" * 64),
    )
    assert unqual["status"] == "PREREQUISITE_NOT_QUALIFIED"

    g = GuardianMonitorState()
    g.observe_stratum_errors(64, {0: 0.9, 1: 0.1})
    g2 = GuardianMonitorState.from_dict(g.to_dict())
    assert g2.ema_errors == g.ema_errors
    sup = build_stratified_support_index(d1.typed_samples)
    batch = sample_fixed_replay_batch(d1.typed_samples, sup, d1.typed_samples, regime="G", b_indices=(0, 1, 2, 3), step_seed=5, guardian=g2)
    assert len(batch) == 8

    # Frozen-weight E16 check
    assert evaluate_e16_memory_regimes(tr.model, spec_p, worlds[:4], model_identity="P", prerequisite_qualified=False)["status"] == "PREREQUISITE_NOT_QUALIFIED"
    e16 = evaluate_e16_memory_regimes(tr.model, spec_p, worlds[:4], model_identity="P", prerequisite_qualified=True)
    assert e16["frozen_weights_verified"] is True

    # Right-censored C24 transfer check
    assert verify_transfer_prerequisites(retention_qualified=False, c_achievable_accuracy=0.85)["status"] == "PREREQUISITE_NOT_QUALIFIED"
    rc = right_censored_acquisition([{"updates": 64, "accuracy": 0.45}, {"updates": 128, "accuracy": 0.62}], threshold=0.80)
    assert rc["right_censored"] is True and rc["right_censored_at_updates"] == 128
    tg = evaluate_transfer_gain({
        "fresh": [{"updates": 64, "examples": 100, "accuracy": 0.60}, {"updates": 128, "examples": 200, "accuracy": 0.82}],
        "ab_qualified_replay": [{"updates": 64, "examples": 100, "accuracy": 0.84}],
    })
    assert tg["meets_transfer_rule"] is True
