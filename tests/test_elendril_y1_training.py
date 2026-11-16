"""Training Group-CE gradient equivalence, restart parity, stage migration, and gate tests."""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

import pytest
import torch

from anra_v5.elendril.foundation_training import TokenSample
from anra_v5.elendril.representation import UTF8
from anra_v5.elendril.training import configure_cpu
from anra_v5.elendril.y1_data import build_development_suite
from anra_v5.elendril.y1_evaluation import (
    ORIGINAL_PRESERVATION_KINDS,
    evaluate_all_heuristics,
    evaluate_f11_bottleneck_admission,
    evaluate_t1_or_t2_gate,
)
from anra_v5.elendril.y1_training import CPU_CANARY_SPEC, create_y1_trainer, verify_group_ce_gradient_equivalence
from tools.elendril_y1_worker import _canary_eval_slice
from v5_training.checkpoint import CheckpointStore


def _tiny_samples(n: int = 16) -> tuple[TokenSample, ...]:
    out = []
    targets = ("A", "Q", "!")
    for i in range(n):
        pr = f"Latest time per source counts. Follow 1 links from A. Query={int(targets[i%3]=='Q')} Memory=0\nA=B@1000/src0; C=D@1001/src1; E=F@1002/log0\nAction:"
        fr = UTF8.frame(pr, targets[i % 3], max_length=256)
        out.append(TokenSample(f"t/{i}", fr.ids, fr.loss_mask, f"w/{i}", "route"))
    return tuple(out)


def test_y1_group_ce_microbatch_equivalence_restart_and_child_migration(tmp_path: Path) -> None:
    configure_cpu()
    samples = _tiny_samples(24)
    prov = {"schema": "test", "split": "train", "license": "repository LICENSE"}
    job = {"job_id": "Y1-train-test", "profile": "S3", "seed": 7, "updates": 2, "batch_size": 8, "micro_batch_size": 2, "lr": 1e-4, "group_ce_weight": 0.5}

    t1, _ = create_y1_trainer(job=job, samples=samples, data_provenance=prov, source_commit="64ed23a433ebb1ce2f620e0db8dcc24f4d8c23be", root=tmp_path / "r1", device="cpu", cpu_canary=True, spec_override=CPU_CANARY_SPEC, updates_override=2, batch_override=8)
    eq = verify_group_ce_gradient_equivalence(t1)
    assert eq["status"] == "PASS"

    r1 = t1.step()
    assert r1["group_ce_weight"] == 0.5 and math.isfinite(r1["group_ce_loss"])
    parent_sha = t1.save()
    r2_direct = t1.step()

    # Exact restart parity from checkpoint after update 1
    t_res, _ = create_y1_trainer(job=job, samples=samples, data_provenance=prov, source_commit="64ed23a433ebb1ce2f620e0db8dcc24f4d8c23be", root=tmp_path / "r1", device="cpu", cpu_canary=True, resume=True, spec_override=CPU_CANARY_SPEC, updates_override=2, batch_override=8)
    r2_resumed = t_res.step()
    assert r2_direct["loss"] == pytest.approx(r2_resumed["loss"], rel=1e-6, abs=1e-7)
    assert t_res.state.global_update == 2

    # Explicit child continuation resetting optimizer/schedule via StageTransition
    child_job = {**job, "job_id": "Y1-child-test", "updates": 1}
    pstore = CheckpointStore(tmp_path / "r1", job["job_id"])
    t_child, _ = create_y1_trainer(job=child_job, samples=samples, data_provenance=prov, source_commit="64ed23a433ebb1ce2f620e0db8dcc24f4d8c23be", root=tmp_path / "r1", device="cpu", cpu_canary=True, spec_override=CPU_CANARY_SPEC, updates_override=1, batch_override=8, parent_store=pstore, parent_checkpoint_sha=parent_sha)
    assert t_child.state.global_update == 0
    assert "stage_transition" in t_child.metadata
    t_child.step()


def test_y1_screening_confirmation_and_bottleneck_gates() -> None:
    # Build synthetic deterministic per-seed metric tables to verify all fail-closed gate conditions.
    fids = {f"f-{h}-{i}": (h, "answerable") for h in (1, 2, 3, 4) for i in range(10)}
    heur = {"last-row-object": {"primary_endpoint": 0.20}, "nearest-to-query-object": {"primary_endpoint": 0.20}, "highest-timestamp-object": {"primary_endpoint": 0.10}, "global-most-frequent-answer": {"primary_endpoint": 0.06}, "always-abstain": {"primary_endpoint": 0.0}, "random-legal": {"primary_endpoint": 0.05}}

    def mk_seed(prim: float, edge: float = 0.60, dq: float = 0.60, qry: float = 0.60, val: float = 0.98, un: float = 0.95, pres: float = 0.80):
        return {
            "primary_endpoint": prim,
            "guardrails": {
                "edge_both_correct_all_three": edge,
                "dual_query_both_correct": dq,
                "query_needed_raw_terminal_rate": qry,
                "valid_action_rate": val,
                "unavailable_abstain_rate": un,
            },
            "preservation_suite": {k: {"accuracy": pres} for k in ORIGINAL_PRESERVATION_KINDS},
            "family_cluster_scores": {fid: prim for fid in fids},
            "family_strata": {fid: list(st) for fid, st in fids.items()},
        }

    arms = {
        "F00": {7: mk_seed(0.10), 19: mk_seed(0.10), 41: mk_seed(0.10)},
        "F01": {7: mk_seed(0.55), 19: mk_seed(0.55), 41: mk_seed(0.55)},
        "F10": {7: mk_seed(0.55), 19: mk_seed(0.55), 41: mk_seed(0.55)},
        # F11 fails preservation regression in seed 41
        "F11": {7: mk_seed(0.70), 19: mk_seed(0.70), 41: mk_seed(0.70, pres=0.70)},
    }
    gate = evaluate_t1_or_t2_gate(stage="t1_factorial", arms_metrics=arms, heuristics=heur, require_ci_lower_gt_zero=True)
    assert gate["status"] == "PASS"
    assert gate["locked_winner"] == "F01"  # Tie within 1e-12 between F01 and F10 resolves F01 > F10 > F11.
    assert gate["arms"]["F11"]["status"] == "FAIL"

    bn = evaluate_f11_bottleneck_admission({
        7: {"known_base_rank1_within_ap_rate": 0.75, "known_base_raw_exact_rate": 0.30, "constrained_edge_both_correct": 0.55, "constrained_dual_query_both_correct": 0.55},
        19: {"known_base_rank1_within_ap_rate": 0.72, "known_base_raw_exact_rate": 0.30, "constrained_edge_both_correct": 0.52, "constrained_dual_query_both_correct": 0.51},
        41: {"known_base_rank1_within_ap_rate": 0.71, "known_base_raw_exact_rate": 0.30, "constrained_edge_both_correct": 0.50, "constrained_dual_query_both_correct": 0.50},
    })
    assert bn["admitted"] is True and bn["recommended_branch"] == "targeted_group_ce"
