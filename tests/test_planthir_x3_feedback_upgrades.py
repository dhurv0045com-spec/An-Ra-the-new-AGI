"""Verification tests for the three V8-X3 PALANTIRI_FEEDBACK upstream Level-6 upgrades."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anra_v5.planthir import (
    CampaignOrchestrator,
    Palantiri,
    SessionJob,
    TypedByteSequenceKernel,
    UniversalAction,
    canonical_ast_hash,
    canonical_graph_isomorphism_hash,
    canonical_horn_clause_hash,
    canonical_structural_hash,
    infer_public_byte_bound,
    parse_typed_output_envelope,
    prune_completed_job_artifacts,
    score_typed_output,
    verify_split_disjointness,
)


def test_typed_byte_sequence_scoring_and_kernel_zero_mod16_collisions() -> None:
    env_ok = parse_typed_output_envelope("ANSWER: 42 kg<EOS>")
    assert env_ok["valid"] is True
    assert env_ok["op"] == "ANSWER"
    assert env_ok["payload"] == "42 kg"
    assert env_ok["has_explicit_op"] is True

    env_abs = parse_typed_output_envelope("ABSTAIN<EOS>")
    assert env_abs["valid"] is True and env_abs["op"] == "ABSTAIN" and env_abs["payload"] == ""

    env_bad = parse_typed_output_envelope("17")
    assert env_bad["valid"] is False and env_bad["error_code"] == "MISSING_TYPED_OP_ENVELOPE"

    # Zero mod-16 collisions ("17" % 16 == "1" % 16, yet exact scoring separates them cleanly)
    mod_collision = score_typed_output("p", "ANSWER:17<EOS>", expected_payload="1")
    assert mod_collision["exact_match"] is False
    assert mod_collision["failure_reason"] == "VALUE_MISMATCH"
    assert score_typed_output("p", "ANSWER:17<EOS>", expected_payload="17")["exact_match"] is True

    # Rational + unit equivalence vs unit mismatch
    rat_ok = score_typed_output("p", "ANSWER: 3/4 kg<EOS>", expected_payload="0.75 kg", value_kind="rational_unit")
    assert rat_ok["exact_match"] is True
    rat_bad_unit = score_typed_output("p", "ANSWER: 0.75 g<EOS>", expected_payload="0.75 kg", value_kind="rational_unit")
    assert rat_bad_unit["exact_match"] is False and rat_bad_unit["failure_reason"] == "UNIT_MISMATCH"
    rat_no_unit = score_typed_output("p", "ANSWER: 0.75<EOS>", expected_payload="0.75 kg", value_kind="rational_unit")
    assert rat_no_unit["exact_match"] is False and rat_no_unit["failure_reason"] == "UNIT_MISMATCH"

    # Grid normalization & shape verification
    grid_ok = score_typed_output("GRID=2x2 Q:12/34->?", "ANSWER: 12 / 34<EOS>", expected_payload="12/34", value_kind="grid")
    assert grid_ok["exact_match"] is True
    grid_bad_shape = score_typed_output(
        "GRID=2x2 Q:12/34->?", "ANSWER: 123 / 456<EOS>", expected_payload="12/34", value_kind="grid"
    )
    assert grid_bad_shape["exact_match"] is False and grid_bad_shape["failure_reason"] == "GRID_SHAPE_MISMATCH"

    # Public shape-derived byte bounds + overflow/unterminated checks
    assert infer_public_byte_bound("MAX_BYTES=20") == 20
    assert infer_public_byte_bound("Q:1234/5678/9012/3456->?") > 24
    ov = score_typed_output("MAX_BYTES=20", "ANSWER:" + "9" * 20, expected_payload="9" * 20)
    assert ov["overflow"] is True and ov["exact_match"] is False and ov["failure_reason"] == "BYTE_BOUND_OVERFLOW"
    unterm = score_typed_output("p", "ANSWER:ok", expected_payload="ok", terminated_by_eos=False)
    assert unterm["unterminated"] is True and unterm["exact_match"] is False and unterm["failure_reason"] == "UNTERMINATED_NO_EOS"

    # TypedByteSequenceKernel truth firewall & verification contract
    kern = TypedByteSequenceKernel(default_value_kind="exact")
    w_pub = {
        "world_id": "tb_0",
        "prompt": "MAX_BYTES=32 compute result",
        "edges": [("Alice", "Bob")],
        "hidden": {"expected": "17"},
    }
    w_mut = {
        "world_id": "tb_0",
        "prompt": "MAX_BYTES=32 compute result",
        "edges": [("Alice", "Bob")],
        "hidden": {"expected": "999"},
    }
    obs_a = kern.observe(kern.initial_state(w_pub))
    obs_b = kern.observe(kern.initial_state(w_mut))
    assert obs_a.rendered_text == obs_b.rendered_text
    assert obs_a.metadata == obs_b.metadata and obs_a.metadata["public_byte_bound"] == 32
    assert kern.ideal_action(kern.initial_state(w_pub)) == ("ANSWER", "17")

    good_act = UniversalAction(
        "ANSWER", "17", -0.01, {"ANSWER": 1.0}, "typed", "ANSWER", "17",
        metadata={"raw_output": "ANSWER:17<EOS>", "terminated_by_eos": True},
    )
    bad_act = UniversalAction(
        "ANSWER", "1", -0.01, {"ANSWER": 1.0}, "typed", "ANSWER", "1",
        metadata={"raw_output": "ANSWER:1<EOS>", "terminated_by_eos": True},
    )
    assert kern.verify(w_pub, [good_act])["correct"] is True
    assert kern.verify(w_pub, [bad_act])["correct"] is False
    assert Palantiri.score_typed_output("p", "ANSWER:17<EOS>", expected_payload="1")["exact_match"] is False
    assert Palantiri.infer_public_byte_bound("MAX_BYTES=44") == 44


def test_canonical_ast_graph_horn_structural_hashing_and_split_disjointness() -> None:
    # 1. Renamed isomorphic graphs produce identical hash; non-isomorphic directed topology differs
    g1 = [("Alice", "Bob", "e"), ("Bob", "Carol", "e"), ("Carol", "Dave", "e")]
    g2 = [("N9", "N4", "e"), ("N4", "N1", "e"), ("N1", "N7", "e")]
    g_flipped = [("Alice", "Bob", "e"), ("Carol", "Bob", "e"), ("Carol", "Dave", "e")]
    assert canonical_graph_isomorphism_hash(g1, directed=True) == canonical_graph_isomorphism_hash(list(reversed(g2)), directed=True)
    assert canonical_graph_isomorphism_hash(g1, directed=True) != canonical_graph_isomorphism_hash(g_flipped, directed=True)

    # 2. Renamed predicates, renamed entities, and reordered Horn clauses produce identical structural hash
    h1 = [
        {"body": ["edge(Alice,Bob)", "edge(Bob,Carol)"], "head": "reach(Alice,Carol)"},
        {"body": ["edge(Carol,Dave)"], "head": "leaf(Dave)"},
    ]
    h2 = [
        {"body": ["rel(X3,X4)"], "head": "sink(X4)"},
        {"body": ["rel(X2,X3)", "rel(X1,X2)"], "head": "path(X1,X3)"},
    ]
    h_bad = [
        {"body": ["edge(Alice,Bob)", "edge(Bob,Carol)"], "head": "reach(Carol,Alice)"},  # reversed head args
        {"body": ["edge(Carol,Dave)"], "head": "leaf(Dave)"},
    ]
    assert canonical_horn_clause_hash(h1, query="reach(Alice,Carol)") == canonical_horn_clause_hash(h2, query="path(X1,X3)")
    assert canonical_horn_clause_hash(h1, query="reach(Alice,Carol)") != canonical_horn_clause_hash(h_bad, query="reach(Alice,Carol)")

    # 3. Alpha-renamed AST hash
    ast1 = {"op": "chain", "entity": "Alpha", "next": {"person": "Beta", "steps": 2}}
    ast2 = {"steps": 2, "next": {"person": "Q2", "steps": 2}, "entity": "Q1", "op": "chain"}
    ast3 = {"op": "chain", "entity": "Alpha", "next": {"person": "Beta", "steps": 3}}
    assert canonical_ast_hash(ast1) == canonical_ast_hash({"op": "chain", "entity": "Q1", "next": {"person": "Q2", "steps": 2}})
    assert canonical_ast_hash(ast1) != canonical_ast_hash(ast3)

    # 4. Split disjointness catches renamed isomorphic Horn/graph leak even when world_id and prompt differ
    train_rec = {"world_id": "tr_1", "prompt": "Train Alice->Bob->Carol", "horn_clauses": h1, "query": "reach(Alice,Carol)"}
    eval_leak = {"world_id": "ev_99", "prompt": "Eval X1->X2->X3", "horn_clauses": h2, "query": "path(X1,X3)"}
    eval_clean = {
        "world_id": "ev_ok",
        "prompt": "Eval clean",
        "horn_clauses": h_bad,
        "query": "reach(Alice,Carol)",
    }

    assert canonical_structural_hash(train_rec) == canonical_structural_hash(eval_leak)
    with pytest.raises(ValueError, match="SPLIT_DISJOINTNESS_VIOLATION"):
        verify_split_disjointness({"train": [train_rec], "eval": [eval_leak]}, fail_closed=True)

    rep_bad = Palantiri.verify_splits({"train": [train_rec], "eval": [eval_leak]}, fail_closed=False)
    assert rep_bad["passed"] is False and rep_bad["structural_leak_count"] == 1

    rep_ok = Palantiri.verify_splits({"train": [train_rec], "eval": [eval_clean]}, fail_closed=True)
    assert rep_ok["passed"] is True and rep_ok["structural_leak_count"] == 0


def _make_fake_ckpt(job_dir: Path) -> Path:
    lin = job_dir / "ckpt" / "main"
    s_old = "a" * 64
    s_cur = "b" * 64
    (lin / "objects" / s_old).mkdir(parents=True, exist_ok=True)
    (lin / "objects" / s_cur).mkdir(parents=True, exist_ok=True)
    for sha in (s_old, s_cur):
        (lin / "objects" / sha / "model.bin").write_bytes(b"weights" * 32)
        (lin / "objects" / sha / "optimizer.bin").write_bytes(b"adamstate" * 64)
        (lin / "objects" / sha / "rng.bin").write_bytes(b"rngstate" * 8)
    (lin / "LATEST").write_text(s_cur + "\n", encoding="utf-8")
    (lin / "PREVIOUS").write_text(s_old + "\n", encoding="utf-8")
    return lin / "objects" / s_cur


def test_dependency_aware_compaction_and_parent_child_protection(tmp_path: Path) -> None:
    # 1. Direct unit checks: child protection & incomplete full_job_updates block optimizer stripping
    d_parent = tmp_path / "u_parent"
    cur_p = _make_fake_ckpt(d_parent)
    pr_parent = prune_completed_job_artifacts(
        d_parent,
        strip_final_optimizer=True,
        preserve_final_optimizer=False,
        child_job_ids=["R7"],
        completed_updates=20,
        full_job_updates=20,
    )
    assert pr_parent["optimizer_stripped"] is False and pr_parent["optimizer_preserved"] is True
    assert "HAS_DOWNSTREAM_CHILDREN:R7" in pr_parent["preservation_reasons"]
    assert (cur_p / "optimizer.bin").is_file()
    assert (cur_p / "rng.bin").is_file()

    d_mid = tmp_path / "u_mid"
    cur_m = _make_fake_ckpt(d_mid)
    pr_mid = prune_completed_job_artifacts(
        d_mid,
        strip_final_optimizer=True,
        preserve_final_optimizer=False,
        completed_updates=10,
        full_job_updates=20,
    )
    assert pr_mid["optimizer_stripped"] is False
    assert "INCOMPLETE_FULL_JOB_UPDATES:10<20" in pr_mid["preservation_reasons"]
    assert (cur_m / "optimizer.bin").is_file()

    d_leaf = tmp_path / "u_leaf"
    cur_l = _make_fake_ckpt(d_leaf)
    pr_leaf = prune_completed_job_artifacts(
        d_leaf,
        strip_final_optimizer=True,
        preserve_final_optimizer=False,
        completed_updates=20,
        full_job_updates=20,
    )
    assert pr_leaf["optimizer_stripped"] is True and pr_leaf["optimizer_preserved"] is False
    assert not (cur_l / "optimizer.bin").exists()
    assert (cur_l / "model.bin").is_file() and (cur_l / "rng.bin").is_file()
    receipt = json.loads((cur_l / "COMPACTION_RECEIPT.json").read_text(encoding="utf-8"))
    assert receipt["stripped_components"] == ["optimizer.bin"] and receipt["inference_and_weights_only"] is True

    # 2. Full CampaignOrchestrator integration: R6 parent keeps optimizer, R2/R7 leaves compact, orphan child isolated
    def _runner(job: SessionJob, wroot: Path, device: str) -> dict:
        _make_fake_ckpt(wroot)
        return {"exit_code": 0, "status": "COMPLETE", "completed_updates": int(job.updates)}

    jobs = [
        SessionJob("R2_LEAF", "P", 7, 7, 10),
        SessionJob("R6_PARENT", "R", 7, 7, 20),
        SessionJob("R7_ADAPT", "R", 7, 7, 10, extra={"adaptation_parent_id": "R6_PARENT"}),
        SessionJob("R7_ORPHAN", "R", 7, 7, 10, extra={"adaptation_parent_id": "MISSING_R6_HOST"}),
    ]
    wdir = tmp_path / "camp_dep"
    res = CampaignOrchestrator(
        campaign_id="X3-DEP-COMPACT",
        source_commit="e" * 40,
        work_dir=wdir,
        max_minutes=120.0,
        reserve_minutes=5.0,
        min_disk_bytes=1024,
    ).run_registered_plan(
        jobs,
        job_runner=_runner,
        devices=("cuda:0",),
        measured_seconds_per_update=0.5,
        per_job_disk_bytes=1024,
        available_disk_bytes=10**8,
        prune_completed_checkpoints=True,
        strip_leaf_optimizers=True,
    )

    assert sorted(res["completed_jobs"]) == ["R2_LEAF", "R6_PARENT", "R7_ADAPT"]
    assert res["blocked_job_ids"] == ["R7_ORPHAN"]
    assert res["pruned_artifacts"]["preserved_optimizer_jobs"] == ["R6_PARENT"]
    assert sorted(res["pruned_artifacts"]["compacted_leaf_jobs"]) == ["R2_LEAF", "R7_ADAPT"]
    assert (wdir / "jobs" / "R6_PARENT" / "ckpt" / "main" / "objects" / ("b" * 64) / "optimizer.bin").is_file()
    assert not (wdir / "jobs" / "R2_LEAF" / "ckpt" / "main" / "objects" / ("b" * 64) / "optimizer.bin").exists()
    assert (wdir / "jobs" / "R2_LEAF" / "ckpt" / "main" / "objects" / ("b" * 64) / "COMPACTION_RECEIPT.json").is_file()
