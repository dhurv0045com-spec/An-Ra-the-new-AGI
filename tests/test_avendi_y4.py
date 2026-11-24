"""Comprehensive CPU verification suite for Avendi V8-Y4."""

from __future__ import annotations

import json
from pathlib import Path
import random
import tempfile
import zipfile

import numpy as np
import pytest
import torch

from anra_v5.elendril.representation import UTF8
from anra_v5.planthir import (
    BytePromptTrainer,
    ConfirmLock,
    GenericTrainRecipe,
    frame_byte_prompt_sample,
    load_legacy_byte_checkpoint,
    verify_session_archive,
)
from experiments.avendi_y4.models import (
    AvendiY4ByteModel,
    P6LearnedMemoryPolicy,
    Y4TypedCompositionModel,
    apply_bounded_p6_weight_update,
    compute_fp64_gradient_cosine,
    compute_y4_byte_loss,
    encode_y4_typed_batch,
    get_y4_byte_spec,
    select_y4_matched_widths,
)
from experiments.avendi_y4.module import (
    TINY_S,
    _resolve_confirmation,
    _split_fams,
    aggregate_full,
    main as y4_main,
    run_full_campaign,
    run_job,
    u0000_smoke_all_y4,
)
from experiments.avendi_y4.plan import PLATFORM_USES, Y1_EXPECTED_CKPT_SHAS, build_plan, validate_plan_dag
from experiments.avendi_y4.tasks import (
    ENTITY_64,
    assert_no_condition_leak,
    audit_y4_split_custody,
    build_functional_graph_families,
    build_p5_blind_worlds,
    build_p6_law_blocks,
    evaluate_y4_shortcuts_and_masks,
    oracle_walk_public_bytes,
    reproduce_y3_defects,
    solve_p5_bellman_scalar,
    solve_p5_bellman_vectorized,
    verify_p6_sham_zero_mi,
)
from experiments.avendi_y4.verify import (
    DISTINCT_Y4_STUDY_AXES,
    stage_historical_checkpoints_to_writable,
    verify_y4_campaign,
)


def test_y3_defect_witnesses_and_mutations() -> None:
    rep = reproduce_y3_defects()
    assert rep["y3_graph_complete_by_depth"] == {1: 8, 2: 8, 3: 0, 4: 0}
    assert rep["y4_graph_complete_by_depth"] == {1: 4, 2: 4, 3: 4, 4: 4}
    assert rep["y4_conflict_mutation_caught"] is True
    assert rep["y3_p5_prefix_shortcut_acc"] == 1.0
    assert rep["y4_p5_flag_only_acc"] < 0.70
    assert rep["y3_batch_unique_items"] == 1
    assert rep["y4_batch_unique_items"] == 16
    assert rep["y3_wrong_donor_matches_intended_and_diverges"] is True
    assert rep["y4_wrong_donor_actual_matches_recipient"] is True

    spec = get_y4_byte_spec("S3", cpu_canary=True)
    bm = AvendiY4ByteModel(spec, seed=7)
    fams = build_functional_graph_families("d_mut", 2, 4402, depths=(1, 2))
    gc = compute_fp64_gradient_cosine(bm, fams[0]["cells"][:4])
    assert gc["valid"] is True and gc["passed"] is True


def test_public_byte_oracle_and_split_custody() -> None:
    bad_prompts = (
        "ROWS: A0->A0 ; B1->C2 || STEP=1 Q:A0",
        "ROWS: A0->B1 ; A0->C2 || STEP=1 Q:A0",
        "ROWS: A0->B1 || STEP=2 Q:A0",
    )
    for bp in bad_prompts:
        with pytest.raises(ValueError):
            oracle_walk_public_bytes(UTF8.frame(bp, "B1").ids)
    with pytest.raises(ValueError):
        oracle_walk_public_bytes([UTF8.pad_id, *UTF8.frame("ROWS: A0->B1 || STEP=1 Q:A0", "B1").ids])

    fams = _split_fams(TINY_S)
    cust = audit_y4_split_custody({k: fams[k] for k in ("train", "dev", "confirm", "transfer")})
    assert cust["passed"] is True
    assert cust["planted_leak_caught"] is True

    sc = evaluate_y4_shortcuts_and_masks(fams["dev"])
    assert sc["oracle_all_families_ok"] is True
    assert sc["all_shortcuts_zero_families"] is True


def test_p5_condition_leak_bellman_and_p6_sham_mi() -> None:
    ws = build_p5_blind_worlds("dev", 28, 4402)
    assert_no_condition_leak(ws)
    with pytest.raises(ValueError, match="CONDITION_LEAK"):
        assert_no_condition_leak([{"public_prompt": "FACTS: stale_contradictory | Q:A0"}])
    with pytest.raises(ValueError, match="forbidden_metadata_key"):
        assert_no_condition_leak([{"public_prompt": "FACTS: A0->B1 | Q:A0", "metadata": {"gold_answer": "B1"}}])

    rew = np.array([[1.0, -2.0], [-2.0, 1.0], [0.0, 0.0]], dtype=np.float64)
    obs = np.array([[[0.95, 0.05], [0.05, 0.95]], [[0.5, 0.5], [0.5, 0.5]]], dtype=np.float64)
    costs = [0.10, 0.05]
    for b in ([0.2, 0.8], [0.5, 0.5], [0.9, 0.1]):
        bv = solve_p5_bellman_vectorized(b, rew, obs, costs, horizon=1)
        bs = solve_p5_bellman_scalar(b, rew.tolist(), obs.tolist(), costs, horizon=1)
        assert float(np.max(np.abs(bv["values"] - np.asarray(bs, dtype=np.float64)))) <= 1e-8

    blocks = build_p6_law_blocks("dev", 16, 4402)
    mi = verify_p6_sham_zero_mi(blocks)
    assert mi["zero_mi_verified"] is True
    assert mi["max_mi_bits"] == 0.0


def test_registry_dag_and_matched_widths() -> None:
    plan = build_plan()
    dag = validate_plan_dag(plan)
    assert dag["valid"] is True
    assert dag["jobs"] == 126
    assert dag["trained"] == 120
    assert dag["historical"] == 6
    assert dag["studies"] == 50
    assert dag["platform_uses"] == 44 == len(PLATFORM_USES)
    assert len(DISTINCT_Y4_STUDY_AXES) == 26
    assert len(set(DISTINCT_Y4_STUDY_AXES.values())) == 26

    widths = select_y4_matched_widths(128)
    assert widths["continuous"]["confounded"] is False
    assert widths["dense"]["confounded"] is False
    assert widths["continuous"]["rel_param_diff"] <= 0.02
    assert widths["dense"]["rel_param_diff"] <= 0.02


def test_typed_invariance_grad_cosine_and_minibatch_equiv() -> None:
    fams = build_functional_graph_families("inv", 4, 4402, depths=(1, 2))
    c1 = next(c for f in fams for c in f["cells"] if c["depth"] == 1)
    c2 = next(c for f in fams for c in f["cells"] if c["depth"] == 2)
    c1_rev = {**c1, "rows": list(reversed(c1["rows"]))}

    for arch in ("categorical", "continuous", "dense"):
        m = Y4TypedCompositionModel(arch, 32, seed=7).eval()
        with torch.no_grad():
            rk1, rv1, q1, vm1, _ = encode_y4_typed_batch([c1], torch.device("cpu"))
            l1 = m.forward_trace(rk1, rv1, q1, vm1, steps=1)["final_logits"][0]
            rk2, rv2, q2, vm2, _ = encode_y4_typed_batch([c1, c2], torch.device("cpu"))
            l_pad = m.forward_trace(rk2, rv2, q2, vm2, steps=1)["final_logits"][0]
            assert float((l1 - l_pad).abs().max().item()) <= 1e-6
            rkr, rvr, qr, vmr, _ = encode_y4_typed_batch([c1_rev], torch.device("cpu"))
            l_rev = m.forward_trace(rkr, rvr, qr, vmr, steps=1)["final_logits"][0]
            assert float((l1 - l_rev).abs().max().item()) <= 1e-6

    spec = get_y4_byte_spec("S3", cpu_canary=True)
    bm = AvendiY4ByteModel(spec, seed=7)
    cells = fams[0]["cells"][:4]
    gc = compute_fp64_gradient_cosine(bm, cells)
    assert gc["valid"] is True
    assert gc["passed"] is True
    assert gc["zero_grad_detected"] is False

    # Zero-grad detection on identical gold/rival (zero margin loss -> zero grad with graph intact)
    z_cells = [{**c, "rival": c["gold"]} for c in cells]
    zm = AvendiY4ByteModel(spec, seed=7)
    zg = compute_fp64_gradient_cosine(zm, z_cells)
    assert zg["zero_grad_detected"] is True

    # Full-batch vs minibatched gradient equivalence
    rec_full = GenericTrainRecipe(seed=7, total_updates=1, batch_size=4, micro_batch_size=4, objective="ce")
    m_full = AvendiY4ByteModel(spec, seed=11)
    m_mb = AvendiY4ByteModel(spec, seed=11)
    m_full.zero_grad()
    compute_y4_byte_loss(m_full, cells, rec_full).backward()
    m_mb.zero_grad()
    for i in range(4):
        (compute_y4_byte_loss(m_mb, [cells[i]], rec_full) * 0.25).backward()
    max_g_diff = max(
        float((p1.grad - p2.grad).abs().max().item())
        for p1, p2 in zip(m_full.parameters(), m_mb.parameters())
        if p1.grad is not None and p2.grad is not None
    )
    assert max_g_diff <= 1e-5


def test_u0000_smoke_immutable_parent_and_p6_rollback(tmp_path: Path) -> None:
    smk = u0000_smoke_all_y4(tmp_path / "u0")
    assert smk["passed"] is True
    assert smk["checked"] == 15

    # Immutable parent check after D & G forks + P6 rollback byte identity
    wr = tmp_path / "fork"
    run_job("A-ce-s7", TINY_S, "cpu", wr)
    lat_p = wr / "ckpts" / "y4-A-ce-s7" / "LATEST"
    p_sha = lat_p.read_text(encoding="ascii").strip()
    p_bytes = (wr / "ckpts" / "y4-A-ce-s7" / "objects" / p_sha / "model.bin").read_bytes()

    d_rec = run_job("D-a_to_a-s7", TINY_S, "cpu", wr)
    g_rec = run_job("G-true_cap-s7", TINY_S, "cpu", wr)
    assert d_rec["result"]["parent_core_sha256"] == d_rec["result"]["child_initial_core_sha256"]
    assert g_rec["result"]["parent_core_sha256"] == g_rec["result"]["child_initial_core_sha256"]
    assert lat_p.read_text(encoding="ascii").strip() == p_sha
    assert (wr / "ckpts" / "y4-A-ce-s7" / "objects" / p_sha / "model.bin").read_bytes() == p_bytes

    pol = P6LearnedMemoryPolicy(width=32, seed=7)
    before = {k: v.detach().clone() for k, v in pol.named_parameters()}
    snap, info = apply_bounded_p6_weight_update(pol, 0, 1, lr=1e-3, max_rel_norm=0.001)
    assert info["rel_norm"] <= 0.001 + 1e-9
    with torch.no_grad():
        for n, p in pol.named_parameters():
            p.copy_(snap[n])
    for n, p in pol.named_parameters():
        assert torch.equal(p, before[n])


def test_mocked_dual_gpu_full_126_campaign_deferral_resume_and_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    y1_root = tmp_path / "y1_valid_store"
    spec = get_y4_byte_spec("S3", cpu_canary=True)
    samp = [frame_byte_prompt_sample("ROWS: A0->B1 || STEP=1 Q:A0", "B1", rival_answer="C2", task_id="t0", world_id="w0") for _ in range(2)]
    src_sha = build_plan()["source_commit"]
    for lin in sorted(Y1_EXPECTED_CKPT_SHAS):
        sd = 7 if lin.endswith("-s7") else (19 if lin.endswith("-s19") else 41)
        tr_y1 = BytePromptTrainer(samp, spec=spec, recipe=GenericTrainRecipe(seed=sd, total_updates=1, batch_size=2, micro_batch_size=2), source_commit=src_sha, root=y1_root, lineage=lin, data_provenance={"s": "t"}, device="cpu", cpu_canary=True)
        tr_y1.step()
        tr_y1.save()
        monkeypatch.setitem(Y1_EXPECTED_CKPT_SHAS, lin, str(tr_y1.parent))

    wr = tmp_path / "full126"
    roots = (str(y1_root),)

    def _inproc(wspec, env, cmd):
        run_job(wspec.worker_id, TINY_S, wspec.device, wr, roots=roots)
        return {"exit_code": 0}

    # 1. Tight-clock slice -> defers cleanly while sealing ConfirmLock
    ticks = iter([0.0] * 10 + [200.0] * 50)
    disp1 = run_full_campaign(
        wr=wr,
        S=TINY_S,
        session_minutes=17.0,
        input_roots=roots,
        devices=("cuda:0", "cuda:1"),
        in_process_runner=_inproc,
        measured_sec_per_update=0.01,
        clock_fn=lambda: next(ticks, 200.0),
    )
    assert disp1["exit_reason"] == "TIME_GUARD_STOP"
    assert len(disp1["pending_job_ids"]) > 0
    assert (wr / "CONFIRM_LOCK.json").is_file()
    assert ConfirmLock(wr / "CONFIRM_LOCK.json").inspect(disp1["executed_job_ids"])["unlocked"] is False

    # 2. Resume to full 126-job completion & aggregate
    disp2 = run_full_campaign(
        wr=wr,
        S=TINY_S,
        session_minutes=330.0,
        input_roots=roots,
        devices=("cuda:0", "cuda:1"),
        in_process_runner=_inproc,
        measured_sec_per_update=0.01,
    )
    assert disp2["exit_reason"] == "ALL_JOBS_COMPLETE"
    assert disp2["dispatched_devices"] == ["cuda:0", "cuda:1"]
    assert len(disp2["pending_job_ids"]) == 0

    fin = aggregate_full(wr, TINY_S, disp2, input_roots=roots)
    assert fin["jobs_verified"] == 126
    assert fin["studies_verified"] == 50
    assert fin["summary_bytes"] <= 2 * 1024 * 1024
    assert (wr / "CONFIRM_RESULT.json").is_file()
    cov = json.loads((wr / "PLATFORM_COVERAGE.json").read_text(encoding="utf-8"))
    assert cov["count"] == 44
    assert verify_session_archive(Path(fin["archives"]["summary"]), expected_source_commit=build_plan()["source_commit"])["verified"] is True

    # 3. Third run on finished work_root -> clean ALREADY_VERIFIED_NO_OP
    disp3 = run_full_campaign(
        wr=wr,
        S=TINY_S,
        session_minutes=330.0,
        input_roots=roots,
        devices=("cuda:0", "cuda:1"),
        in_process_runner=_inproc,
        measured_sec_per_update=0.01,
    )
    assert disp3["exit_reason"] == "ALREADY_VERIFIED_NO_OP"
    assert disp3["executed_job_ids"] == []
    assert len(disp3["reused_job_ids"]) == 126


def test_planted_audit_faults_rejected(tmp_path: Path) -> None:
    wr = tmp_path / "faults"
    a_rec = run_job("A-ce-s7", TINY_S, "cpu", wr)
    run_job("D-a_to_a-s7", TINY_S, "cpu", wr)
    run_job("E-learned_writer-s7", TINY_S, "cpu", wr)

    # Planted token corruption in A-ce-s7 receipt
    rc_p = wr / "jobs" / "A-ce-s7" / "receipt.json"
    bad_a = json.loads(rc_p.read_text(encoding="utf-8"))
    wid0 = next(iter(bad_a["result"]["eval_dev"]["native"]["cases"]))
    bad_a["result"]["eval_dev"]["native"]["cases"][wid0][3] = "~"
    rc_p.write_text(json.dumps(bad_a), encoding="utf-8")

    # Planted duplicate study payload between U08 and U09
    s8_p = wr / "studies" / "U08" / "U08-1jobs" / "dev" / "receipt.json"
    s9_p = wr / "studies" / "U09" / "U09-1jobs" / "dev" / "receipt.json"
    s8_p.parent.mkdir(parents=True, exist_ok=True)
    s9_p.parent.mkdir(parents=True, exist_ok=True)
    dup_pay = {"study_axis": DISTINCT_Y4_STUDY_AXES["U08"], "same": 1}
    s8_p.write_text(json.dumps({"status": "COMPLETE", "measured": dup_pay}), encoding="utf-8")
    s9_p.write_text(json.dumps({"status": "COMPLETE", "measured": dup_pay}), encoding="utf-8")

    # Keep ConfirmLock sealed because confirm-eligible jobs are missing
    res = aggregate_full(wr, TINY_S, {})
    assert res["confirm"] == "DEFERRED_ELIGIBLE_ARMS_INCOMPLETE"
    audit = json.loads((wr / "VERIFIED_AUDIT.json").read_text(encoding="utf-8"))
    assert "A-ce-s7" in audit["failed_job_checks"]
    assert any("CORRUPTED_PREDICTED_TOKEN" in r for r in audit["failed_job_checks"]["A-ce-s7"])
    assert audit["claim_coverage"]["recorded_action_replay"]["status"] == "FAILED"
    assert "WRONG_STUDY_AXIS" in "".join(audit["unverified_studies"]["U09"]["reasons"])


def test_recovery_y1_decoy_two_lane_failure_and_confirmlock_crash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # (1) Read-only valid Y1 archive selected ahead of metadata-only decoy directory
    lin = "Y1-t1_factorial-S3-F00-s7"
    decoy_dir = tmp_path / "decoy_meta_only" / lin
    decoy_dir.mkdir(parents=True)
    (decoy_dir / "LATEST").write_text("0" * 64 + "\n", encoding="ascii")

    valid_src = tmp_path / "real_y1_src"
    spec = get_y4_byte_spec("S3", cpu_canary=True)
    samp = [frame_byte_prompt_sample("ROWS: A0->B1 || STEP=1 Q:A0", "B1", rival_answer="C2", task_id="t0", world_id="w0") for _ in range(2)]
    tr = BytePromptTrainer(samp, spec=spec, recipe=GenericTrainRecipe(seed=7, total_updates=1, batch_size=2, micro_batch_size=2), source_commit=build_plan()["source_commit"], root=valid_src, lineage=lin, data_provenance={"s": "t"}, device="cpu", cpu_canary=True)
    tr.step()
    tr.save()
    real_sha = str(tr.parent)

    y1_zip = tmp_path / "Y1_VALID.zip"
    with zipfile.ZipFile(y1_zip, "w") as zf:
        for f in (valid_src / lin).rglob("*"):
            if f.is_file():
                zf.write(f, str(Path("ckpts") / f.relative_to(valid_src)))

    monkeypatch.setitem(Y1_EXPECTED_CKPT_SHAS, lin, real_sha)
    st_res = stage_historical_checkpoints_to_writable([decoy_dir.parent, y1_zip], [lin], tmp_path / "writable_y1")
    st_info = st_res["staged"][lin]
    assert st_info["status"] == "STAGED_FROM_ZIP"
    assert st_info["manifest_verified"] is True
    assert st_info["checkpoint_sha256"] == real_sha
    ld = load_legacy_byte_checkpoint(tmp_path / "writable_y1", lineage_id=lin, checkpoint_sha256=real_sha, device="cpu")
    assert ld["checkpoint_sha256"] == real_sha

    # (2) Two-lane partial failure preserves good peer receipt/bytes & exports valid PARTIAL archive
    wr_fail = tmp_path / "two_lane"

    def _fail_lane1(wspec, env, cmd):
        if wspec.device == "cuda:1":
            return {"exit_code": 2}
        run_job(wspec.worker_id, TINY_S, "cpu", wr_fail)
        return {"exit_code": 0}

    disp_f = run_full_campaign(
        wr=wr_fail,
        S=TINY_S,
        devices=("cuda:0", "cuda:1"),
        in_process_runner=_fail_lane1,
        measured_sec_per_update=0.01,
    )
    assert disp_f["exit_reason"] == "PARTIAL_WITH_FAILURES"
    assert len(disp_f["executed_job_ids"]) > 0
    assert len(disp_f["failed_job_ids"]) > 0
    for ok_id in disp_f["executed_job_ids"]:
        assert (wr_fail / "jobs" / ok_id / "receipt.json").is_file()
    fin_part = aggregate_full(wr_fail, TINY_S, disp_f)
    assert fin_part["status"] == "PARTIAL"
    assert verify_session_archive(Path(fin_part["archives"]["summary"]), expected_source_commit=build_plan()["source_commit"])["verified"] is True

    # (3) ConfirmLock crash recovery after unlock before CONFIRM_RESULT.json write -> idempotent single score
    wr_cl = tmp_path / "cl_crash"
    plan = build_plan()
    elig = plan["confirm_eligible_jobs"]
    rows = {r["job_id"]: r for r in plan["jobs"]}
    # Materialize minimal valid checkpoint/receipt for one representative confirm job and stub remaining elig IDs as present
    run_job("A-ce-s7", TINY_S, "cpu", wr_cl)
    fams = _split_fams(TINY_S)
    fake_recs = {"A-ce-s7": json.loads((wr_cl / "jobs" / "A-ce-s7" / "receipt.json").read_text(encoding="utf-8"))}
    fake_elig = ["A-ce-s7"]
    lock = ConfirmLock(wr_cl / "CONFIRM_LOCK.json")
    lock.seal(fake_elig, source_commit=plan["source_commit"])
    lock.unlock_if_ready(fake_elig, reason="CRASH_SIM")
    assert not (wr_cl / "CONFIRM_RESULT.json").is_file()

    c1, cu1 = _resolve_confirmation(wr_cl, fake_recs, fake_elig, rows, TINY_S, fams, plan)
    assert c1["status"] == "LOCKED_AND_OPENED_ONCE"
    assert cu1["custody_valid"] is True
    assert (wr_cl / "CONFIRM_RESULT.json").is_file()

    c2, cu2 = _resolve_confirmation(wr_cl, fake_recs, fake_elig, rows, TINY_S, fams, plan)
    assert c2["status"] == "REUSED_SEALED_CONFIRMATION"
    assert c2["result_sha256"] == c1["result_sha256"]
    assert cu2["custody_valid"] is True
