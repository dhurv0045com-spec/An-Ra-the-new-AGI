"""Independent verification for Avendi V8-Y3 receipts, checkpoints, replays, and claims."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np

from anra_v5.planthir import build_verification_matrix
from experiments.avendi_y3.plan import Y1_EXPECTED_CKPT_SHAS, build_plan

DISTINCT_STUDY_AXES = {
    "U08": "crossover_4axis",
    "U09": "literal_position_wording",
    "U10": "decoder_continuations_eos",
    "U11": "changed_query_margin",
    "U12": "irrel_rewire_tsperm",
    "U13": "representation_probe",
    "U14": "legacy_replay_scope",
    "U28": "categorical_recurrence",
    "U29": "continuous_control_param_match",
    "U30": "dense_control_and_oracle",
    "U31": "length_and_topology_transfer",
    "U32": "donor_interchange_step1",
    "U33": "specificity_and_sham_controls",
    "U34": "suffix_rewire_after_patch",
    "U35": "soft_vs_hard_argmax_and_endonly",
    "U36": "seven_condition_executive",
    "U37": "relevant_vs_irrelevant_tool",
    "U38": "stale_contradictory_evidence",
    "U39": "false_abstention_and_dp_regret",
    "U40": "fork_retention_vs_acquisition",
    "U41": "feedback_conditions",
    "U42": "context_vs_persistent_memory",
    "U43": "bounded_weight_update",
    "U44": "diagnostic_action_utility",
    "U45": "split_correction_phases",
    "U46": "version_space_no_info_ceiling",
}


def canonical_digest(obj: Any) -> str:
    """Return canonical SHA-256 hex digest of a JSON-serializable object."""
    return sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def all_finite(value: Any) -> bool:
    """Return False if any float inside a nested structure is NaN or +/-Inf."""
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, str, bytes)) or value is None:
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(all_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_finite(v) for v in value)
    return True


def _plan_job_rows(mode: str = "FULL") -> dict[str, dict[str, Any]]:
    plan = build_plan(mode=mode)
    rows = {r["job_id"]: r for k in ("training_jobs", "short_fork_jobs", "p6_adaptation_jobs") for r in plan[k]}
    rows["P1-frozen"] = {
        "job_id": "P1-frozen",
        "stage": "P1",
        "arm": "frozen",
        "init_seed": 7,
        "order_seed": 7,
        "updates": 1,
        "spec_sha256": canonical_digest("P1-frozen"),
        "status": "PLANNED",
    }
    return rows


def _expected_updates(row: Mapping[str, Any], S: Any) -> int:
    if getattr(S, "full", False):
        return int(row.get("updates", 1))
    if row["stage"] == "P1":
        return 1
    if str(row["arm"]).startswith("u19_"):
        return int(S.u19_u)
    if str(row["arm"]).endswith("endpoint_only") or (row["stage"] == "P7" and str(row["arm"]).startswith("ark")):
        return int(S.fork)
    return int(S.upd)


def _check_npz_and_cases(jdir: Path, res: Mapping[str, Any]) -> tuple[list[str], int, int]:
    """Verify saved first-logit NPZ files against serialized case predictions and token chars."""
    reasons: list[str] = []
    ev_dev = res.get("eval_dev")
    if not isinstance(ev_dev, Mapping):
        return ["MISSING_EVAL_DEV"], 0, 0
    panels_checked = 0
    tokens_checked = 0
    for panel_key, tag in (("native", "dev_native"), ("renamed", "dev_renamed"), ("irrelevant_rows", "dev_irrel"), ("timestamp_permuted", "dev_tsperm"), ("literal", "dev_literal")):
        pdoc = ev_dev.get(panel_key)
        if not isinstance(pdoc, Mapping):
            reasons.append(f"MISSING_PANEL:{panel_key}")
            continue
        cases = pdoc.get("cases")
        if not isinstance(cases, Mapping):
            reasons.append(f"MISSING_CASES:{panel_key}")
            continue
        npz_path = jdir / f"{tag}_first_logits.npz"
        if not npz_path.is_file():
            reasons.append(f"MISSING_NPZ:{npz_path.name}")
            continue
        try:
            with np.load(npz_path, allow_pickle=False) as arr:
                logits = arr["logits"]
                ids = [str(x) for x in arr["ids"].tolist()]
        except Exception as exc:
            reasons.append(f"CORRUPT_NPZ:{npz_path.name}:{type(exc).__name__}")
            continue
        if logits.ndim != 2 or logits.shape[0] != len(ids) or not np.isfinite(logits).all():
            reasons.append(f"INVALID_NPZ_ARRAY:{panel_key}")
            continue
        if len(cases) != len(ids) or set(cases) != set(ids):
            reasons.append(f"CASE_ID_SET_MISMATCH:{panel_key}")
            continue
        panels_checked += 1
        top_tokens = np.argmax(logits, axis=-1)
        for idx, wid in enumerate(ids):
            row = cases[wid]
            if not isinstance(row, list) or len(row) < 4:
                reasons.append(f"MALFORMED_CASE_ROW:{panel_key}:{wid}")
                break
            tok = int(top_tokens[idx])
            exp_ch = chr(tok - 4) if 4 <= tok < 132 else "?"
            if row[3] != exp_ch:
                reasons.append(f"CORRUPTED_PREDICTED_TOKEN:{panel_key}:{wid}:{row[3]}!={exp_ch}")
                break
            tokens_checked += 1
    return reasons, panels_checked, tokens_checked


def verify_job_receipt(
    wr: Path,
    job_id: str,
    row: Mapping[str, Any],
    *,
    S: Any,
    recs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Verify one job's receipt, archive SHA/members, full config, parent lineage, finiteness, and predictions."""
    jdir = wr / "jobs" / job_id
    rc_path = jdir / "receipt.json"
    reasons: list[str] = []
    if not rc_path.is_file():
        return {"job_id": job_id, "stage": row["stage"], "passed": False, "reasons": ["MISSING_RECEIPT"], "details": {}}
    try:
        rec = json.loads(rc_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"job_id": job_id, "stage": row["stage"], "passed": False, "reasons": [f"CORRUPT_RECEIPT:{type(exc).__name__}"], "details": {}}

    plan = build_plan(mode="FULL" if getattr(S, "full", False) else "CANARY")
    if rec.get("job_id") != job_id:
        reasons.append("JOB_ID_MISMATCH")
    if rec.get("init_seed") != row.get("init_seed"):
        reasons.append("INIT_SEED_MISMATCH")
    if rec.get("spec_sha256") != row.get("spec_sha256"):
        reasons.append("SPEC_SHA256_MISMATCH")
    if rec.get("source_commit") != plan["source_commit"]:
        reasons.append("SOURCE_COMMIT_MISMATCH")
    exp_scale = "FULL" if getattr(S, "full", False) else "TINY"
    if rec.get("scale") != exp_scale:
        reasons.append(f"SCALE_MISMATCH:{rec.get('scale')}!={exp_scale}")

    exp_upd = _expected_updates(row, S)
    if int(rec.get("completed_updates", -1)) != exp_upd:
        reasons.append(f"UPDATE_COUNT_MISMATCH:{rec.get('completed_updates')}!={exp_upd}")

    if not all_finite(rec.get("result")):
        reasons.append("NONFINITE_RESULT_VALUES")

    arch_path = jdir / "job_archive.zip"
    if not arch_path.is_file():
        reasons.append("MISSING_JOB_ARCHIVE")
    else:
        actual_sha = sha256(arch_path.read_bytes()).hexdigest()
        if actual_sha != rec.get("archive_sha256"):
            reasons.append("ARCHIVE_SHA256_MISMATCH")
        else:
            try:
                with zipfile.ZipFile(arch_path, "r") as zf:
                    if "receipt.json" not in zf.namelist():
                        reasons.append("MISSING_RECEIPT_IN_ARCHIVE")
                    else:
                        archived = json.loads(zf.read("receipt.json").decode("utf-8"))
                        if archived.get("job_id") != job_id or archived.get("spec_sha256") != row.get("spec_sha256"):
                            reasons.append("ARCHIVE_RECEIPT_MISMATCH")
            except Exception as exc:
                reasons.append(f"CORRUPT_JOB_ARCHIVE:{type(exc).__name__}")

    res = rec.get("result") or {}
    stage = str(row["stage"])
    arm = str(row["arm"])
    details: dict[str, Any] = {"stage": stage, "arm": arm}
    ckpts_root = wr / "ckpts"

    if stage == "P1":
        lins = res.get("lineages") or {}
        expected_lins = tuple(plan["frozen_lineages"])
        if set(expected_lins) != set(lins):
            reasons.append("P1_MISSING_LINEAGE_ROWS")
        lin_summary: dict[str, Any] = {}
        restored_n = 0
        matched_cases = 0
        ref_cases = 0
        parity_passed = 0
        for lin in expected_lins:
            row_l = lins.get(lin) or {}
            st = str(row_l.get("status", "MISSING"))
            if st == "RESTORED":
                restored_n += 1
            p_st = str(row_l.get("parity_status", res.get("y2_logit_parity", "UNAVAILABLE")))
            m_n = int(row_l.get("matched_reference_cases", 0) or 0)
            r_n = int(row_l.get("reference_cases", 0) or 0)
            matched_cases += m_n
            ref_cases += r_n
            if p_st == "PASS" and m_n > 0 and m_n == r_n:
                parity_passed += 1
            lin_summary[lin] = {
                "status": st,
                "expected_checkpoint_sha256": Y1_EXPECTED_CKPT_SHAS[lin],
                "checkpoint_sha256": row_l.get("checkpoint_sha256"),
                "parity_status": p_st,
                "matched_reference_cases": m_n,
                "reference_cases": r_n,
            }
        details["p1_lineages"] = lin_summary
        details["restored_count"] = restored_n
        details["parity_passed_count"] = parity_passed
        details["matched_reference_cases"] = matched_cases
        details["reference_cases"] = ref_cases
    elif stage in ("P2", "P3") or (stage == "P7" and arm.startswith("s20")):
        npz_reasons, panels_ok, toks_ok = _check_npz_and_cases(jdir, res)
        reasons.extend(npz_reasons)
        details["panels_verified"] = panels_ok
        details["tokens_verified"] = toks_ok
        ckpt_sha = (res.get("train") or {}).get("checkpoint_sha256")
        latest_p = ckpts_root / f"y3-{job_id}" / "LATEST"
        if not latest_p.is_file() or latest_p.read_text(encoding="ascii").strip() != ckpt_sha:
            reasons.append("CHECKPOINT_LATEST_MISMATCH")
        else:
            details["checkpoint_sha256"] = ckpt_sha
        if getattr(S, "full", False) and not str(arm).startswith("u19_"):
            dev_cases = len((((res.get("eval_dev") or {}).get("native") or {}).get("cases") or {}))
            if dev_cases < 2048:
                reasons.append(f"UNDER_SIZED_FULL_DEV_EVAL:{dev_cases}")
    elif stage == "P4":
        if "rates" not in res or "causal" not in res:
            reasons.append("P4_MISSING_RATES_OR_CAUSAL")
        ckpt_sha = (res.get("train") or {}).get("checkpoint_sha256")
        latest_p = ckpts_root / f"y3-{job_id}" / "LATEST"
        if not latest_p.is_file() or latest_p.read_text(encoding="ascii").strip() != ckpt_sha:
            reasons.append("CHECKPOINT_LATEST_MISMATCH")
        if getattr(S, "full", False) and ((res.get("causal") or {}).get("distinguishable_pairs", 0) < 256):
            reasons.append("P4_UNDER_SIZED_INTERCHANGE_PAIRS")
    elif stage == "P5" or (stage == "P7" and arm.startswith("ark")):
        parent_id = f"P2-ce-s{row['init_seed']}"
        details["parent_job_id"] = parent_id
        parent_rec = recs.get(parent_id)
        if not parent_rec:
            reasons.append(f"MISSING_PARENT_RECEIPT:{parent_id}")
        else:
            p_ckpt = ((parent_rec.get("result") or {}).get("train") or {}).get("checkpoint_sha256")
            rec_p_ckpt = res.get("parent_checkpoint_sha256")
            p_latest = ckpts_root / f"y3-{parent_id}" / "LATEST"
            if not p_latest.is_file() or p_latest.read_text(encoding="ascii").strip() != p_ckpt:
                reasons.append(f"MISSING_OR_MISMATCHED_PARENT_CKPT:{parent_id}")
            if rec_p_ckpt is not None and rec_p_ckpt != p_ckpt:
                reasons.append(f"PARENT_CHECKPOINT_SHA_MISMATCH:{parent_id}")
            if res.get("parent_core_sha256") and res.get("child_initial_core_sha256") and res["parent_core_sha256"] != res["child_initial_core_sha256"]:
                reasons.append("CHILD_INITIAL_CORE_WEIGHT_MISMATCH")
            details["parent_checkpoint_sha256"] = rec_p_ckpt or p_ckpt
        c_ckpt = (res.get("train") or {}).get("checkpoint_sha256")
        c_latest = ckpts_root / f"y3-{job_id}" / "LATEST"
        if not c_latest.is_file() or c_latest.read_text(encoding="ascii").strip() != c_ckpt:
            reasons.append("CHECKPOINT_LATEST_MISMATCH")
    elif stage == "P6" and arm == "diagnostic_policy":
        ckpt_sha = (res.get("train") or {}).get("checkpoint_sha256")
        latest_p = ckpts_root / f"y3-{job_id}" / "LATEST"
        if not latest_p.is_file() or latest_p.read_text(encoding="ascii").strip() != ckpt_sha:
            reasons.append("CHECKPOINT_LATEST_MISMATCH")
        exp_blocks = 256 if getattr(S, "full", False) else int(S.blocks_dev)
        if int(((res.get("eval_dev") or {}).get("n_blocks", 0))) < exp_blocks:
            reasons.append("P6_UNDER_SIZED_BLOCKS")
    elif stage == "P6" and "adapt" in job_id:
        parent_id = f"P6-diagnostic_policy-s{row['init_seed']}"
        details["parent_job_id"] = parent_id
        parent_rec = recs.get(parent_id)
        if not parent_rec:
            reasons.append(f"MISSING_PARENT_RECEIPT:{parent_id}")
        else:
            p_ckpt = ((parent_rec.get("result") or {}).get("train") or {}).get("checkpoint_sha256")
            rec_p_ckpt = res.get("parent_checkpoint_sha256")
            if rec_p_ckpt != p_ckpt:
                reasons.append(f"P6_ADAPT_PARENT_SHA_MISMATCH:{parent_id}")
            p_latest = ckpts_root / f"y3-{parent_id}" / "LATEST"
            if not p_latest.is_file() or p_latest.read_text(encoding="ascii").strip() != p_ckpt:
                reasons.append(f"MISSING_OR_MISMATCHED_PARENT_CKPT:{parent_id}")
            if not bool((res.get("train") or {}).get("restored_parent", False)):
                reasons.append("P6_ADAPT_MISSING_RESTORED_PARENT_FLAG")
            details["parent_checkpoint_sha256"] = rec_p_ckpt

    return {"job_id": job_id, "stage": stage, "passed": len(reasons) == 0, "reasons": reasons, "details": details}


def verify_all_studies(
    wr: Path,
    *,
    job_checks: Mapping[str, Mapping[str, Any]],
    confirm_custody: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Verify every study U01..U50 for completeness, job backing, study-specific payloads, and non-duplication."""
    out: dict[str, dict[str, Any]] = {}
    measured_digests: dict[str, dict[str, str]] = {}

    for idx in range(1, 51):
        sid = f"U{idx:02}"
        hits = sorted((wr / "studies" / sid).rglob("receipt.json")) if (wr / "studies" / sid).is_dir() else []
        if not hits:
            out[sid] = {"study_id": sid, "execution_status": "MISSING", "evidence_status": "MISSING", "qualification_status": "INCOMPLETE", "reasons": ["MISSING_STUDY_RECEIPT"]}
            continue
        doc = json.loads(hits[0].read_text(encoding="utf-8"))
        reasons: list[str] = []
        if not all_finite(doc):
            reasons.append("NONFINITE_STUDY_PAYLOAD")
        req_jobs = [str(j) for j in (doc.get("required_jobs") or ())]
        miss_jobs = [str(j) for j in (doc.get("missing_jobs") or ())]
        exec_ok = (doc.get("status") == "COMPLETE") and not miss_jobs
        for j in req_jobs:
            jc = job_checks.get(j)
            if jc is None or not jc.get("passed"):
                reasons.append(f"JOB_UNVERIFIED:{j}")
        if sid in DISTINCT_STUDY_AXES:
            measured = doc.get("measured")
            axis = measured.get("study_axis") if isinstance(measured, Mapping) else None
            if axis != DISTINCT_STUDY_AXES[sid]:
                reasons.append(f"MISSING_OR_WRONG_STUDY_AXIS:{axis}!={DISTINCT_STUDY_AXES[sid]}")
            group = "p1" if idx <= 14 else ("p4" if idx <= 35 else ("p5" if idx <= 40 else "p6"))
            measured_digests.setdefault(group, {})[sid] = canonical_digest(measured)

        if sid == "U01" and not doc.get("oracle_ok"):
            reasons.append("U01_ORACLE_FAILED")
        elif sid == "U02" and not (doc.get("passed") and doc.get("planted_leak_caught")):
            reasons.append("U02_CUSTODY_OR_PLANTED_LEAK_FAILED")
        elif sid == "U03" and not (doc.get("oracle_all_families_ok") and doc.get("all_shortcuts_zero_families")):
            reasons.append("U03_SHORTCUT_FIXTURE_FAILED")
        elif sid == "U05" and not (doc.get("passed") and doc.get("eos_flip_detected") and doc.get("case_swap_detected")):
            reasons.append("U05_CORRUPTION_CHECK_FAILED")
        elif sid == "U06" and int(doc.get("verification_rows", 0)) < 6:
            reasons.append("U06_INSUFFICIENT_VERIFICATION_ROWS")
        elif idx in range(7, 15):
            p1_d = (job_checks.get("P1-frozen") or {}).get("details") or {}
            if int(p1_d.get("restored_count", 0)) < 6:
                reasons.append("BLOCKED_MISSING_Y1_PARENTS")
            elif sid in ("U07", "U14") and int(p1_d.get("parity_passed_count", 0)) < 6:
                reasons.append("UNAVAILABLE_Y1_PARITY")
        elif sid == "U15":
            measured = doc.get("measured") or {}
            if not set(req_jobs) <= set(measured):
                reasons.append("U15_MISSING_COMPARISON_JOB")
        elif sid == "U47":
            if not confirm_custody.get("custody_valid"):
                reasons.append(f"CONFIRM_CUSTODY_LIMITATION:{confirm_custody.get('status')}")
        elif sid == "U50":
            measured = doc.get("measured") or doc
            if int(measured.get("jobs_complete", 0)) < 88:
                reasons.append("U50_INCOMPLETE_JOBS")

        if any(r.startswith("BLOCKED_") for r in reasons):
            ev_st = "BLOCKED_MISSING_PARENT"
        elif any(r.startswith("UNAVAILABLE_") for r in reasons):
            ev_st = "UNAVAILABLE_REFERENCE_PARITY"
        elif not exec_ok or reasons:
            ev_st = "UNVERIFIED"
        else:
            ev_st = "VERIFIED"

        qual_st = "MEASURED_VERIFIED" if ev_st == "VERIFIED" else ("BLOCKED" if ev_st.startswith("BLOCKED") else "INCOMPLETE")
        out[sid] = {
            "study_id": sid,
            "execution_status": "COMPLETE" if exec_ok else str(doc.get("status", "INCOMPLETE")),
            "evidence_status": ev_st,
            "qualification_status": qual_st,
            "reasons": reasons,
        }

    # Downgrade duplicate study payloads within multi-study stages
    for grp, sid_map in measured_digests.items():
        seen: dict[str, list[str]] = {}
        for s_k, dig in sid_map.items():
            seen.setdefault(dig, []).append(s_k)
        for dig, sids in seen.items():
            if len(sids) > 1:
                for dup_sid in sids:
                    out[dup_sid]["reasons"].append(f"DUPLICATE_STUDY_PAYLOAD:{grp}")
                    if out[dup_sid]["evidence_status"] == "VERIFIED":
                        out[dup_sid]["evidence_status"] = "UNVERIFIED_DUPLICATE_PAYLOAD"
                        out[dup_sid]["qualification_status"] = "INCOMPLETE"
    return out


def verify_saved_campaign_evidence(
    wr: Path,
    *,
    S: Any,
    confirm_custody: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Independently audit all 88 jobs, 50 studies, and all 6 verification claims from saved disk evidence."""
    wr = Path(wr)
    rows = _plan_job_rows("FULL")
    recs: dict[str, dict[str, Any]] = {}
    for rc in (wr / "jobs").glob("*/receipt.json") if (wr / "jobs").is_dir() else ():
        try:
            recs[rc.parent.name] = json.loads(rc.read_text(encoding="utf-8"))
        except Exception:
            pass

    job_checks = {jid: verify_job_receipt(wr, jid, row, S=S, recs=recs) for jid, row in sorted(rows.items())}
    custody = dict(confirm_custody or {"status": "MISSING_CONFIRM_CUSTODY", "custody_valid": False})
    study_checks = verify_all_studies(wr, job_checks=job_checks, confirm_custody=custody)

    # 1. byte_integrity
    jobs_byte_ok = sum(
        1 for jc in job_checks.values()
        if not any(r in jc["reasons"] for r in ("MISSING_RECEIPT", "CORRUPT_RECEIPT", "MISSING_JOB_ARCHIVE", "ARCHIVE_SHA256_MISMATCH", "MISSING_RECEIPT_IN_ARCHIVE", "CORRUPT_JOB_ARCHIVE", "NONFINITE_RESULT_VALUES", "SPEC_SHA256_MISMATCH", "SOURCE_COMMIT_MISMATCH", "UPDATE_COUNT_MISMATCH"))
        and all("CORRUPT" not in r and "SHA256" not in r and "NONFINITE" not in r and "MISSING_JOB_ARCHIVE" not in r and "MISSING_RECEIPT" not in r for r in jc["reasons"])
    )
    byte_st = "PASS" if (jobs_byte_ok == len(rows) and custody.get("lock_sha256_verified", True)) else ("FAILED" if any(any("SHA256" in r or "CORRUPT" in r or "NONFINITE" in r for r in jc["reasons"]) for jc in job_checks.values()) else "PARTIAL")

    # 2. oracle_fixture_correctness (U01, U02, U03, U05)
    oracle_sids = ("U01", "U02", "U03", "U05")
    oracle_ok = sum(1 for sid in oracle_sids if study_checks[sid]["evidence_status"] == "VERIFIED")
    oracle_st = "PASS" if oracle_ok == len(oracle_sids) else ("FAILED" if any(any("FAILED" in r for r in study_checks[sid]["reasons"]) for sid in oracle_sids) else "PARTIAL")

    # 3. recorded_action_replay (P0 cartridge replay + NPZ/token consistency across all byte jobs)
    p0_rp = wr / "P0_REPLAY_RECEIPT.json"
    p0_ok = False
    if p0_rp.is_file():
        try:
            p0_ok = bool(json.loads(p0_rp.read_text(encoding="utf-8")).get("verified"))
        except Exception:
            p0_ok = False
    byte_job_ids = [jid for jid, row in rows.items() if row["stage"] in ("P2", "P3") or (row["stage"] == "P7" and str(row["arm"]).startswith("s20"))]
    tok_corrupted = [jid for jid in byte_job_ids if any("CORRUPTED_PREDICTED_TOKEN" in r or "CASE_ID_SET_MISMATCH" in r or "INVALID_NPZ_ARRAY" in r for r in job_checks[jid]["reasons"])]
    byte_tok_ok = sum(1 for jid in byte_job_ids if job_checks[jid]["passed"])
    replay_passed = byte_tok_ok + (1 if p0_ok else 0)
    replay_total = len(byte_job_ids) + 1
    if tok_corrupted:
        replay_st = "FAILED"
    elif replay_passed == replay_total:
        replay_st = "PASS"
    elif replay_passed == 0:
        replay_st = "UNAVAILABLE"
    else:
        replay_st = "PARTIAL"

    # 4. checkpoint_restore (6 Y1 parents + 27 child/adapt/fork restores + 1 confirm store)
    p1_det = job_checks["P1-frozen"]["details"]
    y1_restored = int(p1_det.get("restored_count", 0))
    child_ids = [jid for jid, row in rows.items() if row["stage"] == "P5" or (row["stage"] == "P7" and str(row["arm"]).startswith("ark")) or (row["stage"] == "P6" and "adapt" in jid)]
    child_ok = sum(1 for jid in child_ids if not any("PARENT" in r or "CHECKPOINT" in r for r in job_checks[jid]["reasons"]) and job_checks[jid]["passed"])
    ckpt_passed = y1_restored + child_ok
    ckpt_total = 6 + len(child_ids)
    y1_rejected = any((v or {}).get("status") == "REJECTED" for v in (p1_det.get("p1_lineages") or {}).values())
    if y1_restored == 6 and child_ok == len(child_ids):
        ckpt_st = "PASS"
    elif y1_rejected or any("PARENT" in r or "CHECKPOINT" in r for jid in child_ids for r in job_checks[jid]["reasons"]):
        ckpt_st = "BLOCKED" if y1_restored == 0 else "PARTIAL"
    elif y1_restored == 0:
        ckpt_st = "UNAVAILABLE"
    else:
        ckpt_st = "PARTIAL"

    # 5. regenerated_per_case_agreement (real Y1/Y2 parity cases across 6 lineages)
    par_ok = int(p1_det.get("parity_passed_count", 0))
    par_m = int(p1_det.get("matched_reference_cases", 0))
    par_r = int(p1_det.get("reference_cases", 0))
    if par_ok == 6 and par_r > 0 and par_m == par_r:
        regen_st = "PASS"
    elif any((v or {}).get("parity_status") == "FAIL" for v in (p1_det.get("p1_lineages") or {}).values()):
        regen_st = "FAILED"
    elif par_r == 0:
        regen_st = "UNAVAILABLE"
    else:
        regen_st = "PARTIAL"

    # 6. training_reproduction (real TRAINING_REPLAY_RECEIPT.json)
    tr_rp = wr / "TRAINING_REPLAY_RECEIPT.json"
    tr_ok = False
    if tr_rp.is_file():
        try:
            tr_doc = json.loads(tr_rp.read_text(encoding="utf-8"))
            tr_ok = bool(tr_doc.get("passed")) and bool(tr_doc.get("losses_matched")) and bool(tr_doc.get("weights_matched"))
        except Exception:
            tr_ok = False
    tr_st = "PASS" if tr_ok else ("FAILED" if tr_rp.is_file() else "UNAVAILABLE")

    vmat = build_verification_matrix(
        byte_integrity=byte_st,
        oracle_fixture_correctness=oracle_st,
        recorded_action_replay=replay_st,
        checkpoint_restore=ckpt_st,
        regenerated_per_case_agreement=regen_st,
        training_reproduction=tr_st,
    )
    claim_cov = {
        "byte_integrity": {"status": byte_st, "passed": jobs_byte_ok, "total": len(rows)},
        "oracle_fixture_correctness": {"status": oracle_st, "passed": oracle_ok, "total": len(oracle_sids)},
        "recorded_action_replay": {"status": replay_st, "passed": replay_passed, "total": replay_total, "corrupted_tokens_jobs": tok_corrupted},
        "checkpoint_restore": {"status": ckpt_st, "passed": ckpt_passed, "total": ckpt_total, "y1_restored": y1_restored, "y1_total": 6, "child_restored": child_ok, "child_total": len(child_ids)},
        "regenerated_per_case_agreement": {"status": regen_st, "passed": par_ok, "total": 6, "matched_reference_cases": par_m, "reference_cases": par_r},
        "training_reproduction": {"status": tr_st, "passed": 1 if tr_ok else 0, "total": 1},
    }
    verified_jobs = sum(1 for jc in job_checks.values() if jc["passed"])
    verified_studies = sum(1 for sc in study_checks.values() if sc["evidence_status"] == "VERIFIED")
    all_claims_pass = all(c["status"] == "PASS" for c in claim_cov.values())
    sci_verified = (verified_jobs == len(rows)) and (verified_studies == 50) and all_claims_pass and bool(custody.get("custody_valid"))
    return {
        "schema": "avendi-y3-independent-audit-v1",
        "jobs_present": len(recs),
        "jobs_verified": verified_jobs,
        "jobs_total": len(rows),
        "studies_verified": verified_studies,
        "studies_total": 50,
        "scientific_verified": sci_verified,
        "confirm_custody": custody,
        "verification_matrix": vmat,
        "claim_coverage": claim_cov,
        "failed_job_checks": {k: v["reasons"] for k, v in job_checks.items() if not v["passed"]},
        "unverified_studies": {k: v for k, v in study_checks.items() if v["evidence_status"] != "VERIFIED"},
        "job_checks": job_checks,
        "study_checks": study_checks,
    }
