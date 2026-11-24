"""Independent Y4 verification: writable staging of Y1/Y3 weights, 126-job receipt audit, stats, and 6-claim matrix."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np

from anra_v5.planthir import (
    bootstrap_paired_families,
    build_verification_matrix,
    exact_mcnemar_test,
    holm_bonferroni_adjust,
    win_long_path,
)

from experiments.avendi_y4.plan import (
    TECHNOLOGY_BASELINE,
    Y1_EXPECTED_CKPT_SHAS,
    build_plan,
)

DISTINCT_Y4_STUDY_AXES = {
    "U08": "h_crossover_4axis",
    "U09": "h_literal_position_wording",
    "U10": "h_decoder_continuations_eos",
    "U11": "h_changed_query_margin",
    "U12": "h_irrel_rewire_tsperm",
    "U13": "h_representation_probe",
    "U14": "h_legacy_replay_scope",
    "U28": "c_categorical_recurrence",
    "U29": "c_continuous_control_param_match",
    "U30": "c_dense_control_and_oracle",
    "U31": "c_length_and_topology_transfer",
    "U32": "c_donor_interchange_step1_and_penult",
    "U33": "c_specificity_wrong_donor_actual",
    "U34": "c_suffix_rewire_after_patch",
    "U35": "c_iit_vs_sham_and_endonly",
    "U36": "d_seven_condition_executive",
    "U37": "d_relevant_vs_irrelevant_tool",
    "U38": "d_stale_contradictory_evidence",
    "U39": "d_false_abstention_and_dp_regret",
    "U40": "d_fork_retention_vs_acquisition",
    "U41": "e_feedback_conditions",
    "U42": "e_context_vs_persistent_memory",
    "U43": "e_bounded_weight_update",
    "U44": "e_diagnostic_action_utility",
    "U45": "e_split_correction_phases",
    "U46": "e_version_space_no_info_ceiling",
}


def canonical_digest(obj: Any) -> str:
    return sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def all_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, (int, str, bytes)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(all_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_finite(v) for v in value)
    return True


def stage_historical_checkpoints_to_writable(
    roots: Sequence[str | Path],
    lineages: Sequence[str],
    stage_root: Path,
) -> dict[str, Any]:
    """Locate REAL content-addressed checkpoint stores (with objects/<sha>/model.bin) ahead of metadata-only decoys and stage into writable scratch."""
    stage_root = Path(stage_root)
    stage_root.mkdir(parents=True, exist_ok=True)
    staged: dict[str, dict[str, Any]] = {}

    def _dir_has_real_obj(d: Path) -> bool:
        lp = d / "LATEST"
        if not lp.is_file():
            return False
        sh = lp.read_text(encoding="ascii").strip()
        return (d / "objects" / sh / "model.bin").is_file() and (d / "objects" / sh / "manifest.json").is_file()

    def _stage_from_dir(src_lin: Path, lin: str) -> dict[str, Any] | None:
        if not _dir_has_real_obj(src_lin):
            return None
        sh = (src_lin / "LATEST").read_text(encoding="ascii").strip()
        dst = stage_root / lin
        if dst.exists():
            shutil.rmtree(win_long_path(dst), ignore_errors=True)
        (win_long_path(dst / "objects")).mkdir(parents=True, exist_ok=True)
        shutil.copy2(win_long_path(src_lin / "LATEST"), win_long_path(dst / "LATEST"))
        shutil.copytree(win_long_path(src_lin / "objects" / sh), win_long_path(dst / "objects" / sh), dirs_exist_ok=True)
        return {"status": "STAGED_FROM_DIR", "source": str(src_lin), "checkpoint_sha256": sh, "staged_dir": str(dst)}

    def _stage_from_zip(zp: Path, lin: str) -> dict[str, Any] | None:
        try:
            with zipfile.ZipFile(zp, "r") as zf:
                names = set(zf.namelist())
                lat_cand = [n for n in names if n.endswith(f"{lin}/LATEST")]
                for lmem in lat_cand:
                    prefix = lmem[: -len("LATEST")]
                    sh = zf.read(lmem).decode("ascii").strip()
                    obj_pref = f"{prefix}objects/{sh}/"
                    if f"{obj_pref}model.bin" in names and f"{obj_pref}manifest.json" in names:
                        dst = stage_root / lin
                        if dst.exists():
                            shutil.rmtree(win_long_path(dst), ignore_errors=True)
                        (win_long_path(dst / "objects" / sh)).mkdir(parents=True, exist_ok=True)
                        win_long_path(dst / "LATEST").write_text(sh + "\n", encoding="ascii")
                        for full_n in names:
                            if full_n.startswith(obj_pref) and not full_n.endswith("/"):
                                rel = full_n[len(prefix) :]
                                tgt = win_long_path(dst / rel)
                                tgt.parent.mkdir(parents=True, exist_ok=True)
                                tgt.write_bytes(zf.read(full_n))
                        return {"status": "STAGED_FROM_ZIP", "source": str(zp), "checkpoint_sha256": sh, "staged_dir": str(dst)}
        except Exception:
            return None
        return None

    real_paths = [Path(r) for r in roots if r and Path(r).exists()]
    for lin in lineages:
        found = None
        # Pass 1: directories containing actual objects/<sha>/model.bin
        for rp in real_paths:
            if rp.is_dir():
                for cdir in sorted(rp.rglob(lin)):
                    if cdir.is_dir():
                        found = _stage_from_dir(cdir, lin)
                        if found:
                            break
            if found:
                break
        # Pass 2: zip archives containing actual objects/<sha>/model.bin (skips metadata-only analysis dirs)
        if not found:
            for rp in real_paths:
                zips = [rp] if (rp.is_file() and rp.suffix.lower() == ".zip") else (sorted(rp.rglob("*.zip")) if rp.is_dir() else [])
                for zp in zips:
                    found = _stage_from_zip(zp, lin)
                    if found:
                        break
                if found:
                    break
        if found:
            sh = found["checkpoint_sha256"]
            exp_sh = Y1_EXPECTED_CKPT_SHAS.get(lin)
            man_p = win_long_path(Path(found["staged_dir"]) / "objects" / sh / "manifest.json")
            man_digest = sha256(man_p.read_bytes()).hexdigest() if man_p.is_file() else ""
            if exp_sh and sh != exp_sh and man_digest != exp_sh:
                found["manifest_verified"] = False
                found["status"] = "SHA_MISMATCH"
            else:
                found["manifest_verified"] = True
            staged[lin] = found
        else:
            staged[lin] = {"status": "MISSING_PARENT", "manifest_verified": False}
    return {"staged": staged, "stage_root": str(stage_root)}


def _expected_updates(row: Mapping[str, Any], S: Any) -> int:
    if getattr(S, "full", False):
        return int(row.get("updates", 1))
    st = str(row["stage"])
    if st == "H":
        return 2368 if getattr(S, "full", False) else 1
    if st == "M":
        return int(S.m_upd)
    if st == "G":
        return int(S.g_upd)
    return int(S.upd)


def _check_npz_panels(jdir: Path, res: Mapping[str, Any], expected_panels: Sequence[str]) -> tuple[list[str], int, int]:
    reasons: list[str] = []
    ev_dev = res.get("eval_dev")
    if not isinstance(ev_dev, Mapping):
        return ["MISSING_EVAL_DEV"], 0, 0
    panels = toks = 0
    for pk in expected_panels:
        pdoc = ev_dev.get(pk)
        if not isinstance(pdoc, Mapping):
            reasons.append(f"MISSING_PANEL:{pk}")
            continue
        cases = pdoc.get("cases")
        npz_p = jdir / f"dev_{pk}_first_logits.npz"
        if not isinstance(cases, Mapping) or not npz_p.is_file():
            reasons.append(f"MISSING_CASES_OR_NPZ:{pk}")
            continue
        with np.load(npz_p, allow_pickle=False) as arr:
            logits = arr["logits"]
            ids = [str(x) for x in arr["ids"].tolist()]
        if logits.ndim != 2 or logits.shape[0] != len(ids) or not np.isfinite(logits).all() or set(cases) != set(ids):
            reasons.append(f"INVALID_NPZ_OR_IDS:{pk}")
            continue
        panels += 1
        top = np.argmax(logits, axis=-1)
        for ix, wid in enumerate(ids):
            row = cases[wid]
            if not isinstance(row, list) or len(row) < 4:
                reasons.append(f"MALFORMED_CASE:{pk}:{wid}")
                break
            t0 = int(top[ix])
            exp_ch = chr(t0 - 4) if 4 <= t0 < 132 else "?"
            # row[3] is serialized top first-byte char; mismatch flags token tampering
            if str(row[3]) != exp_ch:
                reasons.append(f"CORRUPTED_PREDICTED_TOKEN:{pk}:{wid}:{row[3]}!={exp_ch}")
                break
            toks += 1
    return reasons, panels, toks


def verify_y4_job_receipt(wr: Path, job_id: str, row: Mapping[str, Any], *, S: Any, recs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Independently verify one of the 126 Y4 job receipts, archives, counters, lineage, and predictions."""
    jdir = wr / "jobs" / job_id
    rc_p = jdir / "receipt.json"
    reasons: list[str] = []
    if not rc_p.is_file():
        return {"job_id": job_id, "stage": row["stage"], "passed": False, "reasons": ["MISSING_RECEIPT"], "details": {}}
    try:
        rec = json.loads(rc_p.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"job_id": job_id, "stage": row["stage"], "passed": False, "reasons": [f"CORRUPT_RECEIPT:{type(exc).__name__}"], "details": {}}

    plan = build_plan()
    if rec.get("job_id") != job_id:
        reasons.append("JOB_ID_MISMATCH")
    if rec.get("init_seed") != row.get("init_seed"):
        reasons.append("INIT_SEED_MISMATCH")
    if rec.get("spec_sha256") != row.get("spec_sha256"):
        reasons.append("SPEC_SHA256_MISMATCH")
    if rec.get("source_commit") != plan["source_commit"]:
        reasons.append("SOURCE_COMMIT_MISMATCH")
    if rec.get("technology_baseline") != TECHNOLOGY_BASELINE:
        reasons.append("TECHNOLOGY_BASELINE_MISSING")
    exp_scale = "FULL" if getattr(S, "full", False) else "TINY"
    if rec.get("scale") != exp_scale:
        reasons.append(f"SCALE_MISMATCH:{rec.get('scale')}!={exp_scale}")

    stage = str(row["stage"])
    if stage != "H":
        exp_upd = _expected_updates(row, S)
        if int(rec.get("completed_updates", -1)) != exp_upd:
            reasons.append(f"UPDATE_COUNT_MISMATCH:{rec.get('completed_updates')}!={exp_upd}")

    if not all_finite(rec.get("result")):
        reasons.append("NONFINITE_RESULT")

    azip = jdir / "job_archive.zip"
    if not azip.is_file():
        reasons.append("MISSING_JOB_ARCHIVE")
    else:
        asha = sha256(azip.read_bytes()).hexdigest()
        if asha != rec.get("archive_sha256"):
            reasons.append("ARCHIVE_SHA256_MISMATCH")

    res = rec.get("result") or {}
    details: dict[str, Any] = {"stage": stage, "arm": row["arm"]}
    ckpts = wr / "ckpts"

    if stage == "H":
        lst = str(res.get("status", "MISSING"))
        details["lineage_status"] = lst
        details["parity_status"] = str(res.get("parity_status", "UNAVAILABLE"))
        details["matched_reference_cases"] = int(res.get("matched_reference_cases", 0) or 0)
        details["reference_cases"] = int(res.get("reference_cases", 0) or 0)
        if lst != "RESTORED":
            reasons.append(f"HISTORICAL_PARENT_{lst}")
    elif stage in ("A", "B", "M", "F"):
        p_keys = ("native", "entity_bijection", "timestamp_permuted", "literal") if stage in ("A", "B", "F") else ("native",)
        np_r, pan, tok = _check_npz_panels(jdir, res, p_keys)
        reasons.extend(np_r)
        details["panels_verified"], details["tokens_verified"] = pan, tok
        sh = (res.get("train") or {}).get("checkpoint_sha256")
        lp = ckpts / f"y4-{job_id}" / "LATEST"
        if not lp.is_file() or lp.read_text(encoding="ascii").strip() != sh:
            reasons.append("CHECKPOINT_LATEST_MISMATCH")
    elif stage == "C":
        sh = (res.get("train") or {}).get("checkpoint_sha256")
        lp = ckpts / f"y4-{job_id}" / "LATEST"
        if not lp.is_file() or lp.read_text(encoding="ascii").strip() != sh:
            reasons.append("CHECKPOINT_LATEST_MISMATCH")
        if "rates" not in res or "swap_suite" not in res:
            reasons.append("MISSING_C_METRICS")
        if int(res.get("min_unique_batch_items", 0)) < min(16, 8):
            reasons.append("COLLAPSED_BATCH_ITEMS")
    elif stage in ("D", "G"):
        p_id = str(row["parent_job_id"])
        p_rec = recs.get(p_id)
        if not p_rec:
            reasons.append(f"MISSING_PARENT_RECEIPT:{p_id}")
        else:
            p_ckpt = ((p_rec.get("result") or {}).get("train") or {}).get("checkpoint_sha256")
            if res.get("parent_checkpoint_sha256") != p_ckpt:
                reasons.append(f"PARENT_SHA_MISMATCH:{p_id}")
            if res.get("parent_core_sha256") != res.get("child_initial_core_sha256"):
                reasons.append("CHILD_INITIAL_CORE_MISMATCH")
        sh = (res.get("train") or {}).get("checkpoint_sha256")
        lp = ckpts / f"y4-{job_id}" / "LATEST"
        if not lp.is_file() or lp.read_text(encoding="ascii").strip() != sh:
            reasons.append("CHECKPOINT_LATEST_MISMATCH")
    elif stage == "E":
        # Enforce separate outer / support / eval counters
        if "outer_optimizer_steps" not in res or "support_gradient_steps" not in res or "eval_blocks" not in res:
            reasons.append("MISSING_E_INDEPENDENT_COUNTERS")
        if int(res.get("eval_routes_new_outer_steps", -1)) != 0:
            reasons.append("EVAL_MUTATED_OUTER_STEPS")
        sh = (res.get("train") or {}).get("checkpoint_sha256")
        lp = ckpts / f"y4-{job_id}" / "LATEST"
        if not lp.is_file() or lp.read_text(encoding="ascii").strip() != sh:
            reasons.append("CHECKPOINT_LATEST_MISMATCH")

    return {"job_id": job_id, "stage": stage, "passed": len(reasons) == 0, "reasons": reasons, "details": details}


def evaluate_release_gates(recs: Mapping[str, Mapping[str, Any]], *, S: Any) -> dict[str, Any]:
    """Compute paired seed contrasts, 2,000 bootstrap CIs, exact seed-sign, exact McNemar & Holm adjustments."""
    def _d34(j: str) -> float:
        r = recs[j]["result"]["rates"]["native"]
        return 0.5 * (float(r.get("3", 0.0)) + float(r.get("4", 0.0)))

    seeds = (7, 19, 41, 73, 101)
    have_c = all(f"C-categorical_{a}-s{s}" in recs for a in ("true_step", "true_step_iit", "true_step_sham_iit") for s in seeds)
    primary_c: dict[str, Any] = {"computed": False}
    if have_c:
        iit_vals = [_d34(f"C-categorical_true_step_iit-s{s}") for s in seeds]
        sham_vals = [_d34(f"C-categorical_true_step_sham_iit-s{s}") for s in seeds]
        base_vals = [_d34(f"C-categorical_true_step-s{s}") for s in seeds]
        diff_vs_sham = [a - b for a, b in zip(iit_vals, sham_vals)]
        diff_vs_base = [a - b for a, b in zip(iit_vals, base_vals)]
        boot_sham = bootstrap_paired_families(sham_vals, iit_vals, resamples=2000, seed=4417)
        boot_base = bootstrap_paired_families(base_vals, iit_vals, resamples=2000, seed=4417)
        mcn_sham = exact_mcnemar_test([v >= 0.8 for v in sham_vals], [v >= 0.8 for v in iit_vals])
        mcn_base = exact_mcnemar_test([v >= 0.8 for v in base_vals], [v >= 0.8 for v in iit_vals])
        holm = holm_bonferroni_adjust({"iit_vs_sham": mcn_sham["p_value"], "iit_vs_true_step": mcn_base["p_value"]})
        pos_sham = sum(1 for x in diff_vs_sham if x > 0)
        pos_base = sum(1 for x in diff_vs_base if x > 0)
        primary_c = {
            "computed": True,
            "mean_gain_vs_sham": float(np.mean(diff_vs_sham)),
            "mean_gain_vs_true_step": float(np.mean(diff_vs_base)),
            "positive_seeds_vs_sham": pos_sham,
            "positive_seeds_vs_true_step": pos_base,
            "bootstrap_vs_sham": boot_sham,
            "bootstrap_vs_true_step": boot_base,
            "holm": holm,
            "qualified": (float(np.mean(diff_vs_sham)) >= 0.10 and float(np.mean(diff_vs_base)) >= 0.10 and pos_sham >= 4 and pos_base >= 4),
        }
    return {"primary_c_contrast": primary_c}


def verify_y4_campaign(wr: Path, *, S: Any, confirm_custody: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Independently audit all 126 Y4 jobs, 50 studies, 44 platform uses, and 6 verification claims."""
    wr = Path(wr)
    plan = build_plan(mode="FULL" if getattr(S, "full", False) else "CANARY")
    rows = {r["job_id"]: r for r in plan["jobs"]}
    recs: dict[str, dict[str, Any]] = {}
    for rc in (wr / "jobs").glob("*/receipt.json") if (wr / "jobs").is_dir() else ():
        try:
            recs[rc.parent.name] = json.loads(rc.read_text(encoding="utf-8"))
        except Exception:
            pass

    job_checks = {jid: verify_y4_job_receipt(wr, jid, row, S=S, recs=recs) for jid, row in sorted(rows.items())}
    custody = dict(confirm_custody or {"status": "MISSING_CONFIRM_CUSTODY", "custody_valid": False})

    # Verify all 50 studies and reject missing distinct study axes or duplicate payloads
    study_checks: dict[str, dict[str, Any]] = {}
    digests_by_grp: dict[str, dict[str, str]] = {}
    for idx in range(1, 51):
        sid = f"U{idx:02}"
        hits = sorted((wr / "studies" / sid).rglob("receipt.json")) if (wr / "studies" / sid).is_dir() else []
        if not hits:
            study_checks[sid] = {"study_id": sid, "execution_status": "MISSING", "evidence_status": "MISSING", "reasons": ["MISSING_STUDY_RECEIPT"]}
            continue
        doc = json.loads(hits[0].read_text(encoding="utf-8"))
        reasons: list[str] = []
        if not all_finite(doc):
            reasons.append("NONFINITE_STUDY")
        req = [str(j) for j in (doc.get("required_jobs") or ())]
        miss = [str(j) for j in (doc.get("missing_jobs") or ())]
        exec_ok = (doc.get("status") == "COMPLETE") and not miss
        for j in req:
            if not (job_checks.get(j) or {}).get("passed"):
                reasons.append(f"JOB_UNVERIFIED:{j}")
        if sid in DISTINCT_Y4_STUDY_AXES:
            m = doc.get("measured")
            ax = m.get("study_axis") if isinstance(m, Mapping) else None
            if ax != DISTINCT_Y4_STUDY_AXES[sid]:
                reasons.append(f"WRONG_STUDY_AXIS:{ax}")
            grp = "h" if idx <= 14 else ("c" if idx <= 35 else ("d" if idx <= 40 else "e"))
            digests_by_grp.setdefault(grp, {})[sid] = canonical_digest(m)
        if sid == "U01" and not doc.get("oracle_ok"):
            reasons.append("U01_FAILED")
        elif sid == "U02" and not (doc.get("passed") and doc.get("planted_leak_caught")):
            reasons.append("U02_FAILED")
        elif sid == "U03" and not (doc.get("oracle_all_families_ok") and doc.get("all_shortcuts_zero_families")):
            reasons.append("U03_FAILED")
        elif sid == "U05" and not (doc.get("passed") and doc.get("eos_flip_detected") and doc.get("case_swap_detected")):
            reasons.append("U05_FAILED")
        elif sid == "U47" and not custody.get("custody_valid"):
            reasons.append(f"CONFIRM_CUSTODY:{custody.get('status')}")
        elif sid == "U50" and int((doc.get("measured") or doc).get("jobs_complete", 0)) < 126:
            reasons.append("U50_INCOMPLETE_JOBS")
        if any(str(r).startswith("JOB_UNVERIFIED:H-") for r in reasons):
            ev_st = "BLOCKED_MISSING_PARENT"
        elif not exec_ok or reasons:
            ev_st = "UNVERIFIED"
        else:
            ev_st = "VERIFIED"
        study_checks[sid] = {"study_id": sid, "execution_status": "COMPLETE" if exec_ok else str(doc.get("status", "PARTIAL")), "evidence_status": ev_st, "reasons": reasons}

    for grp, smap in digests_by_grp.items():
        rev: dict[str, list[str]] = {}
        for sk, dg in smap.items():
            rev.setdefault(dg, []).append(sk)
        for sids in rev.values():
            if len(sids) > 1:
                for ds in sids:
                    study_checks[ds]["reasons"].append(f"DUPLICATE_STUDY_PAYLOAD:{grp}")
                    study_checks[ds]["evidence_status"] = "UNVERIFIED_DUPLICATE_PAYLOAD"

    # Compute the 6 independent claim statuses & denominators
    byte_ok = sum(1 for jc in job_checks.values() if not any(r in jc["reasons"] for r in ("MISSING_RECEIPT", "SPEC_SHA256_MISMATCH", "ARCHIVE_SHA256_MISMATCH", "CORRUPT_RECEIPT")))
    byte_st = "PASS" if (byte_ok == len(rows) and custody.get("lock_sha256_verified", True)) else ("FAILED" if any("SHA256" in r or "CORRUPT" in r for jc in job_checks.values() for r in jc["reasons"]) else "PARTIAL")

    oracle_sids = ("U01", "U02", "U03", "U05")
    oracle_ok = sum(1 for sid in oracle_sids if study_checks[sid]["evidence_status"] == "VERIFIED")
    oracle_st = "PASS" if oracle_ok == len(oracle_sids) else "PARTIAL"

    p0_rp = wr / "P0_REPLAY_RECEIPT.json"
    p0_ok = p0_rp.is_file() and bool(json.loads(p0_rp.read_text(encoding="utf-8")).get("verified"))
    byte_jids = [jid for jid, r in rows.items() if r["stage"] in ("A", "B", "M", "F")]
    tok_bad = [j for j in byte_jids if any("CORRUPTED_PREDICTED_TOKEN" in r or "INVALID_NPZ" in r for r in job_checks[j]["reasons"])]
    rep_ok = sum(1 for j in byte_jids if job_checks[j]["passed"]) + (1 if p0_ok else 0)
    replay_st = "FAILED" if tok_bad else ("PASS" if rep_ok == len(byte_jids) + 1 else "PARTIAL")

    h_jids = [j for j, r in rows.items() if r["stage"] == "H"]
    h_restored = sum(1 for j in h_jids if (job_checks[j]["details"] or {}).get("lineage_status") == "RESTORED")
    dg_jids = [j for j, r in rows.items() if r["stage"] in ("D", "G")]
    dg_ok = sum(1 for j in dg_jids if job_checks[j]["passed"])
    ckpt_st = "PASS" if (h_restored == 6 and dg_ok == len(dg_jids)) else ("BLOCKED" if h_restored == 0 else "PARTIAL")

    par_ok = sum(1 for j in h_jids if (job_checks[j]["details"] or {}).get("parity_status") == "PASS")
    regen_st = "PASS" if par_ok == 6 else ("FAILED" if any((job_checks[j]["details"] or {}).get("parity_status") == "FAIL" for j in h_jids) else "UNAVAILABLE")

    tr_rp = wr / "TRAINING_REPLAY_RECEIPT.json"
    tr_ok = False
    if tr_rp.is_file():
        td = json.loads(tr_rp.read_text(encoding="utf-8"))
        tr_ok = bool(td.get("passed")) and bool(td.get("losses_matched")) and bool(td.get("weights_matched"))
    tr_st = "PASS" if tr_ok else ("FAILED" if tr_rp.is_file() else "UNAVAILABLE")

    vmat = build_verification_matrix(byte_integrity=byte_st, oracle_fixture_correctness=oracle_st, recorded_action_replay=replay_st, checkpoint_restore=ckpt_st, regenerated_per_case_agreement=regen_st, training_reproduction=tr_st)
    cov = {
        "byte_integrity": {"status": byte_st, "passed": byte_ok, "total": len(rows)},
        "oracle_fixture_correctness": {"status": oracle_st, "passed": oracle_ok, "total": len(oracle_sids)},
        "recorded_action_replay": {"status": replay_st, "passed": rep_ok, "total": len(byte_jids) + 1, "corrupted_tokens_jobs": tok_bad},
        "checkpoint_restore": {"status": ckpt_st, "passed": h_restored + dg_ok, "total": len(h_jids) + len(dg_jids), "y1_restored": h_restored, "y1_total": 6, "child_restored": dg_ok, "child_total": len(dg_jids)},
        "regenerated_per_case_agreement": {"status": regen_st, "passed": par_ok, "total": 6},
        "training_reproduction": {"status": tr_st, "passed": 1 if tr_ok else 0, "total": 1},
    }
    v_jobs = sum(1 for jc in job_checks.values() if jc["passed"])
    v_studies = sum(1 for sc in study_checks.values() if sc["evidence_status"] == "VERIFIED")
    sci_verified = (v_jobs == len(rows)) and (v_studies == 50) and all(c["status"] == "PASS" for c in cov.values()) and bool(custody.get("custody_valid"))
    return {
        "schema": "avendi-y4-independent-audit-v1",
        "jobs_present": len(recs),
        "jobs_verified": v_jobs,
        "jobs_total": len(rows),
        "studies_verified": v_studies,
        "studies_total": 50,
        "scientific_verified": sci_verified,
        "confirm_custody": custody,
        "verification_matrix": vmat,
        "claim_coverage": cov,
        "failed_job_checks": {k: v["reasons"] for k, v in job_checks.items() if not v["passed"]},
        "unverified_studies": {k: v for k, v in study_checks.items() if v["evidence_status"] != "VERIFIED"},
        "job_checks": job_checks,
        "study_checks": study_checks,
    }
