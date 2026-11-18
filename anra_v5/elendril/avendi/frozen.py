"""Avendi V8-Y2 P1 checkpoint restoration, Y1 parity join verifier, and P2 frozen model diagnosis."""

from __future__ import annotations

from hashlib import sha256
import io
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np
import torch

from anra_v5.elendril.codec import canonical, digest
from anra_v5.elendril.representation import UTF8, SpecialToken
from anra_v5.elendril.y1_data import load_y1_suite
from anra_v5.elendril.y1_diagnostics import (
    extract_and_restore_historical_s3,
    load_verified_y1_checkpoint_backend,
)
from v5_training.checkpoint import CheckpointStore, REQUIRED_COMPONENTS
from .metrics import (
    bootstrap_paired_family_contrasts,
    evaluate_frozen_routing_decision,
    score_single_case_prediction,
    summarize_panel_predictions,
)
from .protocol import (
    INIT_SEEDS,
    ORDER_SEEDS,
    P3_TRAINING_SPEC,
    QUALIFICATION_GATES,
    Y1_BASELINE_LINEAGES,
    Y1_EXECUTED_SOURCE_COMMIT,
    Y1_PROTOCOL_SHA256,
)
from .worlds import AvendiCase, AvendiFamily


def generate_raw_bytes_and_first_logits(
    backend: Any,
    prompt_tokens: Sequence[int],
    *,
    max_new_tokens: int = 3,
) -> tuple[list[int], np.ndarray]:
    """Run raw greedy byte generation until EOS using the Y1 backend and capture 260-way first-action logits."""
    from anra_v5.elendril.hitchhiker_diagnostics import _first_byte_logits

    ans_delim = UTF8.special(SpecialToken.ANSWER)
    seq = [int(t) for t in prompt_tokens]
    if not seq or seq[0] != UTF8.bos_id:
        seq = [UTF8.bos_id, *seq]
    if seq[-1] != ans_delim:
        seq.append(ans_delim)

    first_logits_t = _first_byte_logits(backend, seq)
    first_logits_np = np.asarray(first_logits_t.detach().cpu().numpy(), dtype=np.float32)

    emitted: list[int] = []
    stop_set = (UTF8.eos_id, UTF8.pad_id, UTF8.bos_id, ans_delim)
    for _ in range(max_new_tokens):
        dist = backend.next_distribution(seq)
        next_tok = int(dist.argmax())
        emitted.append(next_tok)
        if next_tok in stop_set:
            break
        seq.append(next_tok)

    return emitted, first_logits_np


def build_deterministic_retraining_recipe(missing_lineages: Sequence[str]) -> dict[str, Any]:
    """Produce a separately labeled deterministic retraining recipe when Y1 FULL weights are absent."""
    return {
        "schema": "anra-avendi-v8-y2-retraining-recipe-v1",
        "status": "REQUIRES_FRESH_IDENTITY_AND_OWNER_COMPUTE_APPROVAL",
        "note": (
            "SUMMARY and EVIDENCE archives do not contain model weights. Recreating missing Y1 S3 lineages "
            "requires a fresh run identity and explicit owner GPU authorization."
        ),
        "original_source_commit": Y1_EXECUTED_SOURCE_COMMIT,
        "original_protocol_sha256": Y1_PROTOCOL_SHA256,
        "missing_lineages": list(missing_lineages),
        "recreation_command_template": (
            "python -m tools.elendril_y1_campaign --phase core "
            f"--source {Y1_EXECUTED_SOURCE_COMMIT} --output <work_dir> --owner-enabled"
        ),
        "hyperparameters": {
            "profile": "S3",
            "updates": 2368,
            "batch_size": 16,
            "micro_batch_size": 2,
            "lr_peak": 1e-4,
            "lr_min_ratio": 0.1,
            "seeds": list(INIT_SEEDS),
            "order_seeds": [ORDER_SEEDS[s] for s in INIT_SEEDS],
        },
    }


def _locate_checkpoint_store_root(checkpoint_root: Path, temp_extract_dir: Path) -> Path | None:
    """Resolve either an unpacked checkpoints directory or a mounted ELENDRIL_V8_Y1_FULL.zip."""
    if checkpoint_root.is_file() and checkpoint_root.suffix.lower() == ".zip":
        with zipfile.ZipFile(checkpoint_root, "r") as zf:
            cp_members = [n for n in zf.namelist() if n.startswith("checkpoints/")]
            if not cp_members:
                return None
            for m in cp_members:
                if ".." in Path(m).parts:
                    raise ValueError(f"unsafe zip member in checkpoint archive: {m}")
                zf.extract(m, path=temp_extract_dir)
        return temp_extract_dir / "checkpoints"

    if checkpoint_root.is_dir():
        if (checkpoint_root / "checkpoints").is_dir():
            return checkpoint_root / "checkpoints"
        # Check if checkpoint_root itself contains lineage subdirectories
        if any((checkpoint_root / jid).is_dir() for jid in Y1_BASELINE_LINEAGES):
            return checkpoint_root
        # Search one level deeper (e.g., /kaggle/input/<dataset>/...)
        for cand in sorted(checkpoint_root.rglob("ELENDRIL_V8_Y1_FULL.zip")):
            if cand.is_file():
                return _locate_checkpoint_store_root(cand, temp_extract_dir)
        for cand in sorted(checkpoint_root.rglob("checkpoints")):
            if cand.is_dir() and any((cand / jid).is_dir() for jid in Y1_BASELINE_LINEAGES):
                return cand
    return None


def _read_evidence_member_bytes(evidence_path: Path, member: str) -> bytes | None:
    if evidence_path.is_file() and evidence_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(evidence_path, "r") as zf:
            if member not in zf.namelist():
                return None
            return zf.read(member)
    if evidence_path.is_dir():
        target = evidence_path / member
        if target.is_file():
            return target.read_bytes()
    return None


def verify_regenerated_y1_case_agreement(
    *,
    job_id: str,
    checkpoint_sha256: str,
    regenerated_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
    reference_Prompt_by_case_id: Mapping[str, str],
    fp32_logit_tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Join regenerated vs archived Y1 predictions/logits by (checkpoint_sha256, rendered_input_sha256, case_id, decoder).

    Enforces exact raw token match and FP32 logit max_abs <= 1e-5. Catches exchanged predictions
    even when aggregate accuracy is identical.
    """
    regen_by_key: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    for row in regenerated_rows:
        cid = str(row["case_id"])
        dec = str(row.get("decoder", "utf8-raw"))
        ckpt = str(row.get("checkpoint_sha256", ""))
        r_sha = str(row.get("rendered_input_sha256") or sha256(str(row.get("public_prompt", "")).encode("utf-8")).hexdigest())
        regen_by_key[(ckpt, r_sha, cid, dec)] = row

    mismatches: list[dict[str, Any]] = []
    max_abs_logit_diff = 0.0
    logit_diffs: list[float] = []
    checked = 0

    for ref in reference_rows:
        cid = str(ref["case_id"])
        dec = str(ref.get("decoder", "utf8-raw"))
        ref_ckpt = str(ref.get("checkpoint_sha256", ""))
        ref_prompt = str(ref.get("public_prompt") or reference_Prompt_by_case_id.get(cid, ""))
        ref_r_sha = str(ref.get("rendered_input_sha256") or sha256(ref_prompt.encode("utf-8")).hexdigest())

        if ref_ckpt != checkpoint_sha256:
            mismatches.append({"case_id": cid, "reason": "CHECKPOINT_SHA_MISMATCH", "expected": ref_ckpt, "observed": checkpoint_sha256})
            continue

        key = (ref_ckpt, ref_r_sha, cid, dec)
        if key not in regen_by_key:
            mismatches.append({"case_id": cid, "reason": "MISSING_OR_HASH_MISMATCHED_CASE", "join_key": list(key)})
            continue

        got = regen_by_key[key]
        checked += 1
        ref_toks = list(ref["raw_tokens"])
        got_toks = list(got["raw_tokens"])
        if ref_toks != got_toks:
            mismatches.append({
                "case_id": cid,
                "reason": "RAW_TOKEN_SEQUENCE_MISMATCH",
                "expected_tokens": ref_toks,
                "observed_tokens": got_toks,
            })

        if ref.get("first_logits") is not None and got.get("first_logits") is not None:
            r_log = np.asarray(ref["first_logits"], dtype=np.float64)
            g_log = np.asarray(got["first_logits"], dtype=np.float64)
            if r_log.shape != g_log.shape or not np.isfinite(g_log).all():
                mismatches.append({"case_id": cid, "reason": "INVALID_LOGIT_SHAPE_OR_NONFINITE"})
            else:
                diff = float(np.max(np.abs(r_log - g_log)))
                max_abs_logit_diff = max(max_abs_logit_diff, diff)
                logit_diffs.append(diff)
                if diff > fp32_logit_tolerance:
                    mismatches.append({
                        "case_id": cid,
                        "reason": "FP32_LOGIT_TOLERANCE_EXCEEDED",
                        "max_abs_diff": diff,
                        "tolerance": fp32_logit_tolerance,
                    })

    tolerance_sensitivity = {
        str(tol): sum(1 for d in logit_diffs if d <= tol)
        for tol in (1e-6, 1e-5, 1e-4, 1e-3)
    }
    matched = len(mismatches) == 0 and checked == len(reference_rows) and len(reference_rows) > 0
    return {
        "schema": "anra-avendi-v8-y2-case-agreement-v1",
        "job_id": job_id,
        "checkpoint_sha256": checkpoint_sha256,
        "status": "PASS" if matched else "FAIL",
        "original_predictions_match": matched,
        "join_keys": ["checkpoint_sha256", "rendered_input_sha256", "case_id", "decoder"],
        "reference_cases": len(reference_rows),
        "checked_cases": checked,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:32],
        "max_abs_logit_diff": max_abs_logit_diff,
        "fp32_logit_tolerance": fp32_logit_tolerance,
        "tolerance_sensitivity_counts": tolerance_sensitivity,
    }


def evaluate_backend_on_avendi_split(
    backend: Any,
    split_bundle: Mapping[str, Any],
    *,
    job_id: str,
    checkpoint_sha256: str,
    logits_npz_path: Path | None = None,
    include_transfer_variants: bool = False,
) -> dict[str, Any]:
    """Score literal copy -> 1-hop crossover -> 3-hop crossover (and optional transfer panels) on the same weights."""
    lit_cases: Sequence[AvendiCase] = split_bundle.get("literal_cases", ())
    one_fams: Sequence[AvendiFamily] = split_bundle.get("one_hop_families", ())
    three_fams: Sequence[AvendiFamily] = split_bundle.get("three_hop_families", ())
    two_fams: Sequence[AvendiFamily] = split_bundle.get("two_hop_families", ())
    four_fams: Sequence[AvendiFamily] = split_bundle.get("four_hop_families", ())

    one_cases = [c for f in one_fams for c in f.cells]
    three_cases = [c for f in three_fams for c in f.cells]
    two_cases = [c for f in two_fams for c in f.cells]
    four_cases = [c for f in four_fams for c in f.cells]

    all_core_cases = [*lit_cases, *one_cases, *three_cases, *two_cases, *four_cases]
    extra_variant_fams: dict[str, dict[str, Sequence[AvendiFamily]]] = {}
    if include_transfer_variants:
        for vk in ("one_hop_transfer_variants", "three_hop_transfer_variants"):
            if vk in split_bundle:
                extra_variant_fams[vk] = dict(split_bundle[vk])
                for vfams in split_bundle[vk].values():
                    for f in vfams:
                        all_core_cases.extend(f.cells)

    case_scores: dict[str, dict[str, Any]] = {}
    logits_rows: list[np.ndarray] = []
    logits_case_ids: list[str] = []

    for c in all_core_cases:
        raw_toks, first_logits = generate_raw_bytes_and_first_logits(backend, c.prompt_tokens)
        sc = score_single_case_prediction(c, raw_tokens=raw_toks, first_logits=first_logits)
        case_scores[c.case_id] = sc
        logits_rows.append(first_logits.astype(np.float32))
        logits_case_ids.append(c.case_id)

    sidecar_receipt = None
    if logits_npz_path is not None and logits_rows:
        logits_npz_path.parent.mkdir(parents=True, exist_ok=True)
        arr = np.stack(logits_rows, axis=0)
        np.savez_compressed(
            logits_npz_path,
            logits=arr,
            case_ids=np.asarray(logits_case_ids, dtype="<U64"),
        )
        raw_bytes = logits_npz_path.read_bytes()
        sidecar_receipt = {
            "path": str(logits_npz_path),
            "sha256": sha256(raw_bytes).hexdigest(),
            "bytes": len(raw_bytes),
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
        }

    out: dict[str, Any] = {
        "schema": "anra-avendi-v8-y2-model-eval-v1",
        "job_id": job_id,
        "checkpoint_sha256": checkpoint_sha256,
        "split": split_bundle.get("split", "dev"),
        "literal_copy": summarize_panel_predictions(
            panel_name="literal_copy",
            cases=lit_cases,
            families=(),
            case_scores_by_id=case_scores,
        ),
        "one_hop": summarize_panel_predictions(
            panel_name="one_hop",
            cases=one_cases,
            families=one_fams,
            case_scores_by_id=case_scores,
        ),
        "three_hop": summarize_panel_predictions(
            panel_name="three_hop",
            cases=three_cases,
            families=three_fams,
            case_scores_by_id=case_scores,
        ),
        "logits_sidecar": sidecar_receipt,
        "case_scores": case_scores,
    }
    if two_fams:
        out["length_two_hop"] = summarize_panel_predictions(
            panel_name="length_two_hop",
            cases=two_cases,
            families=two_fams,
            case_scores_by_id=case_scores,
        )
    if four_fams:
        out["length_four_hop"] = summarize_panel_predictions(
            panel_name="length_four_hop",
            cases=four_cases,
            families=four_fams,
            case_scores_by_id=case_scores,
        )
    if include_transfer_variants and extra_variant_fams:
        tv_summaries: dict[str, Any] = {}
        for vk, vmap in extra_variant_fams.items():
            tv_summaries[vk] = {}
            for vname, vfams in vmap.items():
                vcases = [c for f in vfams for c in f.cells]
                tv_summaries[vk][vname] = summarize_panel_predictions(
                    panel_name=f"{vk}_{vname}",
                    cases=vcases,
                    families=vfams,
                    case_scores_by_id=case_scores,
                )
        out["transfer_variants"] = tv_summaries
    return out


def run_p1_and_p2_frozen_session(
    *,
    suite: Mapping[str, Any],
    output_dir: Path,
    checkpoint_root: Path | None = None,
    evidence_path: Path | None = None,
    device: str = "cpu",
    max_parity_cases: int = 1024,
    expected_spec_override: Any | None = None,
) -> dict[str, Any]:
    """Execute P1 checkpoint restoration + Y1 parity verification and P2 frozen diagnosis across the 6 baseline models."""
    output_dir.mkdir(parents=True, exist_ok=True)
    lineages = sorted(Y1_BASELINE_LINEAGES)
    restored_lineages: dict[str, Any] = {}
    missing_lineages: dict[str, Any] = {}
    parity_reports: dict[str, Any] = {}
    arms_by_seed: dict[str, dict[int, dict[str, Any]]] = {}

    with tempfile.TemporaryDirectory(prefix="avendi-p1-restore-") as tmp_str:
        tmp_dir = Path(tmp_str)
        cp_store_root = (
            _locate_checkpoint_store_root(checkpoint_root, tmp_dir)
            if checkpoint_root is not None and checkpoint_root.exists()
            else None
        )

        # Load reference Y1 development suite from evidence_path if available for P1 parity
        y1_base_cases: list[Any] = []
        y1_prompt_by_cid: dict[str, str] = {}
        if evidence_path is not None and evidence_path.exists():
            dev_bytes = _read_evidence_member_bytes(evidence_path, "prepared/development_suite.json")
            if dev_bytes is not None:
                y1_dev_suite = load_y1_suite(json.loads(dev_bytes.decode("utf-8")))
                y1_base_cases = [c for c in y1_dev_suite["cases"] if c.panel == "main-base"][:max_parity_cases]
                y1_prompt_by_cid = {c.case_id: c.public_prompt for c in y1_base_cases}

        for jid in lineages:
            spec_meta = Y1_BASELINE_LINEAGES[jid]
            if cp_store_root is None or not (cp_store_root / jid).is_dir():
                missing_lineages[jid] = {
                    "job_id": jid,
                    "expected_checkpoint_sha256": spec_meta["checkpoint_sha256"],
                    "missing_components": sorted(REQUIRED_COMPONENTS),
                    "reason": "CHECKPOINT_DIRECTORY_OR_FULL_ARCHIVE_NOT_MOUNTED",
                }
                continue

            lineage_dir = cp_store_root / jid
            try:
                if expected_spec_override is not None:
                    from anra_v5.elendril.hitchhiker_diagnostics import _load_verified_checkpoint_backend

                    st = CheckpointStore(cp_store_root, jid)
                    sha = st.latest_sha256()
                    if sha is None:
                        raise ValueError(f"missing LATEST in {lineage_dir}")
                    state, _, _, _, backend = _load_verified_checkpoint_backend(
                        st,
                        sha,
                        expected_spec=expected_spec_override,
                        device=device,
                    )
                else:
                    state, _, backend, sha = load_verified_y1_checkpoint_backend(
                        cp_store_root,
                        jid,
                        expected_profile="S3",
                        expected_update=int(spec_meta["global_update"]),
                        device=device,
                    )
                    if sha != spec_meta["checkpoint_sha256"]:
                        raise ValueError(
                            f"checkpoint manifest SHA mismatch for {jid}: got {sha}, expected {spec_meta['checkpoint_sha256']}"
                        )
            except Exception as exc:
                missing_lineages[jid] = {
                    "job_id": jid,
                    "expected_checkpoint_sha256": spec_meta["checkpoint_sha256"],
                    "reason": f"RESTORE_VERIFICATION_FAILED: {exc}",
                }
                continue

            restored_lineages[jid] = {
                "job_id": jid,
                "arm": spec_meta["arm"],
                "seed": spec_meta["seed"],
                "checkpoint_sha256": sha,
                "global_update": int(state.global_update),
                "status": "RESTORED_VERIFIED",
            }

            # P1 Reference Parity Check against Y1 EVIDENCE if available
            if y1_base_cases and evidence_path is not None:
                pred_bytes = _read_evidence_member_bytes(evidence_path, f"evidence/predictions/{jid}.json")
                npz_bytes = _read_evidence_member_bytes(evidence_path, f"evidence/logits/{jid}.npz")
                if pred_bytes is not None and npz_bytes is not None:
                    pred_doc = json.loads(pred_bytes.decode("utf-8"))
                    pred_by_cid = {r["case_id"]: r for r in pred_doc["predictions"]}
                    with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as npz:
                        base_logits = npz["base_logits"]
                        base_ids = [str(x) for x in npz["base_case_ids"].tolist()]
                    logit_by_cid = {cid: base_logits[i] for i, cid in enumerate(base_ids)}

                    ref_rows: list[dict[str, Any]] = []
                    regen_rows: list[dict[str, Any]] = []
                    for yc in y1_base_cases:
                        cid = yc.case_id
                        if cid not in pred_by_cid:
                            continue
                        p_rec = pred_by_cid[cid]
                        ref_toks = p_rec["decisions"][0]["raw_tokens"]
                        r_sha = sha256(yc.public_prompt.encode("utf-8")).hexdigest()
                        ref_rows.append({
                            "case_id": cid,
                            "checkpoint_sha256": sha,
                            "decoder": "utf8-raw",
                            "rendered_input_sha256": r_sha,
                            "raw_tokens": ref_toks,
                            "first_logits": logit_by_cid.get(cid),
                        })
                        p_toks = (UTF8.bos_id, *UTF8.encode(yc.public_prompt), UTF8.special(SpecialToken.ANSWER))
                        g_toks, g_logits = generate_raw_bytes_and_first_logits(backend, p_toks)
                        regen_rows.append({
                            "case_id": cid,
                            "checkpoint_sha256": sha,
                            "decoder": "utf8-raw",
                            "rendered_input_sha256": r_sha,
                            "raw_tokens": g_toks,
                            "first_logits": g_logits,
                        })
                    parity_reports[jid] = verify_regenerated_y1_case_agreement(
                        job_id=jid,
                        checkpoint_sha256=sha,
                        regenerated_rows=regen_rows,
                        reference_rows=ref_rows,
                        reference_Prompt_by_case_id=y1_prompt_by_cid,
                    )

            # P2 Frozen Diagnosis on Avendi dev split
            eval_res = evaluate_backend_on_avendi_split(
                backend,
                suite["splits"]["dev"],
                job_id=jid,
                checkpoint_sha256=sha,
                logits_npz_path=output_dir / "evidence" / "logits" / f"{jid}_avendi_dev.npz",
            )
            jdir = output_dir / "analysis" / "jobs" / jid
            jdir.mkdir(parents=True, exist_ok=True)
            compact_eval = {k: v for k, v in eval_res.items() if k != "case_scores"}
            (jdir / "p2_frozen_summary.json").write_bytes(canonical(compact_eval) + b"\n")
            arms_by_seed.setdefault(spec_meta["arm"], {})[int(spec_meta["seed"])] = compact_eval

    # Compute paired family bootstrap between F11 and F00 if both arms were evaluated
    bootstrap_contrasts: dict[str, Any] = {}
    if "F00" in arms_by_seed and "F11" in arms_by_seed:
        for panel in ("one_hop", "three_hop"):
            t_map = {s: arms_by_seed["F11"][s][panel]["family_scores"] for s in arms_by_seed["F11"]}
            c_map = {s: arms_by_seed["F00"][s][panel]["family_scores"] for s in arms_by_seed["F00"]}
            if set(t_map) == set(c_map) and t_map:
                bootstrap_contrasts[panel] = bootstrap_paired_family_contrasts(
                    t_map,
                    c_map,
                    metric_key="all_four_raw_correct",
                )

    routing = evaluate_frozen_routing_decision(arms_by_seed)

    if len(restored_lineages) == len(lineages):
        p1_stage_status = "QUALIFIED"
        p1_ver_status = "PASS"
    elif restored_lineages:
        p1_stage_status = "DIAGNOSTIC"
        p1_ver_status = "PARTIAL"
    else:
        p1_stage_status = "BLOCKED"
        p1_ver_status = "UNAVAILABLE"

    if parity_reports:
        all_par_ok = all(r["original_prediction_agreement"] if "original_prediction_agreement" in r else r["original_predictions_match"] for r in parity_reports.values())
        parity_ver_status = "PASS" if (all_par_ok and len(parity_reports) == len(restored_lineages)) else "FAIL"
    else:
        parity_ver_status = "UNAVAILABLE"

    retraining_recipe = build_deterministic_retraining_recipe(sorted(missing_lineages)) if missing_lineages else None

    receipt = {
        "schema": "anra-avendi-v8-y2-frozen-receipt-v1",
        "p1_stage_status": p1_stage_status,
        "p2_stage_status": routing["status"],
        "verification_rows": {
            "checkpoint_restore": p1_ver_status,
            "regenerated_per_case_agreement": parity_ver_status,
        },
        "restored_lineages_count": len(restored_lineages),
        "expected_lineages_count": len(lineages),
        "restored_lineages": restored_lineages,
        "missing_lineages": missing_lineages,
        "retraining_recipe_if_missing": retraining_recipe,
        "parity_reports": parity_reports,
        "bootstrap_f11_vs_f00": bootstrap_contrasts,
        "routing_decision": routing,
    }
    (output_dir / "analysis").mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis" / "frozen_receipt.json").write_bytes(canonical(receipt) + b"\n")
    return receipt
