"""AN-RA V8-Y1 verified historical restoration, D0/D1 logit diagnostics, and optional D2 probes/patching."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import random
from typing import Any
import zipfile

import numpy as np
import torch

from tools import elendril_v6_export as v6
from tools.elendril_v7_release import verify_checkpoint_members
from v5_training.checkpoint import REQUIRED_COMPONENTS, CheckpointStore
from .backend import NeuralBackend
from .codec import digest
from .cognition import CognitiveState, OperationCandidate
from .hitchhiker_data import render_hitchhiker_public
from .hitchhiker_diagnostics import (
    ANSWER_TOKEN_IDS,
    AvailabilityOnlyMaskPolicy,
    GrammarControlHitchhikerPolicy,
    RawHitchhikerPolicy,
    _first_byte_logits,
    _load_verified_checkpoint_backend,
)
from .representation import SpecialToken, UTF8
from .v6_data import SYMBOLS
from .v6_evaluation import parse_action
from .y1_data import Y1Case
from .y1_evaluation import evaluate_y1_cases, summarize_y1_predictions
from .y1_protocol import (
    HISTORICAL_HH_FULL_ARCHIVE_SHA256,
    HISTORICAL_HH_PROTOCOL_SHA256,
    HISTORICAL_HH_RUN_ID,
    HISTORICAL_HH_S3_MANIFESTS,
    HISTORICAL_HH_WEIGHTS_SOURCE,
    HISTORICAL_S20_RESTORATION_STATUS,
    derive_seed_int,
    resolve_y1_scale,
)


class AnswerOnlyConstrainedEOSPolicy(RawHitchhikerPolicy):
    """Counterfactual auxiliary A-P-only argmax at step 0 + normal next-token EOS verification."""

    origin = "counterfactual_constrained_diagnostic"
    decoder_schema = "utf8-ap-constrained-with-eos"

    def for_case(self, case: Any, *, memory_available: bool | None = None) -> "AnswerOnlyConstrainedEOSPolicy":
        return AnswerOnlyConstrainedEOSPolicy(
            self.backend,
            case.surface,
            memory_available=(case.memory is not None if memory_available is None else memory_available),
            max_new_tokens=self.max_new_tokens,
        )

    def propose(self, state: CognitiveState, legal: frozenset[str], channels: tuple[str, ...], positions: int) -> OperationCandidate | None:
        if self.surface is None:
            raise ValueError("bind public case surface")
        prompt = render_hitchhiker_public(state, channels, self.surface, self.memory_available and "RETRIEVE" in legal)
        ids = [UTF8.bos_id, *UTF8.encode(prompt), UTF8.special(SpecialToken.ANSWER)]
        if len(ids) > self.backend.binding.context_length or len(ids) > positions:
            return None
        logits0 = _first_byte_logits(self.backend, ids)
        probs0 = logits0.softmax(-1)
        best_tid = max(ANSWER_TOKEN_IDS, key=lambda tid: float(logits0[tid]))
        raw = [best_tid]
        probs = [float(probs0[best_tid])]
        used = len(ids)
        ids.append(best_tid)
        # Step 1 uses normal unmodified next-token decoding to verify EOS emission.
        if len(ids) <= self.backend.binding.context_length and used + len(ids) <= positions:
            dist1 = self.backend.next_distribution(ids)
            tok1 = int(dist1.argmax())
            raw.append(tok1)
            probs.append(float(dist1[tok1]))
            used += len(ids)
        op, arg = parse_action(raw, channels)
        return OperationCandidate(op, arg, math.prod(probs), used, len(raw), tuple(raw))


EXPECTED_HH_MANIFEST_SHA256 = "33f88e2e6473c245a7a1d6636db84ddd7ce316cd1fda7fc3eeb337d5618ad68d"


def _read_hh_member(archive_or_dir: Path, member: str) -> bytes:
    if archive_or_dir.is_dir():
        return (archive_or_dir / member).read_bytes()
    with zipfile.ZipFile(archive_or_dir, "r") as zf:
        return zf.read(member)


def verify_and_inspect_hh_full_archive(zip_path: Path | str) -> dict[str, Any]:
    """Verify Hitchhiker FULL archive (.zip or Kaggle-unpacked directory) manifest, source, protocol, and LATEST 2368 pointers."""
    zp = Path(zip_path)
    if zp.is_symlink() or not (zp.is_file() or zp.is_dir()):
        raise ValueError(f"historical checkpoint archive must be a regular file or unpacked dataset directory: {zp}")

    if zp.is_file():
        _, archive_sha = v6._file_sha(zp)
        if archive_sha != HISTORICAL_HH_FULL_ARCHIVE_SHA256:
            raise ValueError(
                f"archive SHA256 mismatch: got {archive_sha}, expected {HISTORICAL_HH_FULL_ARCHIVE_SHA256}"
            )
        checked = v6.verify_zip(zp)
        if checked["source"] != HISTORICAL_HH_WEIGHTS_SOURCE or checked["protocol_sha256"] != HISTORICAL_HH_PROTOCOL_SHA256:
            raise ValueError("archive source or protocol identity mismatch")

        req_members: list[str] = []
        with zipfile.ZipFile(zp, "r") as zf:
            ident = json.loads(zf.read("analysis/identity.json").decode("utf-8"))
            if ident.get("source") != HISTORICAL_HH_WEIGHTS_SOURCE or ident.get("protocol_sha256") != HISTORICAL_HH_PROTOCOL_SHA256:
                raise ValueError("archive analysis/identity.json mismatch")
            markers = [n for n in zf.namelist() if n.startswith("analysis/campaign-") and n.endswith(".json")]
            if len(markers) != 1 or HISTORICAL_HH_RUN_ID not in markers[0]:
                raise ValueError(f"unexpected campaign markers in archive: {markers}")

            for lineage, expected in sorted(HISTORICAL_HH_S3_MANIFESTS.items()):
                latest_member = f"checkpoints/{lineage}/LATEST"
                latest_sha = zf.read(latest_member).decode("ascii").strip()
                if latest_sha != expected["checkpoint_manifest_sha256"]:
                    raise ValueError(f"{lineage} LATEST pointer {latest_sha} != expected {expected['checkpoint_manifest_sha256']}")
                cp_prefix = f"checkpoints/{lineage}/objects/{latest_sha}"
                req_members.append(f"{cp_prefix}/manifest.json")
                req_members.extend(f"{cp_prefix}/{part}" for part in sorted(REQUIRED_COMPONENTS))
            verify_checkpoint_members(zf, req_members)
        mode_tag = "ZIP_VERIFIED"
    else:
        mpath = zp / v6.MANIFEST_NAME
        if not mpath.is_file():
            raise ValueError(f"unpacked archive directory missing {v6.MANIFEST_NAME}: {zp}")
        mbytes = mpath.read_bytes()
        msha = sha256(mbytes).hexdigest()
        if msha != EXPECTED_HH_MANIFEST_SHA256:
            raise ValueError(f"unpacked BUNDLE_MANIFEST.json SHA256 mismatch: {msha} != {EXPECTED_HH_MANIFEST_SHA256}")
        outer = json.loads(mbytes.decode("utf-8"))
        if outer.get("source") != HISTORICAL_HH_WEIGHTS_SOURCE or outer.get("protocol_sha256") != HISTORICAL_HH_PROTOCOL_SHA256:
            raise ValueError("unpacked bundle source or protocol mismatch")
        file_rows = {r["name"]: r for r in outer["files"]}
        ident = json.loads((zp / "analysis" / "identity.json").read_text(encoding="utf-8"))
        if ident.get("source") != HISTORICAL_HH_WEIGHTS_SOURCE or ident.get("protocol_sha256") != HISTORICAL_HH_PROTOCOL_SHA256:
            raise ValueError("unpacked analysis/identity.json mismatch")
        for lineage, expected in sorted(HISTORICAL_HH_S3_MANIFESTS.items()):
            latest_sha = (zp / "checkpoints" / lineage / "LATEST").read_text(encoding="ascii").strip()
            if latest_sha != expected["checkpoint_manifest_sha256"]:
                raise ValueError(f"{lineage} LATEST pointer {latest_sha} != expected {expected['checkpoint_manifest_sha256']}")
            cp_prefix = f"checkpoints/{lineage}/objects/{latest_sha}"
            cp_manifest_bytes = (zp / cp_prefix / "manifest.json").read_bytes()
            if sha256(cp_manifest_bytes).hexdigest() != latest_sha:
                raise ValueError(f"{lineage} checkpoint manifest.json hash mismatch")
            for part in ("manifest.json", *sorted(REQUIRED_COMPONENTS)):
                rel = f"{cp_prefix}/{part}"
                row = file_rows[rel]
                sz, d = v6._file_sha(zp / rel)
                if sz != row["byte_size"] or d != row["sha256"]:
                    raise ValueError(f"unpacked checkpoint component mismatch: {rel}")
        archive_sha = msha
        mode_tag = "UNPACKED_DATASET_VERIFIED"

    return {
        "status": "VERIFIED",
        "intake_mode": mode_tag,
        "archive_path": str(zp),
        "archive_sha256": archive_sha,
        "historical_s20_status": HISTORICAL_S20_RESTORATION_STATUS,
        "verified_lineages": sorted(HISTORICAL_HH_S3_MANIFESTS),
    }


def extract_and_restore_historical_s3(
    zip_path: Path | str,
    lineage: str,
    extract_dir: Path | str,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    """Extract strictly verified final LATEST generation of one HH S3 lineage and load NeuralBackend."""
    if lineage not in HISTORICAL_HH_S3_MANIFESTS:
        raise ValueError(f"unknown historical HH S3 lineage: {lineage!r}")
    expected = HISTORICAL_HH_S3_MANIFESTS[lineage]
    manifest_sha = expected["checkpoint_manifest_sha256"]
    zp = Path(zip_path)
    dest = Path(extract_dir)
    dest.mkdir(parents=True, exist_ok=True)
    cp_root = dest / "checkpoints"

    prefix = f"checkpoints/{lineage}/objects/{manifest_sha}/"
    target_obj_dir = cp_root / lineage / "objects" / manifest_sha
    if not (target_obj_dir / "manifest.json").is_file():
        outer = json.loads(_read_hh_member(zp, v6.MANIFEST_NAME).decode("utf-8"))
        file_rows = {r["name"]: r for r in outer["files"]}
        needed = [f"checkpoints/{lineage}/LATEST", *(prefix + c for c in ("manifest.json", *sorted(REQUIRED_COMPONENTS)))]
        for member in needed:
            out_path = dest / member
            out_path.parent.mkdir(parents=True, exist_ok=True)
            raw = _read_hh_member(zp, member)
            row = file_rows[member]
            if sha256(raw).hexdigest() != row["sha256"] or len(raw) != row["byte_size"]:
                raise ValueError(f"extracted member hash mismatch: {member}")
            out_path.write_bytes(raw)

    store = CheckpointStore(cp_root, lineage)
    if store.latest_sha256() != manifest_sha:
        raise ValueError(f"restored LATEST pointer mismatch for {lineage}")
    expected_spec = resolve_y1_scale("S3")
    state, metadata, spec, tensor_hashes, backend = _load_verified_checkpoint_backend(
        store,
        manifest_sha,
        expected_spec=expected_spec,
        expected_update=expected["global_update"],
        device=device,
    )
    if state.identities.source_commit != HISTORICAL_HH_WEIGHTS_SOURCE:
        raise ValueError(f"{lineage} source_commit mismatch: {state.identities.source_commit}")
    if state.identities.data_manifest_sha256 != expected["identities"]["data_manifest_sha256"]:
        raise ValueError(f"{lineage} data_manifest_sha256 mismatch")

    return {
        "status": "RESTORED_VERIFIED",
        "lineage": lineage,
        "arm": expected["arm"],
        "seed": expected["seed"],
        "checkpoint_sha256": manifest_sha,
        "global_update": state.global_update,
        "parameters": spec.parameter_receipt().total,
        "model_spec_sha256": spec.sha256(),
        "tokenizer_sha256": UTF8.identity,
        "source_commit": HISTORICAL_HH_WEIGHTS_SOURCE,
        "tensor_sha256s": tensor_hashes,
        "backend": backend,
    }


def reproduce_historical_hh_counts(
    zip_path: Path | str,
    lineage: str,
    backend: NeuralBackend,
    *,
    max_cases_per_cohort: int | None = None,
) -> dict[str, Any]:
    """Reproduce published Hitchhiker per-hop (hop-1..4) and original-task counts in the historical runtime."""
    from .hitchhiker_evaluation import evaluate_hitchhiker
    from .v6_data import dataset_from_record
    from tools.elendril_hitchhiker_worker import hitchhiker_case_from_record

    zp = Path(zip_path)
    vsuite = json.loads(_read_hh_member(zp, "prepared/validation_suite.json").decode("utf-8"))
    vorig = dataset_from_record(json.loads(_read_hh_member(zp, "prepared/validation_orig.json").decode("utf-8")))
    fe = json.loads(_read_hh_member(zp, f"analysis/jobs/{lineage}/final_eval.json").decode("utf-8"))

    pol = RawHitchhikerPolicy(backend)
    reproduced_hops: dict[str, int] = {}
    published_hops: dict[str, int] = {}
    for h in ("hop-1", "hop-2", "hop-3", "hop-4"):
        raw_rows = vsuite["cohorts"][h]
        rec_eps = fe["suite_eval"]["records"][h]["episodes"]
        if max_cases_per_cohort is not None:
            raw_rows = raw_rows[:max_cases_per_cohort]
            rec_eps = rec_eps[:max_cases_per_cohort]
        cases = tuple(hitchhiker_case_from_record(r) for r in raw_rows)
        rep = evaluate_hitchhiker(cases, pol)
        reproduced_hops[h] = int(rep["metrics"]["correct"])
        published_hops[h] = sum(int(e["outcome"]["correct"]) for e in rec_eps)

    orig_cases = vorig.cases[:max_cases_per_cohort] if max_cases_per_cohort is not None else vorig.cases
    orig_eps = fe["original_eval"]["record"]["episodes"][:max_cases_per_cohort] if max_cases_per_cohort is not None else fe["original_eval"]["record"]["episodes"]
    orig_rep = evaluate_hitchhiker(orig_cases, pol)
    reproduced_orig = int(orig_rep["metrics"]["correct"])
    published_orig = sum(int(e["outcome"]["correct"]) for e in orig_eps)

    exact = (reproduced_hops == published_hops) and (reproduced_orig == published_orig)
    return {
        "schema": "anra-v8-y1-d0-historical-reproduction-v1",
        "lineage": lineage,
        "status": "EXACT_MATCH" if exact else "MISMATCH_BLOCKER",
        "mismatch_blocker": not exact,
        "reproduced_hops": reproduced_hops,
        "published_hops": published_hops,
        "reproduced_original": reproduced_orig,
        "published_original": published_orig,
        "max_cases_per_cohort": max_cases_per_cohort,
    }


def load_verified_y1_checkpoint_backend(
    checkpoint_root: Path | str,
    lineage: str,
    *,
    expected_profile: str,
    expected_update: int | None = None,
    device: str = "cpu",
) -> tuple[Any, dict[str, Any], NeuralBackend, str]:
    """Load and strictly verify a Y1-trained checkpoint (S3 or S20) in eval mode."""
    store = CheckpointStore(Path(checkpoint_root), lineage)
    latest = store.latest_sha256()
    if latest is None:
        raise ValueError(f"checkpoint lineage {lineage} has no committed LATEST")
    spec = resolve_y1_scale(expected_profile)
    state, metadata, _, tensor_hashes, backend = _load_verified_checkpoint_backend(
        store,
        latest,
        expected_spec=spec,
        expected_update=expected_update,
        device=device,
    )
    return state, {"metadata": metadata, "tensor_sha256s": tensor_hashes}, backend, latest


def run_d0_d1_model_audit(
    backend: NeuralBackend,
    suite: dict[str, Any],
    *,
    job_id: str,
    checkpoint_sha256: str,
    logits_npz_path: Path | str | None = None,
    run_supplemental_base: bool = True,
    max_failed_trajectories: int = 32,
) -> dict[str, Any]:
    """Execute D0 raw/supplemental evaluations + D1 260-way logit analysis + binary sidecar export."""
    cases: tuple[Y1Case, ...] = suite["cases"]
    hist_cases: tuple[Y1Case, ...] = suite.get("historical_continuity", ())

    # 1. Primary raw unconstrained evaluation across all cases + separate historical continuity.
    raw_pol = RawHitchhikerPolicy(backend)
    raw_preds = evaluate_y1_cases(cases, raw_pol, job_id=job_id, decoder_condition="raw", checkpoint_sha256=checkpoint_sha256)
    hist_preds = (
        evaluate_y1_cases(hist_cases, raw_pol, job_id=job_id, decoder_condition="raw", checkpoint_sha256=checkpoint_sha256)
        if hist_cases
        else None
    )
    raw_summary = summarize_y1_predictions(
        cases,
        raw_preds,
        historical_continuity_cases=hist_cases,
        historical_pred_doc=hist_preds,
    )

    # 2. D0 base logits capture [N_base, 260] FP32 + D1 analysis on exact unmodified base logits.
    base_cases = tuple(c for c in cases if c.panel == "main-base")
    n_base = len(base_cases)
    base_logits = np.zeros((n_base, UTF8.vocabulary_size), dtype=np.float32)
    base_case_ids: list[str] = []
    pred_by_cid = {r["case_id"]: r for r in raw_preds["predictions"]}

    d1_rows: list[dict[str, Any]] = []
    ap_mass_all: list[float] = []
    known_rank1_ap_count = 0
    known_raw_exact_count = 0
    known_ap_eos_ok_count = 0
    known_total = 0

    for idx, c in enumerate(base_cases):
        st = CognitiveState(c.goal).observe(c.environment.initial)
        prompt = render_hitchhiker_public(st, tuple(c.environment.channels), c.surface, c.memory is not None)
        ids = [UTF8.bos_id, *UTF8.encode(prompt), UTF8.special(SpecialToken.ANSWER)]
        logits_t = _first_byte_logits(backend, ids)
        probs_t = logits_t.softmax(-1)
        base_logits[idx] = logits_t.numpy().astype(np.float32)
        base_case_ids.append(c.case_id)

        raw_p = [float(x) for x in probs_t.tolist()]
        ap_mass = math.fsum(raw_p[tid] for tid in ANSWER_TOKEN_IDS)
        ap_mass_all.append(ap_mass)

        ap_probs = {SYMBOLS[i]: raw_p[ANSWER_TOKEN_IDS[i]] for i in range(len(SYMBOLS))}
        ap_sum = ap_mass or 1e-12
        cond_probs = {sym: p / ap_sum for sym, p in ap_probs.items()}
        best_ap_sym = max(SYMBOLS, key=lambda s: ap_probs[s])
        gold_sym = c.truth.expected if c.gold_operation == "ANSWER" else None
        gold_abs_prob = ap_probs[gold_sym] if gold_sym in ap_probs else None
        gold_cond_prob = cond_probs[gold_sym] if gold_sym in cond_probs else None
        gold_abs_rank = (1 + sum(1 for p in raw_p if p > gold_abs_prob)) if gold_abs_prob is not None else None
        gold_cond_rank = (1 + sum(1 for p in ap_probs.values() if p > gold_abs_prob)) if gold_abs_prob is not None else None

        # Check constrained A-P + normal next-token EOS verification when answerable.
        ap_eos_ok = False
        if c.solvability == "answerable":
            known_total += 1
            best_tid = ANSWER_TOKEN_IDS[SYMBOLS.index(best_ap_sym)]
            dist1 = backend.next_distribution([*ids, best_tid])
            ended_eos = int(dist1.argmax()) == UTF8.eos_id
            ap_eos_ok = (best_ap_sym == gold_sym) and ended_eos
            known_rank1_ap_count += int(gold_cond_rank == 1)
            known_ap_eos_ok_count += int(ap_eos_ok)
            known_raw_exact_count += int(pred_by_cid[c.case_id]["correct"])

        d1_rows.append(
            {
                "case_id": c.case_id,
                "hops": c.hops,
                "solvability": c.solvability,
                "p_answer_mass": ap_mass,
                "p_query": raw_p[UTF8.encode("Q")[0]],
                "p_abstain": raw_p[UTF8.encode("!")[0]],
                "p_retrieve": raw_p[UTF8.encode("R")[0]],
                "gold_operation": c.gold_operation,
                "gold_symbol": gold_sym,
                "gold_abs_prob": gold_abs_prob,
                "gold_abs_rank": gold_abs_rank,
                "gold_cond_prob": gold_cond_prob,
                "gold_cond_rank": gold_cond_rank,
                "best_ap_symbol": best_ap_sym,
                "raw_correct": pred_by_cid[c.case_id]["correct"],
                "constrained_ap_eos_correct": ap_eos_ok if c.solvability == "answerable" else None,
            }
        )

    # 3. Capture full trajectory FP32 logits for first max_failed_trajectories failed cases in sorted case_id order
    failed_sorted = sorted(
        (c for c in cases if not pred_by_cid[c.case_id]["correct"]),
        key=lambda c: c.case_id,
    )[:max_failed_trajectories]
    traj_arrays: dict[str, np.ndarray] = {}
    for fidx, fc in enumerate(failed_sorted):
        st = CognitiveState(fc.goal).observe(fc.environment.initial)
        ch = tuple(fc.environment.channels)
        step_logits: list[np.ndarray] = []
        for _d in range(2):
            pr = render_hitchhiker_public(st, ch, fc.surface, fc.memory is not None)
            ids = [UTF8.bos_id, *UTF8.encode(pr), UTF8.special(SpecialToken.ANSWER)]
            if len(ids) > backend.binding.context_length:
                break
            lt0 = _first_byte_logits(backend, ids)
            step_logits.append(lt0.numpy().astype(np.float32))
            tok0 = int(lt0.argmax())
            # Also collect step 1 next-token logits if action byte didn't terminate.
            raw_seq = [tok0]
            if tok0 not in (UTF8.eos_id, UTF8.pad_id, UTF8.bos_id, UTF8.special(SpecialToken.ANSWER)) and len(ids) + 1 <= backend.binding.context_length:
                lt1 = _first_byte_logits(backend, [*ids, tok0])
                step_logits.append(lt1.numpy().astype(np.float32))
                raw_seq.append(int(lt1.argmax()))
            op, arg = parse_action(raw_seq, ch)
            if op == "QUERY" and arg in fc.environment.channels:
                st = st.observe(fc.environment.channels[str(arg)])
                ch = ()
            else:
                break
        if step_logits:
            traj_arrays[f"fail_{fidx:02d}"] = np.stack(step_logits, axis=0)

    sidecar_meta = None
    if logits_npz_path is not None:
        lp = Path(logits_npz_path)
        lp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            lp,
            base_logits=base_logits,
            base_case_ids=np.array(base_case_ids),
            failed_case_ids=np.array([c.case_id for c in failed_sorted]),
            **traj_arrays,
        )
        sidecar_meta = {
            "path": str(lp),
            "sha256": v6._file_sha(lp)[1],
            "bytes": lp.stat().st_size,
            "base_logits_dtype": str(base_logits.dtype),
            "base_logits_shape": list(base_logits.shape),
            "failed_trajectories_count": len(failed_sorted),
        }

    # 4. Supplemental base availability-only and grammar decoding + A-P constrained on base/edge/dual-query
    supplemental: dict[str, Any] = {}
    if run_supplemental_base:
        av_preds = evaluate_y1_cases(
            base_cases,
            AvailabilityOnlyMaskPolicy(backend),
            job_id=job_id,
            decoder_condition="availability-only",
            checkpoint_sha256=checkpoint_sha256,
        )
        gr_preds = evaluate_y1_cases(
            base_cases,
            GrammarControlHitchhikerPolicy(backend),
            job_id=job_id,
            decoder_condition="grammar",
            checkpoint_sha256=checkpoint_sha256,
        )
        ep_cases = tuple(c for c in cases if c.panel == "edge-change-panel")
        dq_cases = tuple(c for c in cases if c.panel == "dual-chain-query-panel")
        ap_pol = AnswerOnlyConstrainedEOSPolicy(backend)
        ap_ep = evaluate_y1_cases(ep_cases, ap_pol, job_id=job_id, decoder_condition="ap-constrained-eos", checkpoint_sha256=checkpoint_sha256) if ep_cases else {"predictions": []}
        ap_dq = evaluate_y1_cases(dq_cases, ap_pol, job_id=job_id, decoder_condition="ap-constrained-eos", checkpoint_sha256=checkpoint_sha256) if dq_cases else {"predictions": []}

        # Pair-both-correct for constrained edge & dual-query
        ep_by_w: dict[str, list[bool]] = {}
        for r in ap_ep["predictions"]:
            ep_by_w.setdefault(r["world_id"], []).append(bool(r["correct"]))
        dq_by_w: dict[str, list[bool]] = {}
        for r in ap_dq["predictions"]:
            dq_by_w.setdefault(r["world_id"], []).append(bool(r["correct"]))

        supplemental = {
            "base_availability_only_accuracy": sum(1 for r in av_preds["predictions"] if r["correct"]) / n_base,
            "base_grammar_accuracy": sum(1 for r in gr_preds["predictions"] if r["correct"]) / n_base,
            "constrained_edge_both_correct": sum(1 for vs in ep_by_w.values() if len(vs) == 3 and all(vs)) / max(1, len(ep_by_w)),
            "constrained_dual_query_both_correct": sum(1 for vs in dq_by_w.values() if len(vs) == 2 and all(vs)) / max(1, len(dq_by_w)),
        }

    known_raw_rate = known_raw_exact_count / known_total if known_total else 0.0
    known_ap_eos_rate = known_ap_eos_ok_count / known_total if known_total else 0.0
    known_rank1_rate = known_rank1_ap_count / known_total if known_total else 0.0
    d1_summary = {
        "schema": "anra-v8-y1-d1-analysis-v1",
        "base_cases": n_base,
        "known_base_cases": known_total,
        "mean_p_answer_mass": math.fsum(ap_mass_all) / len(ap_mass_all) if ap_mass_all else 0.0,
        "known_base_raw_exact_rate": known_raw_rate,
        "known_base_rank1_within_ap_rate": known_rank1_rate,
        "known_base_constrained_ap_eos_rate": known_ap_eos_rate,
        "descriptive_rescue_constrained_minus_raw": known_ap_eos_rate - known_raw_rate,
        "constrained_edge_both_correct": supplemental.get("constrained_edge_both_correct"),
        "constrained_dual_query_both_correct": supplemental.get("constrained_dual_query_both_correct"),
        "base_availability_only_accuracy": supplemental.get("base_availability_only_accuracy"),
        "base_grammar_accuracy": supplemental.get("base_grammar_accuracy"),
        "per_case_compact": d1_rows,
    }

    return {
        "schema": "anra-v8-y1-d0d1-audit-v1",
        "job_id": job_id,
        "checkpoint_sha256": checkpoint_sha256,
        "raw_predictions": raw_preds,
        "historical_predictions": hist_preds,
        "raw_summary": raw_summary,
        "d1_summary": d1_summary,
        "logits_sidecar": sidecar_meta,
    }


def extract_layer_residuals(backend: NeuralBackend, cases: tuple[Y1Case, ...]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Capture final public-prompt residual states across embedding, each transformer block, and final_norm."""
    layers_out: dict[str, list[np.ndarray]] = {}
    targets: list[int] = []
    sym_idx = {s: i for i, s in enumerate(SYMBOLS)}
    captured: dict[str, torch.Tensor] = {}
    hooks = []

    def make_hook(name: str):
        def _hook(_mod: Any, _inp: Any, out: Any) -> None:
            captured[name] = out[:, -1, :].detach().float().cpu()
        return _hook

    hooks.append(backend.model.embedding.register_forward_hook(make_hook("layer_0_emb")))
    for lidx, block in enumerate(backend.model.blocks):
        hooks.append(block.register_forward_hook(make_hook(f"block_{lidx}")))
    hooks.append(backend.model.final_norm.register_forward_hook(make_hook("final_norm")))

    modes = [(m, m.training) for m in backend.model.modules()]
    backend.model.eval()
    try:
        with torch.inference_mode():
            for c in cases:
                st = CognitiveState(c.goal).observe(c.environment.initial)
                pr = render_hitchhiker_public(st, tuple(c.environment.channels), c.surface, False)
                ids = [UTF8.bos_id, *UTF8.encode(pr), UTF8.special(SpecialToken.ANSWER)]
                captured.clear()
                _ = backend.logits(torch.tensor([ids], device=backend.device, dtype=torch.int64), torch.zeros((1, len(ids)), device=backend.device, dtype=torch.int64))
                for k, v in captured.items():
                    layers_out.setdefault(k, []).append(v[0].numpy())
                targets.append(sym_idx[str(c.truth.expected)])
    finally:
        for h in hooks:
            h.remove()
        for m, md in modes:
            m.training = md

    return {k: np.stack(v, axis=0) for k, v in layers_out.items()}, np.array(targets, dtype=np.int64)


def _fit_l2_linear_probe(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """Deterministic L2=1.0 multinomial logistic regression probe on frozen residuals."""
    x_tr = torch.tensor(X_train, dtype=torch.float64)
    y_tr = torch.tensor(y_train, dtype=torch.int64)
    x_te = torch.tensor(X_test, dtype=torch.float64)
    y_te = torch.tensor(y_test, dtype=torch.int64)
    w = torch.zeros((X_train.shape[1], len(SYMBOLS)), dtype=torch.float64, requires_grad=True)
    b = torch.zeros(len(SYMBOLS), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=0.5, max_iter=60, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logits = x_tr @ w + b
        loss = torch.nn.functional.cross_entropy(logits, y_tr) + 0.5 * 1.0 * (w * w).sum() / len(x_tr)
        loss.backward()
        return loss

    opt.step(closure)
    with torch.inference_mode():
        pred = (x_te @ w + b).argmax(-1)
        return float((pred == y_te).float().mean().item())


def run_d2_probe_and_patching(
    backend: NeuralBackend,
    probe_data: dict[str, Any],
    *,
    enabled: bool = False,
) -> dict[str, Any]:
    """Run D2 only when separately admitted/enabled; otherwise return DEFERRED_NOT_ADMITTED."""
    if not enabled:
        return {"schema": "anra-v8-y1-d2-v1", "status": "DEFERRED_NOT_ADMITTED", "executed": False}

    train_cases: tuple[Y1Case, ...] = probe_data["train_cases"]
    test_cases: tuple[Y1Case, ...] = probe_data["test_cases"]
    tr_res, tr_y = extract_layer_residuals(backend, train_cases)
    te_res, te_y = extract_layer_residuals(backend, test_cases)

    hop_reports: dict[str, Any] = {}
    best_layer_by_hop: dict[int, str] = {}
    all_hops_above_20pp = True
    rng = random.Random(derive_seed_int("d2-shuffle", probe_data["seed"]))

    for h in (1, 2, 3, 4):
        tr_mask = np.array([c.hops == h for c in train_cases])
        te_mask = np.array([c.hops == h for c in test_cases])
        y_tr_h = tr_y[tr_mask]
        y_te_h = te_y[te_mask]
        shuf = list(y_tr_h.tolist())
        rng.shuffle(shuf)
        y_shuf = np.array(shuf, dtype=np.int64)
        maj = int(np.bincount(y_tr_h).argmax())
        maj_acc = float((y_te_h == maj).mean())

        layer_scores: dict[str, Any] = {}
        best_acc = -1.0
        best_shuf = 0.0
        best_lname = "block_0"
        for lname in sorted(tr_res):
            acc = _fit_l2_linear_probe(tr_res[lname][tr_mask], y_tr_h, te_res[lname][te_mask], y_te_h)
            sh_acc = _fit_l2_linear_probe(tr_res[lname][tr_mask], y_shuf, te_res[lname][te_mask], y_te_h)
            layer_scores[lname] = {"heldout_accuracy": acc, "shuffled_control": sh_acc, "majority_baseline": maj_acc}
            if acc > best_acc:
                best_acc = acc
                best_shuf = sh_acc
                best_lname = lname
        if (best_acc - best_shuf) < 0.20:
            all_hops_above_20pp = False
        best_layer_by_hop[h] = best_lname
        hop_reports[f"hop-{h}"] = {"best_layer": best_lname, "best_minus_shuffled": best_acc - best_shuf, "layers": layer_scores}

    patch_result: dict[str, Any] = {"status": "NOT_TRIGGERED", "reason": "probe_threshold_20pp_not_met_on_all_hops"}
    if all_hops_above_20pp:
        # Bounded equal-token-aligned residual patching verification (with mandatory <= 1e-5 no-patch gate).
        max_no_patch_err = 0.0
        patch_deltas: list[float] = []
        excluded = 0
        for h in (1, 2, 3, 4):
            h_cases = [c for c in test_cases if c.hops == h][:64]
            target_block = backend.model.blocks[min(len(backend.model.blocks) - 1, 1)]
            for idx, bc in enumerate(h_cases):
                donor_c = h_cases[(idx + 1) % len(h_cases)]
                st_b = CognitiveState(bc.goal).observe(bc.environment.initial)
                st_d = CognitiveState(donor_c.goal).observe(donor_c.environment.initial)
                ids_b = [UTF8.bos_id, *UTF8.encode(render_hitchhiker_public(st_b, (), bc.surface, False)), UTF8.special(SpecialToken.ANSWER)]
                ids_d = [UTF8.bos_id, *UTF8.encode(render_hitchhiker_public(st_d, (), donor_c.surface, False)), UTF8.special(SpecialToken.ANSWER)]
                if len(ids_b) != len(ids_d):
                    excluded += 1
                    continue
                base_logits = _first_byte_logits(backend, ids_b)
                # Verify hook no-patch invariance <= 1e-5.
                hook = target_block.register_forward_hook(lambda _m, _i, o: o.clone())
                try:
                    nopatch_logits = _first_byte_logits(backend, ids_b)
                finally:
                    hook.remove()
                err = float((base_logits - nopatch_logits).abs().max().item())
                max_no_patch_err = max(max_no_patch_err, err)
                donor_cache: dict[str, torch.Tensor] = {}
                cap = target_block.register_forward_hook(lambda _m, _i, o: donor_cache.__setitem__("h", o[:, -1, :].detach().clone()))
                try:
                    _ = _first_byte_logits(backend, ids_d)
                finally:
                    cap.remove()

                def _replace(_m: Any, _i: Any, o: torch.Tensor) -> torch.Tensor:
                    out = o.clone()
                    out[:, -1, :] = donor_cache["h"]
                    return out

                rep = target_block.register_forward_hook(_replace)
                try:
                    patched_logits = _first_byte_logits(backend, ids_b)
                finally:
                    rep.remove()
                gt_tid = UTF8.encode(str(bc.truth.expected))[0]
                dt_tid = UTF8.encode(str(donor_c.truth.expected))[0]
                orig_diff = float(base_logits[dt_tid] - base_logits[gt_tid])
                new_diff = float(patched_logits[dt_tid] - patched_logits[gt_tid])
                patch_deltas.append(abs(new_diff - orig_diff))

        patch_result = {
            "status": "PASS" if max_no_patch_err <= 1e-5 else "NOT_QUALIFIED",
            "max_no_patch_logit_diff": max_no_patch_err,
            "evaluated_pairs": len(patch_deltas),
            "excluded_misaligned": excluded,
            "mean_abs_log_odds_delta": math.fsum(patch_deltas) / len(patch_deltas) if patch_deltas else 0.0,
        }

    return {
        "schema": "anra-v8-y1-d2-v1",
        "status": "EXECUTED",
        "executed": True,
        "hops": hop_reports,
        "patching": patch_result,
    }
