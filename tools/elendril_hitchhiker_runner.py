"""Command-line harness for Hitchhiker suite verification, Stage 0 audit, and CPU mechanical checks.

Never fabricates parent checkpoints or logits when weights are absent, never promotes
tiny CPU mechanical runs into scientific capability claims, exports complete Stage 0
evidence packages without truncating inspected cases or 260-way distributions, and keeps
executing commit distinct from the audited V7.17 ancestor and historical training commit.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
import tempfile
from typing import Any

from anra_v5.elendril.hitchhiker_data import (
    V6Dataset,
    arm_comparison_table,
    assert_all_stage_disjoint,
    build_counterfactual_suite,
    build_health_probes,
    build_structural_holdout_suite,
    build_training_stream,
    verify_counterfactual_suite,
)
from anra_v5.elendril.hitchhiker_diagnostics import (
    AvailabilityOnlyMaskPolicy,
    ExternalSlotCanonicalizerPolicy,
    GrammarControlHitchhikerPolicy,
    RawHitchhikerPolicy,
    admit_frozen_checkpoint,
    admit_historical_s20_checkpoint,
    classify_diagnostic_outcomes,
    inspect_first_byte_distribution,
    score_entity_equivariance,
)
from anra_v5.elendril.hitchhiker_evaluation import (
    build_experiment_manifest,
    evaluate_counterfactual_suite,
    evaluate_formation_gate,
    evaluate_health_probes,
    evaluate_original_kinds,
    evaluate_screening_gate,
    holm_adjust,
    independent_rescore_hitchhiker,
    paired_world_bootstrap_contrast,
)
from anra_v5.elendril.hitchhiker_protocol import (
    AUDITED_V717_COMMIT,
    HISTORICAL_S20_LOWLR_MANIFESTS,
    PROTOCOL_SHA,
    resolve_scale_profile,
    stage_jobs,
)
from anra_v5.elendril.hitchhiker_training import (
    aux_penalty_gradient_probe,
    create_hitchhiker_trainer,
    resolve_exposure_schedule,
    run_hitchhiker_updates,
)
from anra_v5.elendril.representation import UTF8
from anra_v5.elendril.training import MODEL, configure_cpu
from anra_v5.elendril.v6_data import assert_disjoint as assert_v6_disjoint, make_dataset
from anra_v5.elendril.v6_evaluation import ControlPolicy


DEFAULT_MECHANICAL_OUTPUT = Path("artifacts/elendril/hitchhiker/mechanical_verification.json")
_REPO_ROOT = Path(__file__).resolve().parents[1]
_LINEAGE_TO_SEED = {m["lineage"]: seed for seed, m in HISTORICAL_S20_LOWLR_MANIFESTS.items()}


def resolve_executing_source_commit(explicit_commit: str | None = None) -> str:
    """Resolve the active repository HEAD commit without substituting AUDITED_V717_COMMIT."""
    if explicit_commit is not None:
        if len(explicit_commit) != 40 or any(c not in "0123456789abcdef" for c in explicit_commit):
            raise ValueError(f"invalid 40-char source commit: {explicit_commit!r}")
        return explicit_commit
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    head = proc.stdout.strip()
    if len(head) != 40 or any(c not in "0123456789abcdef" for c in head):
        raise ValueError(f"unexpected git HEAD output: {head!r}")
    return head


def _canonical_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")) + "\n").encode("utf-8")


def _file_sha_and_size(path: Path) -> tuple[str, int]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected regular non-symlink file: {path}")
    digest = sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _compact_inspection(report: dict[str, Any]) -> dict[str, Any]:
    """Return compact scalar/top-k summary while keeping full-260 vectors in the Stage 0 package."""
    return {
        k: v
        for k, v in report.items()
        if k not in ("raw_logits_260", "raw_probs_260", "availability_masked_probs_260")
    }


def _collect_all_stage0_cases(suite: dict[str, Any], probes: dict[str, Any]) -> list[tuple[str, Any]]:
    ordered: list[tuple[str, Any]] = []
    for cohort_name in sorted(suite["cohorts"]):
        for case in suite["cohorts"][cohort_name]:
            ordered.append((f"suite:{cohort_name}", case))
    for probe_name in sorted(probes):
        for case in probes[probe_name]:
            ordered.append((f"probe:{probe_name}", case))
    return ordered


def _evaluate_backend_decoders(
    backend: Any,
    suite: dict[str, Any],
    probes: dict[str, Any],
    *,
    original_dev: V6Dataset | None = None,
    structural_suite: dict[str, Any] | None = None,
    full_260_inspection: bool = False,
) -> dict[str, Any]:
    """Run raw, availability-only, and grammar decoders plus 260-way first-byte diagnostics."""
    decoder_specs = (
        ("raw", RawHitchhikerPolicy),
        ("availability-only", AvailabilityOnlyMaskPolicy),
        ("grammar", GrammarControlHitchhikerPolicy),
    )
    decoder_reports: dict[str, Any] = {}
    for name, policy_cls in decoder_specs:
        factory = policy_cls(backend)
        suite_eval = evaluate_counterfactual_suite(suite, factory)
        probes_eval = evaluate_health_probes(probes, factory, split=suite["split"])
        orig_eval = (
            evaluate_original_kinds(original_dev, factory, split=original_dev.split)
            if original_dev is not None
            else None
        )
        struct_eval = (
            evaluate_counterfactual_suite(structural_suite, factory)
            if structural_suite is not None
            else None
        )
        decoder_reports[name] = {
            "attribution_origin": suite_eval["attribution_origin"],
            "decoder_schema": suite_eval["decoder_schema"],
            "model_identity": suite_eval.get("model_identity"),
            "cohort_metrics": suite_eval["cohort_metrics"],
            "factor_summary": suite_eval["factor_summary"],
            "health_probe_scores": probes_eval["scores"],
            "original_kind_accuracies": orig_eval["original_kind_accuracies"] if orig_eval else None,
            "structural_holdout_metrics": struct_eval["cohort_metrics"] if struct_eval else None,
            "_suite_eval": suite_eval,
            "_probes_eval": probes_eval,
            "_orig_eval": orig_eval,
            "_struct_eval": struct_eval,
        }

    if full_260_inspection:
        target_cases = _collect_all_stage0_cases(suite, probes)
    else:
        target_cases = [
            ("suite:hop-1", suite["cohorts"]["hop-1"][0]),
            ("suite:factor-entity-permutation", suite["cohorts"]["factor-entity-permutation"][0]),
            ("probe:missing-bridge", probes["missing-bridge"][0]),
            ("probe:missing-bridge", probes["missing-bridge"][1]),
        ]

    full_first_byte = [
        {**inspect_first_byte_distribution(backend, case, top_k=5), "source_group": group}
        for group, case in target_cases
    ]
    compact_first_byte = [_compact_inspection(item) for item in full_first_byte[:4]]

    equivariance = score_entity_equivariance(
        backend,
        suite["cohorts"]["factor-entity-permutation"],
        suite["permutations"],
    )
    raw_metrics = decoder_reports["raw"]["cohort_metrics"]["hop-1"]
    masked_metrics = decoder_reports["availability-only"]["cohort_metrics"]["hop-1"]
    outcome = classify_diagnostic_outcomes(
        raw_metrics,
        masked_metrics,
        factor_metrics=decoder_reports["raw"]["factor_summary"],
        equivariance=equivariance,
    )
    return {
        "decoders": decoder_reports,
        "first_byte_inspections": compact_first_byte,
        "full_first_byte_inspections": full_first_byte,
        "entity_equivariance": equivariance,
        "diagnostic_classification": outcome,
    }


def write_stage0_evidence_package(
    package_dir: Path,
    *,
    executing_source_commit: str,
    historical_seed: int,
    admission_receipt: dict[str, Any],
    diag: dict[str, Any],
    suite_case_count: int,
    probe_case_count: int,
) -> dict[str, Any]:
    """Persist complete Stage 0 evidence (all cases, 260-way distributions, traces, prompts) with hashes."""
    if package_dir.is_symlink():
        raise ValueError(f"linked stage0 package directory: {package_dir}")
    package_dir.mkdir(parents=True, exist_ok=True)

    full_inspections = diag["full_first_byte_inspections"]
    inspections_payload = {
        "schema": "elendril-hitchhiker-stage0-first-byte-v1",
        "protocol_sha256": PROTOCOL_SHA,
        "executing_source_commit": executing_source_commit,
        "audited_v717_ancestor": AUDITED_V717_COMMIT,
        "historical_training_source": admission_receipt.get("manifest", {}).get("source_commit"),
        "historical_seed": historical_seed,
        "expected_case_count": suite_case_count + probe_case_count,
        "actual_case_count": len(full_inspections),
        "cases": full_inspections,
    }
    decoder_records_payload = {
        "schema": "elendril-hitchhiker-stage0-decoders-v1",
        "protocol_sha256": PROTOCOL_SHA,
        "executing_source_commit": executing_source_commit,
        "decoders": {
            name: {
                "attribution_origin": dec["attribution_origin"],
                "decoder_schema": dec["decoder_schema"],
                "model_identity": dec["model_identity"],
                "suite_records": dec["_suite_eval"]["records"],
                "probes_records": dec["_probes_eval"]["records"],
                "original_kinds_record": dec["_orig_eval"]["record"] if dec["_orig_eval"] else None,
                "structural_records": dec["_struct_eval"]["records"] if dec["_struct_eval"] else None,
            }
            for name, dec in diag["decoders"].items()
        },
        "entity_equivariance": diag["entity_equivariance"],
        "diagnostic_classification": diag["diagnostic_classification"],
    }

    sidecar_specs = {
        "first_byte_distributions.json": inspections_payload,
        "decoder_evaluations.json": decoder_records_payload,
    }
    file_entries = []
    for filename, payload in sidecar_specs.items():
        path = package_dir / filename
        if path.is_symlink():
            raise ValueError(f"linked stage0 sidecar file: {path}")
        path.write_bytes(_canonical_bytes(payload))
        file_sha, byte_size = _file_sha_and_size(path)
        file_entries.append({"name": filename, "sha256": file_sha, "byte_size": byte_size})

    manifest = {
        "schema": "elendril-hitchhiker-stage0-package-v1",
        "protocol_sha256": PROTOCOL_SHA,
        "executing_source_commit": executing_source_commit,
        "audited_v717_ancestor": AUDITED_V717_COMMIT,
        "historical_training_source": admission_receipt.get("manifest", {}).get("source_commit"),
        "historical_seed": historical_seed,
        "admission_receipt": admission_receipt,
        "expected_case_count": suite_case_count + probe_case_count,
        "files": file_entries,
    }
    manifest_path = package_dir / "stage0_package_manifest.json"
    manifest_bytes = _canonical_bytes(manifest)
    manifest_path.write_bytes(manifest_bytes)
    manifest_sha = sha256(manifest_bytes).hexdigest()
    return {
        "package_dir": str(package_dir),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "expected_case_count": manifest["expected_case_count"],
        "files": file_entries,
    }


def verify_and_rescore_stage0_package(
    package_dir: Path | str,
    *,
    suite: dict[str, Any] | None = None,
    probes: dict[str, Any] | None = None,
    original_dev: V6Dataset | None = None,
    structural_suite: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Independently verify hashed Stage 0 sidecars, 260-way distributions, and raw trace replays."""
    root = Path(package_dir)
    manifest_path = root / "stage0_package_manifest.json"
    manifest_sha, _ = _file_sha_and_size(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "elendril-hitchhiker-stage0-package-v1":
        raise ValueError("invalid stage0 package schema")
    if manifest.get("protocol_sha256") != PROTOCOL_SHA:
        raise ValueError("stage0 package protocol_sha256 mismatch")

    loaded: dict[str, Any] = {}
    for entry in manifest.get("files", []):
        name = entry["name"]
        sidecar = root / name
        if sidecar.resolve().parent != root.resolve():
            raise ValueError(f"stage0 sidecar escapes package dir: {name}")
        actual_sha, actual_size = _file_sha_and_size(sidecar)
        if actual_sha != entry["sha256"] or actual_size != entry["byte_size"]:
            raise ValueError(f"stage0 sidecar integrity mismatch for {name}")
        loaded[name] = json.loads(sidecar.read_text(encoding="utf-8"))

    first_byte = loaded.get("first_byte_distributions.json")
    decoders_doc = loaded.get("decoder_evaluations.json")
    if first_byte is None or decoders_doc is None:
        raise ValueError("stage0 package missing required sidecar files")

    cases = first_byte.get("cases", [])
    expected_total = manifest["expected_case_count"]
    if len(cases) != expected_total:
        raise ValueError(f"stage0 package case count {len(cases)} disagrees with declared {expected_total}")
    if suite is not None and probes is not None:
        suite_n = sum(len(v) for v in suite["cohorts"].values())
        probe_n = sum(len(v) for v in probes.values())
        if len(cases) != suite_n + probe_n:
            raise ValueError(
                f"stage0 package truncated ({len(cases)} cases vs expected {suite_n + probe_n})"
            )

    for idx, row in enumerate(cases):
        raw_logits = row.get("raw_logits_260")
        raw_probs = row.get("raw_probs_260")
        masked_probs = row.get("availability_masked_probs_260")
        for key, vec in (
            ("raw_logits_260", raw_logits),
            ("raw_probs_260", raw_probs),
            ("availability_masked_probs_260", masked_probs),
        ):
            if not isinstance(vec, list) or len(vec) != UTF8.vocabulary_size:
                raise ValueError(f"case {idx} invalid {key} length")
            if any(not isinstance(x, (int, float)) or not math.isfinite(float(x)) for x in vec):
                raise ValueError(f"case {idx} non-finite entry in {key}")
        if not math.isclose(sum(raw_probs), 1.0, rel_tol=1e-3, abs_tol=1e-3):
            raise ValueError(f"case {idx} raw_probs_260 does not sum to 1.0")
        raw_top1 = max(range(UTF8.vocabulary_size), key=lambda i: raw_probs[i])
        top_k_raw = row.get("top_k_raw") or []
        if not top_k_raw or raw_top1 != top_k_raw[0].get("token_id"):
            raise ValueError(f"case {idx} top_k_raw[0] disagrees with raw_probs_260")

    replayed_cohorts = 0
    decoders = decoders_doc.get("decoders", {})
    for name in ("raw", "availability-only", "grammar"):
        if name not in decoders:
            raise ValueError(f"stage0 package missing decoder {name}")
        dec = decoders[name]
        if suite is not None:
            for cohort_name, cohort_cases in suite["cohorts"].items():
                if not cohort_cases:
                    continue
                independent_rescore_hitchhiker(cohort_cases, dec["suite_records"][cohort_name])
                replayed_cohorts += 1
        if probes is not None:
            for probe_name, plist in probes.items():
                if plist:
                    independent_rescore_hitchhiker(plist, dec["probes_records"][probe_name])
        if original_dev is not None and dec.get("original_kinds_record") is not None:
            independent_rescore_hitchhiker(original_dev.cases, dec["original_kinds_record"])
        if structural_suite is not None and dec.get("structural_records") is not None:
            for cohort_name, cohort_cases in structural_suite["cohorts"].items():
                if cohort_cases:
                    independent_rescore_hitchhiker(cohort_cases, dec["structural_records"][cohort_name])

    return {
        "status": "PASS",
        "manifest_sha256": manifest_sha,
        "verified_files": len(manifest["files"]),
        "verified_cases": len(cases),
        "replayed_cohorts": replayed_cohorts,
    }


def run_verify_suite(*, worlds_per_hop: int = 16, source_commit: str | None = None) -> dict[str, Any]:
    """Verify counterfactual/structural suites, split custody, arm comparison table, and oracle baseline."""
    exec_commit = resolve_executing_source_commit(source_commit)
    val_suite = build_counterfactual_suite(seed=17071, worlds_per_hop=worlds_per_hop, split="validation")
    conf_suite = build_counterfactual_suite(seed=17072, worlds_per_hop=worlds_per_hop, split="confirmation")
    struct_suite = build_structural_holdout_suite(seed=37073, worlds_per_topology=max(2, worlds_per_hop))
    probes = build_health_probes(seed=27173, worlds_per_probe=max(2, worlds_per_hop), split="validation")
    suite_check = verify_counterfactual_suite(val_suite)
    conf_check = verify_counterfactual_suite(conf_suite)

    train_base = make_dataset("train", target_bytes=20_000, seed=606)
    dev_orig = make_dataset("validation", target_bytes=6_000, seed=707)
    probe_cases = tuple(c for plist in probes.values() for c in plist)
    val_bundle = V6Dataset(
        split="validation",
        cases=(*(c for cohort in val_suite["cohorts"].values() for c in cohort), *probe_cases),
        examples=(),
        samples=(),
        manifest={"version": "hitchhiker-verify-val-v1"},
    )
    conf_bundle = V6Dataset(
        split="confirmation",
        cases=tuple(c for cohort in conf_suite["cohorts"].values() for c in cohort),
        examples=(),
        samples=(),
        manifest={"version": "hitchhiker-verify-conf-v1"},
    )
    split_check = assert_all_stage_disjoint(
        train_base,
        val_bundle,
        conf_bundle,
        structural_cohorts=struct_suite,
    )
    assert_v6_disjoint(train_base, dev_orig)

    arm_streams = {}
    for arm in ("B0", "A", "B", "AB", "C", "C+penalty", "AB+C", "S20-contrast", "D-external"):
        arm_streams[arm] = build_training_stream(
            arm,
            base_data=train_base,
            total_updates=4,
            batch_size=4,
            data_seed=606,
            order_seed=7,
            relabel_seed=1009,
            heldout_suites=(val_suite, conf_suite, struct_suite, probes),
        )
    comparison = arm_comparison_table(arm_streams)

    teacher_factory = ControlPolicy("fixed")
    teacher_suite = evaluate_counterfactual_suite(val_suite, teacher_factory)
    teacher_probes = evaluate_health_probes(probes, teacher_factory, stage_records=None, split="validation")
    teacher_orig = evaluate_original_kinds(dev_orig, teacher_factory, split="validation")
    return {
        "schema": "elendril-hitchhiker-suite-verify-v2",
        "protocol_sha256": PROTOCOL_SHA,
        "executing_source_commit": exec_commit,
        "audited_v717_ancestor": AUDITED_V717_COMMIT,
        "claim_status": "IMPLEMENTED",
        "worlds_per_hop": worlds_per_hop,
        "suite_verification": suite_check,
        "confirmation_suite_verification": conf_check,
        "structural_holdout_cohorts": {k: len(v) for k, v in struct_suite["cohorts"].items()},
        "split_disjointness": split_check,
        "arm_comparison_table": comparison,
        "teacher_cohort_metrics": teacher_suite["cohort_metrics"],
        "teacher_health_probe_scores": teacher_probes["scores"],
        "teacher_original_kind_accuracies": teacher_orig["original_kind_accuracies"],
    }


def _resolve_seed(lineage: str | None, seed: int | None) -> int:
    if seed is not None:
        return seed
    if lineage is not None and lineage in _LINEAGE_TO_SEED:
        return _LINEAGE_TO_SEED[lineage]
    if lineage is not None:
        raise ValueError(f"unknown historical S20 lineage {lineage!r}; expected {sorted(_LINEAGE_TO_SEED)}")
    return 7


def run_stage0_audit(
    *,
    checkpoint_root: str | None,
    lineage: str | None = "S20-low-lr-i7-o7",
    seed: int | None = None,
    worlds_per_hop: int = 16,
    package_dir: Path | None = None,
    source_commit: str | None = None,
    expected_data_sha: str | None = None,
) -> dict[str, Any]:
    """Audit historical S20-low-lr weights if present, or emit CHECKPOINT_AUDIT_PENDING receipt."""
    exec_commit = resolve_executing_source_commit(source_commit)
    resolved_seed = _resolve_seed(lineage, seed)
    admission = admit_historical_s20_checkpoint(
        checkpoint_root,
        resolved_seed,
        expected_data_sha=expected_data_sha,
        device="cpu",
    )
    admission_receipt = {k: v for k, v in admission.items() if k != "backend"}
    if admission["status"] != "ADMITTED":
        decision_table = [
            {
                "mechanism": "Stage 0 frozen S20-low-lr audit",
                "status": "PENDING",
                "attribution": "model",
                "evidence_summary": f"Checkpoint weights unavailable at {checkpoint_root!r} for seed {resolved_seed}.",
                "next_action": "Mount archived S20-low-lr checkpoints and rerun stage0-audit.",
            }
        ]
        manifest = build_experiment_manifest(
            source_commit=exec_commit,
            stage="stage0_checkpoint_audit",
            stage0_status=admission_receipt,
            arm_runs=[],
            decision_table=decision_table,
            claim_status="PENDING",
        )
        manifest["executing_source_commit"] = exec_commit
        manifest["audited_v717_ancestor"] = AUDITED_V717_COMMIT
        manifest["historical_training_source"] = HISTORICAL_S20_LOWLR_MANIFESTS[resolved_seed]["source_commit"]
        return manifest

    configure_cpu()
    suite = build_counterfactual_suite(seed=17071, worlds_per_hop=worlds_per_hop, split="validation")
    probes = build_health_probes(seed=27173, worlds_per_probe=max(2, worlds_per_hop), split="validation")
    original_dev = make_dataset("validation", target_bytes=6_000, seed=707)
    struct_suite = build_structural_holdout_suite(seed=37073, worlds_per_topology=max(2, worlds_per_hop))
    diag = _evaluate_backend_decoders(
        admission["backend"],
        suite,
        probes,
        original_dev=original_dev,
        structural_suite=struct_suite,
        full_260_inspection=True,
    )
    clean_decoders = {
        k: {
            dk: dv
            for dk, dv in v.items()
            if dk not in ("_suite_eval", "_probes_eval", "_orig_eval", "_struct_eval")
        }
        for k, v in diag["decoders"].items()
    }
    decision_table = [
        {
            "mechanism": f"Stage 0 frozen S20-low-lr audit (seed={resolved_seed})",
            "status": "EXECUTED",
            "attribution": "model_and_decoder_comparison",
            "evidence_summary": f"Findings: {diag['diagnostic_classification']['findings']}",
            "next_action": "Proceed to Stage 1 15-run S3 comparison on GPU.",
        }
    ]
    manifest = build_experiment_manifest(
        source_commit=exec_commit,
        stage="stage0_checkpoint_audit",
        stage0_status=admission_receipt,
        arm_runs=[],
        decision_table=decision_table,
        claim_status="EXECUTED",
    )
    manifest["executing_source_commit"] = exec_commit
    manifest["audited_v717_ancestor"] = AUDITED_V717_COMMIT
    manifest["historical_training_source"] = admission_receipt["manifest"]["source_commit"]
    manifest["stage0_diagnostics"] = {
        "decoders": clean_decoders,
        "first_byte_inspections": diag["first_byte_inspections"],
        "entity_equivariance": diag["entity_equivariance"],
        "diagnostic_classification": diag["diagnostic_classification"],
    }
    if package_dir is not None:
        suite_n = sum(len(v) for v in suite["cohorts"].values())
        probe_n = sum(len(v) for v in probes.values())
        pkg_receipt = write_stage0_evidence_package(
            package_dir,
            executing_source_commit=exec_commit,
            historical_seed=resolved_seed,
            admission_receipt=admission_receipt,
            diag=diag,
            suite_case_count=suite_n,
            probe_case_count=probe_n,
        )
        verify_receipt = verify_and_rescore_stage0_package(
            package_dir,
            suite=suite,
            probes=probes,
            original_dev=original_dev,
            structural_suite=struct_suite,
        )
        manifest["stage0_evidence_package"] = {**pkg_receipt, "verification": verify_receipt}
    return manifest


def run_cpu_mechanical_check(*, worlds_per_hop: int = 2, source_commit: str | None = None) -> dict[str, Any]:
    """Run bounded tiny CPU mechanical verification across protocol, trainers, custody, and gates."""
    configure_cpu()
    exec_commit = resolve_executing_source_commit(source_commit)
    tiny_spec = replace(MODEL, vocabulary_size=UTF8.vocabulary_size, context_length=512)
    param_total = tiny_spec.parameter_receipt().total
    if param_total > 100_000:
        raise ValueError(f"CPU mechanical model unexpectedly large: {param_total}")

    custom_spec = resolve_scale_profile(
        "S9",
        {
            "spec": tiny_spec,
            "parameters": param_total,
            "model_sha256": tiny_spec.sha256(),
        },
    )

    val_suite = build_counterfactual_suite(seed=17071, worlds_per_hop=worlds_per_hop, split="validation")
    conf_suite = build_counterfactual_suite(seed=17072, worlds_per_hop=worlds_per_hop, split="confirmation")
    struct_suite = build_structural_holdout_suite(seed=37073, worlds_per_topology=max(2, worlds_per_hop))
    probes = build_health_probes(seed=27173, worlds_per_probe=max(2, worlds_per_hop), split="validation")
    suite_check = verify_counterfactual_suite(val_suite)
    train_base = make_dataset("train", target_bytes=12_000, seed=606)
    dev_orig = make_dataset("validation", target_bytes=6_000, seed=707)

    probe_cases = tuple(c for plist in probes.values() for c in plist)
    val_bundle = V6Dataset(
        split="validation",
        cases=(*(c for cohort in val_suite["cohorts"].values() for c in cohort), *probe_cases),
        examples=(),
        samples=(),
        manifest={"version": "hitchhiker-mech-val-v1"},
    )
    conf_bundle = V6Dataset(
        split="confirmation",
        cases=tuple(c for cohort in conf_suite["cohorts"].values() for c in cohort),
        examples=(),
        samples=(),
        manifest={"version": "hitchhiker-mech-conf-v1"},
    )
    custody = assert_all_stage_disjoint(
        train_base,
        val_bundle,
        conf_bundle,
        structural_cohorts=struct_suite,
    )
    assert_v6_disjoint(train_base, dev_orig)
    heldout = (val_bundle, conf_bundle, struct_suite)

    s1_jobs = [j for j in stage_jobs("stage1_s3_screen") if j["seed"] == 7]
    s2_jobs = [j for j in stage_jobs("stage2_followups") if j["seed"] == 7 and j["arm"] != "S20-contrast"]
    jobs = [*s1_jobs, *s2_jobs]

    arm_runs: list[dict[str, Any]] = []
    trainers: dict[str, Any] = {}
    streams_by_arm: dict[str, dict[str, Any]] = {}
    aux_probe_result: dict[str, Any] | None = None

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        missing_audit = admit_historical_s20_checkpoint(tmp_path / "missing_checkpoints", 7)
        if missing_audit["status"] != "CHECKPOINT_AUDIT_PENDING":
            raise RuntimeError("expected CHECKPOINT_AUDIT_PENDING for missing historical S20 root")

        ckpt_root = tmp_path / "mechanical_checkpoints"
        for job in jobs:
            trainer, stream = create_hitchhiker_trainer(
                job,
                source_commit=exec_commit,
                root=ckpt_root,
                base_data=train_base,
                heldout_suites=heldout,
                device="cpu",
                cpu_canary=True,
                spec_override=custom_spec,
                updates_override=2,
                batch_override=4,
            )
            streams_by_arm[job["arm"]] = stream
            if job["arm"] == "C+penalty":
                aux_probe_result = aux_penalty_gradient_probe(trainer)
                if aux_probe_result["status"] != "PASS":
                    raise RuntimeError("aux_penalty_gradient_probe failed on C+penalty")
            run_receipt = run_hitchhiker_updates(trainer, save_every=2)
            ckpt_sha = trainer.store.latest_sha256()
            trainers[job["arm"]] = trainer
            arm_runs.append(
                {
                    "job_id": job["job_id"],
                    "arm": job["arm"],
                    "stage": job["stage"],
                    "attribution_origin": job["attribution_origin"],
                    "parameters": param_total,
                    "updates_executed": run_receipt["updates_executed"],
                    "supervised_tokens": run_receipt["supervised_tokens"],
                    "input_positions": run_receipt["input_positions"],
                    "examples_processed": run_receipt["examples_processed"],
                    "final_loss": run_receipt["final_loss"],
                    "checkpoint_sha256": ckpt_sha,
                    "stream_counts": {
                        "accepted_candidates": stream["stream_metrics"]["accepted_candidates"],
                        "rejected_candidates": stream["stream_metrics"]["rejected_candidates"],
                    },
                }
            )

        b0_lineage = s1_jobs[0]["job_id"]
        admitted_b0 = admit_frozen_checkpoint(
            ckpt_root,
            b0_lineage,
            expected_spec=tiny_spec,
            expected_update=2,
            device="cpu",
        )
        if admitted_b0["status"] != "ADMITTED":
            raise RuntimeError("expected local tiny checkpoint admission to succeed")
        b0_stage0 = _evaluate_backend_decoders(
            admitted_b0["backend"],
            val_suite,
            probes,
            original_dev=dev_orig,
            structural_suite=struct_suite,
            full_260_inspection=True,
        )
        stage0_pkg_dir = tmp_path / "stage0_pkg"
        suite_n = sum(len(v) for v in val_suite["cohorts"].values())
        probe_n = sum(len(v) for v in probes.values())
        stage0_pkg = write_stage0_evidence_package(
            stage0_pkg_dir,
            executing_source_commit=exec_commit,
            historical_seed=7,
            admission_receipt=missing_audit,
            diag=b0_stage0,
            suite_case_count=suite_n,
            probe_case_count=probe_n,
        )
        stage0_pkg_verify = verify_and_rescore_stage0_package(
            stage0_pkg_dir,
            suite=val_suite,
            probes=probes,
            original_dev=dev_orig,
            structural_suite=struct_suite,
        )

        ab_backend = trainers["AB"].backend
        ab_suite_eval = evaluate_counterfactual_suite(val_suite, RawHitchhikerPolicy(ab_backend))
        ab_orig_eval = evaluate_original_kinds(dev_orig, RawHitchhikerPolicy(ab_backend), split="validation")
        ab_suite_with_orig = {**ab_suite_eval, "original_kind_accuracies": ab_orig_eval["original_kind_accuracies"], "source_commit": exec_commit}
        b0_suite_with_orig = {
            **b0_stage0["decoders"]["raw"]["_suite_eval"],
            "original_kind_accuracies": b0_stage0["decoders"]["raw"]["original_kind_accuracies"],
            "source_commit": exec_commit,
        }
        ab_probes_eval = evaluate_health_probes(probes, RawHitchhikerPolicy(ab_backend), split="validation")
        ab_equiv = score_entity_equivariance(
            ab_backend,
            val_suite["cohorts"]["factor-entity-permutation"],
            val_suite["permutations"],
        )

        dext_backend = trainers["D-external"].backend
        dext_factory = ExternalSlotCanonicalizerPolicy(dext_backend, assignment="first-mention")
        dext_suite_eval = evaluate_counterfactual_suite(val_suite, dext_factory)

    # Verify fail-closed gates cleanly on tiny mechanical outputs.
    screening_gate = evaluate_screening_gate(
        {7: ab_suite_with_orig},
        {7: b0_suite_with_orig},
        extra_contrasts={
            "ab_vs_b0_perm": paired_world_bootstrap_contrast(
                ab_suite_eval["records"]["factor-entity-permutation"]["episodes"],
                b0_stage0["decoders"]["raw"]["_suite_eval"]["records"]["factor-entity-permutation"]["episodes"],
            )["p_value"],
            "dext_vs_b0_perm": paired_world_bootstrap_contrast(
                dext_suite_eval["records"]["factor-entity-permutation"]["episodes"],
                b0_stage0["decoders"]["raw"]["_suite_eval"]["records"]["factor-entity-permutation"]["episodes"],
            )["p_value"],
        },
    )
    if screening_gate["status"] != "INCOMPLETE_NOT_QUALIFIED":
        raise RuntimeError(f"expected INCOMPLETE_NOT_QUALIFIED on 1-seed check, got {screening_gate['status']}")

    ab_conf_eval = evaluate_counterfactual_suite(conf_suite, RawHitchhikerPolicy(ab_backend))
    ab_conf_equiv = score_entity_equivariance(
        ab_backend,
        conf_suite["cohorts"]["factor-entity-permutation"],
        conf_suite["permutations"],
    )
    formation_gate = evaluate_formation_gate(
        ab_conf_eval,
        equivariance_report=ab_conf_equiv,
    )
    if formation_gate["status"] != "MECHANICAL_ONLY_NOT_FORMATION_QUALIFIED":
        raise RuntimeError("tiny CPU mechanical check was unexpectedly formation-qualified")

    # Verify production exposure schedule calculation does not double-multiply materialized streams.
    s1_ab_prod = [j for j in stage_jobs("stage1_s3_screen") if j["arm"] == "AB" and j["seed"] == 7][0]
    s1_c_prod = [j for j in stage_jobs("stage1_s3_screen") if j["arm"] == "C" and j["seed"] == 7][0]
    prod_ab_stream = build_training_stream(
        "AB",
        base_data=train_base,
        total_updates=8,
        batch_size=16,
        data_seed=606,
        order_seed=7,
        heldout_suites=heldout,
    )
    prod_c_stream = build_training_stream(
        "C",
        base_data=train_base,
        total_updates=math.ceil(len(train_base.samples) / 16) * s1_c_prod["epochs"],
        batch_size=16,
        data_seed=606,
        order_seed=7,
        heldout_suites=heldout,
    )
    prod_schedule_checks = {
        "AB_materialized_updates": resolve_exposure_schedule(s1_ab_prod, stream=prod_ab_stream, batch_size=16),
        "C_static_pool_updates": resolve_exposure_schedule(s1_c_prod, stream=prod_c_stream, batch_size=16),
    }

    comparison = arm_comparison_table(streams_by_arm)

    decision_table = [
        {
            "mechanism": "Stage 0 frozen S20 audit harness",
            "status": "IMPLEMENTED",
            "attribution": "model_vs_mask_vs_grammar",
            "evidence_summary": f"Missing-weights receipt={missing_audit['status']}; local pkg={stage0_pkg_verify['status']}.",
            "next_action": "Execute against real S20-low-lr checkpoints when mounted.",
        },
        {
            "mechanism": "A (Procedural world diversity)",
            "status": "IMPLEMENTED",
            "attribution": "model",
            "evidence_summary": "Fresh split-disjoint stream built and ran 2 tiny CPU updates.",
            "next_action": "Execute 3-seed S3 screen on GPU.",
        },
        {
            "mechanism": "B (Entity permutation invariance)",
            "status": "IMPLEMENTED",
            "attribution": "model",
            "evidence_summary": "Episode-consistent A-P relabeling stream ran 2 tiny CPU updates.",
            "next_action": "Execute 3-seed S3 screen on GPU.",
        },
        {
            "mechanism": "AB (Fresh + Entity relabeling)",
            "status": "IMPLEMENTED",
            "attribution": "model",
            "evidence_summary": f"2 tiny CPU updates verified; screening_gate={screening_gate['status']}; formation_gate={formation_gate['status']}.",
            "next_action": "Execute 3-seed S3 screen on GPU.",
        },
        {
            "mechanism": "C (Matched coverage across availability)",
            "status": "IMPLEMENTED",
            "attribution": "model",
            "evidence_summary": "Matched 14-stratum static pool ran 2 tiny CPU updates.",
            "next_action": "Execute 3-seed S3 screen on GPU.",
        },
        {
            "mechanism": "C+penalty (Aux illegal-action penalty)",
            "status": "IMPLEMENTED",
            "attribution": "model",
            "evidence_summary": f"Gradient probe={aux_probe_result['status']}; ran 2 tiny CPU updates.",
            "next_action": "Conditional Stage 2 arm if availability illegality persists under C.",
        },
        {
            "mechanism": "D-external (Public canonicalizer control)",
            "status": "IMPLEMENTED",
            "attribution": dext_suite_eval["attribution_origin"],
            "evidence_summary": "Public-only canonicalization + dereferencing ran and replayed cleanly.",
            "next_action": "Run as non-learned diagnostic baseline in Stage 2.",
        },
        {
            "mechanism": "Second-wave E/F/G/H",
            "status": "PENDING",
            "attribution": "deferred",
            "evidence_summary": "Explicitly deferred until Stage 1-3 empirical bottlenecks justify.",
            "next_action": "Remain deferred until Stage 1-3 screening completes.",
        },
    ]

    manifest = build_experiment_manifest(
        source_commit=exec_commit,
        stage="cpu_mechanical_verification",
        stage0_status=missing_audit,
        arm_runs=arm_runs,
        decision_table=decision_table,
        claim_status="IMPLEMENTED",
    )
    manifest["executing_source_commit"] = exec_commit
    manifest["audited_v717_ancestor"] = AUDITED_V717_COMMIT
    manifest["mechanical_checks"] = {
        "cpu_model_spec": asdict(tiny_spec),
        "cpu_model_parameters": param_total,
        "suite_verification": suite_check,
        "split_custody": custody,
        "aux_penalty_gradient_probe": aux_probe_result,
        "arm_comparison_table": comparison,
        "production_schedule_checks": prod_schedule_checks,
        "admitted_local_b0_decoders": {
            k: {
                dk: dv
                for dk, dv in v.items()
                if dk not in ("_suite_eval", "_probes_eval", "_orig_eval", "_struct_eval")
            }
            for k, v in b0_stage0["decoders"].items()
        },
        "admitted_local_b0_first_byte": b0_stage0["first_byte_inspections"],
        "admitted_local_b0_classification": b0_stage0["diagnostic_classification"],
        "local_stage0_package_roundtrip": {
            "files": len(stage0_pkg["files"]),
            "expected_case_count": stage0_pkg["expected_case_count"],
            "verification": stage0_pkg_verify,
        },
        "ab_mechanical_probes": ab_probes_eval["scores"],
        "ab_entity_equivariance": ab_equiv,
        "ab_original_kind_accuracies": ab_orig_eval["original_kind_accuracies"],
        "d_external_factor_summary": dext_suite_eval["factor_summary"],
        "d_external_attribution_origin": dext_suite_eval["attribution_origin"],
        "screening_gate_mechanical": screening_gate,
        "formation_gate_mechanical": formation_gate,
        "holm_example": holm_adjust([0.01, 0.04, 0.20]),
        "dataset_smoke_worlds": len(train_base.cases),
    }
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Elendril Hitchhiker evidence runner")
    parser.add_argument(
        "--mode",
        choices=("verify-suite", "stage0-audit", "cpu-mechanical-check"),
        default="verify-suite",
        help="Execution mode (verify-suite, stage0-audit, or bounded cpu-mechanical-check).",
    )
    parser.add_argument("--checkpoint-root", default=None, help="Historical S20 checkpoint directory for stage0-audit.")
    parser.add_argument("--lineage", default="S20-low-lr-i7-o7", help="Historical S20 lineage for stage0-audit.")
    parser.add_argument("--seed", type=int, default=None, help="Historical S20 seed (7, 19, or 41).")
    parser.add_argument("--package-dir", type=Path, default=None, help="Optional sidecar package directory for full Stage 0 artifacts.")
    parser.add_argument("--expected-data-sha", default=None, help="Optional expected S20 training data SHA-256.")
    parser.add_argument("--source-commit", default=None, help="Explicit executing 40-char git commit SHA.")
    parser.add_argument("--worlds-per-hop", type=int, default=None, help="Override worlds per hop.")
    parser.add_argument("--output", default=None, help="Path to write JSON output artifact.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.mode == "verify-suite":
        payload = run_verify_suite(
            worlds_per_hop=args.worlds_per_hop or 16,
            source_commit=args.source_commit,
        )
        out_path = args.output
    elif args.mode == "stage0-audit":
        out_path = args.output
        pkg_dir = args.package_dir
        if pkg_dir is None and out_path is not None:
            pkg_dir = Path(out_path).parent / "stage0_package"
        payload = run_stage0_audit(
            checkpoint_root=args.checkpoint_root,
            lineage=args.lineage,
            seed=args.seed,
            worlds_per_hop=args.worlds_per_hop or 16,
            package_dir=pkg_dir,
            source_commit=args.source_commit,
            expected_data_sha=args.expected_data_sha,
        )
    else:
        worlds = args.worlds_per_hop or 2
        if not 1 <= worlds <= 8:
            raise ValueError("cpu-mechanical-check limits --worlds-per-hop to 1..8")
        payload = run_cpu_mechanical_check(worlds_per_hop=worlds, source_commit=args.source_commit)
        out_path = args.output or str(DEFAULT_MECHANICAL_OUTPUT)

    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if out_path:
        dest = Path(out_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(serialized, encoding="utf-8")
        print(json.dumps({"status": "OK", "mode": args.mode, "output": str(dest), "claim_status": payload.get("claim_status")}))
    else:
        print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
