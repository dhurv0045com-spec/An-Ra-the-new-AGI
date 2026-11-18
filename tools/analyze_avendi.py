"""Offline Avendi V8-Y2 evidence analyzer and RESULTS.md / METRICS.json generator."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping
import zipfile

from anra_v5.elendril.avendi.protocol import (
    DESIGN_SHA,
    PERMANENT_Y1_DISCOVERY_REGISTER,
    VERIFICATION_ROW_NAMES,
)
from anra_v5.elendril.codec import canonical
from tools import elendril_v6_export as v6


def _read_member(source: Path, rel: str) -> dict[str, Any] | None:
    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source, "r") as zf:
            if rel not in zf.namelist():
                return None
            return json.loads(zf.read(rel).decode("utf-8"))
    if source.is_dir():
        target = source / rel
        if target.is_file():
            return json.loads(target.read_text(encoding="utf-8"))
    return None


def analyze_avendi_evidence(evidence_path: Path | str) -> dict[str, Any]:
    """Verify Avendi evidence archive or directory and assemble complete 6-row verification and stage report."""
    ep = Path(evidence_path)
    zip_check = None
    archive_sha = None
    archive_bytes = None
    if ep.is_file() and ep.suffix.lower() == ".zip":
        zip_check = v6.verify_zip(ep)
        archive_bytes, archive_sha = v6._file_sha(ep)

    stage_doc = _read_member(ep, "analysis/STAGE_STATUS.json") or {}
    p0_doc = _read_member(ep, "analysis/p0_instrument_receipt.json") or {}
    frozen_doc = _read_member(ep, "analysis/frozen_receipt.json")
    p3_doc = _read_member(ep, "analysis/p3_acquisition_gate.json")
    p4_doc = _read_member(ep, "analysis/p4_transition_receipt.json")
    p5_doc = _read_member(ep, "analysis/p5_conditional_receipt.json")
    p6_doc = _read_member(ep, "analysis/p6_correction_receipt.json")
    reg_state = _read_member(ep, "analysis/REGISTRY_STATE.json")

    verification_rows = dict(stage_doc.get("verification_rows", {}))
    if zip_check is not None:
        verification_rows["byte_integrity"] = (
            "PASS" if zip_check.get("status") in ("EXPORT_VERIFIED", "VERIFIED", "ZIP_VERIFIED") else "FAIL"
        )

    return {
        "schema": "anra-avendi-v8-y2-analysis-v1",
        "evidence_path": str(ep),
        "archive_sha256": archive_sha,
        "archive_bytes": archive_bytes,
        "zip_verification": zip_check,
        "run_id": stage_doc.get("run_id", "avendi-local"),
        "phase": stage_doc.get("phase", "fixture"),
        "source": stage_doc.get("source", "unknown"),
        "design_sha256": stage_doc.get("design_sha256", DESIGN_SHA),
        "stage_statuses": stage_doc.get("stage_statuses", {}),
        "verification_rows": verification_rows,
        "p0_instrument": p0_doc,
        "frozen_diagnosis": frozen_doc,
        "p3_acquisition_gate": p3_doc,
        "p4_transition": p4_doc,
        "p5_conditional": p5_doc,
        "p6_correction": p6_doc,
        "registry_design_sha256_preserved": (
            reg_state.get("design_sha256") == DESIGN_SHA if reg_state is not None else None
        ),
    }


def render_avendi_results_markdown(metrics: Mapping[str, Any]) -> str:
    """Render concise RESULTS.md with 6-row verification matrix, stage statuses, P0/P2/P3 tables, and scenario register."""
    lines = [
        f"# Avendi V8-Y2 Results Report (`{metrics['run_id']}`)",
        "",
        f"- **Phase**: `{metrics['phase']}`",
        f"- **Source commit**: `{metrics['source']}`",
        f"- **Design SHA256**: `{metrics['design_sha256']}`",
        f"- **Archive SHA256**: `{metrics.get('archive_sha256') or 'directory_input'}`",
        "",
        "## 1. Six-Row Independent Verification Matrix",
        "",
        "| Verification Row | Status | Notes |",
        "| --- | --- | --- |",
    ]
    vrows = metrics.get("verification_rows", {})
    notes_map = {
        "byte_integrity": "Archive manifest and member SHA-256 verification",
        "oracle_fixture_correctness": "Independent tokenized-input parser, destructive erasure, and blind-policy zero ceiling",
        "recorded_action_replay": "Public environment replay of recorded policy actions",
        "checkpoint_restore": "Y1 S3 F00/F11 manifest, model spec, and finite tensor verification",
        "regenerated_per_case_agreement": "Joined by (checkpoint_sha256, rendered_input_sha256, case_id, decoder), FP32 max_abs <= 1e-5",
        "training_reproduction": "P3 paired margin gradient-sign negative control and family-aligned stream verification",
    }
    for rname in VERIFICATION_ROW_NAMES:
        lines.append(f"| `{rname}` | `{vrows.get(rname, 'UNAVAILABLE')}` | {notes_map.get(rname, '')} |")

    lines.extend([
        "",
        "## 2. Scientific Stage Statuses",
        "",
        "| Stage | Status |",
        "| --- | --- |",
    ])
    for st, val in metrics.get("stage_statuses", {}).items():
        lines.append(f"| `{st}` | `{val}` |")

    p0 = metrics.get("p0_instrument") or {}
    if p0:
        pilot = p0.get("pilot_reference_1801", {})
        dev1 = p0.get("dev_one_hop_policies", {})
        dev3 = p0.get("dev_three_hop_policies", {})
        lines.extend([
            "",
            "## 3. P0 CPU Instrument & Reference Policies",
            "",
            f"- **Custody & Disjointness Audit**: `{p0.get('custody_audit', {}).get('status')}` (canonical 1-hop pool: `{p0.get('custody_audit', {}).get('canonical_one_hop_pool_size')}` families)",
            f"- **Planted Target-Only / ID-Changed Collision Control**: `{p0.get('planted_collision_negative_control', {}).get('status')}`",
            f"- **Destructive Goal-Erasure & Bridge-Masking Checks**: 1-hop `{p0.get('destructive_checks', {}).get('one_hop', {}).get('status')}`, 3-hop `{p0.get('destructive_checks', {}).get('three_hop', {}).get('status')}`",
            f"- **Pilot Seed 1801 Complete Families**: oracle `{pilot.get('all_four_correct', {}).get('traversal')}/256`, last_object `{pilot.get('all_four_correct', {}).get('last_object')}/256`, third_timestamp `{pilot.get('all_four_correct', {}).get('third_timestamp')}/256`, ignore_query `{pilot.get('all_four_correct', {}).get('ignore_query_oracle_elsewhere')}/256`, ignore_bridge `{pilot.get('all_four_correct', {}).get('ignore_bridge_oracle_elsewhere')}/256`",
            "",
            "| Reference Policy | Dev 1-Hop Complete Families | Dev 3-Hop Complete Families |",
            "| --- | ---: | ---: |",
        ])
        for pname in (
            "public_traversal",
            "copied_terminal",
            "last_row",
            "rank_h_timestamp",
            "always_abstain",
            "empirical_majority",
            "query_blind",
            "bridge_blind",
        ):
            c1 = dev1.get("complete_families", {}).get(pname, 0)
            t1 = dev1.get("families", 0)
            c3 = dev3.get("complete_families", {}).get(pname, 0)
            t3 = dev3.get("families", 0)
            lines.append(f"| `{pname}` | `{c1}/{t1}` | `{c3}/{t3}` |")

    fdoc = metrics.get("frozen_diagnosis")
    if fdoc:
        rt = fdoc.get("routing_decision", {})
        lines.extend([
            "",
            "## 4. P1/P2 Frozen Y1 Diagnosis & Routing Decision",
            "",
            f"- **Restored Y1 Lineages**: `{fdoc.get('restored_lineages_count')}/{fdoc.get('expected_lineages_count')}`",
            f"- **Selected Anchor Arm**: `{rt.get('selected_anchor_arm')}`",
            f"- **First Failing Dependency**: `{rt.get('first_failing_dependency')}`",
            f"- **Next Admitted Stage**: `{rt.get('next_admitted_stage')}`",
        ])
        if fdoc.get("missing_lineages"):
            lines.append(f"- **Missing Lineages**: `{', '.join(sorted(fdoc['missing_lineages']))}` (deterministic retraining recipe emitted in `frozen_receipt.json`)")

    lines.extend([
        "",
        "## 5. Permanent Y1 Discovery Register Status (`01`–`25`)",
        "",
        "| ID | Stage Target | Discriminating Test |",
        "| --- | --- | --- |",
    ])
    for row in PERMANENT_Y1_DISCOVERY_REGISTER:
        lines.append(f"| `{row['id']}` | `{row['stage_target']}` | {row['discriminating_test']} |")

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    metrics = analyze_avendi_evidence(args.evidence)
    out = Path(args.output)
    if out.suffix.lower() == ".md":
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_avendi_results_markdown(metrics), encoding="utf-8")
        (out.parent / "METRICS.json").write_bytes(canonical(metrics) + b"\n")
    else:
        out.mkdir(parents=True, exist_ok=True)
        (out / "METRICS.json").write_bytes(canonical(metrics) + b"\n")
        (out / "RESULTS.md").write_text(render_avendi_results_markdown(metrics), encoding="utf-8")
    print(json.dumps({"status": "ANALYZED", "output": str(out), "run_id": metrics["run_id"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
