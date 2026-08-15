"""One-command, fail-closed status for the AN-RA training campaign."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch
from anra.anra_paths import OUTPUT_V2_DIR, ROOT

STATUS_REPORT = OUTPUT_V2_DIR / "campaign_status.json"
FOUNDATION_CORPUS = ROOT / "training_data" / "foundation_records.jsonl"
FOUNDATION_AUDIT = OUTPUT_V2_DIR / "foundation_records_audit.json"
FOUNDATION_PROGRESS = OUTPUT_V2_DIR / "foundation_records_audit.json.progress.json"
DOWNLOAD_PROGRESS = OUTPUT_V2_DIR / "data_manifests" / "download_progress.json"
CAMPAIGN_SLICE = OUTPUT_V2_DIR / "campaign_slice" / "campaign_slice_manifest.json"
V4_BUILD = OUTPUT_V2_DIR / "v4_tokenizer_build.json"
TOKEN_SHARDS = OUTPUT_V2_DIR / "data_manifests" / "native_foundation_v4"


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _check(
    name: str,
    status: str,
    detail: str,
    action: str = "",
    *,
    path: Path | None = None,
) -> dict[str, object]:
    return {
        "name": name,
        "status": status,
        "detail": detail,
        "action": action,
        "path": str(path) if path is not None else None,
    }


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def inspect_campaign_status(
    *,
    phase: str = "pilot",
    target_gb: float = 30.0,
    native_target_gb: float | None = None,
) -> dict[str, object]:
    phase_name = phase.lower()
    if phase_name not in {"pilot", "phase-a", "phase-b", "phase-c", "posttrain"}:
        raise ValueError(f"unknown campaign phase: {phase}")
    checks: list[dict[str, object]] = []

    cuda = torch.cuda.is_available()
    checks.append(
        _check(
            "cuda_runtime",
            "ok" if cuda else "blocker",
            (
                f"{torch.cuda.get_device_name(0)} via torch {torch.__version__}"
                if cuda
                else f"torch {torch.__version__} in {sys.executable} has no CUDA"
            ),
            "Run campaign commands with .venv-cuda\\Scripts\\python.exe."
            if not cuda
            else "",
        )
    )

    corpus_bytes = FOUNDATION_CORPUS.stat().st_size if FOUNDATION_CORPUS.is_file() else 0
    campaign_target_bytes = int(target_gb * 1024**3)
    native_target_bytes = int(
        float(native_target_gb) * 1024**3
        if native_target_gb is not None
        else campaign_target_bytes * 0.90
    )
    audit = _read_json(FOUNDATION_AUDIT)
    progress = _read_json(FOUNDATION_PROGRESS)
    download_progress = _read_json(DOWNLOAD_PROGRESS)
    download_active = (
        download_progress.get("status") == "downloading"
        and time.time() - float(download_progress.get("updated_at", 0.0)) < 1_200
    )
    audit_matches = (
        audit.get("resume_safe") is True
        and int(audit.get("corpus_size_bytes", -1)) == corpus_bytes
    )
    if audit_matches:
        audit_status = "ok"
        audit_detail = f"audited {corpus_bytes / 1024**3:.2f} GiB; resume-safe"
        audit_action = ""
    elif download_active:
        source = str(download_progress.get("source", "native source"))
        source_completion = 100.0 * float(
            download_progress.get("source_completion", 0.0)
        )
        audit_status = "waiting"
        audit_detail = (
            f"audited base is being extended by {source} "
            f"({source_completion:.2f}% source quota); final audit is pending"
        )
        audit_action = "Let the managed downloader finish and publish its incremental audit."
    elif progress and progress.get("status") == "scanning":
        completion = 100.0 * float(progress.get("completion", 0.0))
        audit_status = "waiting"
        audit_detail = (
            f"integrity audit {completion:.2f}% complete; "
            f"{int(progress.get('valid_records', 0)):,} valid records"
        )
        audit_action = "Let scripts.audit_foundation_records finish; it is resumable."
    else:
        audit_status = "blocker"
        audit_detail = "no matching resume-safe corpus audit"
        audit_action = "Run python -m scripts.audit_foundation_records."
    checks.append(
        _check(
            "foundation_audit",
            audit_status,
            audit_detail,
            audit_action,
            path=FOUNDATION_AUDIT,
        )
    )

    corpus_complete = corpus_bytes >= int(native_target_bytes * 0.98)
    checks.append(
        _check(
            "foundation_volume",
            "ok" if corpus_complete else "waiting" if download_active else "blocker",
            f"{corpus_bytes / 1024**3:.2f} / "
            f"{native_target_bytes / 1024**3:.2f} GiB native foundation acquired",
            (
                "After the audit passes, run scripts.download_training_data "
                f"--profile {'120gb' if native_target_bytes >= 120 * 1024**3 else '30gb'} "
                "--bucket base --resume."
                if not corpus_complete and not download_active
                else ""
            ),
            path=FOUNDATION_CORPUS,
        )
    )

    slice_manifest = _read_json(CAMPAIGN_SLICE)
    slice_ready = slice_manifest.get("ready_for_v4") is True
    checks.append(
        _check(
            "campaign_slice",
            "ok" if slice_ready else "blocker",
            (
                f"{float(slice_manifest.get('train_mb', 0.0)):.2f} MiB; "
                f"seven-source mix={slice_manifest.get('campaign_mix_verified', False)}"
            ),
            (
                "The managed Stream-B continuation will build the seven-source slice."
                if download_active
                else "Run scripts.build_campaign_slice after all seven source classes exist."
            )
            if not slice_ready
            else "",
            path=CAMPAIGN_SLICE,
        )
    )

    v4 = _read_json(V4_BUILD)
    v4_ready = str(v4.get("status", "")) in {"built", "passed", "ready"}
    checks.append(
        _check(
            "canonical_v4",
            "ok" if v4_ready else "blocker",
            str(v4.get("status", "missing or not eligible")),
            (
                "The managed Stream-B continuation will build and prove canonical V4."
                if download_active
                else "Build V4 from the ready campaign slice and pass held-out fertility gates."
            )
            if not v4_ready
            else "",
            path=V4_BUILD,
        )
    )

    key_set = bool(os.environ.get("ANRA_MANIFEST_SIGNING_KEY", ""))
    checks.append(
        _check(
            "manifest_signing_key",
            "ok" if key_set else "blocker",
            "configured (value not displayed)" if key_set else "not set",
            "Set ANRA_MANIFEST_SIGNING_KEY in the training shell." if not key_set else "",
        )
    )

    shard_inventories = sorted(TOKEN_SHARDS.glob("*/token_inventory.json"))
    ready_inventories: list[Path] = []
    rejected_inventories: list[str] = []
    for inventory_path in shard_inventories:
        inventory = _read_json(inventory_path)
        manifest_path = Path(str(inventory.get("manifest", "")))
        validation_path = Path(str(inventory.get("validation_manifest", "")))
        test_path = Path(str(inventory.get("test_manifest", "")))
        if not manifest_path.is_absolute():
            manifest_path = (ROOT / manifest_path).resolve()
        if not validation_path.is_absolute():
            validation_path = (ROOT / validation_path).resolve()
        if not test_path.is_absolute():
            test_path = (ROOT / test_path).resolve()
        manifest = _read_json(manifest_path)
        ready = (
            inventory.get("tokenizer_family") == "v4"
            and inventory.get("campaign_sampling_verified") is True
            and inventory.get("campaign_mix_verified") is True
            and manifest.get("campaign_sampling_verified") is True
            and manifest.get("campaign_mix_verified") is True
            and validation_path.is_file()
            and test_path.is_file()
            and int(inventory.get("licensed_tokens", 0)) > 0
        )
        if ready:
            ready_inventories.append(inventory_path)
        else:
            rejected_inventories.append(inventory_path.parent.name)
    checks.append(
        _check(
            "immutable_v4_token_shards",
            "ok" if ready_inventories else "blocker",
            (
                f"{len(ready_inventories)} ready V4 profile(s)"
                + (
                    f"; rejected={','.join(rejected_inventories)}"
                    if rejected_inventories
                    else ""
                )
            ),
            (
                "Publish V4 train/validation/test shards with a verified seven-source "
                "sampling manifest before training."
                if not ready_inventories
                else ""
            ),
            path=TOKEN_SHARDS,
        )
    )
    if phase_name in {"phase-c", "posttrain"}:
        declared = os.environ.get("ANRA_ENABLED_SUBSYSTEMS", "").strip()
        checks.append(
            _check(
                "frozen_subsystem_recipe",
                "ok" if declared else "blocker",
                declared or "no pilot-selected native subsystem set declared",
                "Bind ANRA_ENABLED_SUBSYSTEMS only to separately promoted pilot evidence.",
            )
        )
    if phase_name == "posttrain":
        gate6 = _read_json(OUTPUT_V2_DIR / "scorecards" / "gate6.json")
        passed = gate6.get("passed") is True
        checks.append(
            _check(
                "gate6_base_competence",
                "ok" if passed else "blocker",
                "passed" if passed else "competent base has not passed Gate 6",
                "Do not post-train until the base checkpoint passes Gate 6.",
            )
        )

    free_bytes = shutil.disk_usage(ROOT).free
    minimum_free = max(
        20 * 1024**3,
        max(campaign_target_bytes, native_target_bytes) - corpus_bytes + 10 * 1024**3,
    )
    checks.append(
        _check(
            "disk_headroom",
            "ok" if free_bytes >= minimum_free else "blocker",
            f"{free_bytes / 1024**3:.2f} GiB free; {minimum_free / 1024**3:.2f} GiB required",
            "Free disk space or move campaign artifacts." if free_bytes < minimum_free else "",
        )
    )

    blockers = [row for row in checks if row["status"] in {"blocker", "waiting"}]
    actions = [str(row["action"]) for row in blockers if row.get("action")]
    payload: dict[str, object] = {
        "schema_version": 1,
        "generated_at": time.time(),
        "phase": phase_name,
        "ready": not blockers,
        "checks": checks,
        "next_actions": list(dict.fromkeys(actions)),
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("pilot", "phase-a", "phase-b", "phase-c", "posttrain"),
        default="pilot",
    )
    parser.add_argument("--target-gb", type=float, default=30.0)
    parser.add_argument("--native-target-gb", type=float, default=None)
    parser.add_argument("--json-out", default=str(STATUS_REPORT))
    args = parser.parse_args()
    report = inspect_campaign_status(
        phase=args.phase,
        target_gb=args.target_gb,
        native_target_gb=args.native_target_gb,
    )
    output = Path(args.json_out)
    _atomic_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ready"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
