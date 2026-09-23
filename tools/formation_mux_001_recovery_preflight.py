"""Fail-closed recovery preflight for the 2026-09-23 FORMATION-MUX run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

SCIENCE_COMMIT = "c15ad8beb409537db42d075684ea54847a074ebd"
OPERATOR_COMMIT = "4ee05f6e386f15d34f9dfa7bd7f3300a496b9896"
OPERATOR_BLOB = "e9e1f701b0d4edc509194da55fe1ba37ed62ef86"
PUBLIC_SURFACE_SHA256 = "f1d5200bd05bc28ede97af114b49f616ca72b24af74b7fc530a2cb084db4259c"
TOKENIZER_SHA256 = "97e12db63b343312e5e4abc37df9ef4b01fcb1faba792a6420a4c1b15d0a7fbc"
FRONTIER_EXTENSION = "TIE-ROLE-FRONTIER-001"
S5_EXPERIMENTS = {"CS-MECH-002": 4, "REP-FORM-003A": 2}
FRONTIER_EXPERIMENTS = {"TIE-ROLE-001": 4, "TIE-ROLE-XFER-001": 2}
SEED_LABELS = tuple(f"S{index}" for index in range(1, 5))
REQUIRED_FRONTIER_COMPLETE = {
    ("TIE-ROLE-001", "T0_CANONICAL", "S1"),
    ("TIE-ROLE-001", "T0_CANONICAL", "S2"),
}


class RecoveryError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RecoveryError(f"required recovery metadata missing: {path}") from exc
    except (OSError, ValueError) as exc:
        raise RecoveryError(f"invalid recovery metadata: {path}: {exc}") from exc
    if not isinstance(body, dict):
        raise RecoveryError(f"recovery metadata is not an object: {path}")
    return body


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, body: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(body, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _safe_member_name(name: str) -> PurePosixPath:
    if "\\" in name:
        raise RecoveryError(f"ZIP member uses a Windows path separator: {name}")
    member = PurePosixPath(name)
    if member.is_absolute() or ".." in member.parts or not member.parts:
        raise RecoveryError(f"unsafe ZIP member path: {name}")
    return member


def _archive_prefix(archive: Path) -> tuple[str, int]:
    try:
        handle = zipfile.ZipFile(archive)
    except (OSError, zipfile.BadZipFile) as exc:
        raise RecoveryError(f"invalid recovery ZIP: {archive}: {exc}") from exc
    with handle:
        names = []
        for info in handle.infolist():
            _safe_member_name(info.filename)
            names.append(info.filename)
        if len(names) != len(set(names)):
            raise RecoveryError("recovery ZIP contains duplicate member names")
        suffix = "FORMATION_MUX_001/CAMPAIGN_STATE.json"
        states = [name for name in names if name == suffix or name.endswith("/" + suffix)]
        if len(states) != 1:
            raise RecoveryError(
                f"recovery ZIP must contain exactly one {suffix}; found {len(states)}")
        prefix = states[0][:-len("CAMPAIGN_STATE.json")]
        checkpoints = [
            name for name in names
            if name.startswith(prefix) and PurePosixPath(name).name == "resume.pt"
        ]
        if not checkpoints:
            raise RecoveryError(
                "EVIDENCE_ONLY_NOT_RESUMABLE: the ZIP has campaign JSON but no resume.pt "
                "checkpoint payloads; download the complete saved Kaggle Output tree")
        return prefix, len(checkpoints)


def _extract_archive(archive: Path, destination: Path) -> int:
    prefix, checkpoint_count = _archive_prefix(archive)
    if destination.exists():
        raise RecoveryError(f"refusing to overwrite recovery staging path: {destination}")
    destination.mkdir(parents=True)
    try:
        with zipfile.ZipFile(archive) as handle:
            for info in handle.infolist():
                name = _safe_member_name(info.filename)
                if not info.filename.startswith(prefix):
                    continue
                relative = str(info.filename)[len(prefix):]
                if not relative:
                    continue
                target = destination.joinpath(*_safe_member_name(relative).parts)
                mode = (info.external_attr >> 16) & 0xFFFF
                if stat.S_ISLNK(mode):
                    raise RecoveryError(f"recovery ZIP contains a symlink: {info.filename}")
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with handle.open(info) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                    output.flush()
                    os.fsync(output.fileno())
    except Exception as exc:
        raise RecoveryError(
            f"recovery archive extraction failed; preserved staging path for inspection: "
            f"{destination}: {exc}") from exc
    return checkpoint_count


def _locate(input_path: Path, output_path: Path) -> dict[str, Any]:
    if (output_path / "CAMPAIGN_STATE.json").exists():
        return {"kind": "working", "source": output_path}
    if input_path.is_file():
        if not zipfile.is_zipfile(input_path):
            raise RecoveryError(f"recovery input is not a ZIP archive: {input_path}")
        prefix, checkpoint_count = _archive_prefix(input_path)
        return {
            "kind": "archive",
            "source": input_path,
            "prefix": prefix,
            "archive_checkpoint_count": checkpoint_count,
        }
    if not input_path.exists():
        raise RecoveryError(f"recovery input path does not exist: {input_path}")
    direct = input_path / "FORMATION_MUX_001"
    if (direct / "CAMPAIGN_STATE.json").exists():
        return {"kind": "directory", "source": direct}
    candidates = sorted(input_path.rglob("FORMATION_MUX_001/CAMPAIGN_STATE.json"))
    if len(candidates) > 1:
        names = ", ".join(str(path) for path in candidates)
        raise RecoveryError(f"ambiguous recovery input; preserve each run separately: {names}")
    if len(candidates) == 1:
        return {"kind": "directory", "source": candidates[0].parent}
    evidence = sorted(
        path for path in input_path.rglob("*.zip")
        if "FORMATION_MUX_001_RESULTS" in path.name.upper()
    )
    if evidence:
        names = ", ".join(str(path) for path in evidence)
        raise RecoveryError(
            "EVIDENCE_ONLY_NOT_RESUMABLE: only result ZIP(s) were found, not a saved "
            f"Output tree with resume.pt files: {names}")
    raise RecoveryError(
        f"no FORMATION_MUX_001/CAMPAIGN_STATE.json found under recovery input: {input_path}")


def _official_directories(root: Path) -> Iterator[tuple[str, str, str, Path]]:
    for experiment, arm_count in {**S5_EXPERIMENTS, **FRONTIER_EXPERIMENTS}.items():
        experiment_root = root / experiment
        if not experiment_root.exists():
            continue
        for arm_index in range(1, arm_count + 1):
            arm = experiment_root
            arms = sorted(path for path in experiment_root.iterdir() if path.is_dir())
            if arm_index > len(arms):
                continue
            arm = arms[arm_index - 1]
            for seed_label in SEED_LABELS:
                directory = arm / seed_label
                if directory.is_dir():
                    yield experiment, arm.name, seed_label, directory


def _validate_sealed_markers(root: Path) -> list[dict[str, Any]]:
    receipts = []
    for experiment in (*S5_EXPERIMENTS, *FRONTIER_EXPERIMENTS):
        names = ["SEALED_MARKER.json"]
        if experiment in FRONTIER_EXPERIMENTS:
            names.append("TIE_ROLE_SEALED_MARKER.json")
        for name in names:
            marker_path = root / experiment / name
            if not marker_path.exists():
                continue
            marker = _read_json(marker_path)
            state = marker.get("state")
            final_path = root / experiment / "FINAL_RESULT.json"
            if state == "STARTED" and not final_path.exists():
                raise RecoveryError(
                    f"sealed custody is interrupted before final result: {marker_path}")
            if state == "COMPLETE" and not final_path.exists():
                raise RecoveryError(
                    f"sealed marker is COMPLETE but final result is missing: {marker_path}")
            if final_path.exists() and state != "COMPLETE":
                raise RecoveryError(
                    f"sealed final result exists without COMPLETE marker: {final_path}")
            receipts.append({
                "experiment": experiment,
                "marker": str(marker_path.relative_to(root)),
                "state": state,
                "final_result_present": final_path.exists(),
            })
    return receipts


def _validate_checkpoint(directory: Path, result: dict[str, Any] | None) -> dict[str, Any]:
    checkpoint = directory / "resume.pt"
    receipt_path = directory / "CHECKPOINT_RECEIPT.json"
    progress_path = directory / "LATEST_PROGRESS.json"
    if not checkpoint.exists():
        if progress_path.exists():
            progress = _read_json(progress_path)
            if int(progress.get("updates", 0) or 0) > 0 or int(
                    progress.get("processed_tokens", 0) or 0) > 0:
                raise RecoveryError(
                    "checkpoint missing for an arm with nonzero progress: "
                    f"{directory}")
        if result is not None and result.get("status") == "COMPLETE":
            raise RecoveryError(f"completed arm has no resume.pt checkpoint: {directory}")
        return {"directory": str(directory), "present": False}
    if not receipt_path.exists():
        raise RecoveryError(f"resume.pt has no CHECKPOINT_RECEIPT.json: {directory}")
    receipt = _read_json(receipt_path)
    actual = _sha256_file(checkpoint)
    if receipt.get("sha256") != actual:
        raise RecoveryError(
            f"checkpoint hash mismatch: {checkpoint}: receipt={receipt.get('sha256')} "
            f"actual={actual}")
    if result is not None:
        for key in ("experiment", "arm", "seed_bundle"):
            if key in receipt and receipt[key] != result.get(key):
                raise RecoveryError(
                    f"checkpoint receipt identity mismatch for {key}: {directory}")
        if result.get("engineering_only") not in (None, False):
            raise RecoveryError(f"official checkpoint is marked engineering-only: {directory}")
    return {
        "directory": str(directory),
        "present": True,
        "sha256": actual,
        "byte_size": checkpoint.stat().st_size,
        "updates": receipt.get("updates"),
        "processed_tokens": receipt.get("processed_tokens"),
    }


def validate_source(source: Path, *, located: dict[str, Any]) -> dict[str, Any]:
    campaign = _read_json(source / "CAMPAIGN_STATE.json")
    if campaign.get("science_commit") != SCIENCE_COMMIT:
        raise RecoveryError(
            f"science identity mismatch: {campaign.get('science_commit')} != {SCIENCE_COMMIT}")
    if campaign.get("status") != "ARMS_COMPLETE" or int(campaign.get("complete_arms", -1)) != 24:
        raise RecoveryError(
            "S5 is not 24/24 ARMS_COMPLETE in the recovered Output tree")
    public = _read_json(source / "PUBLIC_SURFACE_MANIFEST.json")
    if public.get("sha256") != PUBLIC_SURFACE_SHA256:
        raise RecoveryError(
            f"public surface identity mismatch: {public.get('sha256')} != "
            f"{PUBLIC_SURFACE_SHA256}")
    tokenizer = _read_json(source / "TOKENIZER_RECEIPT.json")
    if tokenizer.get("artifact_sha256") != TOKENIZER_SHA256:
        raise RecoveryError(
            f"tokenizer identity mismatch: {tokenizer.get('artifact_sha256')} != "
            f"{TOKENIZER_SHA256}")
    if (source / "SURFACE_MANIFEST.json").exists():
        raise RecoveryError("legacy raw sealed-containing SURFACE_MANIFEST.json detected")
    frontier_state = _read_json(source / "TIE_ROLE_FRONTIER_STATE.json")
    if frontier_state.get("extension") != FRONTIER_EXTENSION:
        raise RecoveryError(
            f"frontier extension mismatch: {frontier_state.get('extension')}")

    completed: dict[str, list[str]] = {
        experiment: [] for experiment in (*S5_EXPERIMENTS, *FRONTIER_EXPERIMENTS)
    }
    checkpoints = []
    for experiment, arm, seed_label, directory in _official_directories(source):
        result_path = directory / "ARM_RESULT.json"
        result = _read_json(result_path) if result_path.exists() else None
        if result is not None and result.get("status") == "COMPLETE":
            if result.get("experiment") != experiment or result.get("arm") != arm:
                raise RecoveryError(f"arm result identity/path mismatch: {result_path}")
            completed[experiment].append(f"{arm}/{seed_label}")
        checkpoint = _validate_checkpoint(directory, result)
        if checkpoint["present"]:
            checkpoints.append({"experiment": experiment, **checkpoint})

    for experiment, expected_arms in S5_EXPERIMENTS.items():
        if len(completed[experiment]) != expected_arms * len(SEED_LABELS):
            raise RecoveryError(
                f"{experiment} recovered completed arms "
                f"{len(completed[experiment])} != {expected_arms * len(SEED_LABELS)}")
    frontier_complete = {
        (experiment, item.split("/", 1)[0], item.split("/", 1)[1])
        for experiment in FRONTIER_EXPERIMENTS
        for item in completed[experiment]
    }
    missing_frontier = REQUIRED_FRONTIER_COMPLETE - frontier_complete
    if missing_frontier:
        raise RecoveryError(
            "required completed frontier controls missing: "
            + ", ".join("/".join(item) for item in sorted(missing_frontier)))
    if len(checkpoints) < 26:
        raise RecoveryError(
            f"recovered official checkpoint count {len(checkpoints)} is below 26")
    recorded_frontier = int(frontier_state.get("complete_arms", -1))
    observed_frontier = sum(len(value) for value in completed.values()) - 24
    if recorded_frontier != observed_frontier:
        raise RecoveryError(
            f"frontier state count mismatch: recorded={recorded_frontier} "
            f"observed={observed_frontier}")
    sealed = _validate_sealed_markers(source)
    return {
        "schema": "anra.formation-mux-recovery-preflight/v1",
        "status": "PASS",
        "recovery_only": True,
        "science_commit": SCIENCE_COMMIT,
        "operator_commit": OPERATOR_COMMIT,
        "operator_blob": OPERATOR_BLOB,
        "operator_identity_enforcement": (
            "pinned recovery launcher; recovered ENVIRONMENT.json is not trusted because "
            "the partial frontier path historically reported an older inherited operator"),
        "source_kind": located["kind"],
        "archive_checkpoint_count": located.get("archive_checkpoint_count"),
        "s5_status": campaign.get("status"),
        "s5_completed_arms": 24,
        "frontier_status": frontier_state.get("status"),
        "frontier_completed_arms": observed_frontier,
        "frontier_required_arms": 24,
        "official_checkpoint_count": len(checkpoints),
        "completed_arms": completed,
        "checkpoints": checkpoints,
        "sealed_markers": sealed,
        "raw_sealed_rows_read": False,
        "claim_ceiling": "custody preflight only; no scientific verdict",
    }


def install_from_archive(archive: Path, output_path: Path) -> tuple[Path, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise RecoveryError(f"refusing to overwrite existing working output: {output_path}")
    staging = output_path.with_name(output_path.name + ".recovery.tmp")
    if staging.exists():
        raise RecoveryError(f"stale recovery staging path exists: {staging}")
    checkpoint_count = _extract_archive(archive, staging)
    return staging, checkpoint_count


def install_from_directory(source: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise RecoveryError(f"refusing to overwrite existing working output: {output_path}")
    staging = output_path.with_name(output_path.name + ".recovery.tmp")
    if staging.exists():
        raise RecoveryError(f"stale recovery staging path exists: {staging}")
    shutil.copytree(source, staging)
    return staging


def run_preflight(
    *, input_path: Path, output_path: Path, install: bool
) -> dict[str, Any]:
    located = _locate(input_path.resolve(), output_path.resolve())
    if located["kind"] == "working":
        receipt = validate_source(located["source"], located=located)
        _atomic_json(output_path / "RECOVERY_PREFLIGHT.json", receipt)
        return receipt
    if located["kind"] == "archive":
        if not install:
            raise RecoveryError(
                "archive recovery requires --install so the validated checkpoint tree "
                "is materialized into the working directory")
        staging, archive_checkpoint_count = install_from_archive(
            located["source"], output_path)
        validate_source(
            staging, located={"kind": "staged_archive", "source": staging})
        located["archive_checkpoint_count"] = archive_checkpoint_count
        os.replace(staging, output_path)
        located = {"kind": "installed_archive", "source": output_path}
        receipt = validate_source(output_path, located=located)
        _atomic_json(output_path / "RECOVERY_PREFLIGHT.json", receipt)
        return receipt
    if not install:
        raise RecoveryError(
            "directory recovery requires --install before the pinned operator is allowed "
            "to copy or mutate it")
    staging = install_from_directory(located["source"], output_path)
    validate_source(staging, located={"kind": "staged_directory", "source": staging})
    os.replace(staging, output_path)
    located = {"kind": "installed_directory", "source": output_path}
    receipt = validate_source(output_path, located=located)
    _atomic_json(output_path / "RECOVERY_PREFLIGHT.json", receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("/kaggle/input"))
    parser.add_argument(
        "--out", type=Path, default=Path("/kaggle/working/FORMATION_MUX_001"))
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args(argv)
    try:
        receipt = run_preflight(
            input_path=args.input, output_path=args.out, install=args.install)
    except RecoveryError as exc:
        failure = {
            "schema": "anra.formation-mux-recovery-preflight/v1",
            "status": "FAIL",
            "error": str(exc),
            "science_commit": SCIENCE_COMMIT,
            "operator_commit": OPERATOR_COMMIT,
            "operator_blob": OPERATOR_BLOB,
        }
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
