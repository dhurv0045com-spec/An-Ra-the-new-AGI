from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from tools import formation_mux_001_recovery_preflight as recovery


def _write_json(path: Path, body: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body), encoding="utf-8")


def _arm(
    root: Path,
    experiment: str,
    arm: str,
    seed_label: str,
    *,
    checkpoint: bool = True,
    complete: bool = True,
) -> Path:
    directory = root / experiment / arm / seed_label
    directory.mkdir(parents=True, exist_ok=True)
    seed = int(seed_label[1:])
    if complete:
        _write_json(directory / "ARM_RESULT.json", {
            "status": "COMPLETE",
            "experiment": experiment,
            "arm": arm,
            "seed_bundle": 73010 + seed,
            "engineering_only": False,
        })
    if checkpoint:
        payload = (f"{experiment}/{arm}/{seed_label}").encode()
        (directory / "resume.pt").write_bytes(payload)
        _write_json(directory / "CHECKPOINT_RECEIPT.json", {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "experiment": experiment,
            "arm": arm,
            "seed_bundle": 73010 + seed,
            "updates": 2000,
            "processed_tokens": 0,
        })
        _write_json(directory / "LATEST_PROGRESS.json", {
            "updates": 2000,
            "processed_tokens": 0,
        })
    return directory


def _valid_state(root: Path) -> None:
    _write_json(root / "CAMPAIGN_STATE.json", {
        "status": "ARMS_COMPLETE",
        "complete_arms": 24,
        "science_commit": recovery.SCIENCE_COMMIT,
    })
    _write_json(root / "PUBLIC_SURFACE_MANIFEST.json", {
        "sha256": recovery.PUBLIC_SURFACE_SHA256,
    })
    _write_json(root / "TOKENIZER_RECEIPT.json", {
        "artifact_sha256": recovery.TOKENIZER_SHA256,
    })
    mechanism_arms = (
        "M0_STANDARD", "M1_EXTRA_NO_DECAY", "M2_EXTRA_FROZEN", "M3_EXTRA_FROZEN_MASKED")
    for arm in mechanism_arms:
        for seed_label in recovery.SEED_LABELS:
            _arm(root, "CS-MECH-002", arm, seed_label)
    for arm in ("R0_CANONICAL_R0", "R1_ISOMORPHIC"):
        for seed_label in recovery.SEED_LABELS:
            _arm(root, "REP-FORM-003A", arm, seed_label)
    _arm(root, "TIE-ROLE-001", "T0_CANONICAL", "S1")
    _arm(root, "TIE-ROLE-001", "T0_CANONICAL", "S2")
    _write_json(root / "TIE_ROLE_FRONTIER_STATE.json", {
        "extension": recovery.FRONTIER_EXTENSION,
        "status": "PARTIAL_SESSION",
        "complete_arms": 2,
    })


def test_valid_partial_output_passes_and_installs(tmp_path: Path) -> None:
    source = tmp_path / "input" / "FORMATION_MUX_001"
    _valid_state(source)
    output = tmp_path / "working" / "FORMATION_MUX_001"
    receipt = recovery.run_preflight(
        input_path=tmp_path / "input", output_path=output, install=True)
    assert receipt["status"] == "PASS"
    assert receipt["s5_completed_arms"] == 24
    assert receipt["frontier_completed_arms"] == 2
    assert receipt["official_checkpoint_count"] == 26
    assert (output / "RECOVERY_PREFLIGHT.json").exists()


def test_completed_arm_without_checkpoint_fails(tmp_path: Path) -> None:
    source = tmp_path / "input" / "FORMATION_MUX_001"
    _valid_state(source)
    target = source / "CS-MECH-002" / "M0_STANDARD" / "S1"
    (target / "resume.pt").unlink()
    with pytest.raises(recovery.RecoveryError, match="completed arm has no resume.pt"):
        recovery.run_preflight(
            input_path=tmp_path / "input",
            output_path=tmp_path / "working" / "FORMATION_MUX_001",
            install=True)


def test_checkpoint_hash_mismatch_fails(tmp_path: Path) -> None:
    source = tmp_path / "input" / "FORMATION_MUX_001"
    _valid_state(source)
    target = source / "CS-MECH-002" / "M0_STANDARD" / "S1"
    (target / "resume.pt").write_bytes(b"corrupt")
    with pytest.raises(recovery.RecoveryError, match="checkpoint hash mismatch"):
        recovery.run_preflight(
            input_path=tmp_path / "input",
            output_path=tmp_path / "working" / "FORMATION_MUX_001",
            install=True)


def test_evidence_only_zip_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "input" / "FORMATION_MUX_001"
    _valid_state(source)
    archive = tmp_path / "FORMATION_MUX_001_RESULTS.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for path in source.rglob("*"):
            if path.is_file() and path.name != "resume.pt":
                member = (Path("FORMATION_MUX_001") / path.relative_to(source)).as_posix()
                handle.write(path, member)
    with pytest.raises(recovery.RecoveryError, match="EVIDENCE_ONLY_NOT_RESUMABLE"):
        recovery.run_preflight(
            input_path=archive,
            output_path=tmp_path / "working" / "FORMATION_MUX_001",
            install=True)


def test_ambiguous_saved_outputs_fail_closed(tmp_path: Path) -> None:
    first = tmp_path / "one" / "FORMATION_MUX_001"
    second = tmp_path / "two" / "FORMATION_MUX_001"
    _valid_state(first)
    _valid_state(second)
    with pytest.raises(recovery.RecoveryError, match="ambiguous recovery input"):
        recovery.run_preflight(
            input_path=tmp_path,
            output_path=tmp_path / "working" / "FORMATION_MUX_001",
            install=True)


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "working" / "FORMATION_MUX_001"
    _valid_state(output)
    marker = output / "USER_MARKER.txt"
    marker.write_text("preserve", encoding="utf-8")
    receipt = recovery.run_preflight(
        input_path=tmp_path / "missing", output_path=output, install=False)
    assert receipt["status"] == "PASS"
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_recovery_notebook_is_pinned_and_gated() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (root / "notebooks" / "CYMEK_FORMATION_MUX_001_RECOVERY_T4X2.ipynb").read_text(
            encoding="utf-8"))
    metadata = notebook["metadata"]
    assert metadata["schema"] == "anra.formation-mux-kaggle-recovery-wrapper/v1"
    assert metadata["recovery_commit"] == "7a82d80e6f44756c9e37f3b554500f4f5c664238"
    assert metadata["operator_commit"] == recovery.OPERATOR_COMMIT
    assert metadata["operator_blob"] == recovery.OPERATOR_BLOB
    source = "\n".join(
        line
        for cell in notebook["cells"]
        for line in cell.get("source", []))
    preflight = source.index("tools.formation_mux_001_recovery_preflight")
    operator = source.index("tools/formation_mux_001_kaggle_operator_v12.py")
    assert preflight < operator
    assert "recovery.get('status') != 'PASS'" in source
    assert "RECOVERY_PREFLIGHT.json" in source
