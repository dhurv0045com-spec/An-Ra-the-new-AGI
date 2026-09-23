from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from tools import formation_mux_001_recovery_preflight as recovery


def _write_json(path: Path, body: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body), encoding="utf-8")


def _read_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=True)


def _fake_public_validator(manifest: dict[str, Any]) -> None:
    if manifest.get("schema") != "anra.formation-mux-public-surface/v5":
        raise recovery.RecoveryError("public surface schema mismatch")
    if "sealed" in manifest.get("splits", {}):
        raise recovery.RecoveryError("SEALED_FIREWALL_BREACH")
    if set(manifest.get("splits", {})) != {"training", "development"}:
        raise recovery.RecoveryError("public split set mismatch")
    if manifest.get("sealed_rows_persisted") is not False:
        raise recovery.RecoveryError("sealed persistence mismatch")


def _preserved_public_manifest() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    archive = root / (
        "artifacts/cymek/FORMATION-MUX-001/kaggle-session-20260923/"
        "FORMATION_MUX_001_RESULTS.partial.zip"
    )
    with zipfile.ZipFile(archive) as handle:
        members = [
            name
            for name in handle.namelist()
            if name.endswith("FORMATION_MUX_001/PUBLIC_SURFACE_MANIFEST.json")
        ]
        assert len(members) == 1
        return json.loads(handle.read(members[0]).decode("utf-8"))


@pytest.fixture
def fake_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORMATION_MUX_RECOVERY_NONCE", raising=False)
    monkeypatch.setattr(recovery, "_validate_public_manifest", _fake_public_validator)


def _complete_arm(root: Path, slot: recovery.OfficialSlot) -> None:
    directory = root.joinpath(*slot.relative_directory.split("/"))
    directory.mkdir(parents=True, exist_ok=True)
    updates = slot.exposure_target if slot.exposure_axis == "updates" else 800
    processed_tokens = (
        64_000
        if slot.exposure_axis == "updates"
        else slot.exposure_target + 26
    )
    final_axis = updates if slot.exposure_axis == "updates" else processed_tokens
    trace = [
        {
            "axis": slot.eligible_from,
            "identity_exact_valid_eos": 0.0,
            "per_family": {"fixture": 0.0},
        },
        {
            "axis": final_axis,
            "identity_exact_valid_eos": 0.5,
            "per_family": {"fixture": 0.5},
        },
    ]
    diagnostics: list[dict[str, Any]] = []
    timing = {
        "train_seconds": 1.0,
        "eval_seconds": 0.1,
        "checkpoint_seconds": 0.01,
    }
    state = {
        "updates": updates,
        "processed_tokens": processed_tokens,
        "supervised_tokens": processed_tokens - 1,
        "stream_cursor": 123,
        "trace": trace,
        "diagnostics": diagnostics,
        "timing": timing,
        "last_eval_axis": final_axis,
        "last_checkpoint_axis": max(final_axis - 1, 0),
        "clip_events": 0,
    }
    identity = {
        "experiment": slot.experiment,
        "arm": slot.arm,
        "seed_bundle": slot.seed_bundle,
        "protocol_sha256": slot.protocol_sha256,
        "data_manifest_sha256": recovery.PUBLIC_SURFACE_SHA256,
        "engineering_only": False,
    }
    initial_sha = "a" * 64
    row_optimizer = None
    if slot.experiment == recovery.s5_protocol.EXPERIMENT_A:
        row_optimizer = {
            "step_count": 1,
            "lr": 0.001,
            "exp_avg": torch.zeros((1, 1)),
            "exp_avg_sq": torch.zeros((1, 1)),
        }
    body = {
        "model": {
            "embedding.weight": torch.ones((1, 1)),
            "norm.weight": torch.ones((1,)),
        },
        "main_optimizer": {"state": {}, "param_groups": []},
        "row_optimizer": row_optimizer,
        "cpu_rng_state": torch.tensor([1], dtype=torch.uint8),
        "cuda_rng_state": [torch.tensor([2], dtype=torch.uint8)],
        "initial_model_sha256": initial_sha,
        **identity,
        **state,
    }
    checkpoint = directory / "resume.pt"
    torch.save(body, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    receipt = {
        "sha256": digest,
        "initial_model_sha256": initial_sha,
        **identity,
        **{key: body[key] for key in recovery.CHECKPOINT_STATE_FIELDS},
    }
    _write_json(directory / "CHECKPOINT_RECEIPT.json", receipt)
    formation = recovery._formation_summary(trace, slot.eligible_from)
    progress_identity = {
        key: value for key, value in identity.items() if key != "engineering_only"
    }
    progress = {
        "schema": slot.progress_schema,
        **progress_identity,
        "updates": updates,
        "processed_tokens": processed_tokens,
        "supervised_tokens": body["supervised_tokens"],
        "latest_development": trace[-1],
        "formation_so_far": formation,
        "clip_events": 0,
        "timing": timing,
        "checkpoint_sha256": digest,
        "checkpoint_file": "resume.pt",
    }
    if slot.experiment in recovery.FRONTIER_ARMS:
        progress["extension"] = recovery.FRONTIER_EXTENSION
        progress["science_s5_unchanged"] = True
    else:
        progress["diagnostic_only"] = True
        progress["may_not_change_frozen_science"] = True
    _write_json(directory / "LATEST_PROGRESS.json", progress)
    result = {
        "schema": "anra.formation-mux-arm-result/v2",
        **identity,
        "status": "COMPLETE",
        "updates": updates,
        "processed_tokens": processed_tokens,
        "supervised_tokens": body["supervised_tokens"],
        "stream_cursor": body["stream_cursor"],
        "trace": trace,
        "diagnostics": diagnostics,
        "timing": timing,
        "formation": formation,
        "clip_fraction": 0.0,
        "wall_seconds": 1.2,
        "exposure_target": slot.exposure_target,
        "exposure_axis": slot.exposure_axis,
    }
    _write_json(directory / "ARM_RESULT.json", result)


def _valid_state(root: Path) -> None:
    s5_complete = set()
    frontier_complete = set()
    for slot in recovery.S5_SLOTS:
        _complete_arm(root, slot)
        s5_complete.add(slot.key)
    required_frontier_keys = {
        f"{recovery.frontier_protocol.EXPERIMENT_A}/T0_CANONICAL/S1",
        f"{recovery.frontier_protocol.EXPERIMENT_A}/T0_CANONICAL/S2",
    }
    frontier_complete = {
        slot.key for slot in recovery.FRONTIER_SLOTS if slot.key in required_frontier_keys
    }
    assert frontier_complete == required_frontier_keys
    for slot in recovery.FRONTIER_SLOTS:
        if slot.key in frontier_complete:
            _complete_arm(root, slot)
    _write_json(root / "CAMPAIGN_STATE.json", {
        "schema": "anra.formation-mux-state/v4",
        "status": "ARMS_COMPLETE",
        "science_commit": recovery.SCIENCE_COMMIT,
        "complete_arms": 24,
        "required_arms": 24,
        "arms": {key: "COMPLETE" for key in sorted(s5_complete)},
        "completed_seed_bundles": list(recovery.s5_protocol.SEED_BUNDLES),
        "pending_seed_bundles": [],
        "global_failure": None,
        "sealed": {experiment: "NOT_CONSUMED" for experiment in recovery.S5_ARMS},
    })
    _write_json(root / "TIE_ROLE_FRONTIER_STATE.json", {
        "schema": "anra.tie-role-frontier-state/v1",
        "extension": recovery.FRONTIER_EXTENSION,
        "status": "PARTIAL_SESSION",
        "arms": {key: "COMPLETE" for key in sorted(frontier_complete)},
        "complete_arms": len(frontier_complete),
        "required_arms": 24,
        "completed_seed_bundles": [],
        "pending_seed_bundles": list(recovery.frontier_protocol.SEED_BUNDLES),
        "global_failure": None,
        "sealed": {experiment: "NOT_CONSUMED" for experiment in recovery.FRONTIER_ARMS},
    })
    _write_json(root / "PUBLIC_SURFACE_MANIFEST.json", {
        "schema": "anra.formation-mux-public-surface/v5",
        "sha256": recovery.PUBLIC_SURFACE_SHA256,
        "tokenizer_artifact_sha256": recovery.TOKENIZER_SHA256,
        "sealed_rows_persisted": False,
        "splits": {"training": [], "development": []},
        "sealed_commitments": {experiment: "b" * 64 for experiment in recovery.S5_ARMS},
    })
    _write_json(root / "TOKENIZER_RECEIPT.json", {
        "artifact_sha256": recovery.TOKENIZER_SHA256,
        "vocabulary_size": 24_576,
        "training_rows": 60_000,
        "raw_sealed_rows_persisted": False,
        "sealed_commitments": {experiment: "b" * 64 for experiment in recovery.S5_ARMS},
    })


def test_preserved_evidence_archive_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    archive = root / (
        "artifacts/cymek/FORMATION-MUX-001/kaggle-session-20260923/"
        "FORMATION_MUX_001_RESULTS.partial.zip"
    )
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert digest == "3e3ad68cd7f80bd242733b61153d4bb8f3fedbb1e4fba8fbc0f3ebc5d904423f"
    with zipfile.ZipFile(archive) as handle:
        names = handle.namelist()
        assert len(names) == 755
        assert len(set(names)) == 755
        assert not any(Path(name).name == "resume.pt" for name in names)
        assert handle.testzip() is None


def test_preserved_actual_public_surface_passes_pinned_validator() -> None:
    manifest = _preserved_public_manifest()
    recovery._validate_public_manifest(manifest)
    assert manifest["sha256"] == recovery.PUBLIC_SURFACE_SHA256
    assert set(manifest["splits"]) == {"training", "development"}


def test_token_budget_endpoint_accepts_legal_overshoot() -> None:
    slot = next(
        slot
        for slot in recovery.S5_SLOTS
        if slot.experiment == recovery.s5_protocol.EXPERIMENT_B
    )
    assert not slot.endpoint_reached(800, 499_999)
    assert slot.endpoint_reached(800, 500_000)
    assert slot.endpoint_reached(800, 500_001)
    assert slot.endpoint_reached(800, 500_999)


def test_full_preflight_accepts_preserved_real_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FORMATION_MUX_RECOVERY_NONCE", raising=False)
    source = tmp_path / "input" / "FORMATION_MUX_001"
    _valid_state(source)
    public = _preserved_public_manifest()
    _write_json(source / "PUBLIC_SURFACE_MANIFEST.json", public)
    tokenizer_path = source / "TOKENIZER_RECEIPT.json"
    tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    tokenizer["sealed_commitments"] = public["sealed_commitments"]
    _write_json(tokenizer_path, tokenizer)
    receipt = recovery.run_preflight(
        input_path=tmp_path / "input",
        output_path=tmp_path / "working" / "FORMATION_MUX_001",
        install=True,
    )
    assert receipt["public_surface_validation"].startswith("pinned")
    assert receipt["official_checkpoint_count"] == 26


def test_valid_partial_output_passes_and_installs(
    tmp_path: Path,
    fake_runtime: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "input" / "FORMATION_MUX_001"
    _valid_state(source)
    nonce = "c" * 64
    monkeypatch.setenv("FORMATION_MUX_RECOVERY_NONCE", nonce)
    output = tmp_path / "working" / "FORMATION_MUX_001"
    receipt = recovery.run_preflight(
        input_path=tmp_path / "input",
        output_path=output,
        install=True,
    )
    assert receipt["schema"] == recovery.SCHEMA
    assert receipt["recovery_nonce"] == nonce
    assert receipt["s5_completed_arms"] == 24
    assert receipt["frontier_completed_arms"] == 2
    assert receipt["official_checkpoint_count"] == 26
    rep_rows = [
        row
        for row in receipt["checkpoints"]
        if row["slot"].startswith("REP-FORM-003A/")
    ]
    assert len(rep_rows) == 8
    assert all(row["processed_tokens"] == 500_026 for row in rep_rows)
    assert all(
        row["result_sha256"] is not None and len(row["result_sha256"]) == 64
        for row in receipt["checkpoints"]
        if row["result_complete"]
    )
    assert receipt["raw_sealed_rows_read"] is False
    result = json.loads(
        (output / "REP-FORM-003A" / "R0_PRODUCTION_BPE" / "S1" / "ARM_RESULT.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["formation"]["points"] == 2
    assert result["formation"]["formation_auc"] == 0.25
    assert result["formation"]["first_acquisition_axis"] == 500_026
    assert (output / "RECOVERY_PREFLIGHT.json").exists()


def test_producer_saved_checkpoint_passes_full_preflight(
    tmp_path: Path,
    fake_runtime: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from anra_v5 import formation_mux_train_v2 as base

    original_base_save = base._save_checkpoint
    original_base_model = base.fxm
    original_base_protocol = base.proto
    from anra_v5 import formation_mux_train_v5 as train

    original_train_save = train._ORIGINAL_SAVE
    source = tmp_path / "input" / "FORMATION_MUX_001"
    _valid_state(source)
    slot = next(
        slot
        for slot in recovery.S5_SLOTS
        if slot.experiment == recovery.s5_protocol.EXPERIMENT_B
        and slot.arm == recovery.s5_protocol.ARMS_B[0]
        and slot.seed_label == "S1"
    )
    directory = source.joinpath(*slot.relative_directory.split("/"))
    old_body = _read_checkpoint(directory / "resume.pt")
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    model(torch.ones((1, 1))).sum().backward()
    optimizer.step()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_rng_state_all",
        lambda: [torch.tensor([2], dtype=torch.uint8)],
    )
    payload = {
        "experiment": slot.experiment,
        "arm": slot.arm,
        "seed_bundle": slot.seed_bundle,
        "data_manifest_sha256": recovery.PUBLIC_SURFACE_SHA256,
        "protocol_sha256": slot.protocol_sha256,
        "engineering_only": False,
        "initial_model_sha256": "b" * 64,
        **{
            key: old_body[key]
            for key in (
                "updates",
                "processed_tokens",
                "supervised_tokens",
                "stream_cursor",
                "trace",
                "diagnostics",
                "timing",
                "last_eval_axis",
                "last_checkpoint_axis",
                "clip_events",
            )
        },
    }
    base._save_checkpoint = original_base_save
    train._ORIGINAL_SAVE = original_base_save
    try:
        train._save_checkpoint_with_progress(
            directory / "resume.pt",
            model=model,
            optimizers={"main": optimizer, "rows": None},
            torch=torch,
            payload=payload,
        )
    finally:
        base._save_checkpoint = original_base_save
        base.fxm = original_base_model
        base.proto = original_base_protocol
        train._ORIGINAL_SAVE = original_train_save
    receipt = recovery.run_preflight(
        input_path=tmp_path / "input",
        output_path=tmp_path / "working" / "FORMATION_MUX_001",
        install=True,
    )
    assert receipt["official_checkpoint_count"] == 26


def test_partial_frontier_checkpoint_is_pending_not_complete(
    tmp_path: Path,
    fake_runtime: None,
) -> None:
    source = tmp_path / "input" / "FORMATION_MUX_001"
    _valid_state(source)
    slot = next(
        slot
        for slot in recovery.FRONTIER_SLOTS
        if (slot.experiment, slot.arm, slot.seed_label)
        == (recovery.frontier_protocol.EXPERIMENT_A, "T1_INPUT_X4", "S1")
    )
    _complete_arm(source, slot)
    directory = source.joinpath(*slot.relative_directory.split("/"))
    checkpoint = directory / "resume.pt"
    body = _read_checkpoint(checkpoint)
    body["updates"] = 200
    body["processed_tokens"] = 6_400
    body["supervised_tokens"] = 6_399
    body["stream_cursor"] = 321
    body["trace"] = [{
        "axis": 200,
        "identity_exact_valid_eos": 0.0,
        "per_family": {"fixture": 0.0},
    }]
    body["last_eval_axis"] = 200
    body["last_checkpoint_axis"] = 200
    torch.save(body, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    receipt_path = directory / "CHECKPOINT_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["sha256"] = digest
    for key in recovery.CHECKPOINT_STATE_FIELDS:
        receipt[key] = body[key]
    _write_json(receipt_path, receipt)
    progress_path = directory / "LATEST_PROGRESS.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["updates"] = 200
    progress["processed_tokens"] = 6_400
    progress["supervised_tokens"] = 6_399
    progress["latest_development"] = body["trace"][-1]
    progress["formation_so_far"] = recovery._formation_summary(body["trace"], slot.eligible_from)
    progress["checkpoint_sha256"] = digest
    _write_json(progress_path, progress)
    (directory / "ARM_RESULT.json").unlink()
    receipt = recovery.run_preflight(
        input_path=tmp_path / "input",
        output_path=tmp_path / "working" / "FORMATION_MUX_001",
        install=True,
    )
    assert receipt["frontier_completed_arms"] == 2
    assert receipt["official_checkpoint_count"] == 27
    assert f"{recovery.frontier_protocol.EXPERIMENT_A}/T1_INPUT_X4/S1" in receipt[
        "pending_official_slots"
    ]


def test_completed_arm_without_checkpoint_fails(
    tmp_path: Path,
    fake_runtime: None,
) -> None:
    source = tmp_path / "input" / "FORMATION_MUX_001"
    _valid_state(source)
    target = source / "CS-MECH-002" / "M0_STANDARD" / "S1"
    (target / "resume.pt").unlink()
    with pytest.raises(recovery.RecoveryError, match="checkpoint missing"):
        recovery.run_preflight(
            input_path=tmp_path / "input",
            output_path=tmp_path / "working" / "FORMATION_MUX_001",
            install=True,
        )


def test_checkpoint_hash_mismatch_fails(
    tmp_path: Path,
    fake_runtime: None,
) -> None:
    source = tmp_path / "input" / "FORMATION_MUX_001"
    _valid_state(source)
    target = source / "CS-MECH-002" / "M0_STANDARD" / "S1"
    checkpoint = target / "resume.pt"
    body = _read_checkpoint(checkpoint)
    body["updates"] = 1_800
    torch.save(body, checkpoint)
    with pytest.raises(recovery.RecoveryError, match="checkpoint hash mismatch"):
        recovery.run_preflight(
            input_path=tmp_path / "input",
            output_path=tmp_path / "working" / "FORMATION_MUX_001",
            install=True,
        )


def test_non_endpoint_checkpoint_cannot_back_complete_result(
    tmp_path: Path,
    fake_runtime: None,
) -> None:
    source = tmp_path / "input" / "FORMATION_MUX_001"
    _valid_state(source)
    target = source / "CS-MECH-002" / "M0_STANDARD" / "S1"
    checkpoint = target / "resume.pt"
    body = _read_checkpoint(checkpoint)
    body["updates"] = 1_800
    body["last_checkpoint_axis"] = 1_600
    torch.save(body, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    receipt_path = target / "CHECKPOINT_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["sha256"] = digest
    receipt["updates"] = 1_800
    receipt["last_checkpoint_axis"] = 1_600
    _write_json(receipt_path, receipt)
    progress_path = target / "LATEST_PROGRESS.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["updates"] = 1_800
    progress["checkpoint_sha256"] = digest
    _write_json(progress_path, progress)
    result_path = target / "ARM_RESULT.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["updates"] = 1_800
    _write_json(result_path, result)
    with pytest.raises(recovery.RecoveryError, match="non-endpoint checkpoint"):
        recovery.run_preflight(
            input_path=tmp_path / "input",
            output_path=tmp_path / "working" / "FORMATION_MUX_001",
            install=True,
        )


def test_result_identity_drift_fails_closed(
    tmp_path: Path,
    fake_runtime: None,
) -> None:
    source = tmp_path / "input" / "FORMATION_MUX_001"
    _valid_state(source)
    result_path = source / "CS-MECH-002" / "M0_STANDARD" / "S1" / "ARM_RESULT.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["seed_bundle"] = 73_099
    _write_json(result_path, result)
    with pytest.raises(recovery.RecoveryError, match="seed_bundle identity mismatch"):
        recovery.run_preflight(
            input_path=tmp_path / "input",
            output_path=tmp_path / "working" / "FORMATION_MUX_001",
            install=True,
        )


def test_checkpoint_identity_drift_fails_closed(
    tmp_path: Path,
    fake_runtime: None,
) -> None:
    source = tmp_path / "input" / "FORMATION_MUX_001"
    _valid_state(source)
    checkpoint = source / "CS-MECH-002" / "M0_STANDARD" / "S1" / "resume.pt"
    body = _read_checkpoint(checkpoint)
    body["seed_bundle"] = 73_099
    torch.save(body, checkpoint)
    with pytest.raises(recovery.RecoveryError, match="seed_bundle identity mismatch"):
        recovery.run_preflight(
            input_path=tmp_path / "input",
            output_path=tmp_path / "working" / "FORMATION_MUX_001",
            install=True,
        )


def test_cpu_only_checkpoint_is_rejected_for_t4_recovery(
    tmp_path: Path,
    fake_runtime: None,
) -> None:
    source = tmp_path / "input" / "FORMATION_MUX_001"
    _valid_state(source)
    checkpoint = source / "CS-MECH-002" / "M0_STANDARD" / "S1" / "resume.pt"
    body = _read_checkpoint(checkpoint)
    body["cuda_rng_state"] = None
    torch.save(body, checkpoint)
    with pytest.raises(recovery.RecoveryError, match="CUDA RNG state is missing"):
        recovery.run_preflight(
            input_path=tmp_path / "input",
            output_path=tmp_path / "working" / "FORMATION_MUX_001",
            install=True,
        )


def test_progress_hash_drift_fails_closed(
    tmp_path: Path,
    fake_runtime: None,
) -> None:
    source = tmp_path / "input" / "FORMATION_MUX_001"
    _valid_state(source)
    progress_path = source / "CS-MECH-002" / "M0_STANDARD" / "S1" / "LATEST_PROGRESS.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["checkpoint_sha256"] = "e" * 64
    _write_json(progress_path, progress)
    with pytest.raises(recovery.RecoveryError, match="progress hash mismatch"):
        recovery.run_preflight(
            input_path=tmp_path / "input",
            output_path=tmp_path / "working" / "FORMATION_MUX_001",
            install=True,
        )


def test_evidence_only_zip_is_rejected(
    tmp_path: Path,
    fake_runtime: None,
) -> None:
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
            install=True,
        )


def test_drive_qualified_archive_member_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("FORMATION_MUX_001/CAMPAIGN_STATE.json", "{}")
        handle.writestr("C:/escape.txt", "blocked")
    with pytest.raises(recovery.RecoveryError, match="unsafe ZIP member path"):
        recovery._archive_info(archive)


def test_ambiguous_saved_outputs_fail_closed(
    tmp_path: Path,
    fake_runtime: None,
) -> None:
    first = tmp_path / "one" / "FORMATION_MUX_001"
    second = tmp_path / "two" / "FORMATION_MUX_001"
    _valid_state(first)
    _valid_state(second)
    with pytest.raises(recovery.RecoveryError, match="ambiguous recovery input"):
        recovery.run_preflight(
            input_path=tmp_path,
            output_path=tmp_path / "working" / "FORMATION_MUX_001",
            install=True,
        )


def test_unregistered_arm_directory_fails_closed(
    tmp_path: Path,
    fake_runtime: None,
) -> None:
    source = tmp_path / "input" / "FORMATION_MUX_001"
    _valid_state(source)
    (source / "CS-MECH-002" / "UNREGISTERED" / "S1").mkdir(parents=True)
    with pytest.raises(recovery.RecoveryError, match="unregistered official experiment arms"):
        recovery.run_preflight(
            input_path=tmp_path / "input",
            output_path=tmp_path / "working" / "FORMATION_MUX_001",
            install=True,
        )


def test_direct_campaign_root_is_supported(
    tmp_path: Path,
    fake_runtime: None,
) -> None:
    source = tmp_path / "saved" / "FORMATION_MUX_001"
    _valid_state(source)
    output = tmp_path / "working" / "FORMATION_MUX_001"
    receipt = recovery.run_preflight(
        input_path=source,
        output_path=output,
        install=True,
    )
    assert receipt["source_kind"] == "installed_directory"
    assert receipt["official_checkpoint_count"] == 26


def test_nested_checkpoint_archive_is_supported(
    tmp_path: Path,
    fake_runtime: None,
) -> None:
    source = tmp_path / "source" / "FORMATION_MUX_001"
    _valid_state(source)
    attached = tmp_path / "attached"
    attached.mkdir()
    archive = attached / "saved-output.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for path in source.rglob("*"):
            if path.is_file():
                member = (Path("saved-output/FORMATION_MUX_001") / path.relative_to(source)).as_posix()
                handle.write(path, member)
    output = tmp_path / "working" / "FORMATION_MUX_001"
    receipt = recovery.run_preflight(
        input_path=attached,
        output_path=output,
        install=True,
    )
    assert receipt["source_kind"] == "installed_archive"
    assert receipt["archive_checkpoint_count"] == 26
    assert receipt["official_checkpoint_count"] == 26


def test_directory_and_archive_candidates_are_ambiguous(
    tmp_path: Path,
    fake_runtime: None,
) -> None:
    extracted = tmp_path / "attached" / "extracted" / "FORMATION_MUX_001"
    archived_source = tmp_path / "archive-source" / "FORMATION_MUX_001"
    _valid_state(extracted)
    _valid_state(archived_source)
    archive = tmp_path / "attached" / "saved-output.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for path in archived_source.rglob("*"):
            if path.is_file():
                member = (Path("saved-output/FORMATION_MUX_001") / path.relative_to(archived_source)).as_posix()
                handle.write(path, member)
    with pytest.raises(recovery.RecoveryError, match="ambiguous checkpoint-bearing recovery input"):
        recovery.run_preflight(
            input_path=tmp_path / "attached",
            output_path=tmp_path / "working" / "FORMATION_MUX_001",
            install=True,
        )


def test_output_inside_source_is_rejected_without_mutation(
    tmp_path: Path,
    fake_runtime: None,
) -> None:
    source = tmp_path / "input" / "FORMATION_MUX_001"
    _valid_state(source)
    output = source / "new-output"
    with pytest.raises(recovery.RecoveryError, match="must not be inside"):
        recovery.run_preflight(
            input_path=source,
            output_path=output,
            install=True,
        )
    assert not output.exists()


def test_public_surface_sealed_split_is_rejected(
    tmp_path: Path,
    fake_runtime: None,
) -> None:
    source = tmp_path / "input" / "FORMATION_MUX_001"
    _valid_state(source)
    public_path = source / "PUBLIC_SURFACE_MANIFEST.json"
    public = json.loads(public_path.read_text(encoding="utf-8"))
    public["splits"]["sealed"] = [{"family": "breach"}]
    _write_json(public_path, public)
    with pytest.raises(recovery.RecoveryError, match="SEALED_FIREWALL_BREACH"):
        recovery.run_preflight(
            input_path=tmp_path / "input",
            output_path=tmp_path / "working" / "FORMATION_MUX_001",
            install=True,
        )


def test_real_public_surface_validator_rejects_sealed_split() -> None:
    manifest = _preserved_public_manifest()
    manifest["splits"]["sealed"] = [{"family": "breach"}]
    body = {key: value for key, value in manifest.items() if key != "sha256"}
    manifest["sha256"] = recovery._canonical_sha256(body)
    with pytest.raises(recovery.RecoveryError, match="SEALED_FIREWALL_BREACH"):
        recovery._validate_public_manifest(manifest)


def test_tokenizer_commitment_drift_fails_closed(
    tmp_path: Path,
    fake_runtime: None,
) -> None:
    source = tmp_path / "input" / "FORMATION_MUX_001"
    _valid_state(source)
    tokenizer_path = source / "TOKENIZER_RECEIPT.json"
    tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    tokenizer["sealed_commitments"][recovery.s5_protocol.EXPERIMENT_A] = "d" * 64
    _write_json(tokenizer_path, tokenizer)
    with pytest.raises(recovery.RecoveryError, match="disagree with public manifest"):
        recovery.run_preflight(
            input_path=tmp_path / "input",
            output_path=tmp_path / "working" / "FORMATION_MUX_001",
            install=True,
        )


def test_latched_frontier_failure_fails_closed(
    tmp_path: Path,
    fake_runtime: None,
) -> None:
    source = tmp_path / "input" / "FORMATION_MUX_001"
    _valid_state(source)
    state_path = source / "TIE_ROLE_FRONTIER_STATE.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["global_failure"] = "historical worker failure"
    _write_json(state_path, state)
    with pytest.raises(recovery.RecoveryError, match="latched global failure"):
        recovery.run_preflight(
            input_path=tmp_path / "input",
            output_path=tmp_path / "working" / "FORMATION_MUX_001",
            install=True,
        )


def test_started_sealed_marker_fails_closed(
    tmp_path: Path,
    fake_runtime: None,
) -> None:
    source = tmp_path / "input" / "FORMATION_MUX_001"
    _valid_state(source)
    experiment = "TIE-ROLE-001"
    _write_json(source / experiment / "TIE_ROLE_SEALED_MARKER.json", {
        "schema": "anra.tie-role-sealed-marker/v1",
        "experiment": experiment,
        "state": "STARTED",
        "utc": "2026-09-23T00:00:00Z",
    })
    frontier_path = source / "TIE_ROLE_FRONTIER_STATE.json"
    frontier = json.loads(frontier_path.read_text(encoding="utf-8"))
    frontier["sealed"][experiment] = "STARTED"
    _write_json(frontier_path, frontier)
    with pytest.raises(recovery.RecoveryError, match="requires manual transaction review"):
        recovery.run_preflight(
            input_path=tmp_path / "input",
            output_path=tmp_path / "working" / "FORMATION_MUX_001",
            install=True,
        )


def test_complete_s5_marker_without_final_fails_closed(
    tmp_path: Path,
    fake_runtime: None,
) -> None:
    source = tmp_path / "input" / "FORMATION_MUX_001"
    _valid_state(source)
    experiment = recovery.s5_protocol.EXPERIMENT_A
    _write_json(source / experiment / "SEALED_MARKER.json", {
        "experiment": experiment,
        "state": "COMPLETE",
        "utc": "2026-09-23T00:00:00Z",
    })
    campaign_path = source / "CAMPAIGN_STATE.json"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    campaign["sealed"][experiment] = "COMPLETE"
    _write_json(campaign_path, campaign)
    with pytest.raises(recovery.RecoveryError, match="COMPLETE but final result missing"):
        recovery.run_preflight(
            input_path=tmp_path / "input",
            output_path=tmp_path / "working" / "FORMATION_MUX_001",
            install=True,
        )


def test_existing_output_is_never_overwritten(
    tmp_path: Path,
    fake_runtime: None,
) -> None:
    output = tmp_path / "working" / "FORMATION_MUX_001"
    _valid_state(output)
    marker = output / "USER_MARKER.txt"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(recovery.RecoveryError, match="refusing to overwrite existing working output"):
        recovery.run_preflight(
            input_path=tmp_path / "missing",
            output_path=output,
            install=True,
        )
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_atomic_install_refuses_existing_destination(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    staging.mkdir()
    output.mkdir()
    with pytest.raises(OSError):
        recovery._rename_noreplace(staging, output)
    assert staging.exists()
    assert output.exists()


def test_dangling_output_symlink_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "input" / "FORMATION_MUX_001"
    _valid_state(source)
    output = tmp_path / "working" / "FORMATION_MUX_001"
    output.parent.mkdir(parents=True)
    target = tmp_path / "missing-target"
    try:
        output.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(recovery.RecoveryError, match="symlink component"):
        recovery.run_preflight(
            input_path=tmp_path / "input",
            output_path=output,
            install=True,
        )


def test_recovery_notebook_matches_deterministic_builder() -> None:
    from experiments.COLAB import build_formation_mux_recovery as builder

    root = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (root / "notebooks" / "CYMEK_FORMATION_MUX_001_RECOVERY_T4X2.ipynb").read_text(
            encoding="utf-8"
        )
    )
    assert notebook == builder.build()
    metadata = notebook["metadata"]
    assert metadata["recovery_commit"] == "8609ba95f4e978cf3cdf8d20bd8a907eea8f6728"
    assert metadata["recovery_preflight_blob"] == "b507c2350a885e5e3432cac6f135fa58df52b69e"
    assert metadata["preflight_schema"] == recovery.SCHEMA
    assert metadata["operator_commit"] == recovery.OPERATOR_COMMIT
    assert metadata["operator_blob"] == recovery.OPERATOR_BLOB
    assert metadata["result_hash_immutability"] is True
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) == 2
    for index, cell in enumerate(code_cells):
        compile("".join(cell["source"]), f"formation-mux-recovery-cell-{index}", "exec")
    preflight_source = "".join(code_cells[0]["source"])
    operator_source = "".join(code_cells[1]["source"])
    assert "'tools.formation_mux_001_recovery_preflight'" in preflight_source
    assert "'-u'" not in preflight_source
    assert "'-u'" in operator_source
    assert "'tools/formation_mux_001_kaggle_operator_v12.py'" in operator_source
    assert "COMPLETED ARM IMMUTABILITY: PASS" in operator_source
    assert "same-kernel recovery receipt changed" in operator_source
    canonical = json.loads(
        (root / "notebooks" / "CYMEK_FORMATION_MUX_001_KAGGLE_T4X2.ipynb").read_text(
            encoding="utf-8"
        )
    )
    canonical_markdown = "".join(canonical["cells"][0]["source"])
    assert "This notebook is for fresh starts only" in canonical_markdown
    assert "CYMEK_FORMATION_MUX_001_RECOVERY_T4X2.ipynb" in canonical_markdown


def test_safe_checkpoint_load_accepts_primitive_payload(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "resume.pt"
    torch.save({"model": {"weight": torch.tensor([1])}}, checkpoint)
    assert recovery._safe_checkpoint_load(checkpoint)["model"]["weight"].item() == 1


def test_default_safe_loader_never_falls_back(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    checkpoint = tmp_path / "resume.pt"
    checkpoint.write_bytes(b"not-a-checkpoint")
    with pytest.raises(recovery.RecoveryError, match="safe checkpoint load failed"):
        recovery._safe_checkpoint_load(checkpoint)
