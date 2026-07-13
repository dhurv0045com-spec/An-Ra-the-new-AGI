from __future__ import annotations

import json
from pathlib import Path

import pytest

from training import pilot_factorial
from training.forecast_ledger import ForecastAuditError, audit_pre_launch, read_ledger
from training.launch_manifest import load_and_validate_manifest

KEY = "pilot-factorial-test-key"


def test_factorial_definition_is_complete_and_unique() -> None:
    cells = pilot_factorial.PILOT_FACTORIAL
    ids = [cell.cell_id for cell in cells]

    assert len(cells) >= 20
    assert len(ids) == len(set(ids))
    assert len(pilot_factorial.PILOT_SEEDS) == 3
    assert {cell.scale for cell in cells} == {"50m", "150m"}
    moonshots = {cell.cell_id for cell in cells if cell.moonshot}
    assert moonshots == {"m1-ssm-hybrid", "m3-latent-reasoning", "m5-retriever-head"}
    for cell in cells:
        assert cell.predicted_low <= cell.predicted_high
        assert cell.gate
        assert cell.metric
        # Law 1: no pilot cell may reference the earned lineage as a source.
        assert cell.optimizer in {"adamw", "muon"}
    blocked = {cell.cell_id for cell in cells if cell.blocked_on}
    assert "p150-v4tok" in blocked  # canonical V4 waits on Stream B
    executable = {
        cell.cell_id
        for cell in cells
        if not pilot_factorial.execution_blocker(cell)
    }
    assert {
        "p050-baseline",
        "p050-muon",
        "p150-baseline",
        "p150-muon",
        "p150-qknorm-off",
        "p150-swa-full",
        "p050-mtp",
        "p150-mtp",
        "p150-muon-mtp",
        "p050-moe",
        "p150-moe",
        "p150-muon-moe",
        "p150-moe-mtp",
        "p150-muon-moe-mtp",
        "c050-code-first",
        "c050-math-ramp",
        "c050-identity-late",
    } <= executable


def test_v4_axis_becomes_launchable_only_after_stream_b_is_resolved() -> None:
    cell = next(cell for cell in pilot_factorial.PILOT_FACTORIAL if cell.cell_id == "p150-v4tok")

    assert pilot_factorial.execution_blocker(cell) == "stream-b-canonical-v4"
    assert (
        pilot_factorial.execution_blocker(
            cell,
            resolved_blockers=frozenset({"stream-b-canonical-v4"}),
        )
        == ""
    )


def test_build_manifests_registers_forecasts_first_and_audits(tmp_path: Path) -> None:
    ledger = tmp_path / "forecasts.jsonl"
    cells = pilot_factorial.PILOT_FACTORIAL[:2]
    train_manifest = tmp_path / "train" / "manifest.json"
    validation_manifest = tmp_path / "validation" / "manifest.json"
    train_manifest.parent.mkdir()
    validation_manifest.parent.mkdir()
    train_manifest.write_text("{}", encoding="utf-8")
    validation_manifest.write_text("{}", encoding="utf-8")

    manifests = pilot_factorial.build_pilot_launch_manifests(
        tmp_path,
        owner_authorized=True,
        key=KEY,
        cells=cells,
        ledger_path=ledger,
        data_manifests=(str(train_manifest), str(validation_manifest)),
    )

    assert len(manifests) == len(cells) * len(pilot_factorial.PILOT_SEEDS)
    forecasts = [entry for entry in read_ledger(ledger) if entry["kind"] == "forecast"]
    assert len(forecasts) == 2
    cell_ids = {cell.cell_id for cell in cells}
    for manifest in manifests:
        assert manifest["pilot_cell_id"] in cell_ids
        assert manifest["seeds"] == [manifest["pilot_seed"]]
        assert manifest["seed"] == manifest["pilot_seed"]
        assert manifest["pilot_seed"] in pilot_factorial.PILOT_SEEDS
        assert manifest["checkpoint_source"] == "scratch"
        assert Path(str(manifest["tokenizer_path"])).is_file()
        assert manifest["checkpoint_read_only"] is True
        audit = audit_pre_launch(manifest, path=ledger)
        assert audit["passed"] is True
        assert audit["lead_seconds"] >= 0
        path = (
            tmp_path
            / "cells"
            / str(manifest["pilot_cell_id"])
            / f"seed-{manifest['pilot_seed']}.json"
        )
        assert path.exists()
        loaded = load_and_validate_manifest(path, key=KEY)
        assert loaded["forecast_id"] == manifest["forecast_id"]


def test_build_manifests_requires_owner_and_three_seeds(tmp_path: Path) -> None:
    ledger = tmp_path / "forecasts.jsonl"
    with pytest.raises(PermissionError):
        pilot_factorial.build_pilot_launch_manifests(
            tmp_path,
            owner_authorized=False,
            key=KEY,
            cells=pilot_factorial.PILOT_FACTORIAL[:1],
            ledger_path=ledger,
        )
    with pytest.raises(ValueError, match="three seeds"):
        pilot_factorial.build_pilot_launch_manifests(
            tmp_path,
            owner_authorized=True,
            key=KEY,
            seeds=(1, 2),
            cells=pilot_factorial.PILOT_FACTORIAL[:1],
            ledger_path=ledger,
        )


def test_post_hoc_forecast_swap_is_detected(tmp_path: Path) -> None:
    ledger = tmp_path / "forecasts.jsonl"
    train_manifest = tmp_path / "train" / "manifest.json"
    validation_manifest = tmp_path / "validation" / "manifest.json"
    train_manifest.parent.mkdir()
    validation_manifest.parent.mkdir()
    train_manifest.write_text("{}", encoding="utf-8")
    validation_manifest.write_text("{}", encoding="utf-8")
    manifest = pilot_factorial.build_pilot_launch_manifests(
        tmp_path,
        owner_authorized=True,
        key=KEY,
        cells=pilot_factorial.PILOT_FACTORIAL[:1],
        ledger_path=ledger,
        data_manifests=(str(train_manifest), str(validation_manifest)),
    )[0]
    # A forecast registered after the manifest exists must not certify it.
    late = pilot_factorial.register_forecast(
        cell_id=str(manifest["pilot_cell_id"]),
        metric="token_efficiency_x",
        predicted_low=1.0,
        predicted_high=2.0,
        gate="post-hoc",
        seeds=[int(manifest["seed"]), 2, 3],
        path=ledger,
    )
    doctored = dict(manifest)
    doctored["forecast_id"] = late["forecast_id"]
    doctored["created_at"] = float(late["registered_at"]) - 30.0
    with pytest.raises(ForecastAuditError, match="Post-hoc"):
        audit_pre_launch(doctored, path=ledger)


def test_manifest_files_are_valid_json_with_signatures(tmp_path: Path) -> None:
    ledger = tmp_path / "forecasts.jsonl"
    train_manifest = tmp_path / "train" / "manifest.json"
    validation_manifest = tmp_path / "validation" / "manifest.json"
    train_manifest.parent.mkdir()
    validation_manifest.parent.mkdir()
    train_manifest.write_text("{}", encoding="utf-8")
    validation_manifest.write_text("{}", encoding="utf-8")
    pilot_factorial.build_pilot_launch_manifests(
        tmp_path,
        owner_authorized=True,
        key=KEY,
        cells=pilot_factorial.PILOT_FACTORIAL[:1],
        ledger_path=ledger,
        data_manifests=(str(train_manifest), str(validation_manifest)),
    )
    cell_id = pilot_factorial.PILOT_FACTORIAL[0].cell_id
    payload = json.loads(
        (
            tmp_path
            / "cells"
            / cell_id
            / f"seed-{pilot_factorial.PILOT_SEEDS[0]}.json"
        ).read_text(encoding="utf-8")
    )
    assert payload["signature"]
    assert payload["schema_version"] == 3
    assert set(payload["data_manifest_hashes"]) == set(payload["data_manifests"])
    assert set(payload["data_manifest_roles"].values()) == {"train", "validation"}
    assert payload["pilot_axes"]


def test_blocked_manifest_can_be_audited_but_not_launched(tmp_path: Path) -> None:
    ledger = tmp_path / "forecasts.jsonl"
    train_manifest = tmp_path / "train" / "manifest.json"
    validation_manifest = tmp_path / "validation" / "manifest.json"
    train_manifest.parent.mkdir()
    validation_manifest.parent.mkdir()
    train_manifest.write_text("{}", encoding="utf-8")
    validation_manifest.write_text("{}", encoding="utf-8")
    cell = next(cell for cell in pilot_factorial.PILOT_FACTORIAL if cell.moonshot)
    signed = pilot_factorial.build_pilot_launch_manifests(
        tmp_path,
        owner_authorized=True,
        key=KEY,
        cells=(cell,),
        ledger_path=ledger,
        data_manifests=(str(train_manifest), str(validation_manifest)),
    )[0]
    path = (
        tmp_path
        / "cells"
        / cell.cell_id
        / f"seed-{signed['seed']}.json"
    )

    with pytest.raises(PermissionError, match="blocked on"):
        load_and_validate_manifest(path, key=KEY)
    audited = load_and_validate_manifest(path, key=KEY, allow_blocked=True)
    assert audited["blocked_on"]


def test_signed_launch_rejects_data_manifest_changed_after_signing(tmp_path: Path) -> None:
    ledger = tmp_path / "forecasts.jsonl"
    train_manifest = tmp_path / "train" / "manifest.json"
    validation_manifest = tmp_path / "validation" / "manifest.json"
    train_manifest.parent.mkdir()
    validation_manifest.parent.mkdir()
    train_manifest.write_text('{"version":1}', encoding="utf-8")
    validation_manifest.write_text('{"version":1}', encoding="utf-8")
    manifest = pilot_factorial.build_pilot_launch_manifests(
        tmp_path,
        owner_authorized=True,
        key=KEY,
        cells=pilot_factorial.PILOT_FACTORIAL[:1],
        ledger_path=ledger,
        data_manifests=(str(train_manifest), str(validation_manifest)),
    )[0]
    signed_path = (
        tmp_path
        / "cells"
        / str(manifest["pilot_cell_id"])
        / f"seed-{manifest['pilot_seed']}.json"
    )
    train_manifest.write_text('{"version":2}', encoding="utf-8")

    with pytest.raises(ValueError, match="hash does not match"):
        load_and_validate_manifest(signed_path, key=KEY)
