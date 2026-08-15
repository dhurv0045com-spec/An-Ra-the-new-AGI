from __future__ import annotations

import json
from pathlib import Path

from scripts.run_pilot_queue import build_training_command, completed_run


def test_training_command_is_base_only_and_signed(tmp_path: Path) -> None:
    manifest = tmp_path / "seed-1301.json"
    command = build_training_command("python", manifest)

    assert command[:3] == ["python", "-m", "training.train_unified"]
    assert command[command.index("--mode") + 1] == "session"
    assert command[command.index("--prepare-data") + 1] == "never"
    assert command[command.index("--launch-manifest") + 1] == str(manifest.resolve())


def test_completed_run_requires_matching_artifact_seed_and_run_id(tmp_path: Path) -> None:
    artifact = tmp_path / "seed-1301.pt"
    artifact.write_bytes(b"checkpoint")
    manifest = {
        "artifact_path": str(artifact),
        "seed": 1301,
        "run_id": "run-a",
    }
    report_path = artifact.with_suffix(".run.json")
    report_path.write_text(
        json.dumps(
            {
                "seed": 1301,
                "launch_manifest": {"run_id": "run-a"},
                "stages": {"base": {"exit_code": 0}},
            }
        ),
        encoding="utf-8",
    )

    assert completed_run(manifest) is True
    manifest["run_id"] = "run-b"
    assert completed_run(manifest) is False
