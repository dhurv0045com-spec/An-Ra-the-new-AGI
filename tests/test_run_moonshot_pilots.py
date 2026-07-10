from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.run_moonshot_pilots import run_moonshot_pilots

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_moonshot_executor_blocks_missing_evidence() -> None:
    report = run_moonshot_pilots({})
    assert report["complete"] is False
    assert report["blocked"] is True
    assert report["failed"] is False
    assert len(report["rows"]) == 7


def test_moonshot_executor_accepts_only_complete_passing_evidence() -> None:
    report = run_moonshot_pilots(
        {
            "m1": {
                "short_context_ratio": 0.99,
                "long_context_speedup": 1.6,
                "model_parameters": 150_000_000,
                "seed_count": 3,
            },
            "m2": {
                "reconstruction_mse_improvement": 0.31,
                "contrastive_recall_at_1": 0.41,
                "vision_qa_accuracy": 0.61,
                "heldout_pairs": 5_000,
                "qa_items": 200,
            },
            "m3": {
                "reasoning_score_ratio": 1.16,
                "inference_flops_ratio": 1.0,
                "model_parameters": 150_000_000,
                "seed_count": 3,
            },
            "m4": {
                "calibration_error": 0.05,
                "action_success": 0.8,
                "simulation_baseline_gain": 0.1,
                "digital_top1_accuracy": 0.7,
                "digital_majority_baseline_gain": 0.1,
            },
            "m5": {"training_pairs": 20_000, "recall_at_5_gain": 0.1},
            "m6": {"proof_cases": 100, "deterministic_pass_rate": 0.96},
            "m7": {
                "merged_human_approved_prs": 10,
                "signed_gate_records": 10,
                "reverted_prs": 0,
                "unauthorized_apply_count": 0,
            },
        }
    )
    assert report["complete"] is True


def test_moonshot_executor_direct_entrypoint_uses_this_workspace(tmp_path: Path) -> None:
    report_path = tmp_path / "status.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_moonshot_pilots.py",
            "--json-out",
            str(report_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 3
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["blocked"] is True
    assert report["failed"] is False
