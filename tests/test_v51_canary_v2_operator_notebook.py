"""Static contract checks for the V5.1 Canary-v2 Colab operator launcher.

These tests protect the operator surface only. The scientific executable remains
frozen at 4470a34b7e2d4b2d673c328ef962c84d8e075b89 and is intentionally checked
out detached by the notebook before any scientific qualification/run command.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "cymek_colab_v51_canary_v2_t4.ipynb"
SCIENCE = "4470a34b7e2d4b2d673c328ef962c84d8e075b89"


def _load() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code(nb: dict) -> str:
    return "\n\n".join(
        "".join(cell.get("source", []))
        for cell in nb["cells"]
        if cell.get("cell_type") == "code"
    )


def test_notebook_is_valid_nbformat_and_every_code_cell_compiles():
    nb = _load()
    assert nb["nbformat"] == 4
    assert nb["metadata"]["accelerator"] == "GPU"
    assert nb["metadata"]["scientific_executable_commit"] == SCIENCE
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") == "code":
            compile("".join(cell.get("source", [])), f"canary-v2-cell-{i}", "exec")


def test_operator_detaches_exact_science_commit_and_verifies_blobs():
    code = _code(_load())
    assert f'SCIENCE = "{SCIENCE}"' in code
    assert '"checkout", "--detach", "-q", SCIENCE' in code
    assert "assert HEAD == SCIENCE" in code
    assert "EXPECTED_BLOBS" in code
    for required in (
        "anra_v5/v51_canary_v2_run.py",
        "experiments/V5_1_CANARY_V2/PREREGISTRATION.json",
        "tools/validate_v51_canary_v2.py",
        "tests/test_v51_canary_v2.py",
        "tests/test_v51_canary_v2_durability.py",
        "v5_model/core.py",
        "v5_training/production_backend.py",
        "v5_training/checkpoint.py",
        "v5_training/step.py",
    ):
        assert required in code
    assert "git\", \"hash-object" in NOTEBOOK.read_text(encoding="utf-8")


def test_operator_binds_dedicated_drive_root_and_resume_endpoint():
    code = _code(_load())
    assert "/content/drive/MyDrive/CYMEK/V5_1_CANARY_V2" in code
    assert 'ENV["V51_CANARY_V2_ROOT"] = str(OUT)' in code
    assert '"scientific_executable_commit": SCIENCE' in code
    assert '"--updates", "360"' in code
    assert '"--checkpoint-every", "24"' in code
    assert 'SCAN["action"] in {"START", "RESUME", "COMPLETE"}' in code
    assert 'training["status"] == "COMPLETE_ENDPOINT"' in code
    assert 'list(range(1, 361))' in code
    assert "1_474_560" in code


def test_operator_uses_isolated_preflight_and_cuda_fp32_path():
    code = _code(_load())
    assert 'run_mode("preflight", cuda=True)' in code
    assert 'pre["scientific_state_untouched"] is True' in code
    assert 'pre["isolated_lineage"] == "v51-canary-v2-preflight"' in code
    assert 'run_mode("evaluate", cuda=True)' in code
    assert 'run_mode("finalize", cuda=True)' in code
    assert "--bfloat16" not in code
    assert "MASK_4096" not in code


def test_operator_preserves_one_shot_sealed_firewall_and_no_outcome_branch():
    code = _code(_load())
    evaluate_pos = code.index('run_mode("evaluate", cuda=True)')
    finalize_pos = code.index('run_mode("finalize", cuda=True)')
    assert evaluate_pos < finalize_pos
    between = code[evaluate_pos:finalize_pos]
    assert "formation_gates" not in between
    assert "development_overall_exact_with_valid_eos" not in between
    assert 'marker.get("status") != "CONSUMED_AND_FINALIZED"' in code
    assert "Do not rerun sealed evaluation" in code
    assert 'final["executable_commit"] == SCIENCE' in code


def test_operator_packages_receipts_but_not_checkpoint_or_raw_splits():
    code = _code(_load())
    assert 'CYMEK_V51_CANARY_V2_RESULTS.zip' in code
    assert 'checkpoint_state_excluded": True' in NOTEBOOK.read_text(encoding="utf-8")
    assert 'raw_train_dev_sealed_rows_excluded": True' in NOTEBOOK.read_text(encoding="utf-8")
    # Bundle inclusion is explicit; state and split files are not appended.
    assert 'include += sorted(p for p in RECEIPTS.glob("*.json")' in code
    assert 'include.append(STATE)' not in code
