"""Static contract for the CS-TRANSFER-001 T4 operator notebook.

This test intentionally validates only operator invariants. Scientific code is
frozen at CS_TRANSFER_001_EXECUTABLE_COMMIT and the notebook must detach that
commit rather than execute the moving branch head.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "cymek_colab_cs_transfer_001_t4.ipynb"
SCIENCE = "a916d1c8d2637abb86d16b1c78e418c95461f3c7"
DRIVE_ROOT = "/content/drive/MyDrive/CYMEK/CS_TRANSFER_001"


def _load():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert nb["metadata"]["accelerator"] == "GPU"
    assert nb["metadata"]["colab"]["gpuType"] == "T4"
    return nb


def _source(nb) -> str:
    return "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])


def test_all_code_cells_compile():
    nb = _load()
    for index, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")


def test_notebook_detaches_exact_science_commit_and_verifies_blobs():
    source = _source(_load())
    assert f'SCIENCE = "{SCIENCE}"' in source
    assert '"checkout", "--detach", SCIENCE' in source
    assert 'HEAD == SCIENCE' in source
    for path in (
        "anra_v5/cs_transfer_001_run_v3.py",
        "anra_v5/cs_transfer_001_run_v2.py",
        "anra_v5/cs_transfer_001_data.py",
        "anra_v5/cs_transfer_001_model.py",
        "experiments/CS_TRANSFER_001/PREREGISTRATION.json",
        "experiments/CS_TRANSFER_001/AMENDMENT_1.json",
        "v5_training/production_backend.py",
        "v5_training/checkpoint.py",
    ):
        assert path in source
    assert 'rev-parse", f"HEAD:{path}"' in source


def test_notebook_uses_only_dedicated_drive_root_and_preserves_resume_state():
    source = _source(_load())
    assert DRIVE_ROOT in source
    assert "genisis-arkenstone" not in source
    assert "CYR-GPU-014-R1C" not in source
    assert "V5_1_CANARY_V2" not in source
    assert ".unlink()" not in source.replace("ZIP.unlink()", "")
    assert "shutil.rmtree" not in source
    assert "latest_heads" in source
    assert "after_heads == before_heads" in source


def test_protocol_data_and_cuda_gates_precede_scientific_arms():
    source = _source(_load())
    protocol = source.index('["--mode", "protocol"]')
    prepare = source.index('["--mode", "prepare"]')
    preflight = source.index('["--mode", "preflight", "--cuda"]')
    run_arm = source.index('["--mode", "run-arm"')
    assert protocol < prepare < preflight < run_arm
    assert 'predictive_shortcut_max"] < 0.35' in source
    assert 'acceptance_rate"]) >= 0.15' in source
    assert 'maximum_content_id"]) < 4096' in source
    assert 'contamination"]["clean"] is True' in source


def test_all_eight_arms_are_fixed_and_full_softmax_physical_treatments():
    source = _source(_load())
    assert 'range(4)' in source
    assert '("PHYS_4096", "PHYS_24576")' in source
    assert "MASK_4096" not in source
    assert "PHYS_4096" in source and "PHYS_24576" in source
    assert "480 updates / 1,966,080 real tokens" in source


def test_development_is_frozen_before_one_shot_sealed_finalization():
    source = _source(_load())
    development = source.index('["--mode", "development"]')
    finalize = source.index('["--mode", "finalize", "--cuda"]')
    assert development < finalize
    assert "SEALED_CONSUMPTION.json" in source
    assert "FINAL_RESULT already exists; sealed evaluation will NOT be repeated." in source
    assert "Do not delete it" in source


def test_compact_bundle_excludes_raw_data_and_checkpoint_state():
    source = _source(_load())
    assert 'if "state" in parts or "data" in parts: continue' in source
    assert "CS_TRANSFER_001_RESULTS.zip" in source
