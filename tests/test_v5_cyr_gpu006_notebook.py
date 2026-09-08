import ast
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks" / "cymek_colab_gpu_research_v6.ipynb"
RUNNER = REPO / "anra_v5" / "cyr_gpu006_run.py"


def _cells():
    body = json.loads(NOTEBOOK.read_text("utf-8"))
    return ["".join(cell["source"]) for cell in body["cells"] if cell["cell_type"] == "code"]


def test_three_cells_compile():
    cells = _cells()
    assert len(cells) == 3
    for source in cells:
        ast.parse(source)


def test_cell0_freezes_and_calibrates_generation_plus_training():
    cell0 = _cells()[0]
    for text in ("PREREGISTRATION.json", "checkout", "EXECUTABLE_SHA",
                 "dependency_blobs", "calibrate_candidates", "resolve_from_calibrations",
                 "CYR-GPU-006 PREEXECUTION GATE: PASS"):
        assert text in cell0


def test_cell1_passes_cuda_and_resolved_plan_to_runner():
    cell1 = _cells()[1]
    assert 'DEVICE = torch.device("cuda")' in cell1
    assert "device=DEVICE" in cell1
    assert "resolved=RESOLVED" in cell1
    assert "calibrations=CALIBRATIONS" in cell1
    assert "drive.mount" in cell1
    assert "MyDrive/CYMEK/CYR-GPU-006" in cell1


def test_runner_attempts_all_parent_seeds_without_first_g90_break():
    source = RUNNER.read_text("utf-8")
    marker = "for seed in core.CYR6_PARENT_SEEDS:"
    assert marker in source
    block = source[source.index(marker):source.index('campaign["parents"]', source.index(marker))]
    assert "break" not in block


def test_runner_full_mode_has_no_cpu_fallback():
    source = RUNNER.read_text("utf-8")
    assert 'device = torch.device("cuda")' in source
    assert "refuses non-CUDA device" in source
