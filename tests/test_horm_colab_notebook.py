"""HORM Colab notebook contract: thin operator wrapper only."""

from __future__ import annotations

import ast
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks" / "HORM-colab-gpu.ipynb"
BUILDER = REPO / "experiments" / "COLAB" / "build_horm_colab.py"


def _cells() -> list[str]:
    body = json.loads(NOTEBOOK.read_text("utf-8"))
    return ["".join(cell["source"]) for cell in body["cells"]
            if cell["cell_type"] == "code"]


def test_notebook_is_exactly_three_code_cells():
    assert len(_cells()) == 3


def test_notebook_cells_compile():
    for index, source in enumerate(_cells()):
        try:
            ast.parse(source)
        except SyntaxError as error:  # pragma: no cover - failure path
            raise AssertionError(f"cell {index} does not compile: {error}")


def test_notebook_contains_no_science_logic():
    """No model/hormonal/training implementation may live here."""
    banned = ("class HormonalState", "class HormonalProjection",
              "class HormonalAttentionPatch", "def _run_arm",
              "def _run_horm003", "def _run_horm004",
              "def appraise_committed", "def _score_live_probes",
              "def train", "class Transformer", "class Attention")
    for source in _cells():
        for fragment in banned:
            assert fragment not in source, f"notebook must not define {fragment}"


def test_cell0_implements_the_freeze_sequence():
    cell0 = _cells()[0]
    for required in ("git", "clone", "checkout", "cymek-beta", "HEAD_SHA",
                      "DFDCD883", "95B2331A", "launch_readiness",
                      ".upper().startswith",
                      "torch.cuda.is_available",
                      "HORM COLAB PREEXECUTION GATE: PASS"):
        assert required in cell0, f"CELL 0 missing {required}"


def test_cell1_runs_only_committed_commands():
    cell1 = _cells()[1]
    for required in ("test_hormonal_state.py", "test_hormonal_integration.py",
                      "test_hormonal_session.py", "--horm003", "--horm004",
                      "--force", "-m\", \"pytest\", \"tests",
                      "v5_contracts.import_boundaries"):
        assert required in cell1, f"CELL 1 missing {required}"


def test_cell2_packages_hash_bound_bundle():
    cell2 = _cells()[2]
    for required in ("RESULT", "sha256", "COLAB_BUNDLE_MANIFEST.json",
                      "HORM-COLAB-RESULTS.zip"):
        assert required in cell2, f"CELL 2 missing {required}"


def test_full_mode_requires_cuda():
    assert any("requires Google Colab GPU" in source for source in _cells())


def test_builder_regenerates_notebook_byte_identical():
    import hashlib
    import subprocess
    import sys
    before = hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest()
    subprocess.run([sys.executable, str(BUILDER)], check=True,
                   cwd=str(REPO), capture_output=True)
    after = hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest()
    assert before == after, "builder is not deterministic; regenerate and commit"
