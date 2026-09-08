"""Notebook contract (sections 41, 42, 44): thin operator wrapper only."""

from __future__ import annotations

import ast
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks" / "cymek_colab_gpu_research_v5.ipynb"


def _cells() -> list[str]:
    body = json.loads(NOTEBOOK.read_text("utf-8"))
    return ["".join(cell["source"]) for cell in body["cells"]
            if cell["cell_type"] == "code"]


def test_notebook_is_exactly_three_cells():
    assert len(_cells()) == 3


def test_notebook_cells_compile():
    for index, source in enumerate(_cells()):
        try:
            ast.parse(source)
        except SyntaxError as error:  # pragma: no cover - failure path
            raise AssertionError(f"cell {index} does not compile: {error}")


def test_notebook_contains_no_science_logic():
    """No model/task/optimizer/fork implementation may live here."""
    banned = ("class Transformer", "class Attention", "class Block",
              "class RMSNorm", "def train_arm", "def acquire_parent",
              "def continuation_arm", "def render_batch",
              "def build_future_stream", "def decide_verdict")
    for source in _cells():
        for fragment in banned:
            assert fragment not in source, f"notebook must not define {fragment}"


def test_cell0_implements_the_freeze_sequence():
    cell0 = _cells()[0]
    for required in ("PREREGISTRATION.json",
                     "/content/CYR-GPU-005-PREREGISTRATION.json",
                     "git", "checkout", "assert HEAD == EXECUTABLE_SHA",
                     "assert_freeze_contract",
                     "CYR-GPU-005 PREEXECUTION GATE: PASS"):
        assert required in cell0, f"CELL 0 missing {required}"


def test_full_mode_requires_cuda():
    assert any("requires Google Colab GPU" in source for source in _cells())
