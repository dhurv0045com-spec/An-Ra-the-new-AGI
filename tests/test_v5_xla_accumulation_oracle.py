"""XLA accumulation-boundary oracle + structural guard on the production
entry: the SUM collective happens ONCE per logical update, never inside the
microstep loop."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from v5_experiments.xla_accumulation_oracle import run_xla_accumulation_oracle

REPO = Path(__file__).resolve().parents[1]


def test_oracle_correct_flow_matches_and_buggy_flow_fails():
    receipt = run_xla_accumulation_oracle()
    assert receipt["passed"], receipt
    assert receipt["correct_flow_matches_logical_update"]
    assert receipt["correct_flow_collectives"] == [4]
    assert not receipt["buggy_flow_matches_logical_update"]
    assert receipt["buggy_flow_collectives"] == [1, 2, 3, 4]
    assert receipt["truth"]["denominator"] == receipt["correct"]["denominator"]


def test_oracle_requires_three_replicas_and_four_microsteps():
    with pytest.raises(ValueError, match="at least 3"):
        run_xla_accumulation_oracle(replicas=2)
    with pytest.raises(ValueError, match="4 microsteps"):
        run_xla_accumulation_oracle(microsteps=3)


def test_production_entry_reduces_once_at_the_boundary():
    """AST guard: in `backend_step`, no all_reduce call may live inside the
    microstep loop; exactly one must exist after it."""
    source = (REPO / "v5_training" / "production_entry.py").read_text("utf-8")
    tree = ast.parse(source)
    function = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "backend_step")
    loop = next(node for node in ast.walk(function)
                if isinstance(node, ast.For)
                and isinstance(node.target, ast.Tuple)
                and any(isinstance(name, ast.Name) and name.id == "bucket"
                        for name in node.target.elts))
    def is_reduce_call(node: ast.AST) -> bool:
        return (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "all_reduce_sum_gradients")
    inside = [node for node in ast.walk(loop) if is_reduce_call(node)]
    assert not inside, ("all_reduce inside the microstep loop re-reduces "
                        "accumulated gradients (the CYR-GPU-005 defect)")
    whole = [node for node in ast.walk(function) if is_reduce_call(node)]
    assert len(whole) == 1
    assert loop.end_lineno is not None and whole[0].lineno > loop.end_lineno
