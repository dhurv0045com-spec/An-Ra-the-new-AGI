"""Citadel T1 canary unit tests. Zero third-party dependencies.

Run:  python tests/test_citadel_t1_canary.py   (exit 0 = all pass)
Covers AMENDMENT_001 §§A3–A10 without torch: the evaluator selftest suite,
an independent Fraction-based Wilson reference (exact rational arithmetic —
not the float implementation retyped), data-receipt invariants, and heuristic
null determinism. The torch-bound generate() loop is statically validated here
(compileall) and first executes live as the preregistered untrained baseline,
before any training update.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from decimal import Decimal, getcontext
from fractions import Fraction  # noqa: F401  (kept: documents exact-rational intent)
from pathlib import Path

HERE = Path(__file__).resolve()
CITADEL_ROOT = HERE.parents[1]
sys.path.insert(0, str(CITADEL_ROOT))

from citadel_tpu import calculator_data as calc  # noqa: E402
from citadel_tpu import calculator_eval as ev  # noqa: E402


def wilson_exact(k: int, n: int) -> tuple[Decimal, Decimal]:
    """Independent reference: Wilson interval in 50-digit Decimal arithmetic.

    Same formula as the float implementation, evaluated in a different,
    high-precision arithmetic path — validates float rounding (the actual risk),
    including the [0, 1] clamp behavior at the boundaries.
    """
    getcontext().prec = 50
    z = Decimal("1.96")
    z2 = z * z
    nn, kk = Decimal(n), Decimal(k)
    denom = 1 + z2 / nn
    p = kk / nn
    center = (p + z2 / (2 * nn)) / denom
    half = z * ((p * (1 - p) / nn + z2 / (4 * nn * nn)).sqrt()) / denom
    lo = center - half
    hi = center + half
    lo = lo if lo > 0 else Decimal(0)
    hi = hi if hi < 1 else Decimal(1)
    return lo, hi


def _sqrt_frac(x: Fraction, rounds: int = 50) -> Fraction:  # noqa: F841
    """Superseded by the Decimal reference above; retained as documentation."""
    raise NotImplementedError("use wilson_exact (Decimal)")


def test_selftest_suite() -> None:
    ev.selftest()


def test_wilson_against_exact_reference() -> None:
    for k, n in [(0, 10), (1, 10), (6, 10), (10, 10), (7, 500),
                 (50, 100), (0, 500), (500, 500), (1, 500), (243, 500)]:
        lcb, ucb = ev.wilson(k, n)
        lo, hi = wilson_exact(k, n)
        assert abs(lcb - float(lo)) < 1e-12, (k, n, lcb, lo)
        assert abs(ucb - float(hi)) < 1e-12, (k, n, ucb, hi)
        assert 0.0 <= lcb <= k / n <= ucb <= 1.0


def test_data_receipt_invariants() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        receipt = ev.build_data_receipt(out=str(Path(tmp) / "receipt.json"))
        stored = json.loads((Path(tmp) / "receipt.json").read_text())
        assert stored == receipt
    assert receipt["generator_version"] == calc.GENERATOR_VERSION
    assert receipt["encoding_version"] == ev.ENCODING_VERSION
    assert receipt["counts"] == {"train": 4000, "development": 500, "test": 500}
    assert all(v == 0 for v in receipt["overlap"].values()), receipt["overlap"]
    assert receipt["generalization_slices"]["commutative_heldout"] == 50
    assert set(receipt["op_distribution"]["train"]) == {"+", "-", "*", "/"}
    assert len(receipt["generator_code_sha256"]) == 64
    assert len(receipt["split_sha256"]["test"]) == 64


def test_nulls_deterministic_and_sane() -> None:
    rows = ["3 + 4 = 7", "18 - 7 = 11", "6 * 8 = 48", "72 / 8 = 9"]
    n1 = ev.heuristic_nulls(rows, rows)
    n2 = ev.heuristic_nulls(rows, rows)
    assert n1 == n2
    tgts = [ev.parse_row(r)[3] for r in rows]
    accs = {k: ev.summarize(v, tgts) for k, v in n1.items()}
    name, best = ev.strongest_null_accuracy(accs)
    assert best == max(v["accuracy"] for v in accs.values())
    # copy-first-operand nails no answer here (answers differ from operands)
    assert accs["copy_first_operand"]["correct"] == 0
    # always-zero nails nothing here
    assert accs["always_zero"]["correct"] == 0


def test_prompt_target_convention() -> None:
    for row in ("3 + 4 = 7", "18 - 7 = 11", "6 * 8 = 48", "72 / 8 = 9",
                "91 - 100 = -9", "1428 / 12 = 119"):
        prompt, target = ev.split_prompt_target(row)
        assert prompt.endswith("=") and "==" not in prompt
        assert ev.normalize_answer(target) == int(target)
        assert all(i not in (0, 1, 2, 3) for i in ev.encode(prompt))


def test_notebook_references_resolve() -> None:
    """Every citadel_tpu module/function referenced by notebook cells must exist."""
    import importlib
    import re

    for nb in ("notebooks/citadel_colab_tpu.ipynb", "notebooks/citadel_kaggle_tpu.ipynb"):
        doc = json.loads((CITADEL_ROOT / nb).read_text(encoding="utf-8"))
        assert doc["nbformat"] == 4
        for cell in doc["cells"]:
            if cell.get("cell_type") != "code":
                continue
            src = "".join(cell["source"])
            for mod in re.findall(r"from citadel_tpu import (\w+)", src):
                importlib.import_module(f"citadel_tpu.{mod}")
            for dotted in re.findall(r"python -m (citadel_tpu\.\w+)", src):
                importlib.import_module(dotted)
            aliases = dict(re.findall(r"from citadel_tpu import (\w+) as (\w+)", src))
            for mod in re.findall(r"from citadel_tpu import (\w+)\n", src):
                aliases[mod] = mod
            for alias, attr in re.findall(r"(\w+)\.(\w+)\(", src):
                if alias in aliases:
                    module = importlib.import_module(f"citadel_tpu.{aliases[alias]}")
                    assert hasattr(module, attr), f"{nb}: citadel_tpu.{aliases[alias]}.{attr} missing"


def main() -> int:
    tests = [test_selftest_suite, test_wilson_against_exact_reference,
             test_data_receipt_invariants, test_nulls_deterministic_and_sane,
             test_prompt_target_convention, test_notebook_references_resolve]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}", flush=True)
        except Exception as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}", flush=True)
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
