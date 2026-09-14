"""FORMATION-MUX-001 Kaggle operator — Science S5 + VOCAB-PRESSURE-001.

This wrapper leaves frozen Science S5 untouched and adds one prospective,
development-only mechanism diagnostic after all official arms complete and
before any sealed rows are regenerated. The diagnostic cannot change S5
primary/sealed verdicts.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import formation_mux_001_kaggle_operator_v8 as v8

SCIENCE_COMMIT = v8.SCIENCE_COMMIT
OPERATOR_NAME = "tools/formation_mux_001_kaggle_operator_v9.py"
XFACTOR_SPEC = (
    "docs/cymek/experiments/FORMATION-MUX-001/"
    "XFACTOR_VOCAB_PRESSURE_PREREGISTRATION_V1.json"
)
XFACTOR_SPEC_SHA256 = "ea70f361a9c038d22a87967f611379e772fccd8870855b905c1bfef07ad83969"
XFACTOR_TOOL = "tools/formation_mux_001_vocab_pressure_v1.py"
XFACTOR_TEST = "tests/test_formation_mux_vocab_pressure_v1.py"

_ORIGINAL_QUALIFY = v8.qualify_s5
_ORIGINAL_FINALIZE_SEALED = v8.finalize_sealed


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def verify_xfactor_preregistration(repo: Path) -> dict[str, Any]:
    path = repo / XFACTOR_SPEC
    if not path.exists():
        raise v8.v7.base.GlobalIntegrityError(
            f"VOCAB-PRESSURE preregistration missing: {XFACTOR_SPEC}"
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != XFACTOR_SPEC_SHA256:
        raise v8.v7.base.GlobalIntegrityError(
            "VOCAB-PRESSURE preregistration identity mismatch: "
            f"{digest} != {XFACTOR_SPEC_SHA256}"
        )
    body = json.loads(path.read_text(encoding="utf-8"))
    if body.get("written_before_official_formation_mux_outcomes") is not True:
        raise v8.v7.base.GlobalIntegrityError(
            "VOCAB-PRESSURE preregistration is not marked prospective"
        )
    relation = body.get("relationship_to_science_s5", {})
    if (
        relation.get("diagnostic_only") is not True
        or relation.get("changes_training") is not False
        or relation.get("adds_training_arms") is not False
        or relation.get("may_change_frozen_verdict") is not False
    ):
        raise v8.v7.base.GlobalIntegrityError(
            "VOCAB-PRESSURE preregistration violated diagnostic-only contract"
        )
    return {
        "schema": "anra.formation-mux-xfactor-preregistration-receipt/v1",
        "diagnostic": "VOCAB-PRESSURE-001",
        "path": XFACTOR_SPEC,
        "sha256": digest,
        "prospective": True,
        "diagnostic_only": True,
        "changes_training": False,
        "sealed_rows_used": False,
        "can_change_frozen_s5_verdict": False,
    }


def qualify_s5_with_xfactor(
    repo: Path, public_surface: Path, out: Path
) -> dict[str, Any]:
    receipt = _ORIGINAL_QUALIFY(repo, public_surface, out)
    xfactor_receipt = verify_xfactor_preregistration(repo)
    compile_run = subprocess.run(
        [sys.executable, "-m", "py_compile", XFACTOR_TOOL, OPERATOR_NAME],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    tests = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", XFACTOR_TEST],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    receipt.update(
        {
            "xfactor_preregistration": xfactor_receipt,
            "xfactor_py_compile_returncode": compile_run.returncode,
            "xfactor_py_compile_stderr_tail": compile_run.stderr[-4000:],
            "xfactor_pytest_returncode": tests.returncode,
            "xfactor_pytest_stdout_tail": tests.stdout[-8000:],
            "xfactor_pytest_stderr_tail": tests.stderr[-4000:],
        }
    )
    if compile_run.returncode != 0 or tests.returncode != 0:
        receipt["status"] = "XFACTOR_QUALIFICATION_FAIL"
        v8.v7.base._atomic_json(out / "QUALIFICATION.json", receipt)
        raise v8.v7.base.GlobalIntegrityError(
            "VOCAB-PRESSURE-001 Kaggle qualification failed"
        )
    receipt["status"] = (
        "CPU_STATIC_S5_PASS / GPU_E2E_PASS_ENGINEERING_ONLY / "
        "VOCAB_PRESSURE_QUALIFIED"
    )
    v8.v7.base._atomic_json(out / "QUALIFICATION.json", receipt)
    return receipt


def run_xfactor(public_path: Path, out: Path, torch: Any) -> dict[str, Any]:
    from tools.formation_mux_001_vocab_pressure_v1 import run_diagnostic

    prereg = verify_xfactor_preregistration(_repo_root())
    v8.v7.base._atomic_json(
        out / "XFACTOR_PREREGISTRATION_RECEIPT.json", prereg
    )
    result = run_diagnostic(
        public_path=public_path,
        out=out,
        torch=torch,
        device=torch.device("cuda:0"),
    )
    if result.get("sealed_rows_used") is not False:
        raise v8.v7.base.GlobalIntegrityError(
            "VOCAB-PRESSURE diagnostic unexpectedly used sealed rows"
        )
    if result.get("can_change_frozen_s5_verdict") is not False:
        raise v8.v7.base.GlobalIntegrityError(
            "VOCAB-PRESSURE diagnostic violated frozen-verdict contract"
        )
    return result


def finalize_sealed_with_xfactor(
    public_path: Path, out: Path, torch: Any, tokenizer: Any
) -> dict[str, Any]:
    # Deliberately before sealed regeneration: development-only mechanism
    # localization cannot inspect or adapt to sealed outcomes.
    run_xfactor(public_path, out, torch)
    return _ORIGINAL_FINALIZE_SEALED(public_path, out, torch, tokenizer)


def _bind() -> None:
    # v8 functions resolve these module globals at call time.
    v8.OPERATOR_NAME = OPERATOR_NAME
    v8.v7.OPERATOR_NAME = OPERATOR_NAME
    v8.qualify_s5 = qualify_s5_with_xfactor
    v8.finalize_sealed = finalize_sealed_with_xfactor


def main(argv: list[str] | None = None) -> int:
    _bind()
    return v8.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
