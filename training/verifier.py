from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
import subprocess
import tempfile

try:
    from anra_paths import inject_all_paths

    inject_all_paths()
    from domain_verifiers import (
        verify_citation_grounding,
        verify_constraint_json,
        verify_qiskit,
        verify_rdkit,
        verify_verilog,
    )
except Exception:  # pragma: no cover - optional Phase 3 bridge may be unavailable.
    verify_qiskit = verify_rdkit = verify_verilog = verify_constraint_json = verify_citation_grounding = None


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    return_code: int


@dataclass
class VerificationResult:
    score: float
    tier: int
    reason: str
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0


class VerifierHierarchy:
    def __init__(self, workspace: str | Path = "workspace") -> None:
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)

    def _safe_exec(self, code: str) -> ExecutionResult:
        tmp_path = None
        try:
            fd, raw_path = tempfile.mkstemp(suffix=".py", dir=str(self.workspace))
            os.close(fd)
            tmp_path = Path(raw_path)
            tmp_path.write_text(code, encoding="utf-8")
            env = {k: v for k, v in os.environ.items() if k in {"PATH", "PYTHONPATH", "HOME", "TMPDIR"}}
            proc = subprocess.run(
                ["python3", str(tmp_path)],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.workspace),
                env=env,
            )
            return ExecutionResult(proc.stdout[:4096], proc.stderr[:4096], int(proc.returncode))
        except subprocess.TimeoutExpired as exc:
            out = (exc.stdout or "")[:4096] if isinstance(exc.stdout, str) else ""
            err = (exc.stderr or "")[:4096] if isinstance(exc.stderr, str) else ""
            return ExecutionResult(out, err, 124)
        except Exception as exc:
            return ExecutionResult("", str(exc)[:4096], 1)
        finally:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def verify_code(self, code: str, test_code: str = "") -> VerificationResult:
        joined = code if not test_code else f"{code}\n\n{test_code}\n"
        result = self._safe_exec(joined)
        if result.return_code == 124:
            return VerificationResult(0.0, 1, "timeout", result.stdout, result.stderr, 124)
        if result.return_code == 0 and test_code:
            return VerificationResult(1.0, 1, "tests_passed", result.stdout, result.stderr, 0)
        if result.return_code == 0:
            return VerificationResult(0.7, 1, "ran_without_tests", result.stdout, result.stderr, 0)
        if "SyntaxError" in result.stderr or "Traceback" in result.stderr:
            return VerificationResult(0.2, 1, "runtime_or_syntax_error", result.stdout, result.stderr, result.return_code)
        return VerificationResult(0.0, 1, "crash", result.stdout, result.stderr, result.return_code)

    def verify_math(self, expression: str, expected: str) -> VerificationResult:
        try:
            import sympy as sp

            lhs = sp.sympify(expression)
            rhs = sp.sympify(expected)
            ok = bool(sp.simplify(lhs - rhs) == 0)
            return VerificationResult(1.0 if ok else 0.0, 1, "equivalent" if ok else "not_equivalent")
        except Exception as exc:
            return VerificationResult(0.0, 1, f"math_error: {exc}")

    def verify_file_state(self, check_fn) -> VerificationResult:
        try:
            ok = bool(check_fn())
            return VerificationResult(1.0 if ok else 0.0, 1, "file_state_ok" if ok else "file_state_mismatch")
        except Exception as exc:
            return VerificationResult(0.0, 1, f"file_state_error: {exc}")

    def verify_instruction(self, response: str, pattern: str) -> VerificationResult:
        matched = re.search(pattern, response or "") is not None
        length_ok = len((response or "").strip()) >= 8
        score = 0.8 if matched and length_ok else 0.3 if matched else 0.0
        return VerificationResult(score, 2, "heuristic_instruction")

    def verify_open_ended(self, task: str, response: str) -> VerificationResult:
        response = (response or "").strip()
        if not response:
            return VerificationResult(0.0, 3, "empty_response")

        task_terms = re.findall(r"[a-z0-9]{3,}", (task or "").lower())
        response_terms = re.findall(r"[a-z0-9]{3,}", response.lower())
        if not response_terms:
            return VerificationResult(0.0, 3, "no_content_terms")

        term_counts: dict[str, int] = {}
        for term in response_terms:
            term_counts[term] = term_counts.get(term, 0) + 1

        task_vocab = set(task_terms)
        response_vocab = set(response_terms)
        alignment = (
            sum(1 for term in task_vocab if term in response_vocab) / max(1, len(task_vocab))
            if task_vocab
            else 0.35
        )
        substance = min(1.0, len(response_terms) / 80.0)
        structure_markers = sum(
            marker in response.lower()
            for marker in ("because", "therefore", "for example", "first", "second", "finally")
        )
        structure = min(1.0, 0.2 * structure_markers + 0.2 * response.count("\n"))
        repetition = max(term_counts.values()) / max(1, len(response_terms))
        repetition_penalty = max(0.0, repetition - 0.18) * 1.5

        score = 0.15 + 0.35 * alignment + 0.25 * substance + 0.20 * structure - repetition_penalty
        score = min(0.85, max(0.0, score))
        return VerificationResult(score, 3, "open_ended_semantic_heuristic")

    def score(self, task_type: str, **kwargs) -> VerificationResult:
        if task_type == "code":
            return self.verify_code(kwargs.get("code", ""), kwargs.get("test_code", ""))
        if task_type == "math":
            return self.verify_math(kwargs.get("expression", ""), kwargs.get("expected", ""))
        if task_type == "qiskit" and verify_qiskit is not None:
            return self._from_domain_result(verify_qiskit(kwargs.get("qasm", kwargs.get("code", "")), kwargs.get("target_topology", "linear_nn")))
        if task_type == "rdkit" and verify_rdkit is not None:
            return self._from_domain_result(verify_rdkit(kwargs.get("smiles", kwargs.get("molecule", kwargs.get("response", "")))))
        if task_type in {"verilog", "verilator"} and verify_verilog is not None:
            return self._from_domain_result(verify_verilog(kwargs.get("verilog", kwargs.get("code", "")), kwargs.get("testbench", kwargs.get("test_code", ""))))
        if task_type == "constraint_json" and verify_constraint_json is not None:
            return self._from_domain_result(verify_constraint_json(kwargs.get("constraints", {}), kwargs.get("candidate", {})))
        if task_type == "citation_grounding" and verify_citation_grounding is not None:
            return self._from_domain_result(verify_citation_grounding(kwargs.get("claim", kwargs.get("response", "")), kwargs.get("epg_path"), kwargs.get("memory_nodes")))
        if task_type == "file_state":
            return self.verify_file_state(kwargs.get("check_fn", lambda: False))
        if task_type == "instruction":
            return self.verify_instruction(kwargs.get("response", ""), kwargs.get("pattern", ".*"))
        return self.verify_open_ended(kwargs.get("task", ""), kwargs.get("response", ""))

    def _from_domain_result(self, result) -> VerificationResult:
        return VerificationResult(
            float(getattr(result, "score", 0.0)),
            1,
            str(getattr(result, "reason", "")),
            str(getattr(result, "stdout", "")),
            str(getattr(result, "stderr", "")),
            int(getattr(result, "return_code", 0)),
        )
