from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from anra.anra_paths import WORKSPACE_DIR
from execution.sandbox import CodeSandbox
from verification import DEFAULT_VERIFIER_REGISTRY


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    return_code: int
    limit_reason: str = ""


@dataclass
class VerificationResult:
    score: float
    tier: int
    reason: str
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0


class VerifierHierarchy:
    def __init__(self, workspace: str | Path | None = None) -> None:
        self.workspace = Path(workspace) if workspace is not None else Path(WORKSPACE_DIR)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.sandbox = CodeSandbox(self.workspace, timeout=5)

    def _safe_exec(self, code: str) -> ExecutionResult:
        result = self.sandbox.execute(code)
        return ExecutionResult(
            result.stdout,
            result.stderr,
            result.return_code,
            result.limit_reason,
        )

    def verify_code(self, code: str, test_code: str = "") -> VerificationResult:
        joined = code if not test_code else f"{code}\n\n{test_code}\n"
        result = self._safe_exec(joined)
        if result.return_code == 124:
            return VerificationResult(0.0, 1, "timeout", result.stdout, result.stderr, 124)
        if result.limit_reason:
            return VerificationResult(
                0.0,
                1,
                f"sandbox_{result.limit_reason}",
                result.stdout,
                result.stderr,
                result.return_code,
            )
        if result.return_code == 0 and test_code:
            return VerificationResult(1.0, 1, "tests_passed", result.stdout, result.stderr, 0)
        if result.return_code == 0:
            return VerificationResult(0.7, 1, "ran_without_tests", result.stdout, result.stderr, 0)
        if "SyntaxError" in result.stderr or "Traceback" in result.stderr:
            return VerificationResult(
                0.2, 1, "runtime_or_syntax_error", result.stdout, result.stderr, result.return_code
            )
        return VerificationResult(0.0, 1, "crash", result.stdout, result.stderr, result.return_code)

    def verify_math(self, expression: str, expected: str) -> VerificationResult:
        try:
            import sympy as sp

            lhs = sp.sympify(expression)
            rhs = sp.sympify(expected)
            ok = bool(sp.simplify(lhs - rhs) == 0)
            return VerificationResult(
                1.0 if ok else 0.0, 1, "equivalent" if ok else "not_equivalent"
            )
        except Exception as exc:
            return VerificationResult(0.0, 1, f"math_error: {exc}")

    def verify_file_state(self, check_fn: Callable[[], object]) -> VerificationResult:
        try:
            ok = bool(check_fn())
            return VerificationResult(
                1.0 if ok else 0.0, 1, "file_state_ok" if ok else "file_state_mismatch"
            )
        except Exception as exc:
            return VerificationResult(0.0, 1, f"file_state_error: {exc}")

    def verify_instruction(self, response: str, pattern: str) -> VerificationResult:
        matched = re.search(pattern, response or "") is not None
        length_ok = len((response or "").strip()) >= 8
        score = 0.8 if matched and length_ok else 0.3 if matched else 0.0
        return VerificationResult(score, 2, "heuristic_instruction")

    def verify_exact(self, response: str, expected: str, label: str) -> VerificationResult:
        actual = re.sub(r"\s+", " ", response.strip().lower())
        reference = re.sub(r"\s+", " ", expected.strip().lower())
        passed = bool(reference) and reference in actual
        return VerificationResult(
            1.0 if passed else 0.0, 1, f"{label}_{'verified' if passed else 'mismatch'}"
        )

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

    def score(self, task_type: str, **kwargs: object) -> VerificationResult:
        """Run a verifier and publish its outcome to the shared experience ledger."""
        registered_name = task_type if task_type in DEFAULT_VERIFIER_REGISTRY else "open_ended"
        payload = dict(kwargs)
        if registered_name == "open_ended":
            payload["requested_verifier"] = task_type
        return DEFAULT_VERIFIER_REGISTRY.verify(
            registered_name,
            payload,
            context=self,
        )
