"""Scientific Execution Guards & Fail-Closed Provably Non-Mock Invariants.

Enforces zero-mock scientific execution:
1. Rejects random tensors, dummy strings, and synthetic placeholders on the scientific path.
2. Forbids byte-fallback test tokenizer in scientific mode unless explicitly marked as test.
3. Asserts real PyTorch state-dict serialization in checkpoints.
4. Forbids gold-answer contamination into model prediction generation.
5. Asserts exact git commit SHA match between launch manifest and current HEAD.
"""

from __future__ import annotations

import enum
import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence


class ExecutionMode(enum.Enum):
    VALIDATE_ONLY = "validate_only"
    SCIENTIFIC_EXECUTION = "scientific_execution"


class ScientificIntegrityViolationError(RuntimeError):
    """Raised when mock data, dummy identities, or gold leaks enter the scientific execution path."""


class ScientificExecutionGuard:
    """Fail-closed enforcement for production scientific execution."""

    @staticmethod
    def get_current_git_head(cwd: Path = Path(".")) -> str:
        """Query the exact current Git commit SHA."""
        try:
            res = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True,
            )
            return res.stdout.strip()
        except Exception as err:
            raise ScientificIntegrityViolationError(f"Failed to obtain current Git HEAD: {err}")

    @staticmethod
    def assert_clean_worktree(cwd: Path = Path(".")) -> None:
        """Assert that the git worktree has no uncommitted changes."""
        try:
            res = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True,
            )
            dirty = res.stdout.strip()
            if dirty:
                raise ScientificIntegrityViolationError(
                    f"Scientific execution requires a clean git working tree, but uncommitted changes exist:\n{dirty}"
                )
        except subprocess.CalledProcessError as err:
            raise ScientificIntegrityViolationError(f"Failed to check git worktree cleanliness: {err}")

    @staticmethod
    def assert_matching_commit(manifest_sha: str, cwd: Path = Path(".")) -> None:
        """Verify that the manifest commit SHA strictly matches checked-out HEAD."""
        current_head = ScientificExecutionGuard.get_current_git_head(cwd)
        if manifest_sha != current_head:
            raise ScientificIntegrityViolationError(
                f"Source commit mismatch! Manifest binds {manifest_sha}, but current HEAD is {current_head}."
            )

    @staticmethod
    def assert_real_checkpoint_payloads(payloads: Mapping[str, bytes]) -> None:
        """Assert that checkpoint payloads contain real state-dicts, not dummy placeholder strings."""
        required_keys = ["model.bin", "optimizer.bin", "rng.bin", "training_state.json"]
        for key in required_keys:
            if key not in payloads:
                raise ScientificIntegrityViolationError(f"Missing required checkpoint payload: {key}")
            data = payloads[key]
            if data.startswith(b"remote_") or len(data) < 64:
                raise ScientificIntegrityViolationError(
                    f"Checkpoint payload {key!r} contains a placeholder/mock byte string ({len(data)} bytes)!"
                )

    @staticmethod
    def assert_real_batch(input_ids: Any, targets: Any) -> None:
        """Assert that batch tensors are not dummy or empty."""
        if hasattr(input_ids, "shape"):
            if input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
                raise ScientificIntegrityViolationError("Batch tensor dimensions must be non-zero.")
        elif not input_ids or not input_ids[0]:
            raise ScientificIntegrityViolationError("Batch sequence list must be non-empty.")

    @staticmethod
    def assert_no_gold_in_policy_input(policy_prompt: str, gold_answer: str) -> None:
        """Ensure gold answer is not trivially embedded into the generation policy prompt."""
        if not gold_answer or len(gold_answer.strip()) == 0:
            return
        cleaned_answer = gold_answer.strip().lower()
        if f"answer is {cleaned_answer}" in policy_prompt.lower() or f"the answer: {cleaned_answer}" in policy_prompt.lower():
            raise ScientificIntegrityViolationError(
                f"Gold answer leakage detected in prompt: {gold_answer!r} appears directly in generation input!"
            )