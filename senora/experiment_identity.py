"""Exact experiment identity, cryptographic lineage, and fail-closed promotion gates."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping


SCHEMA = "senora-experiment-identity/v1"


def _assert_sha(name: str, value: str | None, length: int = 64) -> None:
    if value is None:
        raise ValueError(f"{name} is missing")
    if len(value) != length or any(c not in "0123456789abcdef" for c in value.lower()):
        raise ValueError(f"{name} must be a {length}-character lowercase hex SHA string, got {value!r}")


def _canonical_json(data: dict[str, Any]) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True, slots=True)
class ExperimentIdentity:
    """Immutable, fully-bound specification of an experimental run."""

    schema: str
    experiment_id: str
    source_commit_sha: str
    model_spec_sha256: str
    model_constructor_sha256: str
    tokenizer_artifact_sha256: str
    corpus_manifest_sha256: str
    data_manifest_sha256: str
    pack_manifest_sha256: str
    generator_version: str
    split_identities: Mapping[str, str]
    optimizer_spec: Mapping[str, Any]
    schedule_spec: Mapping[str, Any]
    precision: str
    token_budget: int
    tokens_per_update: int
    random_seeds: tuple[int, ...]
    evaluator_spec: Mapping[str, Any]
    scorer_firewall_status: str
    statistical_protocol: Mapping[str, Any]
    promotion_criteria: Mapping[str, Any]
    abort_criteria: Mapping[str, Any]

    def assert_valid(self) -> None:
        if self.schema != SCHEMA:
            raise ValueError(f"unsupported schema {self.schema!r}, expected {SCHEMA!r}")
        if not self.experiment_id:
            raise ValueError("experiment_id cannot be empty")
        _assert_sha("source_commit_sha", self.source_commit_sha, length=40)
        _assert_sha("model_spec_sha256", self.model_spec_sha256)
        _assert_sha("model_constructor_sha256", self.model_constructor_sha256)
        _assert_sha("tokenizer_artifact_sha256", self.tokenizer_artifact_sha256)
        _assert_sha("corpus_manifest_sha256", self.corpus_manifest_sha256)
        _assert_sha("data_manifest_sha256", self.data_manifest_sha256)
        _assert_sha("pack_manifest_sha256", self.pack_manifest_sha256)

        if not self.generator_version:
            raise ValueError("generator_version cannot be empty")
        for split, split_hash in self.split_identities.items():
            _assert_sha(f"split_{split}_sha256", split_hash)

        if self.token_budget <= 0 or self.tokens_per_update <= 0:
            raise ValueError("token_budget and tokens_per_update must be positive integers")
        if self.token_budget < self.tokens_per_update:
            raise ValueError("token_budget cannot be smaller than tokens_per_update")
        if not self.random_seeds:
            raise ValueError("random_seeds must contain at least one seed")

        if not self.scorer_firewall_status:
            raise ValueError("scorer_firewall_status must be explicitly declared")

        required_abort_keys = {
            "max_loss_regression_fraction",
            "fail_on_nan_loss",
            "fail_on_gradient_explosion",
            "fail_on_stagnation",
        }
        if not required_abort_keys.issubset(self.abort_criteria.keys()):
            raise ValueError(f"abort_criteria missing required keys: {required_abort_keys - set(self.abort_criteria.keys())}")

    def is_run_authorized(self) -> tuple[bool, list[str]]:
        """Fail-closed preflight check to determine if compute execution is authorized."""
        blockers: list[str] = []
        try:
            self.assert_valid()
        except ValueError as exc:
            blockers.append(f"invalid identity contract: {exc}")

        # Check for placeholder hashes (e.g. all 0s or all fs)
        for name in [
            "tokenizer_artifact_sha256",
            "corpus_manifest_sha256",
            "data_manifest_sha256",
            "pack_manifest_sha256",
        ]:
            val = getattr(self, name)
            if val in {"0" * 64, "f" * 64, "a" * 64}:
                blockers.append(f"{name} is a placeholder hash ({val[:8]}...), requires genuine certified artifact")

        # Check scorer firewall status
        if self.scorer_firewall_status not in {"PASSED", "PASS_DEVELOPMENT_POLICY", "BYPASS_CANDIDATE_LOGPROB_RAW_CORE_ONLY"}:
            blockers.append(
                f"scorer_firewall_status={self.scorer_firewall_status!r} blocks candidate logprob scoring. "
                "Must use BYPASS_CANDIDATE_LOGPROB_RAW_CORE_ONLY or wait for upstream scorer firewall PASS."
            )

        return (len(blockers) == 0, blockers)

    def canonical(self) -> dict[str, Any]:
        self.assert_valid()
        data = asdict(self)
        data["random_seeds"] = list(self.random_seeds)
        return data

    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.canonical())).hexdigest()