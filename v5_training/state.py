"""Fail-closed state transition for token-indexed V5 training.

This module deliberately contains no framework code.  It defines the state a
real trainer must advance atomically after one successful optimizer update.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Mapping


STATE_SCHEMA = "anra-v5-training-state/v1"
IDENTITY_SCHEMA = "anra-v5-identity-bindings/v1"
CURSOR_SCHEMA = "anra-v5-pack-cursor/v1"


def _assert_sha256(name: str, value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be lowercase SHA-256")


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def next_update_tokens(*, token_budget: int, cumulative_tokens: int, tokens_per_update: int) -> int:
    """Return the exact next update size, including a final partial update."""

    if token_budget <= 0 or tokens_per_update <= 0:
        raise ValueError("token budget and tokens per update must be positive")
    if not 0 <= cumulative_tokens <= token_budget:
        raise ValueError("cumulative tokens must lie inside the run budget")
    return min(tokens_per_update, token_budget - cumulative_tokens)


@dataclass(frozen=True, slots=True)
class IdentityBindings:
    schema: str
    source_commit: str
    model_spec_sha256: str
    tokenizer_sha256: str
    data_manifest_sha256: str
    pack_manifest_sha256: str
    run_spec_sha256: str
    optimizer_spec_sha256: str
    schedule_spec_sha256: str
    curriculum_spec_sha256: str

    def assert_valid(self) -> None:
        if self.schema != IDENTITY_SCHEMA:
            raise ValueError("unsupported identity-binding schema")
        if len(self.source_commit) != 40 or any(c not in "0123456789abcdef" for c in self.source_commit):
            raise ValueError("source commit must be a full lowercase Git SHA-1")
        for name, value in asdict(self).items():
            if name.endswith("_sha256"):
                _assert_sha256(name, value)

    def sha256(self) -> str:
        self.assert_valid()
        return hashlib.sha256(_canonical_json(asdict(self))).hexdigest()


@dataclass(frozen=True, slots=True)
class CursorState:
    schema: str
    pack_manifest_sha256: str
    shard_ordinal: int
    sequence_ordinal: int
    token_offset: int

    def assert_valid(self) -> None:
        if self.schema != CURSOR_SCHEMA:
            raise ValueError("unsupported cursor schema")
        _assert_sha256("cursor pack manifest", self.pack_manifest_sha256)
        if min(self.shard_ordinal, self.sequence_ordinal, self.token_offset) < 0:
            raise ValueError("cursor coordinates cannot be negative")


@dataclass(frozen=True, slots=True)
class TrainingState:
    schema: str
    lineage_id: str
    generation: int
    global_update: int
    cumulative_tokens: int
    token_budget: int
    tokens_per_update: int
    tokens_by_source: Mapping[str, int]
    optimizer_step_max: int
    schedule_tokens: int
    cursor: CursorState
    rng_state_sha256: str
    curriculum_phase: str
    identities: IdentityBindings
    parent_checkpoint_sha256: str | None

    @classmethod
    def initial(
        cls,
        *,
        lineage_id: str,
        token_budget: int,
        tokens_per_update: int,
        cursor: CursorState,
        rng_state_sha256: str,
        curriculum_phase: str,
        identities: IdentityBindings,
    ) -> "TrainingState":
        state = cls(
            schema=STATE_SCHEMA,
            lineage_id=lineage_id,
            generation=0,
            global_update=0,
            cumulative_tokens=0,
            token_budget=token_budget,
            tokens_per_update=tokens_per_update,
            tokens_by_source={},
            optimizer_step_max=0,
            schedule_tokens=0,
            cursor=cursor,
            rng_state_sha256=rng_state_sha256,
            curriculum_phase=curriculum_phase,
            identities=identities,
            parent_checkpoint_sha256=None,
        )
        state.assert_valid()
        return state

    def assert_valid(self) -> None:
        if self.schema != STATE_SCHEMA:
            raise ValueError("unsupported training-state schema")
        if not self.lineage_id or not self.curriculum_phase:
            raise ValueError("lineage and curriculum phase are required")
        if min(self.generation, self.global_update, self.cumulative_tokens) < 0:
            raise ValueError("training counters cannot be negative")
        if self.token_budget <= 0 or self.tokens_per_update <= 0:
            raise ValueError("run token counts must be positive")
        if self.cumulative_tokens > self.token_budget:
            raise ValueError("cumulative tokens exceed the frozen budget")
        if self.optimizer_step_max != self.global_update:
            raise ValueError("optimizer step must equal global update")
        if self.schedule_tokens != self.cumulative_tokens:
            raise ValueError("schedule must be indexed by cumulative tokens")
        if any(not name or value < 0 for name, value in self.tokens_by_source.items()):
            raise ValueError("source ledgers require names and nonnegative counts")
        if sum(self.tokens_by_source.values()) != self.cumulative_tokens:
            raise ValueError("source ledger must equal cumulative tokens")
        _assert_sha256("RNG state", self.rng_state_sha256)
        if self.parent_checkpoint_sha256 is not None:
            _assert_sha256("parent checkpoint", self.parent_checkpoint_sha256)
        self.identities.assert_valid()
        self.cursor.assert_valid()
        if self.cursor.pack_manifest_sha256 != self.identities.pack_manifest_sha256:
            raise ValueError("cursor and frozen pack identities disagree")
        if self.global_update == 0:
            if self.generation != 0 or self.cumulative_tokens != 0 or self.tokens_by_source:
                raise ValueError("initial state must start every lifetime counter at zero")
            if self.parent_checkpoint_sha256 is not None:
                raise ValueError("initial state cannot have a parent checkpoint")
        elif self.generation != self.global_update:
            raise ValueError("this single-update transaction requires generation == global update")

    @property
    def complete(self) -> bool:
        return self.cumulative_tokens == self.token_budget

    def canonical(self) -> dict[str, object]:
        self.assert_valid()
        return asdict(self)

    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.canonical())).hexdigest()

    def advance(
        self,
        *,
        tokens_by_source: Mapping[str, int],
        cursor: CursorState,
        rng_state_sha256: str,
        curriculum_phase: str | None = None,
        parent_checkpoint_sha256: str | None,
    ) -> "TrainingState":
        """Advance exactly once after a completed optimizer update."""

        self.assert_valid()
        expected = next_update_tokens(
            token_budget=self.token_budget,
            cumulative_tokens=self.cumulative_tokens,
            tokens_per_update=self.tokens_per_update,
        )
        consumed = sum(tokens_by_source.values())
        if expected == 0:
            raise ValueError("a completed run cannot advance")
        if consumed != expected or any(value < 0 for value in tokens_by_source.values()):
            raise ValueError(f"update must consume exactly {expected} tokens")
        cursor.assert_valid()
        if cursor.pack_manifest_sha256 != self.cursor.pack_manifest_sha256:
            raise ValueError("an update cannot migrate to an unbound pack")
        if cursor == self.cursor:
            raise ValueError("a successful update must advance the cursor")
        if parent_checkpoint_sha256 is not None:
            _assert_sha256("parent checkpoint", parent_checkpoint_sha256)
        ledger = dict(self.tokens_by_source)
        for source, count in tokens_by_source.items():
            if not source:
                raise ValueError("source name cannot be empty")
            ledger[source] = ledger.get(source, 0) + count
        updated = TrainingState(
            schema=self.schema,
            lineage_id=self.lineage_id,
            generation=self.generation + 1,
            global_update=self.global_update + 1,
            cumulative_tokens=self.cumulative_tokens + consumed,
            token_budget=self.token_budget,
            tokens_per_update=self.tokens_per_update,
            tokens_by_source=ledger,
            optimizer_step_max=self.optimizer_step_max + 1,
            schedule_tokens=self.schedule_tokens + consumed,
            cursor=cursor,
            rng_state_sha256=rng_state_sha256,
            curriculum_phase=curriculum_phase or self.curriculum_phase,
            identities=self.identities,
            parent_checkpoint_sha256=parent_checkpoint_sha256,
        )
        updated.assert_valid()
        return updated

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "TrainingState":
        expected = {
            "schema", "lineage_id", "generation", "global_update", "cumulative_tokens",
            "token_budget", "tokens_per_update", "tokens_by_source", "optimizer_step_max",
            "schedule_tokens", "cursor", "rng_state_sha256", "curriculum_phase",
            "identities", "parent_checkpoint_sha256",
        }
        if set(value) != expected:
            raise ValueError("training-state fields do not match schema")
        cursor = CursorState(**value["cursor"])
        identities = IdentityBindings(**value["identities"])
        state = cls(
            **{key: item for key, item in value.items() if key not in {"cursor", "identities"}},
            cursor=cursor,
            identities=identities,
        )
        state.assert_valid()
        return state
