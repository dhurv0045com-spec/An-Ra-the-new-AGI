"""Fail-closed lifecycle state for a resumable training runner.

The controller owns no model tensors and performs no updates.  It records the
small state machine around an update/checkpoint boundary so a worker failure
cannot be mistaken for a completed run or advance the parent pointer before a
checkpoint is durable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum


RUNNER_SCHEMA = "anra-v5-runner-state/v1"


def _assert_sha256(name: str, value: str | None) -> None:
    if value is None:
        return
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")


class RunStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    CHECKPOINTING = "checkpointing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class RunnerState:
    schema: str
    status: RunStatus
    target_update: int
    committed_update: int
    last_checkpoint_sha256: str | None
    pending_parent_sha256: str | None
    pending_update: int | None
    failure_code: str | None

    @classmethod
    def initial(cls, *, target_update: int) -> "RunnerState":
        state = cls(
            schema=RUNNER_SCHEMA,
            status=RunStatus.CREATED,
            target_update=target_update,
            committed_update=0,
            last_checkpoint_sha256=None,
            pending_parent_sha256=None,
            pending_update=None,
            failure_code=None,
        )
        state.assert_valid()
        return state

    def assert_valid(self) -> None:
        if self.schema != RUNNER_SCHEMA:
            raise ValueError("unsupported runner schema")
        if self.target_update <= 0 or self.committed_update < 0:
            raise ValueError("runner update counters are invalid")
        if self.committed_update > self.target_update:
            raise ValueError("runner committed beyond target")
        _assert_sha256("last checkpoint", self.last_checkpoint_sha256)
        _assert_sha256("pending parent", self.pending_parent_sha256)
        if self.status is RunStatus.CHECKPOINTING:
            if self.pending_update != self.committed_update + 1:
                raise ValueError("checkpointing must target exactly the next update")
            if self.pending_parent_sha256 != self.last_checkpoint_sha256:
                raise ValueError("pending checkpoint parent is not the committed checkpoint")
            if self.failure_code is not None:
                raise ValueError("checkpointing cannot carry a failure")
        elif self.pending_update is not None or self.pending_parent_sha256 is not None:
            raise ValueError("pending checkpoint metadata requires checkpointing status")
        if self.status is RunStatus.CREATED:
            if self.committed_update or self.last_checkpoint_sha256 or self.failure_code:
                raise ValueError("created runner must have no committed work or failure")
        if self.status is RunStatus.COMPLETED:
            if self.committed_update != self.target_update or not self.last_checkpoint_sha256:
                raise ValueError("completed runner lacks a committed target checkpoint")
            if self.failure_code:
                raise ValueError("completed runner cannot carry a failure")
        if self.status is RunStatus.FAILED and not self.failure_code:
            raise ValueError("failed runner requires a failure code")
        if self.failure_code and self.status is not RunStatus.FAILED:
            raise ValueError("failure code requires failed status")

    def canonical(self) -> dict[str, object]:
        self.assert_valid()
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "RunnerState":
        expected = {
            "schema", "status", "target_update", "committed_update",
            "last_checkpoint_sha256", "pending_parent_sha256", "pending_update",
            "failure_code",
        }
        if set(value) != expected:
            raise ValueError("runner fields do not match schema")
        state = cls(
            schema=str(value["schema"]),
            status=RunStatus(str(value["status"])),
            target_update=int(value["target_update"]),
            committed_update=int(value["committed_update"]),
            last_checkpoint_sha256=value["last_checkpoint_sha256"],
            pending_parent_sha256=value["pending_parent_sha256"],
            pending_update=None if value["pending_update"] is None else int(value["pending_update"]),
            failure_code=value["failure_code"],
        )
        state.assert_valid()
        return state


class RunController:
    """Single-writer lifecycle guard around trainer/checkpoint callbacks."""

    def __init__(self, *, target_update: int) -> None:
        self.state = RunnerState.initial(target_update=target_update)

    def start(self) -> RunnerState:
        if self.state.status is not RunStatus.CREATED:
            raise ValueError("only a created runner can start")
        self.state = RunnerState(
            schema=self.state.schema,
            status=RunStatus.RUNNING,
            target_update=self.state.target_update,
            committed_update=self.state.committed_update,
            last_checkpoint_sha256=self.state.last_checkpoint_sha256,
            pending_parent_sha256=None,
            pending_update=None,
            failure_code=None,
        )
        self.state.assert_valid()
        return self.state

    def begin_checkpoint(self, *, update: int) -> RunnerState:
        if self.state.status is not RunStatus.RUNNING:
            raise ValueError("checkpoint can begin only from running status")
        if update != self.state.committed_update + 1:
            raise ValueError("checkpoint update is not the next uncommitted update")
        if update > self.state.target_update:
            raise ValueError("checkpoint update exceeds the run target")
        self.state = RunnerState(
            schema=self.state.schema,
            status=RunStatus.CHECKPOINTING,
            target_update=self.state.target_update,
            committed_update=self.state.committed_update,
            last_checkpoint_sha256=self.state.last_checkpoint_sha256,
            pending_parent_sha256=self.state.last_checkpoint_sha256,
            pending_update=update,
            failure_code=None,
        )
        self.state.assert_valid()
        return self.state

    def commit_checkpoint(self, *, checkpoint_sha256: str) -> RunnerState:
        if self.state.status is not RunStatus.CHECKPOINTING:
            raise ValueError("only a checkpointing runner can commit")
        _assert_sha256("checkpoint", checkpoint_sha256)
        self.state = RunnerState(
            schema=self.state.schema,
            status=RunStatus.RUNNING,
            target_update=self.state.target_update,
            committed_update=self.state.pending_update or 0,
            last_checkpoint_sha256=checkpoint_sha256,
            pending_parent_sha256=None,
            pending_update=None,
            failure_code=None,
        )
        self.state.assert_valid()
        return self.state

    def complete(self) -> RunnerState:
        if self.state.status is not RunStatus.RUNNING:
            raise ValueError("only a running runner can complete")
        if self.state.committed_update != self.state.target_update:
            raise ValueError("completion requires a committed target update")
        self.state = RunnerState(
            schema=self.state.schema,
            status=RunStatus.COMPLETED,
            target_update=self.state.target_update,
            committed_update=self.state.committed_update,
            last_checkpoint_sha256=self.state.last_checkpoint_sha256,
            pending_parent_sha256=None,
            pending_update=None,
            failure_code=None,
        )
        self.state.assert_valid()
        return self.state

    def fail(self, *, code: str) -> RunnerState:
        if self.state.status in {RunStatus.COMPLETED, RunStatus.FAILED}:
            raise ValueError("terminal runner cannot fail again")
        if not code or any(character.isspace() for character in code):
            raise ValueError("failure code must be a compact nonempty identity")
        self.state = RunnerState(
            schema=self.state.schema,
            status=RunStatus.FAILED,
            target_update=self.state.target_update,
            committed_update=self.state.committed_update,
            last_checkpoint_sha256=self.state.last_checkpoint_sha256,
            pending_parent_sha256=None,
            pending_update=None,
            failure_code=code,
        )
        self.state.assert_valid()
        return self.state

    def recover(self) -> RunnerState:
        if self.state.status is not RunStatus.FAILED:
            raise ValueError("only a failed runner can recover")
        self.state = RunnerState(
            schema=self.state.schema,
            status=RunStatus.RUNNING,
            target_update=self.state.target_update,
            committed_update=self.state.committed_update,
            last_checkpoint_sha256=self.state.last_checkpoint_sha256,
            pending_parent_sha256=None,
            pending_update=None,
            failure_code=None,
        )
        self.state.assert_valid()
        return self.state
