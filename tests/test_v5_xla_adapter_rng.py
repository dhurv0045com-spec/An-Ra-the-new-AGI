from __future__ import annotations

import base64
import json
import copy
from types import SimpleNamespace

import pytest
import torch

import v5_training.xla_adapter as xla_adapter


class _FakeXlaRuntime:
    def __init__(self) -> None:
        self.state = 41

    def get_rng_state(self, *, device: str) -> int:
        assert device == "cpu"
        return self.state

    def set_rng_state(self, state: int, *, device: str) -> None:
        assert device == "cpu"
        self.state = state


class _FailingRestoreXlaRuntime(_FakeXlaRuntime):
    def set_rng_state(self, state: int, *, device: str) -> None:
        assert device == "cpu"
        if state == 41:
            self.state = state
            raise ValueError("simulated incompatible target RNG state")
        self.state = state


def _backend() -> xla_adapter.XLAReplicatedBackend:
    backend = object.__new__(xla_adapter.XLAReplicatedBackend)
    backend.replica_backend = SimpleNamespace(model=torch.nn.Linear(2, 2))
    backend.torch = torch
    backend.replicas = 1
    return backend


def test_xla_rng_payload_round_trips_cpu_and_rank_local_generator(monkeypatch) -> None:
    fake_xm = _FakeXlaRuntime()
    monkeypatch.setattr(xla_adapter, "_load_xla", lambda: (object(), fake_xm))
    backend = _backend()
    torch.manual_seed(173)
    payload = backend.capture_rng_state()
    expected_cpu_state = torch.get_rng_state().clone()
    expected_sha256 = backend.rng_state_sha256()

    torch.rand(8)
    fake_xm.state = 999
    backend.restore_rng_state(payload)

    assert torch.equal(torch.get_rng_state(), expected_cpu_state)
    assert fake_xm.state == 41
    assert backend.rng_state_sha256() == expected_sha256


def test_xla_rng_restore_rejects_wrong_device_before_mutating_cpu_rng(monkeypatch) -> None:
    fake_xm = _FakeXlaRuntime()
    monkeypatch.setattr(xla_adapter, "_load_xla", lambda: (object(), fake_xm))
    backend = _backend()
    torch.manual_seed(239)
    before = torch.get_rng_state().clone()
    payload = json.dumps({
        "schema": xla_adapter.XLA_RNG_STATE_SCHEMA,
        "cpu_state_base64": base64.b64encode(
            torch.get_rng_state().numpy().tobytes()
        ).decode("ascii"),
        "xla_state": 41,
        "xla_device": "xla:1",
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")

    with pytest.raises(ValueError, match="device differs"):
        backend.restore_rng_state(payload)

    assert torch.equal(torch.get_rng_state(), before)


def test_xla_rng_restore_rolls_back_cpu_when_xla_setter_fails(monkeypatch) -> None:
    fake_xm = _FailingRestoreXlaRuntime()
    fake_xm.state = 73
    monkeypatch.setattr(xla_adapter, "_load_xla", lambda: (object(), fake_xm))
    backend = _backend()
    torch.manual_seed(311)
    before_cpu = torch.get_rng_state().clone()
    payload = json.dumps({
        "schema": xla_adapter.XLA_RNG_STATE_SCHEMA,
        "cpu_state_base64": base64.b64encode(
            torch.get_rng_state().numpy().tobytes()
        ).decode("ascii"),
        "xla_state": 41,
        "xla_device": "cpu",
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")

    with pytest.raises(RuntimeError, match="previous CPU and XLA states were restored"):
        backend.restore_rng_state(payload)

    assert torch.equal(torch.get_rng_state(), before_cpu)
    assert fake_xm.state == 73


class _FakeVoteRuntime:
    def __init__(self, rank: int = 0) -> None:
        self.rank = rank

    def global_ordinal(self) -> int:
        return self.rank


class _FakeVoteXm:
    def __init__(self, transform=None) -> None:
        self.transform = transform or (lambda claim: [claim])
        self.tags: list[str] = []
        self.all_reduce_calls = 0
        self.all_reduce_scales: list[float | None] = []
        self.mark_step_calls = 0
        self.wait_device_ops_calls = 0
        self.wait_error: Exception | None = None
        self.events: list[str] = []

    def mark_step(self):
        self.mark_step_calls += 1
        self.events.append("mark_step")

    def wait_device_ops(self):
        self.wait_device_ops_calls += 1
        self.events.append("wait_device_ops")
        if self.wait_error is not None:
            raise self.wait_error

    def mesh_reduce(self, tag, claim, reduce_fn):
        self.tags.append(tag)
        self.events.append("mesh_reduce")
        gathered = self.transform(claim)
        return reduce_fn(gathered)

    REDUCE_SUM = "sum"

    def all_reduce(self, operation, gradients, *, scale=None):
        self.all_reduce_calls += 1
        self.all_reduce_scales.append(scale)
        assert operation == self.REDUCE_SUM
        return gradients


def _vote_backend(monkeypatch, *, rank=0, replicas=3, transform=None):
    backend = object.__new__(xla_adapter.XLAReplicatedBackend)
    backend.replicas = replicas
    backend._status_vote_sequence = 0
    runtime = _FakeVoteRuntime(rank)
    xm = _FakeVoteXm(transform)
    monkeypatch.setattr(xla_adapter, "_load_xla", lambda: (runtime, xm))
    return backend, xm


def _claims_for(claim, *, ranks=(0, 1, 2), failures=()):
    claims = []
    for rank in ranks:
        remote = copy.deepcopy(claim)
        remote["rank"] = rank
        if rank in failures:
            remote["error"] = {"type": "builtins.ValueError", "message": "local boom"}
        claims.append(remote)
    return claims


def test_xla_status_vote_all_pass_is_rank_ordered_and_uses_sequence_tags(monkeypatch):
    backend, xm = _vote_backend(
        monkeypatch, transform=lambda claim: _claims_for(claim),
    )

    first = backend.vote_status(stage="forward")
    second = backend.vote_status(stage="backward")

    assert first["status"] == second["status"] == "passed"
    assert [item["rank"] for item in first["claims"]] == [0, 1, 2]
    assert first["sequence"] == 0 and second["sequence"] == 1
    assert xm.tags == [
        f"{xla_adapter.XLA_STATUS_VOTE_SCHEMA}:00000000000000000000",
        f"{xla_adapter.XLA_STATUS_VOTE_SCHEMA}:00000000000000000001",
    ]


def test_xla_status_vote_one_rank_failure_aborts_with_deterministic_detail(monkeypatch):
    backend, _ = _vote_backend(
        monkeypatch,
        transform=lambda claim: _claims_for(claim, failures=(1,)),
    )

    with pytest.raises(RuntimeError, match=r"failed on 1 of 3 ranks \(rank 1: builtins.ValueError: local boom\)"):
        backend.vote_status(stage="backward")


def test_xla_local_compute_stage_votes_before_returning(monkeypatch):
    backend, xm = _vote_backend(
        monkeypatch, transform=lambda claim: _claims_for(claim),
    )

    result = backend.run_local_stage(
        stage="local_update", callback=lambda: {"gradients_ready": True},
    )

    assert result == {"gradients_ready": True}
    assert len(xm.tags) == 1
    assert xm.mark_step_calls == 1
    assert xm.wait_device_ops_calls == 1
    assert xm.events == ["mark_step", "wait_device_ops", "mesh_reduce"]


def test_xla_local_compute_failure_is_shared_before_any_later_collective(monkeypatch):
    def only_local_rank_failed(claim):
        claims = _claims_for(claim, ranks=(0, 1, 2))
        for remote in claims:
            if remote["rank"] != 0:
                remote["error"] = None
        return claims

    backend, xm = _vote_backend(
        monkeypatch, transform=only_local_rank_failed,
    )

    def fail_local_compute():
        raise ValueError("local forward/backward failed")

    with pytest.raises(RuntimeError, match="status vote failed on 1 of 3 ranks"):
        backend.run_local_stage(stage="local_update", callback=fail_local_compute)

    assert len(xm.tags) == 1
    assert xm.mark_step_calls == 1
    assert xm.wait_device_ops_calls == 1
    assert xm.events == ["mark_step", "wait_device_ops", "mesh_reduce"]


def test_xla_device_wait_failure_is_shared_before_gradient_collective(monkeypatch):
    def only_local_rank_failed(claim):
        claims = _claims_for(claim, ranks=(0, 1, 2))
        for remote in claims:
            if remote["rank"] != 0:
                remote["error"] = None
        return claims

    backend, xm = _vote_backend(
        monkeypatch, transform=only_local_rank_failed,
    )
    xm.wait_error = RuntimeError("asynchronous XLA backward failed")

    with pytest.raises(RuntimeError, match="status vote failed on 1 of 3 ranks"):
        backend.run_local_stage(
            stage="local_update", callback=lambda: "lazy graph queued",
        )

    assert xm.events == ["mark_step", "wait_device_ops", "mesh_reduce"]
    assert xm.all_reduce_calls == 0


def test_xla_precollective_vote_rejects_rank_gradient_layout_mismatch(monkeypatch):
    def mismatch_one_rank(claim):
        claims = _claims_for(claim, ranks=(0, 1))
        claims[1]["evidence_sha256"] = "f" * 64
        return claims

    backend, _ = _vote_backend(
        monkeypatch, replicas=2, transform=mismatch_one_rank,
    )

    with pytest.raises(RuntimeError, match="gradient layout differs across ranks"):
        backend.vote_status(
            stage="pre_collective", evidence_sha256="a" * 64,
        )


def test_xla_gradient_layout_vote_runs_before_gradient_reduce(monkeypatch):
    def mismatch_one_rank(claim):
        claims = _claims_for(claim, ranks=(0, 1))
        claims[1]["evidence_sha256"] = "f" * 64
        return claims

    backend, xm = _vote_backend(
        monkeypatch, replicas=2, transform=mismatch_one_rank,
    )
    model = torch.nn.Linear(2, 2)
    model(torch.ones(1, 2)).sum().backward()

    with pytest.raises(RuntimeError, match="gradient layout differs across ranks"):
        backend.all_reduce_sum_gradients(model)

    assert xm.all_reduce_calls == 0


def test_xla_gradient_reduce_applies_optional_global_average_scale(monkeypatch):
    backend, xm = _vote_backend(
        monkeypatch,
        replicas=2,
        transform=lambda claim: _claims_for(claim, ranks=(0, 1)),
    )
    model = torch.nn.Linear(2, 2)
    model(torch.ones(1, 2)).sum().backward()

    backend.all_reduce_sum_gradients(model, scale=0.5)

    assert xm.all_reduce_calls == 1
    assert xm.all_reduce_scales == [0.5]


@pytest.mark.parametrize(
    ("claims_factory", "message"),
    [
        (lambda claim: _claims_for(claim, ranks=(0, 1)), "received 2"),
        (lambda claim: _claims_for(claim, ranks=(0, 0, 2)), "duplicate rank 0"),
        (lambda claim: _claims_for(claim)[:1] + [
            {**_claims_for(claim)[1], "stage": "backward"},
            _claims_for(claim)[2],
        ], "stage mismatch"),
    ],
)
def test_xla_status_vote_rejects_incomplete_or_inconsistent_claims(
        monkeypatch, claims_factory, message):
    backend, _ = _vote_backend(monkeypatch, transform=claims_factory)

    with pytest.raises(RuntimeError, match=message):
        backend.vote_status(stage="forward")
