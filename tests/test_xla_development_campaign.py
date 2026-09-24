"""Host-only wiring test for the explicitly unqualified XLA campaign lane.

This uses one CPU rank behind the real XLA adapter/coordinator interfaces. It
tests campaign orchestration and checkpoint continuation only; it is not TPU,
BF16, multi-rank, runtime, or training-authorization evidence.
"""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
from pathlib import Path
from types import SimpleNamespace

import torch

from anra_v5.miniature_run import MINI_SPEC
import v5_training.production_backend as production_backend_module
from v5_training import production_entry as production_entry_module
from v5_training import xla_adapter as xla_adapter_module
from v5_training.production_entry import frozen_topology, run_campaign


class _Tokenizer:
    identity = SimpleNamespace(artifact_sha256="a" * 64)

    def encode(self, text: str) -> list[int]:
        return [
            4 + int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % 24_572
            for token in text.split()
        ]


class _FakeXlaRuntime:
    @staticmethod
    def global_ordinal() -> int:
        return 0


class _FakeXlaModel:
    REDUCE_SUM = "sum"

    def __init__(self) -> None:
        self.rng_state = 73_011
        self.tags: list[str] = []
        self.all_reduce_calls = 0
        self.mark_step_calls = 0
        self.wait_device_ops_calls = 0

    def mesh_reduce(self, tag, value, reducer):
        self.tags.append(tag)
        return reducer([value])

    def all_reduce(self, operation, gradients):
        assert operation == self.REDUCE_SUM
        self.all_reduce_calls += 1
        return gradients

    def mark_step(self) -> None:
        self.mark_step_calls += 1

    def wait_device_ops(self) -> None:
        self.wait_device_ops_calls += 1

    def get_rng_state(self, *, device: str) -> int:
        assert device == "cpu"
        return self.rng_state

    def set_rng_state(self, state: int, *, device: str) -> None:
        assert device == "cpu"
        self.rng_state = state


def test_xla_development_campaign_wires_update_publish_and_fresh_resume(
    tmp_path: Path, monkeypatch,
) -> None:
    fake_xm = _FakeXlaModel()
    fake_xruntime = _FakeXlaRuntime()
    monkeypatch.setattr(
        xla_adapter_module, "_load_xla", lambda: (fake_xruntime, fake_xm),
    )
    monkeypatch.setattr(
        production_entry_module, "_runtime_name", lambda _device: "xla",
    )
    real_initialize = production_entry_module.initialize
    initializations: list[object] = []

    def counted_initialize(*args, **kwargs):
        initializations.append((args, kwargs))
        return real_initialize(*args, **kwargs)

    monkeypatch.setattr(production_entry_module, "initialize", counted_initialize)
    monkeypatch.setattr(
        production_entry_module,
        "xla_status",
        lambda: {
            "schema": xla_adapter_module.ADAPTER_SCHEMA,
            "status": xla_adapter_module.PENDING_STATUS,
            "device_type": "TPU",
            "world_size": 1,
            "ordinal": 0,
            "versions": {"torch_xla": "fake-test-runtime"},
        },
    )

    topology = dict(frozen_topology())
    topology.update({
        "replicas": 1,
        "tokens_per_replica_microstep": 8,
        "global_tokens_per_microstep": 8,
        "gradient_accumulation_microsteps": 1,
        "global_tokens_per_update": 8,
        "sequences_per_replica_by_bucket": {512: 1},
        "supercycle": [512],
    })
    monkeypatch.setattr(production_entry_module, "frozen_topology", lambda: topology)

    # Keep tensors and model parameters on CPU while making the backend traverse
    # the explicitly unqualified XLA branch. No fake device reaches the product
    # API, and production execution="xla" remains covered by its fail-closed test.
    monkeypatch.setattr(production_backend_module, "_runtime_of", lambda _device: "xla")
    monkeypatch.setattr(
        production_backend_module.ProductionTrainingBackend,
        "_mark_xla_step",
        lambda _self: None,
    )
    real_autocast = torch.autocast

    def host_autocast(*args, **kwargs):
        device_type = kwargs.get("device_type", args[0] if args else None)
        if device_type == "xla":
            return nullcontext()
        return real_autocast(*args, **kwargs)

    monkeypatch.setattr(torch, "autocast", host_autocast)

    documents = [{
        "doc_id": "host-xla-development-doc",
        "source_id": "host-xla-development-source",
        "text": "alpha beta gamma delta " * 80,
        "family": "natural",
    }]
    common = {
        "documents": documents,
        "tokenizer": _Tokenizer(),
        "model_spec": MINI_SPEC,
        "run_id": "host-xla-development-campaign",
        "seed": 73_011,
        "campaign_tokens": 16,
        "max_updates": 1,
        "store_root": tmp_path / "checkpoint-store",
        "device": torch.device("cpu"),
        "torch_module": torch,
        "xb": object(),
        "development_mode": True,
        "cymek_sha": "ab" * 20,
        "execution": "xla-development",
    }

    first = run_campaign(**common)
    assert first["mode"] == "DEVELOPMENT"
    assert first["execution_mode"] == "xla-development"
    assert first["precision"]["status"] == "UNQUALIFIED_DEVELOPMENT_ONLY"
    assert first["resumed"] is False
    assert first["updates_executed"] == 1
    assert first["cumulative_tokens"] == 8
    assert first["resume_equal"] is None
    assert first["resume_verification"] == "DEFERRED_TO_FRESH_WORKER_GROUP"
    first_head = first["checkpoint_head"]
    assert isinstance(first_head, str) and len(first_head) == 64

    second = run_campaign(**common)
    assert second["execution_mode"] == "xla-development"
    assert second["resumed"] is True
    assert second["updates_executed"] == 2
    assert second["cumulative_tokens"] == 16
    assert second["state_complete"] is True
    assert second["resume_equal"] is None
    assert second["resume_verification"] == "DEFERRED_TO_FRESH_WORKER_GROUP"
    assert second["checkpoint_head"] != first_head
    assert second["precision"]["status"] == "UNQUALIFIED_DEVELOPMENT_ONLY"
    assert len(initializations) == 2  # one model per fresh worker-group call

    assert fake_xm.all_reduce_calls == 2
    assert fake_xm.mark_step_calls >= 2
    assert fake_xm.wait_device_ops_calls >= 2
    assert len(fake_xm.tags) > 2  # stage votes, loss aggregation, and v2 publication
    assert second["xla_status"]["versions"]["torch_xla"] == "fake-test-runtime"
