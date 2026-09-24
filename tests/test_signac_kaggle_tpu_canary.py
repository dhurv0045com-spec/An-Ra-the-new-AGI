from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from v5_training.kaggle_tpu_canary import (
    CanaryConfig,
    GRADIENT_PARITY_SCHEMA,
    HOST_MEMORY_FINAL_HASH_SAMPLE_POINT,
    HOST_MEMORY_METRIC,
    HOST_MEMORY_SCHEMA,
    HOST_MEMORY_SAMPLE_POINT,
    HOST_MEMORY_SCOPE,
    PARITY_GRADIENT_ELEMENTS,
    PARITY_TOLERANCES,
    RESTART_SCHEMA,
    SCHEMA,
    _config_sha256,
    _read_host_memory_snapshot,
    _read_memory_snapshot,
    _safe_load_restart_checkpoint,
    _safe_load_rank_stream_checkpoint,
    _run_distributed_gradient_parity_probe,
    _run_synthetic_update,
    _stream_seed,
    _synthetic_cursor,
    _write_restart_checkpoint,
    _write_rank_stream_checkpoint,
    aggregate_restart_receipts,
    aggregate_rank_receipts,
    compare_gradient_parity,
)
from signac_100m.spec import MODEL_SPEC

TEST_SOURCE_SHA256 = "9" * 64
TEST_RUNTIME = {
    "python_version": "3.11.9",
    "torch_version": "2.6.0",
    "torch_xla_version": "2.6.0",
    "platform": "Linux-test",
}


class _TestRankBackend:
    def __init__(self, *, fail_vote: int | None = None) -> None:
        self.fail_vote = fail_vote
        self.vote_calls = 0
        self.gradient_reduce_calls = 0

    def run_local_stage(self, *, stage, callback):
        assert stage == "local_update"
        self.vote_calls += 1
        result = callback()
        if self.vote_calls == self.fail_vote:
            raise RuntimeError("shared local-stage vote failed on a remote rank")
        return result

    def all_reduce_sum_gradients(self, model, *, scale=None):
        self.gradient_reduce_calls += 1


def _rank_receipts(world_size: int = 8) -> list[dict[str, object]]:
    return [
        {
            "ordinal": rank,
            "status": "PASS",
            "schema": SCHEMA,
            "candidate": "m102_primary",
            "source_tree_sha256": TEST_SOURCE_SHA256,
            **TEST_RUNTIME,
            "model_spec_sha256": MODEL_SPEC.sha256(),
            "parameter_count": MODEL_SPEC.parameter_receipt().total,
            "config": {
                "candidate": "m102_primary",
                "source_tree_sha256": TEST_SOURCE_SHA256,
                "expected_world_size": world_size,
                "expected_global_devices": world_size,
                "optimizer_updates": 2,
                "sequence_length": 4096,
                "microbatch_size": 1,
                "accumulation_steps": 4,
            },
            "device_type": "TPU",
            "world_size": world_size,
            "global_device_count": world_size,
            "addressable_device_count": 1,
            "initial_parameter_sha256": "a" * 64,
            "parameter_sha256": "b" * 64,
            "optimizer_moment_sha256": "c" * 64,
            "optimizer_step": 2,
            "rank_sample_stream_sha256": hashlib.sha256(f"rank-stream-{rank}".encode()).hexdigest(),
            "collective_probe_sum": world_size * (world_size + 1) / 2,
            "bf16_probe_finite": True,
            "device_rng_progresses": True,
            "xla_rng_state_sha256_by_update": [
                hashlib.sha256(f"rank-{rank}-xla-state-{update}".encode()).hexdigest()
                for update in range(2)
            ],
            "restart_xla_rng_probe_sha256": hashlib.sha256(
                f"rank-{rank}-xla-probe".encode()
            ).hexdigest(),
            "restart_xla_rng_state_sha256_after_probe": hashlib.sha256(
                f"rank-{rank}-xla-after-probe".encode()
            ).hexdigest(),
            "rank_local_mean_loss_by_update": [2.0 + rank, 1.0 + rank],
            "global_grad_norm_preclip_by_update": [2.0, 0.5],
            "distributed_gradient_parity": {
                "schema": GRADIENT_PARITY_SCHEMA,
                "status": "PASS",
                "world_size": world_size,
                "rank_examples": rank + 1,
                "global_examples": world_size * (world_size + 1) // 2,
                "fixture_sha256": "e" * 64,
                "denominator_policy": "sum_local_loss_over_global_example_count_then_all_reduce_sum",
                "by_precision": {
                    mode: {
                        "status": "PASS",
                        "gradient_elements": PARITY_GRADIENT_ELEMENTS,
                        "absolute_tolerance": tolerances["absolute"],
                        "relative_tolerance": tolerances["relative"],
                        "max_normalized_gradient_error": 0.5,
                        "normalized_loss_error": 0.25,
                    }
                    for mode, tolerances in PARITY_TOLERANCES.items()
                },
            },
            "steady_update_seconds": [0.25],
            "host_memory_samples_by_update": [
                {
                    "schema": HOST_MEMORY_SCHEMA,
                    "status": "AVAILABLE",
                    "metric": HOST_MEMORY_METRIC,
                    "scope": HOST_MEMORY_SCOPE,
                    "unit": "bytes",
                    "sample_point": HOST_MEMORY_SAMPLE_POINT,
                    "source": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
                    "high_water_rss_bytes": 2_097_152 + rank * 1024,
                }
                for _ in range(2)
            ],
            "host_memory_after_final_hashes": {
                "schema": HOST_MEMORY_SCHEMA,
                "status": "AVAILABLE",
                "metric": HOST_MEMORY_METRIC,
                "scope": HOST_MEMORY_SCOPE,
                "unit": "bytes",
                "sample_point": HOST_MEMORY_FINAL_HASH_SAMPLE_POINT,
                "source": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
                "high_water_rss_bytes": 4_194_304 + rank * 1024,
            },
            "local_supervised_tokens_per_update": 16_380,
            "global_supervised_tokens_per_update": 16_380 * world_size,
        }
        for rank in range(world_size)
    ]


def test_kaggle_config_freezes_eight_worker_full_context_canary() -> None:
    config = CanaryConfig(source_tree_sha256=TEST_SOURCE_SHA256)
    config.validate()
    assert config.expected_global_devices == config.expected_world_size == 8
    assert config.sequence_length == 4096
    assert config.accumulation_steps == 4
    assert config.optimizer_updates >= 2


def test_host_rss_high_water_normalizes_linux_kib_to_bytes() -> None:
    resource_module = SimpleNamespace(
        RUSAGE_SELF=0,
        getrusage=lambda _who: SimpleNamespace(ru_maxrss=2048),
    )
    sample = _read_host_memory_snapshot(resource_module=resource_module, system="Linux")
    assert sample["status"] == "AVAILABLE"
    assert sample["high_water_rss_bytes"] == 2 * 1024 * 1024
    assert sample["metric"] == HOST_MEMORY_METRIC
    assert sample["scope"] == HOST_MEMORY_SCOPE


def test_host_rss_high_water_is_optional_outside_linux() -> None:
    sample = _read_host_memory_snapshot(system="Darwin")
    assert sample["status"] == "UNAVAILABLE"
    assert isinstance(sample["reason"], str) and sample["reason"]


def test_host_rss_high_water_is_optional_when_resource_read_fails() -> None:
    def fail_read(_who):
        raise OSError("process counters unavailable")

    resource_module = SimpleNamespace(RUSAGE_SELF=0, getrusage=fail_read)
    sample = _read_host_memory_snapshot(resource_module=resource_module, system="Linux")
    assert sample["status"] == "UNAVAILABLE"
    assert "process counters unavailable" in sample["reason"]


def test_canary_source_identity_is_required_and_validated() -> None:
    with pytest.raises(ValueError, match="source_tree_sha256"):
        CanaryConfig().validate()
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        CanaryConfig(source_tree_sha256="Z" * 64).validate()


def test_run_rejects_source_identity_mismatch_before_runtime_launch(monkeypatch) -> None:
    from signac_100m import source_identity
    from v5_training import kaggle_tpu_canary

    monkeypatch.setattr(
        source_identity, "build_source_identity",
        lambda _root: {"source_tree_sha256": "8" * 64},
    )
    with pytest.raises(ValueError, match="does not match the live Signac source identity"):
        kaggle_tpu_canary.run(CanaryConfig(source_tree_sha256=TEST_SOURCE_SHA256))


def test_aggregate_requires_every_rank_and_replicated_state() -> None:
    aggregate = aggregate_rank_receipts(
        _rank_receipts(), expected_world_size=8, expected_global_devices=8,
        expected_optimizer_step=2,
    )
    assert aggregate["participating_ordinals"] == list(range(8))
    assert aggregate["all_core_update_verified"] is True
    assert aggregate["optimizer_step"] == 2
    assert aggregate["global_mean_loss_by_update_equal_token_rank_average"] == [5.5, 4.5]
    assert aggregate["steady_critical_path_seconds_by_update"] == [0.25]
    assert aggregate["steady_global_tokens_per_second"] == 16_380 * 8 / 0.25
    assert aggregate["distributed_gradient_parity_verified"] is True
    assert aggregate["distributed_gradient_parity_max_normalized_error"] == {
        "fp32": 0.5, "bf16_autocast": 0.5
    }
    assert aggregate["distributed_gradient_parity_fixture_sha256"] == "e" * 64
    assert aggregate["host_memory_observation"]["status"] == "OBSERVED"
    assert aggregate["host_memory_observation"]["sample_point"] == HOST_MEMORY_FINAL_HASH_SAMPLE_POINT
    assert aggregate["host_memory_observation"]["high_water_rss_bytes_by_rank"]["7"] == 4_201_472
    assert aggregate["source_tree_sha256"] == TEST_SOURCE_SHA256
    assert aggregate["runtime_identity"]["python_version"] == TEST_RUNTIME["python_version"]
    assert len(aggregate["runtime_identity"]["sha256"]) == 64


@pytest.mark.parametrize("mutation", [
    "missing_rank", "duplicate_rank", "state_mismatch", "duplicate_data",
    "bad_collective", "bad_timing", "candidate_mismatch", "config_mismatch",
    "token_mismatch", "bad_hash", "no_addressable_devices", "addressable_mismatch",
    "parity_fail", "parity_tolerance", "parity_count", "parity_rank_count",
    "parity_fixture_mismatch", "source_mismatch", "runtime_missing", "runtime_unknown",
    "runtime_mismatch",
])
def test_aggregate_fails_closed_on_incomplete_or_inconsistent_ranks(mutation: str) -> None:
    receipts = _rank_receipts()
    if mutation == "missing_rank":
        receipts.pop()
    elif mutation == "duplicate_rank":
        receipts[-1]["ordinal"] = 6
    elif mutation == "state_mismatch":
        receipts[-1]["parameter_sha256"] = "different"
    elif mutation == "duplicate_data":
        receipts[-1]["rank_sample_stream_sha256"] = receipts[0]["rank_sample_stream_sha256"]
    elif mutation == "bad_collective":
        receipts[-1]["collective_probe_sum"] = -1
    elif mutation == "bad_timing":
        receipts[-1]["steady_update_seconds"] = [0.0]
    elif mutation == "candidate_mismatch":
        receipts[-1]["candidate"] = "tpu_tiled_challenger"
    elif mutation == "config_mismatch":
        receipts[-1]["config"]["seed"] = 99
    elif mutation == "token_mismatch":
        receipts[-1]["global_supervised_tokens_per_update"] += 1
    elif mutation == "bad_hash":
        receipts[-1]["model_spec_sha256"] = "invalid"
    elif mutation == "no_addressable_devices":
        receipts[-1]["addressable_device_count"] = 0
    elif mutation == "addressable_mismatch":
        receipts[-1]["addressable_device_count"] = 2
    elif mutation == "parity_fail":
        receipts[-1]["distributed_gradient_parity"]["by_precision"]["bf16_autocast"]["status"] = "FAIL"
    elif mutation == "parity_tolerance":
        receipts[-1]["distributed_gradient_parity"]["by_precision"]["fp32"]["absolute_tolerance"] = 0.0
    elif mutation == "parity_count":
        receipts[-1]["distributed_gradient_parity"]["global_examples"] += 1
    elif mutation == "parity_rank_count":
        receipts[-1]["distributed_gradient_parity"]["rank_examples"] += 1
    elif mutation == "parity_fixture_mismatch":
        receipts[-1]["distributed_gradient_parity"]["fixture_sha256"] = "f" * 64
    elif mutation == "source_mismatch":
        receipts[-1]["source_tree_sha256"] = "8" * 64
    elif mutation == "runtime_missing":
        receipts[-1]["torch_xla_version"] = ""
    elif mutation == "runtime_unknown":
        receipts[-1]["torch_xla_version"] = "unknown"
    elif mutation == "runtime_mismatch":
        receipts[-1]["platform"] = "other-platform"
    with pytest.raises(ValueError):
        aggregate_rank_receipts(
            receipts, expected_world_size=8, expected_global_devices=8,
            expected_optimizer_step=2,
        )


def test_gradient_parity_comparator_applies_frozen_absolute_and_relative_error() -> None:
    good = compare_gradient_parity(
        [1.000001, -0.499999],
        [1.0, -0.5],
        distributed_loss=0.750001,
        reference_loss=0.75,
        absolute_tolerance=1e-5,
        relative_tolerance=1e-5,
    )
    bad = compare_gradient_parity(
        [1.01, -0.5],
        [1.0, -0.5],
        distributed_loss=0.75,
        reference_loss=0.75,
        absolute_tolerance=1e-5,
        relative_tolerance=1e-5,
    )
    assert good["status"] == "PASS"
    assert good["max_normalized_gradient_error"] <= 1.0
    assert bad["status"] == "FAIL"
    with pytest.raises(ValueError, match="identical dimensions"):
        compare_gradient_parity(
            [1.0], [1.0, 2.0],
            distributed_loss=1.0, reference_loss=1.0,
            absolute_tolerance=1e-5, relative_tolerance=1e-5,
        )


def test_gradient_parity_tolerances_are_immutable() -> None:
    with pytest.raises(TypeError):
        PARITY_TOLERANCES["fp32"]["absolute"] = 1.0


def test_parity_probe_runs_fp32_and_bf16_against_a_cpu_single_device_oracle() -> None:
    torch = pytest.importorskip("torch")
    rank_backend = _TestRankBackend()
    xm = SimpleNamespace(
        REDUCE_SUM=0,
        all_reduce=lambda _reduction, value: value,
        mark_step=lambda: None,
    )
    report = _run_distributed_gradient_parity_probe(
        torch=torch,
        xm=xm,
        rank_backend=rank_backend,
        device=torch.device("cpu"),
        world_size=1,
        ordinal=0,
        seed=731,
    )
    assert report["status"] == "PASS"
    assert report["global_examples"] == report["rank_examples"] == 1
    assert report["by_precision"]["fp32"]["status"] == "PASS"
    assert report["by_precision"]["bf16_autocast"]["status"] == "PASS"
    assert rank_backend.vote_calls == 4
    assert rank_backend.gradient_reduce_calls == 2


def test_parity_probe_votes_local_work_before_any_collective() -> None:
    torch = pytest.importorskip("torch")
    rank_backend = _TestRankBackend(fail_vote=1)
    collective_calls = 0

    def all_reduce(_reduction, value):
        nonlocal collective_calls
        collective_calls += 1
        return value

    with pytest.raises(RuntimeError, match="remote rank"):
        _run_distributed_gradient_parity_probe(
            torch=torch,
            xm=SimpleNamespace(REDUCE_SUM=0, all_reduce=all_reduce),
            rank_backend=rank_backend,
            device=torch.device("cpu"),
            world_size=2,
            ordinal=0,
            seed=731,
        )
    assert collective_calls == 0
    assert rank_backend.gradient_reduce_calls == 0


def test_synthetic_update_votes_local_backward_before_gradient_reduce() -> None:
    torch = pytest.importorskip("torch")

    class TinyLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(32, 8)

        def forward_hidden(self, token_ids, _positions, _mask,
                           use_activation_checkpointing=False):
            return self.embedding(token_ids)

        def forward(self, token_ids, positions, mask, use_activation_checkpointing=False):
            hidden = self.forward_hidden(
                token_ids, positions, mask,
                use_activation_checkpointing=use_activation_checkpointing,
            )
            return torch.nn.functional.linear(hidden, self.embedding.weight)

    backend = _TestRankBackend(fail_vote=1)
    model = TinyLM()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    config = CanaryConfig(
        expected_world_size=2,
        expected_global_devices=2,
        sequence_length=4,
        accumulation_steps=1,
        source_tree_sha256=TEST_SOURCE_SHA256,
    )
    with pytest.raises(RuntimeError, match="remote rank"):
        _run_synthetic_update(
            update_index=0,
            ordinal=0,
            config=config,
            spec=SimpleNamespace(vocabulary_size=32),
            model=model,
            optimizer=optimizer,
            device=torch.device("cpu"),
            positions=None,
            attention_mask=None,
            world_size=2,
            torch=torch,
            rank_backend=backend,
            sample_generator=torch.Generator(device="cpu").manual_seed(7),
        )
    assert backend.vote_calls == 1
    assert backend.gradient_reduce_calls == 0


def test_synthetic_update_completes_with_pre_and_post_update_votes() -> None:
    torch = pytest.importorskip("torch")

    class TinyLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(32, 8)

        def forward_hidden(self, token_ids, _positions, _mask,
                           use_activation_checkpointing=False):
            return self.embedding(token_ids)

        def forward(self, token_ids, positions, mask, use_activation_checkpointing=False):
            hidden = self.forward_hidden(
                token_ids, positions, mask,
                use_activation_checkpointing=use_activation_checkpointing,
            )
            return torch.nn.functional.linear(hidden, self.embedding.weight)

    backend = _TestRankBackend()
    model = TinyLM()
    before = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    config = CanaryConfig(
        expected_world_size=1,
        expected_global_devices=1,
        sequence_length=4,
        accumulation_steps=2,
        source_tree_sha256=TEST_SOURCE_SHA256,
    )
    loss, grad_norm, elapsed, sample_hashes = _run_synthetic_update(
        update_index=0,
        ordinal=0,
        config=config,
        spec=SimpleNamespace(vocabulary_size=32),
        model=model,
        optimizer=optimizer,
        device=torch.device("cpu"),
        positions=None,
        attention_mask=None,
        world_size=1,
        torch=torch,
        rank_backend=backend,
        sample_generator=torch.Generator(device="cpu").manual_seed(9),
    )

    assert loss > 0.0
    assert grad_norm > 0.0
    assert elapsed > 0.0
    assert len(sample_hashes) == 2
    assert backend.vote_calls == 2
    assert backend.gradient_reduce_calls == 1
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))


def test_config_rejects_topology_that_cannot_use_every_frozen_device() -> None:
    with pytest.raises(ValueError, match="one worker per TPU device"):
        CanaryConfig(expected_world_size=4, expected_global_devices=8,
                     source_tree_sha256=TEST_SOURCE_SHA256).validate()


def test_qualification_config_requires_warmup_plus_requested_steady_updates() -> None:
    with pytest.raises(ValueError, match="compile update plus"):
        CanaryConfig(optimizer_updates=20, minimum_steady_updates=20,
                     source_tree_sha256=TEST_SOURCE_SHA256).validate()
    CanaryConfig(optimizer_updates=21, minimum_steady_updates=20,
                 source_tree_sha256=TEST_SOURCE_SHA256).validate()


def test_memory_snapshot_preserves_peak_counters_and_never_invents_legacy_peak() -> None:
    available = _read_memory_snapshot(SimpleNamespace(get_memory_info=lambda _device: {
        "bytes_used": "700", "bytes_limit": 1000, "peak_bytes_used": "820",
    }), "xla:0")
    legacy = _read_memory_snapshot(SimpleNamespace(get_memory_info=lambda _device: {
        "kb_free": 3, "kb_total": 10,
    }), "xla:0")
    assert available == {
        "status": "AVAILABLE",
        "bytes_used": 700,
        "bytes_limit": 1000,
        "peak_bytes_used": 820,
    }
    assert legacy["status"] == "PARTIAL"
    assert legacy["bytes_used"] == 7 * 1024
    assert "peak_bytes_used" not in legacy


def _qualification_rank_receipts(*, peak_fraction: float = 0.80) -> list[dict[str, object]]:
    receipts = _rank_receipts()
    for row in receipts:
        row["config"]["optimizer_updates"] = 21
        row["config"]["minimum_steady_updates"] = 20
        row["optimizer_step"] = 21
        row["rank_local_mean_loss_by_update"] = [1.0] * 21
        row["global_grad_norm_preclip_by_update"] = [0.5] * 21
        row["steady_update_seconds"] = [0.25] * 20
        row["device_memory_samples_by_update"] = [
            {
                "status": "AVAILABLE",
                "bytes_used": int(1000 * peak_fraction),
                "bytes_limit": 1000,
                "peak_bytes_used": int(1000 * peak_fraction),
            }
            for _ in range(21)
        ]
    return receipts


def test_qualification_aggregator_requires_20_updates_and_85_percent_peak_headroom() -> None:
    aggregate = aggregate_rank_receipts(
        _qualification_rank_receipts(), expected_world_size=8,
        expected_global_devices=8, expected_optimizer_step=21,
    )
    assert aggregate["measurement_profile"] == "QUALIFICATION"
    assert aggregate["measured_steady_update_count"] == 20
    assert aggregate["peak_memory_status"] == "PASS"
    assert aggregate["worst_peak_memory_fraction"] == 0.80
    with pytest.raises(ValueError, match="85%"):
        aggregate_rank_receipts(
            _qualification_rank_receipts(peak_fraction=0.86),
            expected_world_size=8, expected_global_devices=8,
            expected_optimizer_step=21,
        )


def test_qualification_aggregator_rejects_missing_peak_counters() -> None:
    receipts = _qualification_rank_receipts()
    receipts[-1]["device_memory_samples_by_update"] = [
        {"status": "PARTIAL", "bytes_limit": 1000}
    ] * 21
    with pytest.raises(ValueError, match="peak-memory and limit counters"):
        aggregate_rank_receipts(
            receipts, expected_world_size=8, expected_global_devices=8,
            expected_optimizer_step=21,
        )


def test_missing_host_rss_does_not_replace_or_block_device_memory_gate() -> None:
    receipts = _qualification_rank_receipts()
    for row in receipts:
        row["host_memory_samples_by_update"] = [
            {
                "schema": HOST_MEMORY_SCHEMA,
                "status": "UNAVAILABLE",
                "metric": HOST_MEMORY_METRIC,
                "scope": HOST_MEMORY_SCOPE,
                "unit": "bytes",
                "sample_point": HOST_MEMORY_SAMPLE_POINT,
                "source": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
                "reason": "resource module unavailable",
            }
            for _ in range(21)
        ]
        row["host_memory_after_final_hashes"] = {
            "schema": HOST_MEMORY_SCHEMA,
            "status": "UNAVAILABLE",
            "metric": HOST_MEMORY_METRIC,
            "scope": HOST_MEMORY_SCOPE,
            "unit": "bytes",
            "sample_point": HOST_MEMORY_FINAL_HASH_SAMPLE_POINT,
            "source": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
            "reason": "resource module unavailable",
        }
    aggregate = aggregate_rank_receipts(
        receipts, expected_world_size=8, expected_global_devices=8,
        expected_optimizer_step=21,
    )
    assert aggregate["peak_memory_status"] == "PASS"
    assert aggregate["worst_peak_memory_fraction"] == 0.80
    assert aggregate["host_memory_observation"]["status"] == "UNAVAILABLE"


def test_restart_checkpoints_round_trip_model_optimizer_cursor_and_rank_rng_stream() -> None:
    torch = pytest.importorskip("torch")
    config = CanaryConfig(verify_restart=True, source_tree_sha256=TEST_SOURCE_SHA256)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    inputs = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    model(inputs).square().sum().backward()
    optimizer.step()
    stream_generator = torch.Generator(device="cpu").manual_seed(_stream_seed(config, 0))
    torch.randint(4, 31, (7,), generator=stream_generator)
    saved_generator_state = stream_generator.get_state().clone()
    uninterrupted_generator = torch.Generator(device="cpu").manual_seed(_stream_seed(config, 0))
    uninterrupted_generator.set_state(saved_generator_state)
    uninterrupted_next = torch.randint(4, 31, (11,), generator=uninterrupted_generator)
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "restart.pt"
        sha256 = _write_restart_checkpoint(
            path=checkpoint,
            model=model,
            optimizer=optimizer,
            torch=torch,
            config=config,
            model_spec_sha256="a" * 64,
            next_update_index=1,
            global_ordinal=0,
            world_size=8,
        )
        payload = _safe_load_restart_checkpoint(checkpoint, torch=torch)
        assert len(sha256) == 64
        assert payload["schema"] == RESTART_SCHEMA
        assert payload["data_cursor"] == {
            "schema": "anra-signac-replicated-update-cursor/v1",
            "next_update_index": 1,
            "next_microstep_index": 0,
        }
        assert payload["model_state_dict"]["weight"].device.type == "cpu"
        restored_model = torch.nn.Linear(3, 2)
        restored_optimizer = torch.optim.AdamW(
            restored_model.parameters(), lr=config.learning_rate
        )
        restored_model.load_state_dict(payload["model_state_dict"])
        restored_optimizer.load_state_dict(payload["optimizer_state_dict"])
        for original, restored in zip(model.parameters(), restored_model.parameters()):
            assert torch.equal(original, restored)
        rank_stream_path = Path(directory) / "rank-00.pt"
        rank_stream_sha = _write_rank_stream_checkpoint(
            path=rank_stream_path,
            torch=torch,
            config=config,
            model_spec_sha256="a" * 64,
            ordinal=0,
            world_size=config.expected_world_size,
            generator=stream_generator,
            xla_rng_state=b'{"schema":"test-xla-rng","state":73}',
            next_update_index=1,
        )
        rank_payload = _safe_load_rank_stream_checkpoint(rank_stream_path, torch=torch)
        assert len(rank_stream_sha) == 64
        assert torch.equal(rank_payload["generator_state"], saved_generator_state)
        assert bytes(rank_payload["xla_rng_state"].tolist()) == b'{"schema":"test-xla-rng","state":73}'
        assert rank_payload["xla_rng_state_sha256"] == hashlib.sha256(
            b'{"schema":"test-xla-rng","state":73}'
        ).hexdigest()
        assert rank_payload["data_cursor"] == _synthetic_cursor(
            config=config, ordinal=0, next_update_index=1,
        )
        rank_payload["xla_rng_state"][0] ^= 1
        torch.save(rank_payload, rank_stream_path)
        with pytest.raises(ValueError, match="XLA RNG-state hash"):
            _safe_load_rank_stream_checkpoint(rank_stream_path, torch=torch)
        restored_generator = torch.Generator(device="cpu").manual_seed(_stream_seed(config, 0))
        restored_generator.set_state(saved_generator_state)
        assert torch.equal(
            uninterrupted_next,
            torch.randint(4, 31, (11,), generator=restored_generator),
        )


def test_controlled_restart_requires_same_optimizer_state_and_next_data_stream() -> None:
    config = CanaryConfig(verify_restart=True, source_tree_sha256=TEST_SOURCE_SHA256)
    config.validate()
    baseline = _rank_receipts()
    for row in baseline:
        ordinal = int(row["ordinal"])
        row["rank_sample_sha256_by_update"] = [
            [
                hashlib.sha256(f"rank-{ordinal}-update-{update}-micro-{micro}".encode()).hexdigest()
                for micro in range(config.accumulation_steps)
            ]
            for update in range(config.optimizer_updates)
        ]
        row["rank_stream_state_sha256_by_update"] = [
            hashlib.sha256(f"rank-{ordinal}-rng-{update}".encode()).hexdigest()
            for update in range(config.optimizer_updates)
        ]
        row["data_cursor_by_update"] = [
            _synthetic_cursor(config=config, ordinal=ordinal, next_update_index=update + 1)
            for update in range(config.optimizer_updates)
        ]
    checkpoint_sha = "d" * 64
    rank_stream_checkpoint_hashes = {
        str(row["ordinal"]): hashlib.sha256(
            f"rank-stream-file-{row['ordinal']}".encode()
        ).hexdigest()
        for row in baseline
    }
    resumed = [
        {
            "schema": RESTART_SCHEMA,
            "status": "PASS",
            "candidate": config.candidate,
            "source_tree_sha256": config.source_tree_sha256,
            **TEST_RUNTIME,
            "ordinal": int(row["ordinal"]),
            "device_type": "TPU",
            "world_size": config.expected_world_size,
            "config_sha256": _config_sha256(config),
            "checkpoint_sha256": checkpoint_sha,
            "rank_stream_checkpoint_sha256": rank_stream_checkpoint_hashes[str(row["ordinal"])],
            "next_update_index": 1,
            "restored_optimizer_step": 1,
            "optimizer_step": config.optimizer_updates,
            "parameter_sha256": row["parameter_sha256"],
            "optimizer_moment_sha256": row["optimizer_moment_sha256"],
            "restored_stream_state_sha256": row["rank_stream_state_sha256_by_update"][0],
            "restored_data_cursor": row["data_cursor_by_update"][0],
            "stream_state_sha256_after_updates": row["rank_stream_state_sha256_by_update"][-1],
            "restored_xla_rng_state_sha256": row["xla_rng_state_sha256_by_update"][0],
            "xla_rng_replay_probe_sha256": row["restart_xla_rng_probe_sha256"],
            "xla_rng_state_sha256_after_probe": row["restart_xla_rng_state_sha256_after_probe"],
            "xla_rng_state_sha256_after_updates": row["xla_rng_state_sha256_by_update"][-1],
            "next_data_cursor": row["data_cursor_by_update"][-1],
            "rank_sample_sha256_by_update": row["rank_sample_sha256_by_update"][1:],
        }
        for row in baseline
    ]
    result = aggregate_restart_receipts(
        baseline, resumed, config=config, checkpoint_sha256=checkpoint_sha,
        rank_stream_checkpoint_sha256=rank_stream_checkpoint_hashes,
    )
    assert result["status"] == "PASS"
    assert result["parameters_match_uninterrupted"] is True
    assert result["optimizer_moments_match_uninterrupted"] is True
    assert result["rank_local_sample_streams_match_uninterrupted_suffix"] is True
    assert result["rank_local_rng_states_match_uninterrupted"] is True
    assert result["rank_local_synthetic_cursors_match_uninterrupted"] is True
    assert result["rank_local_xla_rng_replay_matches_uninterrupted"] is True
    assert len(result["runtime_identity"]["sha256"]) == 64
    assert result["production_exact_resume_certified"] is False


@pytest.mark.parametrize(
    "field, value",
    [
        ("source_tree_sha256", "8" * 64),
        ("torch_xla_version", "9.9.9"),
        ("checkpoint_sha256", "e" * 64),
        ("rank_stream_checkpoint_sha256", "e" * 64),
        ("next_update_index", 0),
        ("optimizer_step", 3),
        ("parameter_sha256", "f" * 64),
        ("optimizer_moment_sha256", "a" * 64),
        ("rank_sample_sha256_by_update", []),
        ("restored_stream_state_sha256", "f" * 64),
        ("restored_data_cursor", {}),
        ("stream_state_sha256_after_updates", "f" * 64),
        ("next_data_cursor", {}),
        ("restored_xla_rng_state_sha256", "f" * 64),
        ("xla_rng_replay_probe_sha256", "f" * 64),
        ("xla_rng_state_sha256_after_probe", "f" * 64),
        ("xla_rng_state_sha256_after_updates", "f" * 64),
    ],
)
def test_controlled_restart_rejects_a_mismatched_continuation(field: str, value: object) -> None:
    config = CanaryConfig(verify_restart=True, source_tree_sha256=TEST_SOURCE_SHA256)
    baseline = _rank_receipts()
    for row in baseline:
        ordinal = int(row["ordinal"])
        row["rank_sample_sha256_by_update"] = [
            [hashlib.sha256(f"{ordinal}-{update}-{micro}".encode()).hexdigest() for micro in range(4)]
            for update in range(2)
        ]
        row["rank_stream_state_sha256_by_update"] = [
            hashlib.sha256(f"{ordinal}-rng-{update}".encode()).hexdigest()
            for update in range(2)
        ]
        row["data_cursor_by_update"] = [
            _synthetic_cursor(config=config, ordinal=ordinal, next_update_index=update + 1)
            for update in range(2)
        ]
    checkpoint_sha = "d" * 64
    rank_stream_checkpoint_hashes = {
        str(row["ordinal"]): hashlib.sha256(
            f"rank-stream-file-{row['ordinal']}".encode()
        ).hexdigest()
        for row in baseline
    }
    resumed = [
        {
            "schema": RESTART_SCHEMA,
            "status": "PASS",
            "candidate": config.candidate,
            "source_tree_sha256": config.source_tree_sha256,
            **TEST_RUNTIME,
            "ordinal": int(row["ordinal"]),
            "device_type": "TPU",
            "world_size": config.expected_world_size,
            "config_sha256": _config_sha256(config),
            "checkpoint_sha256": checkpoint_sha,
            "rank_stream_checkpoint_sha256": rank_stream_checkpoint_hashes[str(row["ordinal"])],
            "next_update_index": 1,
            "restored_optimizer_step": 1,
            "optimizer_step": 2,
            "parameter_sha256": row["parameter_sha256"],
            "optimizer_moment_sha256": row["optimizer_moment_sha256"],
            "restored_stream_state_sha256": row["rank_stream_state_sha256_by_update"][0],
            "restored_data_cursor": row["data_cursor_by_update"][0],
            "stream_state_sha256_after_updates": row["rank_stream_state_sha256_by_update"][-1],
            "restored_xla_rng_state_sha256": row["xla_rng_state_sha256_by_update"][0],
            "xla_rng_replay_probe_sha256": row["restart_xla_rng_probe_sha256"],
            "xla_rng_state_sha256_after_probe": row["restart_xla_rng_state_sha256_after_probe"],
            "xla_rng_state_sha256_after_updates": row["xla_rng_state_sha256_by_update"][-1],
            "next_data_cursor": row["data_cursor_by_update"][-1],
            "rank_sample_sha256_by_update": row["rank_sample_sha256_by_update"][1:],
        }
        for row in baseline
    ]
    resumed[0][field] = value
    with pytest.raises(ValueError):
        aggregate_restart_receipts(
            baseline, resumed, config=config, checkpoint_sha256=checkpoint_sha,
            rank_stream_checkpoint_sha256=rank_stream_checkpoint_hashes,
        )
