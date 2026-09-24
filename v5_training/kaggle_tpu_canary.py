"""All-core Kaggle TPU canary for bounded Signac model updates.

Run as a module in a fresh Kaggle TPU kernel. The parent process never acquires
an XLA device; ``torch_xla.launch`` starts workers and each worker acquires its
assigned device. This validates distributed plumbing and bounded synthetic
updates only. It does not authorize production training or prove capability.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as importlib_metadata
import json
import math
import os
import platform
import shutil
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType

from v5_objectives.causal_lm import causal_lm_loss_from_hidden


SCHEMA = "anra-signac-kaggle-all-core-canary/v6"
RESTART_SCHEMA = "anra-signac-kaggle-controlled-restart/v5"
HOST_MEMORY_SCHEMA = "anra-signac-host-memory-observation/v1"
HOST_MEMORY_METRIC = "process_lifetime_high_water_rss"
HOST_MEMORY_SCOPE = "one_worker_process; not host-wide capacity"
HOST_MEMORY_SAMPLE_POINT = "after each completed optimizer update"
HOST_MEMORY_FINAL_HASH_SAMPLE_POINT = "after final parameter and Adam-moment hashes"
RANK_STREAM_SCHEMA = "anra-signac-kaggle-rank-stream-state/v2"
SYNTHETIC_CURSOR_SCHEMA = "anra-signac-synthetic-stream-cursor/v1"
GRADIENT_PARITY_SCHEMA = "anra-signac-distributed-gradient-parity/v1"
PARITY_FEATURE_DIMENSION = 8
PARITY_CLASS_COUNT = 4
QUALIFICATION_MIN_STEADY_UPDATES = 20
QUALIFICATION_MAX_PEAK_MEMORY_FRACTION = 0.85
PARITY_GRADIENT_ELEMENTS = (
    PARITY_FEATURE_DIMENSION * PARITY_CLASS_COUNT + PARITY_CLASS_COUNT
)
PARITY_TOLERANCES = MappingProxyType({
    "fp32": MappingProxyType({"absolute": 2e-5, "relative": 1e-4}),
    "bf16_autocast": MappingProxyType({"absolute": 1e-3, "relative": 1e-2}),
})
SPEC_NAMES = {
    "m102_primary": "MODEL_SPEC",
    "tpu_depth_preserving_challenger": "TPU_DEPTH_PRESERVING_CHALLENGER",
    "tpu_tiled_challenger": "TPU_TILED_CHALLENGER",
}


@dataclass(frozen=True, slots=True)
class CanaryConfig:
    candidate: str = "m102_primary"
    output_dir: str = "/kaggle/working/signac_100m_distributed"
    expected_global_devices: int = 8
    expected_world_size: int = 8
    seed: int = 73_011
    sequence_length: int = 4096
    microbatch_size: int = 1
    accumulation_steps: int = 4
    optimizer_updates: int = 2
    minimum_steady_updates: int = 1
    learning_rate: float = 3e-4
    max_grad_norm: float = 1.0
    verify_restart: bool = False
    source_tree_sha256: str = ""

    def validate(self) -> None:
        if not _is_sha256(self.source_tree_sha256):
            raise ValueError("source_tree_sha256 must be a lowercase SHA-256")
        if self.candidate not in SPEC_NAMES:
            raise ValueError(f"unknown Signac candidate: {self.candidate}")
        if self.expected_global_devices <= 0 or self.expected_world_size <= 0:
            raise ValueError("expected TPU topology dimensions must be positive")
        if self.expected_world_size != self.expected_global_devices:
            raise ValueError("this Kaggle eight-device canary requires one worker per TPU device")
        if self.sequence_length < 2 or self.microbatch_size <= 0:
            raise ValueError("invalid sequence or microbatch size")
        if self.accumulation_steps <= 0 or self.optimizer_updates < 2:
            raise ValueError("use positive accumulation and at least two updates")
        if type(self.minimum_steady_updates) is not int or self.minimum_steady_updates <= 0:
            raise ValueError("minimum_steady_updates must be a positive integer")
        if self.optimizer_updates - 1 < self.minimum_steady_updates:
            raise ValueError(
                "optimizer_updates must include one compile update plus the required steady updates"
            )
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning rate must be finite and positive")
        if not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be finite and positive")
        if type(self.verify_restart) is not bool:
            raise ValueError("verify_restart must be boolean")


def aggregate_rank_receipts(
    receipts: list[dict[str, object]], *, expected_world_size: int,
    expected_global_devices: int, expected_optimizer_step: int | None = None,
) -> dict[str, object]:
    """Fail closed unless every rank reports the same replicated update."""

    ranks = sorted(int(row.get("ordinal", -1)) for row in receipts)
    expected = list(range(expected_world_size))
    if ranks != expected:
        raise ValueError(f"rank receipts must cover {expected}; got {ranks}")
    if len(receipts) != len(ranks):
        raise ValueError("duplicate rank receipt")
    if any(row.get("schema") != SCHEMA for row in receipts):
        raise ValueError("rank receipt schema does not match the Signac Kaggle canary")
    if any(row.get("status") != "PASS" for row in receipts):
        raise ValueError("a worker did not report a passing canary status")
    candidates = {row.get("candidate") for row in receipts}
    spec_hashes = {row.get("model_spec_sha256") for row in receipts}
    configs = [row.get("config") for row in receipts]
    if len(candidates) != 1 or not isinstance(next(iter(candidates)), str):
        raise ValueError("rank receipts disagree on the candidate identity")
    candidate = str(next(iter(candidates)))
    if candidate not in SPEC_NAMES:
        raise ValueError("rank receipts name an unknown Signac candidate")
    if len(spec_hashes) != 1 or not isinstance(next(iter(spec_hashes)), str):
        raise ValueError("rank receipts disagree on the ModelSpec identity")
    model_spec_sha256 = str(next(iter(spec_hashes)))
    if len(model_spec_sha256) != 64 or any(c not in "0123456789abcdef" for c in model_spec_sha256):
        raise ValueError("rank receipts carry an invalid ModelSpec hash")
    if not isinstance(configs[0], dict) or any(config != configs[0] for config in configs):
        raise ValueError("rank receipts disagree on their frozen canary configuration")
    frozen_config = CanaryConfig(**configs[0])
    frozen_config.validate()
    if configs[0].get("expected_world_size") != expected_world_size:
        raise ValueError("frozen config world size disagrees with the aggregate request")
    if configs[0].get("expected_global_devices") != expected_global_devices:
        raise ValueError("frozen config TPU count disagrees with the aggregate request")
    if configs[0].get("candidate") != next(iter(candidates)):
        raise ValueError("rank receipt candidate does not match the frozen config")
    if any(row.get("source_tree_sha256") != frozen_config.source_tree_sha256 for row in receipts):
        raise ValueError("rank receipt source identity differs from the frozen canary configuration")
    runtime_identity = _aggregate_runtime_identity(receipts)
    from signac_100m.spec import (
        MODEL_SPEC, TPU_DEPTH_PRESERVING_CHALLENGER, TPU_TILED_CHALLENGER,
    )
    expected_specs = {
        "m102_primary": MODEL_SPEC,
        "tpu_depth_preserving_challenger": TPU_DEPTH_PRESERVING_CHALLENGER,
        "tpu_tiled_challenger": TPU_TILED_CHALLENGER,
    }
    expected_spec = expected_specs[candidate]
    if frozen_config.sequence_length > expected_spec.context_length:
        raise ValueError("canary sequence exceeds the registered candidate context")
    if model_spec_sha256 != expected_spec.sha256():
        raise ValueError("rank receipts do not match the registered ModelSpec for this candidate")
    if any(int(row.get("parameter_count", -1)) != expected_spec.parameter_receipt().total for row in receipts):
        raise ValueError("rank parameter count does not match the registered candidate geometry")
    if any(row.get("device_type") != "TPU" for row in receipts):
        raise ValueError("a worker did not report TPU device type")
    if any(row.get("bf16_probe_finite") is not True for row in receipts):
        raise ValueError("a worker failed its finite BF16 probe")
    if any(row.get("device_rng_progresses") is not True for row in receipts):
        raise ValueError("a worker failed its device RNG progression probe")
    parity_reports = [row.get("distributed_gradient_parity") for row in receipts]
    parity_fixture_hashes: set[str] = set()
    for ordinal, report in enumerate(parity_reports):
        if not isinstance(report, dict) or report.get("schema") != GRADIENT_PARITY_SCHEMA:
            raise ValueError("rank omitted the distributed-vs-single-device parity receipt")
        if report.get("status") != "PASS":
            raise ValueError(f"rank {ordinal} failed distributed gradient parity")
        if report.get("world_size") != expected_world_size:
            raise ValueError("gradient parity world size differs from the frozen topology")
        if report.get("rank_examples") != ordinal + 1:
            raise ValueError("gradient parity fixture did not retain its unequal per-rank count")
        fixture_sha256 = report.get("fixture_sha256")
        if (
            not isinstance(fixture_sha256, str)
            or len(fixture_sha256) != 64
            or any(character not in "0123456789abcdef" for character in fixture_sha256)
        ):
            raise ValueError("gradient parity fixture identity is not a SHA-256")
        parity_fixture_hashes.add(fixture_sha256)
        if report.get("denominator_policy") != (
            "sum_local_loss_over_global_example_count_then_all_reduce_sum"
        ):
            raise ValueError("gradient parity report used an unsupported reduction policy")
        expected_examples = expected_world_size * (expected_world_size + 1) // 2
        if report.get("global_examples") != expected_examples:
            raise ValueError("gradient parity did not use the complete unequal-rank reference batch")
        rank_reports = report.get("by_precision")
        if not isinstance(rank_reports, dict) or set(rank_reports) != set(PARITY_TOLERANCES):
            raise ValueError("gradient parity report omitted a frozen precision mode")
        for mode, tolerances in PARITY_TOLERANCES.items():
            mode_report = rank_reports[mode]
            if not isinstance(mode_report, dict) or mode_report.get("status") != "PASS":
                raise ValueError(f"rank {ordinal} failed the {mode} gradient parity probe")
            if mode_report.get("absolute_tolerance") != tolerances["absolute"]:
                raise ValueError("gradient parity absolute tolerance changed")
            if mode_report.get("relative_tolerance") != tolerances["relative"]:
                raise ValueError("gradient parity relative tolerance changed")
            if mode_report.get("gradient_elements") != PARITY_GRADIENT_ELEMENTS:
                raise ValueError("gradient parity gradient shape differs from the frozen oracle")
            for error_key in ("max_normalized_gradient_error", "normalized_loss_error"):
                error = float(mode_report.get(error_key, float("inf")))
                if not math.isfinite(error) or error > 1.0:
                    raise ValueError(f"rank {ordinal} exceeded the {mode} parity tolerance")
    if len(parity_fixture_hashes) != 1:
        raise ValueError("rank workers reconstructed different gradient parity fixtures")
    if any(int(row.get("global_device_count", -1)) != expected_global_devices for row in receipts):
        raise ValueError("worker global device count differs from the frozen Kaggle target")
    if any(int(row.get("world_size", -1)) != expected_world_size for row in receipts):
        raise ValueError("worker world size differs from the frozen Kaggle target")
    addressable_counts = [int(row.get("addressable_device_count", 0)) for row in receipts]
    if any(count <= 0 for count in addressable_counts):
        raise ValueError("a worker reports no addressable TPU devices")
    if len(set(addressable_counts)) != 1:
        raise ValueError("rank workers disagree on addressable TPU device count")
    for field in ("initial_parameter_sha256", "parameter_sha256", "optimizer_moment_sha256"):
        values = {str(row.get(field, "")) for row in receipts}
        if len(values) != 1 or "" in values:
            raise ValueError(f"replicas disagree or omit {field}")
        value = next(iter(values))
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"worker receipt has invalid {field}")
    sample_streams = [row.get("rank_sample_stream_sha256") for row in receipts]
    if any(not isinstance(value, str) or not value for value in sample_streams):
        raise ValueError("a rank omitted its deterministic synthetic sample stream hash")
    if len(set(sample_streams)) != expected_world_size:
        raise ValueError("rank-local synthetic sample streams are missing or duplicated")
    if any(
        len(str(value)) != 64 or any(character not in "0123456789abcdef" for character in str(value))
        for value in sample_streams
    ):
        raise ValueError("rank-local sample stream identity must be a SHA-256")
    expected_collective_sum = expected_world_size * (expected_world_size + 1) / 2
    collective_sums = [float(row.get("collective_probe_sum", float("nan"))) for row in receipts]
    if any(not math.isfinite(value) or abs(value - expected_collective_sum) > 1e-5
           for value in collective_sums):
        raise ValueError("a worker failed the cross-rank collective probe")
    if any(int(row.get("optimizer_step", -1)) <= 0 for row in receipts):
        raise ValueError("optimizer did not advance on every rank")
    if len({int(row["optimizer_step"]) for row in receipts}) != 1:
        raise ValueError("replica optimizer steps disagree")
    if expected_optimizer_step is not None and int(receipts[0]["optimizer_step"]) != expected_optimizer_step:
        raise ValueError("optimizer step count differs from the requested canary budget")
    update_count = len(receipts[0].get("rank_local_mean_loss_by_update", []))
    expected_updates = int(configs[0].get("optimizer_updates", -1))
    expected_steady_updates = expected_updates - 1
    if update_count == 0 or any(
        len(row.get("rank_local_mean_loss_by_update", [])) != update_count
        or update_count != expected_updates
        or len(row.get("global_grad_norm_preclip_by_update", [])) != update_count
        or any(not math.isfinite(float(value)) for value in row["rank_local_mean_loss_by_update"])
        or any(not math.isfinite(float(value)) for value in row["global_grad_norm_preclip_by_update"])
        for row in receipts
    ):
        raise ValueError("worker loss or gradient norm receipts are missing, nonfinite, or misaligned")
    steady_times = [list(map(float, row.get("steady_update_seconds", []))) for row in receipts]
    steady_count = len(steady_times[0]) if steady_times else 0
    if steady_count == 0 or any(
        len(times) != steady_count or steady_count != expected_steady_updates
        or any(not math.isfinite(value) or value <= 0 for value in times)
        for times in steady_times
    ):
        raise ValueError("synchronized steady update timings are missing or invalid")
    if steady_count < frozen_config.minimum_steady_updates:
        raise ValueError("canary did not meet its frozen minimum steady-update count")
    critical_path_seconds = [max(steady_times[rank][step] for rank in range(expected_world_size))
                             for step in range(steady_count)]
    local_tokens = int(receipts[0].get("local_supervised_tokens_per_update", 0))
    expected_local_tokens = (
        int(configs[0].get("microbatch_size", 0))
        * (int(configs[0].get("sequence_length", 0)) - 1)
        * int(configs[0].get("accumulation_steps", 0))
    )
    if local_tokens != expected_local_tokens:
        raise ValueError("local supervised-token count disagrees with the frozen canary shape")
    if local_tokens <= 0 or any(int(row.get("local_supervised_tokens_per_update", -1)) != local_tokens
                                for row in receipts):
        raise ValueError("rank token counts are missing or unequal")
    if any(
        int(row.get("global_supervised_tokens_per_update", -1)) != local_tokens * expected_world_size
        for row in receipts
    ):
        raise ValueError("global supervised-token count disagrees with ranks and workload")
    steady_mean_seconds = sum(critical_path_seconds) / len(critical_path_seconds)
    global_tokens_per_second = local_tokens * expected_world_size / steady_mean_seconds
    global_mean_loss = [
        sum(float(row["rank_local_mean_loss_by_update"][index]) for row in receipts)
        / expected_world_size
        for index in range(update_count)
    ]
    qualification_profile = (
        frozen_config.minimum_steady_updates >= QUALIFICATION_MIN_STEADY_UPDATES
    )
    peak_memory_by_rank: dict[str, int] = {}
    memory_limit_by_rank: dict[str, int] = {}
    memory_samples_complete = True
    for ordinal, row in enumerate(receipts):
        samples = row.get("device_memory_samples_by_update")
        if not isinstance(samples, list) or len(samples) != expected_updates:
            memory_samples_complete = False
            continue
        rank_peaks: list[int] = []
        rank_limits: list[int] = []
        for sample in samples:
            if not isinstance(sample, dict) or sample.get("status") != "AVAILABLE":
                memory_samples_complete = False
                continue
            peak, limit = sample.get("peak_bytes_used"), sample.get("bytes_limit")
            if type(peak) is not int or type(limit) is not int or peak <= 0 or limit <= 0:
                memory_samples_complete = False
                continue
            rank_peaks.append(peak)
            rank_limits.append(limit)
        if rank_peaks and rank_limits:
            peak_memory_by_rank[str(ordinal)] = max(rank_peaks)
            if len(set(rank_limits)) != 1:
                memory_samples_complete = False
            memory_limit_by_rank[str(ordinal)] = min(rank_limits)
    peak_memory_fraction = None
    peak_memory_status = "NOT_REQUIRED"
    if qualification_profile:
        expected_rank_keys = {str(rank) for rank in range(expected_world_size)}
        if (not memory_samples_complete or set(peak_memory_by_rank) != expected_rank_keys
                or set(memory_limit_by_rank) != expected_rank_keys):
            raise ValueError(
                "20-update qualification requires per-update XLA peak-memory and limit counters"
            )
        peak_memory_fraction = max(
            peak_memory_by_rank[str(rank)] / memory_limit_by_rank[str(rank)]
            for rank in range(expected_world_size)
        )
        if peak_memory_fraction > QUALIFICATION_MAX_PEAK_MEMORY_FRACTION:
            raise ValueError(
                "measured peak memory exceeds the frozen 85% per-device headroom limit"
            )
        peak_memory_status = "PASS"
    elif peak_memory_by_rank:
        peak_memory_status = "OBSERVED_NOT_QUALIFIED"
    host_memory_observation = _aggregate_host_memory_observations(
        receipts, expected_update_count=update_count,
    )
    return {
        "world_size": expected_world_size,
        "global_device_count": expected_global_devices,
        "addressable_device_count_per_worker": addressable_counts[0],
        "candidate": str(next(iter(candidates))),
        "source_tree_sha256": frozen_config.source_tree_sha256,
        "runtime_identity": runtime_identity,
        "model_spec_sha256": model_spec_sha256,
        "participating_ordinals": ranks,
        "all_ranks_report_tpu": True,
        "replica_parameters_match": True,
        "replica_optimizer_moments_match": True,
        "optimizer_step": int(receipts[0]["optimizer_step"]),
        "global_mean_loss_by_update_equal_token_rank_average": global_mean_loss,
        "steady_critical_path_seconds_by_update": critical_path_seconds,
        "steady_global_tokens_per_second": global_tokens_per_second,
        "measurement_profile": "QUALIFICATION" if qualification_profile else "SMOKE",
        "measured_steady_update_count": steady_count,
        "minimum_steady_updates_required": frozen_config.minimum_steady_updates,
        "peak_memory_status": peak_memory_status,
        "peak_memory_limit_fraction": QUALIFICATION_MAX_PEAK_MEMORY_FRACTION,
        "worst_peak_memory_fraction": peak_memory_fraction,
        "peak_memory_bytes_by_rank": peak_memory_by_rank,
        "memory_limit_bytes_by_rank": memory_limit_by_rank,
        "host_memory_observation": host_memory_observation,
        "all_core_update_verified": True,
        "distributed_gradient_parity_verified": True,
        "distributed_gradient_parity_max_normalized_error": {
            mode: max(
                float(report["by_precision"][mode]["max_normalized_gradient_error"])
                for report in parity_reports
            )
            for mode in PARITY_TOLERANCES
        },
        "distributed_gradient_parity_fixture_sha256": next(iter(parity_fixture_hashes)),
    }


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_memory_snapshot(xm: object, device: object) -> dict[str, object]:
    """Normalize the XLA memory counters without inventing a peak value."""

    try:
        raw = xm.get_memory_info(device)
    except Exception as exc:
        return {
            "status": "UNAVAILABLE",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    if not isinstance(raw, Mapping):
        return {"status": "UNAVAILABLE", "reason": "get_memory_info returned a non-mapping"}
    values: dict[str, int] = {}
    for name in ("bytes_used", "bytes_limit", "peak_bytes_used", "kb_free", "kb_total"):
        value = raw.get(name)
        if value is not None:
            try:
                values[name] = int(value)
            except (TypeError, ValueError, OverflowError):
                continue
    if "bytes_limit" not in values and values.get("kb_total", 0) > 0:
        values["bytes_limit"] = values["kb_total"] * 1024
    if ("bytes_used" not in values and values.get("kb_total", 0) > 0
            and "kb_free" in values):
        values["bytes_used"] = max(0, values["kb_total"] - values["kb_free"]) * 1024
    if values.get("peak_bytes_used", 0) > 0 and values.get("bytes_limit", 0) > 0:
        return {"status": "AVAILABLE", **values}
    return {
        "status": "PARTIAL",
        "reason": "XLA runtime does not expose both peak_bytes_used and bytes_limit",
        **values,
    }


def _read_host_memory_snapshot(
    *, resource_module: object | None = None, system: str | None = None,
    sample_point: str = HOST_MEMORY_SAMPLE_POINT,
) -> dict[str, object]:
    """Read this Linux worker's cumulative process RSS high-water in bytes.

    ``ru_maxrss`` is reported in KiB on Linux. It is a process-lifetime maximum,
    so it can capture a transient update allocation but is neither a per-update
    peak nor a host-wide memory total. This is diagnostic and never gates a run.
    """

    observed_system = platform.system() if system is None else system
    common: dict[str, object] = {
        "schema": HOST_MEMORY_SCHEMA,
        "metric": HOST_MEMORY_METRIC,
        "scope": HOST_MEMORY_SCOPE,
        "unit": "bytes",
        "sample_point": sample_point,
        "source": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
    }
    if observed_system != "Linux":
        return {
            **common,
            "status": "UNAVAILABLE",
            "reason": "ru_maxrss unit normalization is defined here only for Linux workers",
        }
    try:
        if resource_module is None:
            import resource as resource_module
        usage = resource_module.getrusage(resource_module.RUSAGE_SELF)
        raw_kib = float(usage.ru_maxrss)
        if not math.isfinite(raw_kib) or raw_kib <= 0:
            raise ValueError("ru_maxrss is not a finite positive KiB value")
        high_water_bytes = int(round(raw_kib * 1024))
    except Exception as exc:
        return {
            **common,
            "status": "UNAVAILABLE",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    return {
        **common,
        "status": "AVAILABLE",
        "high_water_rss_bytes": high_water_bytes,
    }


def _aggregate_host_memory_observations(
    receipts: list[dict[str, object]], *, expected_update_count: int,
) -> dict[str, object]:
    """Summarize post-hash worker RSS without treating it as a fit gate."""

    by_rank: dict[str, int] = {}
    all_final_samples_available = True
    update_sample_series_aligned = True
    for ordinal, row in enumerate(receipts):
        update_samples = row.get("host_memory_samples_by_update")
        if not isinstance(update_samples, list) or len(update_samples) != expected_update_count:
            update_sample_series_aligned = False
        else:
            for sample in update_samples:
                _valid_host_memory_sample(sample, sample_point=HOST_MEMORY_SAMPLE_POINT)
        final_sample = row.get("host_memory_after_final_hashes")
        final_value = _valid_host_memory_sample(
            final_sample, sample_point=HOST_MEMORY_FINAL_HASH_SAMPLE_POINT,
        )
        if final_value is None:
            all_final_samples_available = False
        else:
            # Keep rank values separate: RSS pages may be shared across workers.
            by_rank[str(ordinal)] = final_value
    expected_ranks = {str(index) for index in range(len(receipts))}
    if all_final_samples_available and set(by_rank) == expected_ranks:
        status = "OBSERVED"
    elif by_rank:
        status = "PARTIAL"
    else:
        status = "UNAVAILABLE"
    return {
        "schema": HOST_MEMORY_SCHEMA,
        "status": status,
        "metric": HOST_MEMORY_METRIC,
        "scope": HOST_MEMORY_SCOPE,
        "unit": "bytes",
        "sample_point": HOST_MEMORY_FINAL_HASH_SAMPLE_POINT,
        "high_water_rss_bytes_by_rank": by_rank,
        "update_boundary_sample_series_aligned": update_sample_series_aligned,
        "is_qualification_gate": False,
    }


def _valid_host_memory_sample(sample: object, *, sample_point: str) -> int | None:
    if not isinstance(sample, dict):
        if sample is None:
            return None
        raise ValueError("host memory observation must be an object")
    if (sample.get("schema") != HOST_MEMORY_SCHEMA
            or sample.get("metric") != HOST_MEMORY_METRIC
            or sample.get("scope") != HOST_MEMORY_SCOPE
            or sample.get("unit") != "bytes"
            or sample.get("sample_point") != sample_point):
        raise ValueError("host memory observation semantics are inconsistent")
    status = sample.get("status")
    if status == "AVAILABLE":
        value = sample.get("high_water_rss_bytes")
        if type(value) is not int or value <= 0:
            raise ValueError("available host RSS observation has an invalid byte count")
        return value
    if status == "UNAVAILABLE":
        if not isinstance(sample.get("reason"), str) or not sample["reason"].strip():
            raise ValueError("unavailable host RSS observation needs a reason")
        return None
    raise ValueError("host memory observation has an unsupported status")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _config_sha256(config: CanaryConfig) -> str:
    return _sha256(
        json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


RUNTIME_IDENTITY_FIELDS = (
    "python_version", "torch_version", "torch_xla_version", "platform",
)


def _torch_xla_version(torch_xla: object) -> str:
    try:
        return importlib_metadata.version("torch_xla")
    except importlib_metadata.PackageNotFoundError:
        value = getattr(torch_xla, "__version__", None)
        return value if isinstance(value, str) and value.strip() else "unknown"


def _aggregate_runtime_identity(receipts: list[dict[str, object]]) -> dict[str, object]:
    """Require a consistent, nonempty software/platform identity across ranks."""

    metadata: dict[str, str] = {}
    for field in RUNTIME_IDENTITY_FIELDS:
        values = [row.get(field) for row in receipts]
        if any(
            not isinstance(value, str)
            or not value.strip()
            or value.strip().lower() in {"unknown", "unavailable", "n/a", "none"}
            for value in values
        ):
            raise ValueError(f"rank receipts need known {field} metadata")
        if len(set(values)) != 1:
            raise ValueError(f"rank receipts disagree on {field} runtime metadata")
        metadata[field] = str(values[0])
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return {**metadata, "sha256": _sha256(encoded.encode("utf-8"))}


def _stream_seed(config: CanaryConfig, ordinal: int) -> int:
    if not 0 <= ordinal < config.expected_world_size:
        raise ValueError("synthetic stream rank is outside the frozen world")
    return config.seed + ordinal * 1_000_003


def _synthetic_cursor(
    *, config: CanaryConfig, ordinal: int, next_update_index: int,
) -> dict[str, object]:
    """Describe the exact next point in this rank's synthetic token stream."""

    if not 0 <= next_update_index <= config.optimizer_updates:
        raise ValueError("synthetic cursor update is outside the frozen canary")
    return {
        "schema": SYNTHETIC_CURSOR_SCHEMA,
        "config_sha256": _config_sha256(config),
        "rank": ordinal,
        "stream_seed": _stream_seed(config, ordinal),
        "next_update_index": next_update_index,
        "next_microstep_index": 0,
        "next_microbatch_index": next_update_index * config.accumulation_steps,
        "microsteps_per_update": config.accumulation_steps,
    }


def _validate_synthetic_cursor(
    cursor: object, *, config: CanaryConfig, ordinal: int, next_update_index: int,
) -> dict[str, object]:
    expected = _synthetic_cursor(
        config=config, ordinal=ordinal, next_update_index=next_update_index,
    )
    if not isinstance(cursor, dict) or cursor != expected:
        raise ValueError("rank stream cursor differs from the frozen continuation")
    return expected


def _generator_state_sha256(generator: object) -> str:
    state = generator.get_state()
    return _sha256(state.detach().cpu().contiguous().numpy().tobytes())


def _cpu_tree(value: object, torch: object) -> object:
    """Copy a nested model/optimizer state tree to ordinary CPU tensors."""

    tensor_type = torch.Tensor
    if isinstance(value, tensor_type):
        return value.detach().to(device="cpu").contiguous()
    if isinstance(value, dict):
        return {key: _cpu_tree(item, torch) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item, torch) for item in value)
    if isinstance(value, list):
        return [_cpu_tree(item, torch) for item in value]
    return value


def _write_restart_checkpoint(
    *,
    path: Path,
    model: object,
    optimizer: object,
    torch: object,
    config: CanaryConfig,
    model_spec_sha256: str,
    next_update_index: int,
    global_ordinal: int,
    world_size: int,
) -> str:
    """Write one replicated state at a completed optimizer-update boundary."""

    payload = {
        "schema": RESTART_SCHEMA,
        "candidate": config.candidate,
        "config_sha256": _config_sha256(config),
        "model_spec_sha256": model_spec_sha256,
        "completed_optimizer_updates": next_update_index,
        "next_update_index": next_update_index,
        "checkpoint_writer_ordinal": global_ordinal,
        "world_size": world_size,
        "model_state_dict": _cpu_tree(model.state_dict(), torch),
        "optimizer_state_dict": _cpu_tree(optimizer.state_dict(), torch),
        "data_cursor": {
            "schema": "anra-signac-replicated-update-cursor/v1",
            "next_update_index": next_update_index,
            "next_microstep_index": 0,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return _file_sha256(path)


def _write_rank_stream_checkpoint(
    *, path: Path, torch: object, config: CanaryConfig,
    model_spec_sha256: str, ordinal: int, world_size: int,
    generator: object, xla_rng_state: bytes, next_update_index: int,
) -> str:
    """Persist one rank's sampler cursor plus CPU and XLA RNG streams."""

    if world_size != config.expected_world_size:
        raise ValueError("rank stream checkpoint world differs from the frozen config")
    if not isinstance(xla_rng_state, bytes) or not xla_rng_state:
        raise ValueError("rank stream checkpoint requires nonempty serialized XLA RNG state")
    cursor = _synthetic_cursor(
        config=config, ordinal=ordinal, next_update_index=next_update_index,
    )
    payload = {
        "schema": RANK_STREAM_SCHEMA,
        "candidate": config.candidate,
        "config_sha256": _config_sha256(config),
        "model_spec_sha256": model_spec_sha256,
        "rank": ordinal,
        "world_size": world_size,
        "data_cursor": cursor,
        "generator_state": generator.get_state().detach().cpu().contiguous(),
        "generator_state_sha256": _generator_state_sha256(generator),
        "xla_rng_state": torch.tensor(list(xla_rng_state), dtype=torch.uint8),
        "xla_rng_state_sha256": _sha256(xla_rng_state),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return _file_sha256(path)


def _safe_load_rank_stream_checkpoint(path: Path, *, torch: object) -> dict[str, object]:
    """Load a rank-local stream state using tensor/primitive-only deserialization."""

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "controlled restart requires torch.load(weights_only=True) support"
        ) from exc
    expected_fields = {
        "schema", "candidate", "config_sha256", "model_spec_sha256",
        "rank", "world_size", "data_cursor", "generator_state",
        "generator_state_sha256", "xla_rng_state", "xla_rng_state_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ValueError("rank stream checkpoint schema is invalid")
    if payload.get("schema") != RANK_STREAM_SCHEMA:
        raise ValueError("rank stream checkpoint schema is invalid")
    state = payload.get("generator_state")
    if not isinstance(state, torch.Tensor) or state.dtype != torch.uint8 or state.ndim != 1:
        raise ValueError("rank stream checkpoint generator state must be a flat byte tensor")
    state_sha256 = payload.get("generator_state_sha256")
    if (
        not isinstance(state_sha256, str)
        or len(state_sha256) != 64
        or any(character not in "0123456789abcdef" for character in state_sha256)
    ):
        raise ValueError("rank stream checkpoint generator-state hash is invalid")
    xla_state = payload.get("xla_rng_state")
    if not isinstance(xla_state, torch.Tensor) or xla_state.dtype != torch.uint8 \
            or xla_state.ndim != 1 or xla_state.numel() == 0:
        raise ValueError("rank stream checkpoint XLA RNG state must be a nonempty byte tensor")
    xla_state_sha256 = payload.get("xla_rng_state_sha256")
    if (
        not isinstance(xla_state_sha256, str)
        or len(xla_state_sha256) != 64
        or any(character not in "0123456789abcdef" for character in xla_state_sha256)
        or _sha256(xla_state.cpu().contiguous().numpy().tobytes()) != xla_state_sha256
    ):
        raise ValueError("rank stream checkpoint XLA RNG-state hash is invalid")
    return payload


def _safe_load_restart_checkpoint(path: Path, *, torch: object) -> dict[str, object]:
    """Load only tensor/primitive checkpoint values from the canary's own file."""

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "controlled restart requires torch.load(weights_only=True) support"
        ) from exc
    expected_fields = {
        "schema", "candidate", "config_sha256", "model_spec_sha256",
        "completed_optimizer_updates", "next_update_index",
        "checkpoint_writer_ordinal", "world_size", "model_state_dict",
        "optimizer_state_dict", "data_cursor",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != expected_fields
        or payload.get("schema") != RESTART_SCHEMA
        or not isinstance(payload.get("model_state_dict"), dict)
        or not isinstance(payload.get("optimizer_state_dict"), dict)
    ):
        raise ValueError("controlled restart checkpoint schema is invalid")
    return payload


def aggregate_restart_receipts(
    baseline_receipts: list[dict[str, object]],
    resume_receipts: list[dict[str, object]],
    *,
    config: CanaryConfig,
    checkpoint_sha256: str,
    rank_stream_checkpoint_sha256: dict[str, str],
) -> dict[str, object]:
    """Compare fresh-group continuation with the uninterrupted target run."""

    config.validate()
    if not config.verify_restart:
        raise ValueError("restart receipts require verify_restart=True")
    if len(checkpoint_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in checkpoint_sha256
    ):
        raise ValueError("restart checkpoint identity must be a SHA-256")
    expected_ranks = list(range(config.expected_world_size))
    if set(rank_stream_checkpoint_sha256) != {str(rank) for rank in expected_ranks} or any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in rank_stream_checkpoint_sha256.values()
    ):
        raise ValueError("restart reference must bind one valid stream checkpoint per rank")
    for label, rows in (("baseline", baseline_receipts), ("restart", resume_receipts)):
        ranks = sorted(int(row.get("ordinal", -1)) for row in rows)
        if ranks != expected_ranks or len(rows) != len(expected_ranks):
            raise ValueError(f"{label} receipts must contain each worker ordinal exactly once")
        if any(row.get("status") != "PASS" for row in rows):
            raise ValueError(f"{label} worker did not pass")
    baseline = {int(row["ordinal"]): row for row in baseline_receipts}
    resumed = {int(row["ordinal"]): row for row in resume_receipts}
    if any(
        row.get("schema") != SCHEMA or row.get("candidate") != config.candidate
        for row in baseline.values()
    ):
        raise ValueError("uninterrupted receipts are not bound to this canary")
    if any(row.get("source_tree_sha256") != config.source_tree_sha256 for row in baseline.values()):
        raise ValueError("uninterrupted receipts are not bound to the frozen source identity")
    final_parameters = {row.get("parameter_sha256") for row in baseline.values()}
    final_moments = {row.get("optimizer_moment_sha256") for row in baseline.values()}
    if len(final_parameters) != 1 or None in final_parameters:
        raise ValueError("uninterrupted ranks do not agree on final model parameters")
    if len(final_moments) != 1 or None in final_moments:
        raise ValueError("uninterrupted ranks do not agree on final optimizer moments")
    expected_parameter_sha256 = str(next(iter(final_parameters)))
    expected_moment_sha256 = str(next(iter(final_moments)))
    for ordinal in expected_ranks:
        reference = baseline[ordinal]
        row = resumed[ordinal]
        if row.get("schema") != RESTART_SCHEMA or row.get("candidate") != config.candidate:
            raise ValueError("restart receipt schema or candidate identity mismatch")
        if row.get("source_tree_sha256") != config.source_tree_sha256:
            raise ValueError("restart receipt is not bound to the frozen source identity")
        for field in RUNTIME_IDENTITY_FIELDS:
            baseline_value = reference.get(field)
            if not isinstance(baseline_value, str) or not baseline_value.strip():
                raise ValueError(f"uninterrupted receipt omitted {field} runtime metadata")
            if row.get(field) != baseline_value:
                raise ValueError(f"fresh-group restart runtime differs on {field}")
        if row.get("device_type") != "TPU" or row.get("world_size") != config.expected_world_size:
            raise ValueError("restart receipt does not prove the frozen TPU worker topology")
        if row.get("config_sha256") != _config_sha256(config):
            raise ValueError("restart receipt is not bound to the frozen canary configuration")
        if row.get("checkpoint_sha256") != checkpoint_sha256:
            raise ValueError("restart workers did not restore the committed checkpoint bytes")
        if row.get("rank_stream_checkpoint_sha256") != rank_stream_checkpoint_sha256[str(ordinal)]:
            raise ValueError("restart worker did not restore its bound rank stream checkpoint")
        if row.get("next_update_index") != 1 or row.get("restored_optimizer_step") != 1:
            raise ValueError("restart did not continue from the expected update boundary")
        if row.get("optimizer_step") != config.optimizer_updates:
            raise ValueError("restart worker did not reach the frozen optimizer update count")
        if row.get("parameter_sha256") != expected_parameter_sha256:
            raise ValueError("fresh-group parameters differ from the uninterrupted trajectory")
        if row.get("optimizer_moment_sha256") != expected_moment_sha256:
            raise ValueError("fresh-group Adam moments differ from the uninterrupted trajectory")
        baseline_streams = reference.get("rank_sample_sha256_by_update")
        resumed_streams = row.get("rank_sample_sha256_by_update")
        if (
            not isinstance(baseline_streams, list)
            or len(baseline_streams) != config.optimizer_updates
            or any(
                not isinstance(update, list)
                or len(update) != config.accumulation_steps
                or any(
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(character not in "0123456789abcdef" for character in value)
                    for value in update
                )
                for update in baseline_streams
            )
        ):
            raise ValueError("uninterrupted rank sample cursor receipt is incomplete")
        if not isinstance(resumed_streams, list) or resumed_streams != baseline_streams[1:]:
            raise ValueError("fresh-group restart did not replay the expected next data stream")
        baseline_rng_states = reference.get("rank_stream_state_sha256_by_update")
        baseline_cursors = reference.get("data_cursor_by_update")
        baseline_xla_rng_states = reference.get("xla_rng_state_sha256_by_update")
        if (
            not isinstance(baseline_rng_states, list)
            or len(baseline_rng_states) != config.optimizer_updates
            or any(
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in baseline_rng_states
            )
        ):
            raise ValueError("uninterrupted rank RNG-state receipt is incomplete")
        if not isinstance(baseline_cursors, list) or len(baseline_cursors) != config.optimizer_updates:
            raise ValueError("uninterrupted rank sampler-cursor receipt is incomplete")
        if (
            not isinstance(baseline_xla_rng_states, list)
            or len(baseline_xla_rng_states) != config.optimizer_updates
            or any(not _is_sha256(value) for value in baseline_xla_rng_states)
        ):
            raise ValueError("uninterrupted rank XLA RNG-state receipt is incomplete")
        for update_number, cursor in enumerate(baseline_cursors, start=1):
            _validate_synthetic_cursor(
                cursor, config=config, ordinal=ordinal,
                next_update_index=update_number,
            )
        if row.get("restored_stream_state_sha256") != baseline_rng_states[0]:
            raise ValueError("fresh-group restart did not restore this rank's RNG state")
        if row.get("restored_data_cursor") != baseline_cursors[0]:
            raise ValueError("fresh-group restart did not restore this rank's data cursor")
        if row.get("stream_state_sha256_after_updates") != baseline_rng_states[-1]:
            raise ValueError("fresh-group restart RNG state differs from uninterrupted training")
        if row.get("next_data_cursor") != baseline_cursors[-1]:
            raise ValueError("fresh-group restart cursor differs from uninterrupted training")
        if row.get("restored_xla_rng_state_sha256") != baseline_xla_rng_states[0]:
            raise ValueError("fresh-group restart did not restore this rank's XLA RNG state")
        baseline_probe_sha256 = reference.get("restart_xla_rng_probe_sha256")
        if not _is_sha256(baseline_probe_sha256) or row.get(
            "xla_rng_replay_probe_sha256"
        ) != baseline_probe_sha256:
            raise ValueError("fresh-group restart XLA RNG probe differs from uninterrupted run")
        baseline_after_probe_sha256 = reference.get(
            "restart_xla_rng_state_sha256_after_probe"
        )
        if not _is_sha256(baseline_after_probe_sha256) or row.get(
            "xla_rng_state_sha256_after_probe"
        ) != baseline_after_probe_sha256:
            raise ValueError("fresh-group XLA RNG state after probe differs from uninterrupted run")
        if row.get("xla_rng_state_sha256_after_updates") != baseline_xla_rng_states[-1]:
            raise ValueError("fresh-group final XLA RNG state differs from uninterrupted training")
    return {
        "schema": RESTART_SCHEMA,
        "status": "PASS",
        "checkpoint_sha256": checkpoint_sha256,
        "source_tree_sha256": config.source_tree_sha256,
        "runtime_identity": _aggregate_runtime_identity(baseline_receipts),
        "world_size": config.expected_world_size,
        "restored_optimizer_step": 1,
        "final_optimizer_step": config.optimizer_updates,
        "parameters_match_uninterrupted": True,
        "optimizer_moments_match_uninterrupted": True,
        "rank_local_sample_streams_match_uninterrupted_suffix": True,
        "rank_local_rng_states_match_uninterrupted": True,
        "rank_local_synthetic_cursors_match_uninterrupted": True,
        "rank_local_xla_rng_replay_matches_uninterrupted": True,
        "production_exact_resume_certified": False,
    }


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def compare_gradient_parity(
    distributed_gradient: list[float],
    reference_gradient: list[float],
    *,
    distributed_loss: float,
    reference_loss: float,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, object]:
    """Compare a replica-reduced gradient with an independent global-batch oracle."""

    if not distributed_gradient or len(distributed_gradient) != len(reference_gradient):
        raise ValueError("parity gradients must be nonempty and have identical dimensions")
    if (
        not math.isfinite(absolute_tolerance)
        or not math.isfinite(relative_tolerance)
        or absolute_tolerance <= 0
        or relative_tolerance < 0
    ):
        raise ValueError("parity tolerances must be finite and nonnegative")
    if not math.isfinite(distributed_loss) or not math.isfinite(reference_loss):
        raise ValueError("parity losses must be finite")

    absolute_errors: list[float] = []
    normalized_errors: list[float] = []
    for actual, expected in zip(distributed_gradient, reference_gradient):
        if not math.isfinite(actual) or not math.isfinite(expected):
            raise ValueError("parity gradients must be finite")
        absolute_error = abs(actual - expected)
        threshold = absolute_tolerance + relative_tolerance * abs(expected)
        absolute_errors.append(absolute_error)
        normalized_errors.append(absolute_error / threshold)
    loss_absolute_error = abs(distributed_loss - reference_loss)
    loss_threshold = absolute_tolerance + relative_tolerance * abs(reference_loss)
    max_normalized_error = max(normalized_errors)
    normalized_loss_error = loss_absolute_error / loss_threshold
    return {
        "status": (
            "PASS"
            if max_normalized_error <= 1.0 and normalized_loss_error <= 1.0
            else "FAIL"
        ),
        "gradient_elements": len(distributed_gradient),
        "max_absolute_gradient_error": max(absolute_errors),
        "max_normalized_gradient_error": max_normalized_error,
        "loss_absolute_error": loss_absolute_error,
        "normalized_loss_error": normalized_loss_error,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
    }


def _run_distributed_gradient_parity_probe(
    *,
    torch: object,
    xm: object,
    rank_backend: object,
    device: object,
    world_size: int,
    ordinal: int,
    seed: int,
) -> dict[str, object]:
    """Compare an XLA SUM-reduced unequal-rank gradient with a full local batch."""

    autocast_device_type = getattr(device, "type", str(device).split(":", 1)[0])
    def fresh_probe_model():
        model = torch.nn.Linear(
            PARITY_FEATURE_DIMENSION, PARITY_CLASS_COUNT, bias=True
        ).to(device)
        weight = (
            torch.arange(
                PARITY_FEATURE_DIMENSION * PARITY_CLASS_COUNT, dtype=torch.float32
            ).reshape(PARITY_CLASS_COUNT, PARITY_FEATURE_DIMENSION)
            - 15.5
        ) / 50.0
        bias = (
            torch.arange(PARITY_CLASS_COUNT, dtype=torch.float32) - 1.5
        ) / 10.0
        with torch.no_grad():
            model.weight.copy_(weight.to(device))
            model.bias.copy_(bias.to(device))
        return model

    by_precision: dict[str, dict[str, object]] = {}
    for mode, tolerances in PARITY_TOLERANCES.items():
        def compute_local_gradient():
            rank_batches: list[tuple[object, object]] = []
            for rank in range(world_size):
                count = rank + 1
                generator = torch.Generator(device="cpu").manual_seed(
                    seed + 8_000_017 + rank * 10_009
                )
                features = torch.randn(
                    (count, PARITY_FEATURE_DIMENSION), generator=generator,
                    dtype=torch.float32,
                )
                labels = torch.randint(
                    0, PARITY_CLASS_COUNT, (count,), generator=generator,
                    dtype=torch.int64,
                )
                rank_batches.append((features, labels))
            rank_data_identities = [
                {
                    "rank": rank,
                    "examples": int(features.shape[0]),
                    "features_sha256": _sha256(features.contiguous().numpy().tobytes()),
                    "labels_sha256": _sha256(labels.contiguous().numpy().tobytes()),
                }
                for rank, (features, labels) in enumerate(rank_batches)
            ]
            fixture_sha256 = _sha256(
                json.dumps(
                    rank_data_identities, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            )
            global_examples = sum(features.shape[0] for features, _ in rank_batches)
            local_features, local_labels = rank_batches[ordinal]
            local_features = local_features.to(device)
            local_labels = local_labels.to(device)
            all_features = torch.cat(
                [features for features, _ in rank_batches], dim=0,
            ).to(device)
            all_labels = torch.cat(
                [labels for _, labels in rank_batches], dim=0,
            ).to(device)
            local_model = fresh_probe_model()
            local_model.zero_grad(set_to_none=True)
            if mode == "bf16_autocast":
                with torch.autocast(device_type=autocast_device_type, dtype=torch.bfloat16):
                    local_logits = local_model(local_features)
            else:
                local_logits = local_model(local_features)
            local_loss_sum = torch.nn.functional.cross_entropy(
                local_logits.float(), local_labels, reduction="sum"
            )
            (local_loss_sum / global_examples).backward()
            if any(parameter.grad is None for parameter in local_model.parameters()):
                raise RuntimeError("parity probe has an incomplete local gradient layout")
            return (
                fixture_sha256, global_examples, all_features, all_labels,
                local_model, local_loss_sum,
            )

        (
            fixture_sha256, global_examples, all_features, all_labels,
            local_model, local_loss_sum,
        ) = rank_backend.run_local_stage(
            stage="local_update", callback=compute_local_gradient,
        )
        rank_backend.all_reduce_sum_gradients(local_model)
        distributed_loss_sum = xm.all_reduce(xm.REDUCE_SUM, local_loss_sum.detach())

        def compare_with_global_batch():
            distributed_gradient = torch.cat(
                [
                    parameter.grad.detach().float().to("cpu").reshape(-1)
                    for parameter in local_model.parameters()
                ]
            ).tolist()
            distributed_loss = float(distributed_loss_sum.cpu().item()) / global_examples
            reference_model = fresh_probe_model()
            reference_model.zero_grad(set_to_none=True)
            if mode == "bf16_autocast":
                with torch.autocast(device_type=autocast_device_type, dtype=torch.bfloat16):
                    reference_logits = reference_model(all_features)
            else:
                reference_logits = reference_model(all_features)
            reference_loss = torch.nn.functional.cross_entropy(
                reference_logits.float(), all_labels, reduction="mean"
            )
            reference_loss.backward()
            reference_gradient = torch.cat(
                [
                    parameter.grad.detach().float().to("cpu").reshape(-1)
                    for parameter in reference_model.parameters()
                ]
            ).tolist()
            summary = compare_gradient_parity(
                distributed_gradient,
                reference_gradient,
                distributed_loss=distributed_loss,
                reference_loss=float(reference_loss.detach().cpu().item()),
                absolute_tolerance=float(tolerances["absolute"]),
                relative_tolerance=float(tolerances["relative"]),
            )
            summary["global_examples"] = global_examples
            if summary["status"] != "PASS":
                raise RuntimeError(f"{mode} distributed gradient parity probe failed: {summary}")
            return summary

        summary = rank_backend.run_local_stage(
            stage="local_update", callback=compare_with_global_batch,
        )
        summary["fixture_sha256"] = fixture_sha256
        by_precision[mode] = summary

    return {
        "schema": GRADIENT_PARITY_SCHEMA,
        "status": "PASS",
        "world_size": world_size,
        "rank_examples": ordinal + 1,
        "global_examples": sum(range(1, world_size + 1)),
        "fixture_sha256": by_precision[next(iter(PARITY_TOLERANCES))]["fixture_sha256"],
        "denominator_policy": "sum_local_loss_over_global_example_count_then_all_reduce_sum",
        "by_precision": by_precision,
    }


def _run_synthetic_update(
    *,
    update_index: int,
    ordinal: int,
    config: CanaryConfig,
    spec: object,
    model: object,
    optimizer: object,
    device: object,
    positions: object,
    attention_mask: object,
    world_size: int,
    torch: object,
    rank_backend: object,
    sample_generator: object,
) -> tuple[float, float, float, list[str]]:
    """Run one update from a rank-local, stateful synthetic token stream."""

    started = time.perf_counter()
    autocast_device_type = getattr(device, "type", str(device).split(":", 1)[0])

    def compute_local_gradients():
        optimizer.zero_grad(set_to_none=True)
        loss_total = None
        sample_hashes: list[str] = []
        for _micro_index in range(config.accumulation_steps):
            token_ids_cpu = torch.randint(
                4, spec.vocabulary_size,
                (config.microbatch_size, config.sequence_length),
                generator=sample_generator, dtype=torch.int64,
            )
            sample_hashes.append(_sha256(token_ids_cpu.numpy().tobytes()))
            token_ids = token_ids_cpu.to(device)
            segment_ids = torch.zeros_like(token_ids)
            with torch.autocast(device_type=autocast_device_type, dtype=torch.bfloat16):
                hidden = model.forward_hidden(
                    token_ids, positions, attention_mask,
                    use_activation_checkpointing=True,
                )
                loss, supervised_count = causal_lm_loss_from_hidden(
                    hidden, model.embedding.weight, token_ids, segment_ids,
                    torch_module=torch,
                )
            expected_supervised_count = (
                config.microbatch_size * (config.sequence_length - 1)
            )
            if supervised_count != expected_supervised_count:
                raise RuntimeError("synthetic loss did not supervise every shifted token")
            scaled_loss = loss / config.accumulation_steps
            scaled_loss.backward()
            loss_total = (
                scaled_loss.detach() if loss_total is None
                else loss_total + scaled_loss.detach()
            )
        parameters_with_grad = [
            parameter for parameter in model.parameters() if parameter.grad is not None
        ]
        if not parameters_with_grad:
            raise RuntimeError("no gradients were materialized")
        if len(parameters_with_grad) != sum(1 for _ in model.parameters()):
            raise RuntimeError("a trainable parameter is missing its local gradient")
        return loss_total, sample_hashes

    loss_total, sample_hashes = rank_backend.run_local_stage(
        stage="local_update", callback=compute_local_gradients,
    )
    # The adapter verifies identical parameter/gradient layouts on every rank
    # before entering the gradient collective.
    rank_backend.all_reduce_sum_gradients(model, scale=1.0 / world_size)

    def apply_global_update():
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        update_loss = float(loss_total.cpu().item())
        grad_norm = float(norm.detach().cpu().item())
        if not math.isfinite(update_loss):
            raise RuntimeError("nonfinite loss after synchronized update")
        if not math.isfinite(grad_norm):
            raise RuntimeError("nonfinite globally reduced gradient norm")
        return update_loss, grad_norm

    # Vote again after the optimizer boundary so a rank-local update failure
    # cannot strand its peers at the next update's pre-collective vote.
    update_loss, grad_norm = rank_backend.run_local_stage(
        stage="local_update", callback=apply_global_update,
    )
    elapsed = time.perf_counter() - started
    return update_loss, grad_norm, elapsed, sample_hashes


def _worker_impl(_local_index: int, raw_config: dict[str, object]) -> None:
    """One process per TPU device; all XLA device access stays in this worker."""

    import torch
    import torch_xla
    import torch_xla.runtime as xr
    import torch_xla.core.xla_model as xm

    config = CanaryConfig(**raw_config)
    config.validate()
    device = xm.xla_device()
    device_type = str(xr.device_type()).upper()
    world_size = int(xr.world_size())
    ordinal = int(xr.global_ordinal())
    global_device_count = int(xr.global_runtime_device_count())
    addressable_device_count = int(xr.addressable_runtime_device_count())
    if device_type != "TPU":
        raise RuntimeError(f"PJRT reported {device_type}; Kaggle TPU canary requires TPU")
    if world_size != config.expected_world_size:
        raise RuntimeError(f"worker world size {world_size} != {config.expected_world_size}")
    if global_device_count != config.expected_global_devices:
        raise RuntimeError(
            f"global TPU count {global_device_count} != {config.expected_global_devices}"
        )
    if not 0 <= ordinal < world_size:
        raise RuntimeError(f"invalid global ordinal {ordinal} for world size {world_size}")

    from signac_100m.spec import (
        MODEL_SPEC, TPU_DEPTH_PRESERVING_CHALLENGER, TPU_TILED_CHALLENGER,
    )
    from v5_model.core import assert_receipt, initialize, packed_layout
    from v5_training.mutation import moment_fingerprint, optimizer_step, parameter_sha
    from v5_training.optimizer import build_adamw_optimizer, validate_parameter_ownership
    from v5_training.xla_adapter import XLAReplicatedBackend, capture_xla_rng_state

    rank_backend = XLAReplicatedBackend(
        replica_backend=None,
        replicas=world_size,
        world_size=world_size,
        torch_module=torch,
    )

    specs = {
        "m102_primary": MODEL_SPEC,
        "tpu_depth_preserving_challenger": TPU_DEPTH_PRESERVING_CHALLENGER,
        "tpu_tiled_challenger": TPU_TILED_CHALLENGER,
    }
    spec = specs[config.candidate]
    if config.sequence_length > spec.context_length:
        raise ValueError("canary sequence exceeds the model's frozen context length")

    # Deterministic identical initialization plus rank-specific batches tests
    # that the all-reduce, rather than duplicate data, synchronizes replicas.
    def build_model_and_optimizer():
        model = initialize(spec, config.seed, torch_module=torch).train().to(device)
        assert_receipt(model, spec)
        optimizer = build_adamw_optimizer(
            model, lr=config.learning_rate, torch_module=torch,
        )
        validate_parameter_ownership(model, optimizer)
        return model, optimizer, parameter_sha(model, torch_module=torch)

    model, optimizer, initial_sha = rank_backend.run_local_stage(
        stage="local_update", callback=build_model_and_optimizer,
    )

    def run_device_probes():
        rng_probe_a = torch.rand((32,), dtype=torch.float32, device=device)
        rng_probe_b = torch.rand((32,), dtype=torch.float32, device=device)
        bf16_probe = torch.randn((32, 32), dtype=torch.bfloat16, device=device)
        rng_sha_a = _sha256(rng_probe_a.detach().cpu().contiguous().numpy().tobytes())
        rng_sha_b = _sha256(rng_probe_b.detach().cpu().contiguous().numpy().tobytes())
        device_rng_progresses = rng_sha_a != rng_sha_b
        bf16_probe_finite = bool(torch.isfinite(bf16_probe.float()).all().item())
        if not device_rng_progresses or not bf16_probe_finite:
            raise RuntimeError("device RNG or BF16 runtime probe failed")
        return rng_sha_a, rng_sha_b, device_rng_progresses, bf16_probe_finite

    rng_sha_a, rng_sha_b, device_rng_progresses, bf16_probe_finite = (
        rank_backend.run_local_stage(stage="local_update", callback=run_device_probes)
    )

    probe = rank_backend.run_local_stage(
        stage="local_update",
        callback=lambda: torch.tensor(
            [float(ordinal + 1)], dtype=torch.float32, device=device,
        ),
    )
    reduced_probe = xm.all_reduce(xm.REDUCE_SUM, probe)
    expected_sum = world_size * (world_size + 1) / 2

    def validate_collective_probe():
        collective_sum = float(reduced_probe.cpu().item())
        if abs(collective_sum - expected_sum) > 1e-5:
            raise RuntimeError(
                f"collective probe returned {collective_sum}; expected {expected_sum}"
            )
        return collective_sum

    collective_sum = rank_backend.run_local_stage(
        stage="local_update", callback=validate_collective_probe,
    )

    gradient_parity = _run_distributed_gradient_parity_probe(
        torch=torch,
        xm=xm,
        rank_backend=rank_backend,
        device=device,
        world_size=world_size,
        ordinal=ordinal,
        seed=config.seed,
    )
    def prepare_attention_layout():
        segment_ids = torch.zeros(
            (config.microbatch_size, config.sequence_length),
            dtype=torch.int64, device=device,
        )
        return packed_layout(segment_ids, torch_module=torch)

    positions, attention_mask = rank_backend.run_local_stage(
        stage="local_update", callback=prepare_attention_layout,
    )
    before_parameters = initial_sha
    update_seconds: list[float] = []
    mean_losses: list[float] = []
    grad_norms: list[float] = []
    sample_hashes: list[str] = []
    sample_hashes_by_update: list[list[str]] = []
    stream_state_sha256_by_update: list[str] = []
    data_cursor_by_update: list[dict[str, object]] = []
    xla_rng_state_sha256_by_update: list[str] = []
    memory_samples_by_update: list[dict[str, object]] = []
    host_memory_samples_by_update: list[dict[str, object]] = []
    restart_checkpoint_sha256: str | None = None
    restart_xla_rng_probe_sha256: str | None = None
    restart_xla_rng_state_sha256_after_probe: str | None = None
    stream_generator = rank_backend.run_local_stage(
        stage="local_update",
        callback=lambda: torch.Generator(device="cpu").manual_seed(
            _stream_seed(config, ordinal)
        ),
    )
    supervised_per_update = (
        config.microbatch_size * (config.sequence_length - 1) * config.accumulation_steps
    )
    for update_index in range(config.optimizer_updates):
        update_loss, grad_norm, elapsed, update_sample_hashes = _run_synthetic_update(
            update_index=update_index,
            ordinal=ordinal,
            config=config,
            spec=spec,
            model=model,
            optimizer=optimizer,
            device=device,
            positions=positions,
            attention_mask=attention_mask,
            world_size=world_size,
            torch=torch,
            rank_backend=rank_backend,
            sample_generator=stream_generator,
        )
        update_seconds.append(elapsed)
        mean_losses.append(update_loss)
        grad_norms.append(grad_norm)
        sample_hashes.extend(update_sample_hashes)
        sample_hashes_by_update.append(update_sample_hashes)

        def capture_rank_update_state():
            memory_sample = _read_memory_snapshot(xm, device)
            host_memory_sample = _read_host_memory_snapshot()
            stream_state_sha256 = _generator_state_sha256(stream_generator)
            cursor = _synthetic_cursor(
                config=config, ordinal=ordinal, next_update_index=update_index + 1,
            )
            xla_rng_payload = None
            if config.verify_restart:
                xla_rng_payload = capture_xla_rng_state(
                    device=str(next(model.parameters()).device),
                    torch_module=torch,
                )
            return (
                memory_sample, host_memory_sample, stream_state_sha256,
                cursor, xla_rng_payload,
            )

        (
            memory_sample,
            host_memory_sample,
            stream_state_sha256,
            cursor,
            xla_rng_payload,
        ) = rank_backend.run_local_stage(
            stage="local_update", callback=capture_rank_update_state,
        )
        memory_samples_by_update.append(memory_sample)
        host_memory_samples_by_update.append(host_memory_sample)
        stream_state_sha256_by_update.append(stream_state_sha256)
        data_cursor_by_update.append(cursor)
        if xla_rng_payload is not None:
            xla_rng_state_sha256_by_update.append(_sha256(xla_rng_payload))
        if config.verify_restart and update_index == 0:
            def write_restart_checkpoints():
                restart_hash = None
                if ordinal == 0:
                    restart_hash = _write_restart_checkpoint(
                        path=Path(config.output_dir) / config.candidate / "restart.pt",
                        model=model,
                        optimizer=optimizer,
                        torch=torch,
                        config=config,
                        model_spec_sha256=spec.sha256(),
                        next_update_index=update_index + 1,
                        global_ordinal=ordinal,
                        world_size=world_size,
                    )
                _write_rank_stream_checkpoint(
                    path=Path(config.output_dir) / config.candidate
                         / f"restart-rank-{ordinal:02d}.pt",
                    torch=torch,
                    config=config,
                    model_spec_sha256=spec.sha256(),
                    ordinal=ordinal,
                    world_size=world_size,
                    generator=stream_generator,
                    xla_rng_state=xla_rng_payload,
                    next_update_index=update_index + 1,
                )
                return restart_hash

            restart_checkpoint_sha256 = rank_backend.run_local_stage(
                stage="local_update", callback=write_restart_checkpoints,
            )
        if config.verify_restart and update_index == 0:
            rendezvous = getattr(xm, "rendezvous", None)
            if not callable(rendezvous):
                raise RuntimeError("XLA runtime lacks the barrier required by the restart canary")
            rendezvous("signac-controlled-restart-checkpoint-written")

            def replay_restart_rng_probe():
                xla_rng_probe = torch.rand((32,), dtype=torch.float32, device=device)
                probe_sha256 = _sha256(
                    xla_rng_probe.detach().cpu().contiguous().numpy().tobytes()
                )
                state_sha256 = _sha256(capture_xla_rng_state(
                    device=str(next(model.parameters()).device), torch_module=torch,
                ))
                return probe_sha256, state_sha256

            (
                restart_xla_rng_probe_sha256,
                restart_xla_rng_state_sha256_after_probe,
            ) = rank_backend.run_local_stage(
                stage="local_update", callback=replay_restart_rng_probe,
            )
        if ordinal == 0:
            print(json.dumps({
                "candidate": config.candidate,
                "ordinal": ordinal,
                "update": update_index + 1,
                "updates_total": config.optimizer_updates,
                "loss": update_loss,
                "preclip_grad_norm": grad_norms[-1],
                "synchronized_update_seconds": elapsed,
            }, sort_keys=True), flush=True)

    after_parameters = parameter_sha(model, torch_module=torch)
    if before_parameters == after_parameters:
        raise RuntimeError("model parameters did not change during all-core canary")
    moment_sha = moment_fingerprint(optimizer, torch_module=torch)
    step = optimizer_step(optimizer)
    memory = (memory_samples_by_update[-1] if memory_samples_by_update else {
        "status": "UNAVAILABLE", "reason": "no completed updates"
    })
    xm.mark_step()
    host_memory_after_final_hashes = _read_host_memory_snapshot(
        sample_point=HOST_MEMORY_FINAL_HASH_SAMPLE_POINT,
    )

    steady_seconds = update_seconds[1:]
    receipt = {
        "schema": SCHEMA,
        "status": "PASS",
        "candidate": config.candidate,
        "source_tree_sha256": config.source_tree_sha256,
        "model_spec_sha256": spec.sha256(),
        "parameter_count": spec.parameter_receipt().total,
        "device_type": device_type,
        "device": str(device),
        "global_ordinal": ordinal,
        "ordinal": ordinal,
        "world_size": world_size,
        "global_device_count": global_device_count,
        "addressable_device_count": addressable_device_count,
        "collective_probe_sum": collective_sum,
        "expected_collective_probe_sum": expected_sum,
        "distributed_gradient_parity": gradient_parity,
        "bf16_probe_finite": bf16_probe_finite,
        "device_rng_probe_sha256_a": rng_sha_a,
        "device_rng_probe_sha256_b": rng_sha_b,
        "device_rng_progresses": device_rng_progresses,
        "initial_parameter_sha256": initial_sha,
        "parameter_sha256": after_parameters,
        "optimizer_moment_sha256": moment_sha,
        "optimizer_step": step,
        "torch_version": torch.__version__,
        "torch_xla_version": _torch_xla_version(torch_xla),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "config": asdict(config),
        "microsteps_per_update": config.accumulation_steps,
        "local_supervised_tokens_per_update": supervised_per_update,
        "global_supervised_tokens_per_update": supervised_per_update * world_size,
        "rank_sample_sha256": sample_hashes,
        "rank_sample_sha256_by_update": sample_hashes_by_update,
        "rank_stream_state_sha256_by_update": stream_state_sha256_by_update,
        "data_cursor_by_update": data_cursor_by_update,
        "xla_rng_state_sha256_by_update": xla_rng_state_sha256_by_update,
        "restart_xla_rng_probe_sha256": restart_xla_rng_probe_sha256,
        "restart_xla_rng_state_sha256_after_probe": restart_xla_rng_state_sha256_after_probe,
        "rank_sample_stream_sha256": _sha256("".join(sample_hashes).encode("ascii")),
        "rank_sample_stream_is_distinct_by_seed": True,
        "rank_local_mean_loss_by_update": mean_losses,
        "global_grad_norm_preclip_by_update": grad_norms,
        "compile_inclusive_first_update_seconds": update_seconds[0],
        "steady_update_seconds": steady_seconds,
        "device_memory": memory,
        "device_memory_samples_by_update": memory_samples_by_update,
        "host_memory_samples_by_update": host_memory_samples_by_update,
        "host_memory_after_final_hashes": host_memory_after_final_hashes,
        "host_memory_observation_semantics": {
            "schema": HOST_MEMORY_SCHEMA,
            "metric": HOST_MEMORY_METRIC,
            "scope": HOST_MEMORY_SCOPE,
            "unit": "bytes",
            "update_sample_point": HOST_MEMORY_SAMPLE_POINT,
            "final_sample_point": HOST_MEMORY_FINAL_HASH_SAMPLE_POINT,
            "qualification_gate": False,
        },
        "activation_checkpointing": True,
        "precision": "BF16 autocast with FP32 model parameters and Adam state",
        "synthetic_input": True,
        "checkpoint_resume_tested": False,
        "restart_checkpoint_sha256": restart_checkpoint_sha256,
        "limitations": [
            "This is a bounded synthetic all-core canary, not production training or capability evidence.",
            "Per-rank receipts prove participation and replicated updates; exact multi-process restart remains a separate gate.",
            "The optional restart now exercises rank-local CPU and XLA RNG restore, but its synthetic random probe does not certify production stochastic operations.",
            "Memory receipts are point-in-time runtime counters; production fit needs measured peak headroom at the frozen workload.",
            "Host RSS is a per-worker process high-water observation and does not certify host-wide memory fit.",
            "The canary captures host RSS after its final model/Adam hashes; production captures hashes every update and still needs target measurement.",
        ],
    }
    output = Path(config.output_dir) / config.candidate / f"rank-{ordinal:02d}.json"
    _atomic_json(output, receipt)


def _worker(local_index: int, raw_config: dict[str, object]) -> None:
    """Write a rank-scoped failure receipt before propagating worker errors."""

    config = CanaryConfig(**raw_config)
    try:
        _worker_impl(local_index, raw_config)
    except Exception as exc:
        ordinal = int(local_index)
        try:
            import torch_xla.runtime as xr
            ordinal = int(xr.global_ordinal())
        except Exception:
            pass
        failure = {
            "schema": SCHEMA,
            "status": "FAIL",
            "candidate": config.candidate,
            "source_tree_sha256": config.source_tree_sha256,
            "ordinal": ordinal,
            "local_index": int(local_index),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "config": asdict(config),
        }
        _atomic_json(
            Path(config.output_dir) / config.candidate / f"rank-{ordinal:02d}.json",
            failure,
        )
        raise


def _restore_restart_worker_state(
    *,
    config: CanaryConfig,
    candidate_dir: Path,
    device: object,
    ordinal: int,
    torch: object,
    spec: object,
) -> dict[str, object]:
    """Load one rank's restart state before peers enter the next collective."""

    from v5_model.core import assert_receipt, initialize, packed_layout
    from v5_training.mutation import optimizer_step
    from v5_training.optimizer import build_adamw_optimizer, validate_parameter_ownership
    from v5_training.xla_adapter import capture_xla_rng_state, restore_xla_rng_state

    checkpoint_path = candidate_dir / "restart.pt"
    reference_path = candidate_dir / "restart_reference.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    checkpoint_sha256 = _file_sha256(checkpoint_path)
    if reference.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("restart checkpoint bytes differ from the parent reference")
    if reference.get("config_sha256") != _config_sha256(config):
        raise ValueError("restart configuration differs from its parent reference")
    if reference.get("source_tree_sha256") != config.source_tree_sha256:
        raise ValueError("restart source identity differs from its parent reference")

    payload = _safe_load_restart_checkpoint(checkpoint_path, torch=torch)
    expected_payload = {
        "candidate": config.candidate,
        "config_sha256": _config_sha256(config),
        "model_spec_sha256": spec.sha256(),
        "completed_optimizer_updates": 1,
        "next_update_index": 1,
        "world_size": config.expected_world_size,
    }
    if any(payload.get(key) != value for key, value in expected_payload.items()):
        raise ValueError("restart payload identity or cursor is inconsistent")
    if payload.get("data_cursor") != {
        "schema": "anra-signac-replicated-update-cursor/v1",
        "next_update_index": 1,
        "next_microstep_index": 0,
    }:
        raise ValueError("restart checkpoint omitted the next-update data cursor")

    rank_stream_path = candidate_dir / f"restart-rank-{ordinal:02d}.pt"
    expected_rank_stream_hashes = reference.get("rank_stream_checkpoint_sha256")
    if not isinstance(expected_rank_stream_hashes, dict):
        raise ValueError("restart reference omitted per-rank stream checkpoint identities")
    expected_rank_stream_hash = expected_rank_stream_hashes.get(str(ordinal))
    if not isinstance(expected_rank_stream_hash, str) or _file_sha256(
        rank_stream_path
    ) != expected_rank_stream_hash:
        raise ValueError("rank stream checkpoint bytes differ from the parent reference")
    stream_payload = _safe_load_rank_stream_checkpoint(rank_stream_path, torch=torch)
    if any(stream_payload.get(key) != value for key, value in {
        "candidate": config.candidate,
        "config_sha256": _config_sha256(config),
        "model_spec_sha256": spec.sha256(),
        "rank": ordinal,
        "world_size": config.expected_world_size,
    }.items()):
        raise ValueError("rank stream checkpoint identity does not match this worker")
    restored_cursor = _validate_synthetic_cursor(
        stream_payload.get("data_cursor"), config=config, ordinal=ordinal,
        next_update_index=1,
    )
    generator_state = stream_payload.get("generator_state")
    if not isinstance(generator_state, torch.Tensor):
        raise ValueError("rank stream checkpoint omitted the generator state tensor")
    stream_generator = torch.Generator(device="cpu").manual_seed(
        _stream_seed(config, ordinal)
    )
    stream_generator.set_state(generator_state.detach().cpu())
    restored_stream_state_sha256 = _generator_state_sha256(stream_generator)
    if restored_stream_state_sha256 != stream_payload.get("generator_state_sha256"):
        raise ValueError("rank stream generator-state hash does not match its tensor")

    model = initialize(spec, config.seed, torch_module=torch).train().to(device)
    assert_receipt(model, spec)
    optimizer = build_adamw_optimizer(model, lr=config.learning_rate, torch_module=torch)
    validate_parameter_ownership(model, optimizer)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    restored_step = optimizer_step(optimizer)
    if restored_step != 1:
        raise ValueError(f"checkpoint optimizer step is {restored_step}, expected one")

    segment_ids = torch.zeros(
        (config.microbatch_size, config.sequence_length), dtype=torch.int64, device=device,
    )
    positions, attention_mask = packed_layout(segment_ids, torch_module=torch)
    xla_rng_state = stream_payload.get("xla_rng_state")
    if not isinstance(xla_rng_state, torch.Tensor):
        raise ValueError("rank stream checkpoint omitted the XLA RNG state tensor")
    xla_rng_payload = xla_rng_state.detach().cpu().contiguous().numpy().tobytes()
    restored_xla_rng_state_sha256 = _sha256(xla_rng_payload)
    restore_xla_rng_state(
        xla_rng_payload,
        device=str(next(model.parameters()).device),
        torch_module=torch,
    )
    if _sha256(capture_xla_rng_state(
        device=str(next(model.parameters()).device), torch_module=torch,
    )) != restored_xla_rng_state_sha256:
        raise ValueError("restored XLA RNG state differs from the rank stream checkpoint")
    xla_rng_probe = torch.rand((32,), dtype=torch.float32, device=device)
    xla_rng_replay_probe_sha256 = _sha256(
        xla_rng_probe.detach().cpu().contiguous().numpy().tobytes()
    )
    xla_rng_state_sha256_after_probe = _sha256(capture_xla_rng_state(
        device=str(next(model.parameters()).device), torch_module=torch,
    ))
    return {
        "checkpoint_sha256": checkpoint_sha256,
        "expected_rank_stream_hash": expected_rank_stream_hash,
        "payload": payload,
        "restored_cursor": restored_cursor,
        "stream_generator": stream_generator,
        "restored_stream_state_sha256": restored_stream_state_sha256,
        "model": model,
        "optimizer": optimizer,
        "restored_step": restored_step,
        "positions": positions,
        "attention_mask": attention_mask,
        "restored_xla_rng_state_sha256": restored_xla_rng_state_sha256,
        "xla_rng_replay_probe_sha256": xla_rng_replay_probe_sha256,
        "xla_rng_state_sha256_after_probe": xla_rng_state_sha256_after_probe,
    }


def _resume_worker_impl(_local_index: int, raw_config: dict[str, object]) -> None:
    """Restore one shared replicated checkpoint in a fresh XLA process group."""

    import torch
    import torch_xla
    import platform
    import torch_xla.runtime as xr
    import torch_xla.core.xla_model as xm

    config = CanaryConfig(**raw_config)
    config.validate()
    if not config.verify_restart:
        raise ValueError("resume worker requires verify_restart=True")
    device = xm.xla_device()
    ordinal = int(xr.global_ordinal())
    world_size = int(xr.world_size())
    device_type = str(xr.device_type()).upper()
    if device_type != "TPU" or world_size != config.expected_world_size:
        raise RuntimeError("fresh process group does not match the frozen Kaggle TPU topology")
    if int(xr.global_runtime_device_count()) != config.expected_global_devices:
        raise RuntimeError("fresh process group global TPU count changed")

    from signac_100m.spec import (
        MODEL_SPEC, TPU_DEPTH_PRESERVING_CHALLENGER, TPU_TILED_CHALLENGER,
    )
    from v5_training.mutation import moment_fingerprint, optimizer_step, parameter_sha
    from v5_training.xla_adapter import XLAReplicatedBackend

    rank_backend = XLAReplicatedBackend(
        replica_backend=None,
        replicas=world_size,
        world_size=world_size,
        torch_module=torch,
    )

    spec = {
        "m102_primary": MODEL_SPEC,
        "tpu_depth_preserving_challenger": TPU_DEPTH_PRESERVING_CHALLENGER,
        "tpu_tiled_challenger": TPU_TILED_CHALLENGER,
    }[config.candidate]
    candidate_dir = Path(config.output_dir) / config.candidate
    restored = rank_backend.run_local_stage(
        stage="local_update",
        callback=lambda: _restore_restart_worker_state(
            config=config,
            candidate_dir=candidate_dir,
            device=device,
            ordinal=ordinal,
            torch=torch,
            spec=spec,
        ),
    )
    checkpoint_sha256 = restored["checkpoint_sha256"]
    expected_rank_stream_hash = restored["expected_rank_stream_hash"]
    payload = restored["payload"]
    restored_cursor = restored["restored_cursor"]
    stream_generator = restored["stream_generator"]
    restored_stream_state_sha256 = restored["restored_stream_state_sha256"]
    model = restored["model"]
    optimizer = restored["optimizer"]
    restored_step = restored["restored_step"]
    positions = restored["positions"]
    attention_mask = restored["attention_mask"]
    restored_xla_rng_state_sha256 = restored["restored_xla_rng_state_sha256"]
    xla_rng_replay_probe_sha256 = restored["xla_rng_replay_probe_sha256"]
    xla_rng_state_sha256_after_probe = restored["xla_rng_state_sha256_after_probe"]
    losses: list[float] = []
    grad_norms: list[float] = []
    update_seconds: list[float] = []
    sample_hashes_by_update: list[list[str]] = []
    for update_index in range(int(payload["next_update_index"]), config.optimizer_updates):
        loss, grad_norm, elapsed, sample_hashes = _run_synthetic_update(
            update_index=update_index,
            ordinal=ordinal,
            config=config,
            spec=spec,
            model=model,
            optimizer=optimizer,
            device=device,
            positions=positions,
            attention_mask=attention_mask,
            world_size=world_size,
            torch=torch,
            rank_backend=rank_backend,
            sample_generator=stream_generator,
        )
        losses.append(loss)
        grad_norms.append(grad_norm)
        update_seconds.append(elapsed)
        sample_hashes_by_update.append(sample_hashes)
    final_step = optimizer_step(optimizer)
    xla_rng_state_sha256_after_updates = _sha256(capture_xla_rng_state(
        device=str(next(model.parameters()).device), torch_module=torch,
    ))
    receipt = {
        "schema": RESTART_SCHEMA,
        "status": "PASS",
        "candidate": config.candidate,
        "source_tree_sha256": config.source_tree_sha256,
        "torch_version": torch.__version__,
        "torch_xla_version": _torch_xla_version(torch_xla),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "ordinal": ordinal,
        "device_type": device_type,
        "world_size": world_size,
        "config_sha256": _config_sha256(config),
        "checkpoint_sha256": checkpoint_sha256,
        "rank_stream_checkpoint_sha256": expected_rank_stream_hash,
        "next_update_index": int(payload["next_update_index"]),
        "restored_optimizer_step": restored_step,
        "optimizer_step": final_step,
        "parameter_sha256": parameter_sha(model, torch_module=torch),
        "optimizer_moment_sha256": moment_fingerprint(optimizer, torch_module=torch),
        "restored_stream_state_sha256": restored_stream_state_sha256,
        "restored_data_cursor": restored_cursor,
        "restored_xla_rng_state_sha256": restored_xla_rng_state_sha256,
        "xla_rng_replay_probe_sha256": xla_rng_replay_probe_sha256,
        "xla_rng_state_sha256_after_probe": xla_rng_state_sha256_after_probe,
        "xla_rng_state_sha256_after_updates": xla_rng_state_sha256_after_updates,
        "stream_state_sha256_after_updates": _generator_state_sha256(stream_generator),
        "next_data_cursor": _synthetic_cursor(
            config=config, ordinal=ordinal,
            next_update_index=config.optimizer_updates,
        ),
        "rank_sample_sha256_by_update": sample_hashes_by_update,
        "rank_local_mean_loss_by_update": losses,
        "global_grad_norm_preclip_by_update": grad_norms,
        "update_seconds": update_seconds,
        "synthetic_input": True,
        "limitations": [
            "This verifies one deterministic synthetic update-boundary restart in a fresh TPU process group.",
            "Per-rank CPU/XLA generator state and a synthetic stream cursor are restored; this does not certify the production packed-data sampler.",
            "Bucket-lane cursor, durable checkpoint custody, and full-model numerical parity remain separate gates.",
        ],
    }
    _atomic_json(candidate_dir / f"resume-rank-{ordinal:02d}.json", receipt)


def _resume_worker(local_index: int, raw_config: dict[str, object]) -> None:
    config = CanaryConfig(**raw_config)
    try:
        _resume_worker_impl(local_index, raw_config)
    except Exception as exc:
        failure = {
            "schema": RESTART_SCHEMA,
            "status": "FAIL",
            "candidate": config.candidate,
            "source_tree_sha256": config.source_tree_sha256,
            "ordinal": int(local_index),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _atomic_json(
            Path(config.output_dir) / config.candidate / f"resume-rank-{int(local_index):02d}.json",
            failure,
        )
        raise


def run(config: CanaryConfig) -> dict[str, object]:
    config.validate()
    from signac_100m.source_identity import build_source_identity

    live_identity = build_source_identity(Path(__file__).resolve().parents[1])
    if config.source_tree_sha256 != live_identity["source_tree_sha256"]:
        raise ValueError(
            "frozen source_tree_sha256 does not match the live Signac source identity"
        )
    try:
        import torch_xla
    except ImportError as exc:
        raise RuntimeError("Kaggle TPU canary requires the Kaggle-provided torch_xla runtime") from exc
    if not callable(getattr(torch_xla, "launch", None)):
        raise RuntimeError("installed torch_xla has no launch API; use the Kaggle matched TPU image")

    output_dir = Path(config.output_dir).resolve()
    candidate_dir = output_dir / config.candidate
    if candidate_dir.exists():
        raise FileExistsError(
            f"candidate receipt directory already exists: {candidate_dir}; choose a fresh output directory"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.verify_restart:
        from signac_100m.spec import (
            MODEL_SPEC, TPU_DEPTH_PRESERVING_CHALLENGER, TPU_TILED_CHALLENGER,
        )

        spec = {
            "m102_primary": MODEL_SPEC,
            "tpu_depth_preserving_challenger": TPU_DEPTH_PRESERVING_CHALLENGER,
            "tpu_tiled_challenger": TPU_TILED_CHALLENGER,
        }[config.candidate]
        minimum_free_bytes = math.ceil(spec.parameter_receipt().total * 12 * 1.25)
        if shutil.disk_usage(output_dir).free < minimum_free_bytes:
            raise RuntimeError(
                "insufficient free disk for the bounded model-plus-Adam restart checkpoint"
            )
    candidate_dir.mkdir(parents=True)
    # The callable is a top-level importable function. Device acquisition is
    # intentionally deferred to _worker, as required by torch_xla.launch.
    torch_xla.launch(_worker, args=(asdict(config),))

    rank_paths = sorted(candidate_dir.glob("rank-*.json"))
    receipts = [json.loads(path.read_text(encoding="utf-8")) for path in rank_paths]
    aggregate = aggregate_rank_receipts(
        receipts,
        expected_world_size=config.expected_world_size,
        expected_global_devices=config.expected_global_devices,
        expected_optimizer_step=config.optimizer_updates,
    )
    restart_aggregate = None
    restart_checkpoint = None
    restart_checkpoint_sha256 = None
    rank_stream_checkpoint_hashes: dict[str, str] = {}
    if config.verify_restart:
        restart_checkpoint = candidate_dir / "restart.pt"
        if not restart_checkpoint.is_file():
            raise FileNotFoundError("rank zero did not publish the controlled restart checkpoint")
        checkpoint_sha256 = _file_sha256(restart_checkpoint)
        restart_checkpoint_sha256 = checkpoint_sha256
        rank_stream_paths = sorted(candidate_dir.glob("restart-rank-*.pt"))
        expected_rank_names = [
            f"restart-rank-{ordinal:02d}.pt"
            for ordinal in range(config.expected_world_size)
        ]
        if [path.name for path in rank_stream_paths] != expected_rank_names:
            raise ValueError("controlled restart is missing an exact per-rank stream checkpoint set")
        rank_stream_checkpoint_hashes = {
            str(ordinal): _file_sha256(candidate_dir / f"restart-rank-{ordinal:02d}.pt")
            for ordinal in range(config.expected_world_size)
        }
        reference = {
            "schema": RESTART_SCHEMA,
            "config_sha256": _config_sha256(config),
            "source_tree_sha256": config.source_tree_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "expected_final_parameter_sha256": receipts[0]["parameter_sha256"],
            "expected_final_optimizer_moment_sha256": receipts[0]["optimizer_moment_sha256"],
            **{field: receipts[0][field] for field in RUNTIME_IDENTITY_FIELDS},
            "rank_stream_checkpoint_sha256": rank_stream_checkpoint_hashes,
            "rank_sample_sha256_by_update": {
                str(row["ordinal"]): row["rank_sample_sha256_by_update"]
                for row in receipts
            },
            "rank_stream_state_sha256_by_update": {
                str(row["ordinal"]): row["rank_stream_state_sha256_by_update"]
                for row in receipts
            },
            "data_cursor_by_update": {
                str(row["ordinal"]): row["data_cursor_by_update"]
                for row in receipts
            },
        }
        _atomic_json(candidate_dir / "restart_reference.json", reference)
        torch_xla.launch(_resume_worker, args=(asdict(config),))
        resume_paths = sorted(candidate_dir.glob("resume-rank-*.json"))
        resume_receipts = [
            json.loads(path.read_text(encoding="utf-8")) for path in resume_paths
        ]
        restart_aggregate = aggregate_restart_receipts(
            receipts,
            resume_receipts,
            config=config,
            checkpoint_sha256=checkpoint_sha256,
            rank_stream_checkpoint_sha256=rank_stream_checkpoint_hashes,
        )
    result: dict[str, object] = {
        "schema": SCHEMA,
        "status": "PASS",
        "candidate": config.candidate,
        "source_tree_sha256": config.source_tree_sha256,
        "config": asdict(config),
        "aggregate": aggregate,
        "controlled_restart": restart_aggregate,
        "restart_checkpoint": str(restart_checkpoint) if restart_checkpoint else None,
        "restart_checkpoint_sha256": restart_checkpoint_sha256,
        "rank_stream_checkpoints": [
            {
                "ordinal": int(ordinal),
                "path": f"restart-rank-{int(ordinal):02d}.pt",
                "sha256": sha256,
            }
            for ordinal, sha256 in sorted(rank_stream_checkpoint_hashes.items())
        ],
        "rank_receipts": [str(path) for path in rank_paths],
        "replica_consistency": {
            "initial_parameter_sha256": receipts[0]["initial_parameter_sha256"],
            "parameter_sha256": receipts[0]["parameter_sha256"],
            "optimizer_moment_sha256": receipts[0]["optimizer_moment_sha256"],
        },
        "synthetic_input": True,
        "production_training_authorized": False,
        "limitations": receipts[0]["limitations"],
    }
    _atomic_json(candidate_dir / "aggregate.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=sorted(SPEC_NAMES), default="m102_primary")
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument("--output-dir", default="/kaggle/working/signac_100m_distributed")
    parser.add_argument("--expected-global-devices", type=int, default=8)
    parser.add_argument("--expected-world-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=73_011)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--optimizer-updates", type=int, default=2)
    parser.add_argument("--minimum-steady-updates", type=int, default=1)
    parser.add_argument("--verify-restart", action="store_true")
    args = parser.parse_args()
    config = CanaryConfig(
        candidate=args.candidate,
        output_dir=args.output_dir,
        expected_global_devices=args.expected_global_devices,
        expected_world_size=args.expected_world_size,
        seed=args.seed,
        sequence_length=args.sequence_length,
        microbatch_size=args.microbatch_size,
        accumulation_steps=args.accumulation_steps,
        optimizer_updates=args.optimizer_updates,
        minimum_steady_updates=args.minimum_steady_updates,
        verify_restart=args.verify_restart,
        source_tree_sha256=args.source_tree_sha256,
    )
    result = run(config)
    print(json.dumps({
        "status": result["status"],
        "candidate": result["candidate"],
        "aggregate": result["aggregate"],
        "receipt": str(Path(config.output_dir) / config.candidate / "aggregate.json"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
