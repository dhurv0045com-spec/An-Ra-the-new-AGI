import hashlib
import json
from dataclasses import asdict

import pytest

from signac_100m.spec import (
    MODEL_SPEC,
    TPU_DEPTH_PRESERVING_CHALLENGER,
    TPU_TILED_CHALLENGER,
    candidate_receipts,
    parameter_receipt,
    resource_estimate,
)
from signac_100m.source_identity import build_source_identity
from tools.signac_100m_preflight import REPO_ROOT, _manifest_gate, build_report
from v5_training.kaggle_tpu_canary import (
    CanaryConfig,
    GRADIENT_PARITY_SCHEMA,
    HOST_MEMORY_FINAL_HASH_SAMPLE_POINT,
    HOST_MEMORY_METRIC,
    HOST_MEMORY_SAMPLE_POINT,
    HOST_MEMORY_SCHEMA,
    HOST_MEMORY_SCOPE,
    PARITY_GRADIENT_ELEMENTS,
    PARITY_TOLERANCES,
    SCHEMA as CANARY_SCHEMA,
    aggregate_rank_receipts,
)

SOURCE_IDENTITY = build_source_identity(REPO_ROOT)


@pytest.fixture(autouse=True)
def _use_one_source_snapshot_for_preflight_tests(monkeypatch):
    from signac_100m import source_identity

    monkeypatch.setattr(
        source_identity, "build_source_identity", lambda _root: SOURCE_IDENTITY
    )


def test_signac_uses_supported_100m_class_geometry_and_exact_receipt():
    MODEL_SPEC.assert_valid()
    receipt = parameter_receipt()
    assert receipt["embedding"] == 15_728_640
    assert receipt["attention_per_layer"] == 1_228_800
    assert receipt["ffn_per_layer"] == 3_072_000
    assert receipt["total"] == 101_790_080
    assert MODEL_SPEC.tied_embeddings is True
    assert MODEL_SPEC.vocabulary_size == 24_576


def test_resource_estimate_labels_approximations_and_token_prior():
    estimate = resource_estimate()
    parameters = MODEL_SPEC.parameter_receipt().total
    assert estimate["schema"] == "anra-signac-100m-resource-estimate/v2"
    assert estimate["parameters_exact"] == parameters == 101_790_080
    assert estimate["model_parameter_bytes"] == 4 * parameters
    assert estimate["gradient_bytes"] == 4 * parameters
    assert estimate["adam_moment_bytes"] == 8 * parameters
    assert estimate["core_parameter_gradient_adam_bytes_approx"] == 16 * parameters
    assert estimate["checkpoint_bytes_params_plus_moments"] == 12 * parameters
    assert "FP32 persistent parameters" in estimate["precision_assumption"]
    assert "not a peak-memory estimate" in estimate["estimate_scope"]
    assert "activations" in estimate["caveat"]
    assert estimate["attention_score_tensor_bytes_bf16_per_replica_per_active_layer"] == 335_544_320


def test_tpu_tiled_challenger_is_an_explicit_unpromoted_candidate():
    TPU_TILED_CHALLENGER.assert_valid()
    assert TPU_TILED_CHALLENGER.parameter_receipt().total == 100_303_104
    assert TPU_TILED_CHALLENGER.width % 128 == 0
    assert (TPU_TILED_CHALLENGER.kv_heads * TPU_TILED_CHALLENGER.head_dimension) % 128 == 0
    assert TPU_TILED_CHALLENGER.ffn_width % 128 == 0
    receipts = candidate_receipts()
    assert set(receipts) == {
        "m102_primary", "tpu_tiled_challenger", "tpu_depth_preserving_challenger"
    }
    assert receipts["m102_primary"]["resources"]["parameters_exact"] == 101_790_080


def test_depth_preserving_challenger_changes_only_ffn_width():
    challenger = TPU_DEPTH_PRESERVING_CHALLENGER
    challenger.assert_valid()
    assert challenger.parameter_receipt().total == 99_332_480
    assert challenger.ffn_width % 128 == 0
    assert challenger.width == MODEL_SPEC.width
    assert challenger.layers == MODEL_SPEC.layers
    assert challenger.query_heads == MODEL_SPEC.query_heads
    assert challenger.kv_heads == MODEL_SPEC.kv_heads
    assert challenger.context_length == MODEL_SPEC.context_length
    assert challenger.vocabulary_size == MODEL_SPEC.vocabulary_size


def test_preflight_never_authorizes_training_without_external_gates():
    report = build_report(target="tpu")
    assert report["verdict"] == "BLOCKED"
    assert report["training_authorized"] is False
    assert report["gates"]["target_runtime"]["state"] == "TPU_EVIDENCE_REQUIRED"
    assert set(report["blockers"]) == {"data", "evaluation", "runtime"}


def _canary_receipt_payload(profile="SMOKE", host_memory_available=True):
    source_tree_sha256 = SOURCE_IDENTITY["source_tree_sha256"]
    steady_updates = 1 if profile == "SMOKE" else 20
    optimizer_updates = steady_updates + 1
    config = asdict(CanaryConfig(
        optimizer_updates=optimizer_updates,
        minimum_steady_updates=steady_updates,
        verify_restart=profile == "SMOKE",
        source_tree_sha256=source_tree_sha256,
    ))
    rank_rows = _canary_rank_receipts(
        config, profile, host_memory_available=host_memory_available,
    )
    aggregate = aggregate_rank_receipts(
        rank_rows, expected_world_size=8, expected_global_devices=8,
        expected_optimizer_step=optimizer_updates,
    )
    runtime_identity = aggregate["runtime_identity"]
    restart = None
    if profile == "SMOKE":
        restart = {
            "schema": "anra-signac-kaggle-controlled-restart/v5",
            "status": "PASS",
            "checkpoint_sha256": "d" * 64,
            "source_tree_sha256": source_tree_sha256,
            "runtime_identity": runtime_identity,
            "world_size": 8,
            "restored_optimizer_step": 1,
            "final_optimizer_step": optimizer_updates,
            "parameters_match_uninterrupted": True,
            "optimizer_moments_match_uninterrupted": True,
            "rank_local_sample_streams_match_uninterrupted": True,
            "rank_local_rng_states_match_uninterrupted": True,
            "rank_local_synthetic_cursors_match_uninterrupted": True,
            "rank_local_xla_rng_replay_matches_uninterrupted": True,
            "production_exact_resume_certified": False,
        }
    return {
        "schema": CANARY_SCHEMA,
        "status": "PASS",
        "candidate": "m102_primary",
        "source_tree_sha256": source_tree_sha256,
        "config": config,
        "aggregate": aggregate,
        "controlled_restart": restart,
        "rank_receipts": [f"rank-{rank:02d}.json" for rank in range(8)],
        "production_training_authorized": False,
    }


def _canary_rank_receipts(
    config, profile, runtime_identity=None, host_memory_available=True,
):
    if runtime_identity is None:
        runtime_identity = {
            "python_version": "3.11.0",
            "torch_version": "2.6.0",
            "torch_xla_version": "2.6.0",
            "platform": "Linux-test",
        }
    updates = config["optimizer_updates"]
    steady_updates = updates - 1
    world_size = config["expected_world_size"]
    parity = lambda rank: {
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
    }
    rows = []
    for rank in range(world_size):
        row = {
            "ordinal": rank,
            "global_ordinal": rank,
            "status": "PASS",
            "schema": CANARY_SCHEMA,
            "candidate": "m102_primary",
            "source_tree_sha256": config["source_tree_sha256"],
            **runtime_identity,
            "model_spec_sha256": MODEL_SPEC.sha256(),
            "parameter_count": MODEL_SPEC.parameter_receipt().total,
            "config": config,
            "device_type": "TPU",
            "world_size": world_size,
            "global_device_count": world_size,
            "addressable_device_count": 1,
            "initial_parameter_sha256": "a" * 64,
            "parameter_sha256": "b" * 64,
            "optimizer_moment_sha256": "c" * 64,
            "optimizer_step": updates,
            "rank_sample_stream_sha256": hashlib.sha256(
                f"rank-stream-{rank}".encode()
            ).hexdigest(),
            "collective_probe_sum": world_size * (world_size + 1) / 2,
            "bf16_probe_finite": True,
            "device_rng_progresses": True,
            "xla_rng_state_sha256_by_update": [
                hashlib.sha256(f"rank-{rank}-xla-state-{update}".encode()).hexdigest()
                for update in range(updates)
            ],
            "restart_xla_rng_probe_sha256": hashlib.sha256(
                f"rank-{rank}-xla-probe".encode()
            ).hexdigest(),
            "restart_xla_rng_state_sha256_after_probe": hashlib.sha256(
                f"rank-{rank}-xla-after-probe".encode()
            ).hexdigest(),
            "rank_local_mean_loss_by_update": [1.0] * updates,
            "global_grad_norm_preclip_by_update": [0.5] * updates,
            "distributed_gradient_parity": parity(rank),
            "steady_update_seconds": [0.25] * steady_updates,
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
                for _ in range(updates)
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
        if not host_memory_available:
            for sample in row["host_memory_samples_by_update"]:
                sample["status"] = "UNAVAILABLE"
                sample.pop("high_water_rss_bytes")
                sample["reason"] = "host telemetry unavailable"
            final_sample = row["host_memory_after_final_hashes"]
            final_sample["status"] = "UNAVAILABLE"
            final_sample.pop("high_water_rss_bytes")
            final_sample["reason"] = "host telemetry unavailable"
        if profile == "QUALIFICATION":
            row["device_memory_samples_by_update"] = [
                {
                    "status": "AVAILABLE",
                    "bytes_used": 80,
                    "bytes_limit": 100,
                    "peak_bytes_used": 80,
                }
                for _ in range(updates)
            ]
        rows.append(row)
    return rows


def _write_canary_receipt(tmp_path, payload, name="canary.json"):
    bundle_name = name[:-5] if name.endswith(".json") else name
    bundle_dir = tmp_path / bundle_name
    bundle_dir.mkdir()
    path = bundle_dir / "aggregate.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    config = payload["config"]
    profile = "QUALIFICATION" if config["minimum_steady_updates"] == 20 else "SMOKE"
    runtime_identity = payload["aggregate"]["runtime_identity"]
    runtime_fields = ("python_version", "torch_version", "torch_xla_version", "platform")
    rank_rows = _canary_rank_receipts(
        config, profile, {field: runtime_identity[field] for field in runtime_fields},
        host_memory_available=(
            payload["aggregate"]["host_memory_observation"]["status"] != "UNAVAILABLE"
        ),
    )
    for rank, row in enumerate(rank_rows):
        (bundle_dir / f"rank-{rank:02d}.json").write_text(json.dumps(row), encoding="utf-8")
    return path


def test_preflight_reviews_hash_bound_smoke_receipt_without_clearing_production_gate(tmp_path):
    receipt = _write_canary_receipt(tmp_path, _canary_receipt_payload())

    report = build_report(target="tpu", runtime_receipt=receipt)
    runtime = report["gates"]["target_runtime"]
    evidence = runtime["canary_evidence"]

    assert evidence["state"] == "PRESENT_FOR_REVIEW"
    assert evidence["profile"] == "SMOKE"
    assert evidence["host_memory_observation"]["status"] == "OBSERVED"
    assert evidence["path"] == str(receipt.resolve())
    assert evidence["sha256"] == hashlib.sha256(receipt.read_bytes()).hexdigest()
    assert len(evidence["rank_receipts"]) == 8
    for rank, rank_evidence in enumerate(evidence["rank_receipts"]):
        rank_path = receipt.parent / f"rank-{rank:02d}.json"
        assert rank_evidence["ordinal"] == rank
        assert rank_evidence["sha256"] == hashlib.sha256(rank_path.read_bytes()).hexdigest()
    assert runtime["state"] == "TPU_EVIDENCE_REQUIRED"
    assert report["verdict"] == "BLOCKED"
    assert "runtime" in report["blockers"]
    assert report["training_authorized"] is False


def test_preflight_reviews_qualification_only_when_20_updates_and_memory_pass(tmp_path):
    receipt = _write_canary_receipt(
        tmp_path, _canary_receipt_payload("QUALIFICATION"), "qualification.json"
    )

    report = build_report(target="tpu", qualification_receipt=receipt)
    runtime = report["gates"]["target_runtime"]
    evidence = runtime["qualification_evidence"]

    assert evidence["state"] == "PRESENT_FOR_REVIEW"
    assert evidence["profile"] == "QUALIFICATION"
    assert evidence["measured_steady_update_count"] == 20
    assert runtime["state"] == "TPU_EVIDENCE_REQUIRED"
    assert report["training_authorized"] is False


def test_preflight_keeps_host_rss_observational_when_device_gate_passes(tmp_path):
    receipt = _write_canary_receipt(
        tmp_path,
        _canary_receipt_payload("QUALIFICATION", host_memory_available=False),
        "qualification-host-rss-unavailable.json",
    )

    report = build_report(target="tpu", qualification_receipt=receipt)
    evidence = report["gates"]["target_runtime"]["qualification_evidence"]

    assert evidence["state"] == "PRESENT_FOR_REVIEW"
    assert evidence["host_memory_observation"]["status"] == "UNAVAILABLE"
    assert report["training_authorized"] is False


def test_preflight_rejects_mixed_smoke_and_qualification_runtime_identities(tmp_path):
    smoke = _canary_receipt_payload("SMOKE")
    qualification = _canary_receipt_payload("QUALIFICATION")
    metadata = {
        "python_version": "3.11.0",
        "torch_version": "2.6.0",
        "torch_xla_version": "2.7.0",
        "platform": "Linux-test",
    }
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    changed_identity = {**metadata, "sha256": hashlib.sha256(encoded).hexdigest()}
    qualification["aggregate"]["runtime_identity"] = changed_identity
    smoke_path = _write_canary_receipt(tmp_path, smoke, "smoke.json")
    qualification_path = _write_canary_receipt(
        tmp_path, qualification, "qualification-mismatched.json"
    )

    report = build_report(
        target="tpu", runtime_receipt=smoke_path,
        qualification_receipt=qualification_path,
    )
    runtime = report["gates"]["target_runtime"]
    assert runtime["canary_evidence"]["state"] == "PRESENT_FOR_REVIEW"
    assert runtime["qualification_evidence"]["state"] == "BLOCKED"
    assert "runtime identity differs" in runtime["qualification_evidence"]["reason"]
    assert runtime["state"] == "TPU_EVIDENCE_REQUIRED"
    assert report["training_authorized"] is False


def test_preflight_leaves_missing_runtime_receipt_explicitly_unreviewed():
    runtime = build_report(target="tpu")["gates"]["target_runtime"]
    assert runtime["canary_evidence"]["state"] == "NOT_SUPPLIED"
    assert runtime["qualification_evidence"]["state"] == "NOT_SUPPLIED"
    assert runtime["state"] == "TPU_EVIDENCE_REQUIRED"


def test_preflight_requires_raw_rank_receipt_bundle(tmp_path):
    receipt = _write_canary_receipt(tmp_path, _canary_receipt_payload())
    (receipt.parent / "rank-07.json").unlink()

    runtime = build_report(target="tpu", runtime_receipt=receipt)["gates"]["target_runtime"]
    assert runtime["canary_evidence"]["state"] == "BLOCKED"
    assert runtime["state"] == "TPU_EVIDENCE_REQUIRED"


def test_preflight_rejects_tampered_smoke_receipts(tmp_path):
    def wrong_schema(payload):
        payload["schema"] = "other/v1"

    def failed_status(payload):
        payload["status"] = "FAIL"

    def wrong_candidate(payload):
        payload["candidate"] = "tpu_tiled_challenger"

    def wrong_source(payload):
        payload["source_tree_sha256"] = "a" * 64

    def wrong_model_spec(payload):
        payload["aggregate"]["model_spec_sha256"] = "b" * 64

    def missing_rank(payload):
        payload["aggregate"]["participating_ordinals"].pop()

    def failed_parity(payload):
        payload["aggregate"]["distributed_gradient_parity_verified"] = False

    def changed_topology(payload):
        payload["config"]["expected_world_size"] = 4

    def changed_profile(payload):
        payload["aggregate"]["measurement_profile"] = "QUALIFICATION"

    def missing_restart(payload):
        payload["controlled_restart"] = None

    def aggregate_summary_mismatch(payload):
        payload["aggregate"]["steady_global_tokens_per_second"] += 1.0

    for index, mutate in enumerate((
        wrong_schema, failed_status, wrong_candidate, wrong_source, wrong_model_spec,
        missing_rank, failed_parity, changed_topology, changed_profile, missing_restart,
        aggregate_summary_mismatch,
    )):
        payload = _canary_receipt_payload()
        mutate(payload)
        receipt = _write_canary_receipt(tmp_path, payload, f"invalid-{index}.json")
        report = build_report(target="tpu", runtime_receipt=receipt)
        assert report["gates"]["target_runtime"]["canary_evidence"]["state"] == "BLOCKED"
        assert report["gates"]["target_runtime"]["state"] == "TPU_EVIDENCE_REQUIRED"
        assert report["training_authorized"] is False


def test_preflight_recomputes_aggregate_from_each_rank_receipt(tmp_path):
    receipt = _write_canary_receipt(tmp_path, _canary_receipt_payload())
    rank_path = receipt.parent / "rank-03.json"
    row = json.loads(rank_path.read_text(encoding="utf-8"))
    row["status"] = "FAIL"
    rank_path.write_text(json.dumps(row), encoding="utf-8")

    runtime = build_report(target="tpu", runtime_receipt=receipt)["gates"]["target_runtime"]
    assert runtime["canary_evidence"]["state"] == "BLOCKED"
    assert runtime["state"] == "TPU_EVIDENCE_REQUIRED"


def test_preflight_rejects_qualification_without_update_or_memory_thresholds(tmp_path):
    for name, mutate in (
        ("too-few-updates", lambda p: p["aggregate"].update(measured_steady_update_count=19)),
        ("missing-memory", lambda p: p["aggregate"].update(peak_memory_status="UNAVAILABLE")),
        ("over-memory", lambda p: p["aggregate"].update(worst_peak_memory_fraction=0.86)),
        ("wrong-memory-summary", lambda p: p["aggregate"].update(worst_peak_memory_fraction=0.7)),
    ):
        payload = _canary_receipt_payload("QUALIFICATION")
        mutate(payload)
        receipt = _write_canary_receipt(tmp_path, payload, f"{name}.json")
        report = build_report(target="tpu", qualification_receipt=receipt)
        assert report["gates"]["target_runtime"]["qualification_evidence"]["state"] == "BLOCKED"
        assert report["gates"]["target_runtime"]["state"] == "TPU_EVIDENCE_REQUIRED"
        assert report["training_authorized"] is False


def test_preflight_library_rejects_unknown_target_names():
    import pytest

    with pytest.raises(ValueError, match="target must be one of"):
        build_report(target="invented-accelerator")


def test_preflight_rejects_boolean_token_count(tmp_path):
    manifest = {
        "schema": "anra-v5-data-manifest/v1",
        "lifecycle_state": "RUNNABLE",
        "training_manifest_sha256": "a" * 64,
        "tokenizer_sha256": "b" * 64,
        "source_ledger_sha256": "c" * 64,
        "pack_manifest_sha256": "d" * 64,
        "contamination_audit_sha256": "e" * 64,
        "evaluation_manifest_sha256": "f" * 64,
        "qualified_real_tokens": True,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert _manifest_gate(path)["state"] == "BLOCKED"


def test_preflight_rejects_arbitrary_evaluation_file_and_schema_label(tmp_path):
    empty = tmp_path / "empty.json"
    empty.write_text("{}", encoding="utf-8")
    blocked = build_report(evaluation_receipt=empty)
    assert blocked["gates"]["evaluation"]["state"] == "BLOCKED"

    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"receipt_schema": "anra-citadel-readiness/v1"}), encoding="utf-8")
    report = build_report(evaluation_receipt=receipt)
    assert report["gates"]["evaluation"]["state"] == "BLOCKED"
    assert "frozen Citadel readiness schema" in report["gates"]["evaluation"]["reason"]
    assert report["training_authorized"] is False

    development_certificate = REPO_ROOT / "artifacts/e0/scoring_adapter_certificate.json"
    development_report = build_report(evaluation_receipt=development_certificate)
    assert development_report["gates"]["evaluation"]["state"] == "BLOCKED"
    assert development_report["training_authorized"] is False


def test_preflight_requires_complete_hash_bound_citadel_readiness_contract(tmp_path):
    from tools.signac_100m_preflight import EVALUATION_READINESS_HASH_FIELDS

    receipt = tmp_path / "receipt.json"
    payload = {
        "schema": "anra-signac-citadel-readiness/v1",
        "scope": "signac-100m-citadel-readiness",
        "status": "PASS",
        "firewall_status": "PASS",
        "executable_truth_status": "PASS",
        "eos_status": "PASS",
        **{field: "a" * 64 for field in EVALUATION_READINESS_HASH_FIELDS},
    }
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    reviewed = build_report(evaluation_receipt=receipt)
    assert reviewed["gates"]["evaluation"]["state"] == "PRESENT_FOR_REVIEW"
    assert reviewed["training_authorized"] is False

    payload["eos_status"] = "PENDING"
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    blocked = build_report(evaluation_receipt=receipt)
    assert blocked["gates"]["evaluation"]["state"] == "BLOCKED"
    assert "eos_status" in blocked["gates"]["evaluation"]["checks_not_passed"]
