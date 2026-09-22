import json

from signac_100m.spec import (
    MODEL_SPEC,
    TPU_DEPTH_PRESERVING_CHALLENGER,
    TPU_TILED_CHALLENGER,
    candidate_receipts,
    parameter_receipt,
    resource_estimate,
)
from tools.signac_100m_preflight import _manifest_gate, build_report


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
    assert estimate["parameters_exact"] == 101_790_080
    assert estimate["planning_tokens_at_20x_generic_prior"] == 2_035_801_600
    assert "excludes activations" in estimate["caveat"]
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


def test_preflight_rejects_arbitrary_evaluation_file_but_only_reviews_schema(tmp_path):
    empty = tmp_path / "empty.json"
    empty.write_text("{}", encoding="utf-8")
    blocked = build_report(evaluation_receipt=empty)
    assert blocked["gates"]["evaluation"]["state"] == "BLOCKED"

    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"receipt_schema": "anra-citadel-readiness/v1"}), encoding="utf-8")
    reviewed = build_report(evaluation_receipt=receipt)
    assert reviewed["gates"]["evaluation"]["state"] == "PRESENT_FOR_REVIEW"
    assert reviewed["training_authorized"] is False
