from __future__ import annotations

from pathlib import Path

import pytest

from v5_experiments import cyr_gpu011 as core

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "docs/cymek/experiments/CYR-GPU-011/ARK002B_TASK_MANIFEST.json"


def test_v9_and_v10_were_underexposed_relative_to_ark002b() -> None:
    assert core.CYR11_ARK_MAX_ROW_PRESENTATIONS == 18_000 * 64 == 1_152_000
    assert 18_000 * 16 / core.CYR11_ARK_MAX_ROW_PRESENTATIONS == 0.25
    assert (15_632 * 16) / core.CYR11_ARK_MAX_ROW_PRESENTATIONS < 0.22


def test_compact_vocab_matches_arkenstone_symbol_table_without_hidden_bos() -> None:
    tok = core.CompactCharTokenizer()
    assert tok.vocabulary_size == 19
    assert tok.special == {"pad_id": 0, "bos_id": 1, "eos_id": 2}
    text = "74 + 15 = "
    ids = tok.encode(text)
    assert ids and ids[0] != tok.bos_id
    assert tok.decode(ids) == text
    with pytest.raises(ValueError):
        tok.encode("What is 1 plus 2?")


def test_research_small_bridge_geometry_and_counts() -> None:
    compact = core.research_small_spec(19)
    production = core.research_small_spec(24_576)
    for spec in (compact, production):
        assert spec.layers == 4
        assert spec.width == 128
        assert spec.query_heads == 4
        assert spec.kv_heads == 2
        assert spec.head_dimension == 32
        assert spec.ffn_width == 512
        spec.assert_valid()
    assert compact.parameter_receipt().total == 987_392
    assert production.parameter_receipt().total == 4_130_688


def test_exact_ark002b_manifest_and_firewall() -> None:
    data = core.load_ark002b_manifest(MANIFEST)
    assert data["source_split_sha256"] == core.CYR11_ARK002B_SPLIT_SHA
    assert data["source_blob_sha"] == core.CYR11_ARK002B_BLOB_SHA
    assert len(data["train"]) == 500
    assert len(data["dev_controller"]) == 64
    assert len(data["dev_measurement"]) == 85
    assert len(data["sealed_reserved"]) == 48
    assert data["train_test_canonical_overlap"] == 0
    train_pairs = {tuple(r["canonical_pair"]) for r in data["train"]}
    eval_pairs = {tuple(r["canonical_pair"]) for role in ("dev_controller", "dev_measurement", "sealed_reserved") for r in data[role]}
    assert train_pairs.isdisjoint(eval_pairs)


def test_firewall_partition_is_content_deterministic() -> None:
    first = core.load_ark002b_manifest(MANIFEST)
    second = core.load_ark002b_manifest(MANIFEST)
    assert first["role_sha256"] == second["role_sha256"]
    assert [r["world_id"] for r in first["sealed_reserved"]] == [r["world_id"] for r in second["sealed_reserved"]]


def test_reasoning_battery_is_structural_and_train_pair_clean_where_required() -> None:
    data = core.load_ark002b_manifest(MANIFEST)
    battery = core.make_reasoning_battery(data)
    assert len(battery["STANDARD"]) == 85
    assert len(battery["COMMUTED"]) == 85
    assert len(battery["LOCALITY"]) >= 64
    assert len(battery["CARRY"]) == 64
    assert len(battery["TRIPLE_ADD"]) == 48
    assert len(battery["THREE_DIGIT"]) == 48
    assert len(battery["VERBAL"]) == 48
    train_pairs = {tuple(r["canonical_pair"]) for r in data["train"]}
    for row in battery["STANDARD"] + battery["COMMUTED"]:
        assert tuple(sorted((int(row["a"]), int(row["b"])))) not in train_pairs
    for row in battery["LOCALITY"]:
        assert tuple(sorted((int(row["a"]), int(row["b"])))) not in train_pairs


def _cal(regime: str, batch: int, ups: float, eps: float = 100.0) -> dict:
    return {"status": "PASS", "regime": regime, "batch_rows": batch,
            "training_updates_per_sec": ups, "training_real_tokens_per_sec": ups * batch * 8,
            "semantic_rows_per_sec": ups * batch, "generation_examples_per_sec": eps}


def test_resolver_prefers_batch64_when_exposure_is_competitive() -> None:
    calibrations = {
        "COMPACT_B64": _cal("COMPACT", 64, 30.0),
        "COMPACT_B32": _cal("COMPACT", 32, 40.0),
        "PRODUCTION_B64": _cal("PRODUCTION", 64, 1.0),
        "PRODUCTION_B32": _cal("PRODUCTION", 32, 1.8),
        "PRODUCTION_B16": _cal("PRODUCTION", 16, 2.2),
    }
    resolved = core.resolve_from_calibrations(calibrations)
    assert resolved["compact_batch_rows"] == 64
    assert resolved["production_batch_rows"] == 64
    assert 0 < resolved["production_projected_ark_exposure_fraction"] <= 1
    core.validate_resolved(resolved)


def test_resolver_can_trade_batch_for_materially_more_semantic_exposure() -> None:
    calibrations = {
        "COMPACT_B64": _cal("COMPACT", 64, 30.0),
        "PRODUCTION_B64": _cal("PRODUCTION", 64, 0.25),
        "PRODUCTION_B32": _cal("PRODUCTION", 32, 1.8),
        "PRODUCTION_B16": _cal("PRODUCTION", 16, 2.5),
    }
    resolved = core.resolve_from_calibrations(calibrations)
    assert resolved["production_batch_rows"] == 32
    assert resolved["production_projected_row_presentations"] > 0


def test_resolver_does_not_repeat_all_or_nothing_hardware_failure() -> None:
    calibrations = {
        "COMPACT_B64": _cal("COMPACT", 64, 10.0),
        "PRODUCTION_B16": _cal("PRODUCTION", 16, 0.2),
    }
    resolved = core.resolve_from_calibrations(calibrations)
    assert resolved["production_available"] is True
    assert resolved["production_projected_updates"] > 0
    assert resolved["production_promotion_authorized"] is False


def test_compact_only_hardware_is_still_a_valid_one_shot_diagnostic() -> None:
    resolved = core.resolve_from_calibrations({"COMPACT_B32": _cal("COMPACT", 32, 2.0)})
    assert resolved["production_available"] is False
    assert resolved["production_batch_rows"] is None
    core.validate_resolved(resolved)


def test_final_decision_requires_replication_for_replicated_label() -> None:
    compact = {"g90_confirm_update": 12_000}
    p1 = {"g90_confirm_update": 13_000, "ark_exposure_fraction": 0.8}
    p2 = {"g90_confirm_update": None, "ark_exposure_fraction": 0.5}
    assert core.final_decision(compact=compact, production_primary=p1, production_replication=None)["verdict"] == "PRODUCTION_REPRESENTATION_G90_SINGLE_SEED_DEVELOPMENT"
    assert core.final_decision(compact=compact, production_primary=p1, production_replication={"g90_confirm_update": 15_000})["verdict"] == "PRODUCTION_REPRESENTATION_G90_REPLICATED_DEVELOPMENT"
    assert core.final_decision(compact=compact, production_primary={"g90_confirm_update": None, "ark_exposure_fraction": 0.9}, production_replication=p2)["verdict"] == "BRIDGE_DIVERGENCE_COMPACT_G90_PRODUCTION_NO_G90"


def test_structural_flags_are_orthogonal_not_one_reasoning_score() -> None:
    result = {
        "STANDARD": {"complete_exact_with_valid_stop": 0.95},
        "COMMUTED": {"complete_exact_with_valid_stop": 0.94},
        "LOCALITY": {"structural": {"counterfactual_relation_consistency": 0.85, "numeric_usable_fraction": 0.90}},
        "CARRY": {"complete_exact_with_valid_stop": 0.10},
        "TRIPLE_ADD": {"complete_exact_with_valid_stop": 0.55},
        "THREE_DIGIT": {"complete_exact_with_valid_stop": 0.20},
        "VERBAL": {"complete_exact_with_valid_stop": 0.60},
    }
    flags = core.structural_flags(result)
    assert flags["STANDARD_G90"] and flags["COMMUTATION_INVARIANCE"] and flags["COUNTERFACTUAL_LOCALITY"]
    assert flags["OPERATION_COMPOSITION"] and flags["SURFACE_TRANSFER"]
    assert not flags["CARRY_TRANSFER"] and not flags["LENGTH_EXTRAPOLATION"]


def test_live_cymek_compact_bridge_model_builds_exactly() -> None:
    torch = pytest.importorskip("torch")
    from v5_model.core import initialize
    spec = core.research_small_spec(19)
    model = initialize(spec, 123, torch_module=torch)
    assert sum(p.numel() for p in model.parameters()) == 987_392


def test_optimizer_compat_is_semantic_noop_and_rejects_drift() -> None:
    torch = pytest.importorskip("torch")
    import v5_training.optimizer as opt
    from anra_v5.cyr_gpu011_optimizer_compat import install
    from v5_model.core import initialize

    install()
    spec = core.research_small_spec(19)
    model = initialize(spec, 7, torch_module=torch)
    optimizer = opt.build_adamw_optimizer(model, torch_module=torch, lr=1e-3,
        betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    assert optimizer.param_groups
    with pytest.raises(ValueError):
        opt.build_adamw_optimizer(model, torch_module=torch, lr=1e-3, betas=(0.9, 0.999))
