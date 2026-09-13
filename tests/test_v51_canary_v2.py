"""Qualification tests for V5.1 Canary-v2.

V1 already owns the heavy checkpoint/corruption/fresh-process tests.  This
suite attacks only the V2 delta: frozen scientific identity, fresh data seed,
multi-epoch mapping, schedule continuity, trace durability, formation-gate
math, and the post-R1C full-softmax boundary.
"""
from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from anra_v5 import v51_canary_data as data_mod
from anra_v5 import v51_canary_v2_run as v2


REPO = Path(__file__).resolve().parents[1]


def test_v2_preregistration_is_frozen_and_self_consistent():
    p = v2.load_prereg()
    assert p["schema"] == "anra-v51-canary-v2-preregistration/v1"
    assert p["parent_canary"]["verdict"] == "CANARY_FAIL_FORMATION"
    assert p["post_r1c_context"]["verdict"] == "SOFTMAX_COMPETITION_NOT_SUFFICIENT"
    assert p["training"]["target_updates"] == 360
    assert p["training"]["tokens_per_update"] == 4096
    assert p["training"]["token_budget"] == 360 * 4096 == 1_474_560
    assert p["training"]["checkpoint_every"] == 24
    assert p["model"]["parameter_count"] == 10_227_456


def test_v2_is_state_isolated_from_v1():
    assert v2.LINEAGE_ID == "v51-canary-v2"
    assert v2.CANARY_ROOT.name == "V5_1_CANARY_V2"
    assert v2.PREREG_PATH != v2.REPO / "experiments" / "V5_1_CANARY" / "PREREGISTRATION.json"


def test_post_r1c_boundary_forbids_masked_softmax_promotion():
    p = v2.load_prereg()
    assert p["model"]["output_path"] == "tied full softmax, canonical only"
    assert p["model"]["experimental_output_treatments"] == "FORBIDDEN"
    text = json.dumps(p).lower()
    assert "softmax_competition_not_sufficient" in text
    assert "mask_4096 is not a promoted fix" in text


def test_v2_uses_fresh_seed_and_fresh_split_identity():
    p = v2.load_prereg()
    assert p["seed"] == 2026091302
    assert p["seed"] != 20260913  # V1 seed
    small_v2 = data_mod.build_dataset(seed=p["seed"], worlds_per_family=60)
    small_v1 = data_mod.build_dataset(seed=20260913, worlds_per_family=60)
    assert small_v2["split_hashes"] != small_v1["split_hashes"]
    assert data_mod.contamination_screen(small_v2)["clean"] is True


def test_v2_shortcut_screen_fails_closed():
    assert v2._shortcut_violations({"x": {"a": 0.349999}}, 0.35) == []
    bad = v2._shortcut_violations({"x": {"a": 0.35, "b": 0.7}}, 0.35)
    assert [b["heuristic"] for b in bad] == ["a", "b"]


def test_multi_epoch_layout_is_deterministic_and_epoch_sensitive(monkeypatch):
    p = v2.load_prereg()
    calls: list[int] = []

    def fake_stream(pack, *, seed, epoch, tokens_per_update):
        calls.append(epoch)
        assert seed == p["seed"]
        assert tokens_per_update == 4096
        # equal cardinality is mandatory; contents are epoch-specific
        return [(epoch, i) for i in range(127)]

    monkeypatch.setattr(v2, "_epoch_stream", fake_stream)
    layout = v2._stream_layout({}, p, 360)
    assert layout == {
        "windows_per_epoch": 127,
        "epochs_needed": 3,
        "epoch_window_counts": [127, 127, 127],
    }
    assert calls == [0, 0, 1, 2]
    # fixed 360-update endpoint reaches the third epoch rather than trying to
    # index beyond one frozen stream as V1 did.
    assert divmod(359, layout["windows_per_epoch"]) == (2, 105)


def test_multi_epoch_layout_rejects_cardinality_drift(monkeypatch):
    p = v2.load_prereg()

    def bad_stream(pack, *, seed, epoch, tokens_per_update):
        return list(range(127 if epoch == 0 else 126))

    monkeypatch.setattr(v2, "_epoch_stream", bad_stream)
    with pytest.raises(SystemExit, match="epoch 1"):
        v2._stream_layout({}, p, 360)


def test_trace_merge_is_contiguous_and_resume_safe():
    old = [{"update": 1, "loss": 2.0}, {"update": 2, "loss": 1.5}]
    new = [{"update": 3, "loss": 1.2}, {"update": 4, "loss": 1.0}]
    merged = v2._merge_trace(old, new)
    assert [r["update"] for r in merged] == [1, 2, 3, 4]
    # exact duplicate is idempotent
    assert v2._merge_trace(merged, [merged[-1]]) == merged
    with pytest.raises(SystemExit, match="conflicting"):
        v2._merge_trace(old, [{"update": 2, "loss": 99.0}])
    with pytest.raises(SystemExit, match="gap"):
        v2._merge_trace(old, [{"update": 4, "loss": 1.0}])


def test_weighted_overall_metric_is_mathematically_correct():
    scores = {
        "a": {"n": 3, "exact_with_valid_eos": 1.0},
        "b": {"n": 1, "exact_with_valid_eos": 0.0},
    }
    assert v2._weighted_exact(scores) == pytest.approx(0.75)


def test_v1_formation_thresholds_are_retained_without_moving_goalposts():
    p = v2.load_prereg()
    t = p["evaluation"]["v1_thresholds_retained_without_change"]
    assert t == {
        "identity_dev_min": 0.30,
        "binding_dev_min": 0.25,
        "dev_overall_min": 0.15,
        "formation_min_any_family": 0.30,
    }
    dev = {
        "identity": {"n": 10, "exact_with_valid_eos": 0.30},
        "binding": {"n": 10, "exact_with_valid_eos": 0.25},
        "termination": {"n": 10, "exact_with_valid_eos": 0.30},
        "composition": {"n": 10, "exact_with_valid_eos": 0.10},
        "state_order": {"n": 10, "exact_with_valid_eos": 0.00},
        "missing_info": {"n": 10, "exact_with_valid_eos": 0.10},
    }
    gates = v2.formation_gates(p, dev)
    assert gates["identity_acquisition_dev"] is True
    assert gates["binding_acquisition_dev"] is True
    assert gates["formation_positive"] is True
    assert gates["dev_transfer"] is True  # 1.05 / 6 = 0.175


def test_360_update_wsd_has_all_phases_and_no_resume_rewarm():
    p = v2.load_prereg()
    plan = v2.base.canary_wsd_receipt(token_budget=p["training"]["token_budget"])
    trace = v2.base.wsd_trace(
        plan,
        updates=p["training"]["target_updates"],
        tokens_per_update=p["training"]["tokens_per_update"],
        resume_at_update=127,
    )
    assert trace["all_match"] is True
    assert trace["rewarm_events"] == 0
    assert {r["phase"] for r in trace["rows"]} == {"warmup", "stable", "decay"}
    assert len(trace["rows"]) == 360


def test_rung_b_is_not_authorized_by_v2_preregistration():
    p = v2.load_prereg()
    with pytest.raises(SystemExit, match="Rung A only"):
        v2._prereg_rung(p, "B")


def test_substantive_run_cannot_move_the_frozen_endpoint(monkeypatch):
    args = Namespace(rung="A", updates=359, bfloat16=False, cuda=False,
                     allow_cpu=True, checkpoint_every=24)
    # endpoint check occurs before expensive pack/model construction
    assert v2.mode_run(args) == 1


def test_substantive_run_rejects_bfloat16_before_expensive_work():
    args = Namespace(rung="A", updates=360, bfloat16=True, cuda=False,
                     allow_cpu=True, checkpoint_every=24)
    assert v2.mode_run(args) == 1


def test_sealed_test_is_fresh_and_one_shot_by_contract():
    p = v2.load_prereg()
    assert "fresh V2 sealed split" in p["evaluation"]["sealed"]
    source = (REPO / "anra_v5" / "v51_canary_v2_run.py").read_text(encoding="utf-8")
    assert "SEALED_CONSUMPTION.json" in source
    assert "CONSUMED_AND_FINALIZED" in source
    assert "sealed-consumption marker exists without FINALIZATION" in source


def test_v2_runner_keeps_production_spine_not_parallel_training_code():
    source = (REPO / "anra_v5" / "v51_canary_v2_run.py").read_text(encoding="utf-8")
    assert "backend.step(state, batch)" in source
    assert "base.batch_from_window" in source
    assert "base.certify_update" in source
    assert "base.CheckpointStore" in source
    assert "base.production_payloads" in source
    assert "MASK_4096" not in source
