"""Adversarial source-contract tests for Canary-v2 durability ordering.

These tests protect the two failure modes most likely to silently invalidate a
multi-session operator run: a resumable checkpoint without a durable training
trace, and sealed evaluation before all repeatable gates/results are frozen.
"""
from __future__ import annotations

from pathlib import Path

from anra_v5 import v51_canary_v2_run as v2

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "anra_v5" / "v51_canary_v2_run.py"


def _source() -> str:
    return RUNNER.read_text(encoding="utf-8")


def test_training_trace_is_written_before_checkpoint_publication():
    src = _source()
    start = src.index("if checkpoint_every and state.global_update % checkpoint_every == 0:")
    publish = src.index("published = store.publish(", start)
    pre_publish = src[start:publish]
    assert 'base.write_receipt("TRAINING"' in pre_publish
    assert 'status="IN_PROGRESS"' in pre_publish
    assert "state.lineage_id == LINEAGE_ID" in pre_publish


def test_resume_trims_trace_ahead_of_last_durable_checkpoint():
    src = _source()
    assert 'old_trace = [r for r in old_trace if int(r["update"]) <= int(state.global_update)]' in src
    assert "checkpoint exists without a complete durable training trace" in src
    assert "trace_prefix=old_trace" in src


def test_preflight_checkpoint_cannot_become_scientific_update_one():
    src = _source()
    wrapper = src[src.index("def mode_preflight"):src.index("def mode_scan")]
    assert "TemporaryDirectory" in wrapper
    assert 'base.LINEAGE_ID = "v51-canary-v2-preflight"' in wrapper
    assert 'base.STATE_ROOT = temp_root / "state"' in wrapper
    assert 'preflight_payload.pop("checkpoint_sha256", None)' in wrapper
    assert 'preflight_payload["scientific_state_untouched"] = True' in wrapper
    main = src[src.index("def main"):]
    assert 'if args.mode == "preflight":\n        return mode_preflight(args)' in main


def test_development_result_is_frozen_before_sealed_marker():
    src = _source()
    final = src[src.index("def mode_finalize"):src.index("def mode_preflight")]
    dev_eval = final.index('dev = evaluate_split(backend, pack["tokenizer"], dev_rows)')
    dev_receipt = final.index('base.write_receipt("EVALUATION"', dev_eval)
    marker = final.index("SEALED_LOCK.write_text", dev_receipt)
    sealed_eval = final.index('sealed = evaluate_split(backend, pack["tokenizer"], sealed_rows)', marker)
    assert dev_eval < dev_receipt < marker < sealed_eval


def test_sealed_ambiguity_is_fail_closed_not_retried():
    src = _source()
    final = src[src.index("def mode_finalize"):src.index("def mode_preflight")]
    assert "if SEALED_LOCK.exists():" in final
    assert "sealed-consumption marker exists without FINALIZATION" in final
    assert "invalidate/regenerate a new sealed split" in final


def test_training_receipt_reports_complete_lr_identity():
    plan = v2.base.canary_wsd_receipt(token_budget=1_474_560)
    trace = [
        {"update": 1, "epoch": 0, "lr_expected": 1e-6, "lr_actual": 1e-6},
        {"update": 2, "epoch": 0, "lr_expected": 2e-6, "lr_actual": 2e-6},
    ]
    receipt = v2._training_receipt(
        trace, wsd_receipt=plan, status="IN_PROGRESS",
        metrics={"durable_through_update": 2},
    )
    assert receipt["all_lr_match"] is True
    assert receipt["status"] == "IN_PROGRESS"
    assert receipt["epoch_transitions"] == [{"update": 1, "epoch": 0}]
