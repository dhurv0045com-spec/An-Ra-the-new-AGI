from anra_v5 import cyr_gpu006_run_final as runner
from v5_experiments import cyr_gpu006_final as core


def _sealed(floor=0.95, qualified=True):
    return {"qualified": qualified, "robust_floor": floor}


def _old(score=0.95):
    return {"complete_exact_with_valid_stop": score}


def _state(confirm=300000, floor=0.95, old=0.95, status="COMPLETE"):
    return {
        "status": status,
        "robust_g90_confirm_tokens": confirm,
        "sealed_binding": _sealed(floor=floor, qualified=True),
        "old_t2_measurement_final": _old(old),
    }


def _source_arm(final_g=0.95, tokens=500000, tail="tail", parent="parent"):
    return {
        "status": "COMPLETE",
        "redteam_pass": True,
        "final_g": final_g,
        "actual_real_tokens": tokens,
        "future_tail_sha256": tail,
        "parent_model_sha256": parent,
        "final_checkpoint": "/tmp/final",
    }


def _source_parent():
    return {
        "seed": 707,
        "parent_status": "G90_CONFIRMED",
        "parent_equivalence": {"identical": True},
        "future_tail": {"identical": True},
        "arms": {
            core.CYR6_TRANSFER_CANDIDATE: _source_arm(),
            core.CYR6_TRANSFER_COMPARATOR: _source_arm(),
        },
    }


def test_binding_task_is_deterministic_and_orthogonalized():
    first = runner.robust_binding_task()
    second = runner.robust_binding_task()
    assert first["manifest"] == second["manifest"]
    assert len(first["train_rows"]) == 96 * 4
    assert set(first["control"]) == {"canonical", "query_only", "order_only", "query_order"}
    assert set(first["sealed"]) == {"canonical", "query_only", "order_only", "query_order"}
    assert all(len(rows) == 32 for rows in first["control"].values())
    assert all(len(rows) == 32 for rows in first["sealed"].values())


def test_transfer_sources_must_be_equal_age_and_t2_qualified():
    parent = _source_parent()
    ok, reasons = runner._source_eligible(parent)
    assert ok, reasons

    parent["arms"][core.CYR6_TRANSFER_CANDIDATE]["actual_real_tokens"] += 1
    ok, reasons = runner._source_eligible(parent)
    assert not ok
    assert any("exposure differs" in reason for reason in reasons)


def test_transfer_sources_must_share_parent_and_future_tail():
    parent = _source_parent()
    parent["arms"][core.CYR6_TRANSFER_COMPARATOR]["future_tail_sha256"] = "other"
    ok, reasons = runner._source_eligible(parent)
    assert not ok
    assert any("continuation identity differs" in reason for reason in reasons)


def test_noninferior_pair_and_advantage_are_separate():
    pair = {
        "comparator": _state(confirm=300000, floor=0.95, old=0.90),
        "candidate": _state(confirm=300000, floor=0.95, old=0.91),
    }
    summary = runner._transfer_pair_summary(pair)
    assert summary["compatible"] is True
    assert summary["advantage"] is False

    pair["candidate"] = _state(confirm=250000, floor=0.95, old=0.96)
    summary = runner._transfer_pair_summary(pair)
    assert summary["compatible"] is True
    assert summary["advantage"] is True


def test_material_plasticity_slowdown_fails_pair():
    pair = {
        "comparator": _state(confirm=200000),
        "candidate": _state(confirm=300000),
    }
    summary = runner._transfer_pair_summary(pair)
    assert summary["compatible"] is False
    assert any("slower" in reason for reason in summary["reasons"])


def test_transfer_comparator_is_low_not_precontinuation_parent():
    assert core.CYR6_TRANSFER_CANDIDATE == "HYSTERETIC_HIGH_LOW"
    assert core.CYR6_TRANSFER_COMPARATOR == "LOW_CONTINUE"
    assert core.CYR6_TRANSFER_MODE == "REPLAY_1_IN_10_ROWS"
