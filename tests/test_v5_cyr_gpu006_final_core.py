from v5_experiments import cyr_gpu006_final as core


def _cal(train_tps=100000.0, eval_eps=1000.0):
    return {
        name: {
            "status": "PASS",
            "training_real_tokens_per_sec": train_tps,
            "generation_examples_per_sec": eval_eps,
        }
        for name in ("MIDI", "MICRO", "RESEARCH_SMALL")
    }


def _arm(score):
    return {
        "status": "COMPLETE",
        "actual_real_tokens": 500_000,
        "target_actual_real_tokens": 500_000,
        "redteam_pass": True,
        "retention_ret90": score,
    }


def _parent(seed, scores):
    return {
        "seed": seed,
        "parent_status": "G90_CONFIRMED",
        "parent_equivalence": {"identical": True},
        "future_tail": {"identical": True},
        "arms": {arm: _arm(scores[arm]) for arm in core.CYR6_ARMS},
    }


def test_resolver_keeps_full_replicated_design_and_prospective_transfer():
    resolved = core.resolve_from_calibrations(_cal())
    assert resolved["parents"] == 3
    assert resolved["transfer_candidate"] == "HYSTERETIC_HIGH_LOW"
    assert resolved["transfer_comparator"] == "LOW_CONTINUE"
    assert resolved["transfer_mode"] == "REPLAY_1_IN_10_ROWS"
    assert resolved["transfer_target_parents"] == 3
    assert resolved["transfer_min_parents"] == 2
    assert resolved["target_actual_tokens_acquisition"] >= core.CYR6_ACQ_MIN_TOKENS
    assert resolved["target_actual_tokens_continuation"] >= core.CYR6_FORK_MIN_TOKENS
    assert resolved["transfer_target_actual_tokens"] >= core.CYR6_TRANSFER_MIN_TOKENS
    assert set(resolved["stage_budgets_seconds"]) == {"acquisition", "retention", "transfer"}
    core.validate_resolved(resolved)


def test_eval_cost_can_make_design_unaffordable():
    try:
        core.resolve_from_calibrations(_cal(eval_eps=0.001))
    except ValueError as exc:
        assert "affords" in str(exc)
    else:
        raise AssertionError("resolver ignored candidate-free evaluation cost")


def test_transfer_cost_is_in_stage_model():
    with_transfer = core.estimate_stage_seconds(
        training_tokens_per_sec=100000,
        eval_examples_per_sec=1000,
        acquisition_tokens=2_000_000,
        continuation_tokens=500_000,
        transfer_tokens=500_000,
        transfer_parents=3,
    )
    no_transfer_parents = core.estimate_stage_seconds(
        training_tokens_per_sec=100000,
        eval_examples_per_sec=1000,
        acquisition_tokens=2_000_000,
        continuation_tokens=500_000,
        transfer_tokens=500_000,
        transfer_parents=0,
    )
    assert with_transfer["transfer"] > no_transfer_parents["transfer"]


def test_research_candidate_requires_retention_hyst_winner_and_transfer():
    scores = {
        "HIGH_CONTINUE": 0.50,
        "LOW_CONTINUE": 0.65,
        "FIXED_TIME_HIGH_TO_LOW": 0.70,
        "HYSTERETIC_HIGH_LOW": 0.95,
    }
    parents = [_parent(707, scores), _parent(808, scores)]
    positive_transfer = {"status": "REPLICATED_PLASTICITY_NONINFERIOR"}
    result = core.final_decision(parents, transfer=positive_transfer)
    assert result["research_candidate"] is True
    assert result["production_promotion_authorized"] is False

    result = core.final_decision(parents, transfer={"status": "INCONCLUSIVE"})
    assert result["research_candidate"] is False
    assert result["production_promotion_authorized"] is False


def test_one_parent_can_never_become_research_candidate():
    scores = {
        "HIGH_CONTINUE": 0.50,
        "LOW_CONTINUE": 0.65,
        "FIXED_TIME_HIGH_TO_LOW": 0.70,
        "HYSTERETIC_HIGH_LOW": 0.95,
    }
    result = core.final_decision(
        [_parent(707, scores)],
        transfer={"status": "REPLICATED_PLASTICITY_ADVANTAGE"},
    )
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["research_candidate"] is False
    assert result["production_promotion_authorized"] is False
