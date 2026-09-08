from v5_experiments import cyr_gpu006 as core


def _arm(score):
    return {"status": "COMPLETE", "actual_real_tokens": 500000,
            "target_actual_real_tokens": 500000, "redteam_pass": True,
            "retention_ret90": score}


def _parent(seed, scores):
    return {"seed": seed, "parent_status": "G90_CONFIRMED",
            "parent_equivalence": {"identical": True},
            "future_tail": {"identical": True},
            "arms": {arm: _arm(scores[arm]) for arm in core.CYR6_ARMS}}


def test_three_parent_contract_and_counterbalance():
    assert core.CYR6_PARENT_SEEDS == (707, 808, 909)
    assert len(set(core.arm_order(i) for i in range(3))) == 3


def test_one_parent_can_never_win():
    scores = {"HIGH_CONTINUE": .60, "LOW_CONTINUE": .70,
              "FIXED_TIME_HIGH_TO_LOW": .75, "HYSTERETIC_HIGH_LOW": .95}
    result = core.decide_campaign([_parent(707, scores)])
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["winner"] is None


def test_replicated_margin_required():
    strong = {"HIGH_CONTINUE": .55, "LOW_CONTINUE": .60,
              "FIXED_TIME_HIGH_TO_LOW": .65, "HYSTERETIC_HIGH_LOW": .90}
    result = core.decide_campaign([_parent(707, strong), _parent(808, strong)])
    assert result["verdict"] == "REPLICATED_WINNER"
    assert result["winner"] == "HYSTERETIC_HIGH_LOW"


def test_small_or_reversed_effect_is_inconclusive():
    first = {"HIGH_CONTINUE": .70, "LOW_CONTINUE": .71,
             "FIXED_TIME_HIGH_TO_LOW": .72, "HYSTERETIC_HIGH_LOW": .73}
    second = {"HIGH_CONTINUE": .72, "LOW_CONTINUE": .71,
              "FIXED_TIME_HIGH_TO_LOW": .73, "HYSTERETIC_HIGH_LOW": .70}
    result = core.decide_campaign([_parent(707, first), _parent(808, second)])
    assert result["verdict"] == "INCONCLUSIVE"


def test_resolver_is_hardware_only_and_keeps_all_parents():
    cal = {
        "MIDI": {"status": "PASS", "training_real_tokens_per_sec": 100000,
                 "generation_examples_per_sec": 500},
        "MICRO": {"status": "PASS", "training_real_tokens_per_sec": 150000,
                  "generation_examples_per_sec": 700},
        "RESEARCH_SMALL": {"status": "PASS", "training_real_tokens_per_sec": 200000,
                           "generation_examples_per_sec": 900},
    }
    resolved = core.resolve_from_calibrations(cal)
    assert resolved["proxy"] == "MIDI"
    assert resolved["parents"] == 3
    assert resolved["target_actual_tokens_acquisition"] >= core.CYR6_ACQ_MIN_TOKENS
    assert resolved["target_actual_tokens_continuation"] >= core.CYR6_FORK_MIN_TOKENS
    core.validate_resolved(resolved)


def test_eval_cost_can_make_hardware_unaffordable():
    cal = {name: {"status": "PASS", "training_real_tokens_per_sec": 100000,
                  "generation_examples_per_sec": 0.01}
           for name in ("MIDI", "MICRO", "RESEARCH_SMALL")}
    try:
        core.resolve_from_calibrations(cal)
    except ValueError as exc:
        assert "affords" in str(exc)
    else:
        raise AssertionError("resolver ignored generation evaluation cost")
