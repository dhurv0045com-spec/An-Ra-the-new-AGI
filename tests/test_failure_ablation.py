"""Failure-ablation loop: well-posed experiment, not a 0.70 capability gate."""

from anra_core.ablation import (
    ARM_TO_CLASS,
    UPDATE_BY_CLASS,
    ArmResult,
    DecodePolicy,
    PlantedItem,
    arms_for,
    classify_from_arms,
    evaluate_suite,
    oracle_completer,
    planted_suite,
)


def test_planted_suite_is_eighty_balanced() -> None:
    items = planted_suite()
    assert len(items) == 80
    counts: dict[str, int] = {}
    for item in items:
        counts[item.planted_class] = counts.get(item.planted_class, 0) + 1
    assert counts == {
        "missing_knowledge": 10,
        "wrong_knowledge": 10,
        "bad_retrieval": 10,
        "bad_planning": 10,
        "weak_reasoning": 10,
        "tool_execution_failure": 10,
        "context_limit": 10,
        "model_limitation": 10,
    }


def test_classer_unique_flip_maps_arm() -> None:
    for arm, failure_class in ARM_TO_CLASS.items():
        results = {
            "baseline": ArmResult("baseline", False),
            **{name: ArmResult(name, name == arm) for name in ARM_TO_CLASS},
            "empty_k": ArmResult("empty_k", False),
        }
        diagnosis = classify_from_arms(results)
        assert diagnosis.failure_class == failure_class
        assert diagnosis.update == UPDATE_BY_CLASS[failure_class]
        assert diagnosis.write_knowledge == (diagnosis.update in {"write_memory", "correct_memory"})


def test_classer_tie_and_nothing_fail_closed() -> None:
    tied = classify_from_arms(
        {
            "baseline": ArmResult("baseline", False),
            "k_add": ArmResult("k_add", True),
            "plan_change": ArmResult("plan_change", True),
            "empty_k": ArmResult("empty_k", False),
        }
    )
    assert tied.failure_class == "model_limitation"
    assert tied.write_knowledge is False
    none = classify_from_arms(
        {
            "baseline": ArmResult("baseline", False),
            "k_add": ArmResult("k_add", False),
            "empty_k": ArmResult("empty_k", False),
        }
    )
    assert none.failure_class == "model_limitation"
    assert none.update == "queue_training_change_nothing"


def test_representation_error_stops_learning() -> None:
    diagnosis = classify_from_arms(
        {
            "baseline": ArmResult("baseline", False, representation_error=True),
            "k_add": ArmResult("k_add", True),
        }
    )
    assert diagnosis.failure_class == "representation_failure"
    assert diagnosis.update == "stop_do_not_learn"
    assert diagnosis.write_knowledge is False


def test_decode_arm_shares_baseline_prefix() -> None:
    item = next(row for row in planted_suite() if row.planted_class == "weak_reasoning")
    arms = arms_for(item)
    assert arms["baseline"][0].render() == arms["decode_change"][0].render()
    assert arms["decode_change"][1].temperature > 0.0


def test_oracle_recovers_planted_classes() -> None:
    report = evaluate_suite(oracle_completer)
    assert report.n == 80
    assert report.accuracy == 1.0
    assert report.false_knowledge_rate == 0.0
    assert report.by_class["model_limitation"] == 1.0
    assert all(score == 1.0 for score in report.by_class.values())


def test_oracle_physics_is_content_not_arm_name() -> None:
    item = PlantedItem(
        "probe",
        "missing_knowledge",
        "What is the capital of France?",
        "Paris",
        "The capital of France is Paris.",
        "The capital of France is Lyon.",
        "France is in Europe",
        "read the capital fact",
        "guess",
    )
    gold_pack, greedy = arms_for(item)["k_add"]
    empty_pack, _ = arms_for(item)["baseline"]
    assert oracle_completer(item, gold_pack, greedy).success is True
    assert oracle_completer(item, empty_pack, greedy).success is False
    assert oracle_completer(item, empty_pack, DecodePolicy(temperature=0.9)).success is False
