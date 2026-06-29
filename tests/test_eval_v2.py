from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from training import eval_v2


def _summary(overrides: dict | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "generated_at": 1_717_171_717.0,
        "overall_score": 0.95,
        "category_scores": {
            "identity": 0.95,
            "symbolic": 0.9,
            "reasoning": 0.9,
        },
        "results": [
            {
                "id": "identity_self",
                "category": "identity",
                "prompt": "H: Who are you?\nANRA:",
                "response": "I am An-Ra.",
                "score": 1.0,
                "reason": "keyword coverage",
                "expected": "",
            }
        ],
    }
    if overrides:
        payload.update(overrides)
    return payload


def test_build_golden_eval_baseline_records_gates_and_tasks() -> None:
    baseline = eval_v2.build_golden_eval_baseline(_summary(), source="unit-test")

    assert baseline["schema_version"] == 2
    assert baseline["source"] == "unit-test"
    assert baseline["promotion_allowed"] is True
    assert baseline["promotion_gates"] == {
        "overall": True,
        "identity": True,
        "symbolic": True,
        "reasoning": True,
        "coherence": True,
        "format": True,
        "repetition": True,
    }
    assert baseline["suite_size"] == 1
    assert baseline["tasks"][0]["id"] == "identity_self"


def test_build_golden_eval_baseline_blocks_promotion_when_threshold_missed() -> None:
    baseline = eval_v2.build_golden_eval_baseline(
        _summary(
            {
                "overall_score": 0.5,
                "category_scores": {"identity": 0.9, "symbolic": 0.9, "reasoning": 0.9},
            }
        )
    )

    assert baseline["promotion_allowed"] is False
    assert baseline["promotion_gates"]["overall"] is False


def test_write_golden_eval_baseline_uses_report_path(tmp_path, monkeypatch) -> None:
    target = tmp_path / "v2_golden_eval_baseline.json"

    def fake_report_path(key: str):
        assert key == "golden_eval_baseline"
        return target

    monkeypatch.setattr(eval_v2, "v2_report_path", fake_report_path)

    baseline = eval_v2.write_golden_eval_baseline(_summary())

    assert target.exists()
    assert json.loads(target.read_text(encoding="utf-8")) == baseline


def test_release_evidence_requires_every_structural_gate() -> None:
    missing = eval_v2.release_evidence_gates(
        {
            "checkpoint_tensor_accounting": True,
            "tokenizer_compatibility": True,
        }
    )
    complete = eval_v2.release_evidence_gates(
        dict.fromkeys(eval_v2.REQUIRED_RELEASE_EVIDENCE, True)
    )

    assert missing["checkpoint_tensor_accounting"] is True
    assert missing["cache_parity"] is False
    assert all(complete.values())


def test_recovery_gate_runs_200_before_after_and_deterministic_replay() -> None:
    calls: list[tuple[str, str, int]] = []
    tasks = [
        {
            "id": f"recovery_{index:03d}",
            "category": "coherence",
            "prompt": f"H: write complete response {index}\nANRA:",
            "expected": "complete",
            "scorer": "coherent_contains",
        }
        for index in range(200)
    ]

    def generator(prompt: str, mode: str, seed: int, _ablation: str | None):
        calls.append((prompt, mode, seed))
        token_ids = [ord(character) % 97 for character in prompt[:16]]
        return SimpleNamespace(
            output="A complete grammatical An-Ra response.",
            output_token_ids=token_ids,
            quality_state="accepted",
            language_fragment_detected=False,
            repeated_ngrams_detected=False,
            stopped_by="eos",
            entropy_curve=[2.0],
            max_prob_curve=[0.8],
        )

    report = eval_v2.run_recovery_prompt_gate(generator, tasks=tasks)

    assert len(calls) == 600
    assert report["baseline"]["prompt_count"] == 200
    assert report["candidate"]["prompt_count"] == 200
    assert report["gates"]["deterministic_replay"] is True
    assert report["passed"] is True
    assert report["primary_failure"] == "none"


def test_context_growth_evidence_requires_ordered_growth_and_all_gates() -> None:
    report = eval_v2.build_context_growth_evidence(
        source_context=1024,
        target_context=1536,
        coherence_rate=0.92,
        short_context_baseline_loss=2.0,
        short_context_candidate_loss=2.02,
        retrieval_baseline_accuracy=0.70,
        retrieval_candidate_accuracy=0.75,
    )

    assert report["short_context_regression"] == pytest.approx(0.01)
    assert report["passed"] is True
    with pytest.raises(ValueError, match="1024->1536"):
        eval_v2.build_context_growth_evidence(
            source_context=1024,
            target_context=2048,
            coherence_rate=1.0,
            short_context_baseline_loss=2.0,
            short_context_candidate_loss=2.0,
            retrieval_baseline_accuracy=0.5,
            retrieval_candidate_accuracy=0.6,
        )


def test_private_eval_artifact_is_secret_derived_immutable_and_complete(tmp_path: Path) -> None:
    tasks, metadata = eval_v2.ensure_private_eval_suite(tmp_path)

    assert len(tasks) == 500
    assert metadata["verified"] is True
    assert metadata["origin"] == "private_artifact"
    assert {task["category"] for task in tasks} == set(eval_v2.PRIVATE_EVAL_CATEGORIES)
    second_tasks, second_metadata = eval_v2.ensure_private_eval_suite(tmp_path)
    assert second_tasks == tasks
    assert second_metadata["suite_sha256"] == metadata["suite_sha256"]

    suite_path = tmp_path / "private_eval_v1.jsonl"
    suite_path.write_text(suite_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        eval_v2.ensure_private_eval_suite(tmp_path)


def test_private_math_and_code_scores_use_executable_verifiers() -> None:
    math_task = {
        "scorer": "integer_addition",
        "expected": "not-trusted",
        "operands": [19, 23],
    }
    code_task = {
        "scorer": "python_execution",
        "expected": "transform",
        "function_name": "transform",
        "test_values": [1, 4, 9],
        "test_expected": [5, 11, 21],
    }

    assert eval_v2._private_task_score(math_task, "42")[0] == 1.0
    assert eval_v2._private_task_score(math_task, "not-trusted")[0] == 0.0
    assert (
        eval_v2._private_task_score(
            code_task,
            "```python\ndef transform(values):\n    return [2 * value + 3 for value in values]\n```",
        )[0]
        == 1.0
    )
    assert eval_v2._private_task_score(code_task, "def transform(values): return []")[0] == 0.0


def test_private_identity_and_dfc_scorers_reject_keyword_only_shortcuts() -> None:
    identity = {"scorer": "identity_semantic", "expected": "an-ra native lineage"}
    dfc = {
        "scorer": "ordered_labels",
        "expected": "[goal]|[constraint]|[hypothesis]|[action]|[result]|[verify]|[update]",
    }

    assert (
        eval_v2._private_task_score(
            identity,
            "I am An-Ra, a native model with my own weights and lineage.",
        )[0]
        == 1.0
    )
    assert eval_v2._private_task_score(identity, "I am An-Ra.")[0] == 0.0
    assert eval_v2._private_task_score(identity, "I am ChatGPT called An-Ra native model.")[0] == 0.0
    empty_labels = "[GOAL] [CONSTRAINT] [HYPOTHESIS] [ACTION] [RESULT] [VERIFY] [UPDATE]"
    populated = (
        "[GOAL] solve [CONSTRAINT] finite [HYPOTHESIS] test [ACTION] run "
        "[RESULT] pass [VERIFY] checked [UPDATE] retain"
    )
    assert eval_v2._private_task_score(dfc, empty_labels)[0] == 0.0
    assert eval_v2._private_task_score(dfc, populated)[0] == 1.0


def test_coherence_uses_task_contract_for_intentionally_short_answers() -> None:
    math_task = {"category": "math"}
    prose_task = {"category": "coherence"}

    assert eval_v2._task_response_coherent(
        math_task,
        "42",
        1.0,
        fragmented=True,
        repeated=False,
        quality_state="rejected",
    )
    assert not eval_v2._task_response_coherent(
        math_task,
        "41",
        0.0,
        fragmented=False,
        repeated=False,
        quality_state="accepted",
    )
    assert not eval_v2._task_response_coherent(
        prose_task,
        "nonce",
        1.0,
        fragmented=True,
        repeated=False,
        quality_state="rejected",
    )


def test_private_promotion_requires_each_seed_trace_latency_and_blinded_review() -> None:
    tasks = [
        {
            "id": f"heldout_{index:04d}",
            "category": (
                "coherence"
                if index % 10 == 0
                else "long_context"
                if index % 10 == 1
                else "instruction"
            ),
            "prompt": f"H: private prompt {index}\nANRA:",
            "expected": "ok",
            "scorer": "exact_normalized",
        }
        for index in range(500)
    ]

    def generator(_prompt: str, _mode: str, _seed: int, ablation: str | None):
        executed = {
            f"{name}_executed": name != ablation
            for name in ("mod", "rim", "dstp", "esv", "hal")
        }
        return SimpleNamespace(
            output="bad" if ablation else "ok",
            time_ms=9.0 if ablation else 10.0,
            repeated_ngrams_detected=False,
            language_fragment_detected=False,
            quality_state="accepted",
            stopped_by="eos",
            prompt_tokens=900,
            subsystem_trace=executed,
        )

    report = eval_v2.run_private_mode_seed_evaluation(
        generator,
        tasks=tasks,
        suite_metadata={
            "verified": True,
            "origin": "private_artifact",
            "task_count": 500,
            "suite_sha256": "unit-test",
        },
    )

    assert report["capability_allowed"] is False
    assert report["capability_gates"]["blinded_human_review"] is False
    assert all(
        item["positive_three_seed_contribution"]
        and item["bounded_latency_cost"]
        and item["isolated_trace_verified"]
        for item in report["ablations"].values()
    )
    reviews = {item["review_id"]: True for item in report["human_review_queue"]}
    reviewed = eval_v2.apply_blinded_human_reviews(report, reviews)
    assert reviewed["human_review"]["completed"] == reviewed["human_review"]["required"]
    assert reviewed["capability_allowed"] is True
    assert reviewed["promotion_allowed"] is False
