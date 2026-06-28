from __future__ import annotations

import json

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
