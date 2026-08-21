from __future__ import annotations

from scripts.prepare_high_quality_sft_v4 import _classify, _normalize_messages


def test_quality_filter_preserves_multiline_code_and_rejects_collapse() -> None:
    messages, disposition = _normalize_messages(
        [
            {"role": "user", "content": "Write a small Python function."},
            {
                "role": "assistant",
                "content": "```python\ndef add(a, b):\n    return a + b\n```\nThis returns the sum.",
            },
        ]
    )
    assert disposition == "accepted"
    assert messages is not None
    assert "    return a + b" in messages[-1]["content"]

    rejected, reason = _normalize_messages(
        [
            {"role": "user", "content": "Explain gravity."},
            {"role": "assistant", "content": "repeat repeat repeat repeat " * 30},
        ]
    )
    assert rejected is None
    assert reason in {"answer_repetition", "low_lexical_diversity"}


def test_category_rules_prioritize_structured_contracts_and_correction() -> None:
    assert _classify("Return only valid JSON matching this schema.", "smol-contraints")[0] == (
        "tool_contracts"
    )
    assert _classify("Rewrite this paragraph for clarity.", "smollm-rewrite-30k")[0] == (
        "correction"
    )
