from __future__ import annotations

import pytest

from anra.sft_conversation import (
    render_chat_prompt,
    render_prompt_from_context,
    split_training_conversation,
)


def test_training_and_interactive_prompt_bytes_match() -> None:
    messages = [
        {"role": "system", "content": "Be exact."},
        {"role": "user", "content": "Say hello."},
        {"role": "assistant", "content": "Hello."},
        {"role": "user", "content": "What next?"},
        {"role": "assistant", "content": "Continue."},
    ]
    context, answer = split_training_conversation(messages)
    interactive = render_chat_prompt(messages[:-2], "What next?")

    assert answer == "Continue."
    assert render_prompt_from_context(context) == interactive
    assert interactive == (
        "H: SYSTEM: Be exact.\nUSER: Say hello.\nANRA: Hello.\n"
        "USER: What next?\nANRA:"
    )


def test_prompt_contract_rejects_empty_or_non_assistant_training_record() -> None:
    with pytest.raises(ValueError, match="current user message is empty"):
        render_chat_prompt([], "  ")
    with pytest.raises(ValueError, match="must end with an assistant"):
        split_training_conversation([{"role": "user", "content": "hello"}])
