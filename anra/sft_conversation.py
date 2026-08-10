"""Canonical V4 supervised-conversation rendering.

Training, evaluation, and interactive inference must use the same byte-level
prompt contract. Keeping this module independent of Torch lets every surface
import it without loading the model runtime.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

SFT_PROMPT_SCHEMA = "anra-v4-sft-prompt/v1"
_ROLE_LABELS = {"system": "SYSTEM", "user": "USER", "assistant": "ANRA"}


def _validated_message(message: Mapping[str, object], index: int) -> tuple[str, str]:
    role = str(message.get("role", "")).strip().lower()
    content = str(message.get("content", "")).strip()
    if role not in _ROLE_LABELS or not content:
        raise ValueError(f"invalid V4 SFT message at index {index}")
    return role, content


def render_context(messages: Sequence[Mapping[str, object]]) -> str:
    """Render context messages without the outer ``H:``/``ANRA:`` frame."""

    parts: list[str] = []
    for index, message in enumerate(messages):
        role, content = _validated_message(message, index)
        parts.append(f"{_ROLE_LABELS[role]}: {content}")
    if not parts:
        raise ValueError("V4 SFT context is empty")
    return "\n".join(parts)


def split_training_conversation(
    messages: Sequence[Mapping[str, object]],
) -> tuple[str, str]:
    """Return the context and final assistant answer from one SFT record."""

    if not messages:
        raise ValueError("V4 SFT conversation is empty")
    final_role, answer = _validated_message(messages[-1], len(messages) - 1)
    if final_role != "assistant":
        raise ValueError("V4 SFT record must end with an assistant message")
    return render_context(messages[:-1]), answer


def render_prompt_from_context(context: str) -> str:
    """Wrap an already-rendered context in the canonical causal prompt."""

    normalized = str(context).strip()
    if not normalized:
        raise ValueError("V4 SFT prompt context is empty")
    return f"H: {normalized}\nANRA:"


def render_chat_prompt(
    history: Sequence[Mapping[str, object]], current_user_message: str
) -> str:
    """Render interactive history exactly like a training conversation prefix."""

    current = str(current_user_message).strip()
    if not current:
        raise ValueError("current user message is empty")
    messages = [*history, {"role": "user", "content": current}]
    return render_prompt_from_context(render_context(messages))
