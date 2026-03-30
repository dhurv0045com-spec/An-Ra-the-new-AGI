"""
45P Ghost State Memory — Ghost Context prompt injection.

Formats retrieved snippets as an invisible-to-user-style prefix block that
chat front-ends can prepend before the live user message so the model sees
durable context without re-embedding the full transcript each turn.
"""

from __future__ import annotations

from typing import Iterable, List, Mapping

# Delimiters: kept ASCII so log-friendly; integrators may strip for display.
_GHOST_OPEN = "<!-- ghost_context -->"
_GHOST_CLOSE = "<!-- /ghost_context -->"


def format_ghost_block(
    snippets: Iterable[Mapping[str, object]],
    header: str = "Ghost Context (retrieved memory)",
) -> str:
    """
    Render ranked retrieval results into a single Ghost Context string.

    Each snippet mapping may include: text (required), score, role, memory_id.
    The block is wrapped in HTML-style comments so UIs that hide HTML comments
    can treat it as non-visible metadata while the model still receives it.
    """
    lines: List[str] = [_GHOST_OPEN, f"[{header}]"]
    for i, sn in enumerate(snippets, start=1):
        text = str(sn.get("text", "")).strip()
        if not text:
            continue
        score = sn.get("score", "")
        mid = sn.get("memory_id", "")
        role = sn.get("role", "")
        meta = []
        if mid != "":
            meta.append(f"id={mid}")
        if score != "":
            meta.append(f"score={score:.4f}" if isinstance(score, (int, float)) else f"score={score}")
        if role != "":
            meta.append(f"role={role}")
        prefix = f"{i}. "
        if meta:
            prefix += "(" + ", ".join(meta) + ") "
        lines.append(prefix + text)
    lines.append(_GHOST_CLOSE)
    return "\n".join(lines)


def build_prompt(
    ghost_block: str,
    user_message: str,
    *,
    separator: str = "\n\n",
) -> str:
    """
    Prepend Ghost Context before the next user message for the model call.

    Why: downstream chat code typically concatenates system + memory + user;
    this helper keeps ordering consistent (ghost first, then fresh user text).
    """
    gb = ghost_block.strip()
    um = user_message
    if not gb:
        return um
    return gb + separator + um


def strip_ghost_for_display(full_text: str) -> str:
    """
    Remove the ghost block from a string for UI display (optional helper).

    If no ghost markers are present, returns the original string unchanged.
    """
    if _GHOST_OPEN not in full_text:
        return full_text
    start = full_text.find(_GHOST_OPEN)
    end = full_text.find(_GHOST_CLOSE)
    if end == -1:
        return full_text
    end += len(_GHOST_CLOSE)
    rest = full_text[:start] + full_text[end:]
    return rest.strip()
