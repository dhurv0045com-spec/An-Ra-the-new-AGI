"""Single canonical public renderer (I02): one serialization for preparation,
prediction and cognition. Private generator state cannot enter tokens
through an indirect helper because the renderer accepts only declared public
fields and rejects undeclared keys at the boundary.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from bramastra_lab.research.experience.codec import (
    SPECIAL_BOUNDARY,
    SPECIAL_EOS,
    encode_event,
)
from bramastra_lab.research.experience.trajectory import (
    PROVENANCE_ONLY_FIELDS,
    PUBLIC_VIEW_FIELDS,
    ExperienceError,
)

RENDERER_IDENTITY_FIELDS = {
    "schema": "bramastra-public-renderer/v1",
    "roles": ["goal", "observation", "action", "feedback", "workspace",
              "memory_context"],
}


class RendererError(ExperienceError):
    """A public-rendering request carried undeclared or private fields."""


def render_event_sequence(events: Sequence[tuple[str, Any]], *,
                          include_boundary: bool = True,
                          terminal_eos: bool = False) -> list[int]:
    """Render an ordered public event list into tokens.

    Each event is (role, payload) with role in the declared public roles.
    Undeclared roles and provenance-only fields reject here — the boundary,
    not the caller, owns the information boundary.
    """
    tokens: list[int] = [SPECIAL_BOUNDARY] if include_boundary else []
    for role, payload in events:
        if role not in RENDERER_IDENTITY_FIELDS["roles"]:
            raise RendererError(f"undeclared public role {role!r}")
        if isinstance(payload, Mapping):
            leaking = set(payload) & PROVENANCE_ONLY_FIELDS
            if leaking:
                raise RendererError(
                    f"provenance-only fields cannot enter public tokens: {sorted(leaking)}")
        tokens.extend(encode_event(role, payload))
    if terminal_eos:
        tokens.append(SPECIAL_EOS)
    return tokens


def renderer_identity() -> str:
    from bramastra_lab.research.contracts.core import content_identity

    return content_identity(RENDERER_IDENTITY_FIELDS)
