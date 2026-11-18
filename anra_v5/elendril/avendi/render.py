"""Avendi V8-Y2 native Y1 and transfer renderer, token-boundary extractor, and destructive input checks."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import re
from typing import Any, Mapping, Sequence

from anra_v5.elendril.codec import canonical, digest
from anra_v5.elendril.representation import UTF8, SpecialToken
from .protocol import PROMPT_TOKEN_BOUND, RESERVED_TARGET_TOKENS, SYMBOLS


INSTRUCTIONS_HEADER = "Output A-P, Q=query, !=abstain, R=memory, ~X=think.\n"
ACTION_MARKER = "\nAction:"
LEAK_MARKERS = ("family_id", "truth", "expected_answer", "proof_path", "generator_seed", "swapped_flag")


@dataclass(frozen=True, slots=True)
class RenderedRow:
    """Public fact row prior to surface formatting."""

    subject: str
    value: str | None
    tick: int
    source: str


def format_public_row(
    row: RenderedRow,
    *,
    template: int = 0,
    alias_map: Mapping[str, str] | None = None,
) -> str:
    """Render a single fact row in native Y1 (template 0) or transfer syntax (templates 1..3)."""
    def _ent(sym: str) -> str:
        if alias_map is not None and sym in alias_map:
            return str(alias_map[sym])
        return sym

    subj = _ent(row.subject)
    val = "null" if row.value is None else _ent(row.value)
    if template == 0:
        return f"{subj}={val}@{int(row.tick)}/{row.source}"
    if template == 1:
        return f"At {int(row.tick)} {row.source} says {subj}:{val}"
    if template == 2:
        return f"({row.source},{int(row.tick)},{subj},{val})"
    if template == 3:
        return f"[{int(row.tick)}|{subj}->{val}|{row.source}]"
    raise ValueError(f"unsupported render template: {template!r}")


def render_avendi_prompt(
    *,
    query_subject: str,
    hops: int,
    rows: Sequence[RenderedRow],
    query_available: bool = False,
    memory_available: bool = False,
    template: int = 0,
    alias_map: Mapping[str, str] | None = None,
    is_literal_copy: bool = False,
    convention_header: str | None = None,
) -> str:
    """Render a public Avendi observation using native Y1 action framing and enforce the 512-token bound."""
    if query_subject not in SYMBOLS and (alias_map is None or query_subject not in alias_map):
        raise ValueError(f"invalid query_subject: {query_subject!r}")
    if hops < 1 or hops > 6:
        raise ValueError(f"invalid hop count: {hops!r}")
    if not rows:
        raise ValueError("render_avendi_prompt requires at least one public row")

    q_disp = str(alias_map[query_subject]) if (alias_map and query_subject in alias_map) else query_subject
    if is_literal_copy:
        question = f"Follow 1 links from {q_disp}. Return the symbol in the marked register {q_disp}."
    else:
        question = f"Follow {int(hops)} links from {q_disp}."
    if alias_map is not None:
        inv_items = ", ".join(f"{alias_map[k]}->{k}" for k in sorted(alias_map))
        question += f" Alias map: {inv_items}."
    if convention_header:
        question += f" {convention_header.strip()}"

    header = f"{question} Query={int(bool(query_available))} Memory={int(bool(memory_available))}\n"
    rendered_rows = [format_public_row(r, template=template, alias_map=alias_map) for r in rows]
    prompt = INSTRUCTIONS_HEADER + header + "; ".join(rendered_rows) + ACTION_MARKER

    low = prompt.lower()
    for marker in LEAK_MARKERS:
        if marker in low:
            raise ValueError(f"private truth/metadata leak marker {marker!r} detected in rendered prompt")

    tokens = UTF8.encode(prompt)
    if len(tokens) + RESERVED_TARGET_TOKENS > PROMPT_TOKEN_BOUND:
        raise ValueError(
            f"rendered prompt token length {len(tokens)} + {RESERVED_TARGET_TOKENS} exceeds bound {PROMPT_TOKEN_BOUND}"
        )
    return prompt


def encode_prompt_and_target(prompt: str, target_action_text: str) -> dict[str, Any]:
    """Tokenize prompt and target with BOS/ANSWER/EOS and extract exact prompt-only prefix at target boundary."""
    if not prompt.endswith(ACTION_MARKER):
        raise ValueError("prompt must end with explicit '\\nAction:' marker")
    if not target_action_text or any(ord(c) < 32 for c in target_action_text):
        raise ValueError(f"malformed target_action_text: {target_action_text!r}")

    framed = UTF8.frame(prompt, target_action_text, max_length=PROMPT_TOKEN_BOUND)
    full_tokens = tuple(int(t) for t in framed.ids)
    target_boundary_index = framed.loss_mask.index(True)
    prompt_tokens = full_tokens[:target_boundary_index]
    target_tokens = full_tokens[target_boundary_index:]

    if len(full_tokens) > PROMPT_TOKEN_BOUND:
        raise ValueError(f"framed sequence length {len(full_tokens)} exceeds {PROMPT_TOKEN_BOUND}")
    if full_tokens[-1] != UTF8.eos_id:
        raise ValueError("target token sequence missing terminal EOS")

    prefix_at_boundary = full_tokens[:target_boundary_index]
    if prefix_at_boundary != prompt_tokens:
        raise ValueError("target boundary prefix mismatch")

    prompt_bytes = UTF8.decode(prompt_tokens[1:-1]).encode("utf-8")
    return {
        "prompt_tokens": prompt_tokens,
        "target_tokens": target_tokens,
        "full_tokens": full_tokens,
        "target_boundary_index": target_boundary_index,
        "prompt_token_sha256": digest(list(prompt_tokens)),
        "prompt_utf8_sha256": sha256(prompt_bytes).hexdigest(),
        "full_token_sha256": digest(list(full_tokens)),
    }


def extract_prompt_prefix_from_framed_tokens(
    full_tokens: Sequence[int],
    target_boundary_index: int,
) -> tuple[int, ...]:
    """Extract only the public prompt tokens up to target_boundary_index (excluding supervised target+EOS)."""
    if target_boundary_index <= 2 or target_boundary_index >= len(full_tokens):
        raise ValueError(f"invalid target_boundary_index {target_boundary_index} for len {len(full_tokens)}")
    prefix = tuple(int(t) for t in full_tokens[:target_boundary_index])
    ans_delim = UTF8.special(SpecialToken.ANSWER)
    body = prefix[1:-1] if (prefix[0] == UTF8.bos_id and prefix[-1] == ans_delim) else (
        prefix[1:] if prefix[0] == UTF8.bos_id else prefix
    )
    decoded = UTF8.decode(body)
    if not decoded.endswith(ACTION_MARKER):
        raise ValueError("token prefix at target_boundary_index does not end at '\\nAction:' boundary")
    return prefix


def erase_goal_token(prompt: str) -> str:
    """Destructive check helper: replace the queried start symbol in the header with <MASK_GOAL>."""
    out, count = re.subn(
        r"(Follow \d+ links from )([A-Za-z0-9_-]+)(\.)",
        r"\1<MASK_GOAL>\3",
        prompt,
        count=1,
    )
    if count != 1:
        raise ValueError(f"failed to locate goal token in prompt header: {prompt!r}")
    return out


def mask_bridge_edges_in_prompt(prompt: str, bridge_subjects: Sequence[str]) -> str:
    """Destructive check helper: mask values of the specified bridge subjects across rewires."""
    masked = prompt
    for subj in bridge_subjects:
        esc = re.escape(subj)
        # Template 0: C=E@tick/src
        masked = re.sub(rf"\b{esc}=[A-Za-z0-9_-]+@", f"{subj}=<MASK_BRIDGE>@", masked)
        # Template 1: says C:E
        masked = re.sub(rf"(says {esc}:)[A-Za-z0-9_-]+", r"\1<MASK_BRIDGE>", masked)
        # Template 2: (src,tick,C,E)
        masked = re.sub(rf"(,\d+,{esc},)[A-Za-z0-9_-]+\)", r"\1<MASK_BRIDGE>)", masked)
        # Template 3: [tick|C->E|src]
        masked = re.sub(rf"(\|{esc}->)[A-Za-z0-9_-]+\|", r"\1<MASK_BRIDGE>|", masked)
    return masked
