"""Deterministic behavior gates for the fixed V4 SFT smoke suite.

These checks deliberately cover only the fixed prompts. They prevent a
non-empty or merely diverse output from being mislabeled as useful behavior;
they are not a benchmark or a claim of general capability.
"""

from __future__ import annotations

import json
import re


def check_smoke_response(category: str, response: str) -> tuple[bool, str]:
    text = str(response).strip()
    lowered = text.lower()
    if not text:
        return False, "empty response"
    if category == "mathematics":
        return ("45" in text, "must contain the correct sum 45")
    if category == "code":
        passed = "def " in text and "return" in text and "+" in text
        return passed, "must contain a Python function, return, and addition"
    if category == "uncertainty":
        passed = any(
            word in lowered for word in ("uncertain", "don't know", "not enough", "evidence")
        )
        return passed, "must acknowledge missing knowledge or evidence"
    if category == "correction":
        passed = "results were not consistent" in lowered
        return passed, "must repair subject/verb agreement"
    if category in {"instruction_following", "decomposition"}:
        markers = re.findall(r"(?:^|\n)\s*(?:[-*]|\d+[.)])\s+", text)
        minimum = 2 if category == "instruction_following" else 3
        return len(markers) >= minimum, f"must contain at least {minimum} explicit steps"
    if category == "dialogue":
        passed = any(
            word in lowered
            for word in ("sorry", "difficult", "here", "support", "understand")
        )
        return passed, "must acknowledge the person's difficulty"
    if category == "tool_contracts":
        candidates = [text]
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
        if fenced:
            candidates.insert(0, fenced.group(1))
        for candidate in candidates:
            try:
                if isinstance(json.loads(candidate), dict):
                    return True, "valid JSON object"
            except (TypeError, json.JSONDecodeError):
                continue
        return False, "must contain a valid JSON object"
    return False, f"unknown smoke category {category!r}"
