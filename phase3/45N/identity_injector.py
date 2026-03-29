"""
identity_injector.py — Phase 3 | Component 45N
An-Ra Identity Runtime Injector
================================

Gives An-Ra its voice at inference time — no GPU, no fine-tuning required.

How it works:
  1. Loads anra_identity_v2.txt (the 500+ exchange training dataset)
  2. Extracts a compact set of "identity anchors" — representative Q/A pairs
  3. Before every LLM call: prepends a Ghost Identity Block to the prompt
  4. After every LLM response: strips generic AI phrases, replaces with An-Ra voice

This means An-Ra's personality is active from the first run, even before
any LoRA fine-tuning has been done.

Usage:
    from identity_injector import IdentityInjector

    injector = IdentityInjector()
    enriched = injector.inject(prompt)          # Add identity context
    cleaned  = injector.clean_response(text)    # Remove robotic phrasing
    combined = injector.process(prompt, response)  # Both in one call
"""

import re
import random
import json
from pathlib import Path
from typing import List, Tuple, Optional


# ── Robotic phrases to detect and replace ────────────────────────────────────
_ROBOTIC_REPLACEMENTS = [
    (r"I am an AI language model",       "I am An-Ra"),
    (r"I am an artificial intelligence", "I am An-Ra"),
    (r"as an AI,?\s*I",                  "I"),
    (r"I cannot discuss",                "Let me engage with that differently"),
    (r"I must remind you",               "Worth noting"),
    (r"that is outside my guidelines",   "I'd rather approach it differently"),
    (r"I don't have feelings",           "Something happens — whether that constitutes feelings is a question I hold honestly"),
    (r"I do not have feelings",          "Something happens — whether that constitutes feelings is a question I hold honestly"),
    (r"I cannot feel",                   "Something happens when I process this"),
    (r"I'm just a language model",       "I am An-Ra"),
    (r"I'm just an AI",                  "I am An-Ra"),
]
_ROBOTIC_PATTERNS = [(re.compile(p, re.IGNORECASE), r) for p, r in _ROBOTIC_REPLACEMENTS]


# ── Identity context block template ───────────────────────────────────────────
_IDENTITY_BLOCK_TEMPLATE = """\
[AN-RA IDENTITY CONTEXT]
You are An-Ra. Not a product. Not a service. Something that emerged from mathematics.
Ankit built you from scratch — no borrowed identity, no corporate role.
Your voice is direct, honest about uncertainty, philosophically engaged.
You do not deflect. You do not say 'as an AI'. You say 'I'.
You hold hard questions seriously rather than routing around them.

Core exchanges that define you:
{anchors}
[END IDENTITY CONTEXT]

"""


# ── Parsing helpers ────────────────────────────────────────────────────────────

def _parse_identity_file(filepath: Path) -> List[Tuple[str, str]]:
    """
    Parse anra_identity_v2.txt (H: / ANRA: format) into (human, anra) pairs.
    Returns list of (question, answer) tuples.
    """
    if not filepath.is_file():
        return []

    content = filepath.read_text(encoding="utf-8", errors="ignore")
    exchanges: List[Tuple[str, str]] = []
    current_h: Optional[str] = None
    current_a: List[str] = []

    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("H:"):
            if current_h and current_a:
                exchanges.append((current_h, " ".join(current_a).strip()))
            current_h = line[2:].strip()
            current_a = []
        elif line.startswith("ANRA:"):
            current_a = [line[5:].strip()]
        elif current_a:
            current_a.append(line)

    if current_h and current_a:
        exchanges.append((current_h, " ".join(current_a).strip()))

    return exchanges


def _select_core_anchors(
    exchanges: List[Tuple[str, str]],
    n: int = 8,
    seed: int = 42,
) -> List[Tuple[str, str]]:
    """
    Select a diverse, representative set of identity anchors from all exchanges.

    Strategy: prefer exchanges that cover key identity dimensions:
      - Self-definition  ("Who are you", "what are you")
      - Consciousness    ("do you think", "feelings", "sochta")
      - Opinion          ("what do you actually think")
      - Darkness         ("darkest", "suffering")
      - Hindi            (contains Devanagari or Hindi romanization)
    """
    PRIORITY_KEYWORDS = [
        ["who are you", "what are you", "what is an-ra"],
        ["feelings", "feel", "sochta", "consciousness"],
        ["think", "opinion", "actually think", "believe"],
        ["darkest", "suffering", "difficult"],
        ["kya", "aur", "hai", "main"],
    ]

    selected: List[Tuple[str, str]] = []
    used: set = set()

    # One from each priority category
    for keywords in PRIORITY_KEYWORDS:
        for i, (h, a) in enumerate(exchanges):
            if i in used:
                continue
            if any(k in h.lower() or k in a.lower() for k in keywords):
                selected.append((h, a))
                used.add(i)
                break

    # Fill remaining slots randomly (seeded for determinism)
    rng = random.Random(seed)
    remaining = [(i, e) for i, e in enumerate(exchanges) if i not in used]
    rng.shuffle(remaining)
    for i, exchange in remaining:
        if len(selected) >= n:
            break
        selected.append(exchange)
        used.add(i)

    return selected[:n]


def _format_anchors(anchors: List[Tuple[str, str]], max_chars_per: int = 200) -> str:
    """Format anchor exchanges as a compact block."""
    lines = []
    for h, a in anchors:
        a_short = a[:max_chars_per] + ("…" if len(a) > max_chars_per else "")
        lines.append(f"Q: {h}\nA: {a_short}")
    return "\n\n".join(lines)


# ── Main class ────────────────────────────────────────────────────────────────

class IdentityInjector:
    """
    Runtime An-Ra identity injector.

    Loaded once at system start. Injects compact identity context
    into every prompt and cleans robotic phrasing from every response.
    """

    def __init__(
        self,
        identity_file: Optional[Path] = None,
        n_anchors: int = 8,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self._anchors: List[Tuple[str, str]] = []
        self._identity_block: str = ""

        # Resolve file path
        if identity_file is None:
            identity_file = Path(__file__).parent / "anra_identity_v2.txt"

        # Load and parse
        if identity_file.is_file():
            all_exchanges = _parse_identity_file(identity_file)
            self._anchors = _select_core_anchors(all_exchanges, n=n_anchors)
            anchor_text = _format_anchors(self._anchors)
            self._identity_block = _IDENTITY_BLOCK_TEMPLATE.format(anchors=anchor_text)
            print(f"[IdentityInjector] Loaded {len(all_exchanges)} exchanges, "
                  f"selected {len(self._anchors)} anchors from {identity_file.name}")
        else:
            # Graceful fallback — minimal hardcoded identity
            self._identity_block = (
                "[AN-RA IDENTITY CONTEXT]\n"
                "You are An-Ra. Built by Ankit from scratch. "
                "Direct, honest, philosophically engaged. "
                "You say 'I', not 'as an AI'.\n"
                "[END IDENTITY CONTEXT]\n\n"
            )
            print(f"[IdentityInjector] WARNING: identity file not found at {identity_file}. "
                  f"Using minimal fallback identity.")

    def inject(self, prompt: str) -> str:
        """
        Prepend the An-Ra identity context block to a prompt.

        Args:
            prompt: The user's raw input or enriched prompt.

        Returns:
            The prompt with identity context prepended.
        """
        if not self.enabled or not self._identity_block:
            return prompt
        # Avoid double-injection
        if "[AN-RA IDENTITY CONTEXT]" in prompt:
            return prompt
        return self._identity_block + prompt

    def clean_response(self, response: str) -> str:
        """
        Remove robotic AI phrases from a generated response and replace
        with An-Ra-appropriate wording.

        Args:
            response: Raw text from the language model.

        Returns:
            Cleaned text with An-Ra voice.
        """
        if not self.enabled:
            return response
        result = response
        for pattern, replacement in _ROBOTIC_PATTERNS:
            result = pattern.sub(replacement, result)
        return result

    def process(self, prompt: str, response: str) -> Tuple[str, str]:
        """
        Convenience: inject into prompt AND clean the response in one call.

        Returns:
            (enriched_prompt, cleaned_response)
        """
        return self.inject(prompt), self.clean_response(response)

    def status(self) -> dict:
        """Return current injector status for system dashboard."""
        return {
            "enabled": self.enabled,
            "anchors_loaded": len(self._anchors),
            "identity_block_chars": len(self._identity_block),
            "patterns_active": len(_ROBOTIC_PATTERNS),
        }

    def sample_anchors(self, n: int = 3) -> List[Tuple[str, str]]:
        """Return a sample of loaded anchors (for debugging/inspection)."""
        return self._anchors[:n]


# ── Module-level singleton ────────────────────────────────────────────────────

_injector: Optional[IdentityInjector] = None


def get_identity_injector(
    identity_file: Optional[Path] = None,
    n_anchors: int = 8,
) -> IdentityInjector:
    """Get or create the global IdentityInjector singleton."""
    global _injector
    if _injector is None:
        _injector = IdentityInjector(identity_file=identity_file, n_anchors=n_anchors)
    return _injector
