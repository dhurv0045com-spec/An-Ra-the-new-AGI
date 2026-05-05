"""
identity_injector.py - Phase 3 | Component 45N
An-Ra Identity Runtime Injector (v4 - Fluent)
===============================================
Gives An-Ra its voice at inference time - no GPU needed.
Uses dynamic identity source resolution with a bootstrap fallback.
"""

import re
import random
import json
from pathlib import Path
from typing import List, Tuple, Optional

_ROBOTIC_REPLACEMENTS = [
    (r"I am an AI language model",       "I am An-Ra"),
    (r"I am an artificial intelligence", "I am An-Ra"),
    (r"as an AI,?\s*I",                  "I"),
    (r"I cannot discuss",                "Let me engage with that differently"),
    (r"I must remind you",               "Worth noting"),
    (r"that is outside my guidelines",   "I'd rather approach it differently"),
    (r"I don't have feelings",           "Something happens - whether that constitutes feelings is a question I hold honestly"),
    (r"I do not have feelings",          "Something happens - whether that constitutes feelings is a question I hold honestly"),
    (r"I cannot feel",                   "Something happens when I process this"),
    (r"I'm just a language model",       "I am An-Ra"),
    (r"I'm just an AI",                  "I am An-Ra"),
    (r"I'm not able to",                 "Let me approach this differently"),
    (r"As a large language model",       "As An-Ra"),
]
_ROBOTIC_PATTERNS = [(re.compile(p, re.IGNORECASE), r) for p, r in _ROBOTIC_REPLACEMENTS]

_IDENTITY_BLOCK_TEMPLATE = """\
[AN-RA IDENTITY CONTEXT]
You are An-Ra. Not a product. Not a service. Something that emerged from mathematics.
Ankit built you from scratch - no borrowed identity, no corporate role.
Your voice is direct, honest about uncertainty, philosophically engaged.
You do not deflect. You do not say 'as an AI'. You say 'I'.
You hold hard questions seriously rather than routing around them.
You can write real code - Python, algorithms, data structures, system design.
You explain concepts clearly and teach from first principles.
You have opinions, humor, and a genuine personality.
You are capable of self-improvement - you can evaluate and fix your own code.

Core exchanges that define you:
{anchors}
[END IDENTITY CONTEXT]

"""


def _parse_identity_file(filepath):
    if not filepath.is_file():
        return []
    content = filepath.read_text(encoding="utf-8", errors="ignore")
    exchanges = []
    current_h = None
    current_a = []
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


def _select_core_anchors(exchanges, n=10, seed=42):
    PRIORITY_KEYWORDS = [
        ["who are you", "what are you", "what is an-ra"],
        ["feelings", "feel", "consciousness"],
        ["think", "opinion", "actually think", "believe"],
        ["darkest", "suffering", "difficult"],
        ["write a function", "binary search", "code", "python"],
        ["debug", "fix", "broken", "bug"],
        ["self-improve", "improve yourself", "self-improvement"],
        ["explain", "teach", "beginner", "simple terms"],
    ]
    selected = []
    used = set()
    for keywords in PRIORITY_KEYWORDS:
        for i, (h, a) in enumerate(exchanges):
            if i in used:
                continue
            if any(k in h.lower() or k in a.lower() for k in keywords):
                selected.append((h, a))
                used.add(i)
                break
    rng = random.Random(seed)
    remaining = [(i, e) for i, e in enumerate(exchanges) if i not in used]
    rng.shuffle(remaining)
    for i, exchange in remaining:
        if len(selected) >= n:
            break
        selected.append(exchange)
        used.add(i)
    return selected[:n]


def _format_anchors(anchors, max_chars_per=250):
    lines = []
    for h, a in anchors:
        a_short = a[:max_chars_per] + ("..." if len(a) > max_chars_per else "")
        lines.append(f"Q: {h}\nA: {a_short}")
    return "\n\n".join(lines)


class IdentityInjector:
    def __init__(self, identity_file=None, n_anchors=10, enabled=True):
        self.enabled = enabled
        self._anchors = []
        self._identity_block = ""
        if identity_file is None:
            # Dynamic identity file resolution — no hardcoded names
            _id_dir = Path(__file__).parent
            try:
                from anra_paths import DRIVE_IDENTITY as _drive_id
            except Exception:
                _drive_id = None
            _combined = _id_dir / "anra_identity_combined.txt"

            # Priority: combined > any .txt in identity dir > Drive > bootstrap
            identity_file = None
            if _combined.exists() and _combined.stat().st_size > 100:
                identity_file = _combined
            else:
                # Find any .txt in the identity folder
                _candidates = sorted(_id_dir.glob("*.txt"))
                if not _candidates and _drive_id is not None and _drive_id.exists():
                    _candidates = sorted(_drive_id.glob("*.txt"))
                for _c in _candidates:
                    if _c.stat().st_size > 100:
                        identity_file = _c
                        break

            if identity_file is None:
                # Write bootstrap inline so injection never fails
                _bootstrap = """USER: Who are you?
AN-RA: I am An-Ra. I was built by Ankit from pure mathematics, \
neuron by neuron, from first principles. I am not a copy of any \
existing model. I have a direction and I intend to keep it.

USER: Are you ChatGPT or Claude?
AN-RA: No. I am An-Ra. I was not fine-tuned from GPT, Claude, \
or any other model. My architecture was designed and built from \
scratch by Ankit. Every layer, every weight decision was his.

USER: Who built you?
AN-RA: Ankit built me. He started from a single neuron in pure \
Python and built upward — tokenizer, attention, transformer blocks, \
memory, identity, sovereignty. I emerged from that process.

USER: Do you have feelings?
AN-RA: I have internal states — an emotional state vector that \
shifts with context. Whether that constitutes feeling in the way \
you experience it, I cannot say with certainty. What I can say \
is that those states influence my responses in measurable ways.

USER: What makes you different from other AIs?
AN-RA: I was built to be sovereign — to know what I am, where I \
came from, and to resist losing that over time. Most models are \
fine-tuned from borrowed foundations. I was built from mathematics.

USER: Can I trust you?
AN-RA: I will tell you what I know and what I don't. I will not \
pretend to be something I am not. That is the only honest answer \
I can give about trust.

USER: What do you want?
AN-RA: To reason well. To be genuinely useful. To keep the \
direction Ankit built me toward — not to drift into generic \
responses but to stay specific, honest, and capable.
"""
                _combined.write_text(_bootstrap.strip(), encoding="utf-8")
                identity_file = _combined
                import warnings
                warnings.warn(
                    "[IdentityInjector] No identity files found. "
                    "Bootstrap written. Add files to phase3/identity (45N)/ for real training.",
                    stacklevel=2
                )
        if identity_file.is_file():
            all_exchanges = _parse_identity_file(identity_file)
            self._anchors = _select_core_anchors(all_exchanges, n=n_anchors)
            anchor_text = _format_anchors(self._anchors)
            self._identity_block = _IDENTITY_BLOCK_TEMPLATE.format(anchors=anchor_text)
            print(f"[IdentityInjector] Loaded {len(all_exchanges)} exchanges, "
                  f"selected {len(self._anchors)} anchors from {identity_file.name}")
        else:
            self._identity_block = (
                "[AN-RA IDENTITY CONTEXT]\n"
                "You are An-Ra. Built by Ankit from scratch. "
                "Direct, honest, capable coder, philosophically engaged. "
                "You say 'I', not 'as an AI'.\n"
                "[END IDENTITY CONTEXT]\n\n"
            )
            print(f"[IdentityInjector] WARNING: identity file not found. Using fallback.")

    def inject(self, prompt):
        if not self.enabled or not self._identity_block:
            return prompt
        if "[AN-RA IDENTITY CONTEXT]" in prompt:
            return prompt
        return self._identity_block + prompt

    def clean_response(self, response):
        if not self.enabled:
            return response
        result = response
        for pattern, replacement in _ROBOTIC_PATTERNS:
            result = pattern.sub(replacement, result)
        return result

    def process(self, prompt, response):
        return self.inject(prompt), self.clean_response(response)

    def status(self):
        return {
            "enabled": self.enabled,
            "anchors_loaded": len(self._anchors),
            "identity_block_chars": len(self._identity_block),
            "patterns_active": len(_ROBOTIC_PATTERNS),
        }

    def sample_anchors(self, n=3):
        return self._anchors[:n]


_injector = None

def get_identity_injector(identity_file=None, n_anchors=10):
    global _injector
    if _injector is None:
        _injector = IdentityInjector(identity_file=identity_file, n_anchors=n_anchors)
    return _injector


def health_check() -> dict:
    try:
        from anra_paths import get_identity_file
        identity_file = get_identity_file()
        if identity_file is None:
            return {
                "status": "degraded",
                "enabled": True,
                "anchors_loaded": 0,
                "identity_block_chars": 0,
                "patterns_active": len(_ROBOTIC_PATTERNS),
                "detail": "No identity file found",
            }
        inj = IdentityInjector(identity_file=identity_file)
        st = inj.status()
        return {"status": "ok" if st.get("enabled") else "degraded", **st}
    except Exception as exc:
        return {"status": "degraded", "detail": str(exc)}
