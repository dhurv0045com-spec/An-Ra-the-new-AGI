"""
ouroboros_numpy.py — Phase 3 | Component 45O
NumPy-Native Ouroboros Recursive Pass System
=============================================

Implements the same 3-pass recursive reasoning philosophy as ouroboros.py
but uses the LLMBridge's generate() method directly — no PyTorch required.

How the torch version works:
  - Three forward passes through the transformer layers
  - Each pass has a learned "gate vector" that nudges attention

How this NumPy version works:
  - Three full inference calls through the LLM
  - Each pass has a "cognitive mode prefix" — a few words prepended to the
    hidden context that steer the LLM toward a different thinking mode
  - Outputs are blended using tunable weights (loaded from config or defaults)

The three passes:
  Pass 1 — Semantic Anchoring:   "Understand the question thoroughly."
  Pass 2 — Logic Integration:    "Reason step by step about the answer."
  Pass 3 — Adversarial Check:    "Verify: could the answer be wrong? Correct if so."

Results are blended: the final response prioritises the adversarially-verified
pass while retaining depth from passes 1 and 2.

This is operationally equivalent to asking An-Ra the same question three times
from three cognitive angles and synthesizing the best answer — exactly the
spirit of the Ouroboros architecture.

Usage:
    from ouroboros_numpy import OuroborosNumpy

    ouro = OuroborosNumpy(generate_fn=llm_bridge.generate)
    response = ouro.recursive_generate("What is consciousness?")

    # Or with adaptive pass count:
    response = ouro.adaptive_generate("What is 2+2?")    # → 1 pass (simple)
    response = ouro.adaptive_generate("Prove P≠NP")      # → 3 passes (complex)
"""

import re
import json
from pathlib import Path
from typing import Callable, List, Optional, Tuple


# ── Cognitive mode prefixes (the "gate vectors" in prompt space) ───────────────

_PASS_PREFIXES = {
    1: (
        "First, understand the question thoroughly before answering. "
        "Focus on what is actually being asked, the domain, and what kind of answer is needed.\n\n"
    ),
    2: (
        "Now reason step by step. Think through the chain of logic: what follows from what, "
        "what assumptions are being made, and what the most precise answer is.\n\n"
    ),
    3: (
        "Now verify: could this answer be wrong? Is there a contradiction, an edge case, "
        "or a simpler truth? Correct any errors and give the final, verified answer.\n\n"
    ),
}

# Synthesis prompt — blends the three pass outputs
_SYNTHESIS_PREFIX = (
    "You have considered this question from three angles:\n"
    "  1. Semantic understanding\n"
    "  2. Logical reasoning\n"
    "  3. Adversarial verification\n\n"
    "Synthesize the best answer, incorporating the depth from all three perspectives:\n\n"
)

# Complexity signals — queries with these patterns likely need 3 passes
_COMPLEX_PATTERNS = [
    r'\bprove\b', r'\bwhy\b.*\bwhy\b', r'\bexplain\b.*\bhow\b',
    r'\bphilosoph', r'\bconsciousness\b', r'\bmeaning\b',
    r'\bintegrat', r'\bdifferentiat', r'\beigenvalue',
    r'\brecursive\b', r'\bparadox', r'\bproof\b',
    r'\btautology\b', r'\bsyllogism\b', r'\bdeduction',
    r'\bcomplex\b.*\bsystem', r'\bP\s*[=≠]\s*NP',
]
_COMPLEX_RE = [re.compile(p, re.IGNORECASE) for p in _COMPLEX_PATTERNS]

# Simple signals — queries with these patterns can use 1 pass
_SIMPLE_PATTERNS = [
    r'^\s*hi\b', r'^\s*hello\b', r'^\s*what is \d',
    r'^\s*(what|who|when|where) is \w+ \??\s*$',
    r'^\s*\d+\s*[\+\-\*\/]\s*\d+',
]
_SIMPLE_RE = [re.compile(p, re.IGNORECASE) for p in _SIMPLE_PATTERNS]


def _estimate_complexity(query: str) -> int:
    """
    Estimate how many Ouroboros passes a query needs.

    Returns:
        1 — simple query (one pass)
        2 — moderate (two passes)
        3 — complex (full recursive depth)
    """
    # Check simple first
    for pat in _SIMPLE_RE:
        if pat.search(query):
            return 1

    # Count complex signal matches
    hits = sum(1 for pat in _COMPLEX_RE if pat.search(query))
    if hits >= 2:
        return 3
    if hits == 1:
        return 2

    # Length heuristic
    words = len(query.split())
    if words > 50:
        return 3
    if words > 20:
        return 2
    return 2   # default to 2 passes for unknown queries


class OuroborosNumpy:
    """
    NumPy-native implementation of the Ouroboros recursive depth architecture.

    Requires only a callable generate function — no PyTorch, no extra models.
    The generate function should have the signature: (prompt: str) -> str.

    This is designed to wrap the LLMBridge.generate() method.
    """

    def __init__(
        self,
        generate_fn: Callable[[str], str],
        n_passes: int = 3,
        max_new_tokens: int = 250,
        blend_weights: Optional[List[float]] = None,
        config_path: Optional[Path] = None,
        enabled: bool = True,
    ):
        """
        Args:
            generate_fn:    The LLM generation callable: (prompt: str) -> str
            n_passes:       Default number of passes (overridden by adaptive)
            max_new_tokens: Tokens per pass (kept lower to control length)
            blend_weights:  How much each pass contributes to synthesis.
                            Defaults to [0.15, 0.30, 0.55] — later passes weighted more.
            config_path:    Optional JSON file with blend_weights override.
            enabled:        If False, recursive_generate() acts as a passthrough.
        """
        self.generate_fn    = generate_fn
        self.n_passes       = n_passes
        self.max_new_tokens = max_new_tokens
        self.enabled        = enabled

        # Load blend weights
        self.blend_weights = blend_weights or [0.15, 0.30, 0.55]
        if config_path and Path(config_path).is_file():
            try:
                cfg = json.loads(Path(config_path).read_text())
                self.blend_weights = cfg.get("blend_weights", self.blend_weights)
            except Exception:
                pass

        # Ensure weights sum to 1
        total = sum(self.blend_weights)
        self.blend_weights = [w / total for w in self.blend_weights]

        self._pass_count_history: List[int] = []

    def _run_pass(self, context: str, pass_idx: int) -> str:
        """
        Run a single Ouroboros pass.

        Args:
            context:   The current context (original query + accumulated insight)
            pass_idx:  0-indexed pass number (0=semantic, 1=logic, 2=adversarial)

        Returns:
            The LLM's response for this pass.
        """
        prefix = _PASS_PREFIXES.get(pass_idx + 1, "")
        full_prompt = prefix + context
        try:
            return self.generate_fn(full_prompt, max_new_tokens=self.max_new_tokens)
        except TypeError:
            # Fallback if generate_fn doesn't accept max_new_tokens kwarg
            return self.generate_fn(full_prompt)

    def recursive_generate(self, prompt: str, n_passes: Optional[int] = None) -> str:
        """
        Generate a response using N recursive Ouroboros passes.

        Each pass feeds the previous output back as context, so later passes
        reason about earlier passes — not about raw tokens.

        Args:
            prompt:   The user's input.
            n_passes: Override number of passes (uses self.n_passes if None).

        Returns:
            The synthesized final response.
        """
        if not self.enabled:
            return self.generate_fn(prompt)

        n = n_passes or self.n_passes
        n = max(1, min(n, 3))
        self._pass_count_history.append(n)

        if n == 1:
            # Single pass — no recursion needed
            return self._run_pass(prompt, 0)

        # ── Multi-pass recursive loop ──────────────────────────────────────────
        pass_outputs: List[str] = []
        accumulated_context = prompt

        for pass_idx in range(n):
            output = self._run_pass(accumulated_context, pass_idx)
            pass_outputs.append(output)

            # Feed output back as context for next pass (the "tail-eating" loop)
            accumulated_context = (
                f"Original question: {prompt}\n\n"
                f"Pass {pass_idx + 1} reasoning:\n{output}\n\n"
            )

        # ── Synthesis ─────────────────────────────────────────────────────────
        if n >= 3:
            # For 3 passes, synthesize explicitly
            synthesis_context = (
                f"{_SYNTHESIS_PREFIX}"
                f"Original question: {prompt}\n\n"
                + "".join(
                    f"--- Pass {i+1} ---\n{out}\n\n"
                    for i, out in enumerate(pass_outputs)
                )
            )
            try:
                final = self.generate_fn(synthesis_context, max_new_tokens=self.max_new_tokens)
            except TypeError:
                final = self.generate_fn(synthesis_context)
            return final

        # For 2 passes: weight and combine (simpler merge)
        # Weight: pass 1 is context, pass 2 is the primary answer
        # Return pass 2 with a brief acknowledgement of pass 1
        p1, p2 = pass_outputs[0], pass_outputs[1]
        merge_prompt = (
            f"Original question: {prompt}\n\n"
            f"Initial understanding: {p1[:300]}\n\n"
            f"Refined answer (based on the above): {p2}\n\n"
            "Give the final, clean answer incorporating this reasoning:"
        )
        try:
            return self.generate_fn(merge_prompt, max_new_tokens=self.max_new_tokens)
        except TypeError:
            return self.generate_fn(merge_prompt)

    def adaptive_generate(self, prompt: str) -> Tuple[str, int]:
        """
        Generate with automatic pass count based on query complexity.

        Args:
            prompt: The user's query.

        Returns:
            (response, n_passes_used)
        """
        n = _estimate_complexity(prompt)
        response = self.recursive_generate(prompt, n_passes=n)
        return response, n

    def status(self) -> dict:
        """Return Ouroboros status for system dashboard."""
        avg_passes = (
            sum(self._pass_count_history) / len(self._pass_count_history)
            if self._pass_count_history else 0
        )
        return {
            "enabled":       self.enabled,
            "default_passes": self.n_passes,
            "blend_weights": self.blend_weights,
            "calls_made":    len(self._pass_count_history),
            "avg_passes":    round(avg_passes, 2),
        }


def health_check() -> dict:
    return {
        "status": "ok",
        "component": "ouroboros_numpy",
        "adaptive_passes": True,
    }
