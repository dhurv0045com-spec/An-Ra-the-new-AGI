"""
================================================================================
FILE: agent/intelligence/reasoning.py
PROJECT: Agent Loop — 45K
PURPOSE: Chain of thought reasoning + self-critique before major decisions
================================================================================
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ReasoningChain:
    """A recorded chain of thought for one decision."""
    decision_id:  str
    question:     str
    steps:        List[str]       # Intermediate reasoning steps
    conclusion:   str
    critique:     str             # Self-critique of the reasoning
    confidence:   float           # 0.0–1.0
    timestamp:    float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "id":         self.decision_id,
            "question":   self.question,
            "steps":      self.steps,
            "conclusion": self.conclusion,
            "critique":   self.critique,
            "confidence": self.confidence,
        }

    def formatted(self) -> str:
        lines = [f"REASONING: {self.question}", ""]
        for i, step in enumerate(self.steps, 1):
            lines.append(f"  {i}. {step}")
        lines.append(f"\nCONCLUSION: {self.conclusion}")
        lines.append(f"SELF-CRITIQUE: {self.critique}")
        lines.append(f"Confidence: {self.confidence:.0%}")
        return "\n".join(lines)


class ReasoningEngine:
    """
    Structured chain-of-thought reasoning for the agent.

    Before important decisions (which tool, how to handle failure, etc.),
    the agent reasons through options step by step, then self-critiques
    before acting. All reasoning is stored for auditability.

    In production this calls the language model. Here: deterministic
    heuristic reasoning that covers the common cases well.
    """

    def __init__(self):
        self._chains: List[ReasoningChain] = []

    def reason_about_step(self, step_title: str, instruction: str,
                           available_tools: List[str],
                           context: str = "") -> ReasoningChain:
        """Reason about how to approach a step before executing it using LLM."""
        import hashlib
        decision_id = "R_" + hashlib.md5(f"{step_title}{time.time()}".encode()).hexdigest()[:6]

        try:
            import sys
            from pathlib import Path
            m_path = Path(__file__).resolve().parent.parent / "45M"
            if str(m_path) not in sys.path:
                sys.path.insert(0, str(m_path))
            import llm_bridge
            llm = llm_bridge.get_llm_bridge()
            
            prompt = (
                f"Analyze the following step for an AI agent.\n"
                f"Goal: {step_title}\n"
                f"Instruction: {instruction}\n"
                f"Tools avail: {', '.join(available_tools)}\n"
                f"Context: {context}\n"
                "List your step-by-step plan, then state a conclusion and self-critique."
            )
            response = llm.generate(prompt, max_new_tokens=250)
            
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            steps = lines[:max(1, len(lines)-2)]
            conclusion = lines[-2] if len(lines) >= 2 else "LLM produced plan."
            critique = lines[-1] if len(lines) >= 1 else "Critique omitted."
            
            confidence = 0.85
        except Exception as e:
            logger.warning(f"LLM Reasoning failed, falling back: {e}")
            steps = [
                f"What is the goal of this step? → '{step_title}'",
                f"What information do I have? → instruction length: {len(instruction)} chars"
                + (f", context available: yes" if context else ", no prior context"),
                f"Which tools are available? → {', '.join(available_tools)}",
                f"What is the most direct path to completing this step?",
                f"What could go wrong? → tool failure, insufficient info, ambiguous instruction",
                f"What is my fallback if the primary approach fails?",
            ]

            best_tool = available_tools[0] if available_tools else "memory_tool"
            conclusion = (
                f"Use {best_tool} as the primary tool. "
                f"If it fails, fall back to the next available tool. "
                f"Store intermediate results in memory_tool for downstream steps."
            )

            critique = (
                "This plan assumes the primary tool will return useful output. "
                "If the step requires information not available to any tool, "
                "I should store what I know and flag the gap rather than loop."
            )
            confidence = 0.75

        chain = ReasoningChain(
            decision_id=decision_id,
            question=f"How do I complete: {step_title}?",
            steps=steps,
            conclusion=conclusion,
            critique=critique,
            confidence=confidence,
        )
        self._chains.append(chain)
        logger.debug(f"Reasoning chain {decision_id} created for step '{step_title}'")
        return chain

    def reason_about_failure(self, step_title: str, error: str,
                               retries_remaining: int) -> ReasoningChain:
        """Reason about how to handle a step failure using LLM."""
        import hashlib
        decision_id = "R_" + hashlib.md5(f"fail{step_title}{time.time()}".encode()).hexdigest()[:6]

        try:
            import sys
            from pathlib import Path
            m_path = Path(__file__).resolve().parent.parent / "45M"
            if str(m_path) not in sys.path:
                sys.path.insert(0, str(m_path))
            import llm_bridge
            llm = llm_bridge.get_llm_bridge()
            prompt = (
                f"Agent step '{step_title}' failed.\n"
                f"Error: {error}\n"
                f"Retries left: {retries_remaining}\n"
                "Plan a recovery strategy step by step. End with CONCLUSION and CRITIQUE."
            )
            response = llm.generate(prompt, max_new_tokens=200)
            
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            steps = lines[:max(1, len(lines)-2)]
            conclusion = lines[-2] if len(lines) >= 2 else "Refine parameters and retry."
            critique = lines[-1] if len(lines) >= 1 else "Root cause might be systemic."
            
            confidence = 0.6 if retries_remaining > 0 else 0.8
        except Exception as e:
            logger.warning(f"LLM Reasoning failed, falling back: {e}")
            steps = [
                f"What failed? → '{step_title}': {error[:100]}",
                f"Is this a permanent or transient failure?",
                f"Retries remaining: {retries_remaining}",
                "Can a different tool or approach succeed?",
                "Does this failure block downstream steps?",
            ]

            if retries_remaining > 0:
                conclusion = (
                    f"Retry with a modified approach. {retries_remaining} retries remaining. "
                    "Change the tool input or try a different tool."
                )
                confidence = 0.6
            else:
                conclusion = (
                    "No retries remaining. Escalate to human if critical, "
                    "or mark as failed and continue with remaining steps."
                )
                confidence = 0.8
                
            critique = "This recovery plan may not work if the root cause is systemic."

        chain = ReasoningChain(
            decision_id=decision_id,
            question=f"How do I handle failure of: {step_title}?",
            steps=steps,
            conclusion=conclusion,
            critique=critique,
            confidence=confidence,
        )
        self._chains.append(chain)
        return chain

    def get_history(self) -> List[ReasoningChain]:
        return list(self._chains)

    def explain_decision(self, decision_id: str) -> Optional[str]:
        for c in self._chains:
            if c.decision_id == decision_id:
                return c.formatted()
        return None
