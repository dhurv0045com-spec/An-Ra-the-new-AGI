"""
intelligence/extractor.py — Step 4: Automatic Memory Extraction
Extracts structured facts from raw conversation text.
Uses rule-based patterns + LLM extraction (when model available).
No manual tagging required.
"""

import re
import json
import time
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class ExtractedFact:
    content: str
    category: str
    importance: float
    tags: List[str]
    confidence: float


# ─────────────────────────────────────────────
# Pattern-based extraction (zero dependencies)
# ─────────────────────────────────────────────

# (pattern, category, importance, tags)
EXTRACTION_PATTERNS = [
    # Identity
    (r"(?:my name is|i(?:'m| am) called|call me)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
     "identity", 4.5, ["name", "identity"]),
    (r"i(?:'m| am) (\d+) years? old",
     "identity", 4.0, ["age", "identity"]),
    (r"i(?:'m| am) (?:a|an) ([a-z]+ (?:developer|engineer|designer|writer|researcher|scientist|manager|student|teacher|doctor|lawyer|artist|musician))",
     "identity", 4.5, ["occupation", "identity"]),

    # Location
    (r"i(?:'m| am) (?:based in|living in|located in|from)\s+([A-Z][a-z]+(?:[\s,]+[A-Z][a-z]+)*)",
     "location", 4.0, ["location"]),
    (r"i live in\s+([A-Z][a-z]+(?:[\s,]+[A-Z][a-z]+)*)",
     "location", 4.0, ["location"]),

    # Preferences
    (r"i (?:love|really like|enjoy|prefer|always use)\s+(.{5,60}?)(?:\.|,|$)",
     "preference", 3.5, ["preference", "positive"]),
    (r"i (?:hate|dislike|don(?:'t| not) like|avoid|never use)\s+(.{5,60}?)(?:\.|,|$)",
     "preference", 3.5, ["preference", "negative"]),
    (r"my (?:favorite|favourite)\s+(?:[a-z]+\s+)?(?:is|are)\s+(.{3,60}?)(?:\.|,|$)",
     "preference", 3.5, ["preference"]),

    # Projects
    (r"(?:i(?:'m| am) (?:working on|building|creating|developing)|my project(?:\s+\w+)? is)\s+(.{5,100}?)(?:\.|$)",
     "project", 4.0, ["project", "work"]),
    (r"(?:my (?:startup|company|app|product|service) is called|i(?:'m| am) (?:building|launching))\s+(.{3,60}?)(?:\.|,|$)",
     "project", 4.5, ["project", "startup"]),

    # Goals
    (r"(?:i (?:want|need|plan|intend|hope) to|my goal is to|i(?:'m| am) trying to)\s+(.{5,100}?)(?:\.|$)",
     "goal", 4.0, ["goal"]),
    (r"(?:i(?:'m| am) (?:learning|studying|practicing))\s+(.{3,60}?)(?:\.|,|$)",
     "goal", 3.5, ["goal", "learning"]),

    # Relationships
    (r"my (?:partner|wife|husband|girlfriend|boyfriend|spouse)(?:'s name)? is\s+([A-Z][a-z]+)",
     "relationship", 4.0, ["relationship", "personal"]),
    (r"my (?:boss|manager|colleague|coworker|friend|mentor)(?:'s name)? is\s+([A-Z][a-z]+)",
     "relationship", 3.5, ["relationship", "professional"]),
    (r"my (?:son|daughter|child|kid)(?:'s name)? is\s+([A-Z][a-z]+)",
     "relationship", 4.5, ["relationship", "family"]),

    # Skills / Tech stack
    (r"i(?:'m| am) (?:good at|experienced in|expert in|specialized in|proficient in)\s+(.{3,60}?)(?:\.|,|$)",
     "skill", 3.5, ["skill"]),
    (r"i(?:'m| am) using\s+((?:[A-Z][a-z]*|[A-Z]+)(?:\s+[A-Z][a-z]*)*)\s+(?:for|to|as)",
     "skill", 3.0, ["technology"]),
    (r"my (?:tech stack|stack) (?:is|includes?)\s+(.{3,100}?)(?:\.|$)",
     "skill", 4.0, ["technology", "stack"]),

    # Constraints / Context
    (r"i(?:'m| am) (?:on|using|running)\s+((?:mac|windows|linux|ubuntu|debian)[a-z\s]*?)(?:\.|,|$)",
     "context", 3.0, ["platform", "environment"]),
    (r"(?:my (?:budget|time) is|i only have)\s+(.{3,60}?)(?:\.|$)",
     "constraint", 3.5, ["constraint"]),

    # Explicit requests to remember
    (r"(?:remember|note|keep in mind) that\s+(.{5,200}?)(?:\.|$)",
     "explicit", 5.0, ["explicit", "user-flagged"]),
    (r"(?:important:|fyi:|note:)\s+(.{5,200}?)(?:\.|$)",
     "explicit", 4.5, ["explicit", "user-flagged"]),
]


class PatternExtractor:
    """Fast rule-based extraction. Works offline, zero latency."""

    def __init__(self):
        self._compiled = [
            (re.compile(p, re.IGNORECASE), cat, imp, tags)
            for p, cat, imp, tags in EXTRACTION_PATTERNS
        ]

    def extract(self, text: str,
                speaker: str = "user") -> List[ExtractedFact]:
        """Extract facts from a single utterance."""
        if speaker not in ("user", "human"):
            return []

        facts = []
        for pattern, category, importance, tags in self._compiled:
            for match in pattern.finditer(text):
                captured = match.group(1).strip()
                if len(captured) < 3:
                    continue
                # Build natural language fact
                content = self._naturalize(text, match, captured, category)
                facts.append(ExtractedFact(
                    content=content,
                    category=category,
                    importance=importance,
                    tags=tags,
                    confidence=0.85,
                ))

        return self._deduplicate(facts)

    def _naturalize(self, original: str, match, captured: str,
                     category: str) -> str:
        """Turn extracted fragment into a proper sentence."""
        # Use the full sentence containing the match
        start = max(0, original.rfind(".", 0, match.start()) + 1)
        end = original.find(".", match.end())
        if end == -1:
            end = len(original)
        sentence = original[start:end].strip()
        return sentence if len(sentence) > 10 else f"User: {captured}"

    def _deduplicate(self, facts: List[ExtractedFact]) -> List[ExtractedFact]:
        seen = set()
        unique = []
        for f in facts:
            key = f.content[:50].lower()
            if key not in seen:
                seen.add(key)
                unique.append(f)
        return unique

    def extract_conversation(self, turns: List[Dict]) -> List[ExtractedFact]:
        """Extract from a full conversation (list of {role, content} dicts)."""
        all_facts = []
        for turn in turns:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            all_facts.extend(self.extract(content, speaker=role))
        return all_facts


# ─────────────────────────────────────────────
# LLM-based extraction (rich, structured)
# ─────────────────────────────────────────────

LLM_EXTRACTION_PROMPT = """You are a memory extraction system. Extract factual information about the user from this conversation.

Return a JSON array of facts. Each fact:
{
  "content": "Full sentence stating the fact",
  "category": "one of: preference|goal|project|relationship|skill|location|identity|context|constraint|event|belief|habit",
  "importance": 1-5 (5=critical personal info),
  "tags": ["relevant", "keywords"],
  "confidence": 0-1
}

Only extract facts about the USER (not the assistant).
Only include facts that would still be true in the future.
Skip questions, instructions, and temporary context.

CONVERSATION:
{conversation}

Respond ONLY with a valid JSON array. No explanation."""


class LLMExtractor:
    """
    Uses the main language model for rich fact extraction.
    Falls back to PatternExtractor if model unavailable.
    """

    def __init__(self, model_fn=None):
        """
        model_fn: callable(prompt: str) -> str
        If None, uses PatternExtractor only.
        """
        self._model_fn = model_fn
        self._pattern = PatternExtractor()

    def extract(self, conversation_text: str,
                turns: Optional[List[Dict]] = None) -> List[ExtractedFact]:
        # Always run pattern extraction
        pattern_facts = []
        if turns:
            pattern_facts = self._pattern.extract_conversation(turns)
        else:
            pattern_facts = self._pattern.extract(conversation_text, "user")

        # Add LLM extraction if model available
        if self._model_fn:
            try:
                llm_facts = self._llm_extract(conversation_text)
                # Merge — LLM facts take priority for same content
                return self._merge(pattern_facts, llm_facts)
            except Exception:
                pass

        return pattern_facts

    def _llm_extract(self, conversation_text: str) -> List[ExtractedFact]:
        prompt = LLM_EXTRACTION_PROMPT.format(
            conversation=conversation_text[:3000]
        )
        response = self._model_fn(prompt)
        try:
            # Clean response
            response = response.strip()
            if response.startswith("```"):
                response = re.sub(r"```(?:json)?", "", response).strip()
            data = json.loads(response)
            facts = []
            for item in data:
                facts.append(ExtractedFact(
                    content=item.get("content", ""),
                    category=item.get("category", "general"),
                    importance=float(item.get("importance", 3.0)),
                    tags=item.get("tags", []),
                    confidence=float(item.get("confidence", 0.8)),
                ))
            return [f for f in facts if f.content and f.confidence >= 0.5]
        except (json.JSONDecodeError, KeyError):
            return []

    def _merge(self, pattern: List[ExtractedFact],
               llm: List[ExtractedFact]) -> List[ExtractedFact]:
        combined = list(llm)
        llm_contents = {f.content[:40].lower() for f in llm}
        for f in pattern:
            if f.content[:40].lower() not in llm_contents:
                combined.append(f)
        return combined


# ─────────────────────────────────────────────
# Memory Extractor — ties extraction to semantic store
# ─────────────────────────────────────────────

class MemoryExtractor:
    """
    Full pipeline: conversation → extracted facts → semantic memory.
    Called automatically after every conversation.
    """

    def __init__(self, semantic_memory, model_fn=None):
        self.semantic = semantic_memory
        self.extractor = LLMExtractor(model_fn=model_fn)

    def process_conversation(self, turns: List[Dict],
                               session_id: Optional[str] = None) -> List[Any]:
        """Extract and store all facts from a completed conversation."""
        if not turns:
            return []

        conversation_text = "\n".join(
            f"{t['role'].upper()}: {t['content']}" for t in turns
        )

        facts = self.extractor.extract(conversation_text, turns=turns)
        stored = []
        for fact in facts:
            if fact.confidence < 0.6:
                continue
            mem = self.semantic.store_fact(
                content=fact.content,
                summary=fact.content[:120],
                category=fact.category,
                importance=fact.importance,
                tags=fact.tags,
                metadata={"confidence": fact.confidence,
                          "source": "extraction",
                          "session_id": session_id},
                deduplicate=True,
            )
            stored.append(mem)

        return stored

    def process_single_turn(self, role: str, content: str,
                             session_id: Optional[str] = None) -> List[Any]:
        """Extract from a single utterance in real-time."""
        if role not in ("user", "human"):
            return []
        facts = self.extractor.extract(content, speaker="user")
        stored = []
        for fact in facts:
            if fact.confidence >= 0.7:
                mem = self.semantic.store_fact(
                    content=fact.content,
                    summary=fact.content[:120],
                    category=fact.category,
                    importance=fact.importance,
                    tags=fact.tags,
                    metadata={"confidence": fact.confidence,
                              "source": "realtime",
                              "session_id": session_id},
                    deduplicate=True,
                )
                stored.append(mem)
        return stored
