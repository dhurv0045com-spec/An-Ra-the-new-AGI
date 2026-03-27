"""
memory/episodic.py — Specific past conversations
memory/semantic.py — Facts and knowledge
memory/working.py  — Current session context
All three in one file for brevity; each is a clean class.
"""

import time
import json
from typing import List, Optional, Dict, Any
from .store import Memory, MemoryType, ImportanceLevel, IMPORTANCE_SCORES


# ─────────────────────────────────────────────
# EPISODIC memory — what happened, when
# ─────────────────────────────────────────────

class EpisodicMemory:
    """
    Stores specific past conversations with full context.
    Automatically expires old low-importance episodes.
    """
    DEFAULT_TTL_DAYS = 180  # 6 months

    def __init__(self, store, vector_store, user_id: str = "default"):
        self.store = store
        self.vectors = vector_store
        self.user_id = user_id

    def record(self, content: str, summary: str,
               session_id: Optional[str] = None,
               importance: float = 3.0,
               tags: Optional[List[str]] = None,
               metadata: Optional[Dict] = None) -> Memory:
        """
        Store a conversation episode.
        content: full conversation text or turn
        summary: 1-line summary
        """
        # Low-importance episodes expire after 6 months;
        # high-importance ones never expire.
        ttl = None if importance >= 4.0 else self.DEFAULT_TTL_DAYS * (importance / 3.0)

        mem = Memory.create(
            content=content,
            type=MemoryType.EPISODIC,
            summary=summary,
            metadata=metadata or {},
            importance=importance,
            user_id=self.user_id,
            session_id=session_id,
            tags=tags or [],
            ttl_days=ttl,
        )
        self.store.save(mem)
        self.vectors.embed_and_store(mem)
        return mem

    def get_recent(self, n: int = 20) -> List[Memory]:
        return self.store.list(
            user_id=self.user_id,
            type=MemoryType.EPISODIC,
            limit=n,
            order_by="created_at DESC"
        )

    def search(self, query: str, top_k: int = 5) -> List[tuple]:
        return self.vectors.search(
            query, top_k=top_k, user_id=self.user_id,
            type=MemoryType.EPISODIC
        )

    def get_session(self, session_id: str) -> List[Memory]:
        """All episodes from a specific session."""
        rows = self.store._conn.execute(
            """SELECT * FROM memories
               WHERE user_id = ? AND type = ? AND session_id = ?
               ORDER BY created_at ASC""",
            (self.user_id, MemoryType.EPISODIC.value, session_id)
        ).fetchall()
        return [self.store._row_to_memory(r) for r in rows]


# ─────────────────────────────────────────────
# SEMANTIC memory — facts extracted from conversations
# ─────────────────────────────────────────────

class SemanticMemory:
    """
    Stores durable facts and knowledge about the user.
    Supports: upsert (avoid duplicates), structured facts,
    topic-based retrieval.
    """
    FACT_CATEGORIES = [
        "preference", "goal", "project", "relationship",
        "skill", "location", "belief", "habit", "event",
        "constraint", "identity", "context",
    ]

    def __init__(self, store, vector_store, user_id: str = "default"):
        self.store = store
        self.vectors = vector_store
        self.user_id = user_id

    def store_fact(self, content: str, summary: str,
                   category: str = "general",
                   importance: float = 3.5,
                   tags: Optional[List[str]] = None,
                   metadata: Optional[Dict] = None,
                   deduplicate: bool = True) -> Memory:
        """
        Store a semantic fact. Deduplicates by similarity to avoid
        storing the same fact twice.
        """
        if deduplicate:
            existing = self._find_duplicate(content)
            if existing:
                # Update importance and refresh timestamp
                existing.importance = max(existing.importance, importance)
                existing.accessed_at = time.time()
                existing.access_count += 1
                if tags:
                    existing.tags = list(set(existing.tags + tags))
                self.store.update(existing)
                return existing

        mem = Memory.create(
            content=content,
            type=MemoryType.SEMANTIC,
            summary=summary,
            metadata={**(metadata or {}), "category": category},
            importance=importance,
            user_id=self.user_id,
            tags=([category] + (tags or [])),
        )
        self.store.save(mem)
        self.vectors.embed_and_store(mem)
        return mem

    def _find_duplicate(self, content: str,
                         threshold: float = 0.92) -> Optional[Memory]:
        """Check if a very similar fact already exists."""
        results = self.vectors.search(
            content, top_k=1, user_id=self.user_id,
            type=MemoryType.SEMANTIC
        )
        if results and results[0][1] >= threshold:
            return results[0][0]
        return None

    def get_by_category(self, category: str, limit: int = 20) -> List[Memory]:
        return self.store.keyword_search(
            category, user_id=self.user_id,
            type=MemoryType.SEMANTIC, limit=limit
        )

    def search(self, query: str, top_k: int = 10) -> List[tuple]:
        return self.vectors.search(
            query, top_k=top_k, user_id=self.user_id,
            type=MemoryType.SEMANTIC
        )

    def get_all_facts(self, limit: int = 500) -> List[Memory]:
        return self.store.list(
            user_id=self.user_id,
            type=MemoryType.SEMANTIC,
            limit=limit,
            order_by="importance DESC, accessed_at DESC"
        )

    def get_categories(self) -> Dict[str, int]:
        """Count facts per category."""
        facts = self.get_all_facts(limit=10000)
        cats: Dict[str, int] = {}
        for f in facts:
            cat = f.metadata.get("category", "general")
            cats[cat] = cats.get(cat, 0) + 1
        return cats


# ─────────────────────────────────────────────
# WORKING memory — current session state
# ─────────────────────────────────────────────

class WorkingMemory:
    """
    Volatile in-memory state for the current session.
    Automatically persists to DB at end of session.
    Tracks: active context, recent turns, current goals, entities.
    """

    def __init__(self, store, user_id: str = "default",
                 session_id: Optional[str] = None,
                 max_turns: int = 50):
        self.store = store
        self.user_id = user_id
        self.session_id = session_id or store.start_session(user_id)
        self.max_turns = max_turns

        # In-memory state (fast access, no DB roundtrip)
        self._turns: List[Dict] = []           # conversation history
        self._entities: Dict[str, Any] = {}    # named entities seen this session
        self._active_goals: List[str] = []     # what user is trying to do
        self._context_notes: List[str] = []    # scratchpad
        self._session_facts: List[str] = []    # facts learned this session

    def add_turn(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Record a single conversation turn."""
        turn = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {},
        }
        self._turns.append(turn)
        if len(self._turns) > self.max_turns:
            self._turns = self._turns[-self.max_turns:]

    def set_entity(self, name: str, value: Any):
        """Track named entity in current session."""
        self._entities[name] = {
            "value": value,
            "seen_at": time.time(),
        }

    def add_goal(self, goal: str):
        if goal not in self._active_goals:
            self._active_goals.append(goal)

    def complete_goal(self, goal: str):
        self._active_goals = [g for g in self._active_goals if g != goal]

    def note(self, text: str):
        self._context_notes.append(text)

    def add_session_fact(self, fact: str):
        self._session_facts.append(fact)

    def get_recent_turns(self, n: int = 10) -> List[Dict]:
        return self._turns[-n:]

    def get_full_history(self) -> List[Dict]:
        return self._turns

    def get_context_window(self, max_tokens: int = 2000) -> str:
        """Build compact context string from working memory."""
        parts = []
        if self._active_goals:
            parts.append("Active goals: " + "; ".join(self._active_goals))
        if self._entities:
            ents = [f"{k}={v['value']}" for k, v in
                    list(self._entities.items())[-10:]]
            parts.append("Context: " + ", ".join(ents))
        if self._context_notes:
            parts.append("Notes: " + "; ".join(self._context_notes[-5:]))

        # Recent turns
        turns_text = []
        for t in self._turns[-20:]:
            turns_text.append(f"{t['role'].upper()}: {t['content']}")

        context = "\n".join(parts)
        history = "\n".join(turns_text)

        # Rough token budget (4 chars ≈ 1 token)
        budget_chars = max_tokens * 4
        if len(context) + len(history) > budget_chars:
            # Truncate history, keep context
            history = history[-(budget_chars - len(context)):]

        return f"{context}\n\n{history}".strip()

    def persist_session(self, summary: Optional[str] = None):
        """Save session to episodic memory at end of conversation."""
        if not self._turns:
            return
        full_text = "\n".join(
            f"{t['role'].upper()}: {t['content']}" for t in self._turns
        )
        auto_summary = summary or self._auto_summarize()
        # Save as episodic memory
        mem = Memory.create(
            content=full_text,
            type=MemoryType.EPISODIC,
            summary=auto_summary,
            metadata={
                "turn_count": len(self._turns),
                "entities": list(self._entities.keys()),
                "goals": self._active_goals,
                "facts_learned": self._session_facts,
            },
            importance=3.0,
            user_id=self.user_id,
            session_id=self.session_id,
            tags=list(self._entities.keys())[:10],
        )
        self.store.save(mem)
        self.store.end_session(self.session_id, auto_summary)
        return mem

    def _auto_summarize(self) -> str:
        if not self._turns:
            return "Empty session"
        topics = list(self._entities.keys())[:5]
        n = len(self._turns)
        topic_str = f" about {', '.join(topics)}" if topics else ""
        return f"Conversation with {n} turns{topic_str}"

    def reset(self):
        """Clear working memory for new session."""
        self._turns.clear()
        self._entities.clear()
        self._active_goals.clear()
        self._context_notes.clear()
        self._session_facts.clear()
        self.session_id = self.store.start_session(self.user_id)
