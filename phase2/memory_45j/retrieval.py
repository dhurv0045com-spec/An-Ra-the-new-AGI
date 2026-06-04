"""
intelligence/retrieval.py — Step 5: Memory Retrieval Engine
Hybrid search: semantic similarity + keyword + recency + importance.
Fits best memories into token budget. Injects into prompt automatically.
"""

import math
import time
import re
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class RetrievedMemory:
    memory: Any          # Memory object
    semantic_score: float
    keyword_score: float
    recency_score: float
    importance_score: float
    final_score: float

    def to_context_str(self) -> str:
        age = self.memory.age_days()
        age_str = f"{age:.0f}d ago" if age > 1 else "recent"
        return f"[{self.memory.type.value}|{age_str}] {self.memory.summary}"


class HybridRetriever:
    """
    Multi-signal retrieval with configurable weights.
    Returns memories ranked by combined relevance score.
    """

    def __init__(self, vector_store, memory_store,
                 semantic_w: float = 0.50,
                 keyword_w: float = 0.20,
                 recency_w: float = 0.15,
                 importance_w: float = 0.15):
        self.vs = vector_store
        self.store = memory_store
        self.w_sem = semantic_w
        self.w_kw  = keyword_w
        self.w_rec = recency_w
        self.w_imp = importance_w

    def retrieve(self, query: str,
                 user_id: str = "default",
                 top_k: int = 10,
                 memory_types=None,
                 exclude_ids: Optional[List[str]] = None,
                 recency_bias: float = 1.0,
                 min_score: float = 0.05) -> List[RetrievedMemory]:
        """
        Full hybrid retrieval pipeline.
        recency_bias > 1.0 = prefer newer memories
        """
        candidates: Dict[str, RetrievedMemory] = {}

        # ── Semantic search ──────────────────
        types = memory_types or [None]
        for mt in types:
            sem_results = self.vs.search(
                query, top_k=top_k * 3,
                user_id=user_id, type=mt,
                exclude_ids=exclude_ids,
            )
            for mem, score in sem_results:
                if mem.id not in candidates:
                    candidates[mem.id] = RetrievedMemory(
                        memory=mem,
                        semantic_score=score,
                        keyword_score=0.0,
                        recency_score=0.0,
                        importance_score=0.0,
                        final_score=0.0,
                    )
                else:
                    candidates[mem.id].semantic_score = max(
                        candidates[mem.id].semantic_score, score
                    )

        # ── Keyword search ───────────────────
        kw_results = self.store.keyword_search(
            query, user_id=user_id, limit=top_k * 2
        )
        max_kw = len(kw_results) or 1
        for rank, mem in enumerate(kw_results):
            if mem.id in candidates:
                candidates[mem.id].keyword_score = 1.0 - (rank / max_kw)
            else:
                if exclude_ids and mem.id in exclude_ids:
                    continue
                candidates[mem.id] = RetrievedMemory(
                    memory=mem,
                    semantic_score=0.0,
                    keyword_score=1.0 - (rank / max_kw),
                    recency_score=0.0,
                    importance_score=0.0,
                    final_score=0.0,
                )

        # ── Score recency and importance ─────
        now = time.time()
        for rm in candidates.values():
            mem = rm.memory
            # Recency: exponential decay, half-life = 30 days
            age_days = (now - mem.created_at) / 86400
            rm.recency_score = math.exp(-0.023 * age_days * recency_bias)

            # Importance: normalized 0-1
            rm.importance_score = min(mem.importance / 5.0, 1.0)

            # Boost by access frequency (popular memories matter more)
            access_boost = min(math.log(1 + mem.access_count) / 5.0, 0.2)

            rm.final_score = (
                self.w_sem * rm.semantic_score +
                self.w_kw  * rm.keyword_score +
                self.w_rec * rm.recency_score +
                self.w_imp * rm.importance_score +
                access_boost
            )

        # ── Filter and rank ──────────────────
        results = [r for r in candidates.values()
                   if r.final_score >= min_score and not r.memory.is_expired()]
        results.sort(key=lambda r: r.final_score, reverse=True)

        # Touch retrieved memories
        for r in results[:top_k]:
            r.memory.touch()
            self.store.update(r.memory)

        return results[:top_k]

    def retrieve_for_prompt(self, query: str,
                             user_id: str = "default",
                             token_budget: int = 1500,
                             include_types=None) -> "RetrievalResult":
        """
        Retrieve memories and pack them into a token budget.
        Returns structured result ready for prompt injection.
        """
        memories = self.retrieve(
            query, user_id=user_id,
            top_k=20,
            memory_types=include_types,
        )
        return self._pack_into_budget(query, memories, token_budget)

    def _pack_into_budget(self, query: str,
                           memories: List[RetrievedMemory],
                           token_budget: int) -> "RetrievalResult":
        """Greedily pack most important memories into token budget."""
        char_budget = token_budget * 4  # ~4 chars/token
        selected = []
        used = 0

        # Prioritize: critical/high importance first, then by score
        sorted_mems = sorted(
            memories,
            key=lambda r: (r.memory.importance >= 4.0, r.final_score),
            reverse=True
        )

        for rm in sorted_mems:
            text = rm.to_context_str()
            cost = len(text)
            if used + cost <= char_budget:
                selected.append(rm)
                used += cost
            if used >= char_budget * 0.9:
                break

        return RetrievalResult(query=query, memories=selected,
                                tokens_used=used // 4)


@dataclass
class RetrievalResult:
    query: str
    memories: List[RetrievedMemory]
    tokens_used: int

    def to_prompt_block(self, header: bool = True) -> str:
        """Format retrieved memories for injection into prompt."""
        if not self.memories:
            return ""

        lines = []
        if header:
            lines.append("=== RELEVANT MEMORIES ===")

        # Group by type
        by_type: Dict[str, List[RetrievedMemory]] = {}
        for rm in self.memories:
            t = rm.memory.type.value
            by_type.setdefault(t, []).append(rm)

        for type_name, rms in by_type.items():
            lines.append(f"\n[{type_name.upper()} MEMORIES]")
            for rm in rms:
                age = rm.memory.age_days()
                age_str = f"{age:.0f} days ago" if age >= 1 else "today"
                lines.append(f"• {rm.memory.summary} ({age_str})")

        if header:
            lines.append("=========================\n")

        return "\n".join(lines)

    def has_relevant_memory(self, threshold: float = 0.3) -> bool:
        return any(r.final_score >= threshold for r in self.memories)


# ─────────────────────────────────────────────
# Auto-injection wrapper
# ─────────────────────────────────────────────

class MemoryInjector:
    """
    Wraps the full pipeline: given a prompt, transparently
    retrieves + injects memory context before generation.
    """

    def __init__(self, retriever: HybridRetriever,
                 knowledge_graph=None,
                 context_builder=None):
        self.retriever = retriever
        self.graph = knowledge_graph
        self.context_builder = context_builder

    def prepare_prompt(self, user_message: str,
                        user_id: str = "default",
                        token_budget: int = 1500,
                        working_memory=None) -> str:
        """
        Full context assembly. Returns enriched prompt.
        The model never sees this machinery — it just gets a richer prompt.
        """
        parts = []

        # Retrieved memories
        result = self.retriever.retrieve_for_prompt(
            user_message, user_id=user_id,
            token_budget=token_budget // 2,
        )
        if result.has_relevant_memory(threshold=0.2):
            parts.append(result.to_prompt_block())

        # Knowledge graph summary
        if self.graph:
            graph_summary = self.graph.get_user_summary(user_id)
            if graph_summary:
                parts.append(f"=== WHO YOU ARE TALKING TO ===\n{graph_summary}\n")

        # Working memory / current session context
        if working_memory:
            ctx = working_memory.get_context_window(max_tokens=400)
            if ctx.strip():
                parts.append(f"=== CURRENT SESSION ===\n{ctx}\n")

        # Append original message
        parts.append(user_message)
        return "\n".join(parts)
