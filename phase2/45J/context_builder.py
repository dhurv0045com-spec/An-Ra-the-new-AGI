"""
knowledge/context_builder.py — Step 10: Context Builder
Pre-generation context assembly. Pulls from all memory sources.
Token-budget-aware. Most important context always included.
"""

import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass


@dataclass
class ContextBlock:
    name: str
    content: str
    priority: int       # lower = higher priority (always included)
    token_estimate: int


class ContextBuilder:
    """
    Before every generation, assembles context from:
    1. Retrieved relevant memories (semantic + episodic)
    2. Knowledge graph summary
    3. Active goals and tasks
    4. Recent session history
    5. Current conversation

    Token budget is respected — best context always wins.
    """

    def __init__(self, retriever, knowledge_graph,
                 working_memory=None,
                 total_token_budget: int = 2000,
                 model_token_budget: int = 4096):
        self.retriever = retriever
        self.graph = knowledge_graph
        self.working = working_memory
        self.budget = total_token_budget
        self.model_budget = model_token_budget

    def build(self, user_message: str,
              user_id: str = "default",
              conversation_history: Optional[List[Dict]] = None) -> str:
        """
        Full context assembly for a single user message.
        Returns the enriched prompt string ready for generation.
        """
        blocks: List[ContextBlock] = []

        # ── Priority 1: Knowledge graph (who is this person?) ──
        graph_summary = self.graph.get_user_summary(user_id)
        if graph_summary:
            blocks.append(ContextBlock(
                name="user_profile",
                content=f"[USER PROFILE]\n{graph_summary}",
                priority=1,
                token_estimate=len(graph_summary) // 4,
            ))

        # ── Priority 2: High-importance semantic facts ──────────
        try:
            from memory.store import MemoryType
        except ImportError:
            from ..memory.store import MemoryType
        critical_facts = self.retriever.store.list(
            user_id=user_id,
            type=MemoryType.SEMANTIC,
            limit=10,
            order_by="importance DESC"
        )
        if critical_facts:
            facts_text = "\n".join(
                f"• {m.summary}" for m in critical_facts
                if m.importance >= 4.0
            )
            if facts_text:
                blocks.append(ContextBlock(
                    name="key_facts",
                    content=f"[KEY FACTS ABOUT USER]\n{facts_text}",
                    priority=2,
                    token_estimate=len(facts_text) // 4,
                ))

        # ── Priority 3: Relevant retrieved memories ─────────────
        retrieval = self.retriever.retrieve_for_prompt(
            user_message, user_id=user_id,
            token_budget=self.budget // 2,
        )
        if retrieval.has_relevant_memory(threshold=0.25):
            raw_mem_block = retrieval.to_prompt_block(header=False)
            try:
                import sys
                from pathlib import Path
                m_path = Path(__file__).resolve().parent.parent / "45M"
                if str(m_path) not in sys.path:
                    sys.path.insert(0, str(m_path))
                import llm_bridge
                llm = llm_bridge.get_llm_bridge()
                
                prompt = (
                    "Summarize the following retrieved memories into a concise context for the AI:\n"
                    f"{raw_mem_block}\n"
                    "Summary:"
                )
                summarized_mem = llm.generate(prompt, max_new_tokens=150).strip()
                final_mem_block = f"[SUMMARIZED MEMORIES]\n{summarized_mem}"
                token_est = len(final_mem_block) // 4
            except Exception as e:
                final_mem_block = retrieval.to_prompt_block(header=True)
                token_est = retrieval.tokens_used

            blocks.append(ContextBlock(
                name="retrieved_memories",
                content=final_mem_block,
                priority=3,
                token_estimate=token_est,
            ))

        # ── Priority 4: Active goals ─────────────────────────────
        if self.working and self.working._active_goals:
            goals_text = "; ".join(self.working._active_goals[:5])
            blocks.append(ContextBlock(
                name="active_goals",
                content=f"[ACTIVE GOALS]\n{goals_text}",
                priority=4,
                token_estimate=len(goals_text) // 4,
            ))

        # ── Priority 5: Recent conversation turns ───────────────
        if self.working:
            recent = self.working.get_recent_turns(n=10)
            if recent:
                turns_text = "\n".join(
                    f"{t['role'].upper()}: {t['content']}" for t in recent
                )
                blocks.append(ContextBlock(
                    name="recent_history",
                    content=f"[RECENT CONVERSATION]\n{turns_text}",
                    priority=5,
                    token_estimate=len(turns_text) // 4,
                ))
        elif conversation_history:
            recent = conversation_history[-10:]
            turns_text = "\n".join(
                f"{t['role'].upper()}: {t['content']}" for t in recent
            )
            blocks.append(ContextBlock(
                name="conversation_history",
                content=f"[CONVERSATION HISTORY]\n{turns_text}",
                priority=5,
                token_estimate=len(turns_text) // 4,
            ))

        # ── Pack into token budget ───────────────────────────────
        selected = self._pack_blocks(blocks, self.budget)

        # ── Assemble final prompt ────────────────────────────────
        parts = [b.content for b in sorted(selected, key=lambda b: b.priority)]
        if parts:
            parts.append(f"\n[USER MESSAGE]\n{user_message}")
            return "\n\n".join(parts)
        return user_message

    def _pack_blocks(self, blocks: List[ContextBlock],
                      budget: int) -> List[ContextBlock]:
        """Greedy packing by priority. Critical blocks always included."""
        selected = []
        used = 0

        for block in sorted(blocks, key=lambda b: b.priority):
            if block.priority <= 2:
                # Always include critical context
                selected.append(block)
                used += block.token_estimate
            elif used + block.token_estimate <= budget:
                selected.append(block)
                used += block.token_estimate

        return selected

    def get_system_prompt(self, user_id: str = "default") -> str:
        """
        Build a personalized system prompt based on known user facts.
        Injected once at conversation start.
        """
        try:
            from memory.store import MemoryType
        except ImportError:
            from ..memory.store import MemoryType
        parts = [
            "You are a personal AI assistant with memory of past conversations.",
        ]

        graph_summary = self.graph.get_user_summary(user_id)
        if graph_summary:
            parts.append(f"\nWhat you know about this person:\n{graph_summary}")

        # Top preferences
        prefs = self.retriever.store.keyword_search(
            "prefer like enjoy", user_id=user_id,
            type=MemoryType.SEMANTIC, limit=5
        )
        if prefs:
            pref_lines = "\n".join(f"  - {p.summary}" for p in prefs)
            parts.append(f"\nKnown preferences:\n{pref_lines}")

        parts.append(
            "\nUse this knowledge naturally in your responses. "
            "Don't announce that you remember things — just know them."
        )

        return "\n".join(parts)
