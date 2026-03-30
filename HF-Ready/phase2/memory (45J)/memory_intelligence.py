"""
intelligence/importance.py    — Step 7: Memory Importance Scoring
intelligence/consolidation.py — Step 6: Merge and compress memories
intelligence/forgetting.py    — Step 8: Controlled deletion
"""

import math
import time
import re
from typing import List, Dict, Optional, Tuple, Any


# ═══════════════════════════════════════════════════════════
# STEP 7 — IMPORTANCE SCORING
# ═══════════════════════════════════════════════════════════

RECENCY_HALF_LIFE_DAYS = 30.0
FREQUENCY_LOG_BASE = 2.0
MAX_FREQUENCY_BOOST = 2.0

EMOTIONAL_KEYWORDS = {
    # Strong positive → importance boost
    "love", "amazing", "incredible", "critical", "urgent", "important",
    "must", "always", "never", "best", "worst", "hate", "afraid",
    "excited", "stressed", "anxious", "happy", "sad", "worried",
    # Explicit marking
    "remember", "note", "fyi", "heads up", "keep in mind",
    "don't forget", "crucial", "essential",
}

CATEGORY_IMPORTANCE = {
    "identity":     5.0,
    "relationship": 4.5,
    "goal":         4.0,
    "project":      4.0,
    "explicit":     5.0,
    "preference":   3.5,
    "skill":        3.5,
    "location":     3.5,
    "context":      3.0,
    "constraint":   3.5,
    "habit":        3.0,
    "event":        3.0,
    "belief":       3.5,
    "general":      2.5,
}


class ImportanceScorer:
    """
    Assigns and updates importance scores for memories.
    Score is a float 0.0 – 5.0.
    Updates automatically over time as memories are accessed.
    """

    def score_new(self, content: str, category: str = "general",
                   explicit_level: Optional[str] = None) -> float:
        """Initial importance score for a new memory."""
        base = CATEGORY_IMPORTANCE.get(category, 3.0)

        # Emotional weight
        words = set(content.lower().split())
        emotional_hits = len(words & EMOTIONAL_KEYWORDS)
        emotional_boost = min(emotional_hits * 0.3, 1.0)

        # Length signal: longer = more specific = more important
        length_boost = min(len(content) / 500, 0.3)

        # Explicit override
        if explicit_level:
            from store import ImportanceLevel, IMPORTANCE_SCORES
            override = IMPORTANCE_SCORES.get(
                ImportanceLevel(explicit_level), base
            )
            return min(override + emotional_boost, 5.0)

        score = base + emotional_boost + length_boost
        return round(min(score, 5.0), 2)

    def decay(self, memory) -> float:
        """
        Compute current effective importance after time decay.
        Critical and high-importance memories decay slowly.
        Ephemeral memories decay fast.
        """
        base = memory.importance
        age_days = memory.age_days()
        access_count = memory.access_count

        # Memories that are frequently accessed resist decay
        access_factor = min(math.log(1 + access_count, FREQUENCY_LOG_BASE)
                            / MAX_FREQUENCY_BOOST, 1.0)

        # Half-life varies by base importance
        half_life = RECENCY_HALF_LIFE_DAYS * (base / 3.0)
        decay_factor = math.exp(-0.693 * age_days / half_life)

        effective = base * (decay_factor + access_factor * (1 - decay_factor))
        return round(max(effective, 0.0), 3)

    def update_after_access(self, memory) -> float:
        """Boost importance when memory is retrieved (it proved useful)."""
        boost = 0.1 * math.exp(-memory.access_count / 10)
        memory.importance = min(memory.importance + boost, 5.0)
        return memory.importance

    def recompute_all(self, memories: List) -> List:
        """Batch recompute effective importance. Modifies in place."""
        for mem in memories:
            effective = self.decay(mem)
            if abs(effective - mem.importance) > 0.5:
                mem.importance = effective
        return memories


# ═══════════════════════════════════════════════════════════
# STEP 6 — MEMORY CONSOLIDATION
# ═══════════════════════════════════════════════════════════

CONSOLIDATION_PROMPT = """Given these related memories, create one concise summary fact.
Be specific. Preserve all key details.
Return only the summary sentence, nothing else.

MEMORIES:
{memories}

SUMMARY:"""


class MemoryConsolidator:
    """
    Merges duplicate/similar memories.
    Summarizes old episodic memories into semantic facts.
    Compresses conversation history into key points.
    """

    SIMILARITY_THRESHOLD = 0.88
    MIN_AGE_FOR_CONSOLIDATION_DAYS = 7
    BATCH_SIZE = 50

    def __init__(self, memory_store, vector_store, semantic_memory,
                 model_fn=None):
        self.store = memory_store
        self.vs = vector_store
        self.semantic = semantic_memory
        self._model_fn = model_fn

    def consolidate(self, user_id: str = "default",
                     dry_run: bool = False) -> Dict[str, int]:
        """
        Full consolidation pass. Returns counts of operations performed.
        """
        stats = {
            "merged_duplicates": 0,
            "episodic_to_semantic": 0,
            "compressed_old": 0,
        }

        stats["merged_duplicates"] = self._merge_duplicates(user_id, dry_run)
        stats["episodic_to_semantic"] = self._episodic_to_semantic(user_id, dry_run)
        stats["compressed_old"] = self._compress_old_episodes(user_id, dry_run)

        return stats

    def _merge_duplicates(self, user_id: str,
                           dry_run: bool = False) -> int:
        """Find and merge semantically near-duplicate memories."""
        from store import MemoryType
        merged = 0

        for mtype in [MemoryType.SEMANTIC]:
            memories = self.store.list(
                user_id=user_id, type=mtype, limit=500
            )

            processed = set()
            for mem in memories:
                if mem.id in processed:
                    continue
                # Find memories similar to this one
                similar = self.vs.search(
                    mem.content, top_k=5, user_id=user_id, type=mtype,
                    exclude_ids=[mem.id] + list(processed)
                )
                for sim_mem, score in similar:
                    if score >= self.SIMILARITY_THRESHOLD:
                        if not dry_run:
                            self._merge_pair(mem, sim_mem)
                        processed.add(sim_mem.id)
                        merged += 1

        return merged

    def _merge_pair(self, keep, remove):
        """Merge remove into keep, then delete remove."""
        # Combine tags
        keep.tags = list(set(keep.tags + remove.tags))
        # Take higher importance
        keep.importance = max(keep.importance, remove.importance)
        # If both have content, combine into summary
        if len(remove.content) > len(keep.content):
            keep.summary = remove.summary
        keep.access_count += remove.access_count
        self.store.update(keep)
        self.store.delete(remove.id, reason="consolidation_merge")

    def _episodic_to_semantic(self, user_id: str,
                               dry_run: bool = False) -> int:
        """
        Convert old episodic memories into semantic facts.
        Old episodes (> 14 days, low importance) → compressed semantic facts.
        """
        from store import MemoryType
        cutoff = time.time() - 14 * 86400
        rows = self.store._conn.execute("""
            SELECT * FROM memories
            WHERE user_id = ? AND type = ? AND created_at < ? AND importance < 3.5
            ORDER BY created_at ASC LIMIT 100
        """, (user_id, MemoryType.EPISODIC.value, cutoff)).fetchall()

        if not rows:
            return 0

        converted = 0
        for row in rows:
            mem = self.store._row_to_memory(row)
            # Extract key facts and store as semantic
            facts = self._extract_key_facts(mem.content)
            if facts and not dry_run:
                for fact in facts:
                    self.semantic.store_fact(
                        content=fact,
                        summary=fact[:120],
                        category="episodic_compressed",
                        importance=min(mem.importance + 0.5, 5.0),
                        metadata={"source_episode": mem.id,
                                   "source_date": mem.created_at},
                        deduplicate=True,
                    )
                # Delete the old episode
                self.store.delete(mem.id, reason="consolidated_to_semantic",
                                   user_id=user_id)
                converted += 1

        return converted

    def _extract_key_facts(self, text: str) -> List[str]:
        """Extract key sentences from a conversation."""
        # Simple heuristic: sentences with important keywords
        sentences = re.split(r'[.!?]+', text)
        facts = []
        keywords = {"my", "i am", "i'm", "i use", "i like", "i work",
                    "i prefer", "remember", "important", "always", "never"}
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 20:
                continue
            sent_lower = sent.lower()
            if any(kw in sent_lower for kw in keywords):
                # Clean up turn prefixes
                sent = re.sub(r'^(?:USER|ASSISTANT|HUMAN):\s*', '', sent)
                if sent:
                    facts.append(sent[:300])
        return facts[:5]

    def _compress_old_episodes(self, user_id: str,
                                dry_run: bool = False) -> int:
        """
        Compress multiple old episode turns from same session
        into a single summarized episode.
        """
        from store import MemoryType
        # Find sessions with many episodic memories
        rows = self.store._conn.execute("""
            SELECT session_id, COUNT(*) as cnt
            FROM memories
            WHERE user_id = ? AND type = ? AND session_id IS NOT NULL
              AND created_at < ?
            GROUP BY session_id
            HAVING cnt > 5
            LIMIT 10
        """, (user_id, MemoryType.EPISODIC.value,
               time.time() - 30 * 86400)).fetchall()

        compressed = 0
        for row in rows:
            session_id = row[0]
            if not dry_run:
                self._compress_session(session_id, user_id)
            compressed += 1

        return compressed

    def _compress_session(self, session_id: str, user_id: str):
        """Merge all episodes from a session into one."""
        from store import MemoryType
        rows = self.store._conn.execute("""
            SELECT * FROM memories
            WHERE user_id = ? AND type = ? AND session_id = ?
            ORDER BY created_at ASC
        """, (user_id, MemoryType.EPISODIC.value, session_id)).fetchall()

        if len(rows) < 2:
            return

        memories = [self.store._row_to_memory(r) for r in rows]
        combined_content = "\n\n".join(m.content for m in memories)
        combined_tags = list({t for m in memories for t in m.tags})
        max_importance = max(m.importance for m in memories)
        earliest = min(m.created_at for m in memories)

        # Create compressed memory
        from store import Memory
        compressed = Memory.create(
            content=combined_content[:5000],
            type=MemoryType.EPISODIC,
            summary=f"Compressed session ({len(memories)} turns)",
            importance=max_importance,
            user_id=user_id,
            session_id=session_id,
            tags=combined_tags[:20],
            metadata={"compressed": True, "original_count": len(memories)},
        )
        compressed.created_at = earliest
        self.store.save(compressed)
        self.vs.embed_and_store(compressed)

        # Delete originals
        for mem in memories:
            self.store.delete(mem.id, reason="session_compression",
                               user_id=user_id)


# ═══════════════════════════════════════════════════════════
# STEP 8 — FORGETTING SYSTEM
# ═══════════════════════════════════════════════════════════

class ForgettingSystem:
    """
    Controlled, intelligent forgetting.
    Protects important memories. Expires junk. Enables privacy wipes.
    """

    def __init__(self, memory_store, vector_store, importance_scorer):
        self.store = memory_store
        self.vs = vector_store
        self.scorer = importance_scorer

    def expire_low_importance(self, user_id: str = "default",
                               threshold: float = 1.5,
                               max_age_days: float = 30,
                               dry_run: bool = False) -> int:
        """Delete low-importance memories older than max_age_days."""
        cutoff = time.time() - max_age_days * 86400
        rows = self.store._conn.execute("""
            SELECT id, importance, created_at FROM memories
            WHERE user_id = ? AND importance < ? AND created_at < ?
              AND (expires_at IS NULL OR expires_at > ?)
        """, (user_id, threshold, cutoff, time.time())).fetchall()

        count = 0
        for row in rows:
            mem_id = row[0]
            if not dry_run:
                self.store.delete(mem_id, reason="auto_expire_low_importance",
                                   user_id=user_id)
            count += 1
        return count

    def forget_topic(self, topic: str, user_id: str = "default",
                      confirm: bool = False) -> int:
        """Forget everything about a specific topic."""
        if not confirm:
            raise ValueError("Pass confirm=True to forget a topic")
        memories = self.store.keyword_search(
            topic, user_id=user_id, limit=1000
        )
        count = 0
        for mem in memories:
            self.store.delete(mem.id, reason=f"topic_forget:{topic}",
                               user_id=user_id)
            self.vs.index.remove(mem.id)
            count += 1
        return count

    def forget_by_id(self, memory_id: str, user_id: str = "default"):
        """User explicitly deletes a specific memory."""
        self.store.delete(memory_id, reason="user_explicit_delete",
                           user_id=user_id)
        self.vs.index.remove(memory_id)

    def wipe_all(self, user_id: str = "default", confirm: bool = False) -> int:
        """Full memory wipe — privacy nuclear option."""
        if not confirm:
            raise ValueError("Pass confirm=True to wipe all memory")
        memories = self.store.list(user_id=user_id, limit=100000)
        count = len(memories)
        self.store.wipe(user_id=user_id, confirm=True)
        # Rebuild empty index
        for mem in memories:
            self.vs.index.remove(mem.id)
        return count

    def scheduled_cleanup(self, user_id: str = "default") -> Dict[str, int]:
        """
        Full cleanup pass. Run overnight or when store gets large.
        """
        stats = {}
        # Purge expired
        stats["expired"] = self.store.purge_expired()
        # Expire low importance
        stats["low_importance"] = self.expire_low_importance(
            user_id=user_id, threshold=1.5, max_age_days=14
        )
        return stats

    def preview_deletion(self, user_id: str = "default",
                          threshold: float = 1.5,
                          max_age_days: float = 30) -> List[Dict]:
        """Show what would be deleted — before committing."""
        cutoff = time.time() - max_age_days * 86400
        rows = self.store._conn.execute("""
            SELECT id, summary, importance, created_at
            FROM memories
            WHERE user_id = ? AND importance < ? AND created_at < ?
            ORDER BY importance ASC, created_at ASC
            LIMIT 50
        """, (user_id, threshold, cutoff)).fetchall()
        return [{"id": r[0], "summary": r[1], "importance": r[2],
                 "age_days": (time.time() - r[3]) / 86400}
                for r in rows]
