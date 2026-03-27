"""
memory_manager.py — Master entry point for the memory system.
Single object that wires all components together.
Provides the clean interface used by the inference pipeline.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any

# ── Path setup ────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from memory.store import MemoryStore, Memory, MemoryType, ImportanceLevel, IMPORTANCE_SCORES
from memory.vectors import VectorStore, TFIDFEmbedder, get_embedder
from memory.memory_types import EpisodicMemory, SemanticMemory, WorkingMemory
from intelligence.extractor import MemoryExtractor
from intelligence.retrieval import HybridRetriever, MemoryInjector
from intelligence.memory_intelligence import (
    ImportanceScorer, MemoryConsolidator, ForgettingSystem
)
from knowledge.graph import KnowledgeGraph
from knowledge.context_builder import ContextBuilder


class MemoryManager:
    """
    The one object the rest of the system talks to.
    Owns: store, vectors, episodic, semantic, working,
          extractor, retriever, graph, context builder.

    Usage:
        mm = MemoryManager(data_dir="data/memory")
        mm.start_session("user123")

        # Store
        mm.store("User prefers dark mode", type="semantic", importance="high")

        # Retrieve
        mems = mm.retrieve("what does user prefer")

        # After conversation
        mm.process_conversation(turns)

        # Build context for generation
        prompt = mm.prepare_prompt(user_message)
    """

    def __init__(self, data_dir: str = "data/memory",
                 user_id: str = "default",
                 model_fn=None,
                 use_neural_embedder: bool = True):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.user_id = user_id
        self._model_fn = model_fn

        # ── Core storage ─────────────────────────────────────
        self.store = MemoryStore(str(self.data_dir / "memory.db"))

        # ── Embeddings ───────────────────────────────────────
        embedder = get_embedder(prefer_neural=use_neural_embedder)
        self.vectors = VectorStore(
            self.store, embedder=embedder,
            index_file=str(self.data_dir / "vector_index.json")
        )

        # ── Typed memory interfaces ───────────────────────────
        self.episodic = EpisodicMemory(self.store, self.vectors, user_id)
        self.semantic  = SemanticMemory(self.store, self.vectors, user_id)
        self.working   = WorkingMemory(self.store, user_id)

        # ── Intelligence ─────────────────────────────────────
        self.extractor = MemoryExtractor(self.semantic, model_fn=model_fn)
        self.retriever = HybridRetriever(self.vectors, self.store)
        self.scorer    = ImportanceScorer()
        self.consolidator = MemoryConsolidator(
            self.store, self.vectors, self.semantic, model_fn=model_fn
        )
        self.forgetter = ForgettingSystem(self.store, self.vectors, self.scorer)

        # ── Knowledge graph ───────────────────────────────────
        self.graph = KnowledgeGraph(
            str(self.data_dir / "knowledge_graph.json"),
            user_id=user_id
        )

        # ── Context builder ───────────────────────────────────
        self.context = ContextBuilder(
            retriever=self.retriever,
            knowledge_graph=self.graph,
            working_memory=self.working,
        )

        # ── Injector (wraps everything for transparent injection) ─
        self.injector = MemoryInjector(
            retriever=self.retriever,
            knowledge_graph=self.graph,
            context_builder=self.context,
        )

    # ─────────────────────────────────────────
    # Primary API
    # ─────────────────────────────────────────

    def start_session(self, user_id: Optional[str] = None) -> str:
        if user_id:
            self.user_id = user_id
        self.working = WorkingMemory(self.store, self.user_id)
        return self.working.session_id

    def store_memory(self, content: str,
                     type: str = "semantic",
                     importance: str = "medium",
                     tags: Optional[List[str]] = None,
                     metadata: Optional[Dict] = None) -> Memory:
        """Store a memory with natural language API."""
        mtype = MemoryType(type)
        imp_score = IMPORTANCE_SCORES.get(ImportanceLevel(importance), 3.0)

        if mtype == MemoryType.SEMANTIC:
            return self.semantic.store_fact(
                content=content,
                summary=content[:120],
                importance=imp_score,
                tags=tags or [],
                metadata=metadata,
            )
        elif mtype == MemoryType.EPISODIC:
            return self.episodic.record(
                content=content,
                summary=content[:120],
                importance=imp_score,
                tags=tags or [],
                metadata=metadata,
            )
        else:
            mem = Memory.create(
                content=content, type=mtype,
                summary=content[:120],
                importance=imp_score,
                user_id=self.user_id,
                tags=tags or [],
                metadata=metadata or {},
            )
            self.store.save(mem)
            self.vectors.embed_and_store(mem)
            return mem

    def retrieve(self, query: str, limit: int = 10,
                  type: Optional[str] = None) -> List[Dict]:
        """Retrieve memories relevant to query."""
        mtype = MemoryType(type) if type else None
        results = self.retriever.retrieve(
            query, user_id=self.user_id,
            top_k=limit,
            memory_types=[mtype] if mtype else None,
        )
        return [
            {
                "id": r.memory.id,
                "summary": r.memory.summary,
                "content": r.memory.content,
                "type": r.memory.type.value,
                "importance": r.memory.importance,
                "score": round(r.final_score, 3),
                "age_days": round(r.memory.age_days(), 1),
                "tags": r.memory.tags,
            }
            for r in results
        ]

    def add_turn(self, role: str, content: str):
        """Record a conversation turn to working memory + extract facts."""
        self.working.add_turn(role, content)
        # Real-time extraction for user turns
        if role in ("user", "human"):
            self.extractor.process_single_turn(
                role, content, self.working.session_id
            )
            self.graph.extract_and_add(content)

    def prepare_prompt(self, user_message: str,
                        token_budget: int = 1500) -> str:
        """
        Full context assembly. Call this before every generation.
        Returns enriched prompt — model never sees the plumbing.
        """
        return self.context.build(
            user_message, user_id=self.user_id
        )

    def process_conversation(self, turns: List[Dict]) -> List[Memory]:
        """
        Post-conversation processing.
        Extracts facts, updates graph, persists session.
        Call at end of each conversation.
        """
        # Extract and store semantic facts
        stored = self.extractor.process_conversation(
            turns, session_id=self.working.session_id
        )
        # Update knowledge graph
        for turn in turns:
            if turn.get("role") in ("user", "human"):
                self.graph.extract_and_add(
                    turn["content"],
                    memory_id=None
                )
        # Persist session to episodic memory
        self.working._turns = turns
        self.working.persist_session()
        self.graph.save()
        return stored

    def consolidate(self, dry_run: bool = False) -> Dict[str, int]:
        """Consolidate and compress memories."""
        return self.consolidator.consolidate(
            user_id=self.user_id, dry_run=dry_run
        )

    def wipe(self, confirm: bool = False) -> int:
        """Full memory wipe."""
        return self.forgetter.wipe_all(
            user_id=self.user_id, confirm=confirm
        )

    def forget_topic(self, topic: str, confirm: bool = False) -> int:
        return self.forgetter.forget_topic(
            topic, user_id=self.user_id, confirm=confirm
        )

    def stats(self) -> Dict:
        store_stats = self.store.stats(self.user_id)
        graph_stats = self.graph.stats()
        return {
            "memory": store_stats,
            "graph": graph_stats,
            "index_size": len(self.vectors.index),
            "session_id": self.working.session_id,
            "user_id": self.user_id,
        }

    def cleanup(self):
        """Save indices and close connections."""
        self.vectors.index.save()
        self.graph.save()
        self.store.close()


# ─────────────────────────────────────────────
# CLI interface
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="45J Memory System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python memory_manager.py --store "User prefers Python" --type semantic --importance high
  python memory_manager.py --retrieve "what language does the user prefer" --limit 5
  python memory_manager.py --graph
  python memory_manager.py --consolidate
  python memory_manager.py --stats
  python memory_manager.py --wipe --confirm
        """
    )
    parser.add_argument("--store",       metavar="CONTENT", help="Store a memory")
    parser.add_argument("--retrieve",    metavar="QUERY",   help="Retrieve memories")
    parser.add_argument("--type",        default="semantic", help="Memory type")
    parser.add_argument("--importance",  default="medium",  help="Importance level")
    parser.add_argument("--limit",       type=int, default=10)
    parser.add_argument("--graph",       action="store_true", help="Show knowledge graph")
    parser.add_argument("--export",      action="store_true", help="Export graph to JSON")
    parser.add_argument("--consolidate", action="store_true", help="Run consolidation")
    parser.add_argument("--wipe",        action="store_true", help="Wipe all memory")
    parser.add_argument("--confirm",     action="store_true")
    parser.add_argument("--stats",       action="store_true")
    parser.add_argument("--user",        default="default")
    parser.add_argument("--data-dir",    default="data/memory")
    parser.add_argument("--demo",        action="store_true", help="Run demo")

    args = parser.parse_args()
    mm = MemoryManager(data_dir=args.data_dir, user_id=args.user)

    try:
        if args.store:
            mem = mm.store_memory(args.store, type=args.type, importance=args.importance)
            print(f"✓ Stored [{mem.type.value}|{mem.importance:.1f}]: {mem.summary}")

        elif args.retrieve:
            results = mm.retrieve(args.retrieve, limit=args.limit)
            print(f"\nTop {len(results)} memories for: {args.retrieve!r}\n")
            for i, r in enumerate(results, 1):
                print(f"{i:2d}. [{r['type']:8s}|{r['importance']:.1f}|score={r['score']:.3f}|{r['age_days']:.0f}d]")
                print(f"    {r['summary']}")

        elif args.graph:
            stats = mm.graph.stats()
            print(f"\nKnowledge Graph")
            print(f"  Nodes: {stats['nodes']}  |  Edges: {stats['edges']}")
            print(f"  By type: {stats['by_type']}")
            print(f"\n  Top nodes:")
            for label, count in stats["top_nodes"]:
                print(f"    {label}: {count} mentions")
            summary = mm.graph.get_user_summary()
            if summary:
                print(f"\n{summary}")
            if args.export:
                out_path = Path(args.data_dir) / "graph_export.json"
                with open(out_path, "w") as f:
                    f.write(mm.graph.export_json())
                print(f"\nExported → {out_path}")

        elif args.consolidate:
            print("Running consolidation...")
            stats = mm.consolidate(dry_run=not args.confirm)
            print(f"  Merged duplicates:     {stats['merged_duplicates']}")
            print(f"  Episodic→semantic:     {stats['episodic_to_semantic']}")
            print(f"  Compressed sessions:   {stats['compressed_old']}")
            if not args.confirm:
                print("  (dry run — pass --confirm to apply)")

        elif args.wipe:
            if not args.confirm:
                count = mm.store.count(args.user)
                print(f"Would delete {count} memories. Pass --confirm to proceed.")
            else:
                count = mm.wipe(confirm=True)
                print(f"✓ Wiped {count} memories.")

        elif args.stats:
            s = mm.stats()
            print(json.dumps(s, indent=2))

        elif args.demo:
            _run_demo(mm)

        else:
            parser.print_help()

    finally:
        mm.cleanup()


def _run_demo(mm: MemoryManager):
    """End-to-end demonstration of the memory system."""
    print("\n" + "=" * 60)
    print(" 45J MEMORY SYSTEM — LIVE DEMO")
    print("=" * 60)

    # Simulate a conversation
    conversation = [
        {"role": "user",      "content": "Hi! My name is Alex and I'm a backend engineer at a startup called Helios."},
        {"role": "assistant", "content": "Nice to meet you, Alex! What kind of work does Helios do?"},
        {"role": "user",      "content": "We build real-time data pipelines. I'm working on a Python service that processes about 10M events per day. I love Python and I'm learning Rust for performance-critical parts."},
        {"role": "assistant", "content": "That's impressive scale. What's the main challenge right now?"},
        {"role": "user",      "content": "Latency. We want to cut p99 from 200ms to under 50ms. I prefer concise technical answers by the way — skip the fluff."},
        {"role": "assistant", "content": "Understood. For p99 latency: profile with py-spy first to find the real bottleneck."},
        {"role": "user",      "content": "I'm also based in Berlin and we're hiring. Our CTO is Mia. Remember that I hate verbose answers."},
    ]

    print("\n[1] Processing conversation...")
    mm.start_session()
    stored = mm.process_conversation(conversation)
    print(f"    → Extracted {len(stored)} semantic facts")

    print("\n[2] Knowledge graph:")
    print(mm.graph.get_user_summary() or "  (empty)")

    print("\n[3] Retrieving memories for 'programming preferences'...")
    results = mm.retrieve("programming language preferences", limit=5)
    for r in results:
        print(f"    [{r['score']:.3f}] {r['summary']}")

    print("\n[4] Building context for new message...")
    prompt = mm.prepare_prompt("What should I use for the hot path?")
    print("    Context injected:")
    for line in prompt.split("\n")[:20]:
        print(f"    {line}")

    print("\n[5] Stats:")
    s = mm.stats()
    print(f"    Total memories: {s['memory']['total']}")
    print(f"    Graph nodes:    {s['graph']['nodes']}")
    print(f"    Graph edges:    {s['graph']['edges']}")
    print(f"    Vector index:   {s['index_size']} entries")

    print("\n✓ Demo complete — model now knows Alex, Helios, Berlin, Python/Rust, p99 goal.")
    print("=" * 60)


if __name__ == "__main__":
    main()
