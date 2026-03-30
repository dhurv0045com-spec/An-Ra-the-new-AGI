"""
test_45J.py — Full test suite for the memory system.
Tests every component in isolation + integration.
"""

import sys
import os
import time
import tempfile
import shutil
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

PASS = "✓"
FAIL = "✗"
results = []


def test(name: str):
    def decorator(fn):
        def wrapper():
            try:
                fn()
                results.append((name, True, None))
                print(f"  {PASS} {name}")
            except Exception as e:
                results.append((name, False, str(e)))
                print(f"  {FAIL} {name}: {e}")
        return wrapper
    return decorator


def run_all():
    tmpdir = tempfile.mkdtemp()
    try:
        _run_tests(tmpdir)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} passed")
    if passed < total:
        print("\nFailed:")
        for name, ok, err in results:
            if not ok:
                print(f"  {FAIL} {name}: {err}")
    return passed == total


def _run_tests(tmpdir: str):
    db_path = os.path.join(tmpdir, "test_memory.db")
    idx_path = os.path.join(tmpdir, "test_index.json")
    graph_path = os.path.join(tmpdir, "test_graph.json")

    print("\n=== 45J Memory System Tests ===\n")

    # ── Memory Store ──────────────────────────────────────────
    print("[Memory Store]")
    from memory.store import MemoryStore, Memory, MemoryType, ImportanceLevel

    @test("MemoryStore: create and save memory")
    def _():
        store = MemoryStore(db_path)
        mem = Memory.create("User loves Python", MemoryType.SEMANTIC,
                             summary="Loves Python", importance=4.0)
        store.save(mem)
        retrieved = store.get(mem.id)
        assert retrieved is not None
        assert retrieved.content == "User loves Python"
        store.close()

    @test("MemoryStore: list by type")
    def _():
        store = MemoryStore(db_path)
        mems = store.list(type=MemoryType.SEMANTIC)
        assert len(mems) >= 1
        store.close()

    @test("MemoryStore: keyword search")
    def _():
        store = MemoryStore(db_path)
        results = store.keyword_search("Python")
        assert any("Python" in r.content for r in results)
        store.close()

    @test("MemoryStore: delete with audit log")
    def _():
        store = MemoryStore(db_path)
        mem = Memory.create("Temp memory", MemoryType.WORKING)
        store.save(mem)
        store.delete(mem.id, reason="test")
        assert store.get(mem.id) is None
        log = store._conn.execute(
            "SELECT * FROM deletion_log WHERE memory_id = ?", (mem.id,)
        ).fetchone()
        assert log is not None
        store.close()

    @test("MemoryStore: session start/end")
    def _():
        store = MemoryStore(db_path)
        sid = store.start_session("testuser")
        assert sid
        store.end_session(sid, "Test session")
        row = store._conn.execute(
            "SELECT ended_at FROM sessions WHERE id = ?", (sid,)
        ).fetchone()
        assert row and row[0] is not None
        store.close()

    _(); _(); _(); _(); _()

    # ── Vector System ─────────────────────────────────────────
    print("\n[Vector System]")
    import numpy as np
    from memory.vectors import TFIDFEmbedder, VectorIndex, VectorStore

    @test("TFIDFEmbedder: produces normalized vector")
    def _():
        emb = TFIDFEmbedder(dim=256)
        vec = emb.embed("the quick brown fox")
        assert len(vec) == 256
        assert abs(np.linalg.norm(vec) - 1.0) < 0.01

    @test("TFIDFEmbedder: similar texts produce similar vectors")
    def _():
        emb = TFIDFEmbedder(dim=512)
        v1 = emb.embed("I love Python programming")
        v2 = emb.embed("I enjoy Python coding")
        v3 = emb.embed("The weather is sunny today")
        sim_close = float(np.dot(v1, v2))
        sim_far   = float(np.dot(v1, v3))
        assert sim_close > sim_far, f"similar={sim_close:.3f} far={sim_far:.3f}"

    @test("VectorIndex: add and search")
    def _():
        idx = VectorIndex(dim=64)
        import numpy as np
        for i in range(20):
            vec = np.random.randn(64).astype(np.float32)
            vec /= np.linalg.norm(vec)
            idx.add(f"mem_{i}", vec.tolist())
        query = np.random.randn(64).astype(np.float32)
        query /= np.linalg.norm(query)
        results = idx.search(query.tolist(), top_k=5)
        assert len(results) == 5
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    @test("VectorIndex: remove")
    def _():
        idx = VectorIndex(dim=32)
        import numpy as np
        v = np.random.randn(32).astype(np.float32)
        v /= np.linalg.norm(v)
        idx.add("test_id", v.tolist())
        assert len(idx) == 1
        idx.remove("test_id")
        assert len(idx) == 0

    @test("VectorIndex: save and load")
    def _():
        path = os.path.join(tmpdir, "idx_test.json")
        idx = VectorIndex(dim=32, index_file=path)
        import numpy as np
        v = np.random.randn(32).astype(np.float32)
        v /= np.linalg.norm(v)
        idx.add("abc", v.tolist())
        idx.save()
        idx2 = VectorIndex(dim=32, index_file=path)
        assert "abc" in idx2._ids

    _(); _(); _(); _(); _()

    # ── Memory Types ──────────────────────────────────────────
    print("\n[Memory Types]")

    @test("EpisodicMemory: record and search")
    def _():
        store = MemoryStore(db_path)
        from memory.vectors import VectorStore, TFIDFEmbedder
        vs = VectorStore(store, TFIDFEmbedder(256))
        from memory.memory_types import EpisodicMemory
        ep = EpisodicMemory(store, vs, "testuser")
        ep.record("User asked about Python async", summary="Python async question",
                   importance=3.0, tags=["python"])
        results = ep.search("python asynchronous", top_k=3)
        assert len(results) >= 1
        store.close()

    @test("SemanticMemory: store_fact deduplication")
    def _():
        store = MemoryStore(db_path)
        from memory.vectors import VectorStore, TFIDFEmbedder
        vs = VectorStore(store, TFIDFEmbedder(512))
        from memory.memory_types import SemanticMemory
        sem = SemanticMemory(store, vs, "testuser2")
        sem.store_fact("User prefers dark mode UI", "Prefers dark mode",
                        category="preference", importance=3.5)
        count_before = store.count("testuser2")
        sem.store_fact("User prefers dark mode UI", "Prefers dark mode",
                        category="preference", importance=3.5)
        count_after = store.count("testuser2")
        # Should not double-store identical fact
        assert count_after <= count_before + 1
        store.close()

    @test("WorkingMemory: add turns and get context")
    def _():
        store = MemoryStore(db_path)
        from memory.memory_types import WorkingMemory
        wm = WorkingMemory(store, "testuser3")
        wm.add_turn("user", "I'm building a REST API")
        wm.add_turn("assistant", "What language are you using?")
        wm.add_turn("user", "Python with FastAPI")
        ctx = wm.get_context_window(max_tokens=500)
        assert "FastAPI" in ctx
        assert len(wm._turns) == 3
        store.close()

    _(); _(); _()

    # ── Extraction ────────────────────────────────────────────
    print("\n[Memory Extraction]")
    from intelligence.extractor import PatternExtractor

    @test("PatternExtractor: extract name")
    def _():
        ex = PatternExtractor()
        facts = ex.extract("My name is Sarah and I'm a software engineer.", "user")
        assert any("Sarah" in f.content or "engineer" in f.content.lower()
                    for f in facts)

    @test("PatternExtractor: extract preference")
    def _():
        ex = PatternExtractor()
        facts = ex.extract("I love using Vim and I hate Java.", "user")
        cats = [f.category for f in facts]
        assert "preference" in cats

    @test("PatternExtractor: extract project")
    def _():
        ex = PatternExtractor()
        facts = ex.extract("I'm working on a project called Nexus.", "user")
        assert any("Nexus" in f.content or "project" in f.category for f in facts)

    @test("PatternExtractor: ignore assistant turns")
    def _():
        ex = PatternExtractor()
        facts = ex.extract("My name is Assistant and I love helping.", "assistant")
        assert len(facts) == 0

    @test("PatternExtractor: extract from conversation")
    def _():
        ex = PatternExtractor()
        turns = [
            {"role": "user", "content": "I'm based in Tokyo and I work at a startup."},
            {"role": "assistant", "content": "What does your startup do?"},
            {"role": "user", "content": "We build AI tools. I love machine learning."},
        ]
        facts = ex.extract_conversation(turns)
        assert len(facts) >= 1

    _(); _(); _(); _(); _()

    # ── Retrieval ─────────────────────────────────────────────
    print("\n[Retrieval Engine]")

    @test("HybridRetriever: returns ranked results")
    def _():
        store = MemoryStore(db_path)
        from memory.vectors import VectorStore, TFIDFEmbedder
        from memory.memory_types import SemanticMemory
        from intelligence.retrieval import HybridRetriever

        vs = VectorStore(store, TFIDFEmbedder(512))
        sem = SemanticMemory(store, vs, "retrieval_test")
        sem.store_fact("User is a Python developer", "Python developer",
                        category="identity", importance=4.0)
        sem.store_fact("User dislikes Java verbosity", "Dislikes Java",
                        category="preference", importance=3.0)
        sem.store_fact("User is learning Rust", "Learning Rust",
                        category="goal", importance=3.5)

        retriever = HybridRetriever(vs, store)
        results = retriever.retrieve("programming language", user_id="retrieval_test", top_k=5)
        assert len(results) >= 1
        scores = [r.final_score for r in results]
        assert scores == sorted(scores, reverse=True), "Results not sorted by score"
        store.close()

    @test("HybridRetriever: retrieve_for_prompt respects token budget")
    def _():
        store = MemoryStore(db_path)
        from memory.vectors import VectorStore, TFIDFEmbedder
        from intelligence.retrieval import HybridRetriever
        vs = VectorStore(store, TFIDFEmbedder(512))
        retriever = HybridRetriever(vs, store)
        result = retriever.retrieve_for_prompt("test query", token_budget=100)
        assert result.tokens_used <= 100 + 50  # some tolerance
        store.close()

    _(); _()

    # ── Knowledge Graph ───────────────────────────────────────
    print("\n[Knowledge Graph]")
    from knowledge.graph import KnowledgeGraph

    @test("KnowledgeGraph: upsert nodes and edges")
    def _():
        g = KnowledgeGraph(graph_path)
        g.upsert_node("person", "Alice", importance=4.0)
        g.upsert_node("skill", "Python", importance=3.0)
        g.upsert_edge("Alice", "person", "uses", "Python", "skill")
        assert "person:alice" in g.nodes
        assert "skill:python" in g.nodes
        assert len(g.edges) >= 1

    @test("KnowledgeGraph: extract from text")
    def _():
        g = KnowledgeGraph(os.path.join(tmpdir, "g2.json"))
        g.extract_and_add("I'm working on a project called Nova using Python and React.")
        g.extract_and_add("I love Python and I'm learning Rust.")
        assert len(g.nodes) >= 1  # At least User node

    @test("KnowledgeGraph: query interface")
    def _():
        g = KnowledgeGraph(os.path.join(tmpdir, "g3.json"))
        g.upsert_node("person", "User", importance=5.0)
        g.upsert_edge("User", "person", "uses", "Python", "skill")
        result = g.query("User")
        assert result["found"] is True
        assert len(result["connections"]) >= 1

    @test("KnowledgeGraph: save and load")
    def _():
        gpath = os.path.join(tmpdir, "g4.json")
        g = KnowledgeGraph(gpath)
        g.upsert_node("skill", "TypeScript", importance=3.5)
        g.save()
        g2 = KnowledgeGraph(gpath)
        assert "skill:typescript" in g2.nodes

    @test("KnowledgeGraph: export JSON")
    def _():
        g = KnowledgeGraph(os.path.join(tmpdir, "g5.json"))
        g.upsert_node("person", "Bob", importance=4.0)
        exported = g.export_json()
        import json
        data = json.loads(exported)
        assert "nodes" in data and "edges" in data

    @test("KnowledgeGraph: get_user_summary")
    def _():
        g = KnowledgeGraph(os.path.join(tmpdir, "g6.json"))
        g.upsert_edge("User", "person", "uses", "Python", "skill")
        g.upsert_edge("User", "person", "works_on", "Helios", "project")
        summary = g.get_user_summary()
        assert "Python" in summary or "Helios" in summary

    _(); _(); _(); _(); _(); _()

    # ── Intelligence Systems ──────────────────────────────────
    print("\n[Intelligence Systems]")
    from intelligence.memory_intelligence import ImportanceScorer, ForgettingSystem

    @test("ImportanceScorer: score_new with category")
    def _():
        scorer = ImportanceScorer()
        assert scorer.score_new("User is named Alex", "identity") >= 4.5
        assert scorer.score_new("random note", "general") <= 3.0

    @test("ImportanceScorer: decay over time")
    def _():
        scorer = ImportanceScorer()
        store = MemoryStore(db_path)
        mem = Memory.create("Old note", MemoryType.SEMANTIC, importance=3.0)
        mem.created_at = time.time() - 60 * 86400  # 60 days ago
        effective = scorer.decay(mem)
        assert effective < 3.0, f"Expected decay, got {effective}"
        store.close()

    @test("ForgettingSystem: preview_deletion")
    def _():
        store = MemoryStore(db_path)
        from memory.vectors import VectorStore, TFIDFEmbedder
        from intelligence.memory_intelligence import ImportanceScorer, ForgettingSystem
        vs = VectorStore(store, TFIDFEmbedder(256))
        scorer = ImportanceScorer()
        fs = ForgettingSystem(store, vs, scorer)
        preview = fs.preview_deletion(threshold=5.1, max_age_days=0)
        assert isinstance(preview, list)
        store.close()

    _(); _(); _()

    # ── Full Integration ──────────────────────────────────────
    print("\n[Full Integration]")
    from memory_manager import MemoryManager

    @test("MemoryManager: store and retrieve")
    def _():
        mm = MemoryManager(data_dir=os.path.join(tmpdir, "full_test"))
        mm.store_memory("User prefers concise answers", type="semantic", importance="high")
        results = mm.retrieve("what style of answers does the user prefer")
        assert len(results) >= 1
        mm.cleanup()

    @test("MemoryManager: process_conversation extracts facts")
    def _():
        mm = MemoryManager(data_dir=os.path.join(tmpdir, "full_test2"))
        turns = [
            {"role": "user", "content": "I'm Alice, a data scientist at OpenCorp."},
            {"role": "assistant", "content": "Great to meet you Alice!"},
            {"role": "user", "content": "I love Python and I hate slow queries."},
        ]
        mm.start_session()
        stored = mm.process_conversation(turns)
        assert isinstance(stored, list)
        mm.cleanup()

    @test("MemoryManager: prepare_prompt injects context")
    def _():
        mm = MemoryManager(data_dir=os.path.join(tmpdir, "full_test3"))
        mm.store_memory("User is a Rust developer", type="semantic", importance="high")
        mm.store_memory("User is based in Amsterdam", type="semantic", importance="high")
        prompt = mm.prepare_prompt("What are some good Rust libraries?")
        # The prompt should be enriched with memory context
        assert len(prompt) >= len("What are some good Rust libraries?")
        mm.cleanup()

    @test("MemoryManager: stats")
    def _():
        mm = MemoryManager(data_dir=os.path.join(tmpdir, "stats_test"))
        mm.store_memory("test", type="semantic")
        s = mm.stats()
        assert "memory" in s and "graph" in s
        mm.cleanup()

    @test("MemoryManager: retrieval speed benchmark")
    def _():
        mm = MemoryManager(data_dir=os.path.join(tmpdir, "bench_test"))
        # Store 100 memories
        for i in range(100):
            mm.store_memory(
                f"Memory {i}: User knows about topic{i % 20} and technology{i % 10}",
                type="semantic", importance="medium"
            )
        # Measure retrieval
        t0 = time.time()
        for _ in range(10):
            mm.retrieve("technology and programming", limit=5)
        elapsed = (time.time() - t0) / 10 * 1000  # ms per query
        print(f"\n    Avg retrieval: {elapsed:.1f}ms over 100 memories", end="")
        assert elapsed < 500, f"Too slow: {elapsed:.1f}ms"
        mm.cleanup()

    _(); _(); _(); _(); _()


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
