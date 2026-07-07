from __future__ import annotations

from types import SimpleNamespace

from memory.memory_router import MemoryRouter
from retrieval import (
    HybridRetriever,
    CorpusDedupIndex,
    RetrievalHit,
    RetrievalProvenance,
    RetrievalQuery,
    SkillLibraryRetrieverAdapter,
)
from verification import DEFAULT_VERIFIER_REGISTRY


class StaticRetriever:
    def __init__(self, name: str, hits: list[RetrievalHit]) -> None:
        self.name = name
        self.hits = hits

    def search(self, query: RetrievalQuery) -> list[RetrievalHit]:
        return self.hits[: query.limit * query.candidate_multiplier]


def _hit(record_id: str, text: str, score: float, source: str) -> RetrievalHit:
    return RetrievalHit(
        id=record_id,
        text=text,
        score=score,
        provenance=(RetrievalProvenance(source, 1, score),),
    )


def test_hybrid_rrf_deduplicates_and_preserves_provenance() -> None:
    semantic = StaticRetriever(
        "semantic",
        [_hit("shared", "semantic text", 0.9, "semantic"), _hit("vector", "v", 0.8, "semantic")],
    )
    bm25 = StaticRetriever(
        "bm25",
        [_hit("shared", "keyword text", 8.0, "bm25"), _hit("keyword", "k", 4.0, "bm25")],
    )
    result = HybridRetriever((semantic, bm25)).search(RetrievalQuery("query", limit=3))
    assert result[0].id == "shared"
    assert {item.retriever for item in result[0].provenance} == {"semantic", "bm25"}
    assert len({hit.id for hit in result}) == len(result)


def test_hybrid_weighting_and_ties_are_deterministic() -> None:
    first = StaticRetriever("first", [_hit("b", "b", 1.0, "first")])
    second = StaticRetriever("second", [_hit("a", "a", 1.0, "second")])
    hybrid = HybridRetriever((first, second), weights={"first": 1.0, "second": 1.0})
    assert [hit.id for hit in hybrid.search(RetrievalQuery("q", limit=2))] == ["a", "b"]


def test_memory_router_hybrid_uses_shared_substrate(tmp_path) -> None:
    router = MemoryRouter(dim=16, faiss_index_path=tmp_path / "episodic.faiss")
    stored = router.write(
        "The launch code is amber falcon.",
        metadata={"type": "fact", "salience": 1.0},
    )
    rows = router.read("amber falcon launch code", n=3, tier="hybrid")
    match = next(row for row in rows if row["record_id"] == stored.record_id)
    assert {item["tier"] for item in match["provenance"]} == {"semantic", "bm25"}
    assert match["payload"]["content"] == "The launch code is amber falcon."


def test_skill_library_adapter_uses_canonical_hit_contract() -> None:
    class Skills:
        @staticmethod
        def retrieve(_goal: str, top_k: int) -> list[SimpleNamespace]:
            assert top_k >= 1
            return [
                SimpleNamespace(
                    skill_id="skill-1",
                    name="Repair cache",
                    description="Invalidate stale cache entries",
                    example="Fix cache inconsistency",
                    avg_score=0.9,
                    goal_type="debug",
                )
            ]

    hits = SkillLibraryRetrieverAdapter(Skills()).search(
        RetrievalQuery("debug stale cache", limit=1, filters={"goal_type": "debug"})
    )
    assert hits[0].id == "skill-1"
    assert hits[0].provenance[0].retriever == "skills"


def test_citation_grounding_accepts_shared_retriever() -> None:
    retriever = StaticRetriever(
        "grounding",
        [
            RetrievalHit(
                id="fact-1",
                text="Water has chemical formula H2O.",
                score=1.0,
                metadata={"label": "VERIFIED"},
            )
        ],
    )
    result = DEFAULT_VERIFIER_REGISTRY.verify(
        "citation_grounding",
        {"claim": "Water has formula H2O", "retriever": retriever},
    )
    assert result.score > 0.2
    assert result.reason == "closest EPG memory match found"


def test_corpus_dedup_preserves_exact_default_and_supports_near_duplicate_mode() -> None:
    exact = CorpusDedupIndex()
    assert exact.check_and_add("Alpha beta gamma", record_id="first").duplicate is False
    duplicate = exact.check_and_add("  alpha  BETA gamma ", record_id="second")
    assert duplicate.duplicate is True
    assert duplicate.exact is True
    assert duplicate.matched_id == "first"

    near = CorpusDedupIndex(near_duplicate_threshold=0.7)
    near.check_and_add("alpha beta gamma delta", record_id="base")
    decision = near.check_and_add("alpha beta gamma delta epsilon", record_id="candidate")
    assert decision.duplicate is True
    assert decision.exact is False
    assert decision.matched_id == "base"
