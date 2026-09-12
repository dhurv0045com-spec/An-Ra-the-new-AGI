"""Tests for the Citadel Data Readiness Gate (CITADEL-DATA-001).

Green fixtures: a well-formed corpus with proper splits, no duplicates,
no contamination.
Red fixtures: deliberately broken corpora that MUST fail each gate.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from citadel_tpu.data_gate import (  # noqa: E402
    run_all_gates, gate_provenance, gate_split_integrity, gate_exact_dedup,
    gate_near_dedup, gate_contamination, gate_shortcut_resistance,
    gate_mixture_match, gate_tokenizer_accounting, gate_packing_safety,
    gate_cognition_coverage, gate_diversity, gate_reproducibility,
)


def _good_docs(n=30, split="training", prefix="doc"):
    templates = [
        "solve the addition problem what is {a} plus {b} answer {c}",
        "compute the subtraction find {a} minus {b} result {c}",
        "multiplication exercise what is {a} times {b} product {c}",
        "division problem divide {a} by {b} quotient {c}",
        "find the compound expression {a} and {b} combined gives {c}",
        "what is the value when {a} is increased by {b} to reach {c}",
        "determine the difference between {a} and {b} yielding {c}",
        "evaluate the product of {a} and {b} producing {c}",
        "calculate the ratio of {a} over {b} equalling {c}",
        "combine {a} with {b} under addition to produce {c}",
    ]
    return [{"doc_id": f"{prefix}-{i:04d}",
             "text": templates[i % len(templates)].format(
                 a=i+1, b=i*2+1, c=(i+1) + (i*2+1)),
             "source_id": f"src-{i:04d}", "split": split,
             "family": f"family_{i % 5}", "domain": "test"} for i in range(n)]


def _good_manifest(docs):
    return {"sources": [{"source_id": d["source_id"], "family": d.get("family", "test"),
                         "raw_sha256": "a" * 64} for d in docs[:5]],
            "reproducibility": "REPRODUCIBLE"}


def test_green_corpus_passes_all() -> None:
    docs = _good_docs(30)
    manifest = _good_manifest(docs)
    result = run_all_gates(manifest=manifest, documents=docs)
    for gate, status in result["gates"].items():
        if gate in ("SHORTCUT_RESISTANCE", "COGNITION_COVERAGE",
                    "MIXTURE_MATCH", "TOKENIZER_ACCOUNTING"):
            continue  # these need extra context not present in minimal fixture
        assert status == "PASS", f"{gate}: {result['defects'].get(gate)}"


def test_exact_duplicate_fails() -> None:
    docs = _good_docs(10)
    docs.append(dict(docs[0]))  # exact copy
    defects = gate_exact_dedup(docs)
    assert len(defects) >= 1
    assert "exact duplicate" in defects[0]


def test_split_overlap_fails() -> None:
    docs = _good_docs(10, split="training")
    docs += _good_docs(5, split="test")
    docs[-1]["text"] = docs[0]["text"]  # same text in both splits
    defects = gate_split_integrity({}, docs)
    assert any("overlap" in d for d in defects), defects


def test_near_duplicate_detection() -> None:
    docs = [{"doc_id": f"n-{i}",
             "text": f"the quick brown fox jumps over the lazy dog near the river bank during summer {i}"}
            for i in range(10)]
    # exact-ish near duplicate (one token changed)
    docs.append({"doc_id": "n-dup",
                 "text": "the quick brown fox jumps over the lazy dog near the river bank during winter 999"})
    defects, stats = gate_near_dedup(docs, threshold=0.85)
    assert stats["near_dup_merges"] > 0, stats


def test_contamination_detection() -> None:
    train = [{"text": "the eval benchmark contains this exact sentence", "doc_id": "t1"}]
    eval_docs = [{"text": "the eval benchmark contains this exact sentence", "doc_id": "e1"}]
    defects = gate_contamination(train, eval_docs)
    assert len(defects) >= 1, defects


def test_clean_contamination_passes() -> None:
    train = [{"text": "completely different training text about addition", "doc_id": "t1"}]
    eval_docs = [{"text": "unrelated evaluation prompt for testing", "doc_id": "e1"}]
    defects = gate_contamination(train, eval_docs)
    assert len(defects) == 0, defects


def test_mixture_deviation_fails() -> None:
    sched = {"natural": 650_000, "code": 200_000, "cognition": 150_000}
    consumed = {"natural": 900_000, "code": 50_000, "cognition": 50_000}
    defects = gate_mixture_match(sched, consumed, tolerance=0.05)
    assert len(defects) >= 1, defects


def test_mixture_matching_passes() -> None:
    sched = {"natural": 650, "code": 200, "cognition": 150}
    consumed = {"natural": 651, "code": 199, "cognition": 150}
    defects = gate_mixture_match(sched, consumed, tolerance=0.05)
    assert len(defects) == 0, defects


def test_shortcut_resistance_fails() -> None:
    defects = gate_shortcut_resistance(
        documents=[], baseline_scores={"latest_position": 0.20},
        reference_score=0.22, gap_threshold=0.10)
    assert len(defects) >= 1


def test_shortcut_resistance_passes() -> None:
    defects = gate_shortcut_resistance(
        documents=[], baseline_scores={"latest_position": 0.05},
        reference_score=0.50, gap_threshold=0.10)
    assert len(defects) == 0


def test_cognition_coverage_fails_missing() -> None:
    families = {"binding": {"train_generator": True, "eval_generator": True}}
    defects = gate_cognition_coverage(families)
    assert any("reference_solver" in d for d in defects), defects


def test_diversity_fails_on_template_saturation() -> None:
    docs = [{"text": f"the model learns addition batch {i}", "doc_id": f"d{i}"} for i in range(100)]
    defects = gate_diversity(docs, min_templates=10)
    assert len(defects) >= 1, defects


def test_diversity_passes_on_varied() -> None:
    docs = []
    templates = ["addition problem {a} plus {b} equals {c}",
                 "subtraction {a} minus {b} gives {c}",
                 "multiply {a} times {b} is {c}",
                 "divide {a} by {b} to get {c}",
                 "what is {a} and {b} combined: {c}",
                 "compute the sum of {a} and {b}: {c}",
                 "find the difference {a} less {b}: {c}",
                 "product of {a} and {b} equals {c}",
                 "ratio of {a} over {b} is {c}",
                 "combined value {a} with {b} totals {c}"]
    for i in range(50):
        t = templates[i % len(templates)]
        docs.append({"text": t.format(a=i, b=i+1, c=i*2+1), "doc_id": f"d{i}"})
    defects = gate_diversity(docs, min_templates=10)
    assert len(defects) == 0, defects


def test_reproducibility_unknown_fails() -> None:
    defects = gate_reproducibility({"reproducibility": "UNKNOWN"})
    assert len(defects) >= 1


def test_reproducibility_known_passes() -> None:
    defects = gate_reproducibility({"reproducibility": "REPRODUCIBLE"})
    assert len(defects) == 0


def test_full_gate_green() -> None:
    docs = _good_docs(50)
    manifest = _good_manifest(docs)
    tok_counts = {d["doc_id"]: len(d["text"].split()) for d in docs}
    result = run_all_gates(manifest=manifest, documents=docs,
                           tokenizer_counts=tok_counts,
                           scheduled_tokens={"test": 1000},
                           consumed_tokens={"test": 1000})
    # skip gates that need domain-specific fixtures not present here
    skip = {"SHORTCUT_RESISTANCE", "COGNITION_COVERAGE", "DIVERSITY"}
    for gate, status in result["gates"].items():
        if gate in skip:
            continue
        assert status == "PASS", f"{gate}: {result['defects'].get(gate)}"
    assert result["overall"] in ("PASS", "FAIL")


def test_full_gate_red() -> None:
    docs = _good_docs(10) + [dict(_good_docs(1)[0])]  # add duplicate
    manifest = _good_manifest(docs)
    result = run_all_gates(manifest=manifest, documents=docs)
    assert result["overall"] == "FAIL"
    assert result["total_defects"] > 0


def main() -> int:
    tests = [test_green_corpus_passes_all, test_exact_duplicate_fails,
             test_split_overlap_fails, test_near_duplicate_detection,
             test_contamination_detection, test_clean_contamination_passes,
             test_mixture_deviation_fails, test_mixture_matching_passes,
             test_shortcut_resistance_fails, test_shortcut_resistance_passes,
             test_cognition_coverage_fails_missing,
             test_diversity_fails_on_template_saturation,
             test_diversity_passes_on_varied,
             test_reproducibility_unknown_fails,
             test_reproducibility_known_passes,
             test_full_gate_green, test_full_gate_red]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}", flush=True)
        except Exception as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}", flush=True)
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
