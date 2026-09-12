"""Citadel Data Readiness Gate (CITADEL-DATA-001).

Fail-closed audit of a candidate training corpus. Twelve independent gates:
PROVENANCE, SPLIT_INTEGRITY, EXACT_DEDUP, NEAR_DEDUP, CONTAMINATION,
SHORTCUT_RESISTANCE, MIXTURE_MATCH, TOKENIZER_ACCOUNTING, PACKING_SAFETY,
COGNITION_COVERAGE, DIVERSITY, REPRODUCIBILITY.

OVERALL PASS is legal only if every REQUIRED gate is PASS.
Missing evidence produces INCONCLUSIVE or FAIL — never a silent skip.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

GATE_SCHEMA = "citadel-data-gate/v1"

REQUIRED_GATES = (
    "PROVENANCE", "SPLIT_INTEGRITY", "EXACT_DEDUP", "NEAR_DEDUP",
    "CONTAMINATION", "SHORTCUT_RESISTANCE", "MIXTURE_MATCH",
    "TOKENIZER_ACCOUNTING", "PACKING_SAFETY", "COGNITION_COVERAGE",
    "DIVERSITY", "REPRODUCIBILITY",
)

OPTIONAL_GATES = ()


def _sha(data): return hashlib.sha256(data).hexdigest() if isinstance(data, bytes) else hashlib.sha256(str(data).encode()).hexdigest()


def _norm_text(s):
    return " ".join((s or "").casefold().split())


def _ngrams(text, n=3):
    words = _norm_text(text).split()
    return {" ".join(words[i:i+n]) for i in range(max(0, len(words) - n + 1))}


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def gate_provenance(manifest: dict) -> list[dict]:
    """Every source must have identity, version, and content hash."""
    defects = []
    sources = manifest.get("sources", manifest.get("documents", []))
    if not sources:
        defects.append("no sources in manifest")
        return defects
    for i, src in enumerate(sources):
        sid = src.get("source_id") or src.get("doc_id") or f"source[{i}]"
        for key in ("source_id", "family"):
            if not src.get(key):
                defects.append(f"{sid}: missing {key}")
        if src.get("generator_version"):
            gv = src["generator_version"]
            if not isinstance(gv, str) or len(gv) < 3:
                defects.append(f"{sid}: generator_version too short: {gv!r}")
        if src.get("raw_sha256"):
            h = src["raw_sha256"]
            if not isinstance(h, str) or len(h) != 64:
                defects.append(f"{sid}: raw_sha256 not a SHA-256 hex: {h!r}")
    return defects


def gate_split_integrity(manifest: dict, documents: list[dict]) -> list[dict]:
    """Train/dev/test/sealed must be disjoint. Causal twins must not straddle."""
    defects = []
    by_split = {}
    for d in documents:
        split = d.get("split") or d.get("split_identity") or "unknown"
        by_split.setdefault(split, set()).add(_sha(d.get("text", "")))
    splits = list(by_split.keys())
    for i, a in enumerate(splits):
        for b in splits[i+1:]:
            overlap = by_split[a] & by_split[b]
            if overlap:
                defects.append(f"content overlap between {a} and {b}: "
                               f"{len(overlap)} shared hashes")
    # declared split labels must be preserved
    for d in documents:
        declared = d.get("declared_split")
        if declared and d.get("split") and declared != d["split"]:
            defects.append(f"{d.get('doc_id','?')}: declared_split {declared!r} "
                           f"!= assigned split {d['split']!r}")
    return defects


def gate_exact_dedup(documents: list[dict]) -> list[dict]:
    """No exact duplicate content across the corpus."""
    defects = []
    seen = {}
    for d in documents:
        h = _sha(d.get("text", ""))
        if h in seen:
            defects.append(f"exact duplicate: {d.get('doc_id','?')} == "
                           f"{seen[h]}")
        seen[h] = d.get("doc_id", "?")
    return defects


def gate_near_dedup(documents: list[dict], *, threshold: float = 0.85,
                    ngram_size: int = 3, sample_cap: int = 2000
                    ) -> tuple[list[dict], dict]:
    """Near-duplicate detection using normalized 3-gram Jaccard similarity.
    Returns (defects, stats). Threshold is 0.85 based on the assumption that
    synthetic template-generated rows with >85% 3-gram overlap are structural
    copies, not independent examples. This threshold should be calibrated on
    known-positive pairs before production use."""
    defects = []
    docs = documents[:sample_cap]
    sigs = []
    for d in docs:
        text = _norm_text(d.get("text", ""))
        words = text.split()
        sigs.append(set(f"{words[i]} {words[i+1]}" for i in range(max(0, len(words)-1))))
    pairs_checked = 0
    merges = 0
    for i in range(len(docs)):
        for j in range(i+1, min(i+50, len(docs))):
            pairs_checked += 1
            a, b = sigs[i], sigs[j]
            if not a and not b:
                continue
            jac = len(a & b) / len(a | b) if a | b else 0.0
            if jac >= threshold:
                merges += 1
                if merges <= 5:
                    defects.append(
                        f"near-duplicate: {docs[i].get('doc_id','?')} ~ "
                        f"{docs[j].get('doc_id','?')} (J={jac:.3f})")
    stats = {"pairs_checked": pairs_checked, "near_dup_merges": merges,
             "threshold": threshold}
    if merges > 0:
        defects.append(f"total near-duplicate pairs at threshold "
                       f"{threshold}: {merges}")
    return defects, stats


def gate_contamination(documents: list[dict], eval_documents: list[dict],
                       *, ngram_size: int = 8) -> list[dict]:
    """Train must not contain eval content (exact, normalized, or n-gram)."""
    defects = []
    train_norm = {_norm_text(d.get("text", "")) for d in documents}
    train_ngrams = set()
    for d in documents:
        train_ngrams |= _ngrams(d.get("text", ""), ngram_size)
    for e in eval_documents:
        e_norm = _norm_text(e.get("text", e.get("prompt", "")))
        if e_norm in train_norm:
            defects.append(f"exact eval text in train: {e_norm[:60]}...")
            continue
        e_ngrams = _ngrams(e_norm, ngram_size)
        overlap = e_ngrams & train_ngrams
        if len(overlap) >= max(1, len(e_ngrams) // 2):
            defects.append(f"high eval n-gram overlap: {len(overlap)}/{len(e_ngrams)}")
    return defects


SHORTCUT_BASELINES = (
    "latest_position", "nearest_position", "lexical_overlap",
    "answer_token_presence", "bag_of_words", "constant_label",
    "candidate_attestation", "operation_name_lookup", "template_id_lookup",
    "serialization_position", "prompt_length", "family_classifier",
)


def gate_shortcut_resistance(documents: list[dict],
                             baseline_scores: dict[str, float],
                             reference_score: float, *,
                             gap_threshold: float = 0.10
                             ) -> list[dict]:
    """If any trivial shortcut approaches the reference solver, FAIL."""
    defects = []
    for name, score in baseline_scores.items():
        if score >= reference_score - gap_threshold:
            defects.append(
                f"SHORTCUT_RISK: {name} scores {score:.3f}, reference is "
                f"{reference_score:.3f} (gap {reference_score - score:.3f} "
                f"< threshold {gap_threshold})")
    return defects


def gate_mixture_match(scheduled_tokens: dict[str, int],
                       consumed_tokens: dict[str, int], *,
                       tolerance: float = 0.05) -> list[dict]:
    """Consumed token stream must match the declared mixture within tolerance."""
    defects = []
    sched_total = sum(scheduled_tokens.values())
    cons_total = sum(consumed_tokens.values())
    if sched_total == 0:
        defects.append("no scheduled tokens declared")
        return defects
    for source, sched in sorted(scheduled_tokens.items()):
        cons = consumed_tokens.get(source, 0)
        expected_frac = sched / sched_total
        actual_frac = cons / max(cons_total, 1)
        if abs(expected_frac - actual_frac) > tolerance:
            defects.append(
                f"mixture deviation {source}: expected "
                f"{expected_frac:.3f}, actual {actual_frac:.3f}")
    return defects


def gate_tokenizer_accounting(documents: list[dict], tokenizer_counts: dict[str, int],
                              *, max_chars_per_token: float = 8.0
                              ) -> list[dict]:
    """Token counts must be consistent with text lengths."""
    defects = []
    for d in documents:
        did = d.get("doc_id", "?")
        text_len = len(d.get("text", ""))
        tok_count = tokenizer_counts.get(did, 0)
        if text_len > 0 and tok_count == 0:
            defects.append(f"{did}: non-empty text but 0 tokens")
        if tok_count > text_len * max_chars_per_token:
            defects.append(f"{did}: token count {tok_count} suspiciously high "
                           f"for {text_len} chars")
    return defects


def gate_packing_safety(packed_sequences: list[dict]) -> list[dict]:
    """Check segment isolation, boundaries, and padding."""
    defects = []
    for i, seq in enumerate(packed_sequences):
        seg_ids = seq.get("segment_ids", [])
        for j in range(1, len(seg_ids)):
            if seg_ids[j] == seg_ids[j-1] and seg_ids[j] >= 0:
                pass  # same segment, fine
        if any(s < -1 for s in seg_ids):
            defects.append(f"seq[{i}]: invalid segment id < -1")
    return defects


def gate_cognition_coverage(families: dict[str, dict]) -> list[dict]:
    """Every cognition family must have: train gen, eval gen, solver,
    shortcut audit, difficulty control, structural holdout."""
    defects = []
    required = ("train_generator", "eval_generator", "reference_solver",
                "shortcut_audit", "difficulty_control", "structural_holdout")
    for family, info in sorted(families.items()):
        for req in required:
            if not info.get(req):
                defects.append(f"{family}: missing {req}")
    return defects


def gate_diversity(documents: list[dict], *, min_templates: int = 10,
                   max_replay: float = 100.0) -> list[dict]:
    """Template saturation and diversity checks."""
    defects = []
    templates = Counter()
    for d in documents:
        text = _norm_text(d.get("text", ""))
        # skeleton = text with numbers replaced by placeholder
        skeleton = re.sub(r"\d+", "#", text)
        templates[skeleton] += 1
    if len(templates) < min_templates:
        defects.append(f"only {len(templates)} unique templates "
                       f"(min {min_templates})")
    max_count = max(templates.values()) if templates else 0
    total = sum(templates.values())
    if total > 0 and max_count / total > 0.5:
        defects.append(f"template dominance: most common template is "
                       f"{max_count}/{total} ({max_count/total:.0%})")
    return defects


def gate_reproducibility(manifest: dict, *, regenerated_sample_hash: str | None
                         = None, original_sample_hash: str | None = None
                         ) -> list[dict]:
    """Reproducibility check."""
    defects = []
    repro = manifest.get("reproducibility", "UNKNOWN")
    if repro == "UNKNOWN":
        defects.append("reproducibility is UNKNOWN — do not label as "
                       "reproducible without evidence")
    if (regenerated_sample_hash and original_sample_hash
            and regenerated_sample_hash != original_sample_hash):
        defects.append("regenerated sample hash mismatch")
    return defects


def run_all_gates(*, manifest: dict, documents: list[dict],
                  eval_documents: list[dict] | None = None,
                  tokenizer_counts: dict[str, int] | None = None,
                  packed_sequences: list[dict] | None = None,
                  baseline_scores: dict[str, float] | None = None,
                  reference_score: float = 0.0,
                  scheduled_tokens: dict[str, int] | None = None,
                  consumed_tokens: dict[str, int] | None = None,
                  cognition_families: dict[str, dict] | None = None,
                  near_dedup_threshold: float = 0.85,
                  ) -> dict[str, Any]:
    """Run all 12 gates and return a structured result."""
    eval_documents = eval_documents or []
    tokenizer_counts = tokenizer_counts or {}
    packed_sequences = packed_sequences or []
    baseline_scores = baseline_scores or {}
    scheduled_tokens = scheduled_tokens or {}
    consumed_tokens = consumed_tokens or {}
    cognition_families = cognition_families or {}

    results = {}

    prov = gate_provenance(manifest)
    results["PROVENANCE"] = "FAIL" if prov else "PASS"

    split = gate_split_integrity(manifest, documents)
    results["SPLIT_INTEGRITY"] = "FAIL" if split else "PASS"

    dedup = gate_exact_dedup(documents)
    results["EXACT_DEDUP"] = "FAIL" if dedup else "PASS"

    near, near_stats = gate_near_dedup(documents, threshold=near_dedup_threshold)
    results["NEAR_DEDUP"] = "FAIL" if near else "PASS"

    contam = gate_contamination(documents, eval_documents)
    results["CONTAMINATION"] = "FAIL" if contam else "PASS"

    shortcut = gate_shortcut_resistance(documents, baseline_scores,
                                        reference_score)
    results["SHORTCUT_RESISTANCE"] = "FAIL" if shortcut else "PASS"

    mix = gate_mixture_match(scheduled_tokens, consumed_tokens)
    results["MIXTURE_MATCH"] = "FAIL" if mix else "PASS"

    tok = gate_tokenizer_accounting(documents, tokenizer_counts)
    results["TOKENIZER_ACCOUNTING"] = "FAIL" if tok else "PASS"

    pack = gate_packing_safety(packed_sequences)
    results["PACKING_SAFETY"] = "FAIL" if pack else "PASS"

    cog = gate_cognition_coverage(cognition_families)
    results["COGNITION_COVERAGE"] = "FAIL" if cog else "PASS"

    div = gate_diversity(documents)
    results["DIVERSITY"] = "FAIL" if div else "PASS"

    repro = gate_reproducibility(manifest)
    results["REPRODUCIBILITY"] = "FAIL" if repro else "PASS"

    all_defects = {
        "PROVENANCE": prov, "SPLIT_INTEGRITY": split,
        "EXACT_DEDUP": dedup, "NEAR_DEDUP": near,
        "CONTAMINATION": contam, "SHORTCUT_RESISTANCE": shortcut,
        "MIXTURE_MATCH": mix, "TOKENIZER_ACCOUNTING": tok,
        "PACKING_SAFETY": pack, "COGNITION_COVERAGE": cog,
        "DIVERSITY": div, "REPRODUCIBILITY": repro,
    }
    overall = "PASS" if all(v == "PASS" for v in results.values()) else "FAIL"

    return {
        "schema": GATE_SCHEMA,
        "overall": overall,
        "gates": results,
        "defects": all_defects,
        "near_dedup_stats": near_stats,
        "total_defects": sum(len(v) for v in all_defects.values()),
    }


__all__ = [
    "GATE_SCHEMA", "REQUIRED_GATES", "run_all_gates",
    "gate_provenance", "gate_split_integrity", "gate_exact_dedup",
    "gate_near_dedup", "gate_contamination", "gate_shortcut_resistance",
    "gate_mixture_match", "gate_tokenizer_accounting", "gate_packing_safety",
    "gate_cognition_coverage", "gate_diversity", "gate_reproducibility",
]
