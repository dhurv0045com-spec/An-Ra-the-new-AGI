"""Development training surface for the Signac Phase-One baseline.

Examples come from the E0 training generator, whose namespace is kept disjoint
from evaluation, plus snapshot-bound repository research records. This remains
development data and cannot satisfy production corpus readiness.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from e0_cognition.contracts import EvaluationSuite
from e0_cognition.training_generators import (
    TRAINING_COGNITION_FAMILIES,
    TRAINING_GENERATOR_VERSION,
    assert_training_eval_disjoint,
    build_training_examples,
)
from signac_100m.research_corpus import RESEARCH_FAMILY, build_research_evidence_corpus
from v5_training.production_entry import (
    frozen_cognition_fractions,
    prepare_data,
)


TRAINING_SURFACE_SCHEMA = "anra-signac-phase1-training-surface/v4"
TRAINING_FAMILY = "verified_cognition"


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _generator_sha256() -> str:
    source = Path(__file__).resolve().parents[1] / "e0_cognition" / "training_generators.py"
    return hashlib.sha256(source.read_bytes()).hexdigest()


def _resolve_corpus_seed(*, seed: int | None, corpus_seed: int | None) -> int:
    """Resolve the old single-seed spelling without coupling future runs."""

    if corpus_seed is None:
        if seed is None:
            raise ValueError("corpus_seed is required")
        corpus_seed = seed
    elif seed is not None and seed != corpus_seed:
        raise ValueError("legacy seed and corpus_seed disagree")
    if type(corpus_seed) is not int or corpus_seed < 0:
        raise ValueError("corpus_seed must be a nonnegative integer")
    return corpus_seed


def build_training_surface(
    *,
    corpus_seed: int | None = None,
    seed: int | None = None,
    count: int = 256,
    evaluation_suite: EvaluationSuite | None = None,
    include_research_evidence: bool = True,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Create deterministic cognition exercises and provenance-bound research docs.

    Training prompt/answer strings are the only records returned. Counterfactual
    answers, causal graphs, and all evaluation truth remain outside the model
    input. Research records retain their status, supported/unsupported claims,
    and caveats. The V5 packer appends the frozen EOS token to every document.
    """

    corpus_seed = _resolve_corpus_seed(seed=seed, corpus_seed=corpus_seed)
    if count <= 0:
        raise ValueError("training count must be positive")
    cognition_fractions = frozen_cognition_fractions()
    examples = build_training_examples(
        seed=corpus_seed,
        count=count,
        family_fractions=cognition_fractions,
    )
    if evaluation_suite is not None:
        evaluation_suite.assert_valid()
        assert_training_eval_disjoint(
            examples,
            {case.template_id for case in evaluation_suite.cases},
        )
    synthetic_documents: list[dict[str, Any]] = []
    cognition_family_by_source: dict[str, str] = {}
    cognition_family_counts = {family: 0 for family in TRAINING_COGNITION_FAMILIES}
    difficulty_counts = {"easy": 0, "medium": 0, "hard": 0}
    surface_counts_by_family = {
        family: {"natural": 0, "semi_natural": 0, "formal": 0}
        for family in TRAINING_COGNITION_FAMILIES
    }
    cognition_axis_counts: dict[str, dict[str, Counter[str]]] = {
        family: {} for family in TRAINING_COGNITION_FAMILIES
    }
    for example in examples:
        model_view = example.model_view()
        text = f"{model_view['context']}\nQuestion: {model_view['query']}\nAnswer: {example.answer}"
        raw_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        source_id = f"{example.template_id}:{example.example_id}"
        cognition_family_by_source[source_id] = example.family
        cognition_family_counts[example.family] += 1
        difficulty_counts[example.difficulty_band] += 1
        surface_counts_by_family[example.family][example.surface] += 1
        for axis, value in example.surface_axes:
            cognition_axis_counts[example.family].setdefault(axis, Counter())[value] += 1
        synthetic_documents.append({
            "doc_id": example.example_id,
            "source_id": source_id,
            "text": text,
            "domain": "synthetic-cognition",
            "family": TRAINING_FAMILY,
            "authorization_category": "first-party-authorized",
            "raw_source_sha256": raw_sha256,
        })
    research_corpus = (
        build_research_evidence_corpus(repo_root=repo_root)
        if include_research_evidence
        else None
    )
    research_documents = []
    if research_corpus is not None:
        allowed_fields = {
            "doc_id", "source_id", "text", "domain", "family",
            "authorization_category", "raw_source_sha256",
        }
        research_documents = [
            {key: value for key, value in row.items() if key in allowed_fields}
            for row in research_corpus["documents"]
        ]
    documents = synthetic_documents + research_documents
    generator_sha = _generator_sha256()
    manifest_body = {
        "schema": TRAINING_SURFACE_SCHEMA,
        "generator_version": TRAINING_GENERATOR_VERSION,
        "generator_sha256": generator_sha,
        "corpus_seed": corpus_seed,
        "synthetic_count": len(synthetic_documents),
        "cognition_family_fractions": cognition_fractions,
        "cognition_family_counts": cognition_family_counts,
        "cognition_family_by_source": cognition_family_by_source,
        "difficulty_counts": difficulty_counts,
        "surface_counts_by_family": surface_counts_by_family,
        "cognition_surface_axis_counts": {
            family: {
                axis: dict(sorted(counts.items()))
                for axis, counts in sorted(axes.items())
            }
            for family, axes in sorted(cognition_axis_counts.items())
        },
        "research_evidence_count": len(research_documents),
        "document_count": len(documents),
        "families": [TRAINING_FAMILY] + ([RESEARCH_FAMILY] if research_documents else []),
        "objective": "causal token prediction over context, query, answer, and pack-appended EOS",
        "documents": documents,
        # Bind the stable evaluation template namespace, not a particular
        # evaluation seed. Independent baseline suites can then vary their
        # generator seed while retaining identical training data identity.
        "evaluation_template_ids_sha256": (
            hashlib.sha256(_canonical_json(sorted({case.template_id for case in evaluation_suite.cases}))).hexdigest()
            if evaluation_suite is not None else None
        ),
        "research_corpus_sha256": research_corpus["sha256"] if research_corpus is not None else None,
        "research_ledger_basis_head": research_corpus["ledger_basis_head"] if research_corpus is not None else None,
    }
    manifest = dict(manifest_body)
    manifest["sha256"] = hashlib.sha256(_canonical_json(manifest_body)).hexdigest()
    return manifest


def prepare_phase1_data(
    *,
    corpus_seed: int | None = None,
    sampler_seed: int | None = None,
    seed: int | None = None,
    run_id: str,
    tokenizer: Any,
    count: int = 256,
    evaluation_suite: EvaluationSuite | None = None,
    include_research_evidence: bool = True,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Pack the development cognition and evidence snapshots through V5.

    This is a development-only experiment input. ``production_entry`` still
    enforces the frozen token ledger, cursor, mixture, and checkpoint contracts;
    its XLA path remains blocked until the actual Kaggle qualification gates pass.
    """

    corpus_seed = _resolve_corpus_seed(seed=seed, corpus_seed=corpus_seed)
    if sampler_seed is None:
        if seed is None:
            raise ValueError("sampler_seed is required when using corpus_seed")
        sampler_seed = seed
    if type(sampler_seed) is not int or sampler_seed < 0:
        raise ValueError("sampler_seed must be a nonnegative integer")
    if not run_id or any(character.isspace() for character in run_id):
        raise ValueError("run_id must be a compact nonempty identity")
    surface = build_training_surface(
        corpus_seed=corpus_seed,
        count=count,
        evaluation_suite=evaluation_suite,
        include_research_evidence=include_research_evidence,
        repo_root=repo_root,
    )
    packed = prepare_data(
        documents=surface["documents"],
        tokenizer=tokenizer,
        run_id=run_id,
        # Dataset manifests are content identities shared across matched
        # runs. run_id is an output lineage label and must not perturb data
        # selection or content hashes.
        seed=sampler_seed,
        data_identity=f"signac-phase1-{surface['sha256'][:16]}-{getattr(tokenizer.identity, 'artifact_sha256')[:16]}",
        cognition_map=surface["cognition_family_by_source"],
        mixture_families=(
            (TRAINING_FAMILY, RESEARCH_FAMILY)
            if include_research_evidence
            else (TRAINING_FAMILY,)
        ),
        development_mode=True,
    )
    return {
        "schema": TRAINING_SURFACE_SCHEMA,
        "status": (
            "DEVELOPMENT_SYNTHETIC_AND_RESEARCH_SNAPSHOT"
            if include_research_evidence
            else "DEVELOPMENT_SYNTHETIC_ONLY"
        ),
        "training_surface_sha256": surface["sha256"],
        "corpus_seed": corpus_seed,
        "sampler_seed": sampler_seed,
        "generator_sha256": surface["generator_sha256"],
        "cognition_family_fractions": surface["cognition_family_fractions"],
        "cognition_family_counts": surface["cognition_family_counts"],
        "cognition_family_by_source": surface["cognition_family_by_source"],
        "difficulty_counts": surface["difficulty_counts"],
        "surface_counts_by_family": surface["surface_counts_by_family"],
        "documents": len(surface["documents"]),
        "synthetic_documents": surface["synthetic_count"],
        "research_evidence_documents": surface["research_evidence_count"],
        "research_corpus_sha256": surface.get("research_corpus_sha256"),
        "data": packed,
        "production_corpus_ready": False,
        "claim_ceiling": "Development-only synthetic cognition and repository evidence snapshot; not production corpus readiness or capability evidence.",
    }


__all__ = [
    "RESEARCH_FAMILY",
    "TRAINING_FAMILY",
    "TRAINING_SURFACE_SCHEMA",
    "build_training_surface",
    "prepare_phase1_data",
]
