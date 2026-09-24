from __future__ import annotations

import hashlib
import math
from types import SimpleNamespace

import pytest

from e0_cognition.contracts import Split
from e0_cognition.evaluation_generators import build_evaluation_suite
from e0_cognition.training_generators import (
    TRAINING_COGNITION_FAMILIES,
    build_training_examples,
)
from signac_100m.research_corpus import build_research_evidence_corpus
from signac_100m.training import RESEARCH_FAMILY, TRAINING_FAMILY, build_training_surface, prepare_phase1_data


class _Tokenizer:
    identity = SimpleNamespace(artifact_sha256="a" * 64, vocabulary_size=24_576)

    def encode(self, text: str) -> list[int]:
        return [
            4 + int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % 24_572
            for token in text.split()
        ]


def test_phase1_training_surface_is_deterministic_and_eval_namespace_separate():
    suite = build_evaluation_suite(Split.DEVELOPMENT, seed=31, groups_per_family=1)
    left = build_training_surface(seed=31, count=18, evaluation_suite=suite)
    right = build_training_surface(seed=31, count=18, evaluation_suite=suite)
    other = build_training_surface(seed=32, count=18, evaluation_suite=suite)
    other_eval_seed = build_training_surface(
        corpus_seed=31,
        count=18,
        evaluation_suite=build_evaluation_suite(
            Split.DEVELOPMENT, seed=32, groups_per_family=1
        ),
    )

    assert left["sha256"] == right["sha256"]
    assert left["sha256"] != other["sha256"]
    assert left["sha256"] == other_eval_seed["sha256"]
    assert left["synthetic_count"] == 18
    assert left["research_evidence_count"] == 92
    assert len(left["documents"]) == 110
    assert {document["family"] for document in left["documents"]} == {
        TRAINING_FAMILY, RESEARCH_FAMILY
    }
    synthetic = [doc for doc in left["documents"] if doc["family"] == TRAINING_FAMILY]
    assert all("Answer:" in document["text"] for document in synthetic)
    assert all("counterfactual" not in document["text"].lower() for document in synthetic)
    research = [doc for doc in left["documents"] if doc["family"] == RESEARCH_FAMILY]
    assert any("FORMATION-MUX-001 S5" in document["text"] for document in research)
    assert any("INCONCLUSIVE" in document["text"] for document in research)
    assert len(left["evaluation_template_ids_sha256"]) == 64


def test_phase1_data_uses_canonical_pack_path_and_stays_development_only():
    suite = build_evaluation_suite(Split.DEVELOPMENT, seed=31, groups_per_family=1)
    prepared = prepare_phase1_data(
        seed=31,
        run_id="signac-phase1-smoke",
        tokenizer=_Tokenizer(),
        count=24,
        evaluation_suite=suite,
    )
    assert prepared["status"] == "DEVELOPMENT_SYNTHETIC_AND_RESEARCH_SNAPSHOT"
    assert prepared["production_corpus_ready"] is False
    assert prepared["synthetic_documents"] == 24
    assert prepared["research_evidence_documents"] == 92
    assert prepared["documents"] == 116
    assert prepared["data"]["manifest"].tokenizer_sha256 == "a" * 64
    assert prepared["data"]["packed"]
    assert {source.split for source in prepared["data"]["manifest"].sources} == {"training"}
    assert set(prepared["data"]["manifest"].tokens_by_family) == {
        TRAINING_FAMILY, RESEARCH_FAMILY
    }
    family_by_source = prepared["data"]["cell_of_source"]
    assert set(prepared["cognition_family_counts"]) == set(TRAINING_COGNITION_FAMILIES)
    assert set(prepared["cognition_family_by_source"].values()) == set(TRAINING_COGNITION_FAMILIES)
    assert all(
        family_by_source[source] == (TRAINING_FAMILY, family)
        for source, family in prepared["cognition_family_by_source"].items()
    )
    packed_cognition_cells = {
        cell.split("|", 1)[1]
        for cell in prepared["data"]["pack_audit"]["segregated_cells"]
        if cell.startswith(f"{TRAINING_FAMILY}|")
    }
    assert packed_cognition_cells == set(TRAINING_COGNITION_FAMILIES)


def test_matched_runs_hold_data_constant_while_sampler_seed_changes_order():
    first = prepare_phase1_data(
        corpus_seed=31,
        sampler_seed=101,
        run_id="signac-matched-run-a",
        tokenizer=_Tokenizer(),
        count=24,
    )
    matched = prepare_phase1_data(
        corpus_seed=31,
        sampler_seed=101,
        run_id="signac-matched-run-b",
        tokenizer=_Tokenizer(),
        count=24,
    )
    resampled = prepare_phase1_data(
        corpus_seed=31,
        sampler_seed=102,
        run_id="signac-resampled-run",
        tokenizer=_Tokenizer(),
        count=24,
    )

    assert first["training_surface_sha256"] == matched["training_surface_sha256"]
    assert first["data"]["manifest_sha256"] == matched["data"]["manifest_sha256"]
    assert first["data"]["pack_manifest_sha256"] == matched["data"]["pack_manifest_sha256"]
    assert first["data"]["sampler_order"] == matched["data"]["sampler_order"]
    assert first["data"]["manifest_sha256"] == resampled["data"]["manifest_sha256"]
    assert first["data"]["pack_manifest_sha256"] == resampled["data"]["pack_manifest_sha256"]
    assert first["data"]["sampler_order"] != resampled["data"]["sampler_order"]
    assert first["corpus_seed"] == resampled["corpus_seed"] == 31
    assert first["sampler_seed"] == 101
    assert resampled["sampler_seed"] == 102


def test_research_corpus_retains_snapshot_and_claim_boundaries():
    corpus = build_research_evidence_corpus()
    assert corpus["research_records"] == 81
    assert corpus["curated_index_records"] == 11
    assert len(corpus["ledger_sha256"]) == 64
    assert len(corpus["curated_index_sha256"]) == 64
    assert "not a current cross-branch checkout" in corpus["claim_ceiling"].lower()
    s5 = next(row for row in corpus["documents"] if row["record_id"] == "formation-mux-001-s5")
    assert "INCONCLUSIVE_AT_ZERO_BASELINE" in s5["text"]
    assert "not a general law" in s5["text"]


def test_training_surface_manifest_records_crossed_cognition_axes():
    surface = build_training_surface(seed=2409, count=270, include_research_evidence=False)
    axes = surface["cognition_surface_axis_counts"]
    interference = axes["interference_retrieval"]
    expected_grid = {"0:1"} | {f"2:{quartile}" for quartile in (1, 2, 3)} | {
        f"{dose}:{quartile}"
        for dose in (4, 8, 16, 32)
        for quartile in (1, 2, 3, 4)
    }
    observed_grid = set(interference["dose_position_cell"])
    assert observed_grid == expected_grid
    assert set(axes["semantic_state"]["state_query"]) == {
        "latest", "intermediate", "rollback", "precedence"
    }


def test_training_generator_rejects_nonfinite_and_boolean_mixture_weights():
    fractions = {family: 1 / len(TRAINING_COGNITION_FAMILIES)
                 for family in TRAINING_COGNITION_FAMILIES}
    fractions[TRAINING_COGNITION_FAMILIES[0]] = math.nan
    with pytest.raises(ValueError, match="finite numeric weights"):
        build_training_examples(seed=1, count=18, family_fractions=fractions)

    fractions = {family: 0.0 for family in TRAINING_COGNITION_FAMILIES}
    fractions[TRAINING_COGNITION_FAMILIES[0]] = True
    with pytest.raises(ValueError, match="not booleans"):
        build_training_examples(seed=1, count=18, family_fractions=fractions)
