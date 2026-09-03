"""CPU-only CI: readiness math, firewall-of-fallback, schemas, registry."""

import sys
from pathlib import Path

_XF = Path(__file__).resolve().parents[1]
if str(_XF) not in sys.path:
    sys.path.insert(0, str(_XF))

import json

import pytest

from checkpoint_identity import CheckpointNotFound, resolve_checkpoint
from readiness.identifiability import check_identifiability, required_n_mcnemar
from readiness.ladder import RUNGS, gen_tasks, oracle_prompt
from readiness.schemas import (
    CausalResponseProfile,
    PredictionBeforeInterventionRecord,
    commit_prediction,
)


def test_no_silent_fallback():
    with pytest.raises(CheckpointNotFound) as e:
        resolve_checkpoint("checkpoints/does-not-exist-xyz.pt")
    assert "REQUESTED CHECKPOINT NOT FOUND" in str(e.value)


def test_ladder_determinism_and_coverage():
    assert RUNGS == ("B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7")
    a = gen_tasks("B3", 7, 5)
    b = gen_tasks("B3", 7, 5)
    assert [t["prompt"] for t in a] == [t["prompt"] for t in b]
    assert len(gen_tasks("B0", 7, 4)[0]["codes"]) == 1
    assert "Recall:" in oracle_prompt(a[0])


def test_floor_detection_old_ood_shape():
    # structural-OOD shape: raw 0/240, oracle ~24% -> FLOOR/ORACLE limited, DO_NOT_RUN
    r = check_identifiability(240, 0, 240, 58, 58, chance=0.25)
    assert r["decision"] == "DO_NOT_RUN"
    assert r["substrate_adequacy"] in ("FLOOR_LIMITED", "ORACLE_LIMITED")


def test_adequate_shape_runs():
    # EV-DEV shape: raw 13/120, oracle repairs 50/107, discord 50 -> RUN/ADEQUATE
    r = check_identifiability(120, 13, 107, 50, 50, chance=0.25)
    assert r["decision"] == "RUN"
    assert r["substrate_adequacy"] == "ADEQUATE"


def test_ceiling_shape_stops():
    r = check_identifiability(100, 97, 3, 2, 2)
    assert r["decision"] == "DO_NOT_RUN"
    assert r["substrate_adequacy"] == "CEILING_LIMITED"


def test_power_estimate_sane():
    n = required_n_mcnemar(0.30, 0.05)
    assert 10 <= n <= 500
    assert required_n_mcnemar(0.05, 0.30) == -1


def test_prediction_commitment_deterministic():
    rec = PredictionBeforeInterventionRecord(
        checkpoint_sha="abc", task_id="t1", observation_hash="o1",
        candidate_interventions=("A", "B"),
        predicted_response_distribution=(0.7, 0.2),
        predicted_best_intervention="A", uncertainty=0.3, sequence_id=4)
    c1 = commit_prediction(rec)
    c2 = commit_prediction(rec)
    assert c1["commitment_hash"] == c2["commitment_hash"] and len(c1["commitment_hash"]) == 64


def test_prediction_rejects_bad_best():
    with pytest.raises(ValueError):
        PredictionBeforeInterventionRecord(
            checkpoint_sha="a", task_id="t", observation_hash="o",
            candidate_interventions=("A",),
            predicted_response_distribution=(0.5,),
            predicted_best_intervention="ZZZ", uncertainty=0.1)


def test_response_profile_no_labels():
    p = CausalResponseProfile(task_id="t", raw_result=0)
    p.add("E5dup", observed_result=1, behavioral_delta=1, cost=1, predicted=1)
    assert p.legal_interventions["E5dup"]["observed_result"] == 1


def test_registry_schema():
    reg = json.load(open(Path(_XF) / "registry" / "checkpoints.json"))
    assert reg["schema"] == "anra-checkpoint-registry/v2"
    assert "UNQUALIFIED_NEW" in reg["roles"]
    for c in reg["checkpoints"]:
        for k in ("path", "global_step", "parameter_sha256", "role", "status",
                  "research_subject", "readiness"):
            assert k in c, k
        assert c["research_subject"] is False  # old weak ckpts never subjects
