"""Contract tests: grouped query-swap clean replication (P20).

Each test targets a specific recorded failure of tp-grouped-queryswap-001:
row-level split leakage, dishonest group-size semantics, candidate-level
pseudo-replication, fixture drift, vocabulary collision, trajectory on
micro-steps instead of optimizer updates, and evidence-status laundering.
"""

from __future__ import annotations

import itertools
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


# ---------------------------------------------------------------- helpers

def _jsonl(name):
    return [json.loads(l) for l in
            (ROOT / name).read_text(encoding="utf-8").splitlines() if l.strip()]


def _load_generator_module():
    from connector.experiments import grouped_queryswap as g
    return g


# ----------------------------------------------- 1. group-atomic split (P0)

def test_queryswap_group_atomic_split_zero_overlap() -> None:
    train = _jsonl("data/grouped_queryswap/train.jsonl")
    held = _jsonl("data/grouped_queryswap/heldout.jsonl")
    tg = {x["group_id"] for x in train if x["family"] == "queryswap_group"}
    hg = {x["group_id"] for x in held}
    assert tg and hg, "both sides must contain target groups"
    assert not (tg & hg), f"group overlap: {sorted(tg & hg)}"


def test_committed_split_audit_proves_disjointness_and_hashes() -> None:
    audit = json.loads(
        (ROOT / "data/grouped_queryswap/split_audit.json").read_text(encoding="utf-8"))
    assert audit["group_overlap"] == 0
    assert audit["prompt_overlap"] == 0
    assert audit["full_fact_block_overlap"] == 0
    train = _jsonl("data/grouped_queryswap/train.jsonl")
    held = _jsonl("data/grouped_queryswap/heldout.jsonl")

    def sha(rows):
        blob = "\n".join(json.dumps(x, sort_keys=True) for x in rows)
        import hashlib
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    assert audit["train_data_sha256"] == sha(train)
    assert audit["heldout_data_sha256"] == sha(held)
    expected_train_groups = {
        x["group_id"] for x in train if x["family"] == "queryswap_group"}
    assert set(audit["group_split_manifest"]["train_group_ids"]) == \
        expected_train_groups


# ------------------------- 2-5. group semantics: members, block, k (P0.1)

def test_every_group_has_all_expected_members_and_constant_block() -> None:
    train = _jsonl("data/grouped_queryswap/train.jsonl")
    by_group: dict[str, list[dict]] = {}
    for x in train:
        if x["family"] == "queryswap_group":
            by_group.setdefault(x["group_id"], []).append(x)
    for gid, rows in by_group.items():
        ks = {r["group_size"] for r in rows}
        assert len(rows) == list(ks)[0], f"{gid}: member count != group_size"
        # identical fact blocks across members; only query/target differ
        blocks = {r["fact_block_sha256"] for r in rows}
        assert len(blocks) == 1, f"{gid}: fact block varies within group"
        golds = [r["gold"] for r in rows]
        assert len(set(golds)) == len(golds), "duplicate target within group"
        prompts = [r["prompt"] for r in rows]
        assert len(set(prompts)) == len(prompts), "duplicate prompt in group"


def test_group_size_metadata_matches_actual_k_and_is_deliberate() -> None:
    train = _jsonl("data/grouped_queryswap/train.jsonl")
    seen_sizes = {x["group_size"] for x in train
                  if x["family"] == "queryswap_group"}
    assert seen_sizes == {2, 3, 4}, "k must be exactly the deliberate 2..4 set"
    for x in train:
        if x["family"] != "queryswap_group":
            continue
        n_queries = x["prompt"].count("\nReturn ONLY the ref of ")
        assert n_queries == 1  # one query per row...
    by_group: dict[str, int] = {}
    for x in train:
        if x["family"] == "queryswap_group":
            by_group[x["group_id"]] = by_group.get(x["group_id"], 0) + 1
    assert set(by_group.values()) == {2, 3, 4}


# --------------------------------- 6. proposal claims match reality (P0.1)

def test_proposal_language_matches_generator_reality() -> None:
    proposal = json.loads((ROOT / "data/proposal_grouped_queryswap_replication.json")
                          .read_text(encoding="utf-8"))
    unit_text = json.dumps(proposal).lower()
    assert '"three' not in unit_text and "three-query" not in unit_text
    assert "k in {2,3,4}" in unit_text or "k in {2, 3, 4}" in json.dumps(proposal)
    gen_src = (ROOT / "connector/experiments/grouped_queryswap.py").read_text(
        encoding="utf-8")
    assert "k = 2 + (gi % 3)" in gen_src  # generator really produces 2..4
    old = json.loads((ROOT / "data/proposal_grouped_queryswap.json")
                     .read_text(encoding="utf-8"))
    # original preregistration stays untouched/immutable
    assert old["proposal_id"] == "tp-grouped-queryswap-001"


# -------------------- 7-8. group-level paired stats, no pseudo-replication

def test_sign_flip_p_exact_math_on_independent_units() -> None:
    from connector.experiments.query_influence_v3 import sign_flip_p
    vals = [0.5, -0.2, 0.9, 0.3, -0.1, 0.7]
    obs = sum(vals)
    exact = sum(1 for signs in itertools.product((1, -1), repeat=len(vals))
                if sum(s * v for s, v in zip(signs, vals)) >= obs) / 64
    assert sign_flip_p(vals) == pytest.approx(exact)
    assert sign_flip_p([0.0] * 6) == 1.0
    assert sign_flip_p([5.0] * 6) <= 1 / 32 * 2 + 1e-9


def test_primary_p_value_consumes_group_level_inputs_only() -> None:
    src = (ROOT / "connector/experiments/query_influence_v3.py").read_text(
        encoding="utf-8")
    assert "sign_flip_p(group_means)" in src            # single-model primary
    assert "sign_flip_p(deltas)" in src                 # paired deltas primary
    # candidate lifts are NEVER passed to the permutation test directly
    assert "sign_flip_p(cand_lifts)" not in src
    assert "DIAGNOSTIC_ONLY" in src or "diagnostic_only" in src


# ------------------------------ 9-10. QIM-v3 fixture stability + disjoint

def test_qim_v3_fixture_hash_stable() -> None:
    from connector.experiments.query_influence_v3 import fixture_hash
    assert fixture_hash() == fixture_hash()
    assert len(fixture_hash()) == 64


def test_qim_v3_vocab_disjoint_from_training_v2_and_bank() -> None:
    from connector.experiments.query_influence_v3 import vocabulary_disjointness
    res = vocabulary_disjointness()
    assert res["disjoint"], f"vocab collision: {res['overlaps']}"


# --------------------- 11. trajectory triggers on optimizer updates (P6)

def test_trajectory_grid_is_optimizer_updates_not_microsteps() -> None:
    src = (ROOT / "training/sft_grouped_queryswap.py").read_text(encoding="utf-8")
    assert "TRAJECTORY_UPDATES = [5, 10, 20, 30, 40, 50]" in src
    assert "update += 1" in src and "opt.step()" in src
    assert '"optimizer_update": update_idx' in src
    assert "micro_steps" not in src.split("trajectory")[1].split("def record")[0]


def test_preregistered_trajectory_was_honestly_recorded_as_not_executed() -> None:
    lineage = json.loads((ROOT / "output/lineage_sft5_queryswap.json")
                         .read_text(encoding="utf-8"))
    traj = lineage["why_not_clean_P9"]["trajectory"]
    assert "NOT EXECUTED" in traj


# --------------------------- 12-13. extraction floor + family floors (P7/P8)

def test_extraction_floor_parent_relative_tolerance() -> None:
    from connector.experiments.context_value_extraction import (
        EXTRACTION_TOLERANCE, extraction_floor_ok)
    assert extraction_floor_ok(0.75, 0.75 - EXTRACTION_TOLERANCE + 1e-9)
    assert not extraction_floor_ok(0.75, 0.75 - EXTRACTION_TOLERANCE - 0.01)


def test_protected_family_floors_are_parent_relative_per_family() -> None:
    src = (ROOT / "training/sft_grouped_queryswap.py").read_text(encoding="utf-8")
    assert "PARENT_REGRESSION_TOLERANCE" in src
    for fam in ("single_fact", "tool_result", "copy", "protocol_transfer"):
        assert fam in src
    assert "symbolic_ops" in src and "PROTECTED_FAMS" in src
    # symbolic is monitored, not gated
    protected = re.search(r"PROTECTED_FAMS = \(([^)]*)\)", src).group(1)
    assert "symbolic_ops" not in protected


# ----------------------- 14. lineage resolves every preregistered prediction

def test_lineage_resolves_every_preregistered_prediction() -> None:
    proposal = json.loads((ROOT / "data/proposal_grouped_queryswap_replication.json")
                          .read_text(encoding="utf-8"))
    registered = set(proposal["pre_registered_predictions"])
    receipt_path = ROOT / "output/replication_receipt_final.json"
    if not receipt_path.exists():
        # Lifecycle guard: before the replication has been executed, nothing
        # may claim completion. Once output/replication_receipt_final.json
        # appears, the strict branch below engages PERMANENTLY.
        status = json.loads((ROOT / "output/EVIDENCE_MANIFEST.json")
                            .read_text(encoding="utf-8"))["clean_replication"]["status"]
        assert status != "COMPLETE", \
            "clean_replication marked COMPLETE without a final receipt"
        return
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    resolved = receipt["preregistered_predictions"]
    assert set(resolved) == registered
    for v in resolved.values():
        assert isinstance(v, str) and \
            v.startswith(("PASS", "FAIL", "NOT_MEASURED")), f"bad verdict: {v}"
    assert receipt["proposal_sha256_at_execution"]


# --------------- 15. contaminated SFT5 is never labeled clean replication

def test_contaminated_sft5_never_labeled_clean() -> None:
    lineage = json.loads((ROOT / "output/lineage_sft5_queryswap.json")
                         .read_text(encoding="utf-8"))
    blob = json.dumps(lineage)
    assert "REPLICATION_REQUIRED" in blob or "DEVELOPMENT_ONLY" in blob
    # every "clean replication" mention must be a prohibition ("never cited",
    # "requires") or point at the -002 successor — never an sft5 self-claim
    for m in re.finditer(r"clean replication", blob.lower()):
        ctx = blob[max(0, m.start() - 90): m.end()]
        ok = ("never" in ctx or "requires" in ctx
              or "replication-002" in ctx or "REPLICATION_REQUIRED".lower() in ctx)
        assert ok, f"sft5 lineage appears to self-claim cleanliness near: {ctx!r}"
    manifest = json.loads((ROOT / "output/EVIDENCE_MANIFEST.json")
                          .read_text(encoding="utf-8"))
    gq = manifest["grouped_queryswap_result"]
    assert "NOT" in gq["verdict"] or "DEVELOPMENT_ONLY" in gq["status"]
    assert gq["status"].startswith("DEVELOPMENT_ONLY")


def test_verified_intervention_count_not_inflated_by_training() -> None:
    manifest = json.loads((ROOT / "output/EVIDENCE_MANIFEST.json")
                          .read_text(encoding="utf-8"))
    tax = manifest["evidence_taxonomy"]
    assert tax["VerifiedInterventionExperience"] == 0
