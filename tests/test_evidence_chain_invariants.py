"""Debug sweep 6: end-to-end invariant checks over the whole evidence chain.

Runs as a pytest file so CI keeps guarding these invariants forever.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]

P_SHA = "c3bc615eb3ffc8628f82088c433507baa142a0fecf91e4f6e64f9b17729e0625"
C_SHA = "36c87e5be671ff37951b4b433e21ec14bb5e59f6f0770d7f81e50e99ead9e001"
FIX = "27713accb3aa6825de23cf09540497943a0e56ca3ac177ba94f0f164f740a614"


def _j(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


def test_frozen_shas_consistent_across_all_artifacts() -> None:
    r = _j("output/replication_receipt_final.json")
    par = _j("output/qim3_parent_corrective_rescore.json")
    chi = _j("output/qim3_sft6_corrective_rescore.json")
    assert r["parent"]["parameter_sha256"] == par["parameter_sha256"] == P_SHA
    assert r["child"]["parameter_sha256"] == chi["parameter_sha256"] == C_SHA


def test_fixture_identical_in_every_qim_v3_report() -> None:
    for name in ("qim3_parent_baseline.json", "qim3_sft6_replication.json",
                 "qim3_parent_corrective_rescore.json",
                 "qim3_sft6_corrective_rescore.json"):
        rep = _j(f"output/{name}")
        assert rep["fixture_sha256"] == FIX, name


def test_paired_delta_recomputes_from_per_group_lifts() -> None:
    par = _j("output/qim3_parent_corrective_rescore.json")
    chi = _j("output/qim3_sft6_corrective_rescore.json")
    deltas = [c - p for c, p in zip(chi["per_group_query_lift"],
                                    par["per_group_query_lift"])]
    mean_d = sum(deltas) / len(deltas)
    pos = sum(1 for d in deltas if d > 0)
    rec = chi["paired_vs_parent"]
    assert abs(mean_d - rec["mean_paired_delta"]) < 5e-4
    assert f"{pos}/{len(deltas)}" == rec["positive_delta_groups"] == "35/40"


def test_training_receipt_hashes_match_files_on_disk() -> None:
    """The training-time receipt hashed LOCAL WORKING-COPY bytes. On this
    repo's Windows checkouts (core.autocrlf=true) those are CRLF while the
    committed blob is LF, so CI must compare against the CRLF-normalized
    blob — content-identical to disk on any autocrlf checkout."""
    import subprocess
    rr = _j("output/replication_receipt.json")
    for side, key in (("train", "train_sha256"), ("heldout", "heldout_sha256")):
        blob = subprocess.check_output(
            ["git", "show", f"HEAD:data/grouped_queryswap/{side}.jsonl"])
        h_crlf = hashlib.sha256(blob.replace(b"\n", b"\r\n")).hexdigest()
        h_lf = hashlib.sha256(blob).hexdigest()
        assert rr[key] in (h_crlf, h_lf), \
            f"{side}: receipt hash matches neither CRLF-normalized nor raw blob"


def test_consumed_group_structure_intact() -> None:
    from collections import defaultdict
    rows = [json.loads(l) for l in
            (ROOT / "data/grouped_queryswap/train.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    g = defaultdict(list)
    for r in rows:
        if r["family"] == "queryswap_group":
            g[r["group_id"]].append(r)
    assert len(g) == 49
    for gid, members in g.items():
        sizes = {m["group_size"] for m in members}
        assert len(members) in sizes, gid
        assert len({m["fact_block_sha256"] for m in members}) == 1, gid
    held = {json.loads(l)["group_id"] for l in
            (ROOT / "data/grouped_queryswap/heldout.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip() and json.loads(l).get("family") == "queryswap_group"}
    assert not (set(g) & held)


def test_decomposition_internal_arithmetic_consistent() -> None:
    gd = _j("output/greedy_decomposition.json")
    rows = _j("output/greedy_decomposition_rows.json")
    free = sum(1 for x in rows if x["strict"])
    c1 = sum(1 for x in rows if x["rank"] == 1)
    gap = sum(1 for x in rows if x["rank"] == 1 and not x["strict"])
    fails = [x for x in rows if not x["strict"]]
    tax = gd["failure_taxonomy_n75"]
    assert free == 44 and gd["free_greedy_accuracy"].startswith("44/")
    assert c1 == 63 and gap == 19
    assert tax["selection_failures_gold_not_rank1"] + \
        tax["realization_failures_gold_ranked1_but_wrong_emit"] == len(fails) == 75


def test_sign_flip_p_rejects_empty_input() -> None:
    from connector.experiments.query_influence_v3 import sign_flip_p
    with pytest.raises(ValueError):
        sign_flip_p([])


def test_trainer_preserves_all_repair_invariants() -> None:
    src = (ROOT / "training/sft_grouped_queryswap.py").read_text(encoding="utf-8")
    # group unit averages all members
    assert "torch.stack(losses).mean()" in src
    # alpha at unit selection
    assert "mix_rng.random() < args.alpha" in src
    # eligibility conjunction intact
    assert "all(floors.values()) and ext_ok" in src
    # split audit enforced pre-training
    assert 'group_overlap"] == 0' in src
    # honest trajectory metric names only
    assert "qim_v2_mean_group_lift" not in src
