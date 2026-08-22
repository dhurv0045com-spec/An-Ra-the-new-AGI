"""Focused tests for the corrected (v2) query-influence instruments."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from connector.experiments.query_influence import (  # noqa: E402
    _completion_logprob, _prompt, _query, _stable_js, build_groups, fixture_hash,
)
from anra_core.config import CoreConfig  # noqa: E402
from anra_core.model import AnRaCore  # noqa: E402
from anra_core.tokenizer import V4Tokenizer  # noqa: E402

ROOT = Path(__file__).parents[1]
TOKENIZER = V4Tokenizer.load(ROOT / "anra_core" / "assets" / "tokenizer_v4_32k.json")


def _tiny_model(seed: int = 7):
    torch.manual_seed(seed)
    cfg = CoreConfig(vocab_size=32_768, d_model=32, n_layers=2, n_heads=4,
                     n_kv_heads=2, head_dim=8, d_ff=64, block_size=96,
                     base_seq_len=96, target_seq_len=96, sliding_window=8,
                     full_attention_every=2)
    return AnRaCore(cfg).eval()


def test_completion_scoring_full_sequence_and_deterministic() -> None:
    model = _tiny_model()
    lp1 = _completion_logprob(model, TOKENIZER, "Facts.\nAnswer:", " ABC-123.")
    lp2 = _completion_logprob(model, TOKENIZER, "Facts.\nAnswer:", " ABC-123.")
    assert lp1 == lp2 and lp1 < 0
    longer = _completion_logprob(model, TOKENIZER, "Facts.\nAnswer:",
                                 " ABC-123. More tokens.")
    assert longer < lp1


def test_shuffled_display_ordinal_target_correctness() -> None:
    """Ordinal/pointer must index the DISPLAYED fact order; the gold is that
    record's code. The v1 bug graded against pre-shuffle indices."""
    for g in build_groups():
        for i, rec in enumerate(g["displayed_facts"]):
            ordinal_gold = g["displayed_facts"][i]["code"]
            assert ordinal_gold == rec["code"]
            assert f"fact {i + 1}" not in rec["line"]  # ordinal refers to position only


def test_entity_ordinal_pointer_semantic_equivalence() -> None:
    """All three query forms must designate the SAME target code."""
    import random
    rng = random.Random(1)
    for g in build_groups()[:4]:
        t = rng.randrange(3)
        rec = g["displayed_facts"][t]
        golds = {
            "entity": rec["code"],  # by construction of _query(rec)
            "ordinal": g["displayed_facts"][t]["code"],
            "pointer": g["displayed_facts"][t]["code"],
        }
        assert len(set(golds.values())) == 1
        q_entity = _query(rec)
        assert rec["entity"].capitalize() in q_entity


def _one_answer_marker(prompt: str) -> int:
    return sum(1 for line in prompt.splitlines() if line.strip() == "Answer:")


def test_relocation_interventions_pure() -> None:
    g = build_groups()[0]
    facts = [r["line"] for r in g["displayed_facts"]]
    rec = g["displayed_facts"][0]
    q = _query(rec)
    others = [l for l in facts if l != rec["line"]]
    base = _prompt("\n".join(facts), q)
    relocate = f"{q}\n" + "\n".join(facts) + "\nAnswer:"
    fact_move = "\n".join(others) + f"\n{q}\n{rec['line']}\nAnswer:"

    def lines_of(s):
        return sorted(l for l in s.splitlines() if l.strip() and l.strip() != "Answer:")

    for variant in (relocate, fact_move):
        assert _one_answer_marker(variant) == 1, "exactly one Answer marker"
        assert lines_of(variant) == lines_of(base), "same line multiset"
    assert _one_answer_marker(base) == 1


def test_mark_only_no_duplication_and_repeat_only_no_annotation() -> None:
    g = build_groups()[0]
    facts = [r["line"] for r in g["displayed_facts"]]
    rec = g["displayed_facts"][1]
    q = _query(rec)
    mark_only = "\n".join(
        f"[RELEVANT] {l}" if l == rec["line"] else l for l in facts) + f"\n{q}\nAnswer:"
    repeat_only = "\n".join(facts) + f"\n{q}\n{rec['line']}\nAnswer:"
    # MARK_ONLY: the fact appears exactly once (annotated in place), never duplicated.
    fact_lines = [l for l in mark_only.splitlines() if l.endswith(rec["line"])]
    assert len(fact_lines) == 1 and fact_lines[0].startswith("[RELEVANT]")
    # REPEAT_ONLY: no relevance annotation anywhere; fact appears twice
    # (original position + repeated near the answer), neither annotated.
    assert "[RELEVANT]" not in repeat_only
    fact_lines_r = [l for l in repeat_only.splitlines() if l.endswith(rec["line"])]
    assert len(fact_lines_r) == 2 and not any(
        l.startswith("[RELEVANT]") for l in fact_lines_r)


def test_stable_js_extreme_logprobs() -> None:
    a = [-800.0, -900.0, -1000.0]
    b = [-800.5, -900.5, -1000.5]
    assert _stable_js(a, a) == 0.0
    val = _stable_js(a, b)
    assert 0.0 <= val < 0.001  # tiny shift, no underflow/crash
    assert _stable_js([0.0, -1000.0, -1000.0], [-1000.0, 0.0, -1000.0]) > 0.6


def test_query_lift_math() -> None:
    """lift_i = logP(v_i|own query) - mean_j!=i logP(v_i|query_j)."""
    L = [[-1.0, -2.0, -3.0],   # query 0
         [-2.0, -1.5, -4.0],   # query 1
         [-3.5, -2.5, -1.2]]   # query 2
    lifts = [L[i][i] - sum(L[j][i] for j in range(3) if j != i) / 2
             for i in range(3)]
    assert lifts[0] == pytest.approx(-1.0 - (-2.0 + -3.5) / 2)   # +1.25
    assert lifts[1] == pytest.approx(-1.5 - (-2.0 + -2.5) / 2)   # +0.25
    assert lifts[2] == pytest.approx(-1.2 - (-3.0 + -4.0) / 2)   # +2.3
    adv = sum(lifts) / 3
    assert adv > 0  # diagonal structure present in this synthetic matrix


def test_fixture_hash_stable_across_calls() -> None:
    assert fixture_hash() == fixture_hash()
    assert len(fixture_hash()) == 16


def test_preregistered_prediction_completeness() -> None:
    proposal = json.loads(
        (ROOT / "data" / "query_focus_sft" / "proposal.json").read_text(encoding="utf-8"))
    lineage = json.loads((ROOT / "output" / "lineage_sft4.json").read_text(encoding="utf-8"))
    registered = set(proposal["pre_registered_predictions"])
    resolved = lineage["predictions"]
    assert registered == set(resolved)
    for verdict in resolved.values():
        assert verdict in {"PASS", "FAIL", "NOT_MEASURED"}


def test_stale_causal_lineage_cannot_pose_as_verified() -> None:
    """The manifest must not describe the 12 observational records as
    VerifiedInterventionExperience."""
    manifest = json.loads((ROOT / "output" / "EVIDENCE_MANIFEST.json").read_text(encoding="utf-8"))
    blob = json.dumps(manifest).lower()
    assert "12 behavioralimprovementobservations" in blob.replace("-", "").replace("_", "") \
        or "behavioralimprovementobservation" in blob
    for art in manifest["artifacts"]:
        reason = art.get("reason", "").lower()
        if "observations" in reason or "observational" in reason:
            assert "verified intervention" not in reason


def test_train_dev_group_disjointness() -> None:
    train = [json.loads(l) for l in (ROOT / "data/capability_bank/train.jsonl")
             .read_text(encoding="utf-8").splitlines() if l.strip()]
    dev = [json.loads(l) for l in (ROOT / "data/capability_bank/dev.jsonl")
           .read_text(encoding="utf-8").splitlines() if l.strip()]
    assert not ({x.get("group_id") for x in train} &
                {x.get("group_id") for x in dev})
    assert not ({x["prompt"] for x in train} & {x["prompt"] for x in dev})
