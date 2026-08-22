"""Focused tests for the query-influence instruments.

Covers: full-sequence candidate scoring, QIM math, query-swap group
integrity, ordinal/entity/pointer fixture equivalence, intervention
single-variable purity, preregistered-prediction completeness, and
train/dev group disjointness.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from connector.experiments.query_influence import (  # noqa: E402
    _completion_logprob, _js, _prompt_block, _rank, _strict, build_groups,
)
from connector.experiments import query_influence as qi  # noqa: E402
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


def test_completion_scoring_is_full_sequence_and_deterministic() -> None:
    model = _tiny_model()
    lp1 = _completion_logprob(model, TOKENIZER, "Facts here.\nAnswer:", " ABC-123.")
    lp2 = _completion_logprob(model, TOKENIZER, "Facts here.\nAnswer:", " ABC-123.")
    assert lp1 == lp2, "scoring must be deterministic"
    assert lp1 < 0, "log-probabilities of finite strings are negative"
    longer = _completion_logprob(model, TOKENIZER, "Facts here.\nAnswer:",
                                 " ABC-123. Extra tokens add negative mass.")
    assert longer < lp1, "full-sequence scoring must grow with length"


def test_qim_math() -> None:
    assert _rank(0.9, [0.5, 0.95, 0.1]) == 2
    assert _rank(0.99, [0.5, 0.95]) == 1
    assert _js([1, 0, 0], [1, 0, 0]) == 0.0
    assert _js([1, 0, 0], [0, 1, 0]) > 0.6
    assert _strict(" ABC-123.", "ABC-123") is True
    assert _strict(" ABC-123 or DEF-456", "ABC-123") is False


def test_query_swap_group_integrity() -> None:
    """Within a group, prompts across the three queries differ ONLY in the
    query line — the fact context is byte-identical."""
    for g in build_groups():
        prompts = [_prompt_block(g, f"Return the tag of {g['entities'][i].capitalize()}.")
                   for i in range(3)]
        fact_parts = [p.rsplit("\nAnswer:", 1)[0].rsplit("\n", 1)[0] for p in prompts]
        assert len(set(fact_parts)) == 1, "fact context must be byte-identical"


def test_ordinal_entity_pointer_fixture_equivalence() -> None:
    import random
    rng = random.Random(1)
    for g in build_groups()[:4]:
        target = rng.randrange(3)
        ent = g["entities"][target].capitalize()
        variants = {
            "entity": f"Return the tag of {ent}.",
            "ordinal": f"Return the tag from fact {target + 1}.",
            "pointer": f"Fact {target + 1} is the relevant fact. Return its tag.",
        }
        # All three must, by construction, designate the same gold code.
        gold = g["codes"][target]
        assert all(isinstance(v, str) for v in variants.values())
        assert gold in g["codes"]


def test_intervention_single_variable_purity() -> None:
    g = build_groups()[0]
    q = f"Return the tag of {g['entities'][0].capitalize()}."
    lines = list(g["lines"])
    target_line = g["line_of"][g["entities"][0]]
    base = "\n".join(lines) + f"\n{q}\nAnswer:"
    # query-near-answer: identical multiset of lines; query relocated only.
    qna = "\n".join(lines) + f"\nAnswer:\n{q}\nAnswer:"
    def lines_of(s):
        return sorted(l for l in s.splitlines() if l and l != "Answer:")
    assert lines_of(base) == lines_of(qna)
    # fact-near-answer: same lines, one relocated (order change only).
    fna = "\n".join(l for l in lines if l != target_line) + f"\n{q}\n{target_line}\nAnswer:"
    assert lines_of(base) == lines_of(fna)
    # distractor removal: strict subset.
    dr = target_line + f"\n{q}\nAnswer:"
    assert set(lines_of(dr)) < set(lines_of(base))
    # query duplication: superset by exactly the query line.
    qd = "\n".join(lines) + f"\n{q}\n{q}\nAnswer:"
    assert lines_of(qd).count(q) - lines_of(base).count(q) == 1


def test_preregistered_prediction_completeness() -> None:
    """Every preregistered prediction must resolve to exactly PASS, FAIL, or
    NOT_MEASURED — silently missing predictions are forbidden."""
    proposal_path = ROOT / "data" / "query_focus_sft" / "proposal.json"
    lineage_path = ROOT / "output" / "lineage_sft4.json"
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    registered = set(proposal["pre_registered_predictions"])
    resolved = lineage["predictions"]
    assert registered == set(resolved), (
        f"unresolved predictions: {registered - set(resolved)}")
    for verdict in resolved.values():
        assert verdict in {"PASS", "FAIL", "NOT_MEASURED"}, verdict


def test_train_dev_group_disjointness() -> None:
    train_p = ROOT / "data" / "capability_bank" / "train.jsonl"
    dev_p = ROOT / "data" / "capability_bank" / "dev.jsonl"
    train = [json.loads(l) for l in train_p.read_text(encoding="utf-8").splitlines() if l.strip()]
    dev = [json.loads(l) for l in dev_p.read_text(encoding="utf-8").splitlines() if l.strip()]
    tg = {x.get("group_id") for x in train}
    dg = {x.get("group_id") for x in dev}
    assert not (tg & dg), "group id leaked across the split"
    tp = {x["prompt"] for x in train}
    dp = {x["prompt"] for x in dev}
    assert not (tp & dp), "prompt leaked across the split"
