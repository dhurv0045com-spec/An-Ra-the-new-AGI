"""Citadel T1C unit tests. Zero third-party dependencies.

Run:  python tests/test_citadel_t1c.py   (exit 0 = all pass)
Covers the T1C contract without torch/TPU: indexed-determinism, template
rendering + arithmetic parsing, prompt/target splitting incl. arrow rows,
answer-span positions, MID spec structural rules, budget arithmetic, eval-slice
leakage zeros, and every cross-arm classification rule on synthetic receipts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
CITADEL_ROOT = HERE.parents[1]
sys.path.insert(0, str(CITADEL_ROOT))

from citadel_tpu import arith_data as ad  # noqa: E402
from citadel_tpu import calculator_eval as cev  # noqa: E402
from citadel_tpu import t1c_run as t1c  # noqa: E402


def test_indexed_determinism_and_bounds() -> None:
    n = ad.SPLITS["train"]["n"]
    assert ad.row_at("train", 0) == ad.row_at("train", 0)
    assert ad.row_at("train", n - 1) != ad.row_at("train", 0)
    for bad in (-1, n):
        try:
            ad.row_at("train", bad)
            raise SystemExit(f"no bounds error for {bad}")
        except ValueError:
            pass


def test_train_family_ranges_and_uniqueness() -> None:
    import random as _random

    rng = _random.Random(99)
    rows = [ad.row_at("train", rng.randrange(ad.SPLITS["train"]["n"]))[0] for _ in range(10_000)]
    assert max(len(r) for r in rows) <= 32
    assert len(set(rows)) / len(rows) > 0.98  # no collapsed family
    for r in rows[:2_000]:
        a, op, b, _ = ad.parse_arith(r)
        if op == "*":
            assert 0 <= a <= 999 and 0 <= b <= 999
        if op == "/":
            assert 1 <= b <= 999


def test_templates_render_and_parse() -> None:
    assert ad.render(12, "+", 9, 21, "canon") == "12 + 9 = 21"
    assert ad.render(12, "+", 9, 21, "compact") == "12+9=21"
    assert ad.render(12, "+", 9, 21, "arrow") == "12 + 9 -> 21"
    assert ad.render(12, "+", 9, 21, "words") == "add 12 and 9 = 21"
    assert ad.render(18, "-", 7, 11, "words") == "subtract 7 from 18 = 11"
    assert ad.parse_arith("add 12 and 9 = 21") == (12, "+", 9, 21)
    assert ad.parse_arith("subtract 7 from 18 = 11") == (18, "-", 7, 11)
    assert ad.parse_arith("multiply 6 by 8 = 48") == (6, "*", 8, 48)
    assert ad.parse_arith("divide 72 by 8 = 9") == (72, "/", 8, 9)
    try:
        ad.parse_arith("12 + 9 = 22")
        raise SystemExit("arithmetic mismatch not detected")
    except ValueError:
        pass
    try:
        ad.parse_arith("hello world")
        raise SystemExit("malformed row not detected")
    except ValueError:
        pass


def test_prompt_target_all_templates() -> None:
    assert cev.split_prompt_target("12 + 9 -> 21") == ("12 + 9 ->", "21")
    assert cev.split_prompt_target("add 12 and 9 = 21") == ("add 12 and 9 =", "21")
    assert cev.split_prompt_target("12+9=21") == ("12+9=", "21")
    assert cev.split_prompt_target("3 + 4 = 7") == ("3 + 4 =", "7")  # T1 behavior kept


def test_answer_spans() -> None:
    # answer spans include the separator space ("= 21" -> " 21"): positions, not
    # stripped targets, define supervision.
    spans = t1c.answer_spans(["12 + 9 = 21", "12 + 9 -> 21", "add 12 and 9 = 21"], 32)
    assert spans[0] == (8, 3)
    assert spans[1] == (9, 3)
    assert spans[2] == (14, 3)
    try:
        t1c.answer_spans(["x" * 40 + " = 1"], 32)
        raise SystemExit("overlong row not detected")
    except ValueError:
        pass


def test_mid_spec_rules() -> None:
    k = t1c.MID_SPEC_KWARGS
    assert k["width"] == k["query_heads"] * k["head_dimension"] == 128
    assert k["query_heads"] % k["kv_heads"] == 0
    assert k["tied_embeddings"] and k["dropout"] == 0.0 and not k["linear_bias"]
    assert k["head_dimension"] == 16  # same rope geometry as MINI
    assert t1c.MID_EXPECTED_PARAMS == 3_737_472
    assert t1c.MINI_EXPECTED_PARAMS == 1_647_104


def test_budget_arithmetic() -> None:
    # Budgets need not divide evenly by tokens/update: run_arm floors to whole
    # updates (updates_total = budget // (batch*length)). Invariant: for every
    # calibration shape the wasted remainder is less than one update.
    assert set(t1c.ARMS) == {"A", "B", "C", "D"}
    total = 0
    for tag, cfg in t1c.ARMS.items():
        assert cfg["budget"] == 8_000_000, tag
        total += cfg["budget"]
        for b, ln in t1c.CALIBRATION_SHAPES:
            used = cfg["budget"] // (b * ln) * (b * ln)
            assert 0 <= cfg["budget"] - used < b * ln, (tag, b, ln)
    assert total == 32_000_000
    assert t1c.CALIBRATION_SHAPES[0] == (32, 32)


def test_eval_slice_leakage_zeros() -> None:
    evals = {}
    for name in ("dev", "test_core", "test_template", "test_range", "test_composition"):
        n = ad.SPLITS[name]["n"] if name != "dev" else 5_000
        evals[name] = [ad.row_at(name, i)[0] for i in range(min(n, ad.SPLITS[name]["n"]))]
    names = sorted(evals)
    for x in range(len(names)):
        for y in range(x + 1, len(names)):
            a, b = names[x], names[y]
            assert not (set(evals[a]) & set(evals[b])), f"exact leak {a}x{b}"


def _arm(test_acc=0.0, train_acc=0.0, lcb=0.0, ucb=0.008, status="FAIL",
         hist=None, dev_pairs=None):
    return {"status": status,
            "trained": {"test_core": {"accuracy": test_acc, "wilson_lcb": lcb,
                                      "wilson_ucb": ucb},
                        "train_sample": {"accuracy": train_acc}},
            "untrained": {"test_core": {"accuracy": 0.0, "wilson_lcb": 0.0,
                                        "wilson_ucb": 0.008}},
            "diagnostics": {"stop_histogram": hist or {"NEWLINE": 500}},
            "intermediates": {str(k): {"dev_exact": v} for k, v in (dev_pairs or [])}}


def test_feed_contract_no_gaps_no_wrap() -> None:
    """Feeds advance exactly one row per element: consecutive calls tile the
    corpus prefix with no stride gaps; the largest arm never wraps train."""
    first16 = t1c._rich_feed(0, 8) + t1c._rich_feed(8, 8)
    expect = [ad.row_at("train", i)[0] for i in range(16)]
    assert first16 == expect
    for tag, cfg in t1c.ARMS.items():
        for b, ln in t1c.CALIBRATION_SHAPES:
            updates = cfg["budget"] // (b * ln)
            if cfg["data"] == "rich":
                assert updates * b <= ad.SPLITS["train"]["n"], (tag, b, ln)
    narrow = t1c._narrow_feed(3995, 10, train_rows=["r%d" % i for i in range(4000)])
    assert narrow == ["r%d" % i for i in list(range(3995, 4000)) + list(range(5))]


def test_classify_every_rule() -> None:
    base = {t: _arm() for t in "ABCD"}
    assert t1c.classify_cross_arm(base)["labels"] == ["INCONCLUSIVE"]
    arms = dict(base, B=_arm(test_acc=0.30, lcb=0.26))
    assert "OBJECTIVE_LIMITED" in t1c.classify_cross_arm(arms)["labels"]
    arms = dict(base, B=_arm(test_acc=0.30, lcb=0.26),
                C=_arm(test_acc=0.05, lcb=0.03, ucb=0.08))
    assert "DATA_LIMITED" in t1c.classify_cross_arm(arms)["labels"]
    arms = dict(base, D=_arm(test_acc=0.30, lcb=0.26))
    assert "SCALE_LIMITED" in t1c.classify_cross_arm(arms)["labels"]
    arms = dict(base, C=_arm(test_acc=0.02, train_acc=0.80))
    assert "MEMORIZATION" in t1c.classify_cross_arm(arms)["labels"]
    arms = dict(base, A=_arm(hist={"NON_ALPHABET": 300, "PAD": 100, "NEWLINE": 100}))
    assert "FORMAT_FAILURE" in t1c.classify_cross_arm(arms)["labels"]
    arms = dict(base, B=_arm(dev_pairs=[(100, 0.02), (200, 0.09)]))
    assert "BUDGET_LIMITED" in t1c.classify_cross_arm(arms)["labels"]
    arms = dict(base, B=_arm(test_acc=0.40, lcb=0.35, status="PASS"))
    assert "CAPABILITY_LEARNED" in t1c.classify_cross_arm(arms)["labels"]
    arms = dict(base, C={"arm": "C", "status": "IMPLEMENTATION_FAILURE", "error": "x"})
    out = t1c.classify_cross_arm(
        {t: r for t, r in arms.items() if r.get("status") in ("PASS", "FAIL")})
    assert isinstance(out["labels"], list)  # missing arms must not crash classification


def main() -> int:
    tests = [test_indexed_determinism_and_bounds, test_train_family_ranges_and_uniqueness,
             test_templates_render_and_parse,
             test_prompt_target_all_templates,
             test_feed_contract_no_gaps_no_wrap,
             test_answer_spans, test_mid_spec_rules, test_budget_arithmetic,
             test_eval_slice_leakage_zeros, test_classify_every_rule]
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
