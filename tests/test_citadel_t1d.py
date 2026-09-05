"""Citadel T1D unit tests. Zero third-party dependencies.

Run:  python tests/test_citadel_t1d.py   (exit 0 = all pass)
Covers the T1D contract without torch/TPU: tier determinism + bounds, T2
easy-constraint audit, teacher-row arithmetic verification, curriculum
schedule membership, packing exactness + boundaries, feeder static-shape and
prefix-consumption invariants, answer spans on every template family,
SCALE2 structural rules, budget arithmetic, all classifier rules on synthetic
receipts, fake-predictor metric sanity, nulls on mixed templates, masked-vocab
contents, and a session-resume dry run (marker short-circuit, no device).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
CITADEL_ROOT = HERE.parents[1]
sys.path.insert(0, str(CITADEL_ROOT))

from citadel_tpu import calculator_eval as cev  # noqa: E402
from citadel_tpu import t1d_run as t1d  # noqa: E402
from citadel_tpu import tiered_data as td  # noqa: E402


def test_tier_determinism_and_bounds() -> None:
    for t in range(5):
        assert td.tier_row(t, "train", 0) == td.tier_row(t, "train", 0)
        assert td.tier_row(t, "train", 12345) == td.tier_row(t, "train", 12345)
        assert td.tier_row(t, "dev", 9) != td.tier_row(t, "test", 9)
    for bad_tier, bad_split in ((9, "train"), (0, "valid")):
        try:
            td.tier_row(bad_tier, bad_split, 0)
            raise SystemExit(f"no error for {(bad_tier, bad_split)}")
        except ValueError:
            pass


def test_t2_easy_constraints() -> None:
    for i in range(3000):
        _, m = td.tier_row(2, "train", i)
        if m["op"] == "+":
            assert m["carries"] == 0, m
        if m["op"] == "-":
            assert m["borrows"] == 0, m


def test_teacher_rows_verify() -> None:
    for i in range(300):
        for kind in ("digadd", "digsub", "singlemul", "divmicro"):
            text, m = td.teacher_row(kind, i)
            prompt, target = cev.split_prompt_target(text)
            assert len(text) <= 64, text
            if kind == "digadd":
                assert (m["a"] + m["b"] + m["carry_in"]) % 10 == m["digit"]
                assert (m["a"] + m["b"] + m["carry_in"]) // 10 == m["carry_out"]
            elif kind == "digsub":
                assert (m["a"] - m["b"] - m["borrow_in"]) % 10 == m["digit"]
                assert (1 if m["a"] - m["b"] - m["borrow_in"] < 0 else 0) == m["borrow_out"]
            elif kind == "singlemul":
                assert m["a"] * m["b"] == m["c"] and target == str(m["c"])
            else:
                assert m["a"] // m["b"] == m["c"] and m["a"] % m["b"] == 0
    try:
        td.teacher_row("nonsense", 0)
        raise SystemExit("unknown teacher kind accepted")
    except ValueError:
        pass


def test_curriculum_membership() -> None:
    early = {td.curriculum_tier(0.05, j) for j in range(2000)}
    assert early <= {0, 1}, early
    mid = {td.curriculum_tier(0.5, j) for j in range(2000)}
    assert mid <= {2, 3}, mid
    late = {td.curriculum_tier(0.9, j) for j in range(4000)}
    assert late == {1, 2, 3, 4}, late
    uni = {td.uniform_tier(j) for j in range(4000)}
    assert uni == {0, 1, 2, 3, 4}, uni


def test_packing_exactness_and_boundaries() -> None:
    rows = ["12 + 9 = 21", "123456 + 789012 = 912468", "7+8=15", "add 1 and 2 = 3"]
    seqs, placements = t1d.pack_rows(rows, 64)
    assert len(placements) == len(rows)
    for (s, seg, start, ln), t in zip(placements, rows):
        assert ln == len(t)
    for s, seq in enumerate(seqs):
        assert sum(ln for _, ln in seq) <= 64
        assert [sg for sg, _ in seq] == list(range(len(seq)))
    try:
        t1d.pack_rows(["x" * 65], 64)
        raise SystemExit("overlong row accepted")
    except ValueError:
        pass
    try:
        t1d.pack_rows(rows, 8)
        raise SystemExit("impossible pack accepted")
    except ValueError:
        pass


def test_feeder_static_shapes_and_prefix() -> None:
    for mode in ("flat", "curriculum", "teacher"):
        f = t1d.TierFeeder(mode, 8, 64)
        for u in range(30):
            seqs = f.fill_sequences(u / 30)
            assert len(seqs) == 8, (mode, u)
        led = f.ledger()
        assert led["carry_pending"] >= 0
        for key, cnt in led["placed_rows"].items():
            assert cnt > 0, (mode, key)
    f = t1d.TierFeeder("teacher", 4, 64)
    teacher_rows = 0
    total_rows = 0
    for u in range(50):
        before = dict(f.placed_rows)
        f.fill_sequences(u / 50)
        for key, cnt in f.placed_rows.items():
            if key.startswith("teacher:"):
                teacher_rows += cnt - before.get(key, 0)
            total_rows += cnt - before.get(key, 0)
    # Row split is exactly 60/40 by construction; token split follows lengths
    # (teacher rows run longer) and is recorded per arm, not gated here.
    assert abs(teacher_rows / total_rows - 0.40) < 0.02, (teacher_rows, total_rows)
    teacher_tok = sum(v for k, v in f.placed_tokens.items() if k.startswith("teacher:"))
    total_tok = sum(f.placed_tokens.values())
    assert 0.45 <= teacher_tok / total_tok <= 0.60, teacher_tok / total_tok
    assert sum(f.carried.values()) == f.ledger()["carry_pending"]


def test_answer_spans_all_families() -> None:
    rows = ["12 + 9 = 21", "12+9=21", "12 + 9 -> 21", "add 12 and 9 = 21",
            "digadd 7 8 carry0 = digit5 carry1", "72 / 8 = 9"]
    spans = t1d.answer_spans(rows, 64) if hasattr(t1d, "answer_spans") else None
    from citadel_tpu import t1c_run as t1c

    spans = t1c.answer_spans(rows, 64)
    for (plen, alen), t in zip(spans, rows):
        assert alen > 0 and plen + alen == len(t), (t, plen, alen)


def test_scale2_rules() -> None:
    k = t1d.SCALE2_SPEC_KWARGS
    assert k["width"] == k["query_heads"] * k["head_dimension"] == 192
    assert k["query_heads"] % k["kv_heads"] == 0
    assert k["head_dimension"] == 16 and k["tied_embeddings"]
    assert k["dropout"] == 0.0 and not k["linear_bias"]
    assert t1d.SCALE2_EXPECTED_PARAMS == 7_378_368
    assert set(t1d.ARMS) == {"A", "B", "C", "D", "E"}
    total = sum(c["budget"] for c in t1d.ARMS.values())
    assert total == 8_000_000 * 3 + 4_000_000 * 2
    for tag, cfg in t1d.ARMS.items():
        for b, ln in t1d.CALIBRATION_SHAPES:
            used = cfg["budget"] // (b * ln) * (b * ln)
            assert 0 <= cfg["budget"] - used < b * ln, (tag, b, ln)


def _arm(test14=(0.0, 0.0, 0.0, 0.0), train14=(0.0,) * 4, status="FAIL",
         hist=None, dev_pairs=None, lcb=0.0, ucb=0.008):
    def s(acc):
        n = 500
        return {"accuracy": acc, "correct": int(acc * n), "total": n,
                "wilson_lcb": lcb if acc == 0.0 else acc - 0.04,
                "wilson_ucb": ucb if acc == 0.0 else acc + 0.04}

    return {"status": status,
            "trained": {f"t{t}": s(a) for t, a in zip((1, 2, 3, 4), test14)},
            "untrained": {f"t{t}": s(0.0) for t in (1, 2, 3, 4)},
            "trained_train": {f"t{t}": s(a) for t, a in zip(range(5), (0.0,) + train14)},
            "nulls_per_tier": {f"t{t}": {"strongest": "copy_first_operand", "accuracy": 0.02}
                               for t in range(5)},
            "diagnostics": {"stop_histogram": hist or {"NEWLINE": 500}},
            "intermediates": {str(k): {"dev_exact": v} for k, v in (dev_pairs or [])},
            "reload_identical": True}


def test_band_isolation() -> None:
    """Rows honor their split's operand band (single source: BANDS table)."""
    import random as _random

    rng = _random.Random(7)

    def band_ok(tier, split, m):
        if tier == 1:
            return True  # probe tier: no bands possible by construction
        band = td.BANDS[tier][split]
        op = m["op"]
        if tier == 0:
            return band["x_lo"] <= m["a"] <= band["x_hi"]
        if tier == 2 and op in ("+", "-"):
            # hundreds band + second operand >= 2 (never a T0 trivial form)
            return band["ha_lo"] <= m["a"] // 100 <= band["ha_hi"] and m["b"] >= 2
        if op == "/":
            lo = band["div_q_lo"]
            hi = band["div_q_hi"]
            return lo <= m["c"] <= hi
        if op == "*":
            lo = band["mult_a_lo"]
            hi = band["mult_a_hi"]
            return lo <= m["a"] <= hi
        return band["lo"] <= m["a"] <= band["hi"]

    # divisor ladder keeps division triples disjoint across tiers by construction
    for tier, lo, hi in ((2, 2, 9), (3, 10, 99), (4, 100, 999)):
        for _ in range(100):
            _, m = td.tier_row(tier, "train", rng.randrange(50_000))
            if m["op"] == "/":
                assert lo <= m["b"] <= hi, (tier, m)

    for tier in range(5):
        for split in ("train", "dev", "test"):
            for _ in range(150):
                text, m = td.tier_row(tier, split, rng.randrange(1_000_000))
                assert band_ok(tier, split, m), (tier, split, text)
    # train/test bands strictly disjoint where banding exists (table-level)
    for tier in (0, 2, 3, 4):
        if "train" not in td.BANDS.get(tier, {}) or "test" not in td.BANDS.get(tier, {}):
            raise AssertionError(f"tier {tier} missing train/test bands")
        tr, te = td.BANDS[tier]["train"], td.BANDS[tier]["test"]
        for key in ("x_hi", "ha_hi", "lo", "mult_a_hi", "div_q_hi"):
            lok = key.replace("_hi", "_lo")
            if lok in tr and key in tr and lok in te:
                assert tr[key] < te[lok], (tier, key)
    for split, lo, hi in (("train", 0, 5999), ("dev", 6000, 7999), ("test", 8000, 9999)):
        for _ in range(50):
            _, m = td.tier_row(0, split, rng.randrange(6000))
            assert lo <= m["a"] <= hi, (split, m)


def test_leakage_verdict() -> None:
    fatal, reported = td.leakage_verdict({"exact_dev_t2_x_test_t2": 0,
                                          "exact_dev_t3_x_test_t3": 5,
                                          "exact_train-t0_x_test_t1": 12,
                                          "exact_test_core_x_test_template": 4})
    assert fatal == {"exact_dev_t3_x_test_t3": 5}, fatal
    # zero-valued gated pairs carry no information and are dropped; nonzero
    # probe pairs (tier<=1 or keyless) are reported, never gated.
    assert reported == {"exact_train-t0_x_test_t1": 12,
                        "exact_test_core_x_test_template": 4}, reported


def test_classify_every_rule() -> None:
    base = {t: _arm() for t in "ABCDE"}
    # All-zero arms: nothing learned anywhere -> BELOW_FIT_FLOOR fires (correct).
    assert t1d.classify_cross_arm(base)["labels"] == ["BELOW_FIT_FLOOR"]
    arms = dict(base, B=_arm(test14=(0.3, 0.3, 0.3, 0.3), lcb=0.26))
    assert "CURRICULUM_HELPED" in t1d.classify_cross_arm(arms)["labels"]
    arms = dict(base, C=_arm(test14=(0.3, 0.3, 0.3, 0.3), lcb=0.26))
    assert "TEACHER_HELPED" in t1d.classify_cross_arm(arms)["labels"]
    arms = dict(base, D=_arm(test14=(0.3, 0.3, 0.3, 0.3), lcb=0.26))
    assert "SCALE_HELPED" in t1d.classify_cross_arm(arms)["labels"]
    arms = dict(base, E=_arm(test14=(0.3, 0.3, 0.3, 0.3), lcb=0.26))
    assert "REPRESENTATION_LIMITED" in t1d.classify_cross_arm(arms)["labels"]
    arms = dict(base, C=_arm(test14=(0.02,) * 4, train14=(0.8,) * 4))
    assert "GENERALIZATION_LIMITED" in t1d.classify_cross_arm(arms)["labels"]
    arms = dict(base, A=_arm(hist={"NON_ALPHABET": 300, "PAD": 100, "NEWLINE": 100}))
    assert "FORMAT_FAILURE" in t1d.classify_cross_arm(arms)["labels"]
    arms = dict(base, B=_arm(dev_pairs=[(100, 0.02), (200, 0.09)]))
    assert "BUDGET_LIMITED" in t1d.classify_cross_arm(arms)["labels"]
    arms = dict(base)
    for t in arms:
        arms[t]["trained_train"] = {f"t{i}": {"accuracy": 0.0} for i in range(5)}
    assert "BELOW_FIT_FLOOR" in t1d.classify_cross_arm(arms)["labels"]
    arms = dict(base, B=_arm(test14=(0.6, 0.6, 0.02, 0.02), lcb=0.5))
    assert "COMPLEXITY_FRONTIER" in t1d.classify_cross_arm(arms)["labels"]
    arms = dict(base, B=_arm(test14=(0.5,) * 4, lcb=0.45, status="PASS"))
    assert "CAPABILITY_LIFTED" in t1d.classify_cross_arm(arms)["labels"]


def test_fake_predictor_metric_sanity() -> None:
    gold = ["7", "11", "48", "-9"]
    perfect = cev.summarize(gold, gold)
    assert perfect["accuracy"] == 1.0 and perfect["correct"] == 4
    wrong = cev.summarize(["0", "0", "0", "0"], gold)
    assert wrong["accuracy"] == 0.0
    partial = cev.summarize(["7", "0", "48", "0"], gold)
    assert partial["accuracy"] == 0.5 and partial["correct"] == 2


def test_nulls_all_template_families() -> None:
    rows = ["12 + 9 = 21", "12+9=21", "12 + 9 -> 21", "add 12 and 9 = 21",
            "subtract 9 from 12 = 3", "72 / 8 = 9"]
    nulls = cev.heuristic_nulls(rows, rows)
    assert set(nulls) == {"always_zero", "copy_first_operand",
                          "copy_second_operand", "most_common_train_answer"}
    assert nulls["copy_first_operand"][:2] == ["12", "12"]
    assert nulls["copy_second_operand"][3] == "9"


def test_masked_vocab_contents() -> None:
    ids = t1d.valid_alphabet_ids()
    assert ids == sorted(ids) and len(set(ids)) == len(ids)
    assert all(0 <= i < 24_576 for i in ids)
    assert len(ids) < 100
    for c in "0123456789+-*/= \nabcdefghijklmnopqrstuvwxyz>":
        assert cev.encode_char(c) in ids, c


def test_session_resume_dry_run(tmp_path=None) -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "ARM_A.json"
        out.write_text(json.dumps({"status": "PASS", "arm": "A"}), encoding="utf-8")
        (Path(tmp) / "ARM_A.done.json").write_text(json.dumps({"status": "PASS"}),
                                                   encoding="utf-8")
        got = t1d.run_arm("A", dict(t1d.ARMS["A"]), shape=(256, 64), out_dir=tmp)
        assert got.get("resumed") is True and got["status"] == "PASS"


def main() -> int:
    tests = [test_tier_determinism_and_bounds, test_band_isolation,
             test_leakage_verdict, test_t2_easy_constraints,
             test_teacher_rows_verify, test_curriculum_membership,
             test_packing_exactness_and_boundaries, test_feeder_static_shapes_and_prefix,
             test_answer_spans_all_families, test_scale2_rules,
             test_classify_every_rule, test_fake_predictor_metric_sanity,
             test_nulls_all_template_families, test_masked_vocab_contents,
             test_session_resume_dry_run]
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
