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


def _arm(test14=(0.0, 0.0, 0.0, 0.0), train14=(0.0,) * 4, status="SCIENTIFIC_FAIL",
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
    arms = dict(base, B=_arm(test14=(0.5,) * 4, lcb=0.45, status="SCIENTIFIC_PASS"))
    assert "CAPABILITY_LIFTED" in t1d.classify_cross_arm(arms)["labels"]
    mixed = dict(base, C={"arm": "C", "status": "TIMEBOX_ABORT", "updates_done": 12},
                 D={"arm": "D", "status": "IMPLEMENTATION_FAILURE", "error": "x"})
    out = t1d.classify_cross_arm(
        {t: r for t, r in mixed.items()
         if r.get("status") in ("SCIENTIFIC_PASS", "SCIENTIFIC_FAIL")})
    assert out["labels"] == ["BELOW_FIT_FLOOR"]  # non-scientific arms excluded, rest evaluated


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


def test_packing_adversarial_matrix() -> None:
    """Single/exact-fit/remainder/max-length/tiny/mixed/padding-tail packing."""
    from citadel_tpu import t1d_run as t1d

    cases = [
        ["1 + 2 = 3"],
        ["x" * 64],
        ["x" * 63, "1 + 2 = 3"],  # 1-token remainder forces a new sequence
        ["9" * 64],
        ["1 + 2 = 3"] * 40,  # many tiny rows
        ["12 + 9 = 21", "123456 + 789012 = 912468", "7+8=15",
         "add 12345 and 67890 = 80235"],
    ]
    for rows in cases:
        seqs, placements = t1d.pack_rows(rows, 64)
        assert len(placements) == len(rows)
        assert sum(len(s) for s in seqs) == len(rows)
        for s, seq in enumerate(seqs):
            assert [sg for sg, _ in seq] == list(range(len(seq)))
            assert sum(ln for _, ln in seq) <= 64
    try:
        t1d.pack_rows(["x" * 65], 64)
        raise SystemExit("overlong row accepted")
    except ValueError:
        pass
    seqs, _ = t1d.pack_rows(["1 + 2 = 3", "4 + 5 = 9"], 64)
    assert len(seqs) == 1  # shared sequence, distinct segment ids


def test_teacher_heldout_band() -> None:
    for kind in ("digadd", "digsub", "singlemul", "divmicro"):
        for j in (900_000, 900_199):
            text, meta = td.teacher_row(kind, j)
            prompt, target = cev.split_prompt_target(text)
            assert len(text) <= 64, text
            assert meta["template"] == "teacher"


def test_pre50m_estimators_and_decider() -> None:
    from citadel_tpu import pre50m as p50

    assert p50.PRE50M_TARGET["value_tokens"] == 50_000_000
    assert p50.PRE50M_TARGET["cymek_sha"] == "28bf57a"
    mem = p50.memory_estimate(7_378_368)
    assert mem["parameter_bytes"] == 7_378_368 * 4
    assert mem["optimizer_moment_bytes"] == 7_378_368 * 8
    assert mem["verdict"] == "FIT"
    mid = p50.memory_estimate(3_737_472)
    assert mid["resident_gb"] < mem["resident_gb"]
    est = p50.throughput_estimates(8_000.0)
    assert abs(est["estimates_seconds"]["50M"] - 6_250.0) < 1e-9
    assert abs(est["estimates_seconds"]["1B"] - 125_000.0) < 1e-9
    try:
        p50.throughput_estimates(0.0)
        raise SystemExit("nonpositive rate accepted")
    except ValueError:
        pass
    cands = [{"batch": 256, "length": 64, "tokens_per_second": 5_000.0, "correct": True},
             {"batch": 1024, "length": 64, "tokens_per_second": 0.0,
              "correct": False, "error": "OOMError: mem"}]
    sel = p50.oom_decision(cands)
    assert sel["selected"]["batch"] == 256 and len(sel["rejected"]) == 1
    assert sel["rejected"][0]["reason"].startswith("OOMError")
    assert p50.oom_decision([])["status"] == "NO_FEASIBLE_CONFIG"
    ga = p50.grad_accumulation_status(True, 256)
    assert ga["required"] is False and ga["status"] == "NOT_REQUIRED"
    ga2 = p50.grad_accumulation_status(False, 4096)
    assert ga2["required"] is True
    base = {"target": {"understood": True, "type": "tokens", "parameter_count": None},
            "smoke": {"reload_output_identity": True,
                      "optimizer_resume": {"moments_preserved": True},
                      "grad_norm": {"max": 1.5}, "losses": [9.0, 8.0]},
            "feasibility": {"verdict": "FIT"},
            "data_interface": {"status": "PASS"}, "packing": {"status": "PASS"},
            "recommended_batch": 256, "recommended_sequence_length": 64,
            "rate_tok_s": 8000.0}
    import copy

    assert p50.build_decision(**copy.deepcopy(base))["ready_for_50m_training"] is True
    bad = copy.deepcopy(base)
    bad["feasibility"] = {"verdict": "DOES_NOT_FIT"}
    d = p50.build_decision(**bad)
    assert d["ready_for_50m_training"] is False and len(d["blocking_reasons"]) == 1
    bad2 = copy.deepcopy(base)
    bad2["target"] = {"understood": False}
    assert p50.build_decision(**bad2)["ready_for_50m_training"] is False
    bad3 = copy.deepcopy(base)
    bad3["smoke"] = {"reload_output_identity": False, "optimizer_resume": {},
                     "grad_norm": {"max": 0.0}, "losses": []}
    d3 = p50.build_decision(**bad3)
    assert d3["ready_for_50m_training"] is False and len(d3["blocking_reasons"]) >= 2


def test_loss_alignment_mirror() -> None:
    """Pure mirror of production keep semantics (causal shift + segment +
    BOS/PAD exclusion + eligible mask): prompt/pad never kept, answer kept,
    no cross-segment leakage. The live ELIGIBLE_MISMATCH assert enforces this
    on TPU; these fixtures pin the specification."""

    def keep(tokens, seg, eligible, bos=2, pad=0):
        kept = []
        for t in range(1, len(tokens)):
            if seg[t] != seg[t - 1] or seg[t] < 0:
                continue
            if tokens[t] in (bos, pad):
                continue
            if not eligible[t]:
                continue
            kept.append(t)
        return kept

    from citadel_tpu import t1c_run as t1c

    def spans_of(texts, length=64):
        return t1c.answer_spans(texts, length)

    # single ordinary row: kept == answer span, prompt excluded
    row = "12 + 9 = 21"
    ids = cev.encode(row)
    plen, alen = t1c.answer_spans([row], 64)[0]
    assert plen == 8 and alen == 3
    eligible = [False] * len(ids)
    for p in range(plen, plen + alen):
        eligible[p] = True
    assert keep(ids, [0] * len(ids), eligible) == [8, 9, 10]
    # teacher row: prompt excluded incl. words, target span kept
    trow = "digadd 7 8 carry0 = digit5 carry1"
    tids = cev.encode(trow)
    tpl, tal = t1c.answer_spans([trow], 64)[0]
    tel = [False] * len(tids)
    for p in range(tpl, tpl + tal):
        tel[p] = True
    assert keep(tids, [0] * len(tids), tel) == list(range(tpl, tpl + tal))
    assert tpl > 10  # prompt is long; only the tail is supervised
    # packed multi-row: no cross-segment leakage at the boundary
    r1, r2 = "1 + 2 = 3", "4 + 5 = 13"
    i1, i2 = cev.encode(r1), cev.encode(r2)
    tokens = i1 + i2
    seg = [0] * len(i1) + [1] * len(i2)
    elig = [False] * len(tokens)
    s1 = t1c.answer_spans([r1], 64)[0]
    s2 = t1c.answer_spans([r2], 64)[0]
    for p in range(s1[0], s1[0] + s1[1]):
        elig[p] = True
    for p in range(len(i1) + s2[0], len(i1) + s2[0] + s2[1]):
        elig[p] = True
    kept = keep(tokens, seg, elig)
    # segment boundary: force eligible True exactly at the first token of
    # segment 1 — the segment rule must still exclude it (no cross leakage).
    elig[boundary := len(i1)] = True
    assert boundary not in keep(tokens, seg, elig)
    # every genuinely eligible answer position is kept
    elig[boundary] = False
    for p in range(len(i1) + s2[0], len(i1) + s2[0] + s2[1]):
        assert p in keep(tokens, seg, elig), (p, keep(tokens, seg, elig))
    # exactly-full row and padding tail: pads never kept
    full = "9" * 60 + " = 9"
    assert len(full) <= 64
    ids = cev.encode(full)
    pad = ids + [0] * (64 - len(ids))
    seg = [0] * len(ids) + [-1] * (64 - len(ids))
    el = [False] * 64
    for p in range(len(ids) - 1, len(ids)):
        el[p] = True
    kept = keep(pad, seg, el)
    assert kept == [len(ids) - 1]


def test_perfect_predictor_per_tier() -> None:
    """Perfect predictions score 1.0 on every tier; perturbations behave per
    the frozen normalization contract (each perturbed form is constructed to
    be guaranteed-wrong, except the contractually-correct leading-zero and
    newline forms)."""
    for tier in range(5):
        rows = [td.tier_row(tier, "test", j)[0] for j in range(30)]
        tgts = [cev.split_prompt_target(r)[1] for r in rows]
        assert cev.summarize(list(tgts), tgts)["accuracy"] == 1.0, tier
        wrong_digit = [t[:-1] + ("0" if t[-1:] != "0" else "1") if t else "9"
                       for t in tgts]
        assert cev.summarize(wrong_digit, tgts)["accuracy"] == 0.0, tier
        wrong_sign = [("1" if t in ("0", "-0") else
                       ("-" + t if not t.startswith("-") else t[1:] + "5"))
                      for t in tgts]
        assert cev.summarize(wrong_sign, tgts)["accuracy"] == 0.0, tier
        wrong_len = [t + "1" for t in tgts]
        assert cev.summarize(wrong_len, tgts)["accuracy"] == 0.0, tier
        wordy = ["answer is " + t for t in tgts]
        assert cev.summarize(wordy, tgts)["accuracy"] == 0.0, tier
        assert cev.summarize([""] * len(rows), tgts)["accuracy"] == 0.0, tier
        padded = [("-" + "00" + t[1:] if t.startswith("-") else "00" + t)
                  for t in tgts]
        assert cev.summarize(padded, tgts)["accuracy"] == 1.0, tier
        newlined = [t + "\n" for t in tgts]
        assert cev.summarize(newlined, tgts)["accuracy"] == 1.0, tier


def test_template_enumeration_battery() -> None:
    """Every template family × boundary numbers through the full chain:
    encode/decode, parse, split, spans, buffer geometry."""
    cases = ["0 + 0 = 0", "1 + 1 = 2", "9 + 9 = 18", "10 + 10 = 20",
             "91 - 100 = -9", "999999 + 999999 = 1999998",
             "12+9=21", "12 + 9 -> 21", "add 12 and 9 = 21",
             "subtract 9 from 12 = 3", "multiply 999 by 999 = 998001",
             "divide 72 by 8 = 9", "3 + 4 = 7", "18 - 7 = 11", "6 * 8 = 48"]
    for row in cases:
        ids = cev.encode(row)
        assert all(i not in (0, 1, 2, 3) for i in ids), row
        prompt, target = cev.split_prompt_target(row)
        assert cev.normalize_answer(target) is not None, (row, target)
        plen, alen = t1d_answer_spans(row)
        assert alen > 0 and plen + alen == len(ids), row
    cap = cev.validate_generation_capacity(cases)
    assert cap["max_required_tokens"] <= 64
    for kind in ("digadd", "digsub", "singlemul", "divmicro"):
        text, _ = td.teacher_row(kind, 42)
        prompt, target = cev.split_prompt_target(text)
        plen, alen = t1d_answer_spans(text)
        assert alen > 0 and target, (kind, text)


def t1d_answer_spans(row):
    from citadel_tpu import t1c_run as t1c

    return t1c.answer_spans([row], 64)[0]


def test_bundle_verify_matrix() -> None:
    """verify_bundle: valid session passes; each defect class fails loudly."""
    import tempfile

    from citadel_tpu import t1d_run as _run

    def _arm_json(tag, status="SCIENTIFIC_FAIL"):
        return {"arm": tag, "status": status}

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name in t1d.BUNDLE_FILES:
            if name.startswith("ARM_"):
                tag = name[4]
                (root / name).write_text(json.dumps(_arm_json(tag)), encoding="utf-8")
            else:
                (root / name).write_text(json.dumps({"dummy": True}), encoding="utf-8")
        import zipfile

        with zipfile.ZipFile(root / "CITADEL_T1D_RESULTS.zip", "w") as zf:
            for name in t1d.BUNDLE_FILES:
                zf.write(root / name, name)
        got = t1d.verify_bundle(str(root))
        assert got["status"] == "VALID" and got["files"] == len(t1d.BUNDLE_FILES)
        (root / "ARM_C.json").unlink()
        try:
            t1d.verify_bundle(str(root))
            raise SystemExit("missing-file bundle accepted")
        except RuntimeError as exc:
            assert "BUNDLE_INVALID" in str(exc) and "ARM_C.json" in str(exc), exc
        (root / "ARM_C.json").write_text(json.dumps(_arm_json("C")), encoding="utf-8")
        (root / "ARM_D.json").write_text("{not json", encoding="utf-8")
        try:
            t1d.verify_bundle(str(root))
            raise SystemExit("corrupt-JSON bundle accepted")
        except RuntimeError as exc:
            assert "BUNDLE_INVALID" in str(exc), exc
        (root / "ARM_D.json").write_text(
            json.dumps({"arm": "D", "status": "BOGUS"}), encoding="utf-8")
        try:
            t1d.verify_bundle(str(root))
            raise SystemExit("bad-status bundle accepted")
        except RuntimeError as exc:
            assert "BUNDLE_INVALID" in str(exc), exc


def test_session_resume_dry_run(tmp_path=None) -> None:
    import tempfile

    from citadel_tpu import t1d_run as _run

    with tempfile.TemporaryDirectory() as tmp:
        # all 7 session states from §24, mapped to the arm predicate
        assert _run.should_skip_arm(tmp, "A") == ("run", "nothing complete")
        out = Path(tmp) / "ARM_A.json"
        out.write_text(json.dumps({"status": "SCIENTIFIC_PASS", "arm": "A"}),
                       encoding="utf-8")
        (Path(tmp) / "ARM_A.done.json").write_text(
            json.dumps({"status": "SCIENTIFIC_PASS"}), encoding="utf-8")
        got = t1d.run_arm("A", dict(t1d.ARMS["A"]), shape=(256, 64), out_dir=tmp)
        assert got.get("resumed") is True and got["status"] == "SCIENTIFIC_PASS"
        out.write_text(json.dumps({"status": "TIMEBOX_ABORT", "arm": "A"}),
                       encoding="utf-8")
        got = t1d.run_arm("A", dict(t1d.ARMS["A"]), shape=(256, 64), out_dir=tmp)
        assert got.get("resumed") is True  # timeboxed arms are not rerun
        out.unlink()
        try:
            t1d.run_arm("A", dict(t1d.ARMS["A"]), shape=(256, 64), out_dir=tmp)
            raise SystemExit("marker-without-receipt silently accepted")
        except RuntimeError as exc:
            assert "RESUME_CONFLICT" in str(exc), exc
        out.write_text(json.dumps({"status": "BOGUS"}), encoding="utf-8")
        try:
            t1d.run_arm("A", dict(t1d.ARMS["A"]), shape=(256, 64), out_dir=tmp)
            raise SystemExit("unknown status silently accepted")
        except RuntimeError as exc:
            assert "RESUME_CONFLICT" in str(exc), exc
    # timebox receipt builder is pure and schema-stable
    abort = t1d.timebox_abort_receipt(
        tag="B", cfg=dict(t1d.ARMS["B"]), env={}, n_seq=256, length=64,
        updates_done=12, ledgers=[{"updates": 12}], feeder_ledger={},
        error="TimeoutError: box", wall=1.0)
    assert abort["status"] == "TIMEBOX_ABORT" and abort["training"]["updates"] == 12
    json.dumps(abort)


def main() -> int:
    tests = [test_tier_determinism_and_bounds, test_band_isolation,
             test_leakage_verdict, test_t2_easy_constraints,
             test_teacher_rows_verify, test_curriculum_membership,
             test_packing_exactness_and_boundaries, test_feeder_static_shapes_and_prefix,
             test_answer_spans_all_families, test_scale2_rules,
             test_loss_alignment_mirror, test_perfect_predictor_per_tier,
             test_template_enumeration_battery, test_bundle_verify_matrix,
             test_packing_adversarial_matrix, test_teacher_heldout_band,
             test_pre50m_estimators_and_decider,
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
