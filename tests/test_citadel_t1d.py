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
    assert set(t1d.ARMS) == {"A", "B", "C", "D", "E", "F"}
    total = sum(c["budget"] for c in t1d.ARMS.values())
    assert total == 8_000_000 * 3 + 4_000_000 * 2 + 2_000_000
    for tag, cfg in t1d.ARMS.items():
        for b, ln in t1d.CALIBRATION_SHAPES:
            used = cfg["budget"] // (b * ln) * (b * ln)
            assert 0 <= cfg["budget"] - used < b * ln, (tag, b, ln)


def _arm(test14=(0.0, 0.0, 0.0, 0.0), train14=(0.0,) * 4, status="SCIENTIFIC_FAIL",
         hist=None, dev_pairs=None, lcb=0.0, ucb=0.008):
    """Synthetic arm receipt in the REAL schema: intermediates carry the
    per-tier DEV structure the producer writes (inter[cp][f"t{tier}"]["exact"]),
    nulls_per_tier carries all five tiers with strongest/accuracy/all."""
    def s(acc, n=500):
        return {"accuracy": acc, "correct": int(acc * n), "total": n,
                "wilson_lcb": lcb if acc == 0.0 else acc - 0.04,
                "wilson_ucb": ucb if acc == 0.0 else acc + 0.04}

    inter = {}
    for cp, v in (dev_pairs or []):
        inter[str(cp)] = {f"t{tier}": {"exact": v, "lcb": max(0.0, v - 0.03)}
                          for tier in range(5)}
    if not inter:
        inter = {str(cp): {f"t{tier}": {"exact": 0.0, "lcb": 0.0}
                           for tier in range(5)}
                 for cp in (25, 50, 75, 100)}
    return {"schema": "citadel-t1d-arm/v1", "status": status,
            "trained": {f"t{t}": s(a) for t, a in zip(range(5), (0.0,) + test14)},
            "untrained": {f"t{t}": s(0.0) for t in range(5)},
            "untrained_dev": {f"t{t}": s(0.0, n=200) for t in range(5)},
            "trained_train": {f"t{t}": s(a) for t, a in zip(range(5), (0.0,) + train14)},
            "train_memorization": {f"t{t}": {"consumed_prefix": 200,
                                             "n_verified_consumed": 200,
                                             "status": "OK", "lift_eligible": True}
                                   for t in range(5)},
            "nulls_per_tier": {f"t{t}": {"strongest": "copy_first_operand",
                                         "accuracy": 0.02,
                                         "all": {"copy_first_operand": 0.02}}
                               for t in range(5)},
            "heuristic_nulls": {"copy_first_operand": {"accuracy": 0.02,
                                                       "correct": 10, "total": 500,
                                                       "wilson_lcb": 0.01,
                                                       "wilson_ucb": 0.03}},
            "diagnostics": {"stop_histogram": hist or {"NEWLINE": 500},
                            "first_train_lift_tier": None,
                            "first_test_lift_tier": None},
            "intermediates": inter,
            "gate_rules": {"nonoverlap": False, "beats_null": False,
                           "margin": False, "loss": True, "reload": True},
            "checkpoint": {"path": "t1d_arm_synthetic.pt",
                           "sha256": "a" * 64},
            "pre_reload_prediction_sha256": "0" * 64,
            "post_reload_prediction_sha256": "0" * 64,
            "training": {"updates": 10, "ledgers": [{"updates": 10}]},
            "data": {"feeder": {"placed_rows": {}, "placed_tokens": {},
                                "carry_pending": 0}},
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
    base = {t: _arm() for t in t1d.ARM_ORDER}
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
    green_smoke = {"status": "PASS", "reload_output_identity": True,
                   "optimizer_resume": {"moments_preserved": True,
                                        "continued_update_ok": True},
                   "grad_norm": {"max": 1.5}, "losses": [9.0, 8.0],
                   "param_mutation": True, "production_transaction": True,
                   "checkpoint_compat": {"compatible": True},
                   "writer_fence_probe": "rejected-as-required",
                   "token_accounting": {"consistent": True}}
    green_data = {"status": "PASS", "capacity_tokens": 4096, "real_tokens": 4000,
                  "loss_bearing_tokens": 900, "padding_tokens": 96,
                  "scheduled_rows": 64}
    base = {"target": {"understood": True, "type": p50.PRE50M_TARGET["type"],
                       "value_tokens": p50.PRE50M_TARGET["value_tokens"],
                       "parameter_count": None},
            "smoke": green_smoke, "feasibility": {"verdict": "FIT"},
            "data_interface": green_data, "packing": {"status": "PASS"},
            "recommended_batch": 256, "recommended_sequence_length": 64,
            "rate_tok_s": 8000.0}
    import copy

    green = p50.build_decision(**copy.deepcopy(base))
    assert green["ready_for_50m_training"] is True
    assert green["blocking_reasons"] == []
    bad = copy.deepcopy(base)
    bad["feasibility"] = {"verdict": "DOES_NOT_FIT"}
    d = p50.build_decision(**bad)
    assert d["ready_for_50m_training"] is False
    assert any("does not fit" in r for r in d["blocking_reasons"])
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

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name in t1d.BUNDLE_FILES:
            if name.startswith("ARM_"):
                tag = name[4]
                r = _arm(status="SCIENTIFIC_FAIL")
                r["arm"] = tag
                (root / name).write_text(json.dumps(r), encoding="utf-8")
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
        fixed = _arm(status="SCIENTIFIC_FAIL")
        fixed["arm"] = "C"
        (root / "ARM_C.json").write_text(json.dumps(fixed), encoding="utf-8")
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


def test_train_memorization_plan_matrix() -> None:
    """Frozen candidates verified against the EXACT consumed prefix:
    zero / <200 / ==200 / >200 consumed; lift eligibility follows the plan,
    and FIRST_TRAIN_LIFT can never fire on n < LIFT_MIN_N."""
    class FakeFeeder:
        def __init__(self, prefixes):
            self._p = dict(prefixes)

        def consumed_prefix(self, tier):
            return self._p.get(tier, 0)

    plan = t1d.train_memorization_plan(FakeFeeder({}))
    assert all(plan[t]["n_verified"] == 0 for t in range(5))
    assert all(plan[t]["status"] == "INSUFFICIENT_CONSUMPTION" for t in range(5))

    plan = t1d.train_memorization_plan(FakeFeeder({0: 150, 1: 199}))
    assert plan[0]["n_verified"] == 150 and plan[0]["status"] == "INSUFFICIENT_CONSUMPTION"
    assert plan[1]["n_verified"] == 199 and plan[1]["status"] == "INSUFFICIENT_CONSUMPTION"

    plan = t1d.train_memorization_plan(FakeFeeder({2: 200}))
    assert plan[2]["n_verified"] == 200 and plan[2]["status"] == "OK"

    plan = t1d.train_memorization_plan(FakeFeeder({3: 5000, 4: 999_999}))
    assert plan[3]["n_verified"] == 200 and plan[3]["status"] == "OK"
    assert plan[4]["n_verified"] == 200 and plan[4]["status"] == "OK"
    # frozen candidates are the first 200 indices, deterministic
    cands = t1d.frozen_train_candidates()
    assert all(cands[t] == list(range(t1d.TRAIN_SAMPLE_PER_TIER)) for t in range(5))
    # verified indices are always a prefix of the candidates (index order)
    for t in range(5):
        assert plan[t]["verified_indices"] == cands[t][:plan[t]["n_verified"]]


def test_lift_eligibility_never_fires_below_min_n() -> None:
    """A tier with n_verified < LIFT_MIN_N is INSUFFICIENT_CONSUMPTION and its
    train-lift flag is False even at perfect accuracy."""
    feeder_notes = {"consumed_prefix": lambda tier: 150}
    plan = t1d.train_memorization_plan(
        type("F", (), {"consumed_prefix": staticmethod(lambda tier: 150)})())
    assert plan[0]["status"] == "INSUFFICIENT_CONSUMPTION"
    mem = {"consumed_prefix": 150, "n_frozen_candidates": 200,
           "n_verified_consumed": 150, "evaluated_rows": 150,
           "status": plan[0]["status"],
           "lift_eligible": bool(plan[0]["status"] == "OK" and 1.0 >= t1d.LIFT_THRESHOLD)}
    assert mem["lift_eligible"] is False


def test_null_block_validation() -> None:
    """validate_null_block: complete t0-t4 blocks pass; every defect class
    the nulls_per_tier bug produced is rejected."""
    good = {"heuristic_nulls": {"n": {"accuracy": 0.1}},
            "nulls_per_tier": {f"t{t}": {"strongest": "x", "accuracy": 0.02,
                                         "all": {"x": 0.02}} for t in range(5)}}
    assert t1d.validate_null_block(good) == []
    buggy = {"heuristic_nulls": {"n": {"accuracy": 0.1}},
             "nulls_per_tier": {"t4": {"strongest": "x", "accuracy": 0.02,
                                       "all": {"x": 0.02}}}}
    defects = t1d.validate_null_block(buggy)
    assert defects and "t0" in defects[0], defects
    bad_acc = {"heuristic_nulls": {"n": {"accuracy": 0.1}},
               "nulls_per_tier": {f"t{t}": {"strongest": "x", "accuracy": 0.02,
                                            "all": {"x": 0.02}} for t in range(5)}}
    bad_acc["nulls_per_tier"]["t2"] = {"strongest": "x", "accuracy": 1.5,
                                       "all": {"x": 1.5}}
    assert any("t2" in d and "finite" in d for d in t1d.validate_null_block(bad_acc))
    missing_key = {"heuristic_nulls": {"n": {"accuracy": 0.1}},
                   "nulls_per_tier": {f"t{t}": {"strongest": "x", "accuracy": 0.02,
                                                "all": {"x": 0.02}} for t in range(5)}}
    del missing_key["nulls_per_tier"]["t3"]["all"]
    assert any("t3" in d and "all" in d for d in t1d.validate_null_block(missing_key))
    nan_block = {"heuristic_nulls": {"n": {"accuracy": 0.1}},
                 "nulls_per_tier": {f"t{t}": {"strongest": "x", "accuracy": 0.02,
                                              "all": {"x": 0.02}} for t in range(5)}}
    nan_block["nulls_per_tier"]["t1"]["accuracy"] = float("nan")
    assert t1d.validate_null_block(nan_block)


def _synthetic_records(targets, acc=1.0):
    """Generation records in the exact producer schema (calculator_eval.generate)."""
    out = []
    for i, t in enumerate(targets):
        pred = t if i < int(acc * len(targets)) or acc >= 1.0 else "7"
        out.append({"prompt": "synthetic", "target": t, "prediction": pred,
                    "correct": pred == t, "stop_reason": "EOS",
                    "generated_token_count": len(pred), "valid": True})
    return out


def test_post_training_arm_simulation() -> None:
    """THE §12 test: a completed synthetic arm traverses every post-training
    stage on real (pure) rows — trained summaries, trained_train via the
    verified-consumption plan, nulls global+per-tier, diagnostics, lift,
    scientific gate, receipt + marker serialization, classification — and
    reaches the final receipt. This flow would have caught the
    nulls_per_tier bug (incomplete t0-t3 crashed the gate) and the
    train-memorization bug (n=1 samples)."""
    import tempfile

    from citadel_tpu import calculator_eval as cev

    with tempfile.TemporaryDirectory() as tmp:
        feeder = t1d.TierFeeder("curriculum", 8, 64)
        for u in range(400):
            feeder.fill_sequences(u / 400)
        slices = t1d._tier_slices()
        cands = t1d.frozen_train_candidates()
        plan = t1d.train_memorization_plan(feeder, candidates=cands)
        assert all(plan[t]["status"] == "OK" for t in range(5)), \
            "consumed prefixes must exceed 200 in this scenario"

        def s(acc, n=500):
            return {"correct": int(acc * n), "total": n, "accuracy": acc,
                    "wilson_lcb": max(0.0, acc - 0.04),
                    "wilson_ucb": min(1.0, acc + 0.04)}

        accs = {0: 0.9, 1: 0.5, 2: 0.3, 3: 0.1, 4: 0.02}
        trained = {f"t{t}": s(accs[t]) for t in range(5)}
        trained_recs = {
            f"t{t}": _synthetic_records(slices[f"test_t{t}"]["targets"],
                                        acc=accs[t]) for t in range(5)}
        untrained = {f"t{t}": s(0.0) for t in range(5)}
        untrained_train = {f"t{t}": s(0.0, n=200) for t in range(5)}
        trained_train, train_memorization = {}, {}
        for tier in range(5):
            entry = plan[tier]
            rows = [td.tier_row(tier, "train", i)[0]
                    for i in entry["verified_indices"]]
            tgts = [cev.split_prompt_target(r)[1] for r in rows]
            summ = s(0.6, n=len(rows)) if rows else s(0.0, n=0)
            trained_train[f"t{tier}"] = summ
            train_memorization[f"t{tier}"] = {
                "consumed_prefix": entry["consumed_prefix"],
                "n_frozen_candidates": entry["n_candidates"],
                "n_verified_consumed": entry["n_verified"],
                "evaluated_rows": len(rows),
                "status": entry["status"],
                "lift_eligible": bool(entry["status"] == "OK"
                                      and summ["accuracy"] >= t1d.LIFT_THRESHOLD)}
        inter = {}
        for cp, dev in ((25, 0.05), (50, 0.12), (75, 0.18), (100, 0.22)):
            inter[str(cp)] = {f"t{tier}": {"exact": dev, "lcb": dev - 0.03}
                              for tier in range(5)}
        receipt = t1d.build_arm_receipt(
            tag="A", cfg=dict(t1d.ARMS["A"]), env={"probe_pass": True},
            n_seq=8, length=64, param_count=3_737_472,
            citadel_sha="0" * 40, cymek_sha="1" * 64,
            feeder=feeder, seed=20260904, slices=slices,
            ledgers=[{"updates": 100, "first_loss": 9.0, "last_loss": 6.0}],
            done=100, first_loss=9.0, last_loss=6.0, cap_total=100 * 8 * 64,
            ans_total=1234, whole_total=42_000, gsum=3.0, gmax=0.9, gn=100,
            train_wall=60.0, untrained=untrained,
            untrained_train=untrained_train, trained=trained,
            trained_recs=trained_recs, trained_train=trained_train,
            train_memorization=train_memorization, inter=inter,
            teacher_eval={"skipped": "n/a"}, first_step={"n": 0},
            ckpt_path="t1d_arm_a.pt", ckpt_hash="a" * 64,
            pre_sha="p" * 64, post_sha="p" * 64, reload_ok=True,
            device_count=1, wall=90.0, eval_recovery=False)
        assert receipt["status"] in ("SCIENTIFIC_PASS", "SCIENTIFIC_FAIL")
        assert set(receipt["nulls_per_tier"]) == {"t0", "t1", "t2", "t3", "t4"}
        assert t1d.validate_null_block(receipt) == []
        assert receipt["diagnostics"]["first_test_lift_tier"] == 1  # 0.5 >= 0.20
        assert receipt["diagnostics"]["first_train_lift_tier"] == 1
        json.dumps(receipt)  # serializable
        t1d.write_arm_receipt(tmp, receipt, ckpt_hash="a" * 64)
        assert (Path(tmp) / "ARM_A.json").is_file()
        assert (Path(tmp) / "ARM_A.done.json").is_file()
        out = t1d.classify_cross_arm({"A": receipt})
        assert out["labels"] and "reasons" in out
        # reload identity alone must NEVER create SCIENTIFIC_PASS
        zero = dict(receipt)
        zero["trained"] = {f"t{t}": s(0.0) for t in range(5)}
        zero["trained_recs"] = None  # not re-derived; gate uses counts only
        zero["untrained"] = {f"t{t}": s(0.0) for t in range(5)}
        zero["reload_identical"] = True
        zero2 = t1d.build_arm_receipt(
            tag="B", cfg=dict(t1d.ARMS["B"]), env={"probe_pass": True},
            n_seq=8, length=64, param_count=3_737_472,
            citadel_sha="0" * 40, cymek_sha="1" * 64,
            feeder=feeder, seed=20260904, slices=slices,
            ledgers=[{"updates": 100, "first_loss": 6.0, "last_loss": 9.0}],
            done=100, first_loss=6.0, last_loss=9.0, cap_total=100 * 8 * 64,
            ans_total=1234, whole_total=42_000, gsum=3.0, gmax=0.9, gn=100,
            train_wall=60.0, untrained={f"t{t}": s(0.0) for t in range(5)},
            untrained_train=untrained_train,
            trained={f"t{t}": s(0.0) for t in range(5)},
            trained_recs={f"t{t}": _synthetic_records(slices[f"test_t{t}"]["targets"],
                                                      acc=0.0) for t in range(5)},
            trained_train=trained_train, train_memorization=train_memorization,
            inter=inter, teacher_eval={"skipped": "n/a"}, first_step={"n": 0},
            ckpt_path="t1d_arm_b.pt", ckpt_hash="b" * 64,
            pre_sha="q" * 64, post_sha="q" * 64, reload_ok=True,
            device_count=1, wall=90.0, eval_recovery=False)
        assert zero2["status"] == "SCIENTIFIC_FAIL", zero2["gate_rules"]
        assert zero2["gate_rules"]["reload"] is True
        assert zero2["gate_rules"]["loss"] is False


def _write_synthetic_session(root: Path, *, arm_status="SCIENTIFIC_FAIL"):
    from citadel_tpu import tiered_data as td

    root.mkdir(parents=True, exist_ok=True)
    (root / "CALIBRATION.json").write_text(json.dumps(
        {"schema": "citadel-t1d-throughput-calibration/v1",
         "selected": {"batch": 256, "length": 64},
         "selected_tokens_per_second": 8000.0,
         "candidates": [{"batch": 256, "length": 64, "tokens_per_second": 8000.0,
                         "correct": True}]}, sort_keys=True), encoding="utf-8")
    (root / "DATA_MANIFEST.json").write_text(json.dumps(
        {"schema": "citadel-tiered-manifest/v1",
         "generator_version": td.GENERATOR_VERSION, "total_bytes": 1000,
         "max_row_chars": 64, "leakage": {}}, sort_keys=True), encoding="utf-8")
    arms = {}
    for tag in t1d.ARM_ORDER:
        r = _arm(status=arm_status)
        r["arm"] = tag
        arms[tag] = r
        (root / f"ARM_{tag}.json").write_text(json.dumps(r), encoding="utf-8")
    return arms


def _green_pre50m_runner(root, arm_receipts, *, rt_sha, rate, shape):
    from citadel_tpu import pre50m as p50

    (root / "PRE50M_TARGET.json").write_text(json.dumps(
        {"schema": "citadel-pre50m-target/v1", **p50.PRE50M_TARGET}),
        encoding="utf-8")
    green_smoke = {"status": "PASS", "reload_output_identity": True,
                   "optimizer_resume": {"moments_preserved": True,
                                        "continued_update_ok": True},
                   "grad_norm": {"max": 1.0}, "losses": [9.0, 8.0],
                   "param_mutation": True, "production_transaction": True,
                   "checkpoint_compat": {"compatible": True},
                   "writer_fence_probe": "rejected-as-required",
                   "token_accounting": {"consistent": True}}
    green_data = {"status": "PASS", "capacity_tokens": 4096, "real_tokens": 4000,
                  "loss_bearing_tokens": 900, "padding_tokens": 96,
                  "scheduled_rows": 64}
    (root / "PRE50M_CHECKPOINT_SMOKE.json").write_text(
        json.dumps(green_smoke), encoding="utf-8")
    (root / "PRE50M_DATA_INTERFACE.json").write_text(
        json.dumps(green_data), encoding="utf-8")
    (root / "PRE50M_PACKING.json").write_text(
        json.dumps({"status": "PASS"}), encoding="utf-8")
    (root / "PRE50M_FEASIBILITY.json").write_text(
        json.dumps({"memory": {"SCALE2_7_4M": {"verdict": "FIT"}}}), encoding="utf-8")
    (root / "PRE50M_THROUGHPUT.json").write_text(json.dumps({"curve": {}}),
                                                 encoding="utf-8")
    (root / "DIAGNOSTICS.json").write_text(json.dumps({"arms": {}}),
                                           encoding="utf-8")
    decision = p50.build_decision(
        target={"understood": True, "type": p50.PRE50M_TARGET["type"],
                "value_tokens": p50.PRE50M_TARGET["value_tokens"],
                "parameter_count": None},
        smoke=green_smoke, feasibility={"verdict": "FIT"},
        data_interface=green_data, packing={"status": "PASS"},
        recommended_batch=shape[0], recommended_sequence_length=shape[1],
        rate_tok_s=rate)
    (root / "NEXT_50M_DECISION.json").write_text(json.dumps(decision),
                                                 encoding="utf-8")
    return {"status": "PASS", "decision": decision}


def test_full_session_simulation() -> None:
    """THE §13 test: a deterministic no-TPU session traverses every
    summary/PRE50M/bundle stage and ends BUNDLE VALID — with green PRE50M and
    again with a PRE50M implementation failure (arms preserved, decision
    fail-closed, bundle still valid)."""
    import tempfile

    from citadel_tpu import pre50m as p50

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        arms = _write_synthetic_session(root)
        session = t1d._assemble_session_results(
            root, arms, shape=(256, 64), rate=8000.0, scaled=False,
            budgets={t: t1d.ARMS[t]["budget"] for t in t1d.ARMS},
            rt_sha="1" * 64, pre50m_runner=_green_pre50m_runner)
        assert session["pre50m"]["status"] == "PASS"
        assert session["pre50m"]["decision"]["ready_for_50m_training"] is True
        assert session["bundle"] and session["bundle"] != "pending"
        verdict = t1d.verify_bundle(str(root))
        assert verdict["status"] == "VALID"
        decision = json.loads((root / "NEXT_50M_DECISION.json").read_text())
        assert decision["ready_for_50m_training"] is True

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        arms = _write_synthetic_session(root)

        def failing_runner(*a, **k):
            raise RuntimeError("XLA compile exploded")

        session = t1d._assemble_session_results(
            root, arms, shape=(256, 64), rate=8000.0, scaled=False,
            budgets={t: t1d.ARMS[t]["budget"] for t in t1d.ARMS},
            rt_sha="1" * 64, pre50m_runner=failing_runner)
        assert session["pre50m"]["status"] == "IMPLEMENTATION_FAILURE"
        # every required PRE50M artifact exists as an explicit failure receipt
        for name in ("PRE50M_CHECKPOINT_SMOKE.json", "PRE50M_DATA_INTERFACE.json",
                     "PRE50M_PACKING.json", "DIAGNOSTICS.json"):
            doc = json.loads((root / name).read_text())
            assert doc["status"] == "IMPLEMENTATION_FAILURE", (name, doc)
        decision = json.loads((root / "NEXT_50M_DECISION.json").read_text())
        assert decision["ready_for_50m_training"] is False
        assert any("PRE50M implementation failure" in r
                   for r in decision["blocking_reasons"])
        # the operator still gets a complete, verifiable bundle with all T1D arms
        assert t1d.verify_bundle(str(root))["status"] == "VALID"
        for tag in "ABCDE":
            assert (root / f"ARM_{tag}.json").is_file()


def test_session_partial_arm_failure() -> None:
    """§9: one arm implementation-failed, one timeboxed, three scientific —
    the summary must not crash, classification covers scientific arms only,
    and the curve file represents failed arms by status."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        arms = _write_synthetic_session(root)
        arms["C"] = {"arm": "C", "status": "IMPLEMENTATION_FAILURE",
                     "error": "boom"}
        arms["D"] = {"arm": "D", "status": "TIMEBOX_ABORT", "updates_done": 3}
        session = t1d._assemble_session_results(
            root, arms, shape=(256, 64), rate=8000.0, scaled=False,
            budgets={t: t1d.ARMS[t]["budget"] for t in t1d.ARMS},
            rt_sha="1" * 64, pre50m_runner=_green_pre50m_runner)
        assert session["arms"]["C"] == "IMPLEMENTATION_FAILURE"
        curves = json.loads((root / "LIFT_OFF_CURVES.json").read_text())["arms"]
        assert curves["C"] == {"status": "IMPLEMENTATION_FAILURE"}
        assert curves["D"] == {"status": "TIMEBOX_ABORT"}
        assert "train" in curves["A"] and "test" in curves["A"]
        assert t1d.verify_bundle(str(root))["status"] == "VALID"
        # arm failure receipts are durable terminal states on disk
        receipt = t1d.arm_failure_receipt("C", RuntimeError("boom"),
                                          citadel_sha="0" * 40,
                                          cymek_sha="1" * 64)
        assert receipt["status"] == "IMPLEMENTATION_FAILURE"
        json.dumps(receipt)


def test_budget_limited_classifier_matrix() -> None:
    """BUDGET_LIMITED reads the REAL per-tier DEV schema. Rising DEV at the
    last two checkpoints with low final test fires; flat DEV, declining DEV,
    and high final TEST never fire."""
    base = {t: _arm() for t in "ABCDE"}
    rising = dict(base, B=_arm(dev_pairs=[(75, 0.02), (100, 0.09)]))
    assert "BUDGET_LIMITED" in t1d.classify_cross_arm(rising)["labels"]
    flat = dict(base, B=_arm(dev_pairs=[(75, 0.05), (100, 0.05)]))
    assert "BUDGET_LIMITED" not in t1d.classify_cross_arm(flat)["labels"]
    declining = dict(base, B=_arm(dev_pairs=[(75, 0.09), (100, 0.02)]))
    assert "BUDGET_LIMITED" not in t1d.classify_cross_arm(declining)["labels"]
    high_test = dict(base, B=_arm(test14=(0.5, 0.5, 0.5, 0.5), lcb=0.45,
                                  dev_pairs=[(75, 0.02), (100, 0.09)]))
    assert "BUDGET_LIMITED" not in t1d.classify_cross_arm(high_test)["labels"]


def test_select_calibrated_shape_scale2_guard() -> None:
    """§15: the fastest correctness-passing shape is rejected when it fails
    SCALE2 verification; the selected shape is the next passing one; the
    failed candidate is marked IN PLACE (no duplicate selectable dicts)."""
    results = [
        {"batch": 1024, "length": 64, "tokens_per_second": 9000.0, "correct": True},
        {"batch": 512, "length": 64, "tokens_per_second": 7000.0, "correct": True},
        {"batch": 256, "length": 64, "tokens_per_second": 5000.0, "correct": True},
    ]
    seen = []

    def verifier(batch, length):
        seen.append((batch, length))
        return (batch, length) != (1024, 64)  # fastest fails SCALE2

    best, note = t1d.select_calibrated_shape(results, scale2_verifier=verifier)
    assert note == "pass" and best["batch"] == 512
    assert seen == [(1024, 64), (512, 64)]
    failed = [r for r in results if not r["correct"]]
    assert len(failed) == 1 and failed[0]["batch"] == 1024
    assert failed[0]["error"] == "SCALE2_VERIFICATION_FAILED"
    assert len(results) == 3  # no duplicate dicts appended
    assert all(r["correct"] for r in results if r["batch"] in (512, 256))
    none, note2 = t1d.select_calibrated_shape(
        [{"batch": 1, "length": 64, "tokens_per_second": 1.0, "correct": True}],
        scale2_verifier=lambda b, l: False)
    assert none is None and "safe" in note2
    none2, _ = t1d.select_calibrated_shape([], scale2_verifier=verifier)
    assert none2 is None


def test_pre50m_fail_closed_matrix() -> None:
    """§6: starting from one all-green synthetic receipt, mutating EACH
    required condition must force ready_for_50m_training=false with a
    precise blocking reason (20 independent mutations)."""
    import copy

    from citadel_tpu import pre50m as p50

    green_smoke = {"status": "PASS", "reload_output_identity": True,
                   "optimizer_resume": {"moments_preserved": True,
                                        "continued_update_ok": True},
                   "grad_norm": {"max": 1.0}, "losses": [9.0, 8.0],
                   "param_mutation": True, "production_transaction": True,
                   "checkpoint_compat": {"compatible": True},
                   "writer_fence_probe": "rejected-as-required",
                   "token_accounting": {"consistent": True}}
    base = {"smoke": green_smoke,
            "data_interface": {"status": "PASS", "capacity_tokens": 4096,
                               "real_tokens": 4000, "loss_bearing_tokens": 900,
                               "padding_tokens": 96, "scheduled_rows": 64},
            "target": {"understood": True, "type": p50.PRE50M_TARGET["type"],
                       "value_tokens": p50.PRE50M_TARGET["value_tokens"],
                       "parameter_count": None},
            "feasibility": {"verdict": "FIT"}, "packing": {"status": "PASS"},
            "recommended_batch": 256, "recommended_sequence_length": 64,
            "rate_tok_s": 8000.0}
    assert p50.build_decision(**copy.deepcopy(base))["ready_for_50m_training"] is True
    mutations = [
        ("unknown target", lambda b: b["target"].__setitem__("understood", False)),
        ("wrong target contract", lambda b: b["target"].__setitem__("type", "parameters")),
        ("wrong target value", lambda b: b["target"].__setitem__("value_tokens", 49_000_000)),
        ("model unfit", lambda b: b["feasibility"].__setitem__("verdict", "DOES_NOT_FIT")),
        ("no safe shape", lambda b: b.__setitem__("recommended_batch", 0)),
        ("zero throughput", lambda b: b.__setitem__("rate_tok_s", 0.0)),
        ("nonfinite throughput", lambda b: b.__setitem__("rate_tok_s", float("nan"))),
        ("smoke FAIL", lambda b: b["smoke"].__setitem__("status", "FAIL")),
        ("nonfinite loss", lambda b: b["smoke"].__setitem__("losses", [float("nan")])),
        ("no loss", lambda b: b["smoke"].__setitem__("losses", [])),
        ("zero gradients", lambda b: b["smoke"].__setitem__("grad_norm", {"max": 0.0})),
        ("no param mutation", lambda b: b["smoke"].__setitem__("param_mutation", False)),
        ("no production transaction",
         lambda b: b["smoke"].__setitem__("production_transaction", False)),
        ("checkpoint incompatible",
         lambda b: b["smoke"].__setitem__("checkpoint_compat", {"compatible": False})),
        ("reload mismatch",
         lambda b: b["smoke"].__setitem__("reload_output_identity", False)),
        ("moments not preserved",
         lambda b: b["smoke"]["optimizer_resume"].__setitem__("moments_preserved", False)),
        ("continued update false",
         lambda b: b["smoke"]["optimizer_resume"].__setitem__("continued_update_ok", False)),
        ("writer fence accepts",
         lambda b: b["smoke"].__setitem__("writer_fence_probe", "UNEXPECTED-ACCEPT")),
        ("token accounting inconsistent",
         lambda b: b["smoke"].__setitem__("token_accounting", {"consistent": False})),
        ("data interface FAIL",
         lambda b: b["data_interface"].__setitem__("status", "FAIL")),
        ("capacity<real violation",
         lambda b: b["data_interface"].__setitem__("real_tokens", 9999)),
        ("padding mismatch",
         lambda b: b["data_interface"].__setitem__("padding_tokens", 999)),
        ("negative scheduled",
         lambda b: b["data_interface"].__setitem__("scheduled_rows", -1)),
        ("packing FAIL", lambda b: b["packing"].__setitem__("status", "FAIL")),
    ]
    for name, mutate in mutations:
        bad = copy.deepcopy(base)
        mutate(bad)
        d = p50.build_decision(**bad)
        assert d["ready_for_50m_training"] is False, name
        assert d["blocking_reasons"], name
    # token accounting invariants: capacity >= real >= loss-bearing >= 0
    b = copy.deepcopy(base)
    b["data_interface"]["loss_bearing_tokens"] = 5000  # > real
    d = p50.build_decision(**b)
    assert d["ready_for_50m_training"] is False


def test_verify_bundle_failure_receipts_and_checkpoints() -> None:
    """§14: failure receipts are valid members; unknown statuses rejected;
    bundled checkpoint SHAs are verified against the receipt."""
    import hashlib as _hl
    import tempfile
    import zipfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        arms = _write_synthetic_session(root)
        _green_pre50m_runner(root, arms, rt_sha="1" * 64, rate=8000.0,
                             shape=(256, 64))
        session = t1d._assemble_session_results(
            root, arms, shape=(256, 64), rate=8000.0, scaled=False,
            budgets={t: t1d.ARMS[t]["budget"] for t in t1d.ARMS},
            rt_sha="1" * 64, pre50m_runner=_green_pre50m_runner)
        assert t1d.verify_bundle(str(root))["status"] == "VALID"
        # PRE50M failure placeholders are KNOWN statuses -> still valid
        (root / "PRE50M_CHECKPOINT_SMOKE.json").write_text(json.dumps(
            {"status": "IMPLEMENTATION_FAILURE", "phase": "smoke",
             "error": "x", "citadel_sha": "0" * 40, "cymek_runtime_sha": "1" * 64}),
            encoding="utf-8")
        with zipfile.ZipFile(root / "CITADEL_T1D_RESULTS.zip", "w",
                             zipfile.ZIP_DEFLATED) as zf:
            for name in t1d.BUNDLE_FILES:
                zf.write(root / name, name)
        assert t1d.verify_bundle(str(root))["status"] == "VALID"
        # unknown status anywhere -> invalid
        (root / "PRE50M_PACKING.json").write_text(json.dumps({"status": "WEIRD"}),
                                                  encoding="utf-8")
        with zipfile.ZipFile(root / "CITADEL_T1D_RESULTS.zip", "w",
                             zipfile.ZIP_DEFLATED) as zf:
            for name in t1d.BUNDLE_FILES:
                zf.write(root / name, name)
        try:
            t1d.verify_bundle(str(root))
            raise SystemExit("unknown PRE50M status accepted")
        except RuntimeError as exc:
            assert "WEIRD" in str(exc), exc
        (root / "PRE50M_PACKING.json").write_text(json.dumps({"status": "PASS"}),
                                                  encoding="utf-8")
        # bundled checkpoint sha mismatch -> invalid
        payload = b"not-a-real-checkpoint"
        (root / "t1d_arm_a.pt").write_bytes(payload)
        arm_a = json.loads((root / "ARM_A.json").read_text())
        arm_a["checkpoint"] = {"path": "t1d_arm_a.pt",
                               "sha256": _hl.sha256(payload).hexdigest()}
        (root / "ARM_A.json").write_text(json.dumps(arm_a), encoding="utf-8")
        with zipfile.ZipFile(root / "CITADEL_T1D_RESULTS.zip", "w",
                             zipfile.ZIP_DEFLATED) as zf:
            for name in t1d.BUNDLE_FILES:
                zf.write(root / name, name)
            zf.writestr("t1d_arm_a.pt", payload)
        assert t1d.verify_bundle(str(root))["status"] == "VALID"
        arm_a["checkpoint"]["sha256"] = "0" * 64
        (root / "ARM_A.json").write_text(json.dumps(arm_a), encoding="utf-8")
        with zipfile.ZipFile(root / "CITADEL_T1D_RESULTS.zip", "w",
                             zipfile.ZIP_DEFLATED) as zf:
            for name in t1d.BUNDLE_FILES:
                zf.write(root / name, name)
            zf.writestr("t1d_arm_a.pt", payload)
        try:
            t1d.verify_bundle(str(root))
            raise SystemExit("checkpoint sha mismatch accepted")
        except RuntimeError as exc:
            assert "sha mismatch" in str(exc), exc


def _legacy_untrained_producer():
    """The EXACT pre-fix run_arm producer shape for the untrained block:
    dev_tN / test_tN keys in one dict (the shape that produced the real TPU
    failure KeyError: 't1' when the finalizer read tN)."""
    def s(acc, n=500):
        return {"correct": int(acc * n), "total": n, "accuracy": acc,
                "wilson_lcb": max(0.0, acc - 0.04),
                "wilson_ucb": min(1.0, acc + 0.04)}
    out = {}
    for tier in range(5):
        out[f"dev_t{tier}"] = s(0.0, n=200)
        out[f"test_t{tier}"] = s(0.0)
    return out


def test_exact_keyerror_t1_regression() -> None:
    """Reproduces the REAL TPU failure of 2026-09-05: run_arm stored untrained
    results as dev_tN/test_tN while build_arm_receipt read tN — the scientific
    gate died on KeyError: 't1' AFTER expensive training. The fix chain:
    legacy producer shape -> normalize_untrained_receipt -> canonical t0-t4
    -> build_arm_receipt -> classify_cross_arm -> build_lift_curves,
    with NO KeyError anywhere."""
    legacy = _legacy_untrained_producer()
    # document the original failure mode precisely
    try:
        legacy["t1"]
        raise SystemExit("legacy producer unexpectedly has canonical keys")
    except KeyError as exc:
        assert str(exc) == "'t1'", exc
    # the normalizer bridges producer -> canonical deterministically
    canonical = t1d.normalize_untrained_receipt(legacy)
    assert set(canonical) == {"t0", "t1", "t2", "t3", "t4"}
    assert canonical["t1"] is legacy["test_t1"]
    assert canonical["t0"] is legacy["test_t0"]
    # canonical input passes through untouched (identity of mapping, not dict)
    assert t1d.normalize_untrained_receipt(canonical) == canonical
    # and the FULL bridge on the legacy shape is defect-free
    assert t1d.producer_consumer_contract_probe(
        legacy_untrained_keys=True) == []


def test_normalize_untrained_receipt_validation() -> None:
    """ARM_SCHEMA_INVALID — never a raw KeyError: missing both forms, partial
    legacy form, and out-of-contract summaries all fail loudly."""
    try:
        t1d.normalize_untrained_receipt({"untrained": "garbage"})
        raise SystemExit("unrecognized shape accepted")
    except RuntimeError as exc:
        assert "ARM_SCHEMA_INVALID" in str(exc), exc
    partial = _legacy_untrained_producer()
    del partial["test_t2"]
    try:
        t1d.normalize_untrained_receipt(partial)
        raise SystemExit("partial legacy form accepted")
    except RuntimeError as exc:
        assert "ARM_SCHEMA_INVALID" in str(exc) and "test_t2" in str(exc), exc
    bad = {"t0": {"correct": 1, "total": -5, "accuracy": 2.0,
                  "wilson_lcb": 0.0, "wilson_ucb": 0.0}}
    for t in range(1, 5):
        bad[f"t{t}"] = {"correct": 0, "total": 10, "accuracy": 0.0,
                        "wilson_lcb": 0.0, "wilson_ucb": 0.0}
    try:
        t1d.normalize_untrained_receipt(bad)
        raise SystemExit("out-of-contract summary accepted")
    except RuntimeError as exc:
        assert "ARM_SCHEMA_INVALID" in str(exc) and "total" in str(exc), exc


def test_producer_consumer_contract() -> None:
    """§4: one bridge over the REAL producer-shaped data proves every key
    consumed downstream (build_arm_receipt, validate_arm_receipt,
    classify_cross_arm, build_lift_curves) exists — for BOTH the legacy
    producer keying and the canonical producer keying."""
    assert t1d.producer_consumer_contract_probe(
        legacy_untrained_keys=True) == []
    assert t1d.producer_consumer_contract_probe(
        legacy_untrained_keys=False) == []


def test_prefinal_recovery_simulation() -> None:
    """§5: an expensive arm that dies in the PURE finalizer is never lost or
    retrained. Snapshot write -> hash-verified load -> finalization-only
    rerun -> receipt + marker + consumed sidecar; corrupt snapshot, missing
    checkpoint, and cross-run mismatch are all refused; a finalizer
    exception RETAINS the sidecar."""
    import hashlib as _hl
    import tempfile

    from citadel_tpu import calculator_eval as cev

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        feeder = t1d.TierFeeder("curriculum", 8, 64)
        for u in range(400):
            feeder.fill_sequences(u / 400)
        slices = t1d._tier_slices()
        plan = t1d.train_memorization_plan(feeder)
        plan = t1d.train_memorization_plan(feeder, candidates=None)

        def s(acc, n=500):
            return {"correct": int(acc * n), "total": n, "accuracy": acc,
                    "wilson_lcb": max(0.0, acc - 0.04),
                    "wilson_ucb": min(1.0, acc + 0.04)}

        untrained = _legacy_untrained_producer()  # producer-shaped, as run_arm did pre-fix
        untrained_dev = {f"t{t}": untrained[f"dev_t{t}"] for t in range(5)}
        accs = {0: 0.9, 1: 0.4, 2: 0.2, 3: 0.05, 4: 0.0}
        kwargs = dict(
            tag="A", cfg=dict(t1d.ARMS["A"]), env={"probe_pass": True},
            n_seq=8, length=64, param_count=3_737_472,
            citadel_sha="0" * 40, cymek_sha="1" * 64, seed=20260904,
            feeder_placed_rows={k: int(v) for k, v in feeder.placed_rows.items()},
            feeder_ledger=feeder.ledger(),
            ledgers=[{"updates": 100, "first_loss": 9.0, "last_loss": 6.0}],
            done=100, first_loss=9.0, last_loss=6.0, cap_total=100 * 8 * 64,
            ans_total=1234, whole_total=42_000, gsum=3.0, gmax=0.9, gn=100,
            train_wall=60.0, untrained=untrained, untrained_dev=untrained_dev,
            untrained_self={"correct": 0, "total": 96, "accuracy": 0.0,
                            "wilson_lcb": 0.0, "wilson_ucb": 0.04},
            trained_self={"correct": 0, "total": 96, "accuracy": 0.0,
                          "wilson_lcb": 0.0, "wilson_ucb": 0.04},
            self_diagnostics={"most_common_null": {"wilson_lcb": 0.0},
                              "per_domain": {}},
            untrained_train={f"t{t}": s(0.0, n=200) for t in range(5)},
            trained={f"t{t}": s(accs[t]) for t in range(5)},
            trained_recs={f"t{t}": [
                {"prompt": "p", "target": tg, "prediction": tg, "correct": True,
                 "stop_reason": "EOS", "generated_token_count": len(tg),
                 "valid": True} for tg in slices[f"test_t{t}"]["targets"]]
                for t in range(5)},
            trained_train={f"t{t}": s(0.6, n=200) for t in range(5)},
            train_memorization={f"t{t}": {
                "consumed_prefix": plan[t]["consumed_prefix"],
                "n_frozen_candidates": plan[t]["n_candidates"],
                "n_verified_consumed": plan[t]["n_verified"],
                "evaluated_rows": plan[t]["n_verified"],
                "status": plan[t]["status"],
                "lift_eligible": plan[t]["status"] == "OK"} for t in range(5)},
            inter={str(cp): {f"t{tier}": {"exact": dev, "lcb": max(0.0, dev - 0.03)}
                             for tier in range(5)}
                   for cp, dev in ((25, 0.05), (50, 0.12), (75, 0.18), (100, 0.2))},
            teacher_eval={"skipped": "n/a"}, first_step={"n": 0},
            ckpt_path=str(root / "t1d_arm_a.pt"), ckpt_hash="",
            pre_sha="p" * 64, post_sha="p" * 64, reload_ok=True,
            device_count=1, wall=90.0, eval_recovery=False)
        # a real checkpoint file whose hash matches the snapshot
        ckpt_bytes = b"synthetic-checkpoint-bytes"
        (root / "t1d_arm_a.pt").write_bytes(ckpt_bytes)
        kwargs["ckpt_hash"] = _hl.sha256(ckpt_bytes).hexdigest()
        sidecar = Path(t1d.write_prefinal_snapshot(root, kwargs))
        assert sidecar.is_file()
        # finalization-only recovery: hash-verified load, NO training, NO device
        snap, why = t1d.load_prefinal_snapshot(root, "A", expect_cfg=dict(t1d.ARMS["A"]),
                                               seed=20260904, shape=(8, 64))
        assert snap is not None, why
        receipt = t1d.build_arm_receipt(**snap)
        t1d.write_arm_receipt(root, receipt, ckpt_hash=snap["ckpt_hash"])
        sidecar.unlink()  # consumed on success (run_arm does this)
        assert (root / "ARM_A.json").is_file()
        assert (root / "ARM_A.done.json").is_file()
        assert t1d.validate_arm_receipt(receipt) == []
        # corrupt payload is refused and archived, never trusted
        doc = json.loads(sidecar.read_text()) if False else None
        kwargs2 = dict(kwargs)
        kwargs2["done"] = 999  # tamper AFTER hashing
        sidecar2 = Path(t1d.write_prefinal_snapshot(root, kwargs2))
        doc2 = json.loads(sidecar2.read_text(encoding="utf-8"))
        doc2["done"] = 1
        sidecar2.write_text(json.dumps(doc2), encoding="utf-8")
        snap2, why2 = t1d.load_prefinal_snapshot(root, "A",
                                                 expect_cfg=dict(t1d.ARMS["A"]),
                                                 seed=20260904, shape=(8, 64))
        assert snap2 is None and "hash mismatch" in why2
        assert not sidecar2.is_file()  # archived aside
        # missing checkpoint file is refused
        kwargs3 = dict(kwargs)
        kwargs3["ckpt_path"] = str(root / "missing.pt")
        t1d.write_prefinal_snapshot(root, kwargs3)
        snap3, why3 = t1d.load_prefinal_snapshot(root, "A",
                                                 expect_cfg=dict(t1d.ARMS["A"]),
                                                 seed=20260904, shape=(8, 64))
        assert snap3 is None and "missing" in why3
        # cross-run mismatch (different seed) is refused
        t1d.write_prefinal_snapshot(root, kwargs)
        snap4, why4 = t1d.load_prefinal_snapshot(root, "A",
                                                 expect_cfg=dict(t1d.ARMS["A"]),
                                                 seed=999, shape=(8, 64))
        assert snap4 is None and "mismatch" in why4
        # a finalizer exception RETAINS the sidecar (finalization retried,
        # never retrained): invalid untrained block -> ARM_SCHEMA_INVALID
        broken = dict(kwargs)
        broken["untrained"] = {"garbage": True}
        t1d.write_prefinal_snapshot(root, broken)
        snap5, _ = t1d.load_prefinal_snapshot(root, "A",
                                              expect_cfg=dict(t1d.ARMS["A"]),
                                              seed=20260904, shape=(8, 64))
        try:
            t1d.build_arm_receipt(**snap5)
            raise SystemExit("invalid untrained block finalized silently")
        except RuntimeError as exc:
            assert "ARM_SCHEMA_INVALID" in str(exc), exc
        assert (root / "ARM_A.prefinal.json").is_file()  # retained for retry


def test_should_skip_arm_matrix() -> None:
    """§8: IMPLEMENTATION_FAILURE is NEVER a completed arm — with or without
    a marker it must rerun; scientific and timebox completions skip; unknown
    statuses and marker-without-receipt fail loudly."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        assert t1d.should_skip_arm(tmp, "A") == ("run", "nothing complete")
        receipt = Path(tmp) / "ARM_A.json"
        marker = Path(tmp) / "ARM_A.done.json"
        receipt.write_text(json.dumps({"status": "IMPLEMENTATION_FAILURE",
                                       "arm": "A"}), encoding="utf-8")
        assert t1d.should_skip_arm(tmp, "A")[0] == "run"
        marker.write_text("{}", encoding="utf-8")
        decision, why = t1d.should_skip_arm(tmp, "A")
        assert decision == "run" and "not completion" in why
        for status in ("SCIENTIFIC_PASS", "SCIENTIFIC_FAIL", "TIMEBOX_ABORT"):
            receipt.write_text(json.dumps({"status": status, "arm": "A"}),
                               encoding="utf-8")
            assert t1d.should_skip_arm(tmp, "A") == (
                "skip", f"valid {status} receipt+marker present")
        receipt.write_text(json.dumps({"status": "MYSTERY"}), encoding="utf-8")
        assert t1d.should_skip_arm(tmp, "A")[0] == "raise"
        receipt.unlink()
        assert t1d.should_skip_arm(tmp, "A")[0] == "raise"


def test_validate_arm_receipt_contract() -> None:
    """§11: the terminal validator enforces the full scientific contract and
    verify_bundle rejects malformed scientific receipts."""
    import tempfile
    import zipfile

    probe_ok = t1d.producer_consumer_contract_probe(
        legacy_untrained_keys=True) == []
    assert probe_ok
    # a malformed scientific receipt is caught by the validator directly
    arms = _write_synthetic_session(Path(tempfile.mkdtemp()))
    bad = arms["A"]
    del bad["trained"]["t3"]
    defects = t1d.validate_arm_receipt(bad)
    assert any("trained" in d and "t0-t4" in d for d in defects), defects
    bad2 = dict(arms["B"])
    bad2["gate_rules"] = {}
    assert any("gate_rules" in d for d in t1d.validate_arm_receipt(bad2))
    # IMPLEMENTATION_FAILURE receipts carry partial data by design
    failure = {"status": "IMPLEMENTATION_FAILURE", "arm": "X", "error": "e"}
    assert t1d.validate_arm_receipt(failure) == []
    # a malformed scientific receipt is DEMOTED at the session boundary (the
    # classifier never sees it -> no classifier KeyError is possible) and the
    # bundle stays valid
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        arms = _write_synthetic_session(root)
        bad_arm = arms["A"]
        del bad_arm["trained"]["t3"]
        (root / "ARM_A.json").write_text(json.dumps(bad_arm), encoding="utf-8")
        _green_pre50m_runner(root, arms, rt_sha="1" * 64, rate=8000.0,
                             shape=(256, 64))
        session = t1d._assemble_session_results(
            root, arms, shape=(256, 64), rate=8000.0, scaled=False,
            budgets={t: t1d.ARMS[t]["budget"] for t in t1d.ARMS},
            rt_sha="1" * 64, pre50m_runner=_green_pre50m_runner)
        assert session["arms"]["A"] == "IMPLEMENTATION_FAILURE"
        on_disk = json.loads((root / "ARM_A.json").read_text())
        assert on_disk["status"] == "IMPLEMENTATION_FAILURE"
        assert on_disk["schema_defects"], "defects must be recorded on disk"
        assert t1d.verify_bundle(str(root))["status"] == "VALID"
        # a malformed scientific receipt that somehow reached a bundle is
        # still rejected by verify_bundle itself
        (root / "ARM_A.json").write_text(json.dumps(bad_arm), encoding="utf-8")
        import zipfile as _zf

        with _zf.ZipFile(root / "CITADEL_T1D_RESULTS.zip", "w",
                         _zf.ZIP_DEFLATED) as zf:
            for name in t1d.BUNDLE_FILES:
                zf.write(root / name, name)
        try:
            t1d.verify_bundle(str(root))
            raise SystemExit("malformed scientific receipt accepted by verify_bundle")
        except RuntimeError as exc:
            assert "ARM_A.json" in str(exc) and "t0-t4" in str(exc), exc


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
             test_session_resume_dry_run,
             test_train_memorization_plan_matrix,
             test_lift_eligibility_never_fires_below_min_n,
             test_null_block_validation,
             test_post_training_arm_simulation,
             test_full_session_simulation,
             test_session_partial_arm_failure,
             test_budget_limited_classifier_matrix,
             test_select_calibrated_shape_scale2_guard,
             test_pre50m_fail_closed_matrix,
             test_verify_bundle_failure_receipts_and_checkpoints,
             test_exact_keyerror_t1_regression,
             test_normalize_untrained_receipt_validation,
             test_producer_consumer_contract,
             test_prefinal_recovery_simulation,
             test_should_skip_arm_matrix,
             test_validate_arm_receipt_contract]
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
