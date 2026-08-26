"""Debug-sweep invariants: recompute headline numbers from raw rows.

Permanent guard for the MIXED-CAUSAL evidence chain. Any future edit that
breaks a receipt's internal consistency turns CI red.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parents[1]


def _j(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


def test_mc8_adaptive_and_baselines_recompute_from_rows() -> None:
    r = _j("output/mixed_causal_v8_confirmation.json")
    rows = r["per_task_rows"]
    succ = sum(1 for x in rows
               if x["actions"][x["adaptive_action"]]["pass"])
    assert succ == r["policies"]["ADAPTIVE_v7"]["succ"] == 291
    # constants fall back to NO_CHANGE where inapplicable
    for arm, name in (("NO_CHANGE", "ALWAYS_NO_CHANGE"),
                      ("CONSTRAINED", "ALWAYS_CONSTRAINED"),
                      ("NORMALIZED", "ALWAYS_NORMALIZED")):
        s = 0
        for x in rows:
            if arm in x["actions"]:
                s += x["actions"][arm]["pass"]
            else:
                s += x["actions"]["NO_CHANGE"]["pass"]
        assert s == r["policies"][name]["succ"], (name, s)
    oracle = sum(1 for x in rows
                 if any(e["pass"] for e in x["actions"].values()))
    assert oracle == r["oracle_successes"] == 307


def test_mc8_paired_stats_recompute() -> None:
    """Paired deltas must match the stored values exactly."""
    r = _j("output/mixed_causal_v8_confirmation.json")
    rows = r["per_task_rows"]
    n = len(rows)
    adaptive_pass = [x["actions"][x["adaptive_action"]]["pass"] for x in rows]

    def chooser(name, x):
        a = x["actions"]
        return {"ALWAYS_NO_CHANGE": "NO_CHANGE",
                "ALWAYS_CONSTRAINED": "CONSTRAINED" if "CONSTRAINED" in a else "NO_CHANGE",
                "ALWAYS_NORMALIZED": "NORMALIZED" if "NORMALIZED" in a else "NO_CHANGE",
                }[name]

    for name in ("ALWAYS_NO_CHANGE", "ALWAYS_CONSTRAINED",
                 "ALWAYS_NORMALIZED"):
        diffs = [ap - (1 if x["actions"][chooser(name, x)]["pass"] else 0)
                 for ap, x in zip(adaptive_pass, rows)]
        mean_d = sum(diffs) / n
        stored = r["paired_adaptive_vs_others"][name]["mean_diff"]
        # receipts round to 4dp; allow rounding tolerance
        assert abs(mean_d - stored) < 5e-5, (name, mean_d, stored)


def test_identity_chain_across_artifacts() -> None:
    CKPT_SFT6 = "36c87e5be671ff37951b4b433e21ec14bb5e59f6f0770d7f81e50e99ead9e001"
    pol7 = _j("output/self_model_v7.json")
    v7 = _j("output/mixed_causal_v7_verdict.json")
    v8 = _j("output/mixed_causal_v8_confirmation.json")
    v9v = _j("output/mixed_causal_v9_verdict.json")

    # policy SHA consistent everywhere it appears
    sha7 = pol7["parameter_sha256"]
    assert v7["policy_sha256"] == sha7
    assert v8["policy_sha256"] == sha7
    assert v9v["policy_sha256"] == sha7

    # SFT6 runs carry the SFT6 checkpoint; the transfer run carries SFT7's
    harvest_ckpt = _j("output/harvest_v7_pool.json")["checkpoint_sha256"]
    assert harvest_ckpt == CKPT_SFT6
    assert v8["checkpoint_sha256"] == CKPT_SFT6
    # transfer verdict ran a DIFFERENT checkpoint than training
    assert v9v["checkpoint_sha256"] != CKPT_SFT6

    # VIE bank size matches qualification receipt and manifest
    bank_file = ROOT / "data/experience_bank/experiences.jsonl"
    n_bank = len([l for l in
                  bank_file.read_text(encoding="utf-8").splitlines()
                  if l.strip() and not l.lstrip().startswith("#")])
    q = _j("output/vie_qualification_mc8.json")
    assert q["bank_total_after"] == n_bank == 166


def test_fixture_hashes_match_live_modules() -> None:
    import sys
    sys.path.insert(0, str(ROOT))
    import connector.experiments.mixed_causal_v7 as mc7
    import connector.experiments.mixed_causal_v8 as mc8
    import connector.experiments.mixed_causal_v9 as mc9

    r7 = _j("output/mixed_causal_v7_replication.json")
    r8 = _j("output/mixed_causal_v8_confirmation.json")
    r9 = _j("output/mixed_causal_v9_transfer.json")
    assert mc7.fixture_hash() == r7["fixture_sha256"]
    assert mc8.fixture_hash() == r8["fixture_sha256"]
    assert mc9.fixture_hash() == r9["fixture_sha256"]


def test_vie_audit_honesty_guard() -> None:
    """The MC-v7 audit must still show 0 qualified with the retention gap —
    history may not be laundered after MC-v8 fixed it."""
    audit = _j("output/vie_audit_mc7.json")
    assert audit["contract_qualifying_flips"] == 0
    assert "retained" in audit["blocking_gap"]
