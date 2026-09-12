"""CITADEL-EVAL-001 attack-suite tests.

Run:  python tests/test_citadel_eval_attack.py   (exit 0 = all pass)

Verifies mechanically, against real deterministic tiered-arithmetic rows:
  1. the latest-position shortcut scores 1.000 on EVERY tier (the CRITICAL
     finding in docs/citadel/evaluation/SHORTCUT_ATTACKS.md),
  2. the reference solver (gold computed from operands) is self-consistent
     at 1.000 — the corpus gold is machine-derived, not hand-written,
  3. trivial constant/mode attacks score near zero (no constant-label leak),
  4. the corpus is causally sensitive to an operation swap (answers move),
  5. firewall: the corpus generator imports no eval/scoring module,
     the contamination gate fails closed on a planted leak,
     the tamper digest flips when any gate result changes,
  6. the tool's own readiness verdict is fail-closed (NOT_READY with blockers
     while the shortcut exists).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import citadel_eval_attack as eva  # noqa: E402

N = 120  # rows per tier for the suite (small, deterministic, sufficient)


def test_latest_position_shortcut_scores_one_on_every_tier() -> None:
    for tier in range(5):
        s = eva.attack_tier(tier, N)
        assert s["scores"]["latest_position"] == 1.0, (
            f"t{tier}: latest_position={s['scores']['latest_position']}; "
            "SHORTCUT_ATTACKS.md finding no longer reproduces")


def test_reference_solver_is_self_consistent() -> None:
    for tier in range(5):
        s = eva.attack_tier(tier, N)
        assert s["scores"]["reference_solver"] == 1.0, (
            f"t{tier}: gold disagrees with operands — corpus gold broken")


def test_constant_and_mode_attacks_stay_low() -> None:
    for tier in range(5):
        s = eva.attack_tier(tier, N)["scores"]
        assert s["constant_zero"] <= 0.10, f"t{tier}: constant_zero leak {s['constant_zero']}"
        assert s["most_common_answer"] <= 0.10, (
            f"t{tier}: most_common leak {s['most_common_answer']}")


def test_corpus_is_operation_swap_sensitive() -> None:
    # T1 operands are 0..9, so b=0 rows collide (a+b == a-0) structurally —
    # T1 is a labeled memorization-probe tier. T2+ bands exclude b=0-ish
    # collisions, so the answers must move under the swap >= 90% of the time.
    floors = {1: 0.85, 2: 0.90, 3: 0.90, 4: 0.90}
    for tier in range(1, 5):
        c = eva.check_operation_swap_sensitivity(tier, N)
        assert c["addsub_rows"] > 0, f"t{tier}: no add/sub rows sampled"
        assert c["sensitivity"] >= floors[tier], (
            f"t{tier}: swap sensitivity {c['sensitivity']} < {floors[tier]} — "
            "corpus cannot distinguish operators (data-level causal insensitivity)")


def test_gold_self_consistency_check_reports_no_misses() -> None:
    for tier in range(5):
        g = eva.check_gold_self_consistency(tier, N)
        assert g["self_consistent"], f"t{tier}: {g['misses']} gold misses"


def test_firewall_generator_isolation_passes() -> None:
    c = eva.check_generator_isolation()
    assert c["pass"], f"generator isolation violated: {c['violations']}"


def test_firewall_contamination_gate_fails_closed() -> None:
    c = eva.check_contamination_gate_fails_closed()
    assert c["pass"], f"contamination gate not fail-closed: {c}"


def test_firewall_tamper_digest_flips() -> None:
    c = eva.check_tamper_digest()
    assert c["pass"], "receipt digest failed to detect tampering"


def test_readiness_verdict_is_fail_closed() -> None:
    attacks = eva.run_shortcut_attacks(N)
    causal = eva.run_causal_checks(N)
    firewall = eva.run_firewall_checks()
    r = eva.build_readiness(attacks, causal, firewall)
    assert r["verdict"] == "NOT_READY"
    assert r["blockers"], "no blockers recorded despite open shortcut"
    assert any("LATEST_POSITION_SHORTCUT" in b for b in r["blockers"])
    assert r["causal"]["gold_self_consistent"] is True
    assert firewall["all_pass"] is True


def test_full_tool_runs_and_reports_ok() -> None:
    rc = eva.main(["--n", "60"])
    assert rc == 0, "tool reported BROKEN_MECHANISM"


def test_results_artifact_matches_docs_findings() -> None:
    """The committed attack-results artifact must agree with SHORTCUT_ATTACKS.md:
    latest_position 1.000 everywhere, first_position 0.93 (T0) / 0.107 (T1)."""
    path = ROOT / "docs" / "citadel" / "evaluation" / "eval_attack_results.json"
    assert path.is_file(), "eval_attack_results.json missing"
    r = json.loads(path.read_text(encoding="utf-8"))
    per_tier = {t["tier"]: t["scores"] for t in r["attacks"]["per_tier"]}
    assert all(per_tier[t]["latest_position"] == 1.0 for t in per_tier)
    assert abs(per_tier[0]["first_position"] - 0.93) < 1e-6
    assert abs(per_tier[1]["first_position"] - 0.1067) < 1e-3
    assert r["firewall"]["all_pass"] is True
    assert r["readiness"]["verdict"] == "NOT_READY"


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
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
