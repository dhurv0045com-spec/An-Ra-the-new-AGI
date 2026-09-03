"""Readiness v2 static decision tests (Mission 28/29).

Hand-calculated synthetic fixtures; NO model execution, NO checkpoint I/O.
TESTS_WRITTEN_NOT_EXECUTED_THIS_RUN (local compute prohibited this session).

Cases:
  A raw=0/20, oracle weak        -> NOT_READY (INSUFFICIENT)
  B raw=30/100, legal=55, oracle=85, replicated -> READY_SCOPED (qualify)
  C raw=100/100                  -> NOT_IDENTIFIABLE / CEILING_LIMITED
  D raw=10/100, legal=11, oracle=80 -> ORACLE_ELICITABLE_ONLY, NOT_READY
  E N=12, 1/12, unstable frontier -> CALIBRATION_REQUIRED, never READY
"""

import os
import sys
from pathlib import Path

_XF = Path(__file__).resolve().parents[1]
if str(_XF) not in sys.path:
    sys.path.insert(0, str(_XF))

from checkpoint_identity import (
    CheckpointNotFound,
    UnsupportedArchitecture,
    match_architecture_profile,
    resolve_checkpoint,
)
from execution_policy import LocalComputeForbidden, assert_local_compute_allowed
from observed import make_visible
from readiness.pipeline import check_replication, e5_dup, e5_sham, e7_sel
from readiness.canaries import canary_rule
from readiness.frontier import check_frontier, spearman
from readiness.readiness_v2 import (
    decide_readiness,
    legal_headroom,
    power_gate,
    response_diversity,
    x0_permitted,
    x1_permitted,
)
from readiness.status import (
    assess_identifiability,
    chance_report,
    classify_capability,
    wilson,
)

try:
    from qualify_checkpoint import _subject_allowed
except Exception:  # keep suite import-safe if CLI deps absent
    _subject_allowed = None


def test_wilson_hand_values():
    lo, hi = wilson(0, 20)
    assert lo == 0.0 and 0.16 < hi < 0.165
    lo, hi = wilson(1, 12)  # the v1 "8.3%": actually 1.5%-35%
    assert 0.01 < lo < 0.02 and 0.34 < hi < 0.37
    lo, hi = wilson(95, 100)
    assert 0.88 < lo < 0.90 and 0.97 < hi < 0.985
    assert wilson(3, 0) == (0.0, 1.0)


def test_chance_adjusted_reporting():
    r = chance_report(1, 8, 0.25)  # QV-lite style: 12.5% vs 25% chance
    assert r["diff_from_chance"] < 0
    assert r["chance"] == 0.25 and r["n"] == 8


def test_case_A_insufficient():
    cap = classify_capability(20, 0, 0.10, None, 0.25, None)
    assert cap["capability"] == "INSUFFICIENT"
    ident = assess_identifiability(20, 20, 2, 0.10, 0.25)
    assert ident["identifiability"] in ("NOT_IDENTIFIABLE", "MARGINAL")
    d = decide_readiness("calibrate", cap, ident, True, "STABLE", None,
                         None, None, "proto", False, 20)
    assert d["readiness"] in ("NOT_READY", "CALIBRATION_REQUIRED")


def test_case_B_ready_scoped():
    cap = classify_capability(100, 30, 0.85, 0.55, 0.25, True)
    assert cap["capability"] == "PARTIAL"
    ident = assess_identifiability(100, 70, 30, 0.85, 0.25)
    assert ident["identifiability"] == "IDENTIFIABLE"
    d = decide_readiness("qualify", cap, ident, True, "STABLE", 0.25,
                         "ADEQUATE", "SUFFICIENT", "proto-sha", True, 100,
                         qv_lite_vs_chance=0.05)
    assert d["readiness"] == "READY_SCOPED"


def test_case_C_ceiling():
    ident = assess_identifiability(100, 0, 0, 1.0, 0.25)
    assert ident["flags"]["raw_ceiling"] is True
    assert ident["identifiability"] == "NOT_IDENTIFIABLE"


def test_case_D_oracle_only():
    lh = legal_headroom(0.10, 0.11, 0.80)
    assert lh["legal_gap"] == 0.01 and lh["oracle_gap"] == 0.69
    assert legal_headroom(0.10, 0.10, 0.45)["status"] == "ORACLE_ELICITABLE_ONLY"
    cap = classify_capability(100, 10, 0.45, 0.11, 0.25, True)
    assert cap["capability"] == "PARTIAL"
    ident = assess_identifiability(100, 90, 35, 0.45, 0.25)
    assert ident["identifiability"] == "IDENTIFIABLE"
    d = decide_readiness("qualify", cap, ident, True, "STABLE", 0.01,
                         "ADEQUATE", "SUFFICIENT", "proto-sha", True, 100)
    assert d["readiness"] == "NOT_READY" and "ORACLE" in d["reason"]


def test_case_E_smallN_never_ready():
    cap = classify_capability(12, 1, 0.583, None, 0.25, None)
    assert cap["capability"] in ("WEAK", "INSUFFICIENT")
    d = decide_readiness("calibrate", cap,
                         assess_identifiability(12, 11, 6, 0.583, 0.25),
                         None, "CALIBRATION_UNSTABLE", None, None, None,
                         "proto", False, 12)
    assert d["readiness"] == "CALIBRATION_REQUIRED"
    d2 = decide_readiness("calibrate", cap,
                          assess_identifiability(12, 11, 6, 0.583, 0.25),
                          True, "STABLE", 0.40, "ADEQUATE", "SUFFICIENT",
                          "proto", True, 12)
    assert d2["readiness"] == "NOT_READY"  # calibrate WEAK never promotes


def test_v1_pilot_frontier_unstable():
    stats = [{"rung": r, "raw_k": k, "n": 12} for r, k in
             (("B0", 0), ("B1", 3), ("B2", 1), ("B3", 1))]
    f = check_frontier(stats)
    assert f["verdict"] == "CALIBRATION_UNSTABLE"
    # B0(0%) -> B1(25%) rises against nominal difficulty: the canary anomaly
    assert any(i["pair"] == ["B0", "B1"] and i["kind"] == "CALIBRATION_NOISE"
               for i in f["inversions"])


def test_spearman_direction():
    assert abs(spearman([1, 2, 3, 4], [1, 4, 2, 3]) - 0.4) < 1e-9
    assert spearman([1, 2, 3], [3, 2, 1]) == -1.0
    assert round(spearman([1, 2, 3], [1, 1, 3]), 4) == 0.8660  # tie-averaged


def test_canary_failure_caps():
    r = canary_rule({"P0": {"k": 0, "n": 12}, "P1": {"k": 1, "n": 12},
                     "P2": {"k": 12, "n": 12}, "P3": {"k": 12, "n": 12},
                     "P4": {"k": 2, "n": 12}})
    assert r["primitive_canary_failed"] is True
    cap = classify_capability(12, 1, 0.583, None, 0.25, False)
    assert cap["capability"] == "WEAK"
    assert any("PRIMITIVE_CANARY_FAILED" in n for n in cap["notes"])


def test_x0_x1_permission():
    ok = x0_permitted("PARTIAL", True, 0.15, "ADEQUATE", 60, True)
    assert ok["permitted"] is True
    assert x0_permitted("PARTIAL", True, 0.15, "ADEQUATE", 20, True)["permitted"] is False
    assert x0_permitted("WEAK", True, 0.15, "ADEQUATE", 60, True)["permitted"] is False
    assert x1_permitted(True, True)["permitted"] is True
    assert x1_permitted(True, None)["permitted"] is False


def test_power_gate():
    assert power_gate(120, 48)["status"] == "INSUFFICIENT_POWER"
    assert power_gate(120, 200)["status"] == "SUFFICIENT"
    assert power_gate(-1, 200)["status"] == "NO_EFFECT"


def test_fail_closed_missing_inputs():
    cap = classify_capability(100, 30, 0.85, 0.55, 0.25, True)
    ident = assess_identifiability(100, 70, 30, 0.85, 0.25)
    d = decide_readiness("qualify", cap, ident, True, "STABLE", 0.25,
                         "ADEQUATE", "SUFFICIENT", None, True, 100)
    assert d["readiness"] == "READINESS_UNRESOLVED"


def test_diversity_rule():
    sigs = [(1, 0), (0, 1), (1, 1), (0, 0)]
    assert response_diversity(sigs, {"A": 5, "B": 4, "C": 1})["status"] == "ADEQUATE"
    assert response_diversity([(1, 0)], {"A": 1})["status"] == "SPARSE"
    assert response_diversity(None, None)["status"] == "UNKNOWN"


def test_subject_lock():
    if _subject_allowed is None:
        return
    assert _subject_allowed(None, False)["allowed"] is False
    assert _subject_allowed({"research_subject": True}, False)["allowed"] is True
    r = _subject_allowed(None, True)
    assert r["allowed"] is True and r["mode"] == "historical_control"


def test_exec_guard():
    os.environ["TRIQUETRA_NO_LOCAL_MODEL_COMPUTE"] = "1"
    try:
        try:
            assert_local_compute_allowed("model")
        except LocalComputeForbidden:
            ok = True
        else:
            ok = False
        assert ok is True
    finally:
        del os.environ["TRIQUETRA_NO_LOCAL_MODEL_COMPUTE"]


def test_arch_taxonomy():
    try:
        match_architecture_profile({"architecture_version": "nope-v9"})
    except UnsupportedArchitecture as e:
        assert "UNSUPPORTED_ARCHITECTURE" in str(e)
    else:
        raise AssertionError("expected UnsupportedArchitecture")
    try:
        resolve_checkpoint("checkpoints/does-not-exist-xyz.pt")
    except CheckpointNotFound as e:
        assert "REQUESTED CHECKPOINT NOT FOUND" in str(e)
    else:
        raise AssertionError("expected CheckpointNotFound")


def _vt(query="Return ONLY the ref of Aviary."):
    block = "Aviary keeps ref FMP-939.\nDolmen keeps ref EKH-215."
    return make_visible("t1", block, query, ["FMP-939", "EKH-215"])


def test_legal_arms_answer_blind_templates():
    for q in ("Return ONLY the ref of Aviary.",
              "Which ref belongs to the Aviary? Respond with only the ref.",
              "Give only the ref held by the Aviary."):
        vt = _vt(q)
        dup = e5_dup(vt)
        assert "Aviary keeps ref FMP-939." in dup.splitlines()[-3]
        sel = e7_sel(vt)
        assert sel.startswith("Aviary keeps ref FMP-939.")
        sham = e5_sham(vt)
        assert sham != dup  # sham duplicates a different visible fact
        assert e5_sham(vt) == sham  # deterministic


def test_replication_never_assumed(tmp_path):
    assert check_replication(None, "abc")["replication_ok"] is None
    missing = check_replication({"artifact": str(tmp_path / "nope.json")}, "abc")
    assert missing["replication_ok"] is False
    bad = tmp_path / "rep.json"
    bad.write_text('{"provenance": {"parameter_sha256": "WRONG"}}', encoding="utf-8")
    r = check_replication({"artifact": str(bad)}, "abc")
    assert r["replication_ok"] is False and r["same_checkpoint"] is False
    good = tmp_path / "rep2.json"
    good.write_text('{"provenance": {"parameter_sha256": "abc"}}', encoding="utf-8")
    r2 = check_replication({"artifact": str(good)}, "abc")
    assert r2["replication_ok"] is True and len(r2["artifact_sha256"]) == 64
