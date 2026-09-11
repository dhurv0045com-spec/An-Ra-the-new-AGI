# ARK-020 V3 — PRE-EXECUTION AUDIT

**Verdict: READY FOR OPERATOR COLAB CUDA RUN** (launcher gates re-verify at run time).

## Defect disposition (V2 → V3)

| blocker | live verification | repair | test that catches reintroduction |
|---|---|---|---|
| exact-resume smoke could not execute (11-vs-10 arg mismatch; malformed post-load reconstruction; never executed by tests) | CONFIRMED (run_ark020_v2.py:580,597,624-625,638) | dict-state rewrite; 12 hashed state classes; real save/load identity; production function executed by tests | direct smoke test; mutation: tamper any field → FAIL |
| C positional shortcut (tied orders) | CONFIRMED (same `order` for both blocks; tied score 1.0000) | independent namespaces per block+mode | positional-shortcut test (near 1/3) + tied-mutation test (1.0) |
| D not truly inverse (pre-reversed facts) | CONFIRMED (inverse_split5 built (value,key) pairs) | forward (owner, object) presentation; object queried; owner answered | orientation + exposure tests; pre-reversal mutation caught |
| scan before Drive mount; identity overclaim | CONFIRMED (drive.mount in cell 3; PASS on key existence) | mount-first cell 0; real identity verification; PARTIAL_IDENTITY_CHECK; six SAFE ACTION states | operator-scan test suite (fresh/resume/malformed/version/lock/dose) |
| vacuous smoke assertion | CONFIRMED (`or True`) | deleted; meta-test forbids the construct | TestNoVacuousTests |
| weak lock | CONFIRMED (bare timestamp file) | structured CAMPAIGN_LOCK.json; ACTIVE/STALE/MALFORMED | lock tests |

## Test summary (27 tests, all PASS, CPU)

pure construction · C adversarial validity (oracle 1.0, shortcut 0.3377, tied mutation 1.0) ·
D true-inverse + pre-reversal mutation · phase-seed stream independence (task + real-text) ·
V4 integration contracts (signature pins, real-object bmetrics/mixed_update/dose-pilot) ·
checkpoint fail-closed (5 mutations) · golden decisions (13 scenarios) · operator scan
(6 states) · production exact-resume smoke executed directly (PASS, all 12 fields) ·
vacuous-construct meta-test.

## Readiness standard (mission #38) — each item established

Known V2 resume crash repaired ✓ · smoke actually executed ✓ · C ordinal shortcut broken ✓ ·
shortcut near chance ✓ · composition oracle perfect ✓ · D truly inverse ✓ · no vacuous tests ✓ ·
phase seeds correct ✓ · phase-relative metric correct ✓ · V4 integration correct ✓ ·
SEALED firewall ✓ · Drive mounted before scan ✓ · scan identity truthful ✓ · lock safe ✓ ·
checkpoint fails closed ✓ · tests pass ✓ · red-team clean ✓ · preregistration matches code ✓ ·
executable frozen (RUN_READINESS blob SHAs) ✓.
