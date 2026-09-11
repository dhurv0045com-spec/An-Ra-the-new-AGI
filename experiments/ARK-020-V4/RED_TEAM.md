# ARK-020 V4 RED TEAM (mission §34)

| attack | disposition | test |
|---|---|---|
| scan CLI mismatch (blocker A class) | argparse defines `--drive-ok`; scan emits `@@SCAN_JSON@@`; launcher parses and aborts on failure | TestCLIScanContract (real subprocess) |
| unknown CLI args accepted | argparse strict | test_cli_rejects_unknown_scan_args |
| parent vs source confusion | PARENT_IDENTITIES carries both hashes; scan checks ACQUIRED | TestParentIdentity (fails if source used) |
| dose provenance cross-experiment | local DOSE_SELECTION_IMPORTED receipt; scan validates locally only | TestDoseImportReceipt |
| historical output mutation | `v4_local_output_root` redirect; zero-write assertion | TestZeroHistoricalWrites |
| acquisition counted as forgetting | lifecycle + retention_targets + record_retention_event; decide consumes true events only | TestLifecycleRetention cases 1–3 |
| SPARSE64 = 1/32 in disguise | cadence eligibility; 640-update counts verified | TestReplayCadence (320/640/0) |
| max slots exceeded under contention | eligible-then-risk-order then cap | test_max_two_slots_per_update |
| timebox UnboundLocalError | nested del removed; forced timebox proves RESUME.pt + PARTIAL.json + SessionTimebox | TestForcedTimebox |
| phase-boundary duplicate/skip | phase_step = last-completed; resume computes next deterministically; identity includes phase idx/step/global step | V3 suite + identity tests |
| hard process loss | scan finds bare RESUME.pt → RESUME; PARTIAL without checkpoint is NOT exact resume | TestExactResumeV4 |
| SEALED contamination | controller entry points take CONTROL state only | pinned V3 test |
| C positional/direct/frequency shortcuts | unchanged from V3: independent orders (shortcut 0.3377 ≈ chance), oracle 1.0 | carried adversarial tests |
| D not inverse / pair exposure | forward facts; (q,a) never presented; pre-reversal mutation caught | carried V3 tests |
| global/phase step confusion | phase-relative medians; golden tests distinguish median-vs-median from per-run | carried golden tests |
| incomparable static baseline | task exposure matched; replay replaces real-text only; duty accounted | carried |
| incomplete result accepted | INCONCLUSIVE_INCOMPLETE_MATCHED_SETS | carried golden |
| vacuous tests | meta-test forbids `or True` constructs in V4 sources | carried meta-test |
| calling composition "reasoning" / recovery "prevention" / proxy "AGI" | forbidden by claim ceiling | PLAN + PREREGISTRATION |

VERDICT: no unresolved fatal defect. Accepted risks: C/D formation speed unproven until
run (formation gates make failure honest); 3–5 session operator load; ARK-019 V4 evidence
still provenance-marked until byte-level bundle audit.
