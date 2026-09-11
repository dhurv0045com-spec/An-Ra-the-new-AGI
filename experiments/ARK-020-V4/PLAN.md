# ARK-020 V4 — OPERATOR-GRADE REPAIR

**Status: PREREGISTERED BEFORE IMPLEMENTATION EXECUTION.** V1/V2/V3 immutable.

## Question (unchanged) and confirmed V3 blockers (all reproduced live)

A. **CLI scan contract broken**: committed V3 runner lacked `--drive-ok` argparse wiring;
   notebook ran scan without `check=True`. → V4: single tested CLI contract; scan emits
   `@@SCAN_JSON@@{...}`; launcher parses and aborts on failure; test executes the real
   command line.
B. **Parent identity confusion**: scan compared `parent_sha` against the SOURCE model
   hash. → V4: `PARENT_IDENTITIES.json` records source_model_sha256 AND
   acquired_parent_model_sha256 + parent_result_sha256 + science gate per seed; scan
   compares against ACQUIRED parent; regression test fails if scan uses the source.
C. **Dose provenance**: scan validated against another experiment's mutable Drive dir.
   → V4: `DOSE_SELECTION_IMPORTED.json` (source experiment/file/sha/slots, verification
   status, equivalence statement); resume validates only the local receipt.
D. **Historical output mutation risk**: V4.acquire_parent/select_dose write beneath
   `ARK019_GUARDIAN_V4`. → V4 option C: `v4_local_output_root()` redirects all V4-machinery
   writes into THIS campaign; historical artifacts reused READ-ONLY after hash checks;
   zero-write assertion test.
E. **Formation counted as forgetting**: finalize loop treated every below-qualification
   observation as a retention failure — including the currently-acquiring skill. → V4:
   explicit lifecycle (UNSEEN/ACQUIRING/ACQUIRED/HEALTHY/WARNING/DEGRADED/FAILED/
   RECOVERING/RECOVERED/DORMANT); `retention_targets()` excludes the acquiring skill
   until confirmation; `record_retention_event()` writes true retention-failure events
   at observation time; interference gate, prevention, recovery, and Guardian triggers
   consume ONLY true events. Regression cases 1-3 tested.
F. **SPARSE64 mislabeled**: requested replay every update (=1/32). → V4: cadence-aware
   eligibility (SPARSE64 = every 2nd update per capability), risk-ordered allocation,
   max 2 slots/update; realized exposure reported. Tests: 640-update counts —
   SPARSE64 320, REPLAY32 640, STATIC_1OF64 320, STATIC_1OF32 640, CAP 0.
G. **Timebox closure bug**: nested `del m, o` → UnboundLocalError after checkpoint save,
   converting sessions into FAILURE receipts. → V4: no nested del; forced-timebox test
   proves RESUME.pt + PARTIAL.json + SessionTimebox.

Also: hard-interruption scan (RESUME.pt without PARTIAL/SESSION_STATE still RESUMEs;
PARTIAL without checkpoint is NOT exact resume); phase-boundary semantics
(`phase_step = last completed step`); exact-resume smoke honors explicit device and
covers lifecycle + scheduler state; meta-test forbids vacuous constructs.

## Everything else unchanged from V3 (audited)

Task A/B/C/D constructions and validity (C: independent XM/MY orders, shortcut 0.3377,
oracle 1.0; D: true inverse), splits, seeds, phases (2000/1500/1500), thresholds,
controller state machine, predictive trigger, efficiency gate, formation/interference
gates (now on true events), phase-relative medians, multi-session durability, claim
ceiling: controlled real-text-proxy evidence only.
