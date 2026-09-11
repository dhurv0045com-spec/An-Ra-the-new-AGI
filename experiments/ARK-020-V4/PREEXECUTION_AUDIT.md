# ARK-020 V4 PRE-EXECUTION AUDIT

**Verdict: READY_FOR_OPERATOR_COLAB_CUDA_RUN** (launcher gates re-verify at run time).

## Blocker reproduction (all from live V3 code before repair) and repair evidence

| blocker | live evidence (V3) | V4 repair | proof test |
|---|---|---|---|
| A — CLI scan contract | committed V3 argparse lacked `--drive-ok`; `resume_scan()` called without device flag; notebook scan ran without check=True | single CLI contract: `--drive-ok` defined; scan prints `@@SCAN_JSON@@{...}`; launcher parses + aborts on nonzero exit | real-subprocess test incl. unknown-arg rejection |
| B — parent identity | scan compared parent_sha to SOURCE model hash; no acquired-parent record anywhere | PARENT_IDENTITIES.json records source_model_sha256 AND acquired_parent_model_sha256 (+ result sha + science gate); scan checks ACQUIRED | regression test fails when source hash used, passes with acquired hash |
| C — dose provenance | dose inherited from V4 campaign dir; scan expected a local file that could never exist | DOSE_SELECTION_IMPORTED.json (source experiment/file/sha/slots, verification, equivalence statement) written locally; resume validates only it | import-receipt test |
| D — historical mutation risk | V4.acquire_parent/select_dose write beneath run_ark019_v4.OUT = historical ARK019_GUARDIAN_V4 | option C: `v4_local_output_root()` redirects all V4-machinery writes into this campaign; historical reuse is READ-ONLY after hash verification | zero-write assertion test |
| E — formation ≠ forgetting | finalize loop iterated every metrics row → acquiring skill counted as retention failure | capability lifecycle; retention_targets excludes acquiring skill until confirmation; retention_failure_events recorded at observation time; gates/triggers consume only true events | lifecycle cases 1–3 + interference-gate regression |
| F — SPARSE64 mislabeled | every requesting state replayed every update (=1/32) | cadence-aware eligibility (SPARSE64 every 2nd update per capability) + risk-order + max 2 slots | 640-update counts: SPARSE64 320, REPLAY32 640, STATIC_1OF64 320, STATIC_1OF32 640, CAP 0 |
| G — timebox closure | nested `del m, o` → UnboundLocalError after checkpoint save → session recorded as FAILURE | nested del removed | forced-timebox test: RESUME.pt + PARTIAL.json + SessionTimebox |

Additional repairs: hard-interruption scan (bare RESUME.pt → RESUME; PARTIAL without
checkpoint is NOT exact resume); explicit-device exact-resume smoke including lifecycle
and scheduler state; stale `"ark020_v2_smoke"` default fixed; scan version-prefix and
dose-key bugs caught by the new tests during repair.

## Test summary: 39/39 PASS (CPU), layers

CLI contract (real subprocess) · parent identity · dose import · zero historical writes ·
lifecycle retention · replay cadence · forced timebox · hard interruption · production
exact-resume smoke (executed directly, all state classes) · phase-seed streams · C/D
adversarial validity · V4 integration contracts · checkpoint fail-closed · golden
decisions · operator scan · no-vacuous meta-test.

## Readiness gates (mission §17): all PASS — see RUN_READINESS.json.

## Runtime: scope unchanged (140k main updates); T4 calibration at run time reports
sessions (est. 9–16 GPU-hours ≈ 3–5 × 225-min sessions); estimates cannot change the
frozen protocol. Residual risks: C/D formation speed unproven pre-run (honest formation
gates); operator session load; ARK-019 V4 result still provenance-marked.
