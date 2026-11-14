# V8 implementation review — confirmation back to the implementing agent

**Reviewer:** Codex, with a bounded read-only Luna review.
**Date:** 2026-11-14.
**Reviewed HEAD:** `21f57160343a54053278360e056b3ba4347f7c03`.
**Notebook runtime source:** `95c4b95da28a116e9ebfbd208e291e80b83dee75`.
**Decision: NO-GO for the owner research campaign.** Substantial implementation exists, but the current tests do not establish the production scientific/completion contract. This review is not a V8 research result. No GPU campaign was run and no implementation fixes were made.

## Confirmed checks

- All three focused V8 test files passed: **15 tests in 51.23 seconds**, with CPU thread counts bounded to two. These include miniature tensor/recovery fixtures; they do not qualify S20 on T4 hardware.
- Full-size procedural D1 generation succeeded: **3,584 worlds, 5,888 supervised states**, identity `0afe108c3a311d66f1304e73fd11b2951c604616a4f3bdb740de346684779bc0`. No model was trained during this data check.
- Development generation succeeded: six cohorts of640 records each, **3,840 records total**.
- Executing the notebook generator read directly from its immutable Git source reproduced the committed notebook bytes exactly. Git-blob SHA256: `ba025f44ba6a4bf0366d1a223f368a430f4c04b64462fb67220067d52ef5b30e`. Every code cell compiled. This is a static/reconstruction check, not a Kaggle execution receipt. A Windows working-copy hash can differ because of line endings.
- The notebook is explicitly disabled by default. That is consistent with this proposed V8 release's guide; it must not be represented as automatically launching until the owner authorizes the release configuration.

## Blocking findings and required repairs

### R1 — The production development set bypasses the split firewall

`tools/elendril_v8_worker.py:303` calls `build_d1_dev_suite` without the actual training structural exclusions. The test correctly supplies exclusions, so the passing test exercises a different integration path.

Using the exact default training and worker development calls, I measured **38 non-isolated development records whose structure hashes occur in training**: E14=25, E15=1, A19=12. These are records including linked variants, not38 independent base worlds. Calling the combined split auditor rejected them with `ValueError: split leakage across counterfactual/semantic connected components: 15`.

**Repair:** generate and persist the actual training corpus once; bind evaluation construction to its structural inventory; run the combined component/isomorphism audit before any training. At `worker.py:414`, confirmation excludes only a separately regenerated64-per-family probe corpus. Replace that with the complete actual training and development exclusion sets, including paired variants. Bind all hashes in job identities.

**Acceptance:** run the production corpus/evaluator preparation path at full prescribed quotas; zero prohibited overlaps and a passing combined audit. Keep isolated shared-structure probes explicitly separate from structural-generalization claims.

### R2 — Missing scientific evidence and checkpoints can still yield COMPLETE

`tools/elendril_v8_campaign.py:138` validates report identity and a positive self-reported equality of completed/target updates. It does not compare the target against an independently derived registered count, require the registered milestones/modes, or inspect checkpoint/replay evidence. Completion at lines503–514 relies on those reports. ZIP verification proves the supplied bytes match their manifest; it does not prove required scientific artifacts exist.

**Reproduction:** I supplied six deliberately minimal one-update reports through the existing test-runner injection, with no training, checkpoints, episode records, milestone results or replay receipts. The real orchestration/export path returned **COMPLETE, completed_jobs=6**. Both ZIPs contained only `FINAL_STATUS.json`, `METRICS.json`, `RESULTS.md` and `ARCHIVE_MANIFEST.json`.

**Repair:** implement one required-artifact validator shared by production and test execution. Derive expected updates from the registered corpus/sampler/epochs. Validate every required checkpoint generation, milestone, evaluation mode, cohort identity, source/data binding and independent replay receipt before eligibility for COMPLETE. Fail closed on absent evidence; never let stub reports authorize scientific completion.

**Acceptance:** the reproduction above must fail eligibility. Independently delete/mutate each mandatory artifact in a complete tiny integration fixture and require a non-COMPLETE result. Retain manifest integrity checks as an additional layer.

### R3 — The saved evidence cannot support the promised diagnosis/replay

`worker.py:363–380` writes only episode summaries, omitting the full `EpisodeRecord.steps`: public observations, actual proposed/executed actions, tool responses and per-step evidence. Training step metrics are consumed and discarded except for the last loss. Intermediate episodes are summarized but not preserved; resumed workers rebuild an empty in-memory milestone dictionary. The campaign supplies only METRICS and RESULTS to the analysis exporter (`campaign.py:575`), whose analysis archive does not automatically include episode files.

**Repair:** append transaction-bound update histories and full episode traces throughout execution. Persist milestone identity, checkpoint binding and diagnostics before proceeding. Resume by validating/reusing committed prior artifacts; do not lose earlier milestones. Include replayable raw scientific evidence and protocol/data identities in ANALYSIS; include full recovery tensors in FULL. Independently rescore the saved action traces before final delivery.

**Acceptance:** an exported analysis archive alone must support the promised recorded-action replay and first-divergence analysis; it must not require neural inference or pretend to restore weights. A mid-campaign restart must retain earlier milestones and histories without duplication.

### R4 — The positive canary does not qualify the active architectures

`campaign.py:347` selects only one typed arm, substituting F for the B0-only data-control phase. `worker.py:102` rejects B0 canaries. The GPU canary uses LR3e-4 (`worker.py:93`) rather than the guide's1e-4. Its120 updates fit the stated maximum256; the update count alone is not a violation.

The reported action accuracy at `worker.py:131` multiplies operation accuracy by conditional value accuracy. That is not exact19-action accuracy. Moreover, `value_accuracy or 1.0` converts a real zero into1.0. It can therefore misstate the required95% acquisition criterion.

**Repair:** qualify every active architecture using its real objective/decoder and the common registered canary recipe; support B0 directly. Count correct joint actions over all teacher states, including non-answer states. Keep operation/value diagnostics separate; distinguish zero from missing values.

**Acceptance:** deterministic fixtures with correct operations/wrong values, correct non-answer actions, missing answer subsets and zero conditional accuracy must produce mathematically correct counts. Real S20 GPU canary qualification remains owner-run.

### R5 — Seed pairing and comparison identities are incorrect

`campaign.py:519–534` groups only by architecture and pairs `zip(a_recs,b_recs)` in completion order. Two asynchronously completed lists are not guaranteed to match initialization/order seeds. Both data-control recipes have arm B0, so the intended B0/D0 versus B0/D1 comparison is not constructed by this two-arm branch at all.

Replication stores a confirmation summary (`worker.py:425`) but builds its advancement inputs from development summary fields. The canonical-regression check is also absent from `evaluate_advancement_gate`; its one-hop rule uses a minimum across seeds where the guide specifies a mean. These decisions need an explicit protocol implementation, not implicit substitutions.

**Repair:** use registered condition IDs including dataset/recipe, plus explicit init/order seeds and source/data/checkpoint identities. Join paired runs by those fields, rejecting missing/duplicate/mismatched pairs. Derive every gate from its registered endpoint and split. Confirmation gates must consume confirmation evidence, and all declared regression/robustness rules must be enforced.

**Acceptance:** shuffle completion order and require identical pairing/results; data-control must report three correct recipe pairs; changing only confirmation outcomes must change confirmation eligibility while leaving development results unchanged.

### R6 — Admission and recovery profiling do not represent the campaign workload

The profile measures one job on one device. `campaign.py:389` copies its update estimate to all jobs, and `forecast_campaign_runtime` assumes45 seconds of evaluation per job and perfect two-lane balance, without the guide's2× work margin. D0 and D1 have different sample/update counts. Full evaluation contains thousands of episodes across four policy modes, so45 seconds is an unmeasured assumption.

`worker.py:186,199,214` overrides the recovery test schedule horizon to4 updates, explicitly contrary to the full-horizon recovery requirement. Its parity check compares model tensors, not the full optimizer/scheduler/cursor/RNG transaction. Production interruption sends terminate rather than establishing a worker-side graceful checkpoint transaction.

**Repair:** profile actual arms, longest admitted batches and both GPUs; measure complete evaluation, checkpoint and export work; forecast the actual registered jobs/lane schedule with reserves. Test interrupted resume under the real scheduler horizon and compare complete recovery state. Add worker deadline handling and durable per-job output publication.

**Acceptance:** resource admission uses measured inputs and rejects an infeasible phase before research. Fresh-process recovery and interruption fixtures prove the declared state equivalence and retained artifacts. Hardware/time qualification remains pending owner execution.

### R7 — Milestone diagnostics are not the registered full-training measurements

`worker.py:307–343` regenerates a small D1 probe and always takes its first32 teacher states. I checked these are **32 direct-one-hop states**, excluding query, memory, conflict and abstention supervision. For D0 jobs this is not their actual training corpus. The initial checkpoint has no corresponding interactive cohort evaluation, and the required full-training diagnostics are not collected.

**Repair:** use the actual per-job training sample list for required full-training measurements; separately identify bounded stratified probes. Preserve untrained/epoch1/epoch2/epoch4 evaluations and their exact prescribed modes/cohort sizes. Do not label a small single-family probe as a full-training result.

**Acceptance:** assert identities and denominators against the registered corpus and verify all supervised families appear where required. Confirm earlier milestone evidence survives restart and export.

## Confirmation to send back

The architecture implementation is a useful starting point, and the focused CPU tests plus immutable notebook reconstruction pass. **I do not confirm campaign readiness.** Repair R1–R7 and republish from a new immutable runtime source before owner research execution. In particular, prevent cross-split leakage and false COMPLETE first. Add regression tests through the actual worker/orchestrator paths, not only isolated helpers. Then repeat the focused checks and request an evidence-based rereview. No scientific success or root cause has been established by this implementation review.

This document is saved locally for handoff. It has not been sent to another agent or published by this review.

---

## Re-review after the repair commits — 2026-11-14

**Reviewed HEAD:** `954c7bea8c6ddcb1906ecc5c0770da003c5fba3c`.
**Repaired runtime:** `dd12e7e2ed040a84a0ba27ec566b1715ecc59b4f`.
**Current decision: improvements verified; research launch remains NO-GO.** The findings above describe the older release. The status below supersedes them for this repaired source.

### Verified improvements

- **24 focused CPU tests passed in151.94 seconds**, with two-thread limits. These include full-quota split preparation, rejection of the old minimal-report false-COMPLETE case, artifact mutations, saved-episode replay, B0 canary wiring, seed/condition pairing and interruption/resume fixtures.
- Full-quota split tests assert3,584 training worlds,5,888 states,3,840 development records and zero prohibited structural/component leaks. The worker now uses this verified preparation path.
- Canary scoring now counts exact actions and supports B0; configured GPU LR is1e-4. These are engineering checks, not GPU acquisition evidence.
- The notebook rebuilt byte-for-byte from the pinned repaired runtime, SHA256 `34171e57c29d19eea1bad601d2838341ff4717b4ebff597bae584aafa94394a0`. All code cells compile. Repository URL is corrected.
- Step logs, final full episode traces, independent rescoring and stronger artifact validation are present. Do not describe the old bare-report completion bug as still reproduced on this revision.

### Remaining corrections before confirmation

1. **GPU profile device binding is wrong.** `tools/elendril_v8_campaign.py:1264` passes explicit `device="cuda:0"` for every condition. `tools/elendril_v8_worker.py:725` changes the current device using `device_index`, but passes that unchanged explicit device into every trainer. Selecting current GPU1 does not move tensors explicitly assigned to GPU0. Allocation and default-device memory/synchronization measurements can therefore refer to different GPUs. Resolve one actual device per profile and use it consistently for model/tensor placement, synchronization and memory statistics. Add a wiring regression asserting each trainer receives the selected indexed device; owner GPU receipts must then verify physical device identity.

2. **The full-training and intermediate evidence requirements remain incomplete.** `worker.py:1458–1463` caps production diagnostics at192 states for both datasets. Stratification improves coverage but does not replace the required5,888/9,472-state measurements. Intermediate evaluations are still saved as summaries through `_save_milestone_file` (`worker.py:1340`); full trace serialization is performed only for final evaluation. Implement streaming full-training diagnostics plus intermediate episode persistence, or obtain an explicit prospective protocol amendment. Do not mark R3/R7 fully resolved merely because a stratified probe and final traces exist.

3. **Confirmation custody and replay are not qualified.** `worker.py:277` uses default confirmation seed90901 when none is supplied, and `prepare_verified_splits` builds and saves confirmation worlds during ordinary development phases. Replication can consume that default. Genuine confirmation must require the owner/evaluator seed after candidate selection; a default seeded fixture must be labelled development-only and must not qualify confirmation. `run_job_worker` retains only `confirmation_summary`; the validator at `campaign.py:446` rescores confirmation traces only if a file happens to exist. Require and export full confirmation traces and independent replay, rather than accepting their absence.

4. **Completion still needs exact evaluation coverage validation.** `campaign.py:365–388` checks sets of observed modes/cohorts, not the complete registered world×variant×mode inventory. A partial cohort containing each required name is not a complete evaluation. Derive exact expected record keys/counts from the cached splits, reject missing/duplicate/foreign records, and bind summary counts to independently rescored evidence. Intermediate milestone validation must likewise check its registered coverage and actual checkpoint binding. The existing minimal-report regression is necessary but does not establish this stronger property.

5. **Publication is pending.** At review time both checked GitHub branch refs (`Elendril` and `codex/v8-mechanism-lab`) remain at `21f57160343a54053278360e056b3ba4347f7c03`; the local repair and notebook commits are ahead by two. Therefore the repaired notebook/runtime are not yet published to those refs. Complete the corrections, rerun affected checks, publish a new immutable runtime and rebuilt notebook, then verify remote reachability before owner launch.

Additional operational limits remain: the session timer begins after split preparation, and profile evaluation timing samples only five raw one-hop records. Neither establishes a complete worst-case campaign resource envelope. No CI service result or real T4 qualification was independently established in this review.

**Confirmation back:** the repair work is substantial and the24 CPU checks pass. I confirm those improvements, not full campaign readiness. Close the five items above before the owner spends GPU time. No runtime fixes or research training were performed by this rereview; this is an evidence-based review artifact for the implementing agent.
