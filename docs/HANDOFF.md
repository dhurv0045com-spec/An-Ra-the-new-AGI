# Current S20 investigation — 13 November 2026

Use [V7.17 implementation, scientific interpretation and owner instructions](../research/v7.17/README.md). The new source-pinned notebook investigates optimization and evidence formatting with S20-only research arms. Historical V7.16 results and every prior reproduction remain intact. Local verification covers mechanics; owner GPU results are pending and **CAUSE UNRESOLVED**.

# Historical V7 handoff — 8 November 2026

Use the [V7 engineering and owner handoff](../research/v7/HANDOFF.md), [preregistered protocol](../research/v7/PROTOCOL.md) and [evidence review](../research/v7/EVIDENCE.md). The V6 architecture, evidence and reproduction paths below remain intact. V7 research and real T4 qualification are pending; versioned recorded-action replay verifies 120 historical evaluations without changing them.

# Historical V6 notebook export-retry repair — 7 November 2026

The latest owner Kaggle notebook is [ELENDRIL_V6_KAGGLE_T4X2.ipynb](../notebooks/ELENDRIL_V6_KAGGLE_T4X2.ipynb), published at `90b083bec629ebff4b653108f4ce2eca0fef9f94` and pinning runnable source `9b2f1b29b9e05394e165ed375bae5e94693c6877`. It allows verified ZIP reconstruction from a Kaggle saved output on a CPU-only session, while all training, canary and same-/fresh-VM recovery gates still require live dual T4s. The notebook's old draft-only message was corrected. The revised release passed [GitHub Actions CI](https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI/actions/runs/42382874551): **214 passed, seven subtests**. This establishes CPU/code safety only, not a bug-free or GPU-qualified Kaggle run. The earlier V6 source pin and results documented below remain historical, not the currently recommended notebook.

# ELENDRIL V6 handoff — 7 November 2026

Latest continuation adds four investigator-designed experiments to the same notebook. **Current source pin: `6ad497997f728d8161a42f7525e1dfb91ffe6d22`**, registry `6086c1cf5e437f44cb5a0c10046b36fd19cb39a7399a2bc2ddc559b16af24163`. The original15 panels remain intact. [Prospective supplement and falsifiers](../research/v6/INVESTIGATOR_SUPPLEMENT_2026-11-07.md)

- S16: identical28 worlds at1/2/4 decisions; paired accuracy/utility/work gains and regressions.
- S17: same28 worlds under one consistent subject/value/graph symbol bijection; both-correct and mapped-output equivariance.
- S18: four known worlds with empty versus contradictory older same-source memory, both paying engineered prefetch costs.
- S19: full existing raw evaluation and matched controls repriced at query costs0/0.1/0.5/1, with rankings and break-even price. No claim of adaptive cost-aware behavior.

No training arms or separate notebooks were added. Each condition is saved immutably and independently replayed; cached conditions resume without another model call. A90-second measured admission allowance per checkpoint shares the existing diagnostic ceiling and export reserve. Forecast/deadline skips remain explicit. All extra raw conditions, summaries and the four-view chart join the same results ZIP. New registry/source identities require a fresh run/output; old results remain bound to their original pins.

Integration repaired P14's comparison between two distinct identity types: before/after tensor fingerprints now agree separately from checkpoint-manifest binding. A regression test rejects a wrong checkpoint even when tensor fingerprints agree. Independent export verification now also binds diagnostic raw receipts to an already verified parent checkpoint/result and replays supplemental conditions/summaries.

Closing Windows tests: **212 passed, one Windows symlink skip, seven subtests in75.19 seconds**. Additional generated-notebook/schema/default/export checks: **13 passed, one skip in7.60 seconds**. [Runtime Linux CI](https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI/actions/runs/42381123330): **213 passed, seven subtests in37.13 seconds**, including the symlink test. A scripted test-double chart/ZIP check passed; it is not neural evidence.

[Actual engineered-control audit](../artifacts/elendril/v6-engineering/supplement-control-audit.json) at the clean source pin took1.0731 wall seconds/1.0781 CPU seconds including112-world generation. Fixed controls improved16/28→28/28 between one and two decisions, showed28/28 both-correct/equivariant permutation pairs, and preserved4/4 known answers under stale memory. The full raw traces are preserved. S19 arithmetic, control matching and ties are unit-tested; no neural S19 result exists. No local research training or GPU execution occurred.

**GO remains limited to owner preflight/canaries; the research campaign needs real T4 learning and fresh-VM recovery evidence.** The next scientific decision is whether the same frozen neural checkpoint reproduces these patterns, exposing a bottleneck in effort use, binding, memory interference or economics before architectural expansion.

## Previous V6 delivery and preserved evidence

Continued the existing V5 system and fast-forward incorporated the complete research package at `777232f103312d80810c89f80d40cba94147f648`. All five package files were reviewed and preserved unchanged. The original Cymek worktree and historical checkpoint payloads were not modified. The previous V5 handoff is retained below.

| Exact commit | Delivered change |
| --- | --- |
| `6273b0dddc84083a8ea6193d19f4afc48e3b2937` | V6 raw-world data, neural/action training, 15-panel evaluator, two-worker campaign, independent verifier, safe recovery ZIP, notebook builder, tests and prospective scientific amendment. |
| `1cb3f37772d25c7569314031014d5372d490ad2e` | Canonical LF evaluator source identity for Windows/Linux replay. Immutable runtime and notebook source pin. |

The publication commit containing this handoff adds the generated notebook, its schema/source-pin test, CPU receipts and updated import graph; identify its exact SHA with `git log -1`. [Notebook](../notebooks/ELENDRIL_V6_KAGGLE_T4X2.ipynb) · [Launch/return instructions](V6_CAMPAIGN.md) · [Scientific amendments](../research/v6/IMPLEMENTATION_AMENDMENT_2026-11-07.md)

Implemented code: `v6_data.py` serializes/audits raw evidence, worlds, split state and transformed controls; `v6_registry.py` freezes profiles, arms and gates; `v6_training.py` reuses V5 tensors/checkpoints with true final batches, paired representations and explicit objective/curriculum differences; `v6_evaluation.py` records strict raw actions and independently replays them without exposing truth to neural setup. `foundation_training.py` caches exact epoch permutations. The tools implement bounded physical-UUID worker lanes, canaries, optimizer/RNG/cursor recovery, immutable selection, conditional scales, calibration before reserved evaluation, cached diagnostics, failure-preserving finalization and checksum-verified ZIP export/extraction. No whole branch was copied.

Observed engineering evidence at the pinned source:

- [Linux CPU CI](https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI/actions/runs/42379351625): **204 passed, seven subtests, 29.60 seconds**; one NumPy-absence warning, no failing test. Publication-only notebook checks follow separately.
- Closing Windows suite: **203 passed, one skipped, seven subtests, 62.54 seconds**. The skipped test requires Windows symlink privilege; it passes in Linux CI. The additional publication test then passed independently, validating the actual notebook with nbformat and checking exact generator/runtime bytes at its pin. All-default-disabled execution and recovery/export failure-injection tests are included in the suite.
- Historical verifier: **78 hashes, 63 receipts, 519 scoring snapshots, 237 checkpoint manifests and 1,659 components** reverified. Old S5 custody gaps remain gaps.
- [Dataset audit](../artifacts/elendril/v6-engineering/data-adversarial-audit.json): 2,200,185 prompt/target bytes, 7,335 worlds, 9,472 examples; base maximum268 tokens. Validation/reserved confirmation112 worlds each, 16 per kind. The audit is labeled as the actual pre-commit snapshot it inspected, not relabeled as a final-source receipt.
- [CPU qualification](../artifacts/elendril/v6-engineering/cpu-qualification.json): actual S3/S8/S20 tensors contain **3,084,800 / 7,970,944 / 19,871,616** parameters; no forward/training on these full models. A separate width16 one-layer UTF8 model ran eight updates on four QUERY prompts: **4/4 exact QUERY+EOS**, loss **5.6562614→1.2048372**, update/save/generation wall **0.1733774 seconds**, overall harness1.0967 seconds. This is overfit correctness, not held-out cognition or the GPU positive-control gate.
- The tiny checkpoint is `45f3886d3a71d72ac7a46ab419aa500d422fe996c598dd8083cda1d07e2a637f`, external root `C:/Users/ankit/Downloads/Elendril-V6-engineering-canary`, lineage `utf8-query-eos-micro`.
- [Sampler-only timing](../artifacts/elendril/v6-engineering/sampler-micro.json): 4,096 samples, 256 batches of16, exactly identical index digest, one cached epoch; **0.2050423→0.0011164 seconds** in this single CPU measurement. No model was trained and no GPU efficiency is inferred.

**Launch: GO only for owner-run preflight/canaries; NO-GO for research until live gates pass.** The notebook defaults every launch switch off and pins source `1cb3f377...`, registry `d1b516d4c078ef48e0f15618866e246ceea0c97d5e58266c5b770eefb9cdc673`. Its published SHA256 is `8c3912e3746f3413b9cf7079e1b964323ce4e845050c5bbe0d4f4b693abfaf42`. First run preflight/canaries, save complete output, then qualify optimizer continuation in a genuinely new VM using reattached output. Only then authorize the bundled campaign. The budget is a forecast ceiling, not measured T4 performance. Data is generated automatically under writable `/kaggle/working`, not read-only `/kaggle/input`. Native PyTorch is inspected rather than replaced.

Completed jobs/checkpoints/results persist incrementally. Finalization attempts independent replay and `exports/ELENDRIL_V6_RESULTS.zip` even after failure; export failure preserves raw files and the previous valid ZIP. Download that ZIP and save a Kaggle output version. An abruptly destroyed unsaved VM/full disk cannot be guaranteed recoverable. One canonical VM writer owns each lineage.

Scientific limits: no GPU, held-out neural success, metacognition, cumulative learning, forward transfer or self-improvement has been established. Confirmation is development evidence sharing task design/authorship, not an outside sealed evaluation. A5 uses extra supervision/work; R0/R1 have different vocabulary/embedding sizes; route worlds have extra training mass. P15 requires real separately acquired A/B and matched new C trajectories and stays NOT_QUALIFIED without them. S8/S20 can be skipped when gates/resources fail. The next decisive experiment is the all-kind T4 formation canary, followed by the frozen multi-seed S3 screen to separate action formation, evidence revision, invariance and depth failures before scaling.

## Preserved V5 handoff

Continued `Elendril` from `f516e0bdb8f9137532b35c0c23299f7f9dbdf185`; no reset, bulk branch merge or historical checkpoint overwrite. The working V4 source and paused V3 program are remotely preserved as `elendril-v4-working`. The pre-V3 archive remains `elendril-pre-v3`.

Implemented and pushed source milestones:

| Commit | Change |
| --- | --- |
| `1f630365215166554e4e654f4e114e02700ac54f` | Versioned legacy/UTF8 representation, shared neural backend, action worlds, exact-denominator trainer and decoded recovery. |
| `ade2d5bf69abeb6b07daca162cf1d275a5f91bee` | Separate-process publisher race, Unicode training/resume, scaler/migration corruption tests and staged legacy tensor decoding. |
| `afbf0c45e6fdddc5c6e5eebfc99a3622aeca2141` | Checkpoint CLI, gated action cloud runner, raw-action replay, scientific gates and archive cleanup. |
| `78ce7def7c1b7bdc4a73a3e1bb7022485880ceab` | Canary CPU and CUDA allocation telemetry. |
| `1d0c6aedbdccd05dbc85e57a66fbfe433e75eb12` | Fix undeclared NumPy fingerprint dependency; preserve exact byte identity. |

The notebook/evidence publication follows these source commits. Both active notebooks pin **1d0c6aedbdccd05dbc85e57a66fbfe433e75eb12**, not a moving branch. [Colab](../notebooks/ELENDRIL_ACTION_COLAB_T4.ipynb) · [Kaggle](../notebooks/ELENDRIL_ACTION_KAGGLE_T4X2.ipynb)

The active code adds `representation.py`, `backend.py`, `foundation_training.py`, `action_learning.py` and `checkpoint_tools.py`. Legacy IDs and manifest hashes are unchanged. The UTF8 path has a separate vocabulary/config identity and explicit row migration with optimizer reset. The original frozen policy and new trained-action policy use the same backend/episode contracts. The reference trainer retains old sampling/objective semantics while adding pre-publication decoding. The new trainer supports variable lengths, exact weighted accumulation, finite/ownership checks, explicit curricula/schedules/precision and typed stage migration. Recovery protects PREVIOUS, rejects stale publishers and validates actual tensor shapes/dtypes, optimizer/scaler counters, RNG and cursor before publication.

Final Windows qualification passed **166 tests plus seven subtests in 42.03 seconds**. The new notebook/operator subset passed ten tests including schema, all default-disabled cells with writes/subprocesses forbidden, exact source/generator bytes, physical UUID gates, nonfinite forecasts, recovery challenges and independent replay. Repaired runtime source passed [Linux CPU CI](https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI/actions/runs/42366390091), 164 tests plus seven subtests before the two publication-only tests. No test was disabled or weakened to hide a failure. [Qualification details](FOUNDATION_QUALIFICATION.md)

Observed results and costs:

- Tiny correctness control: four prompts, eight updates, QUERY+EOS 4/4 on the training prompts; loss 3.8082435→0.7462860. Measured update/save/generation wall 0.7044 seconds, zero GPU. The earlier LR .01 attempt failed the unchanged 50% loss-reduction threshold; it remains recorded. This is overfit learnability, not action triage.
- Ten validation worlds: engineered fixed 10/10 with four requests, always-query 10/10 with eight, no-query 6/10. Full raw traces and shortcut/oracle controls are retained. The oracle is explicitly privileged.
- Frozen V4 replay at repaired source: route-memory 4/4, no-memory 0/4, same-token context 1/4, revision 0/4 with zero QUERY. Forty calls/912 input positions. Before/after weights remain `8734cfb9957f510a703c0d04e5f68a305e0af9552238c28f178e848dc9c2e863`.
- Native GQA matched reference forward/gradients in the packed probe, but CPU medians were 3.3545 vs 2.8698 ms (1.169× slower); reference stays default. No GPU speed/memory result is claimed.
- Historical custody: 78 evidence hashes, 63 receipts/519 snapshots, 237 manifests/1,659 components reverified. The current installed wheel passed Unicode/control/inspection checks; the rebuilt NumPy-free wheel additionally passed migration and the exact original-checkpoint fingerprint with Tensor.numpy forbidden. No training occurred during wheel qualification.

[Raw receipts and SHA manifest](../artifacts/elendril/foundation-v5/manifest.json) · [Result chart](../artifacts/elendril/foundation-v5/qualification.png)

Cleanup removed the paused V3 notebook wrappers/generator and three superseded planning/trace documents from the active tree, preserving exact Git objects and their disabled-cell tests. [Removal manifest](ARCHIVED_PROGRAMS.json). The active tree has 204 tracked files, 16 docs and two V5 notebooks, about 15.5 MB including preserved raw evidence; the original pre-V3 tree had 861 files, 228 docs and ten notebooks. Shared low-level cloud verification remains because the action runner imports it. Other branch worktrees and external weights were untouched.

Operator commands:

~~~text
python -m pytest -q
python -m tools.verify_elendril_evidence --checkpoint-base C:/Users/ankit/Downloads
python -m elendril inspect-checkpoint --root <root> --lineage <lineage>
python -m elendril verify-checkpoint --root <root> --lineage <lineage>
python -m elendril resume-dry-run --root <root> --lineage <action-lineage>
python -m elendril compare-checkpoints --root <root> --lineage <lineage> --sha <old> --other-sha <new>
~~~

The tiny canary checkpoint `2a4d108fabaf6b0b8515c94042eabfcf88fa201dc8ff976b0d0db0c35a095eaa` is external at `C:/Users/ankit/Downloads/Elendril-V5-foundation-canary`. Original ELD payloads remain under their sibling custody roots. No giant weights are in Git. Use a separate detached worktree at an exact source SHA for historical reproduction; never reinterpret old weights with UTF8 or overwrite the accepted lineage.

Launch decision: conditional GO for owner device/correctness canaries only. Research campaigns remain **BLOCKED_PENDING_GPU**. Colab plans CE/weighted CE/uncertain-world curriculum × two seeds × 448 updates; Kaggle uses the same arms × five seeds × 896 updates on separate UUIDs and a distinct world seed/split. Caps are 105+15 and 315+30 minutes, not measured runtime promises. All launch switches are OFF. Admission requires actual formation, all-arm timings, same-device exact resume and a genuinely different VM session reattaching durable outputs. Kaggle also needs complete rescored Colab evidence and owner review. [Operations](OPERATIONS.md) · [Frozen decision rules](ALGORITHM_COMPARISONS.md)

Limitations remain substantive. No GPU campaign, real-text language learning, learned action triage, calibrated metacognition, new-skill retention, forward transfer or self-improvement was established. Fresh action weights intentionally test formation and do not replace the working two-read checkpoint. Byte expansion changes context cost and softmax competition; it has no qualified whole-copy-head migration. The verifier is in-process, not a sealed execution sandbox. Older S5 checkpoints remain outside reviewed custody. The next decisive experiment is full multi-action formation under the frozen accuracy/request-cost gates, then a separately qualified warm-start or replay retention comparison. [Next experiment](NEXT_COGNITIVE_EXPERIMENT.md)
