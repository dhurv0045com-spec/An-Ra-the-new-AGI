# B2 integrated build — handoff

Date: 2026-09-13. Author: BRAMASTRA implementation lead (owner-authorized external agent), per `engineering/build_20260912/COMPLETION.md` final-report template.

## 1. Branch, worktree, commits and push result

- Worktree: `C:\Users\ankit\Downloads\An-Ra-the-new-AGI-1\bramastra-build-worktree`, branch `BRAMASTRA`, session start at `02b94d3` (descendant of baseline `415250f`), clean at start. Verified with `git status`, `git worktree list`, `git branch --contains 02b94d3`.
- The workspace clone (`C:\Users\ankit\.zcode\workspace\default\An-Ra-the-new-AGI`, branch `cymek` with user changes) and all other worktrees/nested repositories were left untouched. A temporary detached baseline worktree at `02b94d3` was used to verify pre-existing test failures and was removed afterwards.
- The B2 work is delivered as commits on `BRAMASTRA` (list in `git log`; the handoff commit is the tip at push time). Push was attempted normally over the existing `origin` remote; any credential failure is recorded verbatim in the PROGRESS log rather than worked around. No force-push at any point.

## 2. Implemented modules and integration behavior

All under `bramastra_lab/research/` unless noted:

| Packet | Modules | Behavior delivered |
|---|---|---|
| B00 | `engineering/reports/B2/PROGRESS.md` | Branch/evidence audit; all 10 evidence snapshots re-hashed against `evidence/SOURCES.json` (10/10 OK) |
| B01 | `config.py`, `models/wrapper.py`, `cli.py`, `runtime/readiness.py` | Strict profiles (tiny/development/future_capacity = 117,312 / 6,493,440 / 38,681,088 base params), feature flags defaulting to full-loss controls, config identity, seed initialization; `IntegratedModel` wrapper (logits + optional hidden + finite-action scores + value) with exact analytic-vs-instantiated parameter agreement; `python -m bramastra_lab.research.cli` with side-effect-free help |
| B02 | `experience/codec.py`, `experience/sequences.py` | 260-token byte codec (256 bytes + pad/eos/end-of-event/boundary), canonical key-order-invariant serialization, oversized-event rejection; rows with explicit EOS, loss masks as the exact denominator, segment packing with attention isolation (packed neighbor changes cannot alter episode logits), provenance sidecars that structurally cannot carry answers |
| B03 | `data/manifest.py`, `data/sampler.py` | Local-manifest-only ingestion with `DATA_NOT_READY` states, download refusal, file hashing, semantic duplicate detection across splits, pair groups confined to one split; deterministic group sampler with exact cursor restore and unpaired-control mode |
| B04 | `learning/objectives.py`, `learning/treatments.py`, `learning/schedules.py`, `learning/trainer.py` | Answer+EOS CE with supervised-target normalization; accumulation verified equal to a reference global batch within a declared fp32 reduction-order tolerance (atol 1e-5 / rtol 1e-4); training-only logit treatments (full/participating_mask/inactive_offset with `log((V-|A|)/(K-|A|))` and `|A| < K <= V` validation); inconsistent batches reject rather than lose targets; pair-margin objective with tie/identical-answer exclusion counts; clip certificate with 1e-4 tolerance; post-step telemetry that cannot mutate weights/optimizer/RNG |
| B05 | `runtime/checkpoint.py`, `runtime/resume.py` | Atomic publish (staging → fsync → COMPLETE marker → rename), hash+schema+identity+tamper validation, never-overwrite, writer fencing, LATEST/ACCEPTED_PARENT pointers, milestone-retaining rotation, RNG capture/restore; **fresh-process next-update agreement verified** (see §4) |
| B06 | `learning/plasticity.py` | Pure `transition()` state machine (FORM/STABILIZE/EXPAND/REACQUIRE/HOLD/FIXED_SCHEDULE), pool firewall (sealed/measurement pools raise), freshness/cooldown/hysteresis/max-transition enforcement, plasticity-evidence requirement, collapse never lowers LR, disabled + fixed-schedule controls, serializable state with bounded history and proposed-vs-applied decisions |
| B07 | `experience/ledger.py`, `experience/replay.py` | Hash-chained append-only episode receipts (mutation detection), quality/cost accounting, parent experience survives rejected children; stratified replay with exact counters, no-duplicate epochs, explicit shortfall, state restore |
| B08 | `environments/base.py`, `environments/worlds.py`, `environments/oracles.py`, `runtime/inference.py` | Switch/inventory/program environments with one charging rule, public-only schemas, independent oracles cross-checked against exhaustive enumeration, winning and deliberately-failing baselines, submission cost separated; full-vocabulary free generation with honest complete/stop reporting and separately-typed finite-action scoring |
| B09 | `evaluation/scoring.py`, `evaluation/store.py` | Recomputed exact+EOS scoring (no forgeable success field), paired goal metrics (both-correct, same-answer, goal-swap gap) with pair-completeness rejection, per-family retention with worst-family surfacing, Brier, frozen-margin promotion defaulting to no-promotion, pool-separated append-only stores with sealed-reuse ledger |
| B10 | `planning/planner.py` | No-planning baseline; depth/node/time-bounded planner using only the public model API with all steps marked imagined and real-interaction recording refusing imagined steps; `OracleAdapter` refusing construction without `allow_oracle=True` and stamped in result identity; fixed + learned collection-policy hooks with declared fallback, untrained |
| B11 | `commands.py`, `runtime/smoke.py`, fixtures, `tests/test_research_integration.py` | Full CLI path over real modules; cumulative session ledger with reserve logic; tiny fixtures committed under `engineering/reports/B2/fixtures/` |
| B12 | This handoff, STATUS/paper/contract updates, commits | Review, defect fixes, documentation, commit/push |

## 3. Focused tests and bounded smoke

- **Test suites (all new, all passing):** `tests/test_research_{config,cli,model_wrapper,experience,data,learning,plasticity,checkpoint,ledger_replay,environments,evaluation,planning,integration}.py` — 202 focused tests. Full-tree run: `python -m pytest tests -q -n 8` → 404 passed, 10 failed, all 10 verified as **pre-existing failures at pristine commit `02b94d3`** (historical e1/e2/v5 receipt suites binding older source states; unrelated to B2).
- **Bounded learned smoke (owner authorized 2026-09-13):** exact commands:
  - `BRAMASTRA_LEARNED_CHECKS=1 python tests/_resume_probe.py --workdir <tmp>` (resume verification)
  - `python -m bramastra_lab.research.cli prepare-data --manifest engineering/reports/B2/fixtures/manifest.json --out <dir> --config <config>`
  - `... train --config <config> --data <dir> --run-dir <dir> --max-updates 8 --smoke`
  - `... resume --run-dir <dir> --max-updates 4 --expect-parent <checkpoint-id>`
  - `... infer --config <config> --checkpoint <run-dir> --input <request.json>`
  - `... evaluate --checkpoint <run-dir> --config <config> --data <dir> --split development --out <report>`
  - `... package --run-dir <dir> --out <package.json>`
  - Identities: config/tokenizer/codec/data identities are in `prepared.json` and every package manifest; one full run recorded source identity `02b94d3-dirty` (pre-commit working tree). Profiles: tiny only.
- **Resource ledger (authoritative):** `engineering/reports/B2/SESSION_LEDGER.json` — cumulative **65/200 optimizer updates, 66/300 CPU smoke seconds** (CLI legs exact to 0.001 s; unit-test phases exact update counts with conservative upper-bound seconds). Failures encountered during debugging were counted, not hidden. GPU: **unused** — machine has an RTX 4050 (6 GB) but the installed torch is CPU-only (`torch.cuda.is_available() == False`); the optional single GPU session was left unspent rather than replacing the toolchain.

## 4. Fresh-process restore result

Decisive check via three subprocess phases (`tests/_resume_probe.py`): A = uninterrupted 2 updates; B = 1 update + atomic checkpoint; C = fresh process `restore_run()` + next update. Verdict: `"agrees": true` — model checksum, optimizer checksum and all counters (optimizer_updates=2, presentations=2, supervised_targets_seen=4, encoded_tokens_seen=54) match the uninterrupted reference **exactly** (checksum identity), under the declared criterion "1e-5 relative / 1e-7 absolute". The same publication/restore code is exercised by the CLI resume leg (`--expect-parent` refuses stale/divergent chains).

## 5. Optional mechanisms: implemented / tested / enabled / qualified

| Mechanism | implemented | unit_tested | enabled_in_smoke | scientifically_qualified |
|---|---|---|---|---|
| Pair counterfactual grounding (λ configurable, default 0) | yes | yes | no (λ=0 control in smoke) | **false** |
| Logit treatments (full default; participating_mask/inactive_offset) | yes | yes | no (full in smoke) | **false** |
| Plasticity controller (default disabled; fixed-schedule control kept) | yes | yes | no (disabled) | **false** |
| Experience ledger + stratified replay | yes | yes | no (not in smoke path) | **false** |
| Retrieval | interface only | no | no | **false** |
| Bounded planner / oracle diagnostic | yes | yes | no | **false** |
| Learned collection policy | hook only, untrained | yes (untrained invariant) | no | **false** |

## 6. Data/hardware blockers and operator inputs required

- **DATA_NOT_READY:** no qualified production corpus exists in the repo. To train beyond fixtures the operator must supply a local dataset manifest (`bramastra-dataset-manifest/v1`, schema in `data/manifest.py`; example: `engineering/reports/B2/fixtures/manifest.json`) pointing at local JSONL files with declared license/provenance/trainability, then run `inspect` → `prepare-data` → `train`. Downloads are refused by code.
- **Target device:** CPU verified for tiny smoke only. TPU/accelerator certification remains absent; the GPU present on this laptop is unusable by the installed CPU-only torch build.
- No paid compute was used; no corpus downloads; no 500M/TPU campaigns.

## 7. Scientific claims actually supported

- The integrated system trains, checkpoints, resumes across processes, infers and evaluates with public-only information, exact identities and recomputed metrics (integration evidence).
- Controller, pair objective, treatments, replay and evaluation behave per their deterministic specifications (unit evidence).
- **Not claimed:** AGI, 10x improvement, retention benefit, treatment benefit, controller benefit, or any capability result from tiny-fixture smoke. The D02 inconclusiveness (61/92 vs 61/92, 63/92 vs 65/92) stands unchanged.

## 8. Actual measured effort

- Wall-clock: measured only where instrumented (CLI legs: train 3.765 s, resume 3.312 s; suite durations in PROGRESS log). Total session effort was not separately metered.
- Provider implementation tokens: **unavailable** (the execution system exposes no token counter; per the contract this is recorded as unavailable, not zero).
- Learned smoke: 65 updates / 66 s cumulative (§3).
