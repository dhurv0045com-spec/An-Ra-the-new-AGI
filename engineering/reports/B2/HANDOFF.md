# B2 integrated build — handoff

Date: 2026-09-13. Author: BRAMASTRA implementation lead (owner-authorized external agent), per `engineering/build_20260912/COMPLETION.md` final-report template.

## 1. Branch, worktree, commits and push result

- Worktree: `C:\Users\ankit\Downloads\An-Ra-the-new-AGI-1\bramastra-build-worktree`, branch `BRAMASTRA`, session start at `02b94d3` (descendant of baseline `415250f`), clean at start. Verified with `git status`, `git worktree list`, `git branch --contains 02b94d3`.
- The workspace clone (`C:\Users\ankit\.zcode\workspace\default\An-Ra-the-new-AGI`, branch `cymek` with user changes) and all other worktrees/nested repositories were left untouched. A temporary detached baseline worktree at `02b94d3` was used to verify pre-existing test failures and was removed afterwards.
- The B2 work is delivered as commit **`d9e4c38`** ("feat(bramastra): implement integrated B00-B12 build …"), pushed to `origin/BRAMASTRA` as a normal fast-forward (`02b94d3..d9e4c38`). Push succeeded with existing owner credentials; no force-push was used or needed.

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


---

# B2.1 addendum — evidence-driven architectural upgrade (2026-09-13)

Owner directive: deepen the build using the experiment results and failures from the Arkenstone and Cymek branches; architectural changes only.

## Mined evidence and what it changed

| Source | Lesson | Change implemented |
|---|---|---|
| ARK-007R/010 RESULT+ANALYSIS | collapse90 = onset→confirm two-step; recovery90 = 3-consecutive; relative displacement (0.379 HIGH vs 0.008 LOW); 8/9 recover | Controller: collapse confirmation evaluations; sustained recovery confirmation; relative displacement as plasticity evidence; instability episodes are REACQUIRE-able, never terminal |
| CONTROLLER_SYNTHESIS.md | "a capability enters the Guardian's protection set only at qualification" | Controller: `qualifying_families` tracked separately; STABILIZE entry promotes the acquiring family into `protected_families` |
| ARK-013 | adaptive_switches: 0 must be observable; pre-frozen margins; floor effects | Proposed transitions recorded in history with `status="proposed"`; margins already frozen in PromotionConfig |
| ARK-012 | threshold aliasing under evaluation cadence | Freshness bound + configurable `controller_eval_every`; cadence documented on `_validate_freshness` |
| ARK-011 | OOD_CONTROL vs OOD_SEALED firewall; sealed-qualified flag | Existing pool firewall retained; controller pool separation documented on metrics |
| R1C PLAN | inactive offset formula; dense diagnostics; byte-identity proof; counterfactual gradients never in optimizer | Offset formula verified identical; diagnostics extended: active/inactive mass split, max inactive probability, hidden-state L2, relative displacement, counterfactual gradient cosine via autograd.grad; non-mutation tests unchanged |
| CYR-GPU-011/012/013 | DEV_CONTROLLER vs DEV_MEASUREMENT vs SEALED pools; candidate-free evaluation; both-members-exactly-correct; M99/G50/G90 sustained gates | Pool separation already in evaluation store; `sustained_gate` implements onset/confirmation/AUC with peak claims structurally excluded; pair metrics unchanged |
| Cymek V5 code | CursorState ordinals; tokens_by_source; fail-closed ordered launch gates with hash-bound PASS receipts | `_preflight` gate manifest (PREFLIGHT_GATES) persisted as preflight.json on every run; fail-closed refusal |
| W02_REVIEW | invalid-action accounting; budget unit naming; inventory cannot win by reporting failure | `episode_summary` reports budget_unit="action", inquiries/submissions/invalid_actions separately; environment behavior unchanged (charging was already declared) |
| W08 | bootstrap by semantic world; underpowered refusal | `clustered_bootstrap_delta` resamples task clusters; promotion refuses `clusters < min_clusters` and CI-straddling-zero accepts |
| W11 | idempotent publication; replay identity binding; overlapping manifests dedup | Ledger idempotency + conflict rejection; ReplayEngine bound to `dataset_identity` (changed data/policy ⇒ new cursor); pools dedup by episode content identity |
| EXECUTION_PLAN §7 (taxonomy) | every failed episode gets one primary observable category | `classify_failure` in collection runner; category stored on every receipt |

## New wiring (integration-tested, real updates)

- Shared `_training_loop` for train and resume (single code path): data batches, replay-designated updates, controller boundaries, checkpointing.
- `collect` CLI subcommand: environments → experience ledger (fixed or failed-baseline policies).
- Controller loop: greedy complete-answer scoring per family on the controller split, transition application (LR multiplier, HOLD pause, checkpoint request), first-family introduction, state persisted in checkpoints.
- Replay loop: proportion-designated updates draw from the experience ledger; reconciliation reports designated updates, planned/consumed entries, shortfall, and render modes (full/endpoints/oversized-skipped).

## Resource ledger update

Cumulative learned smoke now **193/200 optimizer updates, 111/300 s** — nearly exhausted through counted debugging iterations (all recorded). No further learned runs without an owner budget reset. The pre-delivery verification was completed before exhaustion: full parallel suite **426 passed / 10 pre-existing failures**; controller, replay, pair, collection and integrated-path tests all green.
