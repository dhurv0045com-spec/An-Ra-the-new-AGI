# K8 real execution handoff (contracts implementation, no GPU run)

Source base: `224af33` (BRAMASTRA, origin/BRAMASTRA in sync). This delivery
implements
`engineering/reports/K8_SEMANTIC_REVIEW_20260914/EXECUTOR_CONTRACTS.md` S1–S7
without bypassing readiness, without local optimizer updates, without paid
compute and without touching the old 206-update ledger. Readiness remains
blocked; no `skip-readiness` flag was added. The next step is chief review of
this code + local evidence, then owner GPU execution.

## Identities (frozen protocol)

- `source_closure_sha256`: `9a206b993694ae35c0d42b38cf876f578ef5f5369c65cd9115c088df1b8fb2a2`
- `tokenizer_identity` (`bramastra-byte-260/v1`): `4db7ac566c37acffcac8d1c23895a5c3610285ceba4c7cd544d779ac13893c09`
- `k8_campaign_config.identity` (development geometry + `max_seq=512`, AdamW
  LR 3e-4 / WD 0.01 / clip 1.0, betas 0.9/0.95): `5d9cf2155c9f84e034313cf5a605bf806a66bea8babcb23b446622cc1b5c4ade`
- Campaign model: vocab 260 / layers 8 / width 256 / heads 4 / FFN 704 /
  `max_seq=512` (matches `campaign.json`; bare `development` 256 is refused).
- Data identity is bound per-run from the prepared bundle
  (`k8_identities(data_dir)`); no generic `k8`/`k8-bundle` placeholder is
  accepted by `publish_checkpoint`.

## Commands (from `bramastra-build-worktree`, `$env:PYTHONPATH='.'`)

```powershell
python engineering/reports/K8_SEMANTIC_REVIEW_20260914/probe.py
python -m pytest tests/test_research_k8.py tests/test_research_k8_readiness.py tests/test_research_k8_launch_gate.py -q -p no:cacheprovider -o addopts=
python -m pytest tests/test_research_checkpoint.py tests/test_research_learning.py tests/test_research_contracts.py -q -p no:cacheprovider -o addopts=
python engineering/reports/K8_REAL_EXECUTION_20260914/verify_local.py
python engineering/experiments/K8_20260913/validate_design.py
git diff --check
```

Results on this worktree:

- Semantic probe (was `probe-001.json` at `5c0567f`:
  E5 `completed`/1 update/0 training calls, E6 `completed`/0 payloads,
  `publish_checkpoint` absent):
  now E5 `failed`/0 updates/0 calls, E6 `failed`/0 payloads,
  `publish_checkpoint` present. Fixture paths no longer report success.
- `22 passed` (K8 slice + readiness + launch gate, single invocation).
- `53 passed, 5 skipped` (checkpoint + learning + contracts).
- `verify_local.py`: `12/12 passed, 0 optimizer steps, 0 GPU` (see below).
- `validate_design.py`: no errors, 0 optimizer updates.
- `git diff --check`: clean.

## Acceptance rows (one per contracts section)

| § | Real called symbols | Failing-before → passing-after (local, no training) | Pending GPU check |
|---|---|---|---|
| S1 job/learner/evidence | `phases/types.py: ParentRef.resolve, JobInput.require_update_target, PhaseResult.validate/qualifies_for_campaign`; `runner._phase_success_for_output` rejects `evidence_kind==fixture` | Before: `update_target=4/3` silent defaults, `parent` bare string `E1-B-1701` treated as state, `committed=tasks_per_block`, no `evidence_kind`. After: `verify_local S1_explicit_update_target` (E1 without target fails), `S1_parent_missing_fails` (no ledger → fail, no reinit), `S7_E6_runner_gates` (fixture never qualifies). `worker` now passes explicit `update_target` (E1 200, E3/E4 80), `job_id`, source/data/tokenizer/config hashes, `parent_ref`. | Owner run must bind reservation + absolute deadline + E0-calibrated targets (`N=min(4000,floor(0.75*slot/worst))`, minimums E1 200/E3 80/E4 80) and confirm `qualifies_for_campaign` only for `local-integration`/`learned-campaign`. |
| S2 shared ops lifecycle | `runtime/checkpoint.py: publish_checkpoint, restore_verify`; `phases/ops.py: k8_campaign_config, k8_identities, initialize_random, restore_parent, fork_child, construct_objectives, apply_update, evaluate_episode, publish_checkpoint, restore_verify, migrate_to_gated` | Before: `publish_checkpoint` missing (probe `False`), `require_allocation=False`, generic `k8` identities, `training_update` hardcoded `committed:1`, `optimizer_updates→0` on exception, `RecordingDouble` permissive. After: probe `publish_checkpoint_api_exists:true`; `verify_local S2_checkpoint_publish_restore` (random-init payload via real API with fencing + full identities → load + `restore_verify` ok; generic `k8` refused); `S2_require_allocation` (production gates allocation). Doubles enforce same window/lineage/eligibility and return `fixture-ckpt-*` (never `.pt`). No `type(ops).__name__` branches remain in E1–E5. | GPU must hold writer lease (`writer_token` + `expected_parent`) at publication, preserve distinct arm/seed/phase dirs, and prove `restore_verify` in a fresh process with `expect_config/tokenizer/data` on the owner storage. |
| S3 real supervision | `phases/compiler.py: load_training_trajectories, load_tool_rows, build_batch_for_trajectory, compile_channels_for_row, build_pair_rows` | Before: `"training" in basename` admitted `training-controller`, synthetic fallback rows, fixed `[[70,71],[80,81]]`/`square()`/`result:ok`, `max_seq=128`, provenance overwritten. After: `verify_local S3_exact_split_compiler` (96 exact-`training` rows, canonical `mechanism_id`/`canonical_identity`, controller excluded, empty fails); `S4_backward_only_no_step` (B live `world/action/value` sums from real history/teacher/return, A `extra=={}`). | Owner prepares the full bundle offline (`prepare` + `validate_bundle`); compiler needs no GPU. Confirm family mixture + canonical sidecars survive rendering on the full 4096-mechanism bundle. |
| S4 E1 formation | `phases/e1.py: execute (require_update_target, initialize_random, construct_objectives, apply_update, publish_checkpoint, _evaluate_heldout)` | Before: default 4, `development`/256, aux terms for A, missing pair loss, fixed `[259,1,2,3]` + EOS-as-accuracy, `verified-by-caller-trace`. After: `verify_local S4_E1_paired_formation` (A/B `completed`/2/fixture with doubles, matched `init_state_hash`, held-out `development-measurement` verifier eval, no old markers, no 128 context); `S4_backward_only_no_step` (ProductionOps backward only, named `action/value/embedding` grads, A disabled-term check, `optimizer_updates==0`). Checkpoints at 25/50/75/100% via `publish_checkpoint` with parent chaining. | GPU: paired A/B per seed with identical init hash + stream, frozen 200-update target, 25/50/75/100% restorable payloads, held-out success/goal-pair/cost via independent verifiers; match at latest common update on failure without peeking at sealed outcomes. |
| S5 E2–E4 parents | `phases/e2.py: _resolve_parent, _load_eval_cases, evaluate_episode`; `phases/e3.py: _resolve_parent, _build_training_stream (75/25), _execute_tool_sum, _verify_tool_sample, _evaluate_retention`; `phases/e4.py: _resolve_parent, migrate_to_gated, _verify_handle_migration`; `ops.migrate_to_gated` | Before: random `init_model` instead of restore, invented `1+case%2` counts, `frozen-...` string, T0/T1 same objectives + `50/50` receipt + stored-answer verifier, tiny-model migration discarded. After: `verify_local S5_parent_gates` (E2/E3/E4 without verified parents fail, no reinit; invented markers gone; `run_fixture_generation(` absent from E5); `S5_defect_rejections` (sealed split leak flagged, falsified `{"sum":"WRONG"}` rejected by `verify_tool`, missing migration rejected, corrupt parent refused). E3 enforces T0 `100/0` vs T1 `75/25` (exact split, heldout `filter_then_aggregate_then_check` never in training, execution receipts from real table filter/sum). E4 migrates the training handle itself (S1 gated zero-equal + grads + head/segment contract + optimizer inventory; S0 disabled slots). | GPU: E2 frozen E1 loads + policy/workspace/planner + contradiction/complementary/goal-swap + success/uncertainty with real counters; E3 isolated T0/T1 forks + 75/25 replay + tool-heldout exclusion + execution-argument/output/receipt verification + retention; E4 independent forks + gate values/gradients/migration identity/cost on the trained handle. Local doubles are `fixture`; learned evidence requires GPU optimizer deltas + real `.pt` parents. |
| S6 E5 RSI | `phases/e5.py: _resolve_anchor, _load_meta_tasks, _MethodSensitiveDoubleTrainer, _apply_method_to_handle (dispatch_method_to_trainer on the real learner), _measure_trial, _capture_proposer_choice, MethodArchive` | Before: `run_fixture_generation(confirmed=True)`, `_FakeTrainer`, `committed=tasks_per_block`, `P_fixed=M2`, self-comparing anchors, constant `future_outcomes_seen=False`. After: `verify_local S6_E5_measured_scheduler` (no `run_fixture_generation(`, no `_FakeTrainer`, `P_fixed` compiled from `M0`, no-training boundary yields `failed`/0 — probe confirms). Three blocks use real `meta-training`/`meta-confirmation` tasks, fixed anchor + separate proposer, per-trial `fork_child` with identical support order, `dispatch_method_to_trainer` on the owning learner, measured archive rows (support/query IDs + lineage, failures kept), immutable `MethodArchive` cutoff, five-policy confirmation captured before fresh outcomes, current-task identities rejected in proposal rendering. | GPU: 12×3×45s archive + 12×3 successor + 6×3×90s confirmation per worker with real trainer updates, measured P0 training, captured decoder output → P1 (selected) vs P_fixed (M0) with equal 4-min allowances, same-table scoring (success/AUC/retention/regret/cost). Local receipts stay `fixture`; learned RSI requires GPU-measured outcomes. |
| S7 E6/export | `phases/e6.py: execute, _verify_bundle (exact REQUIRED_PARENT_JOBS, manifest re-hash, real load_checkpoint, .pt gate)`; `campaigns/k8.py: cmd_export (source.json, data.json, failures, proposer_transcripts, comparisons, checkpoints/ payload bytes, artifact_manifest + independent re-hash)`; `runner._phase_success_for_output` (E6 needs non-fixture) | Before: five JSON names + nonempty IDs accepted, hash-of-whatever-exists, metadata-only export, `completed` with 0 `.pt`. After: probe E6 `failed` with 0 payloads; `verify_local S7_E6_runner_gates` (exact `E1-B-1701`/`E1-B-1702` lineage, `.pt` + `fixture-` gates). `cmd_export` now writes `source/data/failures/proposer_transcripts/comparisons/checkpoints/artifact_manifest` and re-hashes the manifest; partial exports report `incomplete`/missing instead of `completed`. | GPU: complete bundle (protocol + source/data + outcomes + failures + comparisons + transcripts + every required parent/successor payload) exported to durable storage, hashes validated, each required checkpoint `load_checkpoint`-verified in a fresh process with real config/arch; incomplete failed-run export lists missing artifacts and does not satisfy acceptance. |

## What was kept vs removed

Kept (already correct): `runtime/checkpoint.save/load` atomic/hash/fencing/expected-parent,
`learning/k8_trainer.accumulate_full_window/finalize_update` single-window
semantics + allocation admission, `learning/k8_scoring` live tensors +
`forward_hidden` gate path, `evaluation/scoring`, `data/k8_bundle`
build/validate/sampler/manifest split/canonical/cursor logic, slot/termination/
device/lease supervision, E0 canonical update + resume proof.

Removed (production fixtures/fabrications): silent `4/3` defaults, bare
`development`/256 profile, `require_allocation=False`, answer-only-A aux
construction, pair-without-loss, invented candidates/regression/transitions,
`type(ops)` branches, fixed `[259,…]` prompts, substring stream matching,
synthetic fallback rows, random-init-instead-of-restore, invented
action/call/node counts, `frozen-…` strings, same-weight T0/T1 + `50/50`
receipts, stored-answer-as-execution, discarded tiny-model migration,
`_FakeTrainer` + `run_fixture_generation(confirmed=True)` + `P_fixed=M2` +
task-count updates + constant isolation flag, JSON-only E6 + substring parent
checks + metadata-only `cmd_export`.

## Pending chief/owner actions (no additional local work authorized)

1. Chief reviews this diff + `verify_local.py` + probe delta and updates
   `campaigns/readiness.py` dispositions only on code + local evidence (CUDA
   presence alone must not clear it).
2. Owner prepares the full bundle (`k8 prepare` with 4096/family + 256
   controller/development + 128 confirmation + tool/meta sizes), validates it,
   and launches `k8 run --mode e0` then `--mode full` on 2×T4 within the
   480-minute allocation (training cutoff 450, export reserve last 30).
3. GPU must supply E0-calibrated update targets, writer-lease fencing,
   per-phase parent payloads and fresh-process restore proofs; E6 must return
   `completed` only for a complete verified bundle.

No local optimizer updates were performed (`validate_design` reports
`optimizer_updates: 0`; trainers were used for `accumulate`/`backward` only
with `zero_grad` and no `finalize_update` in local checks, except doubles
which perform zero updates by construction). No paid compute was used. No
`skip-readiness` path exists (`readiness_still_blocked` passes).
