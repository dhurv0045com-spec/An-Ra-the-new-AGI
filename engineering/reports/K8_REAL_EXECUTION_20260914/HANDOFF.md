# K8 real execution handoff (contracts implementation, no GPU run)

Source base: `224af33` → `187982c` (BRAMASTRA) + debug hardening below. This
delivery implements
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
python -m pytest tests/test_research_k8.py tests/test_research_k8_readiness.py tests/test_research_k8_launch_gate.py tests/test_research_k8_real.py -q -p no:cacheprovider -o addopts=
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
- `35 passed` (K8 slice + readiness + launch gate + 13 real-execution
  regressions, single invocation).
- `53 passed, 5 skipped` (checkpoint + learning + contracts); combined
  `88 passed, 5 skipped` across all seven files.
- `verify_local.py`: `14/14 passed, 0 optimizer steps, 0 GPU` (see below).
- `validate_design.py`: no errors, 0 optimizer updates.
- `git diff --check`: clean.

## Acceptance rows (one per contracts section)

| § | Real called symbols | Failing-before → passing-after (local, no training) | Pending GPU check |
|---|---|---|---|
| S1 job/learner/evidence | `phases/types.py: ParentRef.resolve (exact ==, run_dir carried), JobInput.require_update_target, PhaseResult.validate/qualifies_for_campaign (explicit phase, no marker spoof)`; `runner._phase_success_for_output` rejects `evidence_kind==fixture`; `worker` + `process_supervision` propagate `job_id/slot/parent/update_target/eval_cases/tasks_per_block` | Before: `update_target=4/3` silent defaults, substring `wanted in jid` (1701 matches 17010), bare parent strings, `committed=tasks_per_block`, no `evidence_kind`, spawn dropped `slot/parent/job_id`, worker synthesized `job_id` and invented 200/80. After: `verify_local S1_explicit_update_target` + `S1_parent_missing_fails` + `S1_exact_lineage` (exact match only, `run_dir` carried) + `worker_propagation` (explicit targets required, spawn propagates); `test_research_k8_real.ParentExactMatchTests` (substring refused). | Owner run must bind reservation + absolute deadline + E0-calibrated targets (`N=min(4000,floor(0.75*slot/worst))`, minimums E1 200/E3 80/E4 80) and confirm `qualifies_for_campaign` only for `local-integration`/`learned-campaign`. |
| S2 shared ops lifecycle | `runtime/checkpoint.py: publish_checkpoint (exact segment lineage, dir_suffix namespaces), restore_verify, save_checkpoint(dir_suffix)`; `phases/ops.py: k8_campaign_config, k8_identities, initialize_random, restore_parent, fork_child, construct_objectives, apply_update(pair_rows), evaluate_episode (no fixed prompt), publish_checkpoint, restore_verify, migrate_to_gated` | Before: `publish_checkpoint` missing (probe `False`), `require_allocation=False`, generic `k8` identities, `training_update` hardcoded `committed:1`, permissive doubles, `save_checkpoint` wrong `run_dir` + hardcoded `E1/compat`, cross-arm `update-000000000000` collisions, `part in run_id` substring. After: probe `publish_checkpoint_api_exists:true`; `verify_local S2_checkpoint_publish_restore` (random-init payload via real API with fencing + full identities → load + `restore_verify` ok; generic `k8` refused; A/B same-index collision-free with `update-…-E1-A-1701` vs `-E1-B-1701`); `S2_require_allocation`; `test_research_k8_real.CheckpointNamespaceTests` (legacy names unchanged). No `type(ops).__name__` branches in executors. | GPU must hold writer lease (`writer_token` + `expected_parent`) at publication, use namespaced dirs per arm/seed/phase, and prove `restore_verify` in a fresh process with `expect_config/tokenizer/data` on owner storage. |
| S3 real supervision | `phases/compiler.py: load_training_trajectories (exact pool), load_tool_rows, build_batch_for_trajectory (512), compile_channels_for_row (real history/teacher/return; program single-action; all-queries coverage), build_pair_rows` | Before: `"training" in basename` admitted `training-controller`, synthetic fallback rows, fixed `[[70,71],[80,81]]`/`square()`/`result:ok`, `max_seq=128`, kind-only teacher fallback, 4-query truncation (random orders missed). After: `verify_local S3_exact_split_compiler` (48/48 compile after fix; 96 exact-`training` rows, canonical IDs, controller excluded, empty fails); `S4_backward_only_no_step` (B live `world/action/value` sums from real history/exact teacher/return, A `extra=={}`). | Owner prepares the full bundle offline (`prepare` + `validate_bundle`); compiler needs no GPU. Confirm family mixture + canonical sidecars on the full 4096-mechanism bundle. |
| S4 E1 formation | `phases/e1.py: execute (require_update_target, initialize_random, construct_objectives, apply_update(pair_rows, no downgrade), publish_checkpoint, _evaluate_heldout with real prompts + evaluate_episode only)` | Before: default 4, `development`/256, aux terms for A, missing pair loss, fixed `[259,1,2,3]` + EOS-as-accuracy, `verified-by-caller-trace`, fragile `str(exc)` downgrade masking pair errors. After: `verify_local S4_E1_paired_formation` (A/B `completed`/4/fixture with doubles incl. program rows, matched `init_state_hash`, held-out verifier eval); `S4_backward_only_no_step` (named grads, A disabled-term check, 0 steps). Checkpoints at 25/50/75/100% with parent chaining. | GPU: paired A/B per seed with identical init hash + stream, frozen 200-update target, 25/50/75/100% restorable payloads, held-out success/goal-pair/cost via independent verifiers; match at latest common update on failure without peeking at sealed outcomes. |
| S5 E2–E4 parents | `phases/e2.py: _resolve_parent (exact multi-part B-primary + controls)`; `phases/e3.py: _resolve_parent (single only), _build_training_stream (75/25 exact ≥4, honest tiny), _execute_tool_sum, _verify_tool_sample, _evaluate_retention, _stream_leaks_protected (exact split only)`; `phases/e4.py: _resolve_parent (single only), migrate_to_gated, _verify_handle_migration`; `ops.migrate_to_gated (real inventory checks)` | Before: random `init_model`, invented counts, `frozen-...`, T0/T1 same objectives + `50/50`, stored-answer verifier, tiny-model migration discarded, `split("/")[0]` picking wrong arm, substring leak backstop, `hasattr` capability branches, small-count mixture always failing. After: `verify_local S5_parent_gates` + `S5_defect_rejections` (combined E2 resolves B-primary with A controls, single-arm E3/E4 reject combined keys, sealed exact-split leak flagged, falsified tool rejected, missing migration rejected); E3 T1 `target=4 → 75/25` verified, tiny targets report true composition; E4 migrates the training handle with gate inventory. | GPU: E2 frozen loads + 6 modes with real counters; E3 isolated forks + 75/25 replay + heldout exclusion + execution receipts + retention; E4 independent forks + gate values/gradients/identity/cost on the trained handle. Doubles are `fixture`; learned needs GPU deltas + real `.pt` parents. |
| S6 E5 RSI | `phases/e5.py: _resolve_anchor (explicit only, no synthesis), _load_meta_tasks, _MethodSensitiveDoubleTrainer, _apply_method_to_handle, _measure_trial (fixture-only for doubles, refusal for production-local), _capture_proposer_choice, MethodArchive (immutable cutoff)` | Before: `run_fixture_generation(confirmed=True)`, `_FakeTrainer`, `committed=tasks_per_block`, `P_fixed=M2`, synthesized `E1-B-{seed}` default, in-place anchor mutation, `isinstance/_NoTrainingMarker` + `type(ops)` branches (dead guard), hash-invented `measured_success` for prod, `e5-archive-…` invented checkpoint. After: `verify_local S6_E5_measured_scheduler` + E5-with-parent fixture run (`completed`/3/fixture, `P1=M1/P_fixed=M0`); no anchor synthesis (missing fails), pristine anchor + forked lineage, `checkpoint_identity=archive_identity` content hash, production-local measurement refuses instead of inventing. | GPU: 12×3×45s archive + 12×3 successor + 6×3×90s confirmation with real adaptation training under allocation, measured P0 training, captured decoder output → P1 vs P_fixed (M0) equal 4-min, same-table scoring. Local stays `fixture`. |
| S7 E6/export | `phases/e6.py: execute, _verify_bundle (exact REQUIRED_PARENT_JOBS, manifest re-hash, real load_checkpoint, .pt + fixture gates, bundle_identity=content_identity)`; `campaigns/k8.py: cmd_export (source/data/failures/transcripts/comparisons/checkpoints/artifact_manifest + re-hash)` | Before: five JSON + nonempty IDs accepted, hash-of-whatever, metadata-only export, `completed` with 0 `.pt`, invented `e6-export-verified`. After: probe E6 `failed` with 0 payloads; `verify_local S7_E6_runner_gates`; `test_research_k8_real.EvidenceKindTests` (marker spoofing refused). `bundle_identity` is content hash of digests+parents, never fixed string. | GPU: complete bundle exported, hashes validated, each required checkpoint `load_checkpoint`-verified fresh with real config/arch; incomplete lists missing and does not satisfy acceptance. |

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
`type(ops)`/`hasattr` branches, fixed `[259,…]` prompts, substring stream matching,
synthetic fallback rows, random-init-instead-of-restore, invented
action/call/node counts, `frozen-…` strings, same-weight T0/T1 + `50/50`
receipts, stored-answer-as-execution, discarded tiny-model migration,
`_FakeTrainer` + `run_fixture_generation(confirmed=True)` + `P_fixed=M2` +
task-count updates + constant isolation flag, JSON-only E6 + substring parent
checks + metadata-only `cmd_export`.

## Debug hardening in this pass (audit-driven, all covered by tests)

- Receipts: `to_dict` puts `extra` first (reserved keys win) + `validate`
  refuses colliding extras; direct-ID parents require a completed ledger
  receipt; ledger queries `ORDER BY rowid`; learned validation requires
  explicit `phase`; `qualifies_for_campaign` enforces work for training.
- Checkpoints: namespaced `update-…-<phase>-<arm>-<seed>` dirs (no cross-arm
  collision, legacy names unchanged); namespaced lineages never inherit global
  `LATEST`; exact segment lineage (no `1701 in 21701`).
- Ops: `apply_update` carries `pair_rows` in the protocol; deprecated
  `save_checkpoint` fails loudly on bad paths/seed/counters/identities;
  `restore_parent` no longer swallows config errors; eval counters derive from
  `new_tokens` (never case arithmetic); fixture publishes get per-handle seq.
- Compiler: exact dict-subset teacher only (no substring fallback), no fixed
  `[259]` fallback, all queries as candidates (no `[:8]` truncation); program
  uses its single real history action.
- Executors: no `hasattr`/`type` branches; E1/E3/E4 publish directly; E2 exact
  arm segment + explicit `eval_cases` + strict counter keys; E3 exact 75/25
  with insufficient-rows refusal + exact-split leak check + retention `>0`;
  E4 real gate-vs-optimizer inventory; E5 explicit anchor + `lookup_key`
  lineage + meta-confirmation only + fixture-only doubles with production
  refusal; E6 `bundle_identity` content hash + `validate()` + recursive
  manifest walk.
- Runner/worker/spawn: E5 parents wired (`E1-B-1701/1702`); specs carry
  `job_id/slot/parent/targets`; worker refuses missing training targets;
  thread test path carries the full spec; worker failures carry
  `evidence_kind: fixture`.
- Tests: `test_research_k8_real.py` 13 checks (exact match, namespaces,
  propagation incl. full-spec thread path, evidence/marker-spoof refusal,
  anchor, E2 explicit, E3 exact multiple); `verify_local.py` 14/14.

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
