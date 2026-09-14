# Operational learner handoff O01–O10 (code-ready, no GPU run)

Source base: `373a9fe` (BRAMASTRA, origin/BRAMASTRA in sync) plus the work
below. Implements
`engineering/next_phase_20260914/{README,ASSESSMENT,ARCHITECTURE,EXECUTION,READINESS}.md`
O01–O10 in dependency order, preserving the reviewed exact-lineage,
collision-free checkpoint, real-supervision, strict-evidence and
parent-restoration work. Root `experiment.md` stays authoritative for
treatments, data and statistics. No local optimizer updates, no paid
compute, no readiness bypass: readiness dispositions are untouched (chief
owns them) and every local check below reports zero optimizer steps.

## Identities (frozen protocol)

- `source_closure_sha256`: `8d45768675381218b1fa7c7b91b23e6b2bb79d61d3a9b3bd3401ba6ad0217b3a`
- `tokenizer_identity` (`bramastra-byte-260/v1`): `4db7ac566c37acffcac8d1c23895a5c3610285ceba4c7cd544d779ac13893c09`
- `k8_campaign_config.identity` (vocab260/L8/W256/H4/FFN704/ctx512, AdamW
  LR 3e-4 / WD 0.01 / clip 1.0, betas 0.9/0.95): `5d9cf2155c9f84e034313cf5a605bf806a66bea8babcb23b446622cc1b5c4ade`
- Model: vocab260/L8/W256/H4/FFN704/ctx512, 6493952 parameters + 2 gate
  scalars for S1. Allocation: single 480min campaign / 960 provisioned
  GPU-min; E0 and full share it (kernel restarts never reset the ledger).

## Commands (from `bramastra-build-worktree`, `$env:PYTHONPATH='.'`)

```powershell
python engineering/reports/K8_SEMANTIC_REVIEW_20260914/probe.py
python -m pytest tests/test_research_k8.py tests/test_research_k8_readiness.py tests/test_research_k8_launch_gate.py tests/test_research_k8_real.py tests/test_research_k8_operational.py -q -p no:cacheprovider -o addopts=
python -m pytest tests/test_research_checkpoint.py tests/test_research_learning.py tests/test_research_contracts.py -q -p no:cacheprovider -o addopts=
python engineering/reports/K8_REAL_EXECUTION_20260914/verify_local.py
python engineering/experiments/K8_20260913/validate_design.py
git diff --check
```

Results on this worktree:

- Semantic probe: E5 `failed`/0 updates/0 calls with training forbidden,
  E6 `failed`/0 payloads without payloads, `publish_checkpoint` present.
  Fixture paths report failure, never success.
- `35 passed` (K8 slice + readiness + launch gate + real-execution
  regressions, single invocation).
- `29 passed` (operational O01–O10 acceptance, single invocation).
- `53 passed, 5 skipped` (checkpoint + learning + contracts).
- Combined: `117 passed, 5 skipped` across all eight files.
- `verify_local.py`: `14/14 passed, 0 optimizer steps, 0 GPU`.
- `validate_design.py`: no errors, 0 optimizer updates.
- `git diff --check`: clean.

## Acceptance rows (one per package)

| ID | Changed production symbols | Local evidence (no training) | Remaining GPU-only checks |
|---|---|---|---|
| O01 session authority | `phases/session.py` (new: `LearnerSession`, `bind_reservation`, `bind_job_reservation`, `check_update_admission`, `drive_noop_boundary`, `step_or_noop`, `job_reservation_record`, `ledger_stepping_allowed`); `phases/ops.py: initialize_random` (unbound by default); `phases/e1.py,e3.py,e4.py` (bind + receipt `authority`); `runner.py` (reservation fields in specs); `worker.py` + `process_supervision.py` (reservation/allocation/deadline plumbed) | `test_research_k8_operational.O01SessionTests` (5): missing authority refuses finalize with 0 steps; job/device/phase/source/deadline mismatches refuse; bound simulated reservation reaches the no-op boundary (admission passes, named grads finite, counters unchanged); doubles classify `zero-update-double`. E1/E3/E4 with ProductionOps locally take no-op boundaries (`noop_boundaries` in receipts) and fail honestly on zero commits — fields alone never authorize a local step. | Owner run: runner reserves per slot, worker binds, first `finalize_update` admitted against the live allocation; rebinding a resumed session keeps charged work (E0 resume proof covers the mechanism). |
| O02 windows + checkpoints | `session.drive_noop_boundary`; `e4._prove_architecture_on_handle` (backward-only gate proof); tests only otherwise (trainer + checkpoint paths already complete) | `O02WindowTests` (3): identical windows give identical gradients with 0 steps; pair-without-rows refused at finalize with 0 steps; real-API round trip (save/load/restore_verify) plus wrong-config and stale-writer refusals. E4 gate-gradient proof runs backward-only with grads discarded. | GPU: unequal-microbatch partition equivalence under controlled stochastic layers; full-schedule LR application already in trainer. |
| O03 calibration | `campaigns/calibration.py` (new: `select_update_target`, `select_confirmation_inventory`, `freeze_protocol`, `load_frozen_protocol`, `check_hardware_match`); `worker._run_e0` pilot block + `calibration_samples`; `_pilot_full_profile` (CUDA-gated full-config heaviest-arm timing + eval-loop throughput); `runner._maybe_freeze_protocol`, `_load_usable_protocol`, `phase_targets` ledger event | `O03CalibrationTests` (3): `min(4000,floor(0.75*slot/worst))` values, below-minimum/insufficient/nonpositive/unknown refusals; 128/64/32 step-down + `EVALUATION_BUDGET_INSUFFICIENT`; freeze-once, hardware-match/mismatch, missing-freeze None. `_pilot_full_profile` on CPU returns `skipped-local-no-accelerator` (covered in O10 E0-shape test path). Runner without frozen artifact labels specs `uncalibrated-minima` (never masquerading). | Owner E0 on 2×T4 measures worst update seconds (B package, full config) + eval cases/sec; runner freezes `frozen_protocol.json`; E1+ consume calibrated targets; changed hardware recalibrates. |
| O04 episode kernel | `cognition/episode.py` (extended: `EpisodeEvent.model_origin/planner_meta`, `ImaginedNode`, `mark_conflicts`, `ModelWorldModel`, `CannedWorldModel`, `SymbolicReferenceAdapter`, `_default_guess`, workspace-gated admission, inquiry+submit budgets, compact lossless rendering with explicit render window); `environments/k8_live.py` (new: live rule/inventory/program/tool envs + `generate_live_mechanism` eval stream + `oracle_answer` controls) | `O04KernelTests` (5): live terminal traces per family with event-chain integrity + counters reduced from events; imagined nodes separate from history; invalid actions become costed outcomes; truncation explicit; negative controls reflected (constant value_fn recorded; seeded reorder identical). All-families live smoke (rule/inventory/program/tools complete with success). | GPU: model-backed modes with the restored B model; planner with `ModelWorldModel` over the shared decoder; longer-horizon behavior under real 512-ctx pressure. |
| O05 matched E2 | `phases/e2.py` (rewritten evaluation core: matched live groups, B + A restores, 7 modes with per-mode adapters/checkpoints, goal-swap before render, contradiction probes, complementary coverage, planner calibration gap, origin-gated evidence) | `O05MatchedEvaluationTests` (2): doubles complete fixture with 7 mode stats, paired deltas, goal-swap/contradiction/complementary rows, `k8-live-eval/v1` source; A-only parent fails. `verify_local` failure paths preserved. | Owner run: 32 matched mechanisms × 7 modes on frozen B/A payloads; paired B−A deltas, goal-swap table, contradiction detection rate, complementary coverage, planner predicted-vs-actual gap. |
| O06 tool E3 | `phases/e3.py` (execution-grounded `_tool_batch` with executed-vs-stored consistency refusal; `_load_protected_canonical` sealed+heldout exclusion incl. replay canonical check; retention `evaluated>0` gate) | `O06CanonicalTests` (2): protected/heldout sets load; corrupt tool row (executed ≠ stored) refused. Existing mixture/replay/retention gates unchanged and passing. | GPU: T0/T1 on full tool bundle with 75/25 replay; acquisition vs retention per family; no average hides forgetting. |
| O07 arch E4 | `phases/e4.py` (`_prove_architecture_on_handle`: gate-gradient backward proof, packed-segment isolation, timed execution cost on the training handle; fixture path records) + real gate-vs-optimizer inventory check | `O07GateTests` (2): fixture migration records verify; unmigrated handle refused; arch-proof fixture path returns all three proofs. Existing `GatedArchitectureTests` still pass. | GPU: S1/S0 full-stream training on migrated handles; gate values/gradients/cost compared at equal updates. |
| O08 trial service | `campaigns/trial_service.py` (new: `TrialRequest/TrialResult`, `run_trial` with explicit `production`/`test_substitute` boundaries, M2 `prepare_trial_handle` migration, ledger-gated stepping, boundary-only records, failed-trial retention) | `O08TrialTests` (3): request validation (unknown method, missing identity, past deadline); no-ledger locally → stepping denied; production trial with tiny real trainer + simulated reservation reaches the boundary (`boundary-only`, 0 steps). | GPU: per-trial 45s allowances with real accumulate/finalize, published trial checkpoints, measured query/protected outcomes. |
| O09 proposer/successor | `phases/e5.py` (rewired: archive via trial service, `_proposer_batches` from measured bests, `_train_method_selection` real/noop/double paths, `_capture_proposer_choice` real-decoder-first with labeled teacher fallback, P0 trained before P1/P_fixed forks, real confirmation trials + same-table regret scoring, `_summarize_confirmation` attribution, learned evidence only with real payload checkpoints) | `O09ProposerTests` (3): batches from measured bests; doubles capture `teacher-majority-control` with best choice; attribution rules (supported / extra-training-only / negative). E5 doubles complete fixture (`capture_origin` labeled, regret table present). E5+ProductionOps locally fails honestly (no allocation). | GPU: 12-task archive trials, P0 method-token training, decoder-captured P1 method, equal-time P1/P_fixed updates, 6-task confirmation with regret/retention; `recursive-benefit-supported` only if P1 beats P_fixed. |
| O10 notebook/export | `notebooks/bramastra_k8.ipynb` (config/tokenizer/model/allocation prints, export failure check); `worker._pilot_full_profile` CUDA gate | `O10NotebookTests` (2): notebook calls all five entry points, prints identities/plan, checks failures, shares one RUN_DIR; export of real non-updated test payloads + fresh-subprocess reload (`fresh-reload-ok`). Pilot returns skipped shape on CPU. | Owner: run cells in order on 2×T4 (see launch steps); copy `K8-results/` to persistent storage. |

## What was preserved vs finished

Preserved (reviewed correct): exact parent lineage + `run_dir` records, namespaced checkpoint dirs + fencing, real supervision channels, evidence-kind validation, E0 resume comparison, slot/termination/device/lease supervision, E3 75/25 + tool verification, E4 zero-gate migration, E5 anchor/confirmation isolation, E6 manifest re-hash + payload gates, readiness gate (untouched).

Finished (were interfaces/fixtures): reservation→optimizer admission bridge with ledger backstop; microbatch/window/checkpoint local proofs; E0 calibration artifact + runner consumption; live env loop + matched E2 with per-mode adapters/checkpoints; execution-grounded tool history + canonical exclusion; on-handle arch proof + optimizer inventory; measured trial service with explicit boundaries; decoder-first proposer capture with labeled fallback; P0/P1/P_fixed training paths; real confirmation tables with regret; notebook identity/failure coverage.

## Launch steps (owner, existing 480min allocation)

1. Open `notebooks/bramastra_k8.ipynb` on the 2×T4 session; run cell 1 and confirm source/config/tokenizer identities match this handoff plus 2 live GPUs.
2. Run prepare (cell 3) with the pinned 4096/256/256/128 + 256/64 + 24/6/6 recipe; run validate (cell 5) — must pass before launch.
3. Run E0 (cell 7): two workers verify hardware, gradients, 3-vs-1+2 resume, pilot timing; confirm `protocol_frozen` event with calibrated targets in the ledger.
4. Run full (cell 9): E1→E6 execute under the frozen protocol; training stops at minute 450.
5. Run summarize (cell 11) and export (cell 13, now failure-checked); copy `K8-results/` to persistent storage before the session ends.
6. Chief compares results against `experiment.md` screening criteria (paired B−A ≥3pp both seeds, retention ≤2pp regression, mechanism-cluster intervals).

## Pending chief/owner actions (no further local work authorized)

1. Chief reviews this diff + evidence and decides code-readiness (readiness dispositions unchanged by this delivery).
2. Owner runs E0 (GPU qualification) then the full campaign within the existing allocation; E0 timing freezes the protocol actually consumed downstream.
3. GPU-only verifications outstanding: real optimizer-step admission under allocation, partition equivalence with stochastic layers live, planner with learned world predictions, P0 decoder-captured choices beating the teacher, full-bundle acquisition/retention and confirmation outcomes.

No local optimizer updates were performed (all 117+29 tests assert zero steps; trainers ran accumulate/backward only with discarded gradients, doubles performed none). No paid compute was used. No `skip-readiness` path exists. Failed diagnostics (probe E5/E6 refusals, uncalibrated-minima labeling, boundary-only trial records) are preserved as evidence, not hidden.
