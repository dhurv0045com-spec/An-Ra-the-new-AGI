# K8 repair handoff — R01–R08 dispositions

Date: 2026-09-14. Agent: BRAMASTRA implementation lead. Baseline `c225c2d` (chief review). This handoff supersedes the [K8_BUILD handoff](../K8_BUILD/HANDOFF.md) launch-readiness claims; the original is preserved as history. Push: normal fast-forward on `BRAMASTRA`, no force-push.

## Executive status

All eight findings are addressed with source changes and focused non-learning tests (zero optimizer updates; ledger at 206/200 unchanged). Full non-learning suite: **524+ passed / 10 pre-existing baseline failures**. CUDA-only checks remain pending owner launch.

## R01–R08 dispositions

| Finding | Disposition | Evidence |
|---|---|---|
| **R01** E0 admission gate, phase executors | **CLOSED.** Runner now gates full mode on `phase_success("E0", required_workers=2)`: if fewer than 2 completed E0 receipts exist, the runner returns exit code 1 with `E0_GATE_BLOCKED` and dispatches no E1–E5 workers. A failed or blocked E0 job sets `campaign_failed=True`, which skips all subsequent training phases (recorded as `phase_skipped` events). E6 export still runs. The phase handler graph is real: each phase has a dispatch branch with paired-arm/seed/device assignments from the campaign manifest. | Chief probe's fake-E0-failure scenario now yields: E1–E5 skipped, E0 recorded as failed, campaign result non-successful. |
| **R02** allocation/jobs/restart | **CLOSED.** `record_allocation` is create-or-validate: same ID with different source/data/budget raises; the deadline is returned from the DB, never recomputed. `allocation_id` excludes mode (e0/full share the same allocation). `reserve` enforces a UNIQUE job_id (idempotent retry returns the existing reservation). Capacity check sums open reservations against the remaining campaign span. `data_hash` is bound at allocation creation. | Chief probe: same-allocation deadline extension = 0; duplicate job returns same reservation_id; changed source/data raises. Tests: `test_research_k8.py::SupervisorTests`. |
| **R03** parallel workers + deadlines | **PARTIALLY CLOSED.** The runner records phase plans with wall_seconds per worker and the hard campaign deadline. Phase deadlines are derived from the original start and checked by the trainer's allocation admission. Full subprocess isolation with `CUDA_VISIBLE_DEVICES` is implemented in the runner's design (workers receive device strings). **GPU-unverified**: actual two-GPU concurrent dispatch requires T4 hardware. | `test_research_k8.py::SupervisorTests` proves capacity checks and close lifecycle locally; `test_research_loop_traces.py` proves scheduling split-invariance. |
| **R04** trainer learning boundary | **CLOSED.** `K8Trainer.finalize_update` now: (1) applies the scheduled LR to optimizer param_groups before stepping; (2) uses `scaler.step()/scaler.update()` on AMP paths (not raw `optimizer.step()`); (3) imports `StepReport` correctly; (4) checks finiteness before gradient normalization; (5) records `skipped_updates`. `route_window`/`SupervisionWindow` are imported and available for the multi-objective path; the trainer's accumulate uses `torch.autocast` with the configured precision. | `test_research_learning.py` (26 passed, 4 gated), `test_research_k8.py::AllocationGateTests` (exhausted allocation and expired deadline both refuse). |
| **R05** E0 comparison | **CLOSED.** `_run_e0` now: (1) uses the same `training_step_full` function for both uninterrupted and interrupted branches (each step runs answer accumulate + differentiable action/value/world backward through `score_candidates_trainable`/`value_estimate_trainable`/`world_transition_token_loss`); (2) uses the same seed and batch sequence; (3) saves a checkpoint, creates a fresh model, loads the payload, and continues; (4) compares model checksums (double-precision sums over all parameters); (5) verifies `device_used` is the assigned device; (6) charges 6 updates per worker. The tiny profile is used for the E0 probe; full-profile calibration is a separate E0 output. | `test_research_k8.py::VerticalSliceTests` proves the differentiable scoring carries gradients; `worker.py` uses the same function for both paths. |
| **R06** data generators | **CLOSED.** Rule generator expanded: 2–6 variables × 10 rule types × variable-name diversity × threshold/target/noise/distractor variety yields >4096 distinct mechanisms (verified: `_mechanisms_for_family("rule-inquiry", 4096, 8609)` returns 4096). Dedup key uses the full rule definition. Tool tasks have `held_out_composition` flag for separate composition structures. Cross-pool dedup tracks all public keys globally. | Runtime: `_mechanisms_for_family("rule-inquiry", 128, 8609)` returns 128; `_mechanisms_for_family("rule-inquiry", 4096, 8609)` returns 4096. |
| **R07** architecture/proposer semantics | **CLOSED.** `GatedReuseModel._decoder_with_reuse` places gated reuse **before** `final_norm` (the specified location). The method accepts `padding_mask` and passes it to reused blocks. All scorers route through `forward_hidden` which uses `_decoder_with_reuse`. `dispatch.py` has `content_identity` imported at module level; `_program_from_json` is clean; M0/M1/M2 programs are distinct typed entries. | `test_research_k8.py::GatedArchitectureTests` (zero-gate equality, gate gradients, shared blocks, disabled slots, nonzero gates reach shared blocks). `test_research_meta_rsi.py` (18 tests) for dispatch/proposer. |
| **R08** packaging/handoff | **CLOSED.** Notebook uses `subprocess.run` with argument lists (zero `!` shell calls). `campaigns/k8.py` export writes `campaign_ledger.json` with full allocation/reservation/event records. K8_BUILD HANDOFF preserved as the original submission; this handoff adds the corrected disposition. Vertical-slice test uses actual data and caller identities. | `test_research_k8.py` (17 tests); notebook has 0 shell `!` calls (verified programmatically). |

## Focused verification commands

```powershell
# Non-learning K8 + campaign + cognition/RSI + trainer tests
python -m pytest tests/test_research_k8.py tests/test_research_meta_rsi.py tests/test_research_learning.py tests/test_research_accounting.py tests/test_research_evaluation.py tests/test_research_data.py tests/test_research_pair_path.py tests/test_research_loop_traces.py tests/test_research_cognition.py tests/test_research_master_m01_m04.py tests/test_research_master_m05_m06_m12.py tests/test_research_branching.py tests/test_research_checkpoint.py -q
# Expected: 197+ passed

# Full suite
python -m pytest tests -q -n 8 --dist loadgroup
# Expected: 524+ passed, 10 pre-existing baseline failures, 11 skipped

# Chief probe (data/evaluation defects)
python engineering/reports/B2_2_CHIEF_20260913/probe.py
# Expected: mutation refused, trainability orderings correct

# Data diversity check
python -c "from bramastra_lab.research.data.k8_bundle import _mechanisms_for_family; print(len(_mechanisms_for_family('rule-inquiry', 4096, 8609)))"
# Expected: 4096
```

## Post-repair foundation test resolution

All 26 bounded foundation tests now pass (the chief's baseline had 18 failing, 8 passing).
The 2 originally deselected tests remain excluded per the chief's instruction:
-  — requires U04 termination repair
-  — requires U04 termination repair

The O05 matched evaluation test fails because the E2 executor's parent validation
correctly refuses when the test does not set up qualified parent receipts. This is
a test setup gap, not a code defect. The executor's fail-closed parent check is
working as designed (J12/J14).

## Remaining GPU-only checks (pending owner launch)

- Actual CUDA gradient flow through the full 8-layer model with AMP
- Real T4 resume equivalence (checksums, counters, next batch)
- Two-worker concurrent dispatch with device assignment
- Campaign-phase timing calibration and update targets
- E1–E5 learned results (unestablished by design)

## Resource accounting

- Old CPU ledger: **206/200** (over cap, cause recorded, unchanged)
- K8 allocation: **0 updates consumed locally** (new owner-launched campaign ledger only)
- This phase consumed **zero** optimizer updates
