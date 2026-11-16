# Planthir Priorities 1–6 Reliability & Universal Infrastructure Upgrade

**Date:** `2026-11-17`
**Base Commit:** `00b2128b34755d332414ffae8eb78dd3f430e04f` (`origin/planthir`)
**Qualification Scope:** Engineering controls, real subprocess crash/publication recovery, non-V8 X1 R/D2 backend parity, resource/scheduler accounting, typed design/cohort/teacher audits, streaming raw evidence/custody replay, and declarative CLI (`CPU canary <= 100k params, <= 32 samples, <= 8 updates`). Interactive multi-seed GPU qualification and paired 6-task productivity observations remain pending.

## Implemented Capabilities (Priorities 1–6)

1. **Priority 1 — Backend Contract, Full Provenance & Non-V8 X1 R/D2 Backend**
   - Added `anra_v5/planthir/backend.py` (`BACKEND_CONTRACT_VERSION`, `PROVENANCE_IDENTITY_SCHEMA`, `BackendCapabilities`, `PlanthirBackend`, `V8PlanthirBackend`, `build_provenance_identity`, `verify_provenance_identity`).
   - Added `anra_v5/planthir/adapters/x1_rd2_adapter.py` (`X1RD2Backend`, `X1RD2EnvironmentKernel`, `X1RelationalModel` supporting `X1-P`, `X1-R`, `X1-R-noedge`, `X1-P-capR`, and `X1RD2Recipe` supporting `ce`, `pair_ce`, `pair_eq`, `pair_eq_cf` grouped D2 losses).
2. **Priority 2 — Real Subprocess Crash Recovery & Atomic `CURRENT.json` Publication**
   - Added `trigger_fault_point` (`execution.py`), `publish_bundle_generation`, `resolve_published_generation`, and `quarantine_partial_generations` (`evidence.py`) using immutable `generations/<gen_id>/` + atomic `CURRENT.json` pointer swaps while preserving top-level compatibility archives.
   - Verified child `os._exit(99)` kills across `after_component:model.bin`, `after_journal_append`, `after_checkpoint_commit`, `publication_stage`, and `publication_pointer`, plus corrupt journal rejection and concurrent reader isolation.
3. **Priority 3 — Measured Resource Controls & Independent Multi-Worker Scheduler**
   - Added `anra_v5/planthir/resources.py` (`ResourceMonitor`, `ResourceQuota`, `RESOURCE_RECEIPT.json`) wired into `CartridgeRunner`, measuring per-phase wall time, CPU RSS (`win32_get_process_memory_info`/`psutil`), disk usage, and staging cleanup on quota breaches.
   - Added `anra_v5/planthir/scheduler.py` (`IndependentWorkerScheduler`, `WorkerJobSpec`, `COORDINATOR_RECEIPT.json`) with per-worker directory locks, CUDA device pinning validation, and partial-failure containment.
4. **Priority 4 — Typed Design, Transitive Cohort Audits & Nonanticipating Teacher**
   - Added `anra_v5/planthir/design.py` (`TypedFactorSpec`, `TypedContrastSpec`, `audit_typed_experimental_design`, Union-Find `validate_cohort_and_split_transitivity`, paired-seed `analyze_paired_seed_runs` with Holm-Bonferroni correction) and `validate_nonanticipating_teacher` in `anra_v5/planthir/instrument.py`.
5. **Priority 5 — Chunked Streaming Raw Evidence, Custody & Independent Replay**
   - Extended `anra_v5/planthir/evidence.py` with `stream_raw_evidence_chunks`, `verify_raw_evidence_chunks` (write-time byte quotas), `raw_evidence_link` inside `CAPSULE_MANIFEST.json`, and `result.zip.sha256` sidecars.
   - Added `anra_v5/planthir/replay.py` (`build_custody_receipt`, `independent_replay_bundle`) verifying custody chains and reconstructing models in isolated temp workspaces.
6. **Priority 6 — Data-Only Spec & Composable CLI + Productivity Protocol**
   - Upgraded `anra_v5/planthir/agent_tools.py` & `__main__.py` with strict data-only `anra-planthir-spec-v1` CLI subcommands (`schema`, `init`, `validate`, `run`, `resume`, `inspect-run`, `verify-bundle`, `replay-bundle`, `explain-error`), gating obsolete CLI launchers (`run-validation-study`, `run-gpu-validation`).
   - Added `anra_v5/planthir/productivity.py` and `research/planthir/SIX_TASK_PRODUCTIVITY_STUDY_PROTOCOL.md` (`status: STUDY_OBSERVATIONS_PENDING`, no fabricated efficiency %).

## Verified Checks (CPU Environment)

- `python -m pytest -q`: **`506 passed, 1 skipped, 14 subtests passed in 903.90s`** (exit `0`).
- `python -m anra_v5.planthir calibrate`: **`6/6` injected faults detected**, `healthy_v8_kernel_passed: true` (exit `0`).
- `python -m anra_v5.planthir agent-info` & `python -m anra_v5.planthir schema`: exit `0`.
- `python -m tools.verify_elendril_evidence`: exit `0` (`237` checkpoint manifests indexed).
- `python -m elendril audit --output ...`: exit `0`.

## Pending Execution

- Owner-run bounded multi-seed GPU validation on local RTX or Kaggle CUDA (`cuda_measured=true`).
- Empirical logging of counterbalanced 6-task productivity runs under `research/planthir/SIX_TASK_PRODUCTIVITY_STUDY_PROTOCOL.md`.
