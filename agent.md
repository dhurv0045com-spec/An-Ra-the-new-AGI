# Planthir agent guide

Follow root `AGENTS.md` and read [the reliability and upgrade handoff](research/planthir/GEMINI_RELIABILITY_AND_UPGRADE_HANDOFF.md) alongside [the 6-task productivity study protocol](research/planthir/SIX_TASK_PRODUCTIVITY_STUDY_PROTOCOL.md) before changing or using Planthir. `purpose.md` and `research/planthir/agent.md` retain historical context and earlier Python-level notes; this root guide defines the current declarative CLI and API contract.

## Agent-first declarative workflow

Prefer the composable data-only JSON CLI (`anra-planthir-spec-v1`) so agents avoid writing training loops or shell boilerplate:

1. **Discover contract & schema (read-only):**
   - `python -m anra_v5.planthir agent-info`
   - `python -m anra_v5.planthir schema`
2. **Calibrate the instrument on CPU (`6/6` injected faults detected):**
   - `python -m anra_v5.planthir calibrate`
3. **Initialize & fail-closed validate a data-only spec (`backend_id`: `v8` or `x1_rd2`):**
   - `python -m anra_v5.planthir init --backend x1_rd2 --out spec.json` (or `--backend v8`)
   - `python -m anra_v5.planthir validate --spec spec.json --canary`
   - Unknown keys, `eval`, dotted Python import strings, shell fragments, non-varying declared factors, parameter/precision/seed confounds, and transitive cohort/split leaks fail closed with exit code `1`. Use `python -m anra_v5.planthir explain-error --code <CODE>` for exact repair steps.
4. **Run, inspect, and resume cleanly:**
   - `python -m anra_v5.planthir run --spec spec.json --work-root runs/exp1 --canary`
   - `python -m anra_v5.planthir inspect-run --work-root runs/exp1` (strictly read-only inspection)
   - `python -m anra_v5.planthir resume --spec spec.json --work-root runs/exp1 --canary` (repairs uncommitted milestone bindings, quarantines partial `.staging-*` dirs, and resumes from committed `CheckpointStore` state; never force a mismatched identity/config onto an existing `work_root`).
5. **Verify custody & independently replay exported bundles:**
   - `python -m anra_v5.planthir verify-bundle --bundle-dir runs/exp1/result_bundle` (checks `CURRENT.json` generation pointer, `result.zip <= 20 MiB`, `checkpoint_full.zip`, `.sha256` custody sidecars, and manifests)
   - `python -m anra_v5.planthir replay-bundle --bundle-dir runs/exp1/result_bundle --output runs/exp1/REPLAY_RECEIPT.json` (extracts into an isolated temp workspace and verifies model reconstruction & predictions).

## Backends, guarantees & open limits

- **Registered backends (`anra_v5/planthir/backend.py`):** `v8` (`V8PlanthirBackend`) and non-V8 relational/D2 grouped loss `x1_rd2` (`X1RD2Backend` supporting `X1-P`, `X1-R`, `X1-R-noedge`, `X1-P-capR`, and `ce`/`pair_ce`/`pair_eq`/`pair_eq_cf`). Custom `trainer_factory` hooks are rejected by qualified backends.
- **Measured runtime & bundle publication controls:** `CartridgeRunner` records `PROVENANCE_IDENTITY.json`, `RESOURCE_RECEIPT.json` (wall time, OS/psutil peak RSS, optional CUDA peak memory, disk usage), publishes bundles atomically via `generations/<gen_id>/` + `CURRENT.json` + `.quarantine/`, and supports parallel isolated subprocess scheduling (`IndependentWorkerScheduler`, `COORDINATOR_RECEIPT.json`).
- **Typed design, cohorts & streaming:** `anra_v5.planthir.design` provides `TypedFactorSpec`, `TypedContrastSpec`, union-find `validate_cohort_and_split_transitivity`, and paired-seed Holm-Bonferroni `analyze_paired_seed_runs`. `validate_nonanticipating_teacher` catches teachers that peek at hidden state. `stream_raw_evidence_chunks` streams bounded `.jsonl`/`.npz` chunks with write-time byte quotas and capsule provenance links.
- **CPU canary bounds:** Local CPU execution is restricted to `<= 100,000` parameters, `<= 32` samples, and `<= 8` updates. Obsolete CLI launchers (`run-gpu-validation`, `run-validation-study`) exit `1` on CLI; interactive owner GPU qualification (`torch` CUDA multi-seed runs) remains pending owner execution. Productivity study benchmarking (`anra_v5.planthir.productivity`) reports `STUDY_OBSERVATIONS_PENDING` until logged counterbalanced paired runs are analyzed.

## Focused verification commands

- Full Planthir suite: `python -m pytest tests/test_planthir_*.py -q`
- Repository evidence verification: `python -m tools.verify_elendril_evidence`
- Before concluding work: never commit `.zip`, `.npz`, `.bin`, or raw run folders (`AGENTS.md`); append a concise `<= 5`-line dated entry to root `progress.md`; distinguish code, CPU checks, and pending GPU/productivity runs.
