# Execution Agent Manual (`execution_agent.md`)

> **Role:** Fast / High-Precision Experiment Builder & Execution Agent
> **Platform Baseline:** **Planthir / Palantiri Level 7 (`PLANTHIR_LEVEL = 7`, schema `anra-planthir-level7-v1`)**
> **Companion Manual:** [`orchestrator_agent.md`](orchestrator_agent.md) (for the Expensive Research Architect / Planner Agent)
> **Infrastructure Reference:** [`agent.md`](agent.md) (`anra_v5.planthir` / `anra_v5.palantiri` / `anra_v5.planter`)
> **Mirrors:** Kept identically at root `execution_agent.md`, `anra_v5/planthir/execution_agent.md`, and `research/planthir/execution_agent.md`. Always follow root [`AGENTS.md`](AGENTS.md).

---

## 1. Mission, Branch-Neutral Policy, & Repository Hygiene

When the user or Orchestrator instructs you:
> *"Read `execution_agent.md` and execute `<spec_path>`."*

Your job is to convert that research specification into **10/10 genuine, verified, production-grade code, tests, CPU qualification receipts, and (when requested) a source-pinned Kaggle/Colab notebook** — using **`anra_v5.palantiri`** (`anra_v5.planthir` / `anra_v5.planter`) for custody, trainers, stats, and archives so you invest compute in **real science and crash-free GPU execution**, not boilerplate reinvention.

- **Branch-Neutral Policy:** Commit and push on the `<target-branch>` specified by the user or spec — or otherwise on your currently checked-out active branch (`git branch --show-current`, such as `hitchhiker`, `Elendril`, or `palantiri`). Never invent an unrequested branch name. *(Only if `anra_v5/planthir` is absent or older on the current branch, pull it from `origin/<source-branch>` using [`agent.md`](agent.md) Section 0.)*
- **Repository Hygiene (`AGENTS.md`):** Never commit `.zip`, `.npz`, `.bin`, `.pt`, or raw `runs/` folders. Commit only focused Markdown reports and curated JSON under `Experiment Result/<Program>/`. Append a `<= 5`-nonempty-line entry (including heading) to root [`progress.md`](progress.md) before your final commit/handoff.

---

## 2. Quality-First Override Law (CRITICAL — Read First!)

**Planthir / Palantiri is in an early starting condition.** It accelerates 80–90% of experiment infrastructure, **but it may have missing features, rough edges, or upstream bugs.**

**Your #1 priority is 10/10 scientific & production quality — NEVER degrade science or GPU reliability just to obey a soft rule or force an incomplete Planthir helper.**

| Category | Status | Rules |
|---|---|---|
| **HARD SCIENTIFIC LAWS** | **NEVER BREAKABLE** | 1. **Zero stubs / zero fake numbers:** Every requested model, control, sham, oracle, DP solver, and causal lesion must compute real outputs. Never let an engineered check satisfy a learned metric.<br>2. **Zero leakage & complete-cohort sealed confirmation:** Disjoint `train` / `dev` / `confirm` / `transfer` splits; never select on `confirm`; write a SHA-256 lock and **NEVER open sealed `confirm` during a `PARTIAL`/deferred session** (`pending_confirm_jobs == []` required); never place truth keys (`gold`, `expected`, `truth`, `hidden`) inside `UniversalObservation.metadata`.<br>3. **Uncompressed `FULL` mode:** Never shrink seeds, dataset denominators, updates, or registered jobs inside `FULL` mode to meet a line limit or speed up a test.<br>4. **Honest provenance & reporting:** Separate CPU engineering checks from GPU scientific findings. Report negative, blocked, or failed rows honestly. Pin non-self-referential Git commit SHAs.<br>5. **Real custody & verification:** Never report replay, restore, or archive verification as passing unless you executed it after final edits. |
| **SOFT ENGINEERING DEFAULTS** | **BREAK FREELY IF IT INCREASES QUALITY** | 1. **Soft line/file budgets:** If genuine `FULL`-mode worker dispatch, batched GPU evaluation, or complete scientific controls need more lines than a soft guideline, **write the clean code needed** instead of compressing `FULL` mode.<br>2. **"Don't read internal `.py` files":** Start from Sections 4–5 below without browsing unrelated history — **BUT** whenever you hit a traceback, schema surprise, or subtle Planthir edge case, **immediately inspect the exact function in `anra_v5/planthir/*.py`** rather than guessing.<br>3. **Forcing an incomplete/buggy Planthir helper:** If `run_registered_plan`, `StaticPromptKernel`, `build_pinned_kaggle_notebook`, `admit_slice`, or a legacy evaluator bridge lacks parent-child DAG ordering, adaptive one-go wall-clock loops, `LATEST`-only checkpoint packing, mixed subprocess routing, or hits a bug, **wrap, extend, or bypass that helper cleanly inside `experiments/<name>/`** while keeping all Hard Scientific Laws intact.<br>4. **Default notebook/session flags:** `build_pinned_kaggle_notebook` defaults to `owner_authorized=True` for zero-edit Kaggle Run All (`owner_authorized=False` remains available for `--inventory-only` dry-runs); override `SESSION_MINUTES=300` (e.g., `SESSION_MINUTES=660`, `ONE_GO=True`) whenever one-click full execution is requested. |

**Mandatory Transparency Rule (`PALANTIRI_FEEDBACK`):** Whenever you break a soft rule or bypass/wrap a Planthir gap or bug, include a short **`PALANTIRI_FEEDBACK`** block in your final handoff specifying: (1) `gap_or_bug`, (2) `rule_overridden`, (3) `local_workaround`, and (4) `suggested_upstream_fix`.

---

## 3. Fast 6-Step Execution Playbook

### Step 1 — Preflight (`<30s`, Zero Wasted Turns)
1. Confirm branch and clean tree (`git branch --show-current`, `git status -sb`).
2. Read the Orchestrator specification completely. Do **not** waste turns reading unrelated historical campaigns.
3. Run Level-7 platform self-checks:
   ```bash
   python -m anra_v5.palantiri level
   python -m anra_v5.palantiri doctor
   python -m anra_v5.palantiri calibrate
   python -m anra_v5.palantiri shield
   ```
   Confirm `PLANTHIR_LEVEL = 7`, `GREEN_SIGNAL`, `6/6` faults detected (plus `SHORTCUTS_REJECTED_AND_ORACLE_QUALIFIED`), and `5/5` `LEVEL7_SHIELD_PASS`.

### Step 2 — Choose Your Integration Path
- **Path A — Declarative `spec.json` (`<50` lines):** Use when varying hyperparameters/arms on already-registered backends (`v8`, `x1_rd2`, `x1_s20`):
  ```bash
  python -m anra_v5.palantiri init --backend x1_rd2 --out spec.json
  python -m anra_v5.palantiri validate --spec spec.json --canary
  python -m anra_v5.palantiri run --spec spec.json --work-root runs/exp1 --canary
  python -m anra_v5.palantiri verify-bundle --bundle-dir runs/exp1/result_bundle
  python -m anra_v5.palantiri replay-bundle --bundle-dir runs/exp1/result_bundle --output runs/exp1/REPLAY_RECEIPT.json
  ```
- **Path B — Plugin Package (`experiments/<name>/`, Recommended):** Put all new experiment code under one clean package (`{__init__,plan,tasks,models,module,verify}.py` — separating independent receipt/claim checks into **`verify.py`** is a first-class recommended layout for multi-stage / high-study audits, read-only donor staging & release gates instead of inflating `module.py`; generate starters via `scaffold_experiment_module(..., include_verify_module=True)`) and `tests/test_<name>.py`. Register via `ExperimentModule(...).register()` + `register_planthir_backends()`, and load in CLIs via **`--backend-module experiments.<name>.module`** or `PLANTHIR_BACKEND_MODULES=experiments.<name>.module` (ensure `"experiments*"` is included in `pyproject.toml`).
- **Path C — Hybrid / Quality-First Override:** Keep `anra_v5.palantiri` for custody, Launch Gates, stats (`bootstrap_paired_families`, `build_paired_seed_observations`), and verified archives (`export_session_archives`, `verify_session_archive`), and add clean local PyTorch/NumPy/DAG orchestration inside `experiments/<name>/` wherever a Planthir helper cannot express your pipeline cleanly.

### Step 3 — Production `FULL`-Mode & Kaggle 2×T4 Checklist (10 Battle-Tested Level-7 Rules)
Follow every item here so live Kaggle 2×T4 runs never stall, overflow `/kaggle/working`, or crash mid-wave:

1. **Wrap Frozen Baselines Directly (Zero Confound):** Import `X1TypedModel`, `solve_x1_capacity_controls`, `core_x1_jobs`, `compute_x1_d2_grouped_loss`, `v5_model.core.initialize`, or **`build_x1_spec(arch, cpu_canary=True/False)`** directly; assert `verify_shared_backbone_and_head_overhead(ctrl, trt, backbone_prefix="core.", max_head_overhead_ratio=0.05, max_control_parity_ratio=0.02)`.
2. **Real Per-Job Multi-GPU Dispatch (`FULL` != Scaled-Up `CANARY`):**
   - Use **`CampaignOrchestrator.run_registered_plan(...)`**, which natively supports **`stage_gates`** (`STAGE_GATES.json`), **`depends_on` / `parent_job_id` / `required_parent_paths` / `parent_checker`** (**`BLOCKED_MISSING_PARENT`** in `BLOCKED_JOBS.json`), **`live_wall_clock_preemption=True` + `clock_fn`**, **per-`profile_group` timing**, **`u0000_smoke_fn` + `u0000_fp16_autocast_fn`**, **`byte_decode_fn` + `stage5_preflight_fn`**, and automatic wave-loop `BaseException` partial custody (`JOB_RECEIPTS.json`, `COMPLETED_JOBS.json`, `PENDING_QUEUE.json`, `REGISTERED_PLAN_STATUS.json`).
   - Only compose manual wave loops if an exotic multi-repo subprocess protocol still needs custom routing; always persist `JOB_RECEIPTS.json`, `PENDING_QUEUE.json`, and resumable session archives on interrupt/deferral.
3. **Smoke-Test EVERY Distinct Architecture, Legacy Adapter, Lossy Byte Decoder & FP16 Autocast Path at `u0000` (CRITICAL):**
   - Pass **`u0000_smoke_fn`** — plus **`u0000_fp16_autocast_fn` / `require_u0000_fp16_autocast=True`** (`validate_fp16_autocast_forward_backward`, `validate_model_and_gradients(..., check_fp16_autocast=True)`, and `ExperimentModule(..., supports_fp16=True)`) for AMP/FP16 runs, **`byte_decode_fn`** (validated against `0xB5` / malformed bytes via `validate_byte_decoder_resilience`), and **`stage5_preflight_fn`** — to `run_registered_plan(...)` AND test in `pytest`: a padded CPU `u0000` smoke check (init -> 1 train step under `torch.amp.autocast(..., dtype=torch.float16)` when AMP is enabled -> 1 eval step -> checkpoint save/load -> unconstrained `0xB5` byte decode) for **EVERY** distinct architecture class (e.g., `B0`, `P`, `F`, `R`, `U`, `S50`, and every new treatment/control) AND every legacy subprocess worker path.
   - **NEVER call strict `UTF8.decode(...)` on unconstrained neural byte emissions** (`generate_bytes` / evaluation rollout / decoding loop): a single `0xB5` byte after hours of training crashes Stage 5 with `ValueError`. Always use **`UTF8.is_ascii_stop_token`**, **`UTF8.decode_lossy`**, or **`decode_generated_bytes`** for unconstrained model outputs; reserve strict `UTF8.decode` ONLY for deterministic oracle/ground-truth contracts. Do not run FP32-only smoke tests on AMP models (`-1e9` / `(~valid_tok) * (-1e9)` overflows `c10::Half` on padded attention!) or assume testing one model family covers all of them (`AttributeError: '_S' object has no attribute 'base_spec'` on `B0` stops live waves at step `u0000`!).
4. **Complete-Cohort Confirmation Lock Guard:** Never open sealed `confirm` during a `PARTIAL` or deferred session (`pending_confirm_jobs == []` and 100% of confirmation-eligible jobs verified complete required before unsealing `confirm`).
5. **Batch ALL Multi-Case Evaluation Loops (50–100× Speedup):** Use **`batched_rollout(..., batch_size=32–64, verify_serial_parity_n=8, parity_atol=1e-4)`** (or custom right-padded causal batching reading logits at `lens - 1` / `lens` with serial parity verified at `atol=1e-4`). Never run unbatched `batch=1` loops across thousands of cells.
6. **Bound Kaggle `/kaggle/working` Disk (`<= 19.5 GiB`) & Windows `MAX_PATH`:** `FP32 model + AdamW ≈ 12 bytes/param`. Set coarse `ckpt_every` (e.g. `512`), use **short `CheckpointStore` lineage keys (`"main"`)**, call `tr.store.prune(keep=set())`, pack only `LATEST`/`MILESTONES` + referenced `objects/<sha>/*` into `*_FULL.zip` (skipping redundant `PREVIOUS` generations), extract archives via **`safe_extract_planthir_archive`** (which applies `win_long_path` automatically + compact `_rs/` & `prp_/s` staging), and mirror only top-level archives/reports to `/kaggle/working/` via zero-copy `os.link`. `export_session_archives` now writes `ANALYSIS`, `EVIDENCE`, and `FULL` **before** capped `SUMMARY.zip` and auto-compacts oversized `SUMMARY.zip` when `strict_summary_cap=False`.
7. **Honest Multi-Backbone GPU Profiling:** Tag jobs with `extra={"profile_group": "S20" | "S50"}` (or pass `measured_seconds_per_update_by_group`), synchronize CUDA via `torch.cuda.synchronize(dev)` around timed updates, and free temporary profiling models (`del m, opt, x; torch.cuda.empty_cache()`).
8. **Fail-Fast Pinned Kaggle Notebook with `try/finally` Crash Mirroring:** Pass `preflight_commands=...` to `build_pinned_kaggle_notebook(...)` (defaults to **`owner_authorized=True`** for zero-edit Kaggle Run All; pass `owner_authorized=False` explicitly for dry-run / `--inventory-only`) alongside `preflight_canary_backend`, `backend_modules=...` (auto-propagated into `PLANTHIR_BACKEND_MODULES`), and `require_dual_gpu=True` so `torch.cuda.device_count() >= 2`, `doctor`, and `shield`/CPU gates run BEFORE multi-hour training; when `mirror_outputs_to_working=True`, launch + mirroring run inside `try ... except BaseException ... finally:` writing `CRASH_RECEIPT.json` and salvaging partial `/kaggle/working` receipts even on mid-run failure.
9. **Strict Telemetry & Receipt JSON Sanitization (`dump_json_safe` / `sanitize_for_json`):** Never pass raw `torch.Tensor`, `np.ndarray`, `Path`, `bytes`, `set`, or `NaN`/`Inf` floats to bare `json.dumps(...)` at the end of a multi-hour run; serialize all campaign/job/stage receipts and metrics with **`dump_json_safe(...)`** or **`sanitize_for_json(...)`**.
10. **End-to-End Stage-1-Through-Final Lifecycle Rehearsal Over the Full Job Ledger (`<30s` Before Wave 0):** Never leave downstream Stage-5 scoring, multi-stage paired-contrast aggregation, `ConfirmLock`, `export_session_archives`, or `/kaggle/working` salvage mirroring untested until Hour 5, and never test campaign reporting/export solely on small synthetic `jobs_override` subsets: run **`validate_paired_arm_records_preflight(...)`** (`run_registered_plan(..., fail_on_paired_arm_collision=True)`), **`validate_end_to_end_lifecycle(...)`**, `stage5_preflight_fn`, unconstrained `0xB5` decode checks, and full-ledger report + archive export/verify in preflight before launching GPU waves.

### Step 4 — Fast Tiered Verification Suite
**NEVER run 560+ legacy tests single-threaded.** Run this fast tiered suite:
1. **Unit, `u0000` All-Architecture Smoke, & `FULL`-Dispatch Suite:**
   - Run `python -m pytest tests/test_<name>.py -q` (or `-n auto` if `pytest-xdist` is installed).
   - Must test: tokenized oracle, split custody at **>= production quota** (`verify_production_split_disjointness` / `verify_split_disjointness(..., min_per_family=production_quota)`, never only `per_family=2` when cloud runs `8`+) + planted leak, shortcut panel, destructive masks, batched-vs-serial eval parity (`atol=1e-4`), **CPU `u0000` init+train+eval+`0xB5`-decode smoke on EVERY architecture/worker adapter**, full multi-stage job-ledger reporting/export & `validate_paired_arm_records_preflight` (never only tiny `jobs_override` subsets), mocked 2-GPU `FULL` dispatch + deferral/resume (`verify_full_mode_dispatch(..., split_disjointness_fn=...)` or `in_process_runner` test), complete-cohort confirmation lock guard, and 3 targeted recovery tests (~90 lines using `in_process_runner`: mid-update crash/resume parity, swapped `spec_sha256`/missing member rejection via `reconcile_verified_resume(..., fail_on_rejection=True)`, and 1-lane failure preserving peer work + `PARTIAL` export).
2. **Fresh-Subprocess Plugin CLI Chain (`--backend-module` or `PLANTHIR_BACKEND_MODULES`):**
   ```bash
   python -m anra_v5.palantiri --backend-module experiments.<name>.module init --backend <id> --out <tmp>/spec.json
   python -m anra_v5.palantiri --backend-module experiments.<name>.module validate --spec <tmp>/spec.json --canary
   python -m anra_v5.palantiri --backend-module experiments.<name>.module run --spec <tmp>/spec.json --work-root <tmp>/run --canary --device cpu
   python -m anra_v5.palantiri verify-bundle --bundle-dir <tmp>/run/result_bundle
   python -m anra_v5.palantiri --backend-module experiments.<name>.module replay-bundle --bundle-dir <tmp>/run/result_bundle --output <tmp>/REPLAY.json
   ```
   Confirm `verified: true` and `MATCHED`. *(Equivalently set `PLANTHIR_BACKEND_MODULES=experiments.<name>.module`.)*
3. **Final Pre-Push Regression Check:**
   - Run `python -m anra_v5.palantiri doctor`, `python -m anra_v5.palantiri calibrate`, `python -m anra_v5.palantiri shield`, and `python -m pytest tests/test_<name>.py tests/test_elendril_tree.py tests/test_planthir_v2_audit_upgrades.py tests/test_planthir_declarative_cli_and_usability.py tests/test_planthir_level7.py -q` (add `-n auto` when `pytest-xdist` is available).

### Step 5 — Two-Commit Source Pinning & Push (Zero Self-Referential SHAs)
1. **Commit 1 (Implementation + Docs + Tests):** Stage code + `.md` manuals + tests cleanly on `<target-branch>`, commit, and record `CODE_SHA = git rev-parse HEAD`. *(Whenever code OR `.md` manuals change on a branch carrying a pinned notebook under `notebooks/*.ipynb`, always re-pin the notebook in Commit 2 so `SOURCE_SHA` never lags behind `HEAD`!)*
2. **Generate & Verify Notebook:** Pin the notebook to `CODE_SHA`, run `ast.parse` on every Python code cell, assert `--check-notebook` (`match=true`), and extract a clean ZIP snapshot outside the repo (`git archive --format=zip -o <tmp>/src.zip <CODE_SHA>`, then `python -m zipfile -e <tmp>/src.zip <tmp>/src` — safe on Windows PowerShell where raw tar pipes corrupt headers) to confirm imports and full job ledger expansion cleanly.
3. **Commit 2 (Notebook + Curated Pre-Run Receipt + `progress.md`) & Push:**
   - Append a `<= 5`-nonempty-line entry (`YYYY-MM-DD | Agent | Title`) to root [`progress.md`](progress.md) distinguishing verified CPU engineering checks from pending GPU runs.
   - Commit, verify `git status -sb` is clean (`doctor` reports `tracked_clean: true`), push to `origin <target-branch>`, and confirm `git ls-remote origin refs/heads/<target-branch>` matches `git rev-parse HEAD`.

### Step 6 — Honest Handoff Report
State concisely:
- Branch name, exact `CODE_SHA`, final `HEAD` SHA, and any clickable repository notebook path.
- Exact commands run and observed pass counts/timings (never extrapolate unprinted metrics).
- Explicit pending GPU / owner execution items.
- **`PALANTIRI_FEEDBACK` block** documenting every overridden soft rule or Planthir gap/workaround encountered.

---

## 4. Verified `anra_v5.palantiri` Live API Cheat-Sheet

```python
# 1. Plugin Registration (NO register_experiment_module() function — call .register()!)
ExperimentModule(
    *, backend_id, backend_version="1.0", program=None,
    architectures=("baseline", "treatment"), objectives=("ce",),
    model_factory, dataset_factory, loss_fn,
    kernel_factory=None, policy_factory=None, action_labels=None,
    default_dataset_config=None, default_model_config=None,
    treatment_architecture=None, treatment_model_config=None,
    extra_model_keys=(), extra_dataset_keys=(), supports_grouped_loss=False, supports_fp16=False,
).register()
# CLI flag: python -m anra_v5.palantiri --backend-module experiments.<name>.module <subcmd> ...

# 2. Trainers (native FP16/AMP GradScaler via precision="amp"|"fp16"), Recipes & X1 Spec Helper
GenericTrainRecipe(seed=7, total_updates=4, batch_size=8, micro_batch_size=4, lr=0.001, clip=1.0, objective="ce", sampling="uniform", schedule="cosine", precision="fp32", eq_weight=0.0, cf_weight=0.0, cf_margin=0.5)
GenericExperimentTrainer(*, backend_id, model, model_spec, recipe, loss_fn, items, data_provenance, order_seed, source_commit, root, lineage, device="cpu", cpu_canary=True, resume=False, max_cpu_params=100000, max_cpu_samples=32, max_cpu_updates=8, post_step_hook=None)
trainer.train_loop(*, total_updates=None, checkpoint_every=0, save_final=True, post_step_hook=None, post_train_hook=None, prune_old=True)
BytePromptTrainer(samples, *, spec, recipe=None, source_commit, root, lineage, data_provenance, order_seed=None, device="cpu", cpu_canary=True, resume=False, post_step_hook=None)
build_x1_spec(architecture, *, cpu_canary=False, base_spec=None, width=None, layers=None, query_heads=None, kv_heads=None, ffn_width=None, context_length=None, pointer_dim=None, slot_dim=None, num_slots=None, slot_edge_hidden=None, agg_iterations=None, msg_iterations=None, recurrent_dim=None, max_sequence_length=None)

# 3. Campaign Orchestration, Rollout & Verification (persists partial receipts on BaseException!)
orch = CampaignOrchestrator(*, campaign_id, source_commit, work_dir, max_minutes=330.0, reserve_minutes=45.0, min_disk_bytes=2*2**30, in_process_runner=None)
orch.run_registered_plan(jobs, *, command_builder=None, job_runner=None, devices=("cuda:0", "cuda:1"), resume_input=None, measured_seconds_per_update=None, measured_seconds_per_update_by_group=None, profile_fn=None, per_job_disk_bytes=128*2**20, outer_remaining_minutes=None, inventory_only=False, fail_on_resume_rejection=True, stage_gates=None, parent_checker=None, live_wall_clock_preemption=True, clock_fn=None, u0000_smoke_fn=None, u0000_fp16_autocast_fn=None, require_u0000_fp16_autocast=False, byte_decode_fn=None, stage5_preflight_fn=None, export_prefix=None, export_dir=None, write_full_zip=True)
orch.admit_slice(jobs, *, measured_seconds_per_update, per_job_disk_bytes, parallel_workers=2, outer_remaining_minutes=None, startup_overhead_seconds=0.0, export_reserve_bytes=0, per_job_seconds_per_update=None)
orch.reconcile_verified_resume(imported_receipts, registry_rows, *, required_archive_members=(), fail_on_rejection=False)
orch.run_waves(jobs, *, command_builder, devices=("cuda:0", "cuda:1"), resume=False)
batched_rollout(policy, kernel, worlds, *, batch_size=64, max_actions=4, verify_serial_parity_n=0, parity_atol=1e-4)
verify_full_mode_dispatch(dispatch_fn, *, expected_job_ids, expected_seeds=(7, 19, 41), required_devices=("cuda:0", "cuda:1"), min_dataset_counts=None, tight_session_minutes=None, full_session_minutes=300.0, require_multi_session_resume=True)

# 4. Archives (ANALYSIS/EVIDENCE/FULL before capped SUMMARY), Long Paths, Notebooks & Parity
win_long_path(path)
safe_extract_planthir_archive(zip_path, dest_dir)
export_session_archives(source_root, output_dir, *, prefix, analysis_files=None, full_files=None, summary_files=None, evidence_files=None, status="COMPLETE", source_commit=None, summary_max_bytes=5*2**20, strict_summary_cap=False, write_full_zip=True)
verify_session_archive(archive_path, *, max_bytes=None, expected_source_commit=None, require_sidecar_sha256=True, require_v6_manifest=True, required_members=())
build_pinned_kaggle_notebook(*, source_commit, title, launch_command, owner_authorized=True, repo_url=None, campaign_mode="FULL", session_minutes=300, resume_input=None, inventory_command=None, discover_kaggle_inputs=(), mirror_outputs_to_working=False, preflight_canary_backend=None, preflight_commands=(), require_dual_gpu=False, preflight_cuda_canary=False, emit_file_links=False, backend_modules=())
verify_shared_backbone_and_head_overhead(control_model, treatment_model, *, backbone_prefix="core.", max_head_overhead_ratio=0.05, max_control_parity_ratio=None)

# 5. Paired Stats & Reporting
validate_paired_arm_records_preflight(records, *, arm_key="arm_id", fallback_arm_keys=("architecture",), scope_keys=("stage", "study", "session"), require_unique_without_scope=True, fail_on_collision=False)
bootstrap_paired_families(control_scores, treatment_scores, *, resamples=2000, seed=1801)
build_paired_seed_observations(arm_records, *, contrast_id="PRIMARY", control_arm=..., treatment_arm=..., metric_key="value", scope_key=None, scope_value=None, deduplicate="error")
compare_paired_episode_records(control_episodes, treatment_episodes)
exact_mcnemar_test(baseline_correct, treatment_correct)
holm_bonferroni_adjust(p_values, *, alpha=0.05)
summarize_diagnostic_coverage(worlds, *, target_families, seed_ids=(7, 19, 41))
build_analyst_first_page_report(*, title, campaign_id, source_commit, decision_summary, control_arm, treatment_arm, paired_seed_bundle, ...)
render_campaign_dashboard(status_payload)

# 6. Level-7 Front Door, Runtime Shield, Lossy Byte Decode, JSON Sanitizer & Custody
from anra_v5.palantiri import Palantiri, ConfirmLock, CampaignLedgerVerifier, verify_campaign_ledger, validate_fp16_autocast_forward_backward, validate_model_and_gradients
from anra_v5.palantiri import collect_minimal_checkpoint_files, prune_completed_job_artifacts, mirror_to_working_dir, verify_pinned_notebook_bytes, build_x1_spec
from anra_v5.palantiri import TypedByteSequenceKernel, decode_generated_bytes, infer_public_byte_bound, score_typed_output, canonical_structural_hash, verify_split_disjointness
from anra_v5.palantiri import sanitize_for_json, dump_json_safe, validate_byte_decoder_resilience, validate_end_to_end_lifecycle, validate_paired_arm_records_preflight, run_level7_runtime_shield_suite
Palantiri.status(work_root) / Palantiri.dashboard(work_root) / Palantiri.scaffold(name, out_dir, include_verify_module=True)
Palantiri.runtime_shield() / Palantiri.validate_paired_arm_records(records, ...) / Palantiri.decode_lossy(ids) / Palantiri.decode_generated_bytes(ids) / Palantiri.validate_byte_decoder() / Palantiri.validate_lifecycle(policy, kernel, worlds) / Palantiri.sanitize_json(obj)
Palantiri.score_typed_output(prompt, raw_output, ...) / Palantiri.verify_splits(splits) / Palantiri.verify_production_splits(builder, production_quota=...) / Palantiri.build_x1_spec(arch, cpu_canary=True/False) / Palantiri.validate_fp16_autocast(model, loss_fn) / Palantiri.amp_context(dev, use_amp=None) / Palantiri.audit_fp16_source(obj)
```

- **Level-7 CLI Shortcuts & Upgraded Safeguards:**
  - `python -m anra_v5.palantiri level` (inspect Level-7 subsystems & registered backends)
  - `python -m anra_v5.palantiri shield` (run 5/5 adversarial Level-7 runtime shields: `0xB5` byte decode, FP16 `-1e9`, `NaN`/`Tensor` JSON sanitizer, archive cap auto-compaction, and `multi_stage_arm_record_collision_preflight`)
  - `python -m anra_v5.palantiri doctor` (includes Check 7 `level7_runtime_shield`), `calibrate` (`6/6` + `shield`)
  - `python -m anra_v5.palantiri scaffold --name my_exp --out-dir experiments` (generate a runnable plugin + smoke test + `ConfirmLock`)
  - `python -m anra_v5.palantiri status --work-root <dir> --markdown` (live read-only campaign dashboard & `pending_confirm_jobs`)
  - `python -m anra_v5.palantiri verify-archive --archive <zip> --expected-commit <sha>` (standalone v6 manifest + sidecar check)
  - Note: Never call strict `UTF8.decode` on unconstrained neural byte emissions (`0xB5` crashes Stage 5!); use `UTF8.is_ascii_stop_token` + `UTF8.decode_lossy` or `decode_generated_bytes`, and pass `byte_decode_fn` + `stage5_preflight_fn` to `run_registered_plan`. Use `sanitize_for_json` / `dump_json_safe` for strict JSON artifacts, `TypedByteSequenceKernel` / `score_typed_output` + `infer_public_byte_bound` instead of 16-class hashing, `verify_split_disjointness` to catch renamed isomorphic Horn/graph/AST split leaks, and `prune_completed_job_artifacts(..., strip_final_optimizer=..., child_job_ids=..., completed_updates=..., full_job_updates=...)` so continuation parents never lose `optimizer.bin`. Declare `ExperimentModule(..., supports_fp16=True)`, set `precision="amp"`/`"fp16"` natively in `GenericExperimentTrainer`, and verify padded batches under `torch.amp.autocast` using `validate_fp16_autocast_forward_backward` / `u0000_fp16_autocast_fn`. `build_pinned_kaggle_notebook` propagates `backend_modules` into `PLANTHIR_BACKEND_MODULES` and wraps launch + zero-copy `os.link` mirroring in `try/except BaseException (CRASH_RECEIPT.json)/finally`; `export_session_archives` writes `ANALYSIS`/`EVIDENCE`/`FULL` before capped `SUMMARY` (auto-compacting oversized `SUMMARY` unless `strict_summary_cap=True`).

---

## 5. Battle-Tested Gotchas (Avoid Every Known Trap)

| Trap / Error | Exact Fix |
|---|---|
| `forbidden_metadata_key:gold` in Gate 1 | `UniversalObservation.metadata` forbids `{"expected", "gold", "gold_answer", "hidden", "truth", "ground_truth"}` recursively **even when the value is `None`**. Never put placeholder truth keys in metadata. |
| `AttributeError: 'str' object has no attribute 'modified_world'` | `EnvironmentKernel.decompose_subgoals(world)` MUST return `Sequence[SubgoalProbeSpec]` (or `()`), NEVER strings. |
| Gate 4 fails on `ablate_mechanism` or stateful policy | `model.ablate_mechanism(self, mode="...")` MUST accept zero required arguments. Custom policies MUST expose `self.policy_id: str` and implement `reset_episode(self) -> None`. |
| `FrozenInstanceError` on `SessionJob` | `SessionJob` is `@dataclass(frozen=True)`. Always use `dataclasses.replace(s, device=...)`. |
| `AttributeError: 'WorkerJobSpec' object has no attribute 'job_id'` | Inside `in_process_runner(job, env, cmd)`, the ID attribute is **`job.worker_id`** (NOT `job.job_id`). Return `int` or `{"exit_code": int}`. |
| `KeyError: 'status'` on `reconcile_verified_resume` | Returns `verified_completed_jobs`, `rejected_jobs`, `pending_job_ids`, `next_resumable_job_id` — **no `"status"` key**. Checks `completed_updates >= target_updates`. |
| `KeyError: 'hashes'` on `export_session_archives` | Digest dict key is **`"archive_sha256"`** (NOT `"hashes"`). Archives write `ANALYSIS`/`EVIDENCE`/`FULL` before capped `SUMMARY` (auto-compacting unless `strict_summary_cap=True`). |
| `ValueError` calling `verify_single_planthir_zip` on session ZIPs | Use **`verify_session_archive(...)`** and **`rescore_compact_analysis_archive(...)`** for `export_session_archives` ZIPs (`*_SUMMARY/EVIDENCE/ANALYSIS/FULL.zip`). Use `verify_result_bundle` / `verify_single_planthir_zip` ONLY for Cartridge `result.zip`. |
| `v5_model.core.Core` `ImportError` | Construct via **`v5_model.core.initialize(spec, seed, torch_module=torch)`** with `ModelSpec(vocabulary=...)` (not `vocab_size`) + `packed_layout(segments, torch_module=torch)`. Exposes `.forward_hidden(...)` and `.blocks` (NO `.logits()`). |
| `UTF8.decode` `ValueError` on unconstrained model bytes (`0xB5` crash) | Strict `UTF8.decode(ids)` raises `ValueError` on tokens `< 4` AND on invalid UTF-8 bytes such as `0xB5` after hours of training. NEVER call strict `UTF8.decode` in neural decoding/eval loops — ALWAYS use **`UTF8.is_ascii_stop_token`**, **`UTF8.decode_lossy`**, or **`decode_generated_bytes`**. `SpecialToken` is an `Enum`, use `UTF8.special(SpecialToken.ANSWER)`. |
| `RuntimeError: element 0 of tensors does not require grad` | Any bounded weight-update step invoked from `@torch.no_grad()` evaluation MUST wrap forward/backward in `with torch.enable_grad():`. |
| `NotImplementedError` subtracting `bool` buffers in weight caps | Iterate over `model.named_parameters()` (floating params only), NEVER `state_dict()` (which includes `bool`/int buffers). |
| `CUDA_VISIBLE_DEVICES` double-indexing on lane 1 | `IndependentWorkerScheduler` sets `CUDA_VISIBLE_DEVICES="1"` and remaps `--device cuda:1` -> `--device cuda:0`. Child workers should always use the `--device` passed to them (`cuda:0`). |
| Legacy `V8BytePolicy` `AttributeError` (`'_S' object has no attribute 'base_spec'`) | Stand-in specs must supply a real `base_spec` (or wrap the native model/spec directly) AND **test EVERY architecture/worker adapter (`B0`, `P`, `F`, `R`, `U`) on CPU at `u0000` before GPU launch**. |
| `c10::Half` `-1e9` overflow under `torch.amp.autocast` | Never apply `-1e9` or `(~valid_tok) * (-1e9)` on FP16 tensors inside `torch.amp.autocast(..., dtype=torch.float16)` (`Half` max `65504`). Cast attention scores and `softmax`/`log_softmax` reductions to `torch.float32`, set `ExperimentModule(..., supports_fp16=True)`, enforce padded `validate_fp16_autocast_forward_backward` (includes CPU Half matmul/`bmm`/`einsum` emulation) + `audit_fp16_autocast_source` / `Palantiri.audit_fp16_source` + `u0000_fp16_autocast_fn` at `u0000`, and wrap inner model autocast helpers with **`amp_autocast_context(device, use_amp=None|True|False)`** (`Palantiri.amp_context`, defers to caller autocast on `None` with zero `CUDA is not available` warnings). |
| Kernel `state` mapping vs object in custom kernels | `cartridge.py` / `replay.py` handle dict-like (`Mapping`) or attribute-based `state.terminal`; return consistent state objects cleanly in custom kernels. |
| Windows `MAX_PATH` (`FileNotFoundError` / `checkpoint manifest is missing`) unpacking or reading checkpoints | `CheckpointStore` (`v5_training/checkpoint.py`), `evidence.py`, `checkpoint_intelligence.py`, and `safe_extract_planthir_archive` (`win_long_path`) handle Windows `>=240`-char `objects/<sha256>/manifest.json` paths + `_rs/` / `prp_/s` staging natively; still keep **`CheckpointStore` lineage keys short (e.g. `"main"`)** and never swallow extraction errors silently. |
| PowerShell 5.1 syntax (`&&`, `||`, `<<'EOF'`, `>`) & `git archive` pipes | Use `; if ($LASTEXITCODE -eq 0) { exit $LASTEXITCODE }` instead of `&&`, never pipe raw `git archive --format=tar` through PowerShell text pipelines (corrupts binary tar headers — use **`git archive --format=zip -o <tmp>/src.zip <CODE_SHA>`** + **`python -m zipfile -e <tmp>/src.zip <tmp>/src`**), and avoid PowerShell `>` for UTF-8 JSON. |
| `GenericExperimentTrainer.save()` `"cannot publish faulted or repeated update"` | `TrainingState` enforces `generation == global_update`, so calling `trainer.save()` twice on the same update fails. Use `post_step_hook` on `GenericExperimentTrainer` / `step(...)`, OR `trainer.train_loop(..., save_final=False, post_train_hook=...)` followed by a single `trainer.save()`. |
| `X1ModelSpec` fields & canary width errors | `X1ModelSpec` uses `base_spec: ModelSpec` (`width == query_heads * head_dimension`) + short names (`"P"`, `"R"`, `"R-noedge"`, `"P-capR"`), and `solve_x1_capacity_controls` defaults target S20 widths rather than `cpu_canary`. Call **`build_x1_spec(arch, cpu_canary=True/False)`** cleanly. |
| Default `owner_authorized=True` & CUDA probe handoff (`wrapper_CUDA_cat`) | `build_pinned_kaggle_notebook` defaults to **`owner_authorized=True`** (pass `owner_authorized=False` explicitly for dry-run / `--inventory-only`). Representation consumers (`causal.py`, `cartridge.py`, `observatory.py`, `runtime.py`, `validation_study.py`) normalize `.detach().cpu()` automatically. |
| Raw `json.dumps` `TypeError`/`ValueError` on `Tensor`, `Path`, `bytes`, `set`, `NaN`/`Inf` | Never call bare `json.dumps(..., allow_nan=False)` on training/eval telemetry. Always serialize campaign, stage, and job receipts via **`dump_json_safe(...)`** or **`sanitize_for_json(...)`** so `Tensor`, `ndarray`/NumPy scalars, `Path`, `bytes`, `set`, and non-finite floats never abort a finished run. |
| `SUMMARY.zip` cap overflow or custom notebook crash hiding partial outputs | `export_session_archives` commits `ANALYSIS`, `EVIDENCE`, and `FULL` **before** capped `SUMMARY.zip` and auto-compacts oversized `SUMMARY.zip` (`strict_summary_cap=False`). Any custom notebook launch cell must wrap execution + `/kaggle/working` mirroring in **`try / except BaseException (CRASH_RECEIPT.json) / finally:`** (`mirror_to_working_dir`) to salvage partial receipts on error. |
| `ValueError: duplicate record for arm ...` in `build_paired_seed_observations` | Multi-stage ledgers sharing an arm across stages (e.g. `S2` vs `S3` sharing `P-R-Iter_D3-Hybrid_S20`) collide over `(arm_id, init_seed, order_seed)`. Run **`validate_paired_arm_records_preflight(...)`** (`Palantiri.validate_paired_arm_records(...)`, wired into `CampaignOrchestrator.prepare` / `run_registered_plan`) over the full ledger before training, pass **`scope_key`/`scope_value`** or **`deduplicate="first"|"last"`** to `build_paired_seed_observations(...)`, and never test campaign reporting only on small `jobs_override` subsets. |
| CLI `--dir` error, `ORIGINAL_PREDICTION_MISMATCH` & bare `pytest` | Always pass **`--bundle-dir`** for `verify-bundle`/`replay-bundle`, run **`python -m pytest`** (so repo root is on `sys.path`), and use `python -m anra_v5.palantiri explain-error --code <CODE>` for instant diagnostics. On replay mismatch, `replay-bundle` emits structured `mismatch_reasons`, `mismatch_counts_by_reason`, `mismatches`, `triage_hints`, and `pythonhashseed` directly on stdout/stderr (`PYTHONHASHSEED=0`). |
