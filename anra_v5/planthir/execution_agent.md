# Execution Agent Manual (`execution_agent.md`)

> **Role:** Fast / High-Precision Experiment Builder & Execution Agent
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
| **SOFT ENGINEERING DEFAULTS** | **BREAK FREELY IF IT INCREASES QUALITY** | 1. **Soft line/file budgets:** If genuine `FULL`-mode worker dispatch, batched GPU evaluation, or complete scientific controls need more lines than a soft guideline, **write the clean code needed** instead of compressing `FULL` mode.<br>2. **"Don't read internal `.py` files":** Start from Sections 4–5 below without browsing unrelated history — **BUT** whenever you hit a traceback, schema surprise, or subtle Planthir edge case, **immediately inspect the exact function in `anra_v5/planthir/*.py`** rather than guessing.<br>3. **Forcing an incomplete/buggy Planthir helper:** If `run_registered_plan`, `StaticPromptKernel`, `build_pinned_kaggle_notebook`, `admit_slice`, or a legacy evaluator bridge lacks parent-child DAG ordering, adaptive one-go wall-clock loops, `LATEST`-only checkpoint packing, mixed subprocess routing, or hits a bug, **wrap, extend, or bypass that helper cleanly inside `experiments/<name>/`** while keeping all Hard Scientific Laws intact.<br>4. **Default notebook/session flags:** Override `OWNER_AUTHORIZED=False` / `SESSION_MINUTES=300` whenever the owner or spec requests one-click full execution (`OWNER_AUTHORIZED=True`, `SESSION_MINUTES=660`, `ONE_GO=True`). |

**Mandatory Transparency Rule (`PALANTIRI_FEEDBACK`):** Whenever you break a soft rule or bypass/wrap a Planthir gap or bug, include a short **`PALANTIRI_FEEDBACK`** block in your final handoff specifying: (1) `gap_or_bug`, (2) `rule_overridden`, (3) `local_workaround`, and (4) `suggested_upstream_fix`.

---

## 3. Fast 6-Step Execution Playbook

### Step 1 — Preflight (`<30s`, Zero Wasted Turns)
1. Confirm branch and clean tree (`git branch --show-current`, `git status -sb`).
2. Read the Orchestrator specification completely. Do **not** waste turns reading unrelated historical campaigns.
3. Run platform self-checks:
   ```bash
   python -m anra_v5.palantiri doctor
   python -m anra_v5.palantiri calibrate
   ```
   Confirm `GREEN_SIGNAL` and `6/6` faults detected (plus `SHORTCUTS_REJECTED_AND_ORACLE_QUALIFIED`).

### Step 2 — Choose Your Integration Path
- **Path A — Declarative `spec.json` (`<50` lines):** Use when varying hyperparameters/arms on already-registered backends (`v8`, `x1_rd2`, `x1_s20`):
  ```bash
  python -m anra_v5.palantiri init --backend x1_rd2 --out spec.json
  python -m anra_v5.palantiri validate --spec spec.json --canary
  python -m anra_v5.palantiri run --spec spec.json --work-root runs/exp1 --canary
  python -m anra_v5.palantiri verify-bundle --bundle-dir runs/exp1/result_bundle
  python -m anra_v5.palantiri replay-bundle --bundle-dir runs/exp1/result_bundle --output runs/exp1/REPLAY_RECEIPT.json
  ```
- **Path B — Plugin Package (`experiments/<name>/`, Recommended):** Put all new experiment code under one clean package (`plan.py`, `tasks.py`, `models.py`, `module.py`, plus an optional clean 5th helper module such as **`verify.py`** for independent receipt/claim verification instead of cramming 400+ lines into `module.py`) and `tests/test_<name>.py`. Register via `ExperimentModule(...).register()` + `register_planthir_backends()`, and load in CLIs via **`--backend-module experiments.<name>.module`** or `PLANTHIR_BACKEND_MODULES=experiments.<name>.module` (ensure `"experiments*"` is included in `pyproject.toml`).
- **Path C — Hybrid / Quality-First Override:** Keep `anra_v5.palantiri` for custody, Launch Gates, stats (`bootstrap_paired_families`, `build_paired_seed_observations`), and verified archives (`export_session_archives`, `verify_session_archive`), and add clean local PyTorch/NumPy/DAG orchestration inside `experiments/<name>/` wherever a Planthir helper cannot express your pipeline cleanly.

### Step 3 — Production `FULL`-Mode & Kaggle 2×T4 Checklist (8 Battle-Tested Rules)
Follow every item here so live Kaggle 2×T4 runs never stall, overflow `/kaggle/working`, or crash mid-wave:

1. **Wrap Frozen Baselines Directly (Zero Confound):** Import `X1TypedModel`, `solve_x1_capacity_controls`, `core_x1_jobs`, `compute_x1_d2_grouped_loss`, or `v5_model.core.initialize` directly; assert `verify_shared_backbone_and_head_overhead(ctrl, trt, backbone_prefix="core.", max_head_overhead_ratio=0.05, max_control_parity_ratio=0.02)`.
2. **Real Per-Job Multi-GPU Dispatch (`FULL` != Scaled-Up `CANARY`):**
   - Use **`CampaignOrchestrator.run_registered_plan(...)`**, which now natively supports **`stage_gates`** (conditional evidence-gated stage unlocking recorded in `STAGE_GATES.json`), **`depends_on` / `parent_job_id` / `required_parent_paths` / `parent_checker`** (isolating missing-parent jobs as **`BLOCKED_MISSING_PARENT`** in `BLOCKED_JOBS.json` without aborting unrelated waves), **`live_wall_clock_preemption=True` + `clock_fn`** (inter-wave reserve checks before dispatching the next wave), **per-`profile_group` timing (`per_job_seconds_per_update`)**, and **`u0000_smoke_fn` preflight**.
   - Only compose manual wave loops if an exotic multi-repo subprocess protocol still needs custom routing; always persist `JOB_RECEIPTS.json`, `PENDING_QUEUE.json`, and resumable session archives on interrupt/deferral.
3. **Smoke-Test EVERY Distinct Architecture & Legacy Worker Adapter at `u0000` (CRITICAL):**
   - Pass **`u0000_smoke_fn`** to `run_registered_plan(...)` AND test in `pytest`: a CPU `u0000` smoke check (init -> 1 train step -> 1 eval step -> checkpoint save/load) for **EVERY** distinct architecture class (e.g., `B0`, `P`, `F`, `R`, `U`, `S50`, and every new treatment/control) AND every legacy subprocess worker path. Do not assume testing one model family covers all of them (`AttributeError: '_S' object has no attribute 'base_spec'` on `B0` stops live waves at step `u0000`!).
4. **Complete-Cohort Confirmation Lock Guard:** Never open sealed `confirm` during a `PARTIAL` or deferred session (`pending_confirm_jobs == []` and 100% of confirmation-eligible jobs verified complete required before unsealing `confirm`).
5. **Batch ALL Multi-Case Evaluation Loops (50–100× Speedup):** Use **`batched_rollout(..., batch_size=32–64, verify_serial_parity_n=8, parity_atol=1e-4)`** (or custom right-padded causal batching reading logits at `lens - 1` / `lens` with serial parity verified at `atol=1e-4`). Never run unbatched `batch=1` loops across thousands of cells.
6. **Bound Kaggle `/kaggle/working` Disk (`<= 19.5 GiB`) & Windows `MAX_PATH`:** `FP32 model + AdamW ≈ 12 bytes/param`. Set coarse `ckpt_every` (e.g. `512`), use **short `CheckpointStore` lineage keys (`"main"`)**, call `tr.store.prune(keep=set())`, pack only `LATEST`/`MILESTONES` + referenced `objects/<sha>/*` into `*_FULL.zip` (skipping redundant `PREVIOUS` generations), extract archives via **`safe_extract_planthir_archive`** (which applies `win_long_path` automatically + compact `_rs/` & `prp_/s` staging), and mirror only top-level archives/reports to `/kaggle/working/` via zero-copy `os.link`.
7. **Honest Multi-Backbone GPU Profiling:** Tag jobs with `extra={"profile_group": "S20" | "S50"}` (or pass `measured_seconds_per_update_by_group`), synchronize CUDA via `torch.cuda.synchronize(dev)` around timed updates, and free temporary profiling models (`del m, opt, x; torch.cuda.empty_cache()`).
8. **Fail-Fast Pinned Kaggle Notebook:** Pass `preflight_commands=...` to `build_pinned_kaggle_notebook(...)` alongside `preflight_canary_backend` and `require_dual_gpu=True` so `torch.cuda.device_count() >= 2` (when `OWNER_AUTHORIZED=True`), `doctor`, and fast CPU/CLI gates run BEFORE multi-hour training; auto-discover both `.zip` archives and unpacked folders under `/kaggle/input/**`.

### Step 4 — Fast Tiered Verification Suite
**NEVER run 560+ legacy tests single-threaded.** Run this fast tiered suite:
1. **Unit, `u0000` All-Architecture Smoke, & `FULL`-Dispatch Suite:**
   - Run `python -m pytest tests/test_<name>.py -q` (or `-n auto` if `pytest-xdist` is installed).
   - Must test: tokenized oracle, split custody + planted leak, shortcut panel, destructive masks, batched-vs-serial eval parity (`atol=1e-4`), **CPU `u0000` init+train+eval smoke on EVERY architecture/worker adapter**, mocked 2-GPU `FULL` dispatch + deferral/resume (`verify_full_mode_dispatch` or `in_process_runner` test), complete-cohort confirmation lock guard, and 3 targeted recovery tests (~90 lines using `in_process_runner`: mid-update crash/resume parity, swapped `spec_sha256`/missing member rejection via `reconcile_verified_resume(..., fail_on_rejection=True)`, and 1-lane failure preserving peer work + `PARTIAL` export).
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
   - Run `python -m anra_v5.palantiri doctor`, `python -m anra_v5.palantiri calibrate`, and `python -m pytest tests/test_<name>.py tests/test_elendril_tree.py tests/test_planthir_v2_audit_upgrades.py tests/test_planthir_declarative_cli_and_usability.py -q` (add `-n auto` when `pytest-xdist` is available).

### Step 5 — Two-Commit Source Pinning & Push (Zero Self-Referential SHAs)
1. **Commit 1 (Implementation + Tests):** Stage code + tests cleanly on `<target-branch>`, commit, and record `CODE_SHA = git rev-parse HEAD`.
2. **Generate & Verify Notebook:** Pin the notebook to `CODE_SHA`, run `ast.parse` on every Python code cell, assert `--check-notebook` (`match=true`), and extract `git archive <CODE_SHA>` to a temp directory outside the repo to confirm the clean source snapshot imports and expands the full job ledger cleanly.
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
    extra_model_keys=(), extra_dataset_keys=(), supports_grouped_loss=False,
).register()
# CLI flag: python -m anra_v5.palantiri --backend-module experiments.<name>.module <subcmd> ...

# 2. Trainers & Recipes
GenericTrainRecipe(seed=7, total_updates=4, batch_size=8, micro_batch_size=4, lr=0.001, clip=1.0, objective="ce", sampling="uniform", schedule="cosine", precision="fp32", eq_weight=0.0, cf_weight=0.0, cf_margin=0.5)
GenericExperimentTrainer(*, backend_id, model, model_spec, recipe, loss_fn, items, data_provenance, order_seed, source_commit, root, lineage, device="cpu", cpu_canary=True, resume=False, max_cpu_params=100000, max_cpu_samples=32, max_cpu_updates=8)
BytePromptTrainer(samples, *, spec, recipe=None, source_commit, root, lineage, data_provenance, order_seed=None, device="cpu", cpu_canary=True, resume=False)

# 3. Campaign Orchestration, Rollout & Verification
orch = CampaignOrchestrator(*, campaign_id, source_commit, work_dir, max_minutes=330.0, reserve_minutes=45.0, min_disk_bytes=2*2**30, in_process_runner=None)
orch.run_registered_plan(jobs, *, command_builder=None, job_runner=None, devices=("cuda:0", "cuda:1"), resume_input=None, measured_seconds_per_update=None, measured_seconds_per_update_by_group=None, profile_fn=None, per_job_disk_bytes=128*2**20, outer_remaining_minutes=None, inventory_only=False, fail_on_resume_rejection=True, stage_gates=None, parent_checker=None, live_wall_clock_preemption=True, clock_fn=None, u0000_smoke_fn=None, export_prefix=None, export_dir=None, write_full_zip=True)
orch.admit_slice(jobs, *, measured_seconds_per_update, per_job_disk_bytes, parallel_workers=2, outer_remaining_minutes=None, startup_overhead_seconds=0.0, export_reserve_bytes=0, per_job_seconds_per_update=None)
orch.reconcile_verified_resume(imported_receipts, registry_rows, *, required_archive_members=(), fail_on_rejection=False)
orch.run_waves(jobs, *, command_builder, devices=("cuda:0", "cuda:1"), resume=False)
batched_rollout(policy, kernel, worlds, *, batch_size=64, max_actions=4, verify_serial_parity_n=0, parity_atol=1e-4)
verify_full_mode_dispatch(dispatch_fn, *, expected_job_ids, expected_seeds=(7, 19, 41), required_devices=("cuda:0", "cuda:1"), min_dataset_counts=None, tight_session_minutes=None, full_session_minutes=300.0, require_multi_session_resume=True)

# 4. Archives, Long Paths, Notebooks, & Backbone Parity
win_long_path(path)
safe_extract_planthir_archive(zip_path, dest_dir)
export_session_archives(source_root, output_dir, *, prefix, analysis_files=None, full_files=None, summary_files=None, evidence_files=None, status="COMPLETE", source_commit=None, summary_max_bytes=5*2**20, write_full_zip=True)
verify_session_archive(archive_path, *, max_bytes=None, expected_source_commit=None, require_sidecar_sha256=True, require_v6_manifest=True, required_members=())
build_pinned_kaggle_notebook(*, source_commit, title, launch_command, owner_authorized=False, repo_url=None, campaign_mode="FULL", session_minutes=300, resume_input=None, inventory_command=None, discover_kaggle_inputs=(), mirror_outputs_to_working=False, preflight_canary_backend=None, preflight_commands=(), require_dual_gpu=False, emit_file_links=False, backend_modules=())
verify_shared_backbone_and_head_overhead(control_model, treatment_model, *, backbone_prefix="core.", max_head_overhead_ratio=0.05, max_control_parity_ratio=None)

# 5. Paired Stats & Reporting
bootstrap_paired_families(control_scores, treatment_scores, *, resamples=2000, seed=1801)
build_paired_seed_observations(arm_records, *, contrast_id="PRIMARY", control_arm=..., treatment_arm=..., metric_key="value")
compare_paired_episode_records(control_episodes, treatment_episodes)
summarize_diagnostic_coverage(worlds, *, target_families, seed_ids=(7, 19, 41))
build_analyst_first_page_report(*, title, campaign_id, source_commit, decision_summary, control_arm, treatment_arm, paired_seed_bundle, ...)
```

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
| `KeyError: 'hashes'` on `export_session_archives` | Digest dict key is **`"archive_sha256"`** (NOT `"hashes"`). |
| `ValueError` calling `verify_single_planthir_zip` on session ZIPs | Use **`verify_session_archive(...)`** and **`rescore_compact_analysis_archive(...)`** for `export_session_archives` ZIPs (`*_SUMMARY/EVIDENCE/ANALYSIS/FULL.zip`). Use `verify_result_bundle` / `verify_single_planthir_zip` ONLY for Cartridge `result.zip`. |
| `v5_model.core.Core` `ImportError` | Construct via **`v5_model.core.initialize(spec, seed, torch_module=torch)`** with `ModelSpec(vocabulary=...)` (not `vocab_size`) + `packed_layout(segments, torch_module=torch)`. Exposes `.forward_hidden(...)` and `.blocks` (NO `.logits()`). |
| `UTF8.decode` `ValueError("expected content tokens only")` | `UTF8.decode(ids)` rejects tokens `< 4` (`pad=0, bos=1, eos=2, ANSWER=3`). Filter `[t for t in ids if t >= 4]`. `SpecialToken` is an `Enum`, use `UTF8.special(SpecialToken.ANSWER)`. |
| `RuntimeError: element 0 of tensors does not require grad` | Any bounded weight-update step invoked from `@torch.no_grad()` evaluation MUST wrap forward/backward in `with torch.enable_grad():`. |
| `NotImplementedError` subtracting `bool` buffers in weight caps | Iterate over `model.named_parameters()` (floating params only), NEVER `state_dict()` (which includes `bool`/int buffers). |
| `CUDA_VISIBLE_DEVICES` double-indexing on lane 1 | `IndependentWorkerScheduler` sets `CUDA_VISIBLE_DEVICES="1"` and remaps `--device cuda:1` -> `--device cuda:0`. Child workers should always use the `--device` passed to them (`cuda:0`). |
| Legacy `V8BytePolicy` `AttributeError` (`'_S' object has no attribute 'base_spec'`) | Stand-in specs must supply a real `base_spec` (or wrap the native model/spec directly) AND **test EVERY architecture/worker adapter (`B0`, `P`, `F`, `R`, `U`) on CPU at `u0000` before GPU launch**. |
| Kernel `state` mapping vs object in custom kernels | `cartridge.py` / `replay.py` handle dict-like (`Mapping`) or attribute-based `state.terminal`; return consistent state objects cleanly in custom kernels. |
| Windows `MAX_PATH` (`FileNotFoundError`) unpacking nested ZIPs | `safe_extract_planthir_archive` (`win_long_path` built in) + `_rs/` + `prp_/s` already handle long paths; keep **`CheckpointStore` lineage keys short (e.g. `"main"`)** and extract custom staging dirs via `win_long_path` (never swallow extraction errors silently). |
| PowerShell 5.1 syntax (`&&`, `||`, `<<'EOF'`, `>`) | Use `; if ($LASTEXITCODE -eq 0) { exit $LASTEXITCODE }` instead of `&&`, use file editing tools instead of heredocs, and avoid PowerShell `>` for UTF-8 JSON (it writes UTF-16LE BOM). |
| CLI `--dir` error & bare `pytest` | Always pass **`--bundle-dir`** for `verify-bundle`/`replay-bundle`, run **`python -m pytest`** (so repo root is on `sys.path`), and use `python -m anra_v5.palantiri explain-error --code <CODE>` for instant diagnostics. |
