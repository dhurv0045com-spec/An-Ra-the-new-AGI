# Execution Agent Manual (`execution_agent.md`)

> **Role:** Fast / High-Precision Experiment Builder & Execution Agent
> **Companion Manual:** [`orchestrator_agent.md`](orchestrator_agent.md) (for the Expensive Research Architect / Planner Agent)
> **Infrastructure Reference:** [`agent.md`](agent.md) (`anra_v5.planthir` / `anra_v5.palantiri` / `anra_v5.planter`)
> **Mirrors:** Kept identically at root `execution_agent.md`, `anra_v5/planthir/execution_agent.md`, and `research/planthir/execution_agent.md`. Always follow root [`AGENTS.md`](AGENTS.md).

---

## 1. Mission & Immediate Operating Rules

When the user or Orchestrator says:
> *"Read `execution_agent.md` and execute `<spec_path>`."*

Your job is to turn that research specification into **10/10 genuine, verified, production-ready code, tests, CPU qualification evidence, and (when requested) a source-pinned Kaggle/Colab notebook** — using **`anra_v5.palantiri`** (`anra_v5.planthir` / `anra_v5.planter`) so you spend your time on **real science and production reliability**, not reinventing training loops, checkpoints, stats, or archive exporters.

- **Git Branch Policy:** Work, commit, and push directly on **`hitchhiker`** (the active main branch in this repo) unless the user or spec explicitly names another branch. Never silently create and push to a random branch.
- **Repository Hygiene (`AGENTS.md`):** Never commit `.zip`, `.npz`, `.bin`, `.pt`, or raw `runs/` folders. Commit only focused Markdown + curated JSON under `Experiment Result/<Program>/`. Append a `<= 5`-nonempty-line entry (including heading) to root [`progress.md`](progress.md) before your final commit/handoff.

---

## 2. Quality-First Override Law (CRITICAL — Read First!)

**Planthir / Palantiri is in a starting condition.** It accelerates 80–90% of experiment engineering, **but it may have missing features, rough edges, or bugs.**

**Your #1 priority is 10/10 scientific & production quality — NOT blind obedience to a soft rule or a misbehaving helper.**

| Category | Status | Rules |
|---|---|---|
| **HARD SCIENTIFIC LAWS** | **NEVER BREAK** | 1. **Zero stubs / zero fake numbers:** Every requested model, control, sham, oracle, DP solver, and causal intervention must run real math. Never let an engineered pass satisfy a learned row.<br>2. **Zero split leakage & sealed confirmation:** Disjoint `train` / `dev` / `confirm` / `transfer` splits; never select on `confirm`; write a SHA-256 lock and open `confirm` once; never put truth keys (`gold`, `expected`, `truth`, `hidden`) in `UniversalObservation.metadata`.<br>3. **Uncompressed `FULL` mode:** Never shrink seeds, dataset denominators, updates, or job counts inside `FULL` mode to fit a line cap or make a test faster.<br>4. **Honest provenance & reporting:** Separate CPU engineering qualification from GPU scientific outcomes. Report negative, failed, or blocked rows honestly.<br>5. **Real custody & verification:** Never claim a test, replay, restore, or archive check passed unless you actually ran and verified it. |
| **SOFT ENGINEERING DEFAULTS** | **BREAK FREELY IF IT INCREASES QUALITY** | 1. **Soft line/file budgets:** If real `FULL`-mode worker dispatch, batched GPU evaluation, or complete scientific controls need more lines than a soft target, **write the extra clean code** instead of compressing `FULL` mode.<br>2. **"Don't read internal `.py` files":** Start from Sections 3–5 below without browsing the whole repo — **BUT** if you hit a traceback, schema error, or subtle bug in `anra_v5/planthir/`, **immediately read the exact internal function** to solve it cleanly instead of guessing.<br>3. **Forcing an incomplete/buggy Planthir helper:** If a helper (e.g. `run_registered_plan`, `StaticPromptKernel`, `build_pinned_kaggle_notebook`, `admit_slice`) lacks multi-tier DAG ordering, adaptive one-go wall-clock loops, `LATEST`-only checkpoint packing, or hits a bug, **wrap, extend, or bypass that helper cleanly inside `experiments/<name>/`** while keeping all Hard Scientific Laws intact.<br>4. **Default notebook flags:** Override `OWNER_AUTHORIZED=False` / `SESSION_MINUTES=300` whenever the owner asks for one-click full execution (e.g. `OWNER_AUTHORIZED=True`, `SESSION_MINUTES=660`, `ONE_GO=True`). |

**Mandatory Transparency Rule:** Whenever you break a soft default or work around a Planthir limitation/bug, include a concise **`PALANTIRI_FEEDBACK`** block in your final handoff stating: (1) what failed or was missing, (2) why you overrode it, (3) your workaround, and (4) the exact upgrade Planthir needs next.

---

## 3. Fast 6-Step Execution Playbook

### Step 1 — Preflight (`<30s`, Zero Wasted Turns)
1. Check branch and status: `git status -sb` (confirm you are on `hitchhiker` or the user-specified branch).
2. Read the Orchestrator's specification completely. Do **not** browse unrelated legacy directories (`v8x/`, `avendi/` sprawl).
3. Run platform self-checks:
   ```bash
   python -m anra_v5.palantiri doctor
   python -m anra_v5.palantiri calibrate
   ```
   Expect `GREEN_SIGNAL` and `6/6` fault detection.

### Step 2 — Choose Your Integration Path

#### Path A — Declarative `spec.json` Only (Existing Backends `v8`, `x1_rd2`, `x1_s20`)
Use when varying hyperparameters/arms on an existing backend without new models/tasks/losses:
```bash
python -m anra_v5.palantiri init --backend x1_rd2 --out spec.json
# Edit only your delta in spec.json
python -m anra_v5.palantiri validate --spec spec.json --canary
python -m anra_v5.palantiri run --spec spec.json --work-root runs/exp1 --canary
python -m anra_v5.palantiri verify-bundle --bundle-dir runs/exp1/result_bundle
python -m anra_v5.palantiri replay-bundle --bundle-dir runs/exp1/result_bundle --output runs/exp1/REPLAY_RECEIPT.json
```

#### Path B — New Science Package (`experiments/<name>/`, Recommended for New Tasks/Models/Campaigns)
Keep everything cleanly isolated under `experiments/<name>/` (`plan.py`, `tasks.py`, `models.py`, `module.py` — or a single `experiments/<name>.py` for small single-stage tests) plus `tests/test_<name>.py`. Never edit 10+ legacy files.

Minimal plugin registration (`experiments/<name>/module.py`):
```python
from anra_v5.palantiri import (
    CampaignOrchestrator,
    ExperimentModule,
    GenericDatasetBundle,
    GenericExperimentTrainer,
    GenericTrainRecipe,
    SessionJob,
    build_pinned_kaggle_notebook,
    export_session_archives,
    verify_full_mode_dispatch,
    verify_session_archive,
)

module = ExperimentModule(
    backend_id="my_exp",
    backend_version="my-exp-1.0",
    program="MY-EXP",
    architectures=("baseline", "treatment"),
    objectives=("ce", "custom_loss"),
    model_factory=build_my_model,          # (arch, model_config, seed) -> nn.Module
    dataset_factory=build_my_dataset,      # (dataset_config, cpu_canary) -> dict with train_data + worlds
    loss_fn=my_loss_fn,                    # (model, batch, recipe) -> torch.Tensor
    kernel_factory=MyKernel,               # optional; defaults to StaticPromptKernel
    policy_factory=build_my_policy,        # optional; defaults to LogitsPolicy
    extra_model_keys=("width", "cpu_canary"),
    extra_dataset_keys=("num_families", "seed"),
    supports_grouped_loss=True,
).register()

def register_planthir_backends() -> ExperimentModule:
    return module
```
- Subprocess CLIs auto-discover your plugin via `PLANTHIR_BACKEND_MODULES=experiments.<name>.module` (make sure `"experiments*"` is in `pyproject.toml`).

### Step 3 — Production-Grade `FULL` Mode & Kaggle 2×T4 Checklist
Apply these battle-tested rules so your experiment succeeds on **real Kaggle 2×T4 GPUs** on the very first run:

1. **Wrap Frozen Baselines Directly (Zero Baseline Confound):**
   - Import `X1TypedModel`, `solve_x1_capacity_controls`, `core_x1_jobs`, `compute_x1_d2_grouped_loss`, or `v5_model.core.initialize` directly.
   - Assert `verify_shared_backbone_and_head_overhead(ctrl, trt, backbone_prefix="core.", max_head_overhead_ratio=0.05, max_control_parity_ratio=0.02)`.
2. **Real Per-Job Multi-GPU Dispatch (`FULL` != Scaled-Up `CANARY`):**
   - For flat job lists, use `CampaignOrchestrator.run_registered_plan(...)`.
   - For multi-tier parent-child DAGs or adaptive one-go 11h runs (`session_minutes=660`), compose `reconcile_verified_resume(..., fail_on_rejection=True)` + `admit_slice(...)` + wave-by-wave `run_waves(...)` over `("cuda:0", "cuda:1")` with a real wall-clock check (`rem_min > 15.0`) before every wave.
   - Always emit `PENDING_QUEUE.json` + resumable `*_FULL.zip` if interrupted or time-guarded.
3. **Batch ALL Multi-Case Evaluation Loops (50–100× Speedup!):**
   - Never run unbatched `batch_size=1` forward loops over thousands of dev/confirm/milestone cells.
   - Right-pad causal byte prompts with `UTF8.pad_id` in chunks of `batch_size=32–64`, read logits at `lens - 1` (step 0) and `lens` (step 1), and respect `max_ctx - max_new`. Verify exact numerical parity (`atol=1e-4`) against the serial single-item decoder in a unit test!
4. **Bound Kaggle Disk Usage (`<= 19.5 GiB` `/kaggle/working` Cap!):**
   - `FP32 model + AdamW ≈ 12 bytes/param`. Saving every 128 updates across 50+ jobs without pruning will fill `/kaggle/working` and crash with `ENOSPC`.
   - Save at coarse intervals (e.g. `ckpt_every=512`), call `tr.store.prune(keep=set())` after each save, and when packaging `*_FULL.zip`, include only each lineage's `LATEST`, `MILESTONES` (if present), and `objects/<latest_sha>/*` (skipping redundant `PREVIOUS` generations).
5. **Honest Multi-Backbone GPU Profiling:**
   - Call `torch.cuda.synchronize(dev)` before and after timed updates, profile both small and large backbones (do NOT apply a 20M-param `p90` blindly to 3M-param or tiny MLP jobs), and free temporary profiling models (`del m, opt, x; torch.cuda.empty_cache()`).
6. **Clean Kaggle Notebook Generation (`build_pinned_kaggle_notebook`):**
   - Put **Config + Source Checkout + Preflight Gate (`assert torch.cuda.device_count() >= 2`, `doctor`, fast CPU canary)** in **Cell 1** BEFORE the multi-hour GPU launch in **Cell 2** so a wrong accelerator fails in 10 seconds.
   - Auto-discover mounted `/kaggle/input/**` `.zip` archives **and** auto-unpacked Kaggle Dataset folders (e.g., `checkpoints/` containing expected lineage dirs).
   - Mirror ONLY `work/archives/*` and top-level reports (`FINAL_STATUS.json`, `PENDING_QUEUE.json`, `RESULTS.md`, `METRICS.json`) to `/kaggle/working/` using zero-copy `os.link` (fallback `shutil.copy2`) — never `rglob("*.zip")` over 80 per-job archives!
   - Honor the owner's requested defaults (`OWNER_AUTHORIZED`, `SESSION_MINUTES`, `ONE_GO`).

### Step 4 — Fast, Decisive Verification Suite (Always Use `-n auto`!)
**NEVER run 560+ legacy tests single-threaded.** Follow this fast tiered chain:

1. **Experiment Unit & Full-Dispatch Tests (`pytest tests/test_<name>.py -n auto -q`):**
   - Exact math invariants: tokenized oracle parser, split custody + planted-leak catch, shortcut baselines, destructive masks, batched-vs-serial eval parity, causal/DP/stateful reset checks.
   - **`FULL`-mode dual-GPU dispatch test** (`verify_full_mode_dispatch(...)` or mocked `in_process_runner` test): verifies ALL registered job IDs, both `CUDA_VISIBLE_DEVICES="0"` & `"1"`, `--device cuda:0` remap, tight-clock `PARTIAL` deferral -> `PENDING_QUEUE.json` -> Session-2 verified resume from `*_FULL.zip` -> `COMPLETE` + `LOCKED_AND_OPENED_ONCE`, plus one-go completion if enabled.
   - **3 targeted recovery tests (~90 lines, using `in_process_runner` instead of spawning 6 cold Python subprocesses):**
     1. Mid-update `InjectedCrash` (`after_publish_before_pointer`) -> resume matches uninterrupted losses, weights, and RNG bytes.
     2. `reconcile_verified_resume(..., fail_on_rejection=True)` rejects swapped `spec_sha256` and missing archive member.
     3. One failed worker lane preserves its peer's output and emits a valid `PARTIAL` archive.
2. **Fresh-Subprocess Palantiri Plugin CLI Chain:**
   ```bash
   PLANTHIR_BACKEND_MODULES=experiments.<name>.module python -m anra_v5.palantiri init --backend <id> --out <tmp>/spec.json
   PLANTHIR_BACKEND_MODULES=experiments.<name>.module python -m anra_v5.palantiri validate --spec <tmp>/spec.json --canary
   PLANTHIR_BACKEND_MODULES=experiments.<name>.module python -m anra_v5.palantiri run --spec <tmp>/spec.json --work-root <tmp>/run --canary --device cpu
   PLANTHIR_BACKEND_MODULES=experiments.<name>.module python -m anra_v5.palantiri verify-bundle --bundle-dir <tmp>/run/result_bundle
   PLANTHIR_BACKEND_MODULES=experiments.<name>.module python -m anra_v5.palantiri replay-bundle --bundle-dir <tmp>/run/result_bundle --output <tmp>/REPLAY.json
   ```
   *(Verify `verified=true` and `MATCHED`.)*
3. **Final Pre-Push Regression Sweep:**
   ```bash
   python -m anra_v5.palantiri doctor
   python -m anra_v5.palantiri calibrate
   python -m pytest tests/test_<name>.py tests/test_elendril_tree.py tests/test_planthir_v2_audit_upgrades.py tests/test_planthir_declarative_cli_and_usability.py -n auto -q
   ```

### Step 5 — Two-Commit Source Pinning & Push (Zero Self-Referential SHAs!)
1. **Commit 1 (Code + Tests):** Stage `experiments/<name>/...` and `tests/test_<name>.py` on `hitchhiker`, commit, and record `CODE_SHA = git rev-parse HEAD`.
2. **Generate & Verify Notebook:**
   - Generate notebook pinned to `CODE_SHA`, run `ast.parse` on every Python code cell, verify `--check-notebook` returns `"match": true`, and run a clean `git archive <CODE_SHA>` extraction in a temp directory outside the repo to verify the archived source imports and expands the exact job/study counts cleanly.
3. **Commit 2 (Notebook + Curated Receipt + `progress.md`) & Push:**
   - Append a `<= 5`-nonempty-line entry to root `progress.md` (`YYYY-MM-DD | Agent | Title`, distinguishing implemented controls vs. pending GPU execution).
   - Commit on `hitchhiker`, push `git push origin hitchhiker`, and verify `git ls-remote origin refs/heads/hitchhiker` equals `git rev-parse HEAD`.

### Step 6 — Honest Handoff Report
Report concisely:
- Exact `CODE_SHA`, final `HEAD` SHA, branch (`hitchhiker`), and clickable GitHub notebook URL.
- Exact observed CPU verification counts & runtimes (never invent or extrapolate numbers).
- What remains pending owner GPU execution.
- **`PALANTIRI_FEEDBACK` block:** every Planthir gap, bug, or soft default you overrode so the platform keeps getting stronger.

---

## 4. Verified `anra_v5.palantiri` API Cheat-Sheet (Live Signatures)

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

# 2. Trainers & Recipes
GenericTrainRecipe(seed=7, total_updates=4, batch_size=8, micro_batch_size=4, lr=0.001, clip=1.0, objective="ce", sampling="uniform", schedule="cosine", precision="fp32", eq_weight=0.0, cf_weight=0.0, cf_margin=0.5)
GenericExperimentTrainer(*, backend_id, model, model_spec, recipe, loss_fn, items, data_provenance, order_seed, source_commit, root, lineage, device="cpu", cpu_canary=True, resume=False, max_cpu_params=100000, max_cpu_samples=32, max_cpu_updates=8)
BytePromptTrainer(samples, *, spec, recipe=None, source_commit, root, lineage, data_provenance, order_seed=None, device="cpu", cpu_canary=True, resume=False)

# 3. Campaign Orchestration & Verification
orch = CampaignOrchestrator(*, campaign_id, source_commit, work_dir, max_minutes=330.0, reserve_minutes=45.0, min_disk_bytes=2*2**30, in_process_runner=None)
orch.run_registered_plan(jobs, *, command_builder=None, job_runner=None, devices=("cuda:0", "cuda:1"), resume_input=None, measured_seconds_per_update=None, profile_fn=None, per_job_disk_bytes=128*2**20, outer_remaining_minutes=None, inventory_only=False, fail_on_resume_rejection=True, export_prefix=None, export_dir=None, write_full_zip=True)
orch.admit_slice(jobs, *, measured_seconds_per_update, per_job_disk_bytes, parallel_workers=2, outer_remaining_minutes=None, startup_overhead_seconds=0.0, export_reserve_bytes=0)
orch.reconcile_verified_resume(imported_receipts, registry_rows, *, required_archive_members=(), fail_on_rejection=False)
orch.run_waves(jobs, *, command_builder, devices=("cuda:0", "cuda:1"), resume=False)
verify_full_mode_dispatch(dispatch_fn, *, expected_job_ids, expected_seeds=(7, 19, 41), required_devices=("cuda:0", "cuda:1"), min_dataset_counts=None, tight_session_minutes=None, full_session_minutes=300.0, require_multi_session_resume=True)

# 4. Archives, Notebooks, & Backbone Parity
export_session_archives(source_root, output_dir, *, prefix, analysis_files=None, full_files=None, summary_files=None, evidence_files=None, status="COMPLETE", source_commit=None, summary_max_bytes=5*2**20, write_full_zip=True)
verify_session_archive(archive_path, *, max_bytes=None, expected_source_commit=None, require_sidecar_sha256=True, require_v6_manifest=True, required_members=())
build_pinned_kaggle_notebook(*, source_commit, title, launch_command, owner_authorized=False, repo_url=None, campaign_mode="FULL", session_minutes=300, resume_input=None, inventory_command=None, discover_kaggle_inputs=(), mirror_outputs_to_working=False, preflight_canary_backend=None, require_dual_gpu=False, emit_file_links=False, backend_modules=())
verify_shared_backbone_and_head_overhead(control_model, treatment_model, *, backbone_prefix="core.", max_head_overhead_ratio=0.05, max_control_parity_ratio=None)
```

---

## 5. Battle-Tested Gotchas (Avoid Every Known Trap!)

| Trap / Error | Exact Fix |
|---|---|
| `forbidden_metadata_key:gold` in Gate 1 | `UniversalObservation.metadata` forbids keys `{"expected", "gold", "gold_answer", "hidden", "truth", "ground_truth"}` recursively **even if the value is `None`**. Never include placeholder truth keys. |
| `AttributeError: 'str' object has no attribute 'modified_world'` | `EnvironmentKernel.decompose_subgoals(world)` MUST return `Sequence[SubgoalProbeSpec]` (or `()`), NEVER strings. |
| Gate 4 fails on `ablate_mechanism` or stateful policy | `model.ablate_mechanism(self, mode="...")` MUST accept zero required arguments. Custom policies MUST expose `self.policy_id: str` and implement `reset_episode(self) -> None`. |
| `FrozenInstanceError` on `SessionJob` | `SessionJob` is `@dataclass(frozen=True)`. Always use `dataclasses.replace(s, device=...)`. |
| `AttributeError: 'WorkerJobSpec' object has no attribute 'job_id'` | Inside `in_process_runner(job, env, cmd)`, the worker ID attribute is **`job.worker_id`** (NOT `job.job_id`). Return `int` or `{"exit_code": int}`. |
| `KeyError: 'status'` on `reconcile_verified_resume` | Returns `verified_completed_jobs`, `rejected_jobs`, `pending_job_ids`, `next_resumable_job_id` — **no `"status"` key**. Checks `completed_updates >= target_updates`. |
| `KeyError: 'hashes'` on `export_session_archives` | Digest dict key is **`"archive_sha256"`** (NOT `"hashes"`). |
| `ValueError` calling `verify_single_planthir_zip` on session ZIPs | Use **`verify_session_archive(...)`** and **`rescore_compact_analysis_archive(...)`** for `export_session_archives` ZIPs (`*_SUMMARY/EVIDENCE/ANALYSIS/FULL.zip`). Use `verify_result_bundle` / `verify_single_planthir_zip` ONLY for Cartridge `result.zip`. |
| `v5_model.core.Core` `ImportError` | Construct via **`v5_model.core.initialize(spec, seed, torch_module=torch)`** + `packed_layout(segments, torch_module=torch)`. Exposes `.forward_hidden(...)` and `.blocks`. |
| `UTF8.decode` `ValueError("expected content tokens only")` | `UTF8.decode(ids)` rejects tokens `< 4` (`pad=0, bos=1, eos=2, ANSWER=3`). Filter `[t for t in ids if t >= 4]`. `SpecialToken` is an `Enum`, use `UTF8.special(SpecialToken.ANSWER)`. |
| `RuntimeError: element 0 of tensors does not require grad` | Any bounded weight-update helper called from an `@torch.no_grad()` evaluator MUST wrap its forward/backward step in `with torch.enable_grad():`. |
| `NotImplementedError` subtracting `bool` buffers in weight-norm caps | Iterate over `model.named_parameters()` (floating params only), NOT `state_dict()` (which includes `bool`/int buffers). |
| `CUDA_VISIBLE_DEVICES` double-indexing on lane 1 | `IndependentWorkerScheduler` sets `CUDA_VISIBLE_DEVICES="1"` and remaps `--device cuda:1` -> `--device cuda:0`. Child workers should always use the `--device` passed to them (`cuda:0`). |
| Windows `MAX_PATH` (`FileNotFoundError`) unpacking nested ZIPs under `pytest-xdist` | Unpack directly into `wr` (avoid deep intermediate `resume_extract/` nesting) and prefix paths `>= 240` chars with `\\?\` on Windows. |
| PowerShell 5.1 syntax (`&&`, `||`, `<<'EOF'`, `>`) | Use `; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }` instead of `&&`, use `edit_file`/`write_file` instead of heredocs, and avoid PowerShell `>` for UTF-8 JSON (it writes UTF-16LE BOM). |
| CLI `--dir` error | Always use **`--bundle-dir`** for `verify-bundle` and `replay-bundle`, and use `python -m anra_v5.palantiri explain-error --code <CODE>` for instant error lookup. |
