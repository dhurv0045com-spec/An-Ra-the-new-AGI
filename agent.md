# Planthir / Palantiri (`anra_v5.planthir` / `anra_v5.palantiri` / `anra_v5.planter`) — Agent Guide

> **Canonical Manual:** Guide for any AI agent installing, verifying, or running controlled experiments with the **Planthir** experiment system (aliases: `planthir`, `palantiri`, `planter`; Python modules `anra_v5.planthir`, `anra_v5.palantiri`, `anra_v5.planter`). Kept identically at root `agent.md` and `anra_v5/planthir/agent.md`. Follow root [AGENTS.md](AGENTS.md) at all times. Read [the reliability & upgrade handoff](research/planthir/GEMINI_RELIABILITY_AND_UPGRADE_HANDOFF.md) and [the 6-task productivity protocol](research/planthir/SIX_TASK_PRODUCTIVITY_STUDY_PROTOCOL.md) when planning infrastructure changes or productivity studies.

---

## 0. Cross-Branch Prompt Protocol (Read First)

The **technology name** (`planthir` / `palantiri` / `planter`) is independent of the **Git branch name** (for example `hitchhiker`, `Elendril`, `palantiri`, or `planthir`).

When the user asks you in natural language to run an experiment using a named branch — for example:

> **"Use the technology planthir (or planter / palantiri) from the branch `hitchhiker` (or `Elendril`) to perform the experiment."**

Execute these steps immediately without asking redundant questions:

1. **Identify `<source-branch>` from the user's request** (e.g. `hitchhiker`, `Elendril`, `palantiri`, or `planthir`) and **work on a new experiment branch**:
   ```bash
   git checkout -b exp/<short-experiment-name>
   ```
2. **Pull the technology files from `origin/<source-branch>` only if `anra_v5/planthir` is absent (or older) in the current branch:**
   ```bash
   git fetch origin <source-branch>
   git checkout origin/<source-branch> -- anra_v5/planthir anra_v5/palantiri anra_v5/planter anra_v5/elendril/v8 pyproject.toml tools/audit_tree.py docs/TREE_INVENTORY.json agent.md research/planthir
   ```
   *(If you are already on `hitchhiker`, `Elendril`, `palantiri`, or `planthir`, these files are already installed locally.)*
3. **Run preflight self-checks (`GREEN_SIGNAL` + `6/6` fault calibration):**
   ```bash
   python -m anra_v5.planthir doctor
   python -m anra_v5.planthir calibrate
   ```
   *(Equivalent CLI aliases: `python -m anra_v5.palantiri` or `python -m anra_v5.planter`.)*
4. **Choose Path 1 or Path 2 below** based on whether your experiment reuses an existing backend (`v8`, `x1_rd2`, `x1_s20`) or introduces a new architecture/environment/loss.

---

## 0.1 Two-Agent Framework Protocol (Expensive Prompt Architect + Fast Execution Builder)

This workflow pairs **(1) a higher-capability Research Architect model** that authors the experiment prompt with **(2) a faster, cheaper Builder model** that implements and tests it on top of `anra_v5.palantiri`. The goal is **minimum tokens & lines (`~70%` smaller, `~3x` faster) with ZERO loss of scientific quality (`9.8/10`)**.

### Instructions for Role 1 — The Expensive Research Architect (Prompt-Authoring Agent)

When writing an experiment handoff prompt for the Builder Agent, obey this **6-Rule Zero-Degradation Specification**:

1. **Require Verification Against the Upgraded §0 Replay Gate (Never Make the Builder Debug Core):**
   - Instruct the builder to verify `init -> validate -> run -> verify-bundle -> replay-bundle` on both the relevant built-in backend (`x1_s20` / `x1_rd2`) and the new plugin (`PLANTHIR_BACKEND_MODULES=experiments.<pkg>.module`) in fresh subprocesses. Do **not** order the builder to rewrite `anra_v5/planthir/*` internals unless a verified upstream bug is identified.
2. **Mandate Wrapping Frozen Baselines Directly (Zero Baseline Confound):**
   - When extending an existing model family (e.g., X1 -> X2), explicitly forbid re-implementing baseline architectures or losses from scratch. Command the builder to import and wrap `X1TypedModel`, `solve_x1_capacity_controls`, `core_x1_jobs`, and `compute_x1_d2_grouped_loss` directly, and assert `verify_shared_backbone_and_head_overhead(...)` (`backbone_digest_match=True`, `<=5%` head overhead, `<=2%` active-control parity).
3. **Delegate Infrastructure & `FULL`-Mode Multi-GPU Execution to `anra_v5.palantiri` (Zero Custom Loops/Exporters):**
   - Forbid writing custom trainers, serial GPU loops, checkpoint stores, bootstrap CI loops, lesion routers, or ZIP exporters. Require `ExperimentModule`, `GenericExperimentTrainer`, `CampaignOrchestrator.run_registered_plan(...)` (which unifies `prepare`, disposable-clone live profiling, `RESUME_INPUT` extraction + `reconcile_verified_resume(..., fail_on_rejection=True)`, dual-clock `admit_slice` -> `PENDING_QUEUE.json`, parallel 2-GPU `run_waves`, and verified `SUMMARY/EVIDENCE/ANALYSIS/FULL` archive emission), `verify_session_archive(...)`, `build_pinned_kaggle_notebook(..., repo_url=..., campaign_mode="FULL", session_minutes=300, resume_input=..., inventory_command=...)`, `bootstrap_paired_families`, `build_paired_seed_observations`, and `build_analyst_first_page_report`.
   - For resilience checks, run `python -m anra_v5.palantiri calibrate` **plus ONLY the 3 campaign-specific recovery tests** (~90 lines total): (a) mid-update interrupt/resume parity, (b) swapped `spec_sha256` / missing archive member rejection via `reconcile_verified_resume(..., fail_on_rejection=True)`, and (c) one failed worker lane preserving peer work + valid partial export.
4. **Specify Vectorized Symbolic & DP Math Instead of Verbose OOP Trees:**
   - Instruct the builder to represent finite-state rule banks & version spaces as compact NumPy transition tensors (`(N_laws, 4)` + `(N_laws, 4, 4)` boolean truth-table masking, ~150 lines) and POMDP public-belief solvers as exact vectorized Bellman recursions (~70 lines). **Zero stubbing** of controls, symbolic solvers, DP references, baselines, or §11 fixtures is permitted.
5. **Command Programmatic Registries, Clear Package Path(s), & Zero-Edit Notebooks:**
   - Pick explicit importable package path(s) (`experiments/<name>/` matching `PLANTHIR_BACKEND_MODULES=experiments.<name>.module`), build registries programmatically (no committed 1,000-line `REGISTRY.json` dumps), and generate any Kaggle 2xT4 notebook via `build_pinned_kaggle_notebook(...)` (with `repo_url`, `inventory_command`, `OWNER_AUTHORIZED=False` safe `--inventory-only` dry-run, and `--check` / clean `git archive` verification).
   - **Never include a §9-style table ordering edits to 10–14 legacy files** (`v8x/*`, `tools/*_campaign.py`, `tools/analyze_*.py`).
6. **Enforce Realistic Tiered Line Budgets & Mandatory `verify_full_mode_dispatch(...)` Gate:**
   - **Single-stage experiment:** `<= 2` files (`experiments/<name>.py` + `tests/test_<name>.py`), **`<= 450` lines total**, `<15s` CPU suite.
   - **Standard multi-world program (1–3 stages, `~10–25` GPU jobs, e.g., V8-X2):** `<= 4` code files + `1` test file = **`~1,600–1,900` lines total**.
   - **Mega-program (`4+` stages / `50+` studies / `50+` GPU jobs, e.g., Avendi V8-Y3):** **NEVER cram 50+ studies into `<= 1,900` lines** (that forces silent `FULL`-mode compression). EITHER split into **2–3 chained experiment modules** (each `<= 1,500` lines, e.g. `foundation_suite` -> `architecture_suite` -> `causal_suite`) OR budget **`~2,500–3,200` lines total** backed by `CampaignOrchestrator.run_registered_plan(...)`.
   - Always mandate **`verify_full_mode_dispatch(...)`** (`<2s` mocked-GPU `FULL`-mode dry-run verifying every registered job ID, all required seeds `(7, 19, 41)`, both `cuda:0` and `cuda:1` in parallel, uncompressed production dataset denominators, and tight-session deferral to `PENDING_QUEUE.json` + Session-2 resume).

#### Copy-Paste Header for Prompts Sent to the Builder Agent

```markdown
## MANDATORY PALANTIRI BUILDER CONTRACT (ZERO DEGRADATION, MINIMUM TOKENS)
1. Read `agent.md` Section 1 (Path 2) & Section 3 (Exact Contracts Cheat-Sheet) FIRST. Do NOT read internal `anra_v5/planthir/*.py` source files unless an error code tells you to.
2. IMPLEMENT 100% OF THE SCIENCE IN BOTH CANARY AND FULL MODE (ZERO STUBS): Build every requested mechanism, active control, symbolic/DP solver, baseline, and decisive mathematical fixture completely. Never compress seeds, dataset denominators, or job lists inside `FULL` mode.
3. ZERO DUPLICATE INFRASTRUCTURE & ZERO LEGACY REWRITES:
   - Import and wrap frozen baselines directly (e.g., `X1TypedModel`, `compute_x1_d2_grouped_loss`, `core_x1_jobs`); check `verify_shared_backbone_and_head_overhead(...)`. Never edit or duplicate `anra_v5/elendril/v8x/*` or create custom `tools/*_campaign.py` / `tools/analyze_*.py` / committed `REGISTRY.json` dumps.
   - Delegate training and FULL-mode 2-GPU multi-session orchestration 100% to `CampaignOrchestrator.run_registered_plan(...)` (`prepare`, disposable-clone live profiling, `RESUME_INPUT` + `reconcile_verified_resume`, dual-clock `admit_slice`, 2-GPU `run_waves`, and verified `export_session_archives` + `verify_session_archive(...)`), plus `build_pinned_kaggle_notebook(..., repo_url=..., inventory_command=...)`.
4. PACKAGE & REALISTIC LINE BUDGET: Place code cleanly under `experiments/<name>/` (or 1 file for single-stage). Respect the matching tier (`<= 450` lines single-stage; `<= 1,900` lines for 1–3 stages / `~10–25` jobs; split `50+`-study mega-programs into 2–3 chained modules <=1,500 lines each or budget `2,500–3,200` lines) using vectorized NumPy/PyTorch tensor ops.
5. REQUIRED ACCEPTANCE GATES: `python -m pytest tests/test_<name>.py -q` (<15s, including `verify_full_mode_dispatch(...)`, `calibrate`, and 3 targeted recovery checks) + fresh-subprocess `PLANTHIR_BACKEND_MODULES=... python -m anra_v5.palantiri init -> validate -> run -> verify-bundle -> replay-bundle` + safe notebook `--inventory-only` dry-run check.
```

### Instructions for Role 2 — The Fast / Cheaper Experiment Builder Agent

1. **Read `agent.md` Sections 1–3 only** before writing code; do not spend turns browsing unrelated repository history or re-reading 2,000-line internal modules.
2. **Never copy `v8x/` or `avendi/` multi-file sprawl.** Start from the Path 2 `ExperimentModule` + `CampaignOrchestrator.run_registered_plan(...)` recipe and Section 3 contracts below.
3. **Run `verify_shared_backbone_and_head_overhead()`, `policy.reset_episode()`, `verify_full_mode_dispatch()`, `verify_session_archive()`,** and **one clean subprocess CLI chain (`init -> validate -> run -> verify-bundle -> replay-bundle`)** to prove both scientific/replay custody and uncompressed `FULL`-mode multi-GPU dispatch.

---

## 1. Two Crisp Workflows (Pick the Right Path Immediately)

### Path 1 — Varying Hyperparams / Existing Arms (`spec.json` only, `<45` lines)

> **Scope:** Use this path **only** when your experiment uses an already-registered backend, architecture, dataset, and loss (`v8`, `x1_rd2`, `x1_s20`, or an `ExperimentModule` exposed via `PLANTHIR_BACKEND_MODULES`). You do **not** need to open Python files inside `anra_v5/planthir/`.

1. **Generate & edit only your delta in `spec.json`:**
   ```bash
   python -m anra_v5.planthir init --backend x1_rd2 --out spec.json   # or --backend v8 or --backend x1_s20
   ```
   - Concise `spec.json` files can omit `contrasts`, `hypotheses`, `factors`, and `typed_contrasts`: when one factor differs between `control_arm` and a `treatment_arm`, Planthir auto-infers the single-factor contrast cleanly (multi-factor confounds still fail Gate 2).
   - Use `"extra_config": {...}` inside `dataset`, `model`, `recipe`, or `execution` for extra scalar parameters.
2. **Validate & run canary:**
   ```bash
   python -m anra_v5.planthir validate --spec spec.json --canary
   python -m anra_v5.planthir explain-error --code <CODE>   # lookup fix without reading .py files
   python -m anra_v5.planthir run --spec spec.json --work-root runs/exp1 --canary
   python -m anra_v5.planthir inspect-run --work-root runs/exp1
   ```
3. **GPU production (`--device cuda:0 --no-canary`) & bundle verification:** Commit tracked edits (`source_commit: "HEAD"`), set `"cpu_canary": false`, `"sample_limit_for_cpu": null`, run/resume, then verify and replay:
   ```bash
   python -m anra_v5.planthir resume --spec spec.json --work-root runs/exp1 --no-canary --device cuda:0
   python -m anra_v5.planthir verify-bundle --bundle-dir runs/exp1/result_bundle
   python -m anra_v5.planthir replay-bundle --bundle-dir runs/exp1/result_bundle --output runs/exp1/REPLAY_RECEIPT.json
   ```

### Path 2 — New Architecture / New Environment / New Loss (`ExperimentModule` Python API, `~300–500` lines)

> **Scope:** When you introduce new model math, a new task environment, or a new loss function, put your novel science in **one standalone module** (e.g. `experiments/<name>.py`) and plug it into `anra_v5.planthir.ExperimentModule` + `GenericExperimentTrainer` + `CampaignOrchestrator`. Make **zero edits** to `anra_v5/planthir/` core files (`backend.py`, `agent_tools.py`, `causal.py`, `design.py`, `checkpoint_intelligence.py`).

Minimal `experiments/<name>.py` recipe (**3 required hooks + optional `kernel_factory` / `policy_factory`**):

```python
from anra_v5.planthir import (
    CampaignOrchestrator,
    ExperimentModule,
    GenericDatasetBundle,
    SessionJob,
    batched_rollout,
    build_pinned_kaggle_notebook,
    export_session_archives,
)

# 1. model_factory(architecture: str, model_config: Mapping, seed: int) -> nn.Module
#    Optional: model.ablate_mechanism() contextmanager + model.compute_depth_context(steps) for Gate 4
# 2. dataset_factory(dataset_config: Mapping, cpu_canary: bool) -> dict with:
#    {"train_data": GenericDatasetBundle(...), "train_worlds": ..., "eval_worlds": ..., "multihop_worlds": ..., "probe_train_worlds": ...}
# 3. loss_fn(model: nn.Module, batch: Sequence[Any], recipe: GenericTrainRecipe) -> torch.Tensor
# 4. Optional kernel_factory / policy_factory: omitted defaults use StaticPromptKernel + LogitsPolicy!

module = ExperimentModule(
    backend_id="my_exp",
    architectures=("baseline", "treatment"),
    objectives=("ce", "custom_loss"),
    model_factory=build_my_model,
    dataset_factory=build_my_dataset,
    loss_fn=my_loss_fn,
    action_labels=("A", "B"),  # optional; defaults cleanly to StaticPromptKernel + LogitsPolicy
    extra_model_keys=("my_width",),
    extra_dataset_keys=("num_samples",),
).register()
```

- Subprocess CLIs auto-load your plugin without touching `backend.py` by setting `PLANTHIR_BACKEND_MODULES=experiments.<name>` (provided your module calls `.register()` or defines `register_planthir_backends()`), and `python -m anra_v5.planthir init --backend my_exp --out spec.json` works out of the box!
- **Built-in helpers you should reuse instead of rewriting:**
  - *Frozen baseline wrapping & capacity parity (NEVER rewrite baselines from scratch):* Always import `X1TypedModel` (`anra_v5/elendril/v8x/models.py`), `solve_x1_capacity_controls` (`anra_v5/elendril/v8x/contracts.py`), `core_x1_jobs` / `extension_x1_jobs` (`v8x/protocol.py`), and `compute_x1_d2_grouped_loss` directly, then assert `verify_shared_backbone_and_head_overhead(ctrl, trt, backbone_prefix="core.", max_head_overhead_ratio=0.05, max_control_parity_ratio=0.02)`.
  - *Legacy byte / V6 / Y1 checkpoints & training:* `LegacyByteObservablePolicy`, `load_legacy_byte_checkpoint`, `locate_checkpoint_store_root`, `build_deterministic_retraining_recipe`, `BytePromptTrainer`, `frame_byte_prompt_sample`, `compute_ce_and_pair_margin_terms`, `audit_pair_margin_gradient_sign_reversal`.
  - *4-cell crossover & family stats:* `verify_crossover_fixture`, `score_crossover_panel`, `bootstrap_paired_families`, `audit_sampled_strata`.
  - *Diagnostic coverage & paired comparison reporting:* `summarize_diagnostic_coverage`, `stratify_intervention_worlds`, `compare_paired_episode_records`, `PairedSeedObservation`, `build_paired_seed_observations`, `run_paired_comparison_preflight`, `build_analyst_first_page_report`, `rescore_compact_analysis_archive`.
  - *Vectorized solver & programmatic registry best practices:* Represent finite-state rule banks & symbolic version spaces as compact NumPy transition tensors (`(N_laws, 4)` + `(N_laws, 4, 4)` boolean truth-table masking, ~150 lines) and POMDP public-belief solvers as vectorized Bellman recursions (~70 lines). Generate multi-block job registries programmatically in Python (zero committed 1,000-line `REGISTRY.json` dumps).
  - *Campaign admission, verified resume, `FULL`-mode dispatch & Kaggle notebooks:* `CampaignStageSpec`, `CampaignOrchestrator.run_registered_plan(...)` (prepare + `RESUME_INPUT` + disposable-clone profiling + `admit_slice` + 2-GPU `run_waves` + `export_session_archives` + `verify_session_archive`), `verify_full_mode_dispatch(...)`, `verify_session_archive(...)`, `CampaignOrchestrator.admit_slice(..., measured_seconds_per_update=..., per_job_disk_bytes=..., outer_remaining_minutes=...)`, `CampaignOrchestrator.reconcile_verified_resume(...)`, `CampaignOrchestrator.run_waves`, `CampaignOrchestrator.run_stage_dag`, `CampaignOrchestrator.evaluate_frozen_checkpoints`, `build_verification_matrix`, `render_campaign_results_markdown`, `export_session_archives(..., summary_files=..., evidence_files=..., full_files=..., write_full_zip=True)`, and `build_pinned_kaggle_notebook(..., repo_url=..., campaign_mode="FULL", session_minutes=300, resume_input=None, inventory_command=..., extra_pinned_repos=..., preflight_canary_backend=..., require_dual_gpu=True, emit_file_links=True, mirror_outputs_to_working=True)`.

---

## 2. Fast Test Budget & Anti-Slowdown Rules

- **Do NOT spawn 6+ cold `subprocess` Python processes inside `pytest`** unless explicitly testing OS crash isolation (`_exit` recovery). Multi-worker scheduler/campaign tests must use `CampaignOrchestrator.run_registered_plan(...)`, `verify_full_mode_dispatch(...)`, `IndependentWorkerScheduler(..., in_process_runner=...)`, or `CampaignOrchestrator(..., in_process_runner=...)` so suites complete in **`<15s`** instead of `70s+`.
- **Use `palantiri calibrate` + 3 targeted recovery tests (~90 lines) instead of a 500-line custom chaos harness:** Run `calibrate` for the 6 instrument faults, and test ONLY (1) mid-update resume parity vs uninterrupted training, (2) swapped `spec_sha256` / missing archive member rejection via `reconcile_verified_resume(..., fail_on_rejection=True)`, and (3) one failed worker lane preserving its peer's work + `PARTIAL` export.
- **Tiered & parallel verification (NEVER run 560+ legacy tests single-threaded):**
  1. Iterate fast on `python -m pytest tests/test_<new>.py -q` (`<15s`, including `verify_full_mode_dispatch(...)` in `<2s`).
  2. Before final handoff/push, run **`python -m pytest tests/test_<new>.py tests/test_planthir_v2_audit_upgrades.py tests/test_planthir_declarative_cli_and_usability.py tests/test_elendril_tree.py -q`** plus `python -m anra_v5.palantiri doctor` and `python -m anra_v5.palantiri calibrate`.
  3. Only run a full repository sweep if explicitly requested, and parallelize it with **`python -m pytest -n auto -q`** when `pytest-xdist` is available.
- **CPU canary budget:** Keep CPU canary runs at `<= 100,000` parameters, `<= 32` samples, and `<= 8` updates.

---

## 3. Gotchas & Exact Core Contracts (Avoid Reading 20 Files)

| Component | Exact Contract / Gotcha |
|---|---|
| **Plugin Registration (`ExperimentModule`)** | Call **`ExperimentModule(...).register()`** (there is NO `register_experiment_module()` function!). For subprocess CLI loading set `PLANTHIR_BACKEND_MODULES=experiments.<pkg>.module` and include `"experiments*"` in `pyproject.toml`. |
| **`dataset_factory(dataset_config, cpu_canary)`** | Must return `{"train_data": GenericDatasetBundle(items=tuple(...), dataset_name=..., data_identity=..., manifest={...}), "train_worlds": [...], "eval_worlds": [...], "multihop_worlds": [...], "probe_train_worlds": [...]}`. `train_worlds` and `eval_worlds` MUST have disjoint `kernel.structural_signature(w)` values (or Gate 2 fails). |
| **`UniversalObservation` & Truth-Leak Firewall** | Required fields: `obs_id`, `rendered_text`, `available_tools: dict[str, bool]`, `consumed_tools: frozenset[str]`, `decision_index: int`, `budget_remaining: int` (`>= 1`), `metadata: dict` (NO `world_id` or `decision_step` top-level fields!). Inside `metadata` (recursively, case-insensitive), keys `{"expected", "gold", "gold_answer", "hidden", "truth", "ground_truth"}` raise **`forbidden_metadata_key:<k>` even when the value is `None`** — never include placeholder truth keys in `metadata`. |
| **`EnvironmentKernel` & `SubgoalProbeSpec`** | Requires `kernel_id`, `initial_state`, `observe`, `ideal_action`, `step`, `verify`, `decompose_subgoals`, plus optional `target_symbol`, `structural_signature`, `inject_distractor`, `permute_world`, `reverse_world`. **`decompose_subgoals(world)` MUST return `Sequence[SubgoalProbeSpec]`** (`subgoal_id, stage_name, hop_index, modified_world, expected_operation, expected_argument, description=""`) or `()` — NEVER return `tuple[str, ...]` (Gate 4 fails). |
| **`ObservablePolicy` & `ablate_mechanism`** | Custom policies MUST expose **`self.policy_id: str`** and implement **`reset_episode(self) -> None`** (called between worlds by evaluator and replay), `propose_with_telemetry`, `extract_representations`. `model.ablate_mechanism(self, mode: str = "...")` MUST accept zero required arguments (Gate 4 calls `model.ablate_mechanism()` with no args!). |
| **`IndependentWorkerScheduler` & `WorkerJobSpec`** | Callback `in_process_runner(job, env, cmd)` receives `WorkerJobSpec` whose ID attribute is **`job.worker_id`** (NOT `job.job_id`). Must return an **`int`** exit code (`0`/`1`) OR **`{"exit_code": int}`** (NEVER `{"returncode": ...}` and NEVER raise unhandled exceptions when testing lane-failure isolation). |
| **`CampaignOrchestrator.run_registered_plan` & `reconcile_verified_resume`** | Use `orch.run_registered_plan(jobs, ...)` for `FULL`-mode multi-GPU campaigns and assert `verify_full_mode_dispatch(...)`. **`reconcile_verified_resume(...)`** returns `{"schema", "campaign_id", "source_commit", "verified_completed_jobs", "rejected_jobs", "pending_job_ids", "next_resumable_job_id"}` (**NO `"status"` key**). |
| **`export_session_archives` & Archive Verifier Split** | **`export_session_archives(...)`** returns per-archive SHA-256 digests under **`"archive_sha256"`** (**NO `"hashes"` key**) and supports `write_full_zip: bool = True`. Use **`verify_session_archive(...)`** and **`rescore_compact_analysis_archive(...)`** for `export_session_archives` ZIPs (`*_SUMMARY.zip`, `*_EVIDENCE.zip`, `*_ANALYSIS.zip`, `*_FULL.zip` with `BUNDLE_MANIFEST.json`/`SHA256SUMS`); use **`verify_result_bundle(...)` / `verify_single_planthir_zip(...)`** ONLY for Cartridge `result.zip` / `checkpoint_full.zip` (`ARCHIVE_MANIFEST.json`). |
| **Reporting & Stats Keyword Signatures (`anra_v5.palantiri`)** | `bootstrap_paired_families(control_scores, treatment_scores, *, resamples=2000, seed=1801)` (in `anra_v5.palantiri` / `design.py`); `build_paired_seed_observations(arm_records, *, contrast_id="PRIMARY", control_arm=..., treatment_arm=..., metric_key="value")`; `build_verification_matrix(*, byte_integrity="PASS", ...)` (keyword status strings); `build_analyst_first_page_report(*, title, campaign_id, source_commit, decision_summary, control_arm, treatment_arm, paired_seed_bundle, ...)`; `render_campaign_results_markdown(*, title, campaign_id, source_commit, status, verification_matrix, ...)`. |
| **Bundle Layout & Replay Cohort (`ARCHIVE_MANIFEST.json`)** | Archive member inside `result.zip` / `checkpoint_full.zip` is **`ARCHIVE_MANIFEST.json`** (NOT `MANIFEST.json`), and `REPLAY_COHORT.json` stores `ordered_world_ids`, `cohort_sha256`, `worlds`, and multi-step `reference_predictions` (`history`). `run --work-root <W>` writes bundles to **`<W>/result_bundle`** (NOT `<W>/bundle`). Missing non-built-in plugins fail closed with `MISSING_BACKEND_PLUGIN:<id>`. |
| **`v5_model.core.ModelSpec` & `v5_model.core.initialize`** | `v5_model.core` does **NOT** export a `Core` class at module level — construct via **`v5_model.core.initialize(spec, seed, torch_module=torch)`** using **`ModelSpec(vocabulary=...)`** (NOT `vocab_size`). The returned module exposes `.forward_hidden(tokens, positions, mask)`, `.blocks`, and `__call__(tokens, positions, mask)` (NO `.logits()` method; compute `positions, mask_attn = packed_layout(segments, torch_module=torch)`). |
| **`v5_training.state.IdentityBindings`** | Field name is **`optimizer_spec_sha256`** (NOT `optimizer_group_sha256`). |
| **`IndependentWorkerScheduler` GPU Pinning** | When `WorkerJobSpec(device="cuda:1")` runs, the scheduler sets `CUDA_VISIBLE_DEVICES="1"` in the child AND automatically rewrites `--device cuda:1` -> `--device cuda:0` (`remap_child_cuda_device=True`). Child processes should read `PLANTHIR_VISIBLE_CUDA_DEVICE` (`cuda:0`) and never manually index `cuda:1` when `CUDA_VISIBLE_DEVICES` is pinned. |
| **Planthir CLI Bundle Flags** | Always use **`--bundle-dir`** (`verify-bundle --bundle-dir ...`, `replay-bundle --bundle-dir ...`), never `--dir`. |
| **Repository Audit Rules (`AGENTS.md` / `audit_tree.py`)** | Never commit `.zip`, `.npz`, `.bin`, `.pt`, or raw `runs/` folders. Keep curated JSON/Markdown only under `Experiment Result/<Program>/`. Every `progress.md` entry must be **`<= 5` non-empty lines** including its heading (`YYYY-MM-DD \| Agent \| Title`). |
| **`UTF8`, `UTF8.decode` & `SpecialToken` (`anra_v5/elendril/representation.py`)** | `SpecialToken` is a string `Enum` (`int(SpecialToken.EOS)` raises `TypeError`). Always use `UTF8.bos_id` (`1`), `UTF8.eos_id` (`2`), `UTF8.pad_id` (`0`), and `UTF8.special(SpecialToken.ANSWER)` (`3`). `UTF8.encode(text)` shifts UTF-8 bytes by `+4`. **`UTF8.decode(ids)` raises `ValueError("expected content tokens only")`** if any token ID `< 4` is passed — always filter `[t for t in ids if t >= 4]`. `UTF8.frame(prompt, answer)` returns `FramedExample` with `.ids` and `.loss_mask` (target starts at `framed.loss_mask.index(True)` right after token `3`). Or call `frame_byte_prompt_sample(...)`. |
| **`V6Trainer` / `FoundationTrainer` (`anra_v5/elendril/foundation_training.py`)** | Samples must be `TokenSample(task_id, tokens, eligible, world_id, category)` (or subclass `BytePromptSample`). Call `configure_device(device)` before trainer init. `Recipe` uses `lr` and `clip` (never `learning_rate` or `grad_clip`), and LR schedule is `self.learning_rate(u)`. `TrainingState.complete` is a `@property`, never pass `complete=` to `dataclasses.replace(self.state, ...)`. Prefer built-in `BytePromptTrainer` for new UTF-8 experiments. |
| **`analyze_paired_seed_runs` (`anra_v5/planthir/design.py`)** | `analyze_paired_seed_runs` takes paired `observations` with keys `{"contrast_id", "pair_id", "init_seed", "order_seed", "control_value", "treatment_value"}` and returns `results["contrasts"][cid]["variance_status"]` (`"DEGENERATE_ZERO_VARIANCE"` or `"NONZERO_VARIANCE"`). Never pass un-paired separate arm rows directly: build typed `PairedSeedObservation` objects via `build_paired_seed_observations(...)` (verified by `run_paired_comparison_preflight()` in `doctor`) so dropped arms report missing pairs cleanly without discarding completed jobs. |
| **`ExperimentModule` `dataset_factory` splits & defaults (`anra_v5/planthir/runtime.py`)** | Omitting `kernel_factory` / `policy_factory` cleanly defaults to `StaticPromptKernel` + `LogitsPolicy`, and omitting `probe_train_worlds` / `multihop_worlds` auto-namespaces fallback slices to avoid self-inflicted `TRANSITIVE_COHORT_LEAKAGE`; for custom disjoint evaluation sets, explicitly return distinct `"probe_train_worlds"` / `"multihop_worlds"` in `dataset_factory`. |

---

## 4. Complete CLI Reference (`anra_v5.planthir` / `anra_v5.palantiri` / `anra_v5.planter`)

All CLI subcommands emit strict JSON (`0` = valid/verified, `1` = error). Deprecated legacy commands (`run-validation-study`, `run-gpu-validation`) intentionally exit `1`.

| Command | Purpose |
|---|---|
| `python -m anra_v5.planthir doctor` | Read-only check of bindings, V8 tool-consumption firewall, starter templates, git status, RSS/disk probes |
| `python -m anra_v5.planthir calibrate` | 6-fault synthetic observatory benchmark (`6/6` detected) |
| `python -m anra_v5.planthir schema` / `agent-info` | Machine-readable allowed fields, backends, and Python input/output contracts |
| `python -m anra_v5.planthir init --backend <id> --out spec.json` | Generate starter `anra-planthir-spec-v1` template for any registered backend |
| `python -m anra_v5.planthir validate --spec spec.json --canary` | Fail-closed preflight check (schema, budgets, Gate 2 confounders, cohort transitivity) |
| `python -m anra_v5.planthir explain-error --code <CODE>` | Instant diagnosis and fix instructions for validation/custody error codes |
| `python -m anra_v5.planthir run` / `resume --spec spec.json --work-root <dir>` | Execute or crash-resume under 5 Launch Gates + atomic bundle publication |
| `python -m anra_v5.planthir inspect-run --work-root <dir>` | Read-only status/identity inspection of an existing run directory |
| `python -m anra_v5.planthir verify-bundle --bundle-dir <dir>` | Verify `.sha256` sidecars, `CURRENT.json`, and ZIP archive integrity |
| `python -m anra_v5.planthir replay-bundle --bundle-dir <dir> --output <out.json>` | Extract checkpoints into temp store, replay full multi-step cohort, and check per-case agreement (`--expected-backend`, `--require-reference`, `--worlds-json`) |
| `python -m anra_v5.planthir rescore-archive --archive <zip> --out <dir>` | Verify a compact analysis archive and rebuild `RESULTS.md` + `METRICS.json` in a fresh directory |
| `python -m anra_v5.planthir historical-atlas` | Summarize curated AN-RA historical V6–V8 evidence |

---

## 5. Starter JSON Templates (`spec.json`)

```json
{
  "schema": "anra-planthir-spec-v1",
  "campaign_id": "EXP-X1-RD2-SLOTS-01",
  "program": "X1-RD2",
  "source_commit": "HEAD",
  "backend_id": "x1_rd2",
  "dataset": {"dataset_name": "X1-D2", "num_groups": 4, "eval_world_count": 4, "multihop_world_count": 2, "probe_world_count": 4, "seed_offset": 19},
  "control_arm": {
    "arm_id": "X1P",
    "architecture": "X1-P",
    "model": {"hidden_dim": 16, "num_layers": 1, "num_heads": 2, "pointer_dim": 8, "max_sequence_length": 512},
    "recipe": {"seed": 7, "total_updates": 2, "batch_size": 8, "micro_batch_size": 4, "lr": 0.001, "objective": "pair_eq", "eq_weight": 0.1, "sampling": "uniform", "schedule": "cosine", "precision": "fp32"}
  },
  "treatment_arms": [
    {
      "arm_id": "X1R",
      "architecture": "X1-R",
      "model": {"hidden_dim": 16, "num_layers": 1, "num_heads": 2, "pointer_dim": 8, "num_slots": 2, "slot_dim": 8, "slot_edge_hidden": 8, "max_sequence_length": 512},
      "recipe": {"seed": 7, "total_updates": 2, "batch_size": 8, "micro_batch_size": 4, "lr": 0.001, "objective": "pair_eq", "eq_weight": 0.1, "sampling": "uniform", "schedule": "cosine", "precision": "fp32"}
    }
  ],
  "execution": {"device": "cpu", "cpu_canary": true, "checkpoint_interval": 1, "parameter_tolerance_ratio": 2.0}
}
```
*(Notice: single-factor contrast & falsifier are auto-inferred here; generate full `v8`, `x1_rd2`, or `x1_s20` templates via `python -m anra_v5.planthir init --backend <id>`.)*

---

## 6. Cross-Branch Copy Fallback (For Any Uninstalled Branch)

- **Bash / Linux / Colab / Kaggle:**
  ```bash
  git fetch origin <source-branch>
  git checkout origin/<source-branch> -- \
    anra_v5/planthir anra_v5/palantiri anra_v5/planter \
    anra_v5/elendril v5_contracts v5_evaluation v5_model v5_training \
    pyproject.toml tools/audit_tree.py docs/TREE_INVENTORY.json \
    agent.md research/planthir tests/test_planthir_*.py
  PYTHONPATH=. python -m anra_v5.planthir doctor
  ```
- **Windows PowerShell:**
  ```powershell
  git fetch origin <source-branch>
  git checkout origin/<source-branch> -- `
    anra_v5/planthir anra_v5/palantiri anra_v5/planter `
    anra_v5/elendril v5_contracts v5_evaluation v5_model v5_training `
    pyproject.toml tools/audit_tree.py docs/TREE_INVENTORY.json `
    agent.md research/planthir tests/test_planthir_*.py
  $env:PYTHONPATH = "."
  python -m anra_v5.planthir doctor
  ```
