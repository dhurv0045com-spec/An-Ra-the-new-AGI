# Planthir / Palantiri (`anra_v5.planthir` / `anra_v5.palantiri` / `anra_v5.planter`) — Agent Guide

> **Canonical Manual:** Guide for any AI agent installing, verifying, or running controlled experiments with the **Planthir** experiment system (aliases: `planthir`, `palantiri`, `planter`; Python modules `anra_v5.planthir`, `anra_v5.palantiri`, `anra_v5.planter`). Follow root [AGENTS.md](AGENTS.md) at all times. Read [the reliability & upgrade handoff](research/planthir/GEMINI_RELIABILITY_AND_UPGRADE_HANDOFF.md) and [the 6-task productivity protocol](research/planthir/SIX_TASK_PRODUCTIVITY_STUDY_PROTOCOL.md) when planning infrastructure changes or productivity studies.

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

Minimal `experiments/<name>.py` recipe (**4 hooks + `ExperimentModule`**):

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
# 4. kernel_factory() -> EnvironmentKernel (optional counterfactual hooks: structural_signature, inject_distractor, permute_world, reverse_world)
#    policy_factory(model, model_spec, *, model_identity: str, mode: str = "raw", copy_enabled_override: bool | None = None) -> ObservablePolicy

module = ExperimentModule(
    backend_id="my_exp",
    architectures=("baseline", "treatment"),
    objectives=("ce", "custom_loss"),
    model_factory=build_my_model,
    dataset_factory=build_my_dataset,
    loss_fn=my_loss_fn,
    kernel_factory=MyKernel,
    policy_factory=MyPolicy,
    extra_model_keys=("my_width",),
    extra_dataset_keys=("num_samples",),
).register()
```

- Subprocess CLIs auto-load your plugin without touching `backend.py` by setting `PLANTHIR_BACKEND_MODULES=experiments.<name>` (provided your module calls `.register()` or defines `register_planthir_backends()`), and `python -m anra_v5.planthir init --backend my_exp --out spec.json` works out of the box!
- For multi-seed / 2xT4 Kaggle campaigns, wrap your wave execution with `CampaignOrchestrator` (`prepare` -> `admit` -> `run_waves`), package `<PREFIX>_ANALYSIS.zip` + `<PREFIX>_FULL.zip` with `export_session_archives`, and emit your pinned notebook via `build_pinned_kaggle_notebook`.
- Use `audit_sampled_strata(records_or_worlds, kernel=..., required_operations=("ANSWER", "QUERY"))` to verify sampled/truncated datasets don't starve any action class.

---

## 2. Fast Test Budget & Anti-Slowdown Rules

- **Do NOT spawn 6+ cold `subprocess` Python processes inside `pytest`** unless explicitly testing OS crash isolation (`_exit` recovery). Multi-worker scheduler/campaign tests must use `IndependentWorkerScheduler(..., in_process_runner=...)` or `CampaignOrchestrator(..., in_process_runner=...)` so suites complete in **`<15s`** instead of `70s+`.
- **Iterate with targeted tests first (`pytest tests/test_<new>.py -q`)**; run the full repo suite once before final handoff/push.
- **CPU canary budget:** Keep CPU canary runs at `<= 100,000` parameters, `<= 32` samples, and `<= 8` updates.

---

## 3. Gotchas & Exact Core Contracts (Avoid Reading 20 Files)

| Component | Exact Contract / Gotcha |
|---|---|
| **`v5_model.core.ModelSpec` & `Core`** | Use keyword **`vocabulary`** (NOT `vocab_size`). `Core` has NO `.logits()` method — compute `positions, mask_attn = packed_layout(segments, torch_module=torch)` and call `logits = core(tokens, positions, mask_attn)`. |
| **`v5_training.state.IdentityBindings`** | Field name is **`optimizer_spec_sha256`** (NOT `optimizer_group_sha256`). |
| **`IndependentWorkerScheduler` GPU Pinning** | When `WorkerJobSpec(device="cuda:1")` runs, the scheduler sets `CUDA_VISIBLE_DEVICES="1"` in the child AND automatically rewrites `--device cuda:1` -> `--device cuda:0` (`remap_child_cuda_device=True`). Child processes should read `PLANTHIR_VISIBLE_CUDA_DEVICE` (`cuda:0`) and never manually index `cuda:1` when `CUDA_VISIBLE_DEVICES` is pinned. |
| **Planthir CLI Bundle Flags** | Always use **`--bundle-dir`** (`verify-bundle --bundle-dir ...`, `replay-bundle --bundle-dir ...`), never `--dir`. |
| **Repository Audit Rules (`AGENTS.md` / `audit_tree.py`)** | Never commit `.zip`, `.npz`, `.bin`, `.pt`, or raw `runs/` folders. Keep curated JSON/Markdown only under `Experiment Result/<Program>/`. Every `progress.md` entry must be **`<= 5` non-empty lines** including its heading (`YYYY-MM-DD \| Agent \| Title`). |

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
| `python -m anra_v5.planthir replay-bundle --bundle-dir <dir> --output <out.json>` | Extract checkpoints into temp store, replay predictions, and check case-by-case agreement |
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
