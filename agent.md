# Planthir (`anra_v5.planthir`) — Agent Operating & Installation Guide

> **Purpose:** Copy-paste manual for any AI agent installing, verifying, or running controlled experiments with **Planthir** across any AN-RA branch. Follow root [AGENTS.md](AGENTS.md) at all times. Read [the reliability & upgrade handoff](research/planthir/GEMINI_RELIABILITY_AND_UPGRADE_HANDOFF.md) and [the 6-task productivity protocol](research/planthir/SIX_TASK_PRODUCTIVITY_STUDY_PROTOCOL.md) when planning infrastructure changes or productivity measurements.

---

## 1. Quick Install & Cross-Branch Setup

Planthir is a Python package at `anra_v5/planthir/` (with CLI entry point `python -m anra_v5.planthir` and console script `planthir` in `pyproject.toml`). Because `anra_v5/planthir/` binds directly into AN-RA's `anra_v5/elendril/` and `v5_*` runtime modules (including the `enforce_tool_consumption=True` evaluator fix), choose **Method A** for new experiments or **Method B** to copy Planthir cleanly into an existing branch.

### Method A — Branch Directly from `planthir` (Recommended)

```bash
git fetch origin planthir
git checkout -b exp/my-new-experiment origin/planthir
pip install -e . --no-build-isolation --no-deps   # optional if using PYTHONPATH=.
python -m anra_v5.planthir doctor
```

### Method B — Copy Planthir into Another Existing Branch

Run from your target branch to pull Planthir, its shared runtime dependencies, docs, and tests cleanly:

- **Bash / Linux / Colab / Kaggle:**
  ```bash
  git fetch origin planthir
  git checkout origin/planthir -- \
    anra_v5/planthir \
    anra_v5/elendril \
    v5_contracts v5_evaluation v5_model v5_training \
    pyproject.toml agent.md research/planthir \
    tests/test_planthir_*.py
  pip install -e . --no-build-isolation --no-deps
  python -m anra_v5.planthir doctor
  ```

- **Windows PowerShell:**
  ```powershell
  git fetch origin planthir
  git checkout origin/planthir -- `
    anra_v5/planthir `
    anra_v5/elendril `
    v5_contracts v5_evaluation v5_model v5_training `
    pyproject.toml agent.md research/planthir `
    tests/test_planthir_*.py
  $env:PYTHONPATH = "."
  python -m anra_v5.planthir doctor
  ```

> **Green Signal Gate:** Stop immediately if `python -m anra_v5.planthir doctor` exits `1` or reports `"green_signal": false`. On Windows/CPU setups without editable install, prefix Python commands with `PYTHONPATH=.` (Bash) or `$env:PYTHONPATH="."` (PowerShell).

---

## 2. The 6-Step Agent-First Declarative CLI Workflow

Every CLI subcommand emits strict JSON to `stdout` (`0` = verified/valid, `1` = validation/custody/replay/runtime error). **Do not hand-write training loops, evaluators, paired statistics, or ZIP packers.** Express only your **experimental delta** in a data-only `anra-planthir-spec-v1` JSON file.

| Step | Exact CLI Command | What It Verifies / Produces |
| :--- | :--- | :--- |
| **1. Preflight & Schema** | `python -m anra_v5.planthir doctor`<br>`python -m anra_v5.planthir schema`<br>`python -m anra_v5.planthir agent-info` | Read-only check of bindings, V8 tool-consumption firewall, starter templates, git clean status, RSS/disk probes, and allowed spec keys/backends. |
| **2. Fault Calibration** | `python -m anra_v5.planthir calibrate` | Runs the 6-fault synthetic observatory benchmark (`all_detected: true`, `6/6` detected, `healthy_v8_kernel_passed: true`). |
| **3. Generate & Edit Spec** | `python -m anra_v5.planthir init --backend x1_rd2 --out spec.json`<br>*(or `--backend v8`)* | Writes a runnable data-only `anra-planthir-spec-v1` starter spec. Unknown keys, `eval`, dotted Python import paths, and shell fragments are rejected. |
| **4. Fail-Closed Validate** | `python -m anra_v5.planthir validate --spec spec.json --canary`<br>`python -m anra_v5.planthir explain-error --code <CODE>` | Verifies closed schema, parameter budgets, Gate 2 Confounder Firewall, typed factors/contrasts, and pairwise disjoint splits **before** training. |
| **5. Run / Inspect / Resume** | `python -m anra_v5.planthir run --spec spec.json --work-root runs/exp1 --canary`<br>`python -m anra_v5.planthir inspect-run --work-root runs/exp1`<br>`python -m anra_v5.planthir resume --spec spec.json --work-root runs/exp1 --canary` | Trains under 5 Launch Gates + `ResourceMonitor` (`RESOURCE_RECEIPT.json`), saves SHA-256 `CheckpointStore` states, atomically publishes `generations/<gen_id>/` + `CURRENT.json`, and exports `result.zip` (`<= 20 MiB`) + `checkpoint_full.zip`. |
| **6. Custody, Replay & Handoff** | `python -m anra_v5.planthir verify-bundle --bundle-dir runs/exp1/result_bundle`<br>`python -m anra_v5.planthir replay-bundle --bundle-dir runs/exp1/result_bundle --output runs/exp1/REPLAY_RECEIPT.json` | Verifies `.sha256` sidecars + `CURRENT.json` + archive manifests, extracts checkpoints into an isolated temp store to independently replay predictions, and emits `~6 KB` `AI_HANDOFF_BRIEF.json`. |

> **CUDA Production Rule:** When setting `"device": "cuda"` (with `--no-canary`), **commit all tracked changes first** (`git status --porcelain --untracked-files=no` must be empty) and set `"source_commit": "HEAD"` (or the exact 40-char `git rev-parse HEAD` SHA). `CartridgeRunner` refuses CUDA runs on dirty or mismatched working trees.
>
> **Deprecated CLI Commands:** `run-validation-study` and `run-gpu-validation` intentionally exit `1` on the CLI with migration JSON. Always use `init` -> `validate` -> `run` / `resume`.

---

## 3. Ready-to-Use `anra-planthir-spec-v1` Templates

### Template A — `x1_rd2` Relational Architecture / Grouped-Loss Contrast

Supports architectures `"X1-P"`, `"X1-R"`, `"X1-R-noedge"`, `"X1-P-capR"` and objectives `"ce"`, `"pair_ce"`, `"pair_eq"`, `"pair_eq_cf"`.

```json
{
  "schema": "anra-planthir-spec-v1",
  "campaign_id": "EXP-X1-SLOTS-01",
  "program": "X1-RD2",
  "source_commit": "HEAD",
  "backend_id": "x1_rd2",
  "dataset": {
    "dataset_name": "X1-D2",
    "num_groups": 4,
    "eval_world_count": 4,
    "multihop_world_count": 2,
    "probe_world_count": 4,
    "seed_offset": 19
  },
  "control_arm": {
    "arm_id": "X1P",
    "architecture": "X1-P",
    "model": {"hidden_dim": 16, "num_layers": 1, "num_heads": 2, "pointer_dim": 8, "max_sequence_length": 512},
    "recipe": {"seed": 7, "total_updates": 4, "batch_size": 8, "micro_batch_size": 4, "lr": 0.001, "objective": "pair_eq_cf", "eq_weight": 0.1, "cf_weight": 0.1, "cf_margin": 0.5, "sampling": "uniform", "schedule": "cosine", "precision": "fp32"},
    "milestone_updates": [2, 4],
    "order_seed": 7
  },
  "treatment_arms": [
    {
      "arm_id": "X1R",
      "architecture": "X1-R",
      "model": {"hidden_dim": 16, "num_layers": 1, "num_heads": 2, "pointer_dim": 8, "num_slots": 2, "slot_dim": 8, "slot_edge_hidden": 8, "max_sequence_length": 512},
      "recipe": {"seed": 7, "total_updates": 4, "batch_size": 8, "micro_batch_size": 4, "lr": 0.001, "objective": "pair_eq_cf", "eq_weight": 0.1, "cf_weight": 0.1, "cf_margin": 0.5, "sampling": "uniform", "schedule": "cosine", "precision": "fp32"},
      "milestone_updates": [2, 4],
      "order_seed": 7,
      "contrasts": [
        {
          "contrast_id": "C_R_VS_P",
          "factor_name": "architecture",
          "control_arm": "X1P",
          "treatment_arm": "X1R",
          "held_constant": ["lr", "objective", "precision", "order_seed"],
          "expected_mechanism_signature": "x1_relational_slot_readout"
        }
      ],
      "hypotheses": [
        {
          "hypothesis_id": "H_X1_REL_SLOTS",
          "statement": "Relational slot readout in X1-R shifts logits under surgical r_to_ptr ablation.",
          "mechanism_target": "x1_relational_slot_readout",
          "falsifiers": [
            {
              "falsifier_id": "F_X1_FINITE",
              "description": "Overall accuracy remains well-defined and finite",
              "metric_name": "overall_accuracy",
              "comparator": ">=",
              "threshold": 0.0
            }
          ]
        }
      ]
    }
  ],
  "factors": [
    {"factor_name": "architecture", "factor_type": "architecture", "control_value": "X1-P", "levels": ["X1-P", "X1-R"]}
  ],
  "typed_contrasts": [
    {"contrast_id": "C_R_VS_P", "factor_name": "architecture", "control_arm_id": "X1P", "treatment_arm_id": "X1R", "target_mechanism": "x1_relational_slot_readout", "expected_treatment_value": "X1-R"}
  ],
  "execution": {
    "device": "cpu",
    "cpu_canary": true,
    "checkpoint_interval": 2,
    "parameter_tolerance_ratio": 2.0,
    "max_wall_seconds": 120.0
  }
}
```

### Template B — `v8` Architecture / Head Contrast (e.g., `J` vs `P` Pointer or `U` Recurrent)

Supports architectures `"B0"`, `"J"`, `"F"`, `"P"`, `"F-cap"`, `"U"` and objective `"ce"`. Use `"sample_limit_for_cpu": null` on CUDA GPU runs (`"cpu_canary": false`).

```json
{
  "schema": "anra-planthir-spec-v1",
  "campaign_id": "EXP-V8-POINTER-01",
  "program": "Elendril-V8",
  "source_commit": "HEAD",
  "backend_id": "v8",
  "dataset": {
    "dataset_name": "D1",
    "d1_worlds_per_family": 16,
    "sample_limit_for_cpu": 8,
    "eval_world_count": 4,
    "multihop_world_count": 2,
    "probe_world_count": 4,
    "seed_offset": 7
  },
  "control_arm": {
    "arm_id": "J_CTRL",
    "architecture": "J",
    "model": {"width": 32, "layers": 1, "query_heads": 2, "kv_heads": 1, "head_dimension": 16, "ffn_width": 64, "context_length": 512},
    "recipe": {"seed": 7, "total_updates": 2, "batch_size": 4, "micro_batch_size": 2, "lr": 0.0001, "objective": "ce", "sampling": "uniform", "schedule": "cosine", "precision": "fp32"},
    "milestone_updates": [1, 2],
    "order_seed": 7
  },
  "treatment_arms": [
    {
      "arm_id": "P_TRT",
      "architecture": "P",
      "model": {"width": 32, "layers": 1, "query_heads": 2, "kv_heads": 1, "head_dimension": 16, "ffn_width": 64, "context_length": 512, "pointer_dim": 16, "copy_enabled": true},
      "recipe": {"seed": 7, "total_updates": 2, "batch_size": 4, "micro_batch_size": 2, "lr": 0.0001, "objective": "ce", "sampling": "uniform", "schedule": "cosine", "precision": "fp32"},
      "milestone_updates": [1, 2],
      "order_seed": 7,
      "contrasts": [
        {
          "contrast_id": "C_P_VS_J",
          "factor_name": "architecture",
          "control_arm": "J_CTRL",
          "treatment_arm": "P_TRT",
          "held_constant": ["lr", "precision", "order_seed"],
          "expected_mechanism_signature": "pointer_copy_gate"
        }
      ]
    }
  ],
  "execution": {
    "device": "cpu",
    "cpu_canary": true,
    "checkpoint_interval": 1,
    "parameter_tolerance_ratio": 1.25
  }
}
```

---

## 4. Advanced Building Blocks (Python APIs)

All symbols below are exported directly from `anra_v5.planthir`:

1. **Multi-Seed Worker Scheduling (`IndependentWorkerScheduler`):**
   Launch isolated subprocess workers (distinct `work_root`, OS lockfile `.planthir_worker.lock`, disjoint CUDA devices, partial-failure containment, `COORDINATOR_RECEIPT.json`):
   ```python
   from pathlib import Path
   from anra_v5.planthir import IndependentWorkerScheduler, WorkerJobSpec

   receipt = IndependentWorkerScheduler(coordinator_root=Path("runs/coord"), max_parallel=2).run([
       WorkerJobSpec("seed7", Path("runs/seed7"), spec_path="spec_s7.json", device="cuda:0", canary=False),
       WorkerJobSpec("seed19", Path("runs/seed19"), spec_path="spec_s19.json", device="cuda:1", canary=False),
   ])
   ```
2. **Paired-Seed Statistics, Cohort Transitivity & Teacher Firewall (`design.py`, `instrument.py`):**
   - `validate_cohort_and_split_transitivity(splits, cohorts=...)`: Union-Find transitive closure across `world_id`, `base_world_id`, `parent_world_id`, `counterfactual_mate_id`, `group_id`, `cohort_id`.
   - `analyze_paired_seed_runs(observations)`: paired differences over independent `(init_seed, order_seed)` pairs with Student-t/bootstrap CIs and Holm-Bonferroni correction across contrasts (rejects missing pairs, duplicate seeds, 1-cluster cohorts, unequal sizes).
   - `validate_nonanticipating_teacher(kernel, world_pairs)`: catches teachers that peek at hidden state when visible history and tool availability match.
3. **Chunked Streaming Raw Evidence & Capsule Replay (`evidence.py`, `replay.py`):**
   - `stream_raw_evidence_chunks(out_dir, records_iter, max_chunk_records=64, max_total_bytes=...)` writes SHA-256-indexed `.jsonl`/`.npz` chunks enforcing write-time byte quotas; pass `raw_evidence_link={"raw_manifest_sha256": ...}` into `write_tier2_diagnostic_capsule` to bind lossy `float16` capsules to exact raw provenance.
   - `build_custody_receipt` + `independent_replay_bundle` + `probe_from_capsule` allow offline probing and independent checkpoint verification without retraining.
4. **Adding a New Architecture or Backend Beyond `v8` / `x1_rd2` (`backend.py`):**
   Implement the `PlanthirBackend` protocol (`BACKEND_CONTRACT_VERSION = "anra-planthir-backend-v1"`) alongside `EnvironmentKernel` + `ObservablePolicy` (`anra_v5/planthir/contracts.py`), then call `register_backend(MyBackend())`. Qualified backends must reject `custom trainer_factory` hooks and emit canonical parameter counts and factor configs. See [anra_v5/planthir/adapters/x1_rd2_adapter.py](anra_v5/planthir/adapters/x1_rd2_adapter.py) as the reference non-V8 implementation.

---

## 5. Strict Boundaries & Repository Discipline

- **CPU canary limits:** CPU runs enforce `<= 100,000` parameters, `<= 32` samples, and `<= 8` updates. Never launch heavy training in local pytest or CPU shells.
- **Keep binaries out of Git:** Never commit `.zip`, `.npz`, `.bin`, `.pt`, or raw `runs/` folders (`AGENTS.md` / `tests/test_elendril_tree.py`). Put only focused Markdown reports and curated JSON metrics/provenance under `Experiment Result/<Program>/`.
- **Progress register:** Append a concise `<= 5`-line dated entry to root [progress.md](progress.md) after every meaningful change, upgrade, or execution.
- **Honest qualification claims:** Always distinguish implemented controls, CPU canary checks, live CUDA runs (`cuda_measured: true`), and empirical human/agent productivity trials (`anra_v5.planthir.productivity` reports `STUDY_OBSERVATIONS_PENDING` until logged counterbalanced runs are supplied).
- **Pre-handoff verification commands:**
  - `python -m anra_v5.planthir doctor`
  - `python -m anra_v5.planthir calibrate`
  - `python -m pytest tests/test_planthir_*.py -q` (Bash) or `python -m pytest (Get-ChildItem tests/test_planthir_*.py) -q` (PowerShell)
  - `python -m tools.verify_elendril_evidence`
