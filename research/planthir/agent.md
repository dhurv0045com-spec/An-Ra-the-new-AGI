# `agent.md` — How AI Agents Must Use Planthir (`anra_v5.planthir`)

> **Scope:** Instructions for any AI coding or research agent running experiments, evaluating architectures, or auditing results in AN-RA.
> **Important:** Always follow repository rules in root `AGENTS.md` as well (including appending `<= 5`-line dated entries to `progress.md` and keeping binary `.zip`/`.npz`/`.bin` files out of Git).

---

## 1. Where Planthir Lives & How to Access It Across Branches

Planthir is a standard importable Python package at **`anra_v5/planthir/`** (also registered as CLI script `planthir` in `pyproject.toml`). You do **not** copy/paste infrastructure code into notebooks or ad-hoc folders.

- **Case A — Starting a new experiment branch (Recommended):**
  Branch from `planthir` (or merge `planthir` into your main branch once). Then `anra_v5/planthir/` is automatically present:
  ```bash
  git checkout -b exp-v9-new-idea planthir
  ```
  In Python inside the repo, just import it by name:
  ```python
  import anra_v5.planthir as planthir
  from anra_v5.planthir import CartridgeArmSpec, CartridgeRunner, ExperimentCartridge
  ```

- **Case B — Pulling Planthir into an older or separate branch:**
  Bring only the Planthir package, docs, and test suite into your current branch without switching branches:
  ```bash
  git checkout planthir -- anra_v5/planthir research/planthir tests/test_planthir_universal_infrastructure.py purpose.md agent.md
  ```
  *(If auditing or running V8 studies that rely on the consumed-tool fix, also verify `anra_v5/elendril/v8/evaluation.py` carries `enforce_tool_consumption=True`.)*

- **Case C — Local Python Environments (Windows Workspace Note):**
  - Default CPU environment (`python`) runs `pytest` and CPU canaries (`python -m pytest ...`, `python -m anra_v5.planthir calibrate`).
  - Local GPU environment (`.venv-cuda`, `torch` + CUDA): set `$env:PYTHONPATH="."` in PowerShell (or `PYTHONPATH=.` in Bash) or install editable once (`pip install -e . --no-build-isolation --no-deps`):
    ```powershell
    $env:PYTHONPATH="."
    python -m anra_v5.planthir run-gpu-validation --output-dir "Experiment Result/Planthir" --updates 60
    ```

---

## 2. The Golden Rule for Agents

**Never build training loops, evaluators, paired-bootstrap calculators, Linear/Subgoal probes, or ZIP packers from scratch.**

Only specify the **experimental delta** (~40–60 lines):
1. Which **control arm** and **treatment arm(s)** you want to compare (`CartridgeArmSpec`).
2. Your preregistered **`HypothesisSpec`** + **`FalsifierSpec`** + **`CausalContrastSpec`**.
3. Pass them to **`ExperimentCartridge`** -> **`CartridgeRunner().run(...)`**.

Planthir's **5 Launch Gates** then automatically verify the environment, block confounded designs, verify gradients, train with SHA-256 milestone checkpoints, run the 4-stage process funnel + 6 causal/ablation counterfactuals, enforce the epistemic firewall, export `result.zip` (`<= 20 MB`) + `checkpoint_full.zip`, reload from `result.zip`, prove `1.000` live-vs-restored prediction parity, and output a `~6 KB` `AI_HANDOFF_BRIEF.json`.

---

## 3. Copy-Paste Templates

### Template A: Controlled Mechanism / Hyperparameter Experiment (`< 60` Lines)

Use this whenever comparing an existing or new V8 configuration against a single-variable control (e.g., testing a different `recurrent_steps`, `pointer_dim`, `lr`, or custom `trainer_factory`):

```python
from dataclasses import replace
from pathlib import Path
import subprocess
from anra_v5.elendril.foundation_training import Recipe
from anra_v5.elendril.v6_registry import MODEL_PROFILES
from anra_v5.elendril.v8.contracts import build_v8_model_spec
from anra_v5.elendril.v8.data import D1_TRAIN_SEED, build_d1_training_dataset
from anra_v5.elendril.v8.holdout import canonical_graph_signature
from anra_v5.elendril.v8.training import prepare_v8_training_data
from anra_v5.planthir import (
    CartridgeArmSpec, CartridgeRunner, CausalContrastSpec,
    ExperimentCartridge, FalsifierSpec, HypothesisSpec, V8EnvironmentKernel,
)

# 1. Data & strictly disjoint evaluation worlds
train_data = prepare_v8_training_data("D1", d1_worlds_per_family=32, sample_limit_for_cpu=None)
train_sigs = {canonical_graph_signature(w)["canonical_hash"] for w in build_d1_training_dataset(quota_per_family=32).worlds}
eval_pool = [w for w in build_d1_training_dataset(seed=D1_TRAIN_SEED + 101, quota_per_family=32).worlds
             if canonical_graph_signature(w)["canonical_hash"] not in train_sigs]
eval_worlds = tuple(eval_pool[:: max(1, len(eval_pool) // 24)][:24])
multihop_worlds = tuple(w for w in eval_pool if int(w.get("hops", 1)) >= 2)[:12]
probe_train_worlds = tuple(build_d1_training_dataset(quota_per_family=16).worlds[:32])

# 2. Shared base spec & single-variable Recipe (Gate 2 rejects mismatched recipes!)
base_spec = replace(MODEL_PROFILES["S20"], width=64, layers=2, query_heads=4, kv_heads=2, head_dimension=16, ffn_width=128, context_length=512)
recipe = Recipe(seed=7, total_updates=60, batch_size=16, micro_batch_size=8, lr=3e-3, schedule="cosine", precision="fp32")

control = CartridgeArmSpec(
    arm_id="J_ctrl", architecture_family="J", is_control=True,
    v8_spec=build_v8_model_spec("J", base_spec=base_spec),
    recipe=recipe, dataset_identity=train_data.data_identity, milestone_updates=(20, 40, 60),
)
treatment = CartridgeArmSpec(
    arm_id="U_treat", architecture_family="U", is_control=False,
    v8_spec=build_v8_model_spec("U", base_spec=base_spec, recurrent_dim=16, recurrent_steps=2),
    recipe=recipe, dataset_identity=train_data.data_identity, milestone_updates=(20, 40, 60),
    hypotheses=(HypothesisSpec(
        hypothesis_id="H-U-RECURRENCE", statement="Recurrent refinement U reduces composition gap vs baseline J.",
        mechanism_target="recurrent_refinement",
        falsifiers=(FalsifierSpec("F-STAGE1", "Positive tool recall", "stage1_tool_recall_when_needed", ">=", 0.0),),
    ),),
    contrasts=(CausalContrastSpec(
        contrast_id="C-U-VS-J", factor_name="architecture", control_arm="J_ctrl", treatment_arm="U_treat",
        held_constant=("base_spec_64x2", "D1_d32", "seed_7", "lr_3e-3", "param_tolerance"),
        expected_mechanism_signature="compute_depth_gain_or_stage4_lift",
    ),),
)

# 3. Execute all 5 Launch Gates + export result.zip (<=20 MB) + checkpoint_full.zip + AI_HANDOFF_BRIEF.json
source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
cartridge = ExperimentCartridge(
    campaign_id="EXP-NEW-001", program="V8-Mechanism-Test", source_commit=source_commit,
    kernel=V8EnvironmentKernel(enforce_tool_consumption=True), control_arm=control, treatment_arms=(treatment,),
    train_data=train_data, eval_worlds=eval_worlds, multihop_worlds=multihop_worlds,
    probe_train_worlds=probe_train_worlds, parameter_tolerance_ratio=2.0,
)
result = CartridgeRunner().run(cartridge, work_root=Path("./tmp_run"), bundle_out_dir=Path("./tmp_bundle"), device="cuda:0", cpu_canary=False)
```

> **Important Gate 2 rule:** Keep `"param"` / `"capacity"` / `"matched"` in `contrasts[*].held_constant` whenever testing parameter-matched comparisons so `audit_experimental_design` enforces `parameter_tolerance_ratio`, and make sure `source_commit` is an actual 40-char hex git SHA (`git rev-parse HEAD`).

---

### Template B: Onboarding a Brand-New Architecture or Task (V9 / Memory / Continual Learning)

When introducing an architecture or environment completely outside `V8`, implement Planthir's two lightweight protocols (`anra_v5/planthir/contracts.py`):

```python
import torch
from anra_v5.planthir.contracts import UniversalAction, UniversalObservation
from anra_v5.planthir.instrument import validate_environment_kernel
from anra_v5.planthir.observatory import ProcessFunnelAnalyzer
from anra_v5.planthir.causal import run_representation_linear_probe
from anra_v5.planthir.interpretation import synthesize_scientific_interpretation

class MyNewKernel:
    kernel_id = "my-v9-task-v1"
    def initial_state(self, world): ...
    def observe(self, state) -> UniversalObservation:
        obs = UniversalObservation(
            obs_id=..., rendered_text=..., available_tools={"query": ..., "retrieve": ...},
            consumed_tools=frozenset(...), decision_index=..., budget_remaining=...,
        )
        assert not obs.validate_invariants(), obs.validate_invariants()
        return obs
    def ideal_action(self, state) -> tuple[str, str | None]: ...
    def step(self, state, operation: str, argument: str | None): ...
    def verify(self, world, actions) -> dict: return {"correct": ..., "terminal_status": ...}
    def decompose_subgoals(self, world) -> tuple: return ()

class MyNewPolicy:
    policy_id = "my-v9-arch"
    def propose_with_telemetry(self, obs: UniversalObservation) -> UniversalAction:
        return UniversalAction(
            operation=..., argument=..., log_probability=..., action_distribution=...,
            mode="raw", raw_operation=..., raw_argument=...,
            telemetry={"custom_gate_mean": ...},
        )
    def extract_representations(self, obs_batch) -> dict[str, torch.Tensor]:
        return {"decision_hidden": hidden_tensor}  # shape [N, D]

# Immediately works with validate_environment_kernel, ProcessFunnelAnalyzer, linear probes, & interpretation:
inst = validate_environment_kernel(MyNewKernel(), worlds[:8])
funnel = ProcessFunnelAnalyzer().evaluate_suite(MyNewPolicy(), MyNewKernel(), worlds)
probe = run_representation_linear_probe(MyNewPolicy(), MyNewKernel(), train_worlds, worlds)
report = synthesize_scientific_interpretation(
    experiment_id="V9-EXP-001", instrument_validation=inst,
    process_funnel=funnel, linear_probe=probe,
)
```

*(Note: See `purpose.md` Section 4 item #6 regarding custom `CartridgeRunner` integration for non-V8 trainers using `trainer_factory`.)*

---

### Template C: Zero-Retraining Causal Probing from `result.zip` or `checkpoint_full.zip`

Never retrain just to test a new hypothesis, ablation, or linear probe on an already-run model:

```python
from anra_v5.planthir import (
    CheckpointCatalog, V8EnvironmentKernel,
    run_surgical_mechanism_ablation, run_compute_depth_scaling_intervention, probe_from_capsule,
)

# 1. Reload frozen weights directly from result.zip (handles full_embedded OR compact_fp16 automatically)
loaded = CheckpointCatalog.load_from_result_zip("path/to/result.zip", experiment_id="EXP-PLANTHIR-004-U", device="cpu")
policy = loaded["policy"]
kernel = V8EnvironmentKernel(enforce_tool_consumption=True)
print("Surgical ablation:", run_surgical_mechanism_ablation(policy, kernel, eval_worlds))
print("Depth scaling:", run_compute_depth_scaling_intervention(policy, kernel, eval_worlds))

# 2. Or run offline linear probes purely from Tier-2 capsules (no weights loaded at all):
offline = probe_from_capsule("path/to/evidence_bundles/U/tier2_capsule")

# 3. Or restore the full SHA-256 CheckpointStore from checkpoint_full.zip:
restored = CheckpointCatalog.restore_full_checkpoint_zip("path/to/checkpoint_full.zip", "./restored_store")
```

---

## 4. Mandatory 4-Step Workflow for Every Agent

| Step | Action | Exact Command / Check | Pass Criteria |
| :--- | :--- | :--- | :--- |
| **1. Calibrate** | Run instrument fault calibration before running research | `python -m anra_v5.planthir calibrate` | `all_detected: true` (`6/6`) and `healthy_v8_kernel_passed: true` |
| **2. Canary / Bounded Run** | Verify gradient flow, Gate 2 confounder purity, & runtime on CPU or local GPU (`.venv-cuda`, stay well below system overload) | `python -m anra_v5.planthir run-gpu-validation --output-dir "Experiment Result/Planthir" --updates 60` (or `CartridgeRunner().run(...)`) | Exit code `0`, reasonable VRAM/time, finite loss/grads |
| **3. Inspect Gates & Verify ZIPs** | Confirm all 5 Launch Gates passed and SHA-256 archives verify | Inspect `LAUNCH_GATE_CERTIFICATE.json` and run `python -m anra_v5.planthir verify-bundle --dir <bundle_dir>` (or `--zip <result.zip>`) | `all_5_gates_passed: true`, `verified: true`, `result.zip <= 20 MiB` (`20,971,520` B), Gate 5 `prediction_parity_rate == 1.0` |
| **4. Minimal-Token Handoff** | Read ONLY `AI_HANDOFF_BRIEF.json` (~6 KB) to inspect metrics, confirmed vs. suspected claims, and ranked next steps before scaling to Kaggle `2xT4` | Open `AI_HANDOFF_BRIEF.json` + `RESULT_OVERVIEW.md` + `tier1_summary/SCIENTIFIC_SUMMARY.md` | Never dump multi-MB raw JSON/traces into prompt; keep `.zip`/`.npz`/`.bin` out of Git and append `<= 5`-line entry to `progress.md` |

---

## 5. Pre-Flight Checklist Before Launching Expensive Kaggle / Colab Compute

- [ ] `python -m anra_v5.planthir calibrate` exited `0` after my final edit.
- [ ] `git status` shows clean commits matching `source_commit` (40-char SHA).
- [ ] Gate 1 & Gate 2 passed with **strictly disjoint** eval worlds (no canonical graph overlap with `train_data`) and matched seeds/LR/updates/param budget.
- [ ] Milestone checkpoints were recorded so trajectory analysis (`representation_leads_readout_detected`, `effective_rank`) can run across time.
- [ ] `result.zip` (`<= 20 MB`) and `checkpoint_full.zip` passed `verify-bundle` and live-vs-restored parity (`1.000`).
- [ ] All relevant tests pass:
  `python -m pytest tests/test_planthir_universal_infrastructure.py tests/test_elendril_v8_contracts_and_data.py tests/test_elendril_v8_training_and_recovery.py tests/test_elendril_v8_campaign_and_export.py tests/test_elendril_tree.py tests/test_elendril_result_curation.py -q`
