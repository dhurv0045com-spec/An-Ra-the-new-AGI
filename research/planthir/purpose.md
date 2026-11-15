# Planthir (`anra_v5/planthir`) — Purpose, Architecture & Adversarial Audit Brief

> **Audience:** Independent AI reviewers and research engineers auditing AN-RA's universal experimental infrastructure (`Planthir` Level 5) for reasoning errors, statistical blind spots, edge cases, or engineering flaws.
>
> **How to use this document:** Read this specification alongside `anra_v5/planthir/` and `tests/test_planthir_universal_infrastructure.py`. Challenge every threshold, mathematical formulation, protocol assumption, and gate invariant below. Do **not** assume correctness merely because existing tests pass.

---

## 1. Why Planthir Exists (Historical Problem Statement)

Across **52 archived primary runs** spanning AN-RA **V6, V7, V7.1, V7.15, V7.16, V7.17** and **V8 (`Elendril`)** (`EXP-PLANTHIR-003`), evaluating models by terminal accuracy alone repeatedly stalled scientific progress due to four structural failures:

1. **Silent Instrument & Evaluator Defects Masquerading as Model Failures:**
   - **V7.16 Aggregator Shadowing (`HIST-DEFECT-V716-SHADOW`):** `tools/elendril_v716_report.py` initialized `final_epoch` as a `(epoch, dict)` tuple and compared `int > tuple` inside a `try/except TypeError`, silently dropping later epochs (`epoch-10` shadowed by earlier epochs).
   - **V8 Consumed-Tool Persistence Leak (`HIST-DEFECT-V8-UNCONSUMED-TOOL`):** In `anra_v5/elendril/v8/evaluation.py::run_episode`, executing `QUERY` or `RETRIEVE` left `query_avail=True` / `mem_avail=True`, trapping models and even a stateless visible-state oracle into repeating already-consumed tools (`6/12 = 50.0%` oracle failure before Planthir's fix `enforce_tool_consumption=True`, vs. `12/12 = 100.0%` after fix in `EXP-PLANTHIR-001`).
   - **Historical Shadow State (`self._last_world`):** Evaluator state leaked across verification vs. observation paths unless strictly separated by a single state machine.
2. **Process-vs-Outcome Decoupling:**
   - Models in V7–V7.17 consistently achieved `0.977–1.000` mean query recall while mean confirmation accuracy stayed `0.573–0.742` and surface-alias robustness dropped to `0.147–0.250`. Scalar accuracy alone could not separate (a) missing internal representations, (b) readout-head/gate under-extraction, (c) multi-hop composition loss, or (d) positional/token-identity shortcuts.
3. **Confounded Comparisons & Ornamental Mechanisms:**
   - Changing architecture, parameter count, LR schedule, dataset slice, or seed simultaneously invalidates causal attribution. Worse, specialized mechanisms (e.g., pointer gates `P`, recurrent refinement `U`, low-rank bottlenecks `F-cap`) can be silently bypassed during training unless surgically ablated post-hoc.
4. **High Token & Bandwidth Overhead ("Rocket Fuel" Waste):**
   - Rebuilding custom training/evaluation/packaging harnesses per experiment consumes high AI authoring token budgets, while dumping multi-gigabyte raw checkpoints or tens of kilobytes of unstructured logs slows iteration and causes context bloat.

**Core Scientific Objective:** Maximize **verified causal knowledge gained per GPU hour** across arbitrary AN-RA architectures, training algorithms, tasks, and parameter scales, while isolating instrument truth from model capability under an explicit epistemic firewall.

---

## 2. System Architecture & Exact File Map (`anra_v5/planthir/`)

| Module | Responsibility | Key Public Symbols |
| :--- | :--- | :--- |
| `contracts.py` | Architecture-agnostic protocols, truth firewall, hypothesis & experiment schemas | `EnvironmentKernel`, `ObservablePolicy`, `UniversalObservation`, `UniversalAction`, `SubgoalProbeSpec`, `ExperimentRegistration`, `HypothesisSpec`, `FalsifierSpec`, `CausalContrastSpec`, `detect_truth_leaks` |
| `adapters/v8_adapter.py` | Unifies V8 (`Elendril`) B0/J/F/P/F-cap/U under Planthir contracts | `V8EnvironmentKernel`, `V8KernelState`, `V8ObservablePolicy` |
| `instrument.py` | Pre-flight calibration, gradient/aggregator audits, 6-fault injection suite | `validate_environment_kernel`, `validate_metric_aggregator`, `validate_model_and_gradients`, `run_fault_injection_suite` |
| `observatory.py` | Internal weight/grad/rep telemetry, 4-stage funnel, milestone trajectory | `InternalTelemetryCollector`, `ProcessFunnelAnalyzer`, `analyze_checkpoint_trajectory`, `compute_effective_rank`, `compute_representation_isotropy`, `DIVERGENCE_TAXONOMY` |
| `causal.py` | Zero-retraining frozen counterfactuals, linear probes, surgical ablations | `run_oracle_information_intervention`, `run_subgoal_hop_decomposition`, `run_invariance_counterfactuals`, `run_representation_linear_probe`, `run_surgical_mechanism_ablation`, `run_compute_depth_scaling_intervention` |
| `interpretation.py` | Epistemic firewall partitioning confirmed, suspected, falsified, unresolved | `synthesize_scientific_interpretation`, `ScientificInterpretationReport`, `ScientificClaim`, `RecommendedExperiment` |
| `evidence.py` | 3-tier evidence packaging, SHA-256 `result.zip` (`<= 20 MB`) & `checkpoint_full.zip` | `write_tier1_summary`, `write_tier2_diagnostic_capsule`, `probe_from_capsule`, `export_result_bundle`, `verify_result_bundle`, `verify_single_planthir_zip` |
| `checkpoint_intelligence.py` | Content-addressed checkpoint discovery, factor diffing, ZIP restoration | `CheckpointCatalog`, `CheckpointRecord` |
| `registry.py` | Experiment registration store + V6–V8 historical atlas synthesizer | `PlanthirRegistry`, `build_historical_atlas` |
| `cartridge.py` | Level-5 self-auditing 5-Gate `ExperimentCartridge` runner & token/handoff brief | `CartridgeArmSpec`, `ExperimentCartridge`, `CartridgeRunner`, `audit_experimental_design`, `build_ai_handoff_brief`, `estimate_token_fuel_savings` |
| `validation_study.py` | Reproducible studies `EXP-PLANTHIR-001/002/003` (CPU) & `004` (live GPU) | `run_exp_001_fault_calibration`, `run_exp_002_causal_validation`, `run_exp_003_historical_synthesis`, `run_exp_004_live_gpu_cartridge_validation` |
| `__main__.py` | CLI: `calibrate`, `historical-atlas`, `run-validation-study`, `run-gpu-validation`, `verify-bundle` | `main` |

---

## 3. Formal Mathematical & Algorithmic Specifications

### 3.1 Universal Protocols & Structural Invariants (`contracts.py`)
- **`UniversalObservation.validate_invariants()`** enforces before execution:
  1. Non-empty `obs_id` and `rendered_text`.
  2. `detect_truth_leaks(rendered_text)`: rejects presence of any marker in `PLANTHIR_LEAK_MARKERS` (e.g., `V5_LEAK_MARKERS` plus `expected=`, `expected:`, `hidden=`, `hidden:`, `ground_truth=`, `ground_truth:`), allowing a single trailing prompt suffix `answer:`.
  3. Strict `int` `decision_index >= 0` and `budget_remaining >= 1`.
  4. **Tool-Consumption Monotonicity:** For every `t in consumed_tools`, `available_tools.get(t, False)` must be `False`.
  5. **Forbidden Metadata Keys:** Rejects `"expected"`, `"gold"`, `"gold_answer"`, `"hidden"`, `"truth"` in `obs.metadata`.

### 3.2 Instrument Calibration & 6-Fault Injection (`instrument.py`)
- **`validate_environment_kernel`**: Runs oracle rollouts (up to 8 steps) verifying:
  - `train_rollout_tri_parity`: oracle rollout `verify()` passes AND matches `static_steps_fn` step count, exact prompt UTF-8 strings, and tool availability flags at every step.
  - `tool_consumption_monotonicity`: previously used `"query"` / `"retrieve"` keys remain `False` on subsequent steps.
  - `actor_truth_firewall`: mutating `world["hidden"]["expected"]` (`"Z"` / `"Y"`) must produce bit-identical `rendered_text` and `available_tools` across the entire rollout.
  - `subgoal_decomposition_validity`: `kernel.ideal_action(initial_state(sg.modified_world))` matches `(sg.expected_operation, sg.expected_argument)` for every emitted `SubgoalProbeSpec`.
  - `v8_evaluator_parity`: Template-0 oracle rollout via `_OracleV8Actor` on `anra_v5.elendril.v8.evaluation.run_episode(..., enforce_tool_consumption=True)`.
- **`validate_model_and_gradients`**: Temporary `model.train()` single forward/backward pass verifying finite loss, finite gradients, non-zero gradient norms, and `active_param_contract` coverage, restoring original `.training` modes and clearing `.grad` cleanly.
- **`validate_metric_aggregator`**: Feeds synthetic `epoch-1`, `epoch-2`, `epoch-4`, `epoch-10` fixtures and asserts `final_epoch == 10` and `final_accuracy == 0.90`.
- **`run_fault_injection_suite()`**: Deliberately injects 6 faults (V8 unconsumed-tool bug, truth-firewall leak, prompt drift, detached `val_head.weight` gradient, V7.16 `ep > final_epoch` tuple shadowing, and SHA-tampered `rng.bin` restore) and requires `6/6` detection + localization.

### 3.3 Representation Geometry & 4-Stage Process Funnel (`observatory.py`)
- **Shannon Effective Rank (`compute_effective_rank`)** on centered `[N, D]` representations $\tilde{H} = H - \bar{H}$:
  $$\sigma = \text{svdvals}(\tilde{H}), \quad p_i = \frac{\sigma_i}{\sum_j \sigma_j} \ (\sigma_i > 10^{-12}), \quad \text{erank}(H) = \exp\!\left(-\sum_i p_i \log p_i\right).$$
- **Representation Isotropy (`compute_representation_isotropy`)**:
  For $L_2$-normalized representations $\hat{h}_i$, $\bar{c} = \frac{1}{N(N-1)}\sum_{i \neq j} \hat{h}_i^\top \hat{h}_j$, $\text{isotropy} = \text{clamp}(1 - |\bar{c}|, 0, 1)$.
- **Confidence Calibration (`_calibration_metrics`)**: Using executed action probability $p = \exp(\text{log\_probability})$ and correctness $y \in \{0, 1\}$:
  - Brier score: $\frac{1}{N}\sum (p_i - y_i)^2$.
  - 10-bin Expected Calibration Error (`ece`).
  - Area Under Risk-Coverage Curve (`aurc`) sorted descending by confidence.
- **4-Stage Process Funnel (`ProcessFunnelAnalyzer.evaluate_suite`)**:
  - **Stage 1 (`stage1_info_seeking`):** Fraction where `initial_needed_tool == initial_called_tool` (plus `stage1_tool_recall_when_needed` and `stage1_unnecessary_tool_rate`).
  - **Stage 2 (`stage2_tool_discipline`):** Among valid initial tool callers, fraction executing $\ge 2$ actions without `REPEATED_CONSUMED_TOOL`.
  - **Stage 3 (`stage3_single_hop_grounding`, `stage3_subgoal_hop1_grounding`):** Accuracy on native `hops <= 1` answerable worlds and decomposed `hop_1` probes.
  - **Stage 4 (`stage4_multi_hop_composition`, `stage4_epistemic_abstention`):** Accuracy on `hops >= 2` answerable worlds and `expected is None` unsolvable worlds.
  - **First-Divergence Taxonomy (`DIVERGENCE_TAXONOMY`):** 9 mutually exclusive first-step divergence labels (`NONE`, `MODE_MASK_OVERRIDE`, `REPEATED_CONSUMED_TOOL`, `MISSED_REQUIRED_TOOL`, `UNNECESSARY_TOOL_CALL`, `PREMATURE_ABSTAIN`, `FAILED_ABSTAIN_ON_UNSOLVABLE`, `HOP1_VALUE_ERROR`, `MULTI_HOP_CHAIN_ERROR`).
- **Checkpoint Trajectory (`analyze_checkpoint_trajectory`)**: Evaluates sorted milestone checkpoints; flags `representation_leads_readout_detected = (max(head_minus_probe_gap) >= 0.10)` and `rank_collapse_detected = (erank_last < 0.70 * erank_first)`.

### 3.4 Frozen-Model Causal & Counterfactual Engine (`causal.py`)
All uncertainty intervals use **`_paired_bootstrap_ci`** (`samples=400`, deterministic seed hashed via `digest(["planthir-bootstrap-v1", seed, len(diffs)])`, 2.5th–97.5th percentile).
1. **`run_oracle_information_intervention`:** Compares baseline rollout accuracy vs. `kernel.inject_oracle_information(world)` (`delta_oracle` and `tool_delta_oracle` with paired 95% bootstrap CIs).
2. **`run_subgoal_hop_decomposition`:** On multi-hop worlds with oracle payloads merged, compares Hop-1 alone (`sg1`) and Hop-2-given-bridge (`sg2`) vs. full chain:
   $$\text{composition\_gap} = P(\text{Hop1} \wedge \text{Hop2}) - P(\text{Full Chain}),$$
   with paired 95% bootstrap CI.
3. **`run_invariance_counterfactuals`:** Tests cyclic shift of `SYMBOLS` (+1) equivariance, row-order reversal invariance, unused 3-symbol distractor chain invariance, and decoder mode overrides (`mode_policies`).
4. **`run_representation_linear_probe`:** Solves closed-form multi-class ridge regression on `decision_hidden` ($[N, D]$ appended with unpenalized bias column $1$, $L_2$ penalty $\lambda=1.0$):
   $$W = (X_a^\top X_a + \operatorname{diag}(\lambda \mathbf{1}_D, 10^{-6}))^{-1} X_a^\top Y_{\text{onehot}}.$$
   Reports `probe_operation_accuracy`, `probe_hop1_symbol_accuracy`, `probe_final_symbol_accuracy`, and `head_minus_probe_gap = probe_final_acc - model_head_acc` (positive = probe outperforms readout head).
5. **`run_surgical_mechanism_ablation`:** Zero-retraining module ablation across V8 specialized mechanisms:
   - `P`: `copy_enabled_override=False` (`pointer_copy_gate`).
   - `U`: `_SpecStepProxy(..., recurrent_steps=0)` (`recurrent_refinement`).
   - `F-cap`: temporary zeroing of `cap_out.weight` and `cap_out.bias` with `finally:` restoration (`value_residual_bottleneck`).
   Flags `ornamental_mechanism_bypass = (ablation_supported and action_agreement_rate >= 0.98 and abs(accuracy_delta) <= 1e-6)`.
6. **`run_compute_depth_scaling_intervention`:** Evaluates `U` across `recurrent_steps` in `sorted(step_grid | {0, trained_steps})` via `_SpecStepProxy` without mutating frozen `V8ModelSpec`.

### 3.5 Epistemic Firewall (`interpretation.py`)
`synthesize_scientific_interpretation` enforces hard stratification:
- If `instrument_validation["passed"]` is `False` -> returns `status="INVALIDATED_BY_INSTRUMENT_DEFECT"`, zero confirmed/suspected/falsified claims, and ranks `REPAIR_INSTRUMENT_DEFECT` #1.
- Confirmed vs. Suspected separation rules currently include:
  - Confirmed: `stage1 - stage4 >= 0.25` (`C-FUNNEL-DROP`), `stage2 < 0.80` (`C-POST-TOOL-DISCIPLINE`), oracle delta `|delta| >= 0.15` with bootstrap 95% CI excluding 0 (`C-TOOL-ACQUISITION-BOTTLENECK`), oracle accuracy `< 0.50` (`C-INCONTEXT-BOTTLENECK`), `composition_gap >= 0.15` with CI excluding 0 (`C-COMPOSITION-GAP`), `surgical_ablation.accuracy_delta >= 0.05` (`C-ACTIVE-MECHANISM-CONTRIBUTION`), `compute_depth_scaling.recurrence_gain_over_zero >= 0.05` (`C-RECURRENT-COMPUTE-GAIN`), and `FalsifierSpec` evaluations.
  - Suspected: directional effects whose CI includes 0, equivariance `< 0.70`, row-reversal `< 0.75`, `head_minus_probe_gap >= 0.15`, `effective_rank < 3.0`, `pointer_gate_mean < 0.15`, `ornamental_mechanism_bypass`, `trajectory_max_probe_lead >= 0.10`.
  - Unresolved: Always appends `U-SCALE-AND-HORIZON` so single-regime experiments cannot claim global scaling laws.

### 3.6 Evidence Tiers, `result.zip` (`<= 20 MB`), and Gate 5 Parity (`evidence.py`, `checkpoint_intelligence.py`, `cartridge.py`)
- **Tier 1 (`< 100 KB`):** `SCIENTIFIC_SUMMARY.json` + `SCIENTIFIC_SUMMARY.md`, plus `AI_HANDOFF_BRIEF.json` (`< 8 KB`) and `LAUNCH_GATE_CERTIFICATE.json`.
- **Tier 2 (`< 2 MB`):** `CAPSULE_MANIFEST.json`, `diagnostics.json`, and float16-compressed `representations.npz`; offline-probable via `probe_from_capsule()` without loading Tier-3 weights.
- **`export_result_bundle()`**:
  - Builds `checkpoint_full.zip` containing complete SHA-256-verified `CheckpointStore` lineages (all milestones, `model.bin`, `optimizer.bin`, `rng.bin`).
  - Builds `result.zip` targeting `<= 20 MiB` (`20,971,520` B): tries `full_embedded` (latest `model.bin` + `optimizer.bin` + store files) first; if size `> max_result_zip_bytes`, automatically falls back to `compact_fp16` (converts floating-point tensors in `model.bin` to `float16` `model_fp16.bin` while keeping `manifest.json` / `scheduler.json` / `training_state.json`).
  - Rejects path traversal (`..`, backslashes, absolute paths) in `_safe_arcname` and `_safe_extract_members`.
- **`CartridgeRunner` 5 Launch Gates (`cartridge.py`):**
  1. `gate1_instrument_integrity`: `validate_environment_kernel` + `validate_metric_aggregator`.
  2. `gate2_confounder_firewall` (`audit_experimental_design`): enforces matched `dataset_identity`, `order_seed`, `recipe.seed`, `total_updates`, `batch_size`, `lr`, `schedule`, and `max_param_ratio` when `held_constant` mentions capacity/param parity.
  3. `gate3_gradient_and_mechanism_health`: pre-update `validate_model_and_gradients` + bounded training + SHA-256 `MILESTONES` recording.
  4. `gate4_trajectory_and_causal_attribution`: milestone trajectory + all 6 causal/ablation diagnostics + Tier-2 capsule verification.
  5. `gate5_bundle_and_live_parity_self_audit`: restores `checkpoint_full.zip`, reloads every arm directly from `result.zip`, and asserts **`prediction_parity_rate == 1.0`** and max hidden abs diff (`<= 1e-3` for `full_embedded`, `<= 5e-2` for `compact_fp16`).

---

## 4. Adversarial Review Checklist (Where Other AI Agents Should Attack Planthir)

**To any auditing AI agent:** Treat every item below as an active hypothesis of a potential flaw, oversimplification, or blind spot in Planthir. Inspect the exact code referenced and report concrete fixes, counterexamples, or upgrades!

1. **Probe Linearity Blind Spot (`causal.py:run_representation_linear_probe` & `evidence.py:probe_from_capsule`):**
   - Currently fits only a linear L2 ridge classifier ($\lambda=1.0$). If a nonlinear recurrent/attention architecture encodes `hop_1` or `final_symbol` along a curved manifold (e.g., XOR / product binding before final projection), the linear probe can score poorly and misclassify a *nonlinear encoding* as *representation absence*. Also, `_fit_ridge_classifier` uses one-hot MSE ridge instead of regularized cross-entropy, and doesn't tune $\lambda$ or guard against `D >> N` rank deficiency beyond `l2_reg=1.0`.
   - *Audit Question:* Should Planthir add a standardized CV-tuned ridge + 2-layer MLP probe pair (reporting the linear-vs-nonlinear gap)?

2. **Fixed Heuristic Thresholds vs. Sample-Size & Statistical Power Calibration (`interpretation.py`):**
   - Rules like `head_minus_probe_gap >= 0.15`, `stage1 - stage4 >= 0.25`, `stage2 < 0.80`, `surgical_ablation.accuracy_delta >= 0.05`, and `effective_rank < 3.0` are fixed constants. On small suites ($N=16\text{–}24$), random noise can exceed `0.15` or fall below statistical power; on tiny architectures (`D=32`), `effective_rank < 3.0` means something different than on `D=512`.
   - Furthermore, **Gate 5 passes `S-SINGLE-HOP-GROUNDING-DEFICIT` or `C-INCONTEXT-BOTTLENECK` even on a 6-update warmup where overall accuracy is `0.000`** (`EXP-PLANTHIR-004` on RTX 4050 showed `loss 2.87 -> 2.52`, `acc=0.000`, yet `C-INCONTEXT-BOTTLENECK` fired as `"confirmed"`!). Should claims that assume convergence be gated by minimum training progress or baseline learnability checks, and should paired binary comparisons use **exact McNemar's test + Holm-Bonferroni / FDR adjustment** rather than fixed delta thresholds alone?

3. **2-Hop Ceiling vs. General $N$-Hop Subgoal DAGs (`adapters/v8_adapter.py:decompose_subgoals`, `causal.py:run_subgoal_hop_decomposition`):**
   - `V8EnvironmentKernel.decompose_subgoals` assumes 2 hops (`hop_1` and `hop_2_given_bridge`). For AN-RA V9 (3+ hops, trees/DAGs, partial credit, or noisy memory retrieval), this must generalize cleanly to arbitrary `hop_k` chains and cumulative intermediate dependency accuracy.

4. **Off-Manifold Artifacts in Zero-Ablation (`causal.py:run_surgical_mechanism_ablation`):**
   - Temporarily zeroing `cap_out` in `F-cap` or clamping `recurrent_steps=0` in `U` tests zero-ablation/skip-ablation. Zero-ablation can push downstream activations off-manifold if a module emits a non-zero mean offset! Also, `run_surgical_mechanism_ablation` currently hardcodes `arch in {"P", "U", "F-cap"}` and skips bootstrap CIs/McNemar $p$-values on `accuracy_delta` (`ornamental_mechanism_bypass` requires `abs(acc_delta) <= 1e-6`, which could miss a nearly-ornamental mechanism with `delta = +0.01`). How should we register arbitrary custom ablation hooks via `ObservablePolicy` for future V9 modules?

5. **Confounder Firewall Coverage & Nested Config Blind Spot (`cartridge.py:audit_experimental_design`):**
   - Gate 2 checks `dataset_identity`, `order_seed`, `recipe.seed`, `total_updates`, `batch_size`, `lr`, `schedule`, and conditionally checks `max_param_ratio` **only if `"param" / "capacity" / "matched"` appears in `held_constant` strings**.
   - *Bug / Loophole risk:* If an agent omits `"matched"` from `held_constant` tags, Gate 2 does NOT enforce `max_param_ratio`! Also, Gate 2 does not inspect `recipe.precision`, optimizer weight decay / betas, gradient clipping threshold, warmup steps, or custom `trainer_factory` kwargs — and in `CartridgeRunner`, **`eval_worlds` is shared directly inside `EXP-PLANTHIR-004` rather than enforcing zero hash overlap against `train_data`!**

6. **Cross-Architecture Universality Gap (`cartridge.py:CartridgeArmSpec`):**
   - Notice `CartridgeArmSpec` still carries `v8_spec: V8ModelSpec` and `CartridgeRunner.run` currently calls `V8TypedTrainer`, `compute_typed_nll`, and `V8ObservablePolicy` directly unless `trainer_factory` is supplied — AND even with `trainer_factory`, Gate 3 lines 439–449 build `toks, segs, apos` assuming V8 typed tensor batches, while Gate 4 calls `V8EnvironmentKernel`-specific `inject_oracle_information`! True V9 universality requires `trainer_factory` + generic `ObservablePolicy` gradient-loss hook + `EnvironmentKernel` capability flags instead of assuming `V8TypedTrainer` sample fields.

7. **Naming / Semantic Sign Clarity (`causal.py:436`):**
   - Look closely at `causal.py:436`:
     `"head_minus_probe_gap": (probe_final_acc - head_final_acc)`!
     Notice the key is literally named **`head_minus_probe_gap`**, yet the subtraction computed is **`probe_final_acc - head_final_acc`** (which is `probe_minus_head_gap`)! When the probe beats the head (`0.190 > 0.000`), `"head_minus_probe_gap"` is positive (`+0.190`), which matches the doc rule in `interpretation.py:286`, even though the key name implies the opposite sign (`head - probe`)! That is a textbook subtle naming-sign trap worth fixing cleanly with backwards compatibility!

8. **Token Fuel Estimate Hardcoding (`cartridge.py:estimate_token_fuel_savings`):**
   - Look at `cartridge.py:184-186`: `baseline_from_scratch_tokens_est = 1_000_000` and `cartridge_workflow_tokens_est = 180_000` are static engineering estimates rather than dynamic measurements derived from actual lines of cartridge code (`author_loc`) and raw diagnostic sizes (`raw_summary_bytes`, `handoff_brief_bytes`). While `context_handoff_compression_ratio` (`8.66x` in `EXP-PLANTHIR-004`) *is* empirically measured byte-for-byte, the headline `82.0%` reduction is a heuristic constant that should either become formula-driven or be clearly separated from measured compression!

9. **`compact_fp16` Checksum & Live-Parity Interaction (`evidence.py` & `checkpoint_intelligence.py`):**
   - When `compact_fp16` is triggered (`result.zip > 20 MB`), `model_fp16.bin` replaces `model.bin` + `optimizer.bin`, so `_restore_compact_fp16_object` bypasses `CheckpointStore.restore()` and validates `model_fp16.bin` only via `ARCHIVE_MANIFEST.json` SHA-256 (verified by `verify_single_planthir_zip`). Verify whether `manifest.json` component checksums inside `compact_fp16` could confuse downstream `CheckpointStore` tools that expect exact FP32 `model.bin` bytes.

---

## 5. Verification Baseline Already Observed on Repo Commit `21f57160` (`planthir` branch)

Auditors can reproduce the current baseline before and after proposing edits:
- **CPU Calibration & Study (`EXP-PLANTHIR-001/002/003`):** `python -m anra_v5.planthir run-validation-study --output-dir "Experiment Result/Planthir"` (`6/6` faults caught; 6 V8 arms packaged & restored).
- **Live GPU Validation (`EXP-PLANTHIR-004`):** `EXP_PLANTHIR_004` passed all 5 Gates on `NVIDIA GeForce RTX 4050 Laptop GPU` (`cuda:0`, 60 updates/arm, `75.56 MB` peak VRAM, `51.464s`, `result.zip = 4,255,655 B`, `checkpoint_full.zip = 12,624,167 B`, `1.000` live-vs-restored prediction parity across `J/F/P/U`).
- **Regression Suite:** `29 passed in 68.04s` across `tests/test_planthir_universal_infrastructure.py` and V8/Elendril tests; `python -m tools.verify_elendril_evidence` exited `0`.
