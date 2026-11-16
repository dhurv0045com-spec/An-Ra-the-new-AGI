# Planthir — Universal Scientific Observatory & Research Infrastructure for AN-RA

> *"Can we systematically improve our ability to discover how intelligence works, rather than merely improving our ability to measure whether a model succeeds?"*

Planthir is AN-RA's universal, architecture-agnostic scientific observatory built on top of the repository's existing Elendril / V5–V8 contracts (`CheckpointStore`, `ModelSpec`, truth firewalls, and deterministic seeds) rather than replacing them. It shifts experimental work from single-number outcome reporting toward **verified causal mechanism discovery per GPU hour**.

---

## 1. Scientific Observatory Philosophy

Across 52 primary archived runs spanning `V6` through `V7.17` (`EXP-PLANTHIR-003`), two structural failure modes recurred:

1. **Process-vs-Outcome Decoupling:** Models routinely achieved near-100% query recall (`V7`: `1.000`, `V7.15`: `1.000`, `V7.16`: `0.995`, `V7.17`: `0.977`) while confirmation accuracy remained bounded (`~0.57–0.74`) and surface-alias robustness collapsed (`0.147–0.250`). Outcome scores alone could not distinguish between missing internal representations, brittle readout heads, closed pointer gates, or multi-hop state decay.
2. **Silent Instrument Defects:** Reporter/evaluator bugs (`V7.16` `ep > final_epoch` TypeError shadowing; `V8` consumed `QUERY`/`RETRIEVE` tools remaining marked available across subsequent episode steps) could masquerade as model failures unless caught by an active calibration layer before interpreting results.

Planthir solves both by separating **Instrument Integrity**, **Internal Observability**, **Causal Counterfactuals**, **Epistemic Interpretation**, **3-Tier Evidence Custody**, and **Checkpoint Intelligence** behind minimal universal interfaces.

---

## 2. Universal Contracts (`anra_v5/planthir/contracts.py`)

Every research program—Transformers, pointer-readers, recurrent state updaters, external memory systems, continual learning replay curricula, or metacognitive confidence heads—integrates with Planthir through two lightweight protocols:

- **`EnvironmentKernel`** (`initial_state`, `observe`, `ideal_action`, `step`, `verify`, `decompose_subgoals`): Single source-of-truth state machine shared across teacher, interactive evaluation, verifier, and counterfactual subgoal probes. Emits **`UniversalObservation`** objects carrying mandatory `available_tools`, `consumed_tools`, `decision_index`, `budget_remaining`, and self-checking `.validate_invariants()`.
- **`ObservablePolicy`** (`propose_with_telemetry`, `extract_representations`): Emits **`UniversalAction`** containing executed action, raw unmasked proposal, full action distribution, calibrated log-probability, and mechanism telemetry (`pointer_gate`, `pointer_attention_entropy`, `recurrent_mean_delta`, etc.), alongside hidden-state tensors (`decision_hidden`) for geometric and linear-probe analysis.
- **Universal Registration (`PlanthirRegistry`, `ExperimentRegistration`, `HypothesisSpec`, `FalsifierSpec`, `CausalContrastSpec`)**: Content-addressed pre-registration binding hypotheses, falsification thresholds, controlled vs. varied factors, dataset SHA-256, source commit, and parameter budgets across architectures.

---

## 3. Instrument Validation & Controlled 6-Fault Calibration (`instrument.py`)

Before Planthir interprets any model failure as a capability failure, three gates run automatically:

1. **`validate_environment_kernel`**: Verifies tri-parity between teacher labels, rollout ideal actions, and independent verification; enforces **tool-consumption monotonicity** (consumed tools must remain unavailable) and **truth-firewall leakage markers** (`detect_truth_leaks`); validates V8 evaluator parity.
2. **`validate_model_and_gradients`**: Executes a forward/backward contract pass confirming finite loss and nonzero gradients across all expected active parameters (catching silent head disconnects).
3. **`validate_metric_aggregator`**: Supplies synthetic multi-epoch fixtures reaching double-digit epochs (`epoch-10`) to fail fast on ordering/type-shadowing bugs.

### Historical V8 Consumed-Tool Fix (`anra_v5/elendril/v8/evaluation.py`)

Auditing `v8/evaluation.py::run_episode` confirmed that executing `QUERY` set `queried = True` without clearing `query_avail`, and executing `RETRIEVE` set `retrieved = True` without clearing `mem_avail`. Planthir added `enforce_tool_consumption: bool = True` to clear tool availability flags immediately upon consumption while retaining `enforce_tool_consumption=False` strictly for fault-reproduction benchmarks.

In `EXP-PLANTHIR-001` (`4.569s` on CPU):
- With `enforce_tool_consumption=False`, a stateless visible-state oracle scored `6/12` (`50.0%`), looping on already-consumed tools in `6` episodes.
- With `enforce_tool_consumption=True`, the identical oracle achieved `12/12` (`100.0%`).
- `run_fault_injection_suite()` detected and localized `6/6` controlled injected faults: `v8_consumed_tool_availability_bug`, `truth_firewall_leak`, `train_eval_prompt_drift`, `disconnected_policy_head_grad`, `v16_epoch_aggregation_shadowing`, and `checkpoint_rng_corruption`.

---

## 4. Scientific Observability: Internal Telemetry & 4-Stage Process Funnel (`observatory.py`)

- **`InternalTelemetryCollector`**: Computes per-module group (`embedding`, `backbone`, `pointer_or_memory`, `recurrent`, `head`) L2 weight/gradient norms, `update_to_weight_ratio`, dormant weight fractions, batch **Shannon effective rank** (`exp(H(sv))`), representation **isotropy**, and mechanism telemetry without heavy activation dumps.
- **`ProcessFunnelAnalyzer`**: Traces every rollout across a four-stage causal bottleneck:
  - **Stage 1 (`stage1_info_seeking`):** Did initial policy correctly identify tool need?
  - **Stage 2 (`stage2_tool_discipline`):** After consuming a tool, did it abstain from repeating consumed tools and incorporate returned facts?
  - **Stage 3 (`stage3_single_hop_grounding` / `stage3_subgoal_hop1_grounding`):** Can it execute atomic single-hop variable binding?
  - **Stage 4 (`stage4_multi_hop_composition` / `stage4_epistemic_abstention`):** Can it compose 2+ hops end-to-end and abstain on `null`/`unavailable` targets?
- **Confidence Calibration:** Computes `brier_score`, Expected Calibration Error (`ece`), and Area Under Risk-Coverage Curve (`aurc`), plus exact `first_divergence` counts across 9 failure taxonomy labels (`DIVERGENCE_TAXONOMY`).

---

## 5. Frozen-Model Causal Counterfactual Engine (`causal.py`)

Working directly on frozen checkpoints (`no retraining`):

1. **`run_oracle_information_intervention`**: Pre-merges hidden tool payloads to measure paired `delta_oracle` (with 95% bootstrap CI), cleanly separating **tool-acquisition failure** from **reasoning-over-visible-evidence failure**.
2. **`run_subgoal_hop_decomposition`**: Decomposes multi-hop worlds into isolated Hop-1 and Hop-2-given-bridge subproblems on identical surface rendering, quantifying `composition_gap = P(Hop-1 & Hop-2 separately) - P(Full Multi-Hop)`.
3. **`run_invariance_counterfactuals`**: Tests cyclic symbol permutation equivariance, row-order reversal invariance, irrelevant distractor chain injection, and decoder mode overrides (`raw`, `grammar`, `availability-mask`, `direct-answer-only`).
4. **`run_representation_linear_probe`**: Solves closed-form multi-class ridge linear probes (`decision_hidden` -> operation, Hop-1 symbol, final symbol) to isolate **representation absence** vs **readout head failure** (`head_minus_probe_gap`).

---

## 6. Epistemic Firewall & Scientific Interpretation (`interpretation.py`)

`synthesize_scientific_interpretation` enforces a strict **epistemic firewall**:

- If `instrument_validation["passed"]` is `False`, interpretation halts immediately (`status = "INVALIDATED_BY_INSTRUMENT_DEFECT"`), emits zero causal claims, and ranks `REPAIR_INSTRUMENT_DEFECT` first.
- When the instrument passes, findings are partitioned into:
  - **`confirmed_causes`** (supported by statistically non-zero paired counterfactuals or exact trace invariants),
  - **`suspected_mechanisms`** (suggestive signatures like closed pointer gate or readout gaps requiring follow-up control),
  - **`falsified_hypotheses`** (explicit `FalsifierSpec` threshold failures),
  - **`unresolved_questions`** (always including explicit single-regime scale/horizon scope guards so certainty is never manufactured), and
  - **`recommended_next_experiments`** sorted by estimated expected information gain per GPU hour (`CPU_SECONDS`, `SINGLE_T4_MINUTES`, `DUAL_T4_HOURS`).

---

## 7. Three-Tier Evidence Architecture, Packaged Result Bundles & Checkpoint Intelligence (`evidence.py`, `checkpoint_intelligence.py`)

1. **Tier-1 Scientific Summary (`< 100 KB`, Git-committed):** `SCIENTIFIC_SUMMARY.json` + `SUMMARY.md` capturing registration metadata, calibration checks, funnel breakdowns, counterfactual deltas, interpretation claims, and provenance.
2. **Tier-2 Diagnostic Capsule (`< 2 MB`, SHA-256 tamper-checked, offline-probable):** Compressed `float16` `representations.npz` + `diagnostics.json` + `CAPSULE_MANIFEST.json`. `probe_from_capsule()` re-runs effective-rank and linear ridge probes without loading Tier-3 checkpoints (`12.4–14.2 KB` per arm in `EXP-PLANTHIR-002`).
3. **Packaged `result.zip` (`<= 20 MB`) & Whole-Store `checkpoint_full.zip` (`export_result_bundle`):** Every campaign directory exposes clean, self-describing `FINAL_STATUS.json`, `CHECKPOINT_INDEX.json`, `RESULT_OVERVIEW.md`, compact `result.zip` (guaranteed `<= 20 MB` by bundling Tier-1 summaries + Tier-2 capsules + full checkpoints when under budget or compact `float16` `model_fp16.bin` weights + `LATEST`/manifests when larger), and `checkpoint_full.zip` containing the complete SHA-256-verified `CheckpointStore` recovery archive. `CheckpointCatalog.load_from_result_zip()` verifies archive checksums and loads frozen models directly from `result.zip` (`full_embedded` or `compact_fp16`), while `CheckpointCatalog.restore_full_checkpoint_zip()` extracts and SHA-256-verifies full stores.
4. **Tier-3 Recovery Archive Custody (`CheckpointStore` + `CheckpointCatalog`):** Never duplicated in Git; indexed via `CheckpointCatalog.discover_checkpoints()` without reading `model.bin` into RAM until explicitly requested via `load_checkpoint_for_causal_probing()` (supports both `J/F/P/F-cap/U` `V8TypedTrainer` checkpoints and `B0` `IndependentOrderTrainer` checkpoints, plus `diff_checkpoints()` to audit held-constant vs. altered factors).

In `EXP-PLANTHIR-002` (bounded CPU canary across `B0`, `J`, `F`, `P`, `F-cap`, `U` at `17,680–26,756` params, `6` updates/arm): all 6 checkpoints were trained, discovered, SHA-256 restored from disk, causally probed across all counterfactual interventions, compressed into `12.4–14.2 KB` Tier-2 capsules (`probe_from_capsule`), packaged into self-verifying `result.zip` + `checkpoint_full.zip` archives, and directly restored/probed back from both archives without retraining.

---

## 7b. Level-5 5-Gate Self-Auditing Experiment Cartridge (`cartridge.py`, `EXP-PLANTHIR-004`)

To prevent human verification overhead ("checking the instrument every single experiment") and cut AI authoring/interpretation token cost ("rocket fuel savings"), `anra_v5.planthir.cartridge` enforces five automated Launch Gates around any `ExperimentCartridge`:

1. **Gate 1 — Instrument Integrity:** Pre-flight kernel determinism, leak checks, consumed-action enforcement, and aggregator monotonicity.
2. **Gate 2 — Confounder Firewall (`audit_experimental_design`):** Rejects mismatched seeds, datasets, update budgets, batch sizes, LRs, schedules, or unmatched capacity confounds before any GPU compute runs.
3. **Gate 3 — Gradient Health & Milestone Checkpoints:** Verifies active gradient flow before optimization and commits SHA-256 `MILESTONES` (`analyze_checkpoint_trajectory`).
4. **Gate 4 — Milestone Trajectory & 6-Intervention Attribution:** Runs 4-stage funnel, milestone trajectory (detecting representation-before-readout phase transitions), oracle injection, subgoal decomposition, invariances, ridge probes, **`run_surgical_mechanism_ablation`** (detecting ornamental/bypassed mechanisms), and **`run_compute_depth_scaling_intervention`**.
5. **Gate 5 — Bundle Integrity & 100% Live-vs-Restored Parity:** Exports `result.zip` (`<= 20 MB`), `checkpoint_full.zip`, `AI_HANDOFF_BRIEF.json` (`<= 8 KB`), and `LAUNCH_GATE_CERTIFICATE.json`, reloads each checkpoint from `result.zip`, and asserts 100% action-prediction parity against the live trained model.

In **`EXP-PLANTHIR-004`** (live on `NVIDIA GeForce RTX 4050 Laptop GPU`, `60` updates/arm, `92,115–121,300` params, `75.56 MB` peak VRAM, `51.46s`): all 5 Launch Gates passed, `100%` live-vs-restored CPU parity held across `J/F/P/U`, surgical ablation detected `S-ORNAMENTAL-MECHANISM-BYPASS` on `U`, and `AI_HANDOFF_BRIEF.json` compressed `48,283` B of diagnostics to `5,978` B (`8.66x` compression, ~`82%` end-to-end token savings).

---

## 8. Minimal Extension Guide (`< 50` Lines for Future AN-RA Programs)

To bring a **V9 architecture**, **continual learning scheduler**, **external episodic memory bank**, or **metacognitive confidence predictor** into Planthir without changing core observatory, causal, or evidence code:

```python
from anra_v5.planthir.contracts import UniversalObservation, UniversalAction

class MyV9Kernel:
    kernel_id = "anra-v9-memory-kernel"
    def initial_state(self, world): ...
    def observe(self, state) -> UniversalObservation: ...  # enforce validate_invariants()
    def ideal_action(self, state): ...
    def step(self, state, operation, argument): ...
    def verify(self, world, actions): ...
    def decompose_subgoals(self, world): ...  # optional multi-stage decomposition

class MyV9Policy:
    policy_id = "v9-episodic-memory"
    def propose_with_telemetry(self, obs: UniversalObservation) -> UniversalAction:
        # Expose e.g. {"memory_hit_entropy": ..., "metacog_confidence": ...}
        ...
    def extract_representations(self, obs_batch):
        return {"decision_hidden": hidden_tensor}  # [N, D]
```

Once those two classes exist (~30–50 lines), `validate_environment_kernel`, `InternalTelemetryCollector`, `ProcessFunnelAnalyzer`, `run_representation_linear_probe`, `synthesize_scientific_interpretation`, and `package_three_tier_evidence` work out of the box.

---

## 9. Limitations & 10–20 Day Reassessment Protocol

### Known Current Limitations
1. **CPU Validation Scope:** Local validation tests tiny models (`17.7k–26.8k` params, 6 updates) to validate pipeline mechanics and calibration. It measures neither `S20`-scale optimization dynamics nor Kaggle `2xT4` throughput.
2. **Closed-Form Linear Probe Scope:** `run_representation_linear_probe` fits L2 ridge classifiers on `decision_hidden` (`D <= 512`). For high-dimensional hidden states (`D > 2048`) or nonlinear encodings, add randomized sketching or MLP probes in a future minor increment.
3. **Subgoal Counterfactual Scope:** `run_subgoal_hop_decomposition` currently decomposes 2-hop chains into `hop_1` and `hop_2_given_bridge`; deeper `3–4` hop chains in V9 will benefit from generalized recursive `hop_k` specs in `EnvironmentKernel.decompose_subgoals`.
4. **Binary Bundle & Capsule Git Exclusion:** `.zip` bundles (`result.zip`, `checkpoint_full.zip`), `.npz` capsules, and `.bin` checkpoint files are excluded from Git by `.gitignore` and `AGENTS.md`; external artifact buckets store the verified bundles while curated Markdown reports, SHA-256 manifests, and summary JSON live in Git.

### Reassessment Checklist (Every 10–20 Days)
1. Re-run `python -m anra_v5.planthir calibrate` before and after any evaluator or data-generator edit.
2. Audit `EXP-PLANTHIR-003` (`historical-atlas`) against newly completed Kaggle/Colab runs (`V8` and beyond).
3. Inspect whether newly emerging research hypotheses (e.g., continual learning forgetting curves, memory overwrite interference, metacognitive abstention calibration) require adding one new reusable counterfactual helper in `causal.py` rather than ad-hoc notebook scripts.
