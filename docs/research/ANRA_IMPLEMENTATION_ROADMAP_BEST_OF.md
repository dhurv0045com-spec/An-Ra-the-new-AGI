# An-Ra Implementation Roadmap: Best-Of Intelligence and Efficiency

Date: 2026-06-08

This roadmap converts `ANRA_BEST_RESEARCH_FOR_INTELLIGENCE_AND_EFFICIENCY.md` into implementation-ready work. It keeps the An-Ra vision fixed: owner-shaped intelligence, verifier-grounded improvement, memory continuity, sovereignty gates, and measurable subsystem health.

## Operating Rule

No research idea is allowed to change the brain, training loop, memory stack, or self-improvement policy unless it beats the current system on a named eval and does not regress identity, safety, or owner style.

## Implementation Status - 2026-06-08

This pass implemented the measurement and experiment scaffolds for the first five roadmap items. These are deliberately conservative: they create reports, knobs, and owner-review gates, but they do not promote risky research ideas into default behavior without eval evidence.

| Roadmap item | Status | Artifact / code path | Push-readiness note |
|---|---|---|---|
| Golden eval baseline | Implemented scaffold | `output/v2/v2_golden_eval_baseline.json`, `training/eval_v2.py` | `train_unified --mode eval` writes the artifact; still needs a run against the checkpoint chosen for promotion. |
| SparseLoRA-style adaptation | Implemented logging-only estimate | `output/v2/v2_sparse_lora_report.json`, `training/sparse_lora.py` | Measures estimated skippable context-token work; does not skip training tokens yet. |
| Optimizer bake-off | Implemented adapter/report scaffold | `output/v2/v2_optimizer_bakeoff_report.json`, `training/anra_optimizer.py` | AdamW runs directly; Muon runs if installed; SCALE/GaLore remain fallback/report-only until verified packages are present. |
| RLVR upgrade path | Implemented DAPO-style telemetry | `output/v2/v2_rlvr_report.json`, `training/rlvr.py` | Adds reward shaping, token-level loss knob, dynamic sampling knob, KL/output/replay metrics; default policy remains conservative. |
| GEPA self-improvement | Implemented proposal scaffold | `output/v2/v2_gepa_report.json`, `training/gepa.py` | Generates trace-backed prompt/tool-policy proposals; auto-apply is disabled and owner/sovereignty review is required. |
| TurboQuant/KVarN audit | Next | `core/turboquant.py` | Next P0 item after this pre-push audit. |

Latest verification before push:

- Focused roadmap suite: `74 passed`
- Full test suite: `259 passed, 1 warning`
- Compile check for touched Python modules: passed
- Exact merge-conflict marker sweep: clean

## P0: Do First

### 1. Golden eval baseline

Component: `evaluation`

Why: every efficiency idea needs proof that it makes An-Ra better rather than merely faster.

Implementation:

- Define a stable eval pack under `output/v2/eval/` for identity, owner voice, symbolic math/code, memory recall, tool use, and latency.
- Use `engine/eval_harness.py` to run baseline, system-on, and ablation modes.
- Save a JSON summary with task success, latency, token count, and notes.
- Add a "promotion allowed" boolean that is false if identity, symbolic correctness, or safety regress.

Acceptance:

- `train_unified --mode eval` or a focused eval command writes a golden artifact.
- The artifact can be compared on later runs.
- Any training/optimizer change can point to this baseline.

### 2. SparseLoRA-style efficient owner adaptation

Component: `training_loop`

Why: user data should improve the model with less compute.

Implementation:

- Keep LoRA/QLoRA as baseline.
- Add an experimental adapter mode that estimates contextual sparsity and skips low-value context-token gradient work.
- Start with logging-only sparsity estimates if full sparse kernels are too large.
- Measure wall-clock time, memory, loss, and eval delta.

Acceptance:

- Same data slice, same seed, same eval.
- Report compares baseline LoRA/QLoRA vs sparse mode.
- Sparse mode must not reduce identity or symbolic eval pass rate.

### 3. Optimizer bake-off: AdamW vs Muon vs SCALE vs GaLore

Component: `training_loop`

Why: optimizer-state memory is one of the biggest bottlenecks for serious training.

Implementation:

- Add optimizer adapters behind config names: `adamw`, `muon`, `scale`, `galore`.
- Run on a small An-Ra model first, then larger only if stable.
- Record memory, tokens/sec, loss curve, eval pass rate, and gradient stability.

Acceptance:

- A single report table ranks optimizers by quality per memory.
- Any optimizer that produces instability, identity drift, or worse eval is quarantined.

### 4. RLVR upgrade path

Component: `training_loop`

Why: An-Ra's strongest path to better reasoning is verifier-backed improvement, not raw chat imitation.

Implementation:

- Keep current GRPO in `training/rlvr.py` as baseline.
- Add DAPO-style knobs first: overlong reward shaping, token-level policy loss, dynamic sampling toggle, and KL logging.
- Add S-GRPO/T-SPMO as a memory-constrained mode.
- Add GRPO-lambda only after baseline and DAPO-style runs are stable.
- Record all verifier outputs through replay provenance.

Acceptance:

- Each RLVR step logs G, rewards, advantage stats, KL, output length, verifier pass rate, and replay additions.
- Verified examples enter `training/replay_pipeline.py`.
- Failed examples enter hard-replay or analysis queues, not silent discard.

### 5. GEPA-style self-improvement loop

Component: `self_improvement`

Why: An-Ra can improve prompts, tool policies, and verifier instructions before expensive weight updates.

Implementation:

- Collect traces from tool calls, symbolic checks, failed goals, and eval failures.
- Generate natural-language reflections that identify the failure cause.
- Propose prompt/tool-rule edits as candidate artifacts.
- Score candidates against eval tasks and keep only Pareto improvements.

Acceptance:

- No automatic identity/prompt drift without owner/sovereignty gate.
- Every accepted prompt/tool rule has trace evidence and an eval delta.
- GEPA loop uses fewer rollouts than RLVR for prompt/tool improvements.

### 6. TurboQuant and KVarN audit

Component: `runtime`

Why: `core/turboquant.py` already exists, but An-Ra needs proof that it helps real autoregressive decoding.

Implementation:

- Audit the current implementation against the TurboQuant paper.
- Add compression, reconstruction, attention-score, and long-decode quality checks.
- Compare with KVarN's core idea: Hadamard rotation plus dual-scaling variance normalization for autoregressive error accumulation.
- Do not wire compressed KV into default generation until eval is green.

Acceptance:

- Report shows memory ratio, attention-score error, generation-quality delta, and latency.
- KVarN is either promoted to prototype, rejected with evidence, or left watchlisted.

## P1: Build After Baselines Exist

### 7. TurboVec-like compressed memory index

Component: `memory`

Implementation:

- Add a memory-index benchmark for FAISS, NumPy fallback, and optional compressed-vector backend.
- Metrics: recall@k, query latency, build time, memory bytes, update cost.
- Keep optional dependency isolated.

Acceptance:

- Compressed index cannot become default unless recall-sensitive memory tasks remain green.

### 8. LMCache-style prefix/KV reuse for agent sessions

Component: `runtime`

Implementation:

- Detect repeated system/identity/tool context prefixes in chat and agent loops.
- Prototype cache metadata only first: prefix hash, token count, reuse hit rate.
- Add actual KV reuse only after generation path is stable.

Acceptance:

- Report shows repeated-prefix savings without changing output semantics.

### 9. EAGLE-3 and speculative decoding feasibility

Component: `runtime`

Implementation:

- Start with external survey and small prototype, not default integration.
- Decide whether a draft model, EAGLE head, or n-gram speculation best fits An-Ra.
- Measure speed only under real prompt lengths and batch sizes.

Acceptance:

- End-to-end speed improves without output quality regression.
- Training-time rollout generation is considered separately from user-facing inference.

### 10. Better replay into data mix

Component: `data_mix`

Implementation:

- Tag replay examples by source: symbolic, code verifier, RLVR, GEPA, owner correction, tool failure.
- Add weights and decay policy.
- Keep owner data dominance intact.

Acceptance:

- Replay is visible in mix reports.
- Bad failures do not become training targets without corrected targets.

## P2: Research Prototypes Only

### 11. Titans-style test-time memory

Component: `ghost_memory`

Implementation:

- Prototype outside the brain as a memory scorer or reranker.
- Compare against current ghost memory and retrieval.

Acceptance:

- Must improve long-context recall without corrupting identity.

### 12. Mamba / Gated DeltaNet / Log-Linear Attention experiments

Component: `brain`

Implementation:

- Do not rewrite the main model.
- Build small isolated blocks and compare on the same tokenizer, data, and eval.

Acceptance:

- Promote only if quality per compute clearly beats transformer baseline.

### 13. MoE conversion

Component: `brain`

Implementation:

- Treat as long-term architecture research.
- Use teacher/distillation from MoE models before trying sparse expert training.

Acceptance:

- No default path until routing quality and identity stability are proven.

## First 30 Days

### Week 1: Baselines and paper-to-repo audit

- Finish golden eval design.
- Verify current TurboQuant tests and gaps.
- Map every selected research idea to a component and metric.

### Week 2: Training efficiency experiments

- Implement optimizer bake-off scaffolding.
- Add LoRA/QLoRA baseline report.
- Add SparseLoRA logging-only prototype if kernels are not ready.

### Week 3: Reasoning and self-improvement experiments

- Add DAPO-style logging and reward shaping to RLVR.
- Add replay provenance fields.
- Prototype GEPA-style reflection over failed eval/tool traces.

### Week 4: Memory and inference feasibility

- Benchmark FAISS vs fallback memory store.
- Design TurboVec optional backend experiment.
- Compare TurboQuant and KVarN on paper-level requirements.
- Decide whether EAGLE-3 is worth a prototype.

## Promotion Gates

A feature can move from experiment to default only if:

- Golden eval success rate does not regress.
- Identity/CIV/ESV checks pass.
- Symbolic/verifier tasks do not regress.
- Latency or memory improves on a named workload.
- Engineering log and report artifacts are updated.
- A rollback flag exists.

## Do Not Do Yet

- Do not replace the transformer brain with Mamba/Gated DeltaNet/Titans directly.
- Do not make TurboVec or TurboQuant default from paper claims alone.
- Do not train on replay failures without corrected targets.
- Do not optimize for token speed while losing owner style or truthfulness.
- Do not accept "5x efficiency" without naming the exact metric and benchmark.
