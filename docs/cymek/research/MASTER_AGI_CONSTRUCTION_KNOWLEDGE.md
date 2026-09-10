# AN-RA MASTER AGI CONSTRUCTION KNOWLEDGE

**Status:** CROSS-BRANCH SYNTHESIS / RESEARCH AUTHORITY MAP / NOT A CLAIM OF AGI  
**Synthesis date:** 2026-09-10  
**Purpose:** give a human or autonomous research agent one file that explains what An-Ra currently knows about building a stronger general-learning Core, what construction is justified now, what is only a hypothesis, what failed, what remains unknown, and what evidence must exist before scaling.

> **Two-sentence definition of this file:** This is the shortest honest path from all current An-Ra evidence to a buildable research system: a conventional neural Core trained on auditable data, measured with causal/anti-shortcut evaluations, and surrounded by a fail-closed training, verification, retention, diagnosis, and promotion loop. It does **not** say we know how to build “perfect AGI”; it says exactly which pieces are demonstrated, which are implemented but unexecuted, which are speculative, and which experiment should change the design next.

---

## 0. Evidence snapshot and authority

This synthesis was built from the live heads of the five requested branches, not from memory or old prose:

| Branch | Live head used | Role in this synthesis | Highest-value Markdown evidence |
|---|---|---|---|
| `cymek` | `28bf57a0d299a2c13a99fe0046616c00a1b8530c` | production Core/training contracts | `blueprint/V5_TRAINING_SPEC_v1.0.md`, `V5_MASTER_BLUEPRINT.md`, `IMPLEMENTATION_BLUEPRINT.md`, `BENCHMARK.md` |
| `citadel` | `1d27f9b0d770e30577de0a8671c909cb783b4ff1` | independent evidence audit, negative results, TPU experiments, 500M gate | `docs/citadel/EVIDENCE_LEDGER.md`, `NEGATIVE_RESULTS.md`, `EXPERIMENTS_BRIEF.md`, `BOTTLENECK_RANKING.md`, `experiments/T1D/RESULTS.md`, `500M/PRODUCTION_PATH_AUDIT.md`, `500M/CYMEK_REQUIRED_CHANGES.md` |
| `triquetra` | `f23f0af42d90847cf1d2c244160c8203d1995b33` | causal diagnosis / binding / self-model instrument research | `AN_RA_PROGRAM.md` |
| `cymek-500m-readiness` | `ba582b4dea3b3fae13d8b0ca6a2f87ff7243c7d0` | latest Cymek integration and real-GPU acquisition evidence | `docs/cymek/research/CROSS_BRANCH_EVIDENCE_AUDIT.md`, `docs/cymek/experiments/CYR-GPU-011/RESULT.md`, `DESIGN_REASONING.md` |
| `Arkenstone` | `c6acca77582069a4c1e0f50ef86cfb50b4637fae` | independent discovery, retention/recovery/invariance mechanisms | `docs/arkenstone/EXPERIMENT_LOG.md`, `MECHANISM_TOURNAMENT.md`, `COGNITION_BOTTLENECK_GRAPH.md`, `AGI_FEATURE_LEDGER.md` |

**Authority rule:** raw execution artifacts and their post-run audits beat older prose; newer documentation without execution does not create a scientific result. `cymek` remains the production authority, while Citadel, Triquetra, Arkenstone and `cymek-500m-readiness` provide evidence and challenger designs unless a result is explicitly promoted through the Cymek gate.

### Evidence labels used here

- **DEMONSTRATED** — executed evidence is strong enough for the exact scoped claim.
- **SUPPORTED** — useful executed evidence, but scope/replication/causal isolation is incomplete.
- **IMPLEMENTED** — code/plan exists and may be audited, but no scientific outcome exists yet.
- **SPECULATIVE** — plausible construction or mechanism requiring a decisive experiment.
- **REJECTED / CONTRADICTED** — the tested claim failed or a prior positive claim was invalidated.
- **NOT_DEMONSTRATED** — do not use the claim as fact.

---

# PART I — WHAT A SERIOUS AGI CONSTRUCTION ACTUALLY NEEDS

“Perfect AGI” is not a scientifically defined endpoint. For An-Ra, the useful target is a system that can **acquire reusable operations, generalize them under structural change, preserve them while learning more, diagnose failures, use tools/memory safely, and continue improving without evaluation leakage**.

A complete research stack therefore needs all of these layers:

```text
TRUSTWORTHY DATA
    ↓
REPRESENTATION / TOKENIZATION
    ↓
NEURAL CORE
    ↓
TRAINING OBJECTIVE
    ↓
OPTIMIZATION + CURRICULUM + REPLAY
    ↓
INTERNAL REPRESENTATIONS
    ↓
ADDRESSING / BINDING
    ↓
TRANSFORMATION / COMPOSITION
    ↓
GENERALIZATION + INVARIANCE
    ↓
RETENTION + RECOVERY + PLASTICITY
    ↓
MULTI-SKILL CONTINUAL LEARNING
    ↓
CAUSAL DIAGNOSIS / SELF-MODEL
    ↓
TOOLS + EXTERNAL MEMORY + VERIFICATION
    ↓
SEALED EVALUATION + FRESH REPLICATION
    ↓
PROMOTION / DURABILITY / SCALE
```

The Core and the complete AGI system are not the same thing. An-Ra’s current best architecture keeps the **Core conventional and causally inspectable**, while tools, permissions, durable memory, experiment routing, verification and promotion remain outside the Core until evidence shows a neural mechanism should be internalized.

---

# PART II — CURRENT BEST CONSTRUCTION

## 1. System architecture: separate learning, execution, measurement, and authority

### 1.1 Neural Core

**Current best baseline: dense decoder-only Transformer.** Do not add MoE, recurrence, SSM blocks, latent-thought heads, learned routers, separate cognition heads, or neural long-term-memory modules merely because they sound “AGI-like.” The strongest reason is experimental: if the baseline fails, a simple system tells us which data/objective/representation pressure was inadequate; architecture soup destroys causal attribution.

The current V5-A production candidate in `cymek` is:

| Component | Current candidate | Status |
|---|---:|---|
| family | dense causal decoder Transformer | **SUPPORTED baseline** |
| parameters | 250,216,960 | **PROVISIONAL; scale not proven** |
| layers × width | 26 × 896 | **PROVISIONAL** |
| Q / KV heads | 14 / 7 | **STRONG INFERENCE** |
| head dim | 64 | **STRONG INFERENCE** |
| FFN | 2,368 SwiGLU | **STRONG INFERENCE** |
| attention | full causal every layer | **PROVISIONAL** |
| native context | 4,096 | **NOT YET LEARNING-VALIDATED** |
| norm | pre-RMSNorm + final RMSNorm | **SUPPORTED family** |
| QK norm | affine per-head RMS normalization | **DEMONSTRATED scale-control mechanism locally; cognition benefit unproven** |
| position | RoPE base 10,000, positions 0–4095 | **LOCAL CONFORMANCE** |
| embedding/output | tied | **PROVISIONAL** |
| bias/dropout | no linear bias, dropout 0 | **STRONG INFERENCE** |

Initialization: embedding, Q/K/V, gate and up use `Normal(0, 0.02)`; attention-output and FFN-down use `Normal(0, 0.02/sqrt(2L))`. Local mechanism tests support residual-output scaling as a way to control depth-dependent residual/gradient growth, but that is not yet evidence that it improves cognition.

Precision: one persistent FP32 parameter set, BF16 autocast compute, FP32 logits/loss/global-gradient-norm reductions, FP32 Adam moments. Native BF16 optimizer state was rejected after clip-norm overshoot; the mixed layout is the current safe local choice, not yet a universal TPU law.

### 1.2 External runtime / Connector

Keep the runtime controller outside the Core:

```text
TASK
 → Core attempt
 → independent verifier
 → observable failure record
 → smallest legal intervention
 → retry / repair
 → evidence receipt
 → optional training proposal
```

The verifier is the only success authority. Hidden answer labels may be used by an evaluator, but they must never leak into diagnosis features, curriculum generation, or runtime policy. External repair can prove that a failure is recoverable; it does **not** prove the neural Core has internalized the missing computation.

### 1.3 Independent evaluator and promoter

Training code must not own its own success metric. Evaluation consumes immutable checkpoints, sealed/fresh fixtures, tokenizer identity and protocol hashes; promotion consumes evaluation receipts and durability receipts, never mutable training state or “latest checkpoint.”

This separation is one of the strongest pieces of the project and should survive every future architecture change.

---

## 2. Representation and tokenizer: now a first-order scientific bottleneck

### 2.1 Existing production candidate

Cymek’s current tokenizer contract is byte-level BPE with byte fallback, 24,576 entries, reserved `PAD=0, UNK=1, BOS=2, EOS=3`, zero expected UNKs, and no destructive normalization, case folding, whitespace rewrite, prefix-space insertion, or dropout. The original local tokenizer tournament put 24k between 16k and 32k in compression/parameter cost, but that result was only a planning prior because the local corpus was not representative.

### 2.2 New CYR-GPU-011 evidence changes priority

The latest completed Cymek-readiness experiment makes representation impossible to treat as a minor implementation choice:

- **COMPACT_BRIDGE**, real Cymek V5 4L/128w with the exact 19-symbol arithmetic representation, received only **44.89%** of the ARK-002B semantic exposure box yet reached M99 at update 1,400, sustained G50, and ended at **56.47% held-out STANDARD exact-with-valid-EOS**. Maximum controller exact was **59.38%**.
- **PRODUCTION_BRIDGE**, same V5 geometry, same task data, same Cymek causal objective, but frozen 24,576-token production representation, received **100%** of the ARK semantic exposure box: 1,152,000 row presentations / 18,000 updates. It reached train M99 but remained **0% held-out STANDARD**, **0/48 SEALED**, and never reached G50 or G90.

This is **DEMONSTRATED as a condition-specific divergence**, not as proof that “24k BPE is bad.” Compact-vs-production changes vocabulary size, segmentation, number atomization, embedding/output burden and correlated representation geometry together; the next causal study must factor these apart.

### 2.3 Construction decision

Do **not** replace the general-language tokenizer with a 19-symbol arithmetic alphabet. Keep a general byte-fallback tokenizer for language, but elevate representation to a primary experimental axis before an expensive scale run: vocabulary size, numeric/symbol segmentation, tied-output burden, and tokenization must be compared under matched model geometry, objective, semantic examples, and compute.

**Highest-information immediate closure:** extend/recreate only the CYR-GPU-011 compact bridge to the full 1,152,000 semantic-row exposure box. If compact reaches G90 while production stays at its already-observed 0%, representation becomes the dominant causal target; if compact also fails, isolate objective/initialization/optimizer/architecture differences next.

---

## 3. Data: quality, causal coverage and replay matter more than raw bytes

### 3.1 Production mixture candidate

Cymek’s current 5B-token candidate is:

| Slice | Share | Tokens | Evidence status |
|---|---:|---:|---|
| high-quality natural text | 65% | 3.25B | **PROVISIONAL** |
| code/math/formal/structured | 20% | 1.00B | **PROVISIONAL** |
| mechanically verified cognition | 15% | 0.75B | **PROVISIONAL** |

This is a useful starting hypothesis, not a discovered optimum. The cognition fraction should be tested against alternatives (the blueprint proposes 5% / 15% / 30% proxy comparisons), and every claim must be normalized for real tokenizer tokens and compute.

### 3.2 Data contract

Every source should carry origin, acquisition, license/terms class, raw hash, filtering version, exact/near-dedup cluster, contamination result, split, tokenizer hash and post-filter token count. Splits must be source/dedup-cluster disjoint before outcome inspection; train/dev/sealed leakage invalidates a capability result even if training was expensive.

Synthetic cognition is useful only when truth is executable. The preferred source is programs, solvers, simulators, formal transforms or databases; LLM paraphrases remain provenance-tagged and bounded, never treated as ground truth merely because they sound natural.

### 3.3 Cognition families worth training

The current best native cognition curriculum covers operations rather than benchmark names:

| Family | Current share inside cognition slice | Why it exists |
|---|---:|---|
| identity / exact copy | 8% | preserve arbitrary symbols and exact values |
| query-conditioned binding | 16% | let the current query select the right arbitrary binding |
| semantic-time state / precedence | 16% | use logical state rather than text position |
| interference-resistant retrieval | 10% | retrieve through distractors |
| relational composition | 20% | compose multi-step relations rather than retrieve endpoints |
| counterfactual sensitivity | 10% | change when relevant premises change |
| held-out-structure rule induction | 10% | infer/apply operations on novel structure |
| missing-information recognition | 5% | avoid unjustified guesses |
| faithful realization / format | 5% | emit the internally preferred answer correctly and terminate |

These weights are **not demonstrated optima**. The family definitions are better supported than the exact ratios.

### 3.4 Critical lesson: continued support can preserve a capability

ARK-015 is one of the most important data results in the entire program. Under canonical-only continuation, `NARROW_HIGH` lost robust order/query invariance in **8/8 matched pairs**, while `NARROW_LOW` lost it in **0/8** and `AUGMENTED_HIGH_REFERENCE` with continued presentation diversity lost it in **0/8**. Canonical accuracy stayed 1.0 even while broader invariance collapsed toward ~0.47 under narrow HIGH.

Therefore, forgetting/narrowing is not simply “the parameters moved too far.” High-plasticity training can preserve an invariant if the training stream continues to support it; current best hypothesis is an interaction between **plasticity/update scale × data support**.

### 3.5 Replay construction rule

When adding a new capability, preserve a small fixed replay/control stream from old capabilities and measure exact token fractions. ARK-013 showed that LOW LR alone did **not** solve prolonged cross-task interference without replay; every arm eventually lost T2 under pure new-skill training.

The exact replay fraction is **NOT_DEMONSTRATED**. Use controlled dose experiments, not folklore.

---

## 4. Objective: causal CE first, but train the behavior the evaluator requires

### 4.1 Current default

The clean baseline objective is causal cross-entropy over eligible target tokens, replica-global token mean, with label smoothing and z-loss zero. Query-swap auxiliary lambda and trace-loss lambda remain zero in the production candidate until a prospective experiment shows transfer beyond the task that trained it.

### 4.2 EOS/termination is a real training contract

Citadel T1D exposed a major measurement/training bug: **15,000/15,000 generations ended MAX_TOKENS** because EOS was used as a generation stop but was never a supervised target. That made content failure and termination failure inseparable; the corrected contract is `prompt + answer + EOS`, supervise answer content **and EOS**, and give a full-length answer one additional generation step to emit EOS.

This is not an AGI mechanism; it is a high-confidence mechanical rule. A system cannot be judged on a termination behavior it was never trained to produce.

### 4.3 Loss is not cognition

The founding negative result remains important: generic continuation improved held-out LM loss from **2.1884 → 1.9710** while preserved raw copy/context/binding/composition probes stayed at zero or chance. Citadel T1C/T1D repeated the broader lesson: large loss movement can occur while exact symbolic behavior remains almost unmoved.

Therefore use loss as a substrate metric, never as the success criterion for cognition.

### 4.4 Query-conditioned pressure remains a serious open objective question

Triquetra found raw query-conditioned candidate ranking near chance on the weak V4 substrate: raw rank-1 **25.0%** with chance 25%, QCS confidence interval including zero, and position effects around 19–35× larger than query-match effects. Correct-value recency could repair failures (~46–47%) while entity identity by itself repaired ~0%, which warns against casually calling a salience effect “addressing.”

A query-swap objective remains scientifically plausible because the current query often fails to control the answer representation, but it is **SPECULATIVE until a capable substrate, candidate-free transfer test and matched compute comparison show real gain**.

---

## 5. Optimization: acquire/recover and retain are different regimes

### 5.1 Safe baseline mechanics

Current V5 optimizer family:

```text
AdamW
β1=0.9, β2=0.95, eps=1e-8
weight_decay=0.1 on ndim>=2 tensors
no decay on ndim<2 norms/QK scales
global gradient clip = 1.0
```

The 5B V5 candidate schedule is token-indexed WSD: 0→3e-4 over first 50M tokens, stable 3e-4 until 4.5B, then decay to 3e-5 at 5B. These exact large-run constants are **NOT_DEMONSTRATED**; they remain a production hypothesis until executed and compared.

### 5.2 Demonstrated Micro retention law — scoped carefully

ARK-007R: after an already generalized T2 state, matched continuation at HIGH `1e-3` failed in **9/12 = 75%** forks while LOW `1e-5` failed in **0/12 = 0%**; risk difference LOW−HIGH = **−0.75**. This is strong Micro-T2 causal evidence that lower LR can protect an already-acquired capability under the tested stream.

ARK-010: after a prospectively defined instability event, continued HIGH recovered sustained G90 in **8/9 = 88.9%** states, while immediate LOW recovered in **2/9 = 22.2%**. Low LR therefore should not be used blindly when capability is absent.

ARK-011: after HIGH recovery, switching LOW reduced recurrent instability: HIGH recollapsed **3/6 = 50%**, SWITCH_LOW **0/6 = 0%**.

The best supported phase hypothesis is therefore:

```text
CAPABILITY ABSENT / NEED MOVEMENT
    → higher-plasticity acquisition or recovery
CAPABILITY CONFIRMED
    → lower-plasticity retention OR continued support/replay
CAPABILITY FALLS
    → recover, then protect again
```

But this is **not** a universal production scheduler. ARK-012 did not identify a unique numeric switch threshold; several behavioral thresholds aliased to the same switch time. ARK-016 also failed to isolate update-magnitude/trust-region mechanism because only 1/12 opportunities qualified.

### 5.3 What not to conclude

- “LOW LR is universally better” — false.
- “Large parameter movement causes forgetting” — unsupported; augmented HIGH moved farther and retained invariance in ARK-015.
- “Exact G90=.90 is a universal state threshold” — unsupported.
- “LR alone solves continual learning” — contradicted by ARK-013 no-replay interference.

---

## 6. Generalization and invariance: canonical accuracy is not enough

A model can score 100% on the narrow surface while losing the actual reusable operation. ARK-015 directly demonstrated this: canonical-only high-LR continuation retained canonical exact at 1.0 while order/query-order robustness collapsed, whereas LOW or continued augmentation preserved the broader capability.

Every important capability should therefore be tested on orthogonal axes:

```text
canonical
query-only change
order-only change
query + order change
relevant-fact intervention
irrelevant-fact intervention
structural OOD
natural/semi-natural analogue
```

Do not collapse these into one “reasoning score.” The latest CYR-GPU-011 post-run audit is another warning: a raw `COMMUTED=100%` flag was **not** accepted as commutation invariance because reversing operands also changed which operand occupied the held-out OOD role. A metric name is not evidence that the metric isolates what its name claims.

---

## 7. Capability formation is currently a more urgent Cymek problem than retention

### 7.1 Citadel low-budget evidence

T1D ran six arms across roughly 3.7–7.4M parameter models and 2–8M token budgets. Held-out exact remained **0–6.6%** across all arms, train exact **0–11%**, even though losses fell substantially. Curriculum, teacher, scale, masking and self-knowledge were therefore not sufficient for arithmetic lift-off at those tested budgets, though T1D had EOS and budget-confound limitations that prevent stronger universal conclusions.

Teacher primitives did move: held-out teacher microtask accuracy reached **51.5%** in the teacher arm while composed T2+ behavior stayed near zero. That is evidence for **primitive learning without compositional transfer**, not evidence that teacher data is useless.

### 7.2 CYR-GPU-009 and CYR-GPU-011 refined the diagnosis

CYR-GPU-009’s TINY model could saturate train probes but never reached candidate-free G90 under ~250k semantic row presentations; later audit showed it received <22% of the ARK-002B positive-reference semantic exposure and was underdosed for a strong negative conclusion.

CYR-GPU-011 corrected the exposure confound. It showed that **production representation can memorize but still produce 0% held-out exact even at the full 1,152,000-row reference box**, while compact representation entered a qualitatively different partially generalizing regime by 44.89% exposure.

**Current Cymek priority:** solve capability formation/representation before spending the primary compute budget on sophisticated retention controllers. Retention science matters, but there must first be a reliable acquired state to retain.

---

## 8. Causal diagnosis and self-modeling: keep the machinery, reject the false greens

Triquetra’s strongest contribution is not a successful learned self-model; it is the discipline for deciding when self-model claims are identifiable.

### 8.1 What worked

- Controlled entity×value factorials showed correct-value recency is a strong elicitation axis: DEV value-only repair **46.7%** vs neutral **3.7%**, replicated at **46.2%** on a second seed.
- Separate selection from realization; rank-1 and free generation can fail differently.
- Use paired interventions and answer-blind/legal observations.
- Require a checkpoint readiness gate before mechanism studies.

### 8.2 What failed

The original X1-REAL self-model “PASS” was invalid. Prospective intervention prediction accuracy 0.9545 sounded excellent, but an always-negative predictor scored **0.9733** because positive-cell prevalence was only 0.0267. The basis was not qualified and the claim was correctly marked contradicted.

The first readiness gate also produced a false green on the weak V4 checkpoint; v2 correctly downgraded it to `NOT_READY / INSUFFICIENT / NOT_IDENTIFIABLE`. This is a critical research lesson: **a self-diagnosis model built on a floor-limited substrate can look accurate because the evaluation basis is degenerate**.

### 8.3 Construction decision

Keep a causal-diagnosis layer, but do not train a neural “self-model” until:

```text
base capability is in an identifiable performance regime
→ intervention basis has adequate positive/negative coverage
→ simple constant baselines are beaten
→ prediction is prospective
→ hidden outcome labels cannot leak into features
→ result replicates on fresh tasks/checkpoints
```

Until then, use the external verifier/experimenter as instrumentation rather than pretending metacognition has been learned.

---

## 9. Evaluation: the project’s strongest reusable asset

The canonical cognitive decomposition is:

```text
REPRESENT → ADDRESS → TRANSFORM → CHOOSE → REALIZE
```

Retention/recovery and transfer should be added as longitudinal axes rather than hidden inside a single final score.

Primary scientific evaluation should be candidate-free generation with explicit valid termination. Candidate ranking is diagnostic only until the scorer itself is nuisance-resistant; Citadel demonstrated that sum/token/byte and calibrated DC-PMI/contextual-calibration policies were badly length/tokenization biased, with the calibrated policies selecting the fewest-token role **100% across 15 CUDA cells**.

A promotion-quality result needs:

1. frozen protocol before outcomes;
2. source/fact/template/structure-disjoint CONTROL and SEALED sets;
3. paired causal interventions;
4. cheap heuristic attacks before model interpretation;
5. at least two independent subjects/seeds for a strong claim;
6. candidate-free behavior, not only assisted ranking;
7. retention/regression checks on old skills and natural substrate;
8. fresh replication after sealed success;
9. exact code/model/tokenizer/data/fixture hashes and receipts;
10. a written alternative explanation and evidence that would change the conclusion.

**Rule:** if loss improves and behavior does not, behavior wins. If a post-run audit shows a probe was confounded, the interpretation is downgraded even if the numeric score looks impressive.

---

## 10. Checkpointing, durability and reproducibility

Cymek’s production-engineering direction is sound and should be preserved:

```text
completed optimizer update
→ collective snapshot
→ immutable temporary generation
→ hash + structural validation
→ publish recovery/milestone pointer
→ durable upload
→ clean redownload
→ clean restore canary
→ only then mark durable
```

Full-resume state must include FP32 parameters, FP32 Adam moments/step, token-indexed schedule state, exact source/family token ledger, sampler/supercycle cursor, all RNG states, topology, model/tokenizer/data/pack/source/code identities. Resuming must not rewarm LR, double-advance token counters, or reconstruct sampler position from wall time.

Local checkpoint/transaction/corruption/cursor canaries are strong. TPU PRE50M-style smoke has also validated important pieces of update/resume/writer-fence behavior, but there is still **no scientific evidence that the complete 500M campaign recipe is ready or good**.

---

## 11. Scale: engineering readiness is not scientific readiness

Cymek production plumbing is substantially stronger than the scientific capability evidence. Citadel’s production-path audit found packing, sampler/cursor, batch assembly, model path, causal loss, optimizer, TrainingState, checkpoint transaction and restore/continuation connected; historical blockers included production corpus materialization, top-level entry, frozen tokenizer identity, canonical schedule execution and wired milestone evaluation.

The `cymek-500m-readiness` branch implemented major missing plumbing, but **data readiness and scientific authorization remain separate gates**. The latest CYR-GPU-011 result explicitly leaves PRE500M and 500M unauthorized because the current production representation failed the controlled acquisition bridge.

Do not use “I can run the compute” as evidence that the run is scientifically justified. A 500M-token run should happen only when the smaller bridge tells us the model can form the target capability and the data/tokenizer/schedule/evaluation identities are frozen and verified.

---

# PART III — THE INTEGRATED TRAINING ALGORITHM CANDIDATE

The following is the best current **research algorithm**, not a proven AGI algorithm.

```text
INPUTS
  immutable data/source manifests
  frozen tokenizer + model + objective + optimizer identities
  frozen development/control/sealed evaluation contracts

INITIALIZE
  conventional dense Core
  verified parameter count / tied weights / precision layout
  exact optimizer ownership and resume state

FOR each training phase:
  1. Train on deterministic, provenance-bound mixed data.
  2. Count actual non-padding target/input tokens exactly.
  3. At fixed token milestones, evaluate candidate-free:
       - substrate LM loss
       - representation
       - addressing/query sensitivity
       - transformation/composition
       - realization + EOS termination
       - invariance under irrelevant/order/surface changes
       - old-skill retention
       - new-skill acquisition
  4. Never select a mechanism from sealed outcomes.
  5. If capability is absent:
       keep acquisition/plasticity high enough to learn;
       do not freeze it merely to preserve a capability it does not possess.
  6. Once a capability is prospectively confirmed:
       compare retention controls under matched future data:
         HIGH / LOW / fixed-time / state-dependent policy / replay-support controls.
  7. When adding a new skill:
       keep an explicitly measured replay/support stream for old skills;
       report old and new capability separately.
  8. If capability falls:
       diagnose whether failure is representation, addressing,
       transformation, realization, data-support narrowing or optimization;
       apply the smallest intervention that distinguishes hypotheses.
  9. Checkpoint only after a completed update and publish only after clean restore.
 10. Choose a development checkpoint behaviorally, not because it is final.

PROMOTION
  one chosen immutable checkpoint
  → sealed evaluation once
  → source/family-disjoint fresh replication
  → natural-transfer check
  → regression/retention check
  → integrity/durability check
  → promote only if all conjunctive gates pass
```

The future **Capability Guardian** idea—reactively changing protection/replay based on CONTROL probes—is scientifically attractive, but ARK-019 is not implemented/executed and remains blocked on ARK-017/018. Do not silently install it into production before the causal mechanism and real-data transfer tests run.

---

# PART IV — EVIDENCE MATURITY PERCENTAGES

These percentages are **not “percent to AGI” and not probability of success**. They are a transparent engineering/research maturity estimate using this rubric: 0 = idea only, 25 = implemented/unexecuted, 50 = executed in one narrow regime, 75 = replicated and/or transferred across controlled regimes, 100 = replicated on real data and target-scale production conditions with no unresolved major confound.

| Subsystem | Evidence maturity | Why |
|---|---:|---|
| experiment integrity / receipts / fail-closed contracts | **90%** | strong hashes, canaries, preregistration, negative-result preservation; still not full production custody at scale |
| causal evaluation methodology | **85%** | rich decomposition, anti-shortcut, sealed/fresh design; scorer policy remains unresolved |
| checkpoint/resume/durability mechanics | **80%** | strong local + TPU canaries; full remote production campaign not demonstrated |
| Core architecture mechanics | **65%** | conventional design + local QK/init/precision evidence; exact V5 learning benefit unproven |
| data governance / provenance design | **70%** | strong contracts and implementation; production-quality 5B corpus not yet qualified |
| actual production corpus readiness | **20%** | pipeline exists, but complete campaign supply/qualification is not demonstrated |
| representation/tokenizer scientific understanding | **35%** | CYR-011 exposed a major divergence but causal factor is not isolated |
| objective design | **45%** | CE mechanics + EOS contract solid; query-conditioned/compositional objective pressure unresolved |
| capability formation/generalization | **45%** | replicated Micro transitions exist; Cymek production representation currently fails controlled heldout lift-off |
| retention/recovery under same-skill stress | **75% Micro / ~20% production-transfer** | replicated Micro T2 + non-arithmetic narrowing effects; mechanism and scale transfer unresolved |
| multi-skill continual learning | **20%** | ARK-013 failed to qualify new skill; Guardian not executed |
| causal self-diagnosis / learned self-model | **15%** | instrumentation improved, but major positive self-model claim was invalidated |
| real-text representation/retention transfer | **10%** | ARK-018 implemented but not executed |
| target-scale 500M/5B scientific readiness | **20%** | engineering path advanced, but capability/data/representation gates remain open |
| demonstrated AGI | **0%** | no current result supports a general AGI claim |

A rough weighted program maturity for **doing credible AGI research** is much higher than the maturity of the AGI capability itself. The repository is becoming good at falsifying itself; the neural Core is still far from demonstrating broad continual general intelligence.

---

# PART V — THE MOST IMPORTANT NUMBERS WE ACTUALLY KNOW

| Finding | Number | Honest interpretation |
|---|---:|---|
| generic continuation LM loss | 2.1884 → 1.9710 | better loss did not produce tested cognition |
| Citadel T1D heldout arithmetic exact | 0–6.6% | no lift-off across tested low-budget arms; confounds recorded |
| T1D teacher primitive heldout | 51.5% | primitives learnable without composition transfer |
| Triquetra value-only repair | 46.7% DEV; 46.2% replication | value recency strongly elicits behavior on weak V4 |
| Triquetra raw query-conditioned rank | 25.0% at 25% chance | measurable query control approximately absent on that substrate |
| ARK-007R HIGH retention failure | 9/12 = 75% | high LR unstable after acquired Micro-T2 state |
| ARK-007R LOW retention failure | 0/12 = 0% | low LR protective in that matched Micro-T2 regime |
| ARK-010 HIGH post-collapse recovery | 8/9 = 88.9% | movement/plasticity helps reacquisition |
| ARK-010 immediate LOW recovery | 2/9 = 22.2% | freezing too early can prevent recovery |
| ARK-011 HIGH recurrent collapse | 3/6 = 50% | recovered state can destabilize again |
| ARK-011 HIGH→LOW recurrent collapse | 0/6 = 0% | post-recovery low switch protected in Micro T2 |
| ARK-015 narrow HIGH invariance failure | 8/8 = 100% | high plasticity + narrow support eroded broader capability |
| ARK-015 narrow LOW failure | 0/8 = 0% | low LR protected |
| ARK-015 augmented HIGH failure | 0/8 = 0% | continued diverse support also protected despite large movement |
| CYR-011 compact exposure | 44.89% | compact bridge timeboxed early |
| CYR-011 compact heldout STANDARD | 56.47% | partial generalization, not G90 |
| CYR-011 production exposure | 100% | full ARK reference semantic dose |
| CYR-011 production heldout STANDARD | 0% | memorization without structural heldout generalization |
| CYR-011 production SEALED | 0/48 = 0% | same negative on reserved measurement |

---

# PART VI — WHERE WE FAILED, IN SMALL FORM

| Failure | What it taught us | Status now |
|---|---|---|
| lower LM loss without cognitive movement | loss cannot be the promotion metric | **DEMONSTRATED negative** |
| candidate scorers dominated by token/length bias | scoring instrument must be certified before model conclusions | **DEMONSTRATED failure** |
| T1D/T1C arithmetic did not lift off | low-budget CE can learn distribution/format without exact computation | **executed negative; T1D had known confounds** |
| EOS never trained in T1D | termination and content were conflated | **contract repaired conceptually/elsewhere** |
| curriculum did not accelerate Micro T2 | easy→hard staging is not automatically helpful | **REJECTED in tested Micro regime** |
| teacher rows did not create composition | primitives can improve while composition stays absent | **null for sufficient composition at tested dose** |
| selective-binding headline failed independent replication | format familiarity was mistaken for robust selection | **REJECTED** |
| entity-addressing interpretation | value recency, not entity identity, explained much of the repair | **reattributed** |
| X1 self-model “PASS” | class imbalance let a trivial predictor win | **CONTRADICTED** |
| readiness gate v1 | floor substrate produced a false green | **CONTRADICTED; v2 replaces it** |
| LOW LR as universal cure | failed under cross-task no-replay interference and post-collapse recovery | **REJECTED as universal** |
| exact LR switch threshold | time/state aliasing prevented identification | **NOT_IDENTIFIED** |
| update-cap mechanism | event rate too low to assign causal credit | **INCONCLUSIVE** |
| native BF16 optimizer state | clip-norm invariant violated | **REJECTED locally** |
| CYR-GPU-006 | hardware feasibility gate correctly stopped before science | **non-scientific preexecution failure** |
| CYR-GPU-009 | semantic exposure <22% of ARK reference | **valid narrow null, insufficient for strong acquisition conclusion** |
| CYR-GPU-010 | batch-16 design would still underdose semantic rows | **superseded before execution** |
| CYR-GPU-011 production representation | M99 but 0% heldout at full ARK exposure | **current major scientific bottleneck** |
| ARK-017 | causal mechanism test not run | **IMPLEMENTED / NOT EXECUTED** |
| ARK-018 | real-text science+Birth study not run | **IMPLEMENTED / NOT EXECUTED** |
| ARK-019 | closed-loop Guardian | **blocked / not implemented or executed** |

Failures are not waste; they remove bad explanations. But a failed experiment with a confound does not justify a universal negative claim.

---

# PART VII — WHAT TO BUILD NEXT, IN ORDER

## Gate A — close Cymek capability formation

**Do first:** complete the compact CYR-GPU-011 bridge to 1,152,000 semantic row presentations with corrected structural probes. This is cheaper and more informative than another full campaign because production already completed the reference exposure and stayed at zero heldout.

Decision:

```text
compact reaches G90
    → representation burden becomes primary target
    → run representation factorial
compact does not reach G90 at full exposure
    → isolate objective / init / optimizer / architecture differences
```

## Gate B — isolate representation rather than changing everything

Prospective factorial should keep model geometry, semantic examples, objective, optimizer and exposure fixed while separately testing:

```text
vocabulary size
BPE segmentation / number-symbol atomization
embedding/output parameter burden
tied vs controlled output burden if justified
byte/character fallback behavior
```

Do not infer a general-language tokenizer from arithmetic alone.

## Gate C — run ARK-017 mechanism credit

ARK-017 is designed to distinguish update magnitude from sparse invariant-support replay and their interaction using a known high-event failure generator. It should run before claiming why LOW or augmentation protects capability.

## Gate D — run ARK-018 real-data bridge

ARK-018 is implemented and preregistered to train a ~20–25M conventional decoder primarily on the exact peS2o scientific shard with periodic Birth Book vs token-matched small-science control exposures. Its result can establish content internalization and real-text substrate behavior; **Birth content learning is not evidence of identity, consciousness, reasoning or AGI**.

## Gate E — only then test a closed-loop Capability Guardian

If a protection mechanism survives ARK-017 and behaves on a real-text-trained substrate, ARK-019 can ask whether a controller preserves SKILL_A while SKILL_B and real-text learning continue. The controller must beat static LOW and static protection while retaining new-skill plasticity; otherwise it is merely a complicated freeze mechanism.

## Gate F — PRE500M / 500M only after scientific and data gates

Before large training:

```text
qualified production corpus + exact manifests
frozen tokenizer identity
one production entry point
executed token-indexed schedule
candidate-free milestone evaluation
exact resume/durability
capability formation demonstrated on the relevant representation
M102 / multi-seed replication when required
explicit PRE500M green decision
```

Only then is a 500M-token campaign a useful scale experiment rather than an expensive attempt to make an unresolved small-scale problem larger.

---

# PART VIII — CURRENT BEST AGI RESEARCH HYPOTHESES

## H1 — representation burden can change whether structural capability emerges

**Status: SUPPORTED / causal factor unresolved.** CYR-011 is the strongest evidence: compact partially generalized under <45% reference exposure while production stayed at 0% under 100%. The causal unit is not yet “vocabulary size”; it is the bundle of representation changes.

## H2 — capability emergence can be delayed far beyond memorization

**Status: DEMONSTRATED at Micro symbolic scale.** ARK-002B replicated the qualitative memorize-first → delayed-generalize transition with large seed variance. This means stopping immediately after train saturation can miss a later structural transition.

## H3 — acquired capability can narrow without canonical accuracy falling

**Status: DEMONSTRATED at controlled Micro non-arithmetic scale.** ARK-015 retained canonical exact while order/query-order invariance eroded under narrow high-plasticity continuation.

## H4 — retention is an interaction between plasticity and ongoing support

**Status: strongest current mechanism hypothesis, not fully isolated.** LOW protected; continued augmentation at HIGH also protected despite much larger path length. ARK-017 is the causal-credit experiment.

## H5 — acquisition/recovery and retention need different control regimes

**Status: SUPPORTED at Micro T2.** HIGH helps recover absent capability, LOW protects present capability, and HIGH→LOW after recovery reduces recurrence. Exact state thresholds and scale transfer are unresolved.

## H6 — query-conditioned addressing is a real missing operation, but current weak-substrate evidence cannot justify a learned self-model

**Status: SUPPORTED as a bottleneck hypothesis / self-model NOT_DEMONSTRATED.** Triquetra measured chance-level query ranking and strong recency effects, while its self-model basis failed qualification.

## H7 — primitives do not automatically compose

**Status: SUPPORTED.** T1D teacher primitives learned to ~51.5% while full arithmetic composition stayed near floor. Training should measure composition explicitly rather than assuming it emerges from microtask mastery.

## H8 — better data/measurement should be tested before exotic architecture

**Status: STRONG INFERENCE.** No current evidence shows MoE, recurrence, SSM, latent thought or neural long-term memory is the binding bottleneck. Adding them before resolving representation, objective, data support and capability formation would reduce interpretability.

---

# PART IX — NON-NEGOTIABLE RESEARCH LAWS

1. **Behavior beats loss.** Loss is substrate evidence, not a cognition verdict.
2. **Execution artifacts beat prose.** A plan, README or newer commit is not a result.
3. **Preregister before outcome-sensitive execution.** Changing a metric after seeing the result creates a new exploratory claim.
4. **Candidate-free behavior is primary.** Assisted scoring is diagnostic unless its own bias screen is qualified.
5. **Train the behavior you evaluate.** If EOS terminates generation, EOS must be part of the training contract.
6. **Keep causal variables separate.** Do not call a combined query+order change “query sensitivity.”
7. **Shared-parent matched forks are the default for retention experiments.** Same bytes, same future stream, one treatment variable.
8. **A strong claim needs replication.** One seed can generate a hypothesis, not a universal law.
9. **Keep CONTROL and SEALED firewalled.** Sealed results cannot become training data or tuning feedback.
10. **Preserve negative results and false greens.** They are part of the system’s anti-self-deception memory.
11. **Do not promote from aggregate accuracy.** Inspect worst family, invariance, sensitivity, transfer, realization and retention.
12. **Scale only after the smaller experiment identifies what should scale.** Compute is not a substitute for causal clarity.
13. **No architecture soup.** Add one mechanism when a measured bottleneck demands it.
14. **No AGI/consciousness/identity claims from narrow task competence or Birth Book internalization.** Those conclusions require entirely different evidence.

---

# PART X — ONE-PAGE AGENT HANDOFF

If an agent reads only this section, it should act as follows:

**Core:** preserve the conventional Cymek V5 design as the production control; do not add exotic neural modules. **Immediate science:** solve the representation/capability-formation bottleneck exposed by CYR-GPU-011 before retention or scale work.

**Training:** use exact provenance-bound data, deterministic packing, causal CE with answer+EOS supervision, AdamW with verified precision/ownership, token-indexed schedules, and candidate-free evaluations. Treat the 65/20/15 data mix, 24,576 tokenizer, exact WSD schedule and 250M scale as candidate constants, not discovered laws.

**Cognition:** train reusable operations—copy, binding, semantic state, interference-resistant retrieval, composition, counterfactual sensitivity, rule induction, missing-information restraint and faithful realization. Measure each operation under relevant and irrelevant interventions, structural OOD, natural analogues and retention stress.

**Continual learning:** when a capability exists, Micro evidence supports lower LR and/or continued capability-supporting data as retention controls; when capability is absent, HIGH can recover better than immediate LOW. Never call this universal; exact thresholds, mechanism credit, multi-skill plasticity and scale transfer remain open.

**Diagnosis:** keep Triquetra-style causal instrumentation and the independent verifier, but do not claim learned self-modeling until a nondegenerate intervention basis prospectively beats trivial baselines on a capable substrate. A false green is a failed instrument, not intelligence.

**Scale:** no 500M/5B scientific launch is justified merely because the engineering can run it. First close compact exposure, representation factorization, real-data transfer, protection mechanism and data/evaluation identities; then use the smallest scale ladder that can falsify the next hypothesis.

**North-star loop:**  
`BUILD → MEASURE → UNDERSTAND → DIAGNOSE → PREDICT → INTERVENE → VERIFY → INTERNALIZE → SCALE`, while Arkenstone’s discovery loop remains `BUILD → MEASURE → UNDERSTAND → IMPROVE → VERIFY NOVELTY`.

---

## Final state

The project’s strongest result is **not “we have AGI.”** The strongest position is that An-Ra now has a fairly rigorous experimental framework plus several real, narrow discoveries: delayed generalization after memorization, state-dependent acquisition/recovery/retention behavior, non-arithmetic invariance narrowing, and a newly exposed representation-dependent capability-formation gap in real Cymek V5.

The next breakthrough is most likely to come from **causally isolating why representation changes lift-off, then combining that with a tested stability–plasticity/data-support mechanism on a real-data-trained substrate**. If those effects survive fresh replication and scale, only then should they become part of the production Core or training controller.
