# AN-RA MASTER AGI CONSTRUCTION KNOWLEDGE

**Status:** CROSS-BRANCH SYNTHESIS / RESEARCH AUTHORITY MAP / NOT A CLAIM OF AGI  
**Synthesis date:** 2026-09-13 (supersedes the 2026-09-10 synthesis; audited against primary artifacts on all 14 live branches)  
**Machine-readable evidence:** [`docs/research/EXPERIMENT_EVIDENCE_LEDGER.json`](../../research/EXPERIMENT_EVIDENCE_LEDGER.json) · human ledger: [`EXPERIMENT_EVIDENCE_LEDGER.md`](../../research/EXPERIMENT_EVIDENCE_LEDGER.md) · negatives: [`NEGATIVE_RESULTS_LEDGER.md`](../../research/NEGATIVE_RESULTS_LEDGER.md) · causal graph: [`CAUSAL_KNOWLEDGE_GRAPH.md`](../../research/CAUSAL_KNOWLEDGE_GRAPH.md) · decisions: [`NEXT_PHASE_DECISION_MEMO.md`](../../research/NEXT_PHASE_DECISION_MEMO.md)  
**Purpose:** give a human or autonomous research agent one file that explains what An-Ra currently knows about building a stronger general-learning Core, what construction is justified now, what is only a hypothesis, what failed, what remains unknown, and what evidence must exist before scaling.

> **Two-sentence definition of this file:** This is the shortest honest path from all current An-Ra evidence to a buildable research system: a conventional neural Core trained on auditable data, measured with causal/anti-shortcut evaluations, and surrounded by a fail-closed training, verification, retention, diagnosis, and promotion loop. It does **not** say we know how to build “perfect AGI”; it says exactly which pieces are demonstrated, which are implemented but unexecuted, which are speculative, and which experiment should change the design next.

---

## 0. Evidence snapshot and authority

This synthesis was audited against the live heads of all fourteen origin branches on 2026-09-13, plus git-forensics over unreachable history. Raw artifact receipts and post-run audits outrank prose; per-experiment provenance lives in the evidence ledger.

| Branch | Live head (2026-09-13) | Role | Highest-value evidence |
|---|---|---|---|
| `cymek-500m-readiness` | `f2c27a6` | Cymek V5 production core + controlled GPU campaigns | `docs/cymek/experiments/CYR-GPU-011..014-R1C/*`, `docs/cymek/research/*` |
| `Arkenstone` | `4ae9e3b` | Discovery program (ARK-001…022, Guardian/continual line) | `docs/arkenstone/CURRENT_STATE.md`, `experiments/ARK-017/RESULT_V2.md`, `experiments/ARK-018/FINAL_RESULT_AUDIT.md`, `experiments/ARK-019/FINAL_RESULT_AUDIT_V3.md` |
| `arkenstone-ark020-v4` | `65d1ef3` | ARK-020 V4 execution branch + durability amendments A1.1–A1.3 (no results) | `experiments/ARK-020-V4/ENGINEERING_AMENDMENT_A1.md` |
| `codex/arkenstone-improvements` | `1511321` | Runtime integrity + independent ARK-014 replication with raw receipts | `artifacts/arkenstone/ark014/ark014-cuda-2201-03/ARK-014_RESULT.json` |
| `arkenstone-astra` | `ccb84fb` | BRAMASTRA + research-environments code (no results) | — |
| `BRAMASTRA` | `02b94d3` | Discovery/binding lab + its own cross-branch audit | `docs/bramastra/RESULTS.md`, `docs/bramastra/EVIDENCE.md`, `artifacts/bramastra/*` |
| `triquetra` | `f23f0af` | Cognition laboratory (V4-substrate diagnostics); WAITING_FOR_STRONGER_CHECKPOINT | `AN_RA_PROGRAM.md`, `output/*` |
| `citadel` | `1d27f9b` | Independent auditor: T1-series, scorer tournament, 500M path audit, negatives registry | `docs/citadel/EVIDENCE_LEDGER.md`, `NEGATIVE_RESULTS.md`, `experiments/T1D/RESULTS.md`, `500M/PRODUCTION_PATH_AUDIT.md` |
| `esoes` | `85f44b7` | TPU-era V5 blueprint + founding negative + mechanism canaries (D-024…D-028) | `docs/esoes/EVIDENCE_AND_CONTEXT.md` |
| `core-exp` | `51124de` | Historical V4-era self-model/policy line (unre-audited) + milestone 0001 | commit `20d8841` / tag `milestone/0001-honest-loop` |
| `core-frozen-v4` | `f72f193` | Frozen inference-only V4 core (32,768-token tokenizer) | README |
| `main` | `b620f1c` | Frozen V4 research system (2026-08-15) | — |
| `iterate500` / `iterate900` | `b438420` / `6fbd2c0` | Historical TPU/SFT engineering lineages | — |
| *(deleted)* `senora` | unreachable `30a8fa7` | Entire P35-CMS-1 + CAD program survives ONLY in unreachable commits | EVIDENCE_GAPS.md |

**History warning:** the repository has two disconnected shards; the cymek-500m-readiness/Arkenstone shard is rooted at a parentless squash `28bf57a` (2026-09-05) and the local branch `cymek` (`4abeaeb`) is the only bridge to pre-2026-09-05 history. Commit `6653b4ce` (the R1C audited executable chain head) is an ancestor of **no** live branch and is **not served by GitHub** — the R1C launcher currently binds it and cannot launch from a fresh clone until rebound.

### 0.1 Corrections to the 2026-09-10 synthesis (changelog)

1. **ARK-017 V2 is EXECUTED** (was "implemented/not executed"): verdict `BOTH_LEVERS_SUFFICIENT` — HIGH failed 4/6; LOW, CAP1X, exact-noncanonical 1/16 replay, CAP+replay and augmented-HIGH each 0/6; secondary one-order screens CAP4X/CAP16X/replay-1/32/replay-1/64 all 0/3.
2. **ARK-018 V4 is EXECUTED and audited**: Birth-content internalization threshold **NOT MET** (+0.000/+0.067 vs required +0.10, both seeds); SEALED science NLL +3.68%/+3.91% vs matched science replay; replicated temporary-binding acquisition slowdown (1200/>1500 vs 300/300 steps).
3. **ARK-019 V3.1 is EXECUTED**: official verdict **`CONTROLLER_NOT_SUPPORTED`** — the required new capability never formed in any arm (new-skill-formation bottleneck); PLASTIC_HIGH destroyed old skill 4/4 (robust-min ≈0.005), STATIC_REPLAY_1OF64 ≈0.882, Guardian variants ≈0.963 (recovery, not prevention); CAP16X never triggered. ARK-019 V4 carries a **transcribed, externally audited** `GUARDIAN_CONTINUAL_PROXY_CANDIDATE` result whose raw bundle is NOT in the repository (byte re-audit required before it becomes DEMONSTRATED).
4. **ARK-020 V4 remains NOT EXECUTED** in-repo (V1→V4 repair chain: 5 defects → 4 blockers → 7 blockers, 39/39 tests, A1.1–A1.3 durability amendments whose own audit found the repo readiness claim stronger than the executable evidence). The mission-brief belief in a completed V4 result was independently audited as absent.
5. **CYR-GPU-013/R1B is EXECUTED** (was "ready, not executed"): replicated six-level class-space response curve, verdict `MIXED_OR_SEED_SENSITIVE_RESPONSE_CURVE`.
6. **CYR-GPU-012/R1 and 013/R1B change the representation question**: with active token IDs fixed, declared tied class-space size alone moves held-out formation (V4096 100% vs V24576 0% at 512k rows; intermediate 4096–16384 region strongest across two fresh seeds). The effect is non-monotonic and seed-sensitive; `smaller vocabulary is better` and `more parameters are better` are both falsified.
7. **CYR-GPU-014/R1C is frozen and twice engineering-repaired** (optimizer-constructor TypeError; CLIP_BREACH float32 reduction-order 1.0000042915 > 1.0 at 1e-6 tolerance), status `R1C_READY_FOR_OPERATOR_CUDA_RUN`, scientific `NOT_EXECUTED` — **and its launcher pins a commit absent from origin (launch-blocking; see EVIDENCE_GAPS).**
8. **BRAMASTRA is live again** (2026-09-12): EOS contract experiment (0/32→32/32 ×2 seeds), query-blind binding null (48.4% ≈ 50% query-blind baseline), discovery-controller null, D02 depth-two null, plus its own independent cross-branch audit (which corrected T1C prose 0/500 → raw 0/1,000).

**Authority rule:** raw execution artifacts and their post-run audits beat older prose; newer documentation without execution does not create a scientific result. `cymek-500m-readiness` carries the production Core contracts; Citadel, Triquetra, Arkenstone, BRAMASTRA and the codex branch provide evidence and challenger designs unless a result is explicitly promoted through the Cymek gate.

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

### 2.2 The representation evidence chain (CYR-GPU-011 → 012 → 013 → 014)

The latest completed Cymek-readiness experiments make representation impossible to treat as a minor implementation choice:

- **CYR-GPU-011 (DEMONSTRATED):** **COMPACT_BRIDGE**, real Cymek V5 4L/128w with the exact 19-symbol arithmetic representation, received only **44.89%** of the ARK-002B semantic exposure box yet reached M99 at update 1,400, sustained G50, and ended at **56.47%** held-out STANDARD exact-with-valid-EOS (maximum controller exact 59.38%). **PRODUCTION_BRIDGE**, same V5 geometry/task/objective with the frozen 24,576-token production representation, received **100%** of the exposure box (1,152,000 rows / 18,000 updates): train M99 but **0% held-out STANDARD**, **0/48 SEALED**, never G50 or G90.
- **CYR-GPU-012/R1 (DEMONSTRATED):** with the active arithmetic token IDs, segmentation, data, geometry, optimizer and seeds fixed, changing only the **declared tied embedding/output class-space size** produced V19 **12.94%** / **V4096 100%** / V24576 **0%** at the 512k-row endpoint. Non-monotonic; parameter displacement does not explain capability (V4096 moved farther than V19).
- **CYR-GPU-013/R1B (SUPPORTED, R1):** two fresh matched seeds × six levels at 128k rows: V19/V1024/V24576 ≈ 0; V4096 0.506/0.494 (most stable); V8192 0.718/0.0 and V16384 0.647/0.129 (seed-sensitive). Verdict `MIXED_OR_SEED_SENSITIVE_RESPONSE_CURVE`: the intermediate 4096–16384 region is the reproducible developmental regime; the exact optimum is not identified.
- **CYR-GPU-014/R1C (IMPLEMENTED_NOT_EXECUTED):** the causal follow-up — keep the physical 24,576-row matrix in every arm and manipulate only training-time softmax participation (MASK_19/4096/8192/16384, OFFSET_EQ4096 vs FULL_24576; 4 seeds × 6 arms). It is frozen, twice engineering-repaired (optimizer-constructor API; CLIP_BREACH float32 reduction-order tolerance), and **launch-blocking**: the operator notebook pins commit `6653b4ce`, which exists on no branch and is not served by GitHub.

Falsified simple stories: `smaller vocabulary → better capability` and `more classes/parameters → better capability`. A post-run audit also downgraded CYR-011's `COMMUTED=100%` flag: reversing operands changed the OOD tens-band role, so it is operand-role asymmetry evidence, not proven commutation invariance.

### 2.3 Construction decision

Do **not** replace the general-language tokenizer with a 19-symbol arithmetic alphabet, and do **not** change the production tokenizer yet. Keep a general byte-fallback tokenizer for language. Representation is now the primary experimental axis before any expensive scale run: the single highest-information pending experiment is **R1C**, which partitions inactive-softmax competition from tied-matrix size while preserving full-vocabulary evaluation. If R1C shows the MASK_4096 arm rescues formation on the fixed 24,576 matrix, a softmax-partition mechanism (e.g. inactive-row exclusion/regularization or untied low-rank output) becomes a justified production-representation change candidate; if not, the search moves to tied initialization/optimization geometry. A positive R1B-style intermediate result alone does **not** authorize a production vocabulary change.

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

### 7.2 CYR-GPU-009/011/012/013 refined the diagnosis

CYR-GPU-009's TINY model could saturate train probes but never reached candidate-free G90 under ~250k semantic row presentations; the later audit showed it received <22% of the ARK-002B positive-reference semantic exposure and was underdosed for a strong negative conclusion (CYR-GPU-010, which would have repeated that dose, was superseded pre-execution).

CYR-GPU-011 corrected the exposure confound and showed that **production representation can memorize but still produce 0% held-out exact even at the full 1,152,000-row reference box**, while compact representation entered a qualitatively different partially generalizing regime by 44.89% exposure. CYR-GPU-012/013 then isolated the surprising core: the effect tracks the **declared tied class-space size** (non-monotonically, seed-sensitively), not compact-vs-production tokenization per se.

**Current Cymek priority:** execute the R1C softmax-competition mechanism dissection (after rebinding its launcher to a pushed commit) before spending the primary compute budget on sophisticated retention controllers or scale. Retention science matters, but there must first be a reliable acquired state to retain.

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
| experiment integrity / receipts / fail-closed contracts | **90%** | strong hashes, canaries, preregistration, negative-result preservation; receipt meta-checks currently STALE-BY-DESIGN after the R1C constant consolidation; launcher-provenance gap found by audit |
| causal evaluation methodology | **85%** | rich decomposition, anti-shortcut, sealed/fresh design; scorer policy remains unresolved (`production_scoring_mode: null`) |
| checkpoint/resume/durability mechanics | **80%** | strong local + TPU canaries; ARK-020-V4 A1 amendments show resume edge cases still surface on real hardware |
| Core architecture mechanics | **65%** | conventional design + local QK/init/precision evidence; exact V5 learning benefit unproven |
| data governance / provenance design | **70%** | strong contracts and implementation; production-quality 5B corpus not yet qualified |
| actual production corpus readiness | **20%** | pipeline exists, but complete campaign supply/qualification is not demonstrated |
| representation/tokenizer scientific understanding | **55%** | non-monotonic class-space effect isolated (R1, R1 replicated directionally in R1B) but seed-sensitive and mechanism untested (R1C pending) |
| objective design | **45%** | CE mechanics + EOS contract solid cross-program; query-conditioned/compositional objective pressure unresolved |
| capability formation/generalization | **55%** | delayed transition replicated; class-space response curve mapped at development scale; production-representation lift-off still unsolved |
| retention/recovery under same-skill stress | **75% Micro / ~20% production-transfer** | replicated Micro T2 levers + ARK-017 both-levers result; mechanism and scale transfer unresolved |
| multi-skill continual learning | **15%** | ARK-019 V3.1 controller NOT supported (formation bottleneck); V4 candidate result transcribed-only; ARK-020 unexecuted |
| causal self-diagnosis / learned self-model | **15%** | instrumentation improved, but major positive self-model claim was invalidated and no qualified subject exists |
| real-text representation/retention transfer | **45%** | ARK-018 V4 executed and audited (internalization threshold not met; plasticity cost replicated); ARK-019 used its parents |
| target-scale 500M/5B scientific readiness | **20–25%** | engineering path advanced ~236 commits since the citadel audit, but capability/data/representation gates remain open |
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
| CYR-012/R1 class-space endpoint (512k rows) | V19 12.94% / V4096 100% / V24576 0% | declared class-space size alone moves formation; non-monotonic |
| CYR-013/R1B response curve (128k rows, 2 seeds) | V4096 0.506/0.494; V8192 0.718/0.0; V16384 0.647/0.129; extremes ≈ 0 | intermediate regime reproducible, amplitude seed-sensitive |
| ARK-017 V2 primary screen | HIGH 4/6 fail; LOW/CAP1X/replay-1/16/joint/augmented 0/6 | both retention levers independently sufficient |
| ARK-018 V4 Birth internalization | +0.000 / +0.067 vs required +0.10 | preregistered threshold NOT MET (2 seeds) |
| ARK-018 V4 binding-acquisition slowdown | 1200 / >1500 vs 300 / 300 steps | replicated narrow plasticity cost of Birth-10% |
| ARK-019 V3.1 old-skill robust-min | PLASTIC_HIGH ≈0.005; STATIC 1/64 ≈0.882; Guardian ≈0.963 | plastic continuation destroys; replay protects; Guardian recovers |
| BRAMASTRA EOS experiment | 0/32 → 32/32 (both seeds) | EOS supervision mechanically required |

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
| CYR-GPU-012/013 | monotonic vocab stories falsified (V4096 100% > V19 12.94% > V24576 0%; intermediate regime seed-sensitive) | **DEMONSTRATED non-monotonic class-space effect** |
| CYR-GPU-014-R1C engineering | optimizer-constructor TypeError; CLIP_BREACH float32 reduction-order abort | **repaired with regression tests; scientific run NOT_EXECUTED; launcher pins an origin-absent commit (launch-blocking)** |
| ARK-017 V2 | both retention levers independently sufficient (HIGH 4/6 fail; LOW/CAP1X/replay/joint/augmented 0/6) | **DEMONSTRATED mechanism dissection** |
| ARK-018 V4 | Birth internalization threshold NOT MET (+0.000/+0.067 vs +0.10); science NLL +3.7–3.9%; binding-acquisition slowdown replicated | **DEMONSTRATED negative on primary; real-text transfer partially answered** |
| ARK-019 V3.1 | `CONTROLLER_NOT_SUPPORTED`; new skill never formed in any arm; PLASTIC_HIGH destroyed old skill 4/4 | **DEMONSTRATED: formation gates continual learning, Guardians recover but did not prevent** |
| ARK-019 V4 | transcribed external `GUARDIAN_CONTINUAL_PROXY_CANDIDATE` (old 4/4 + new 4/4 under Guardian arms) | **SUPPORTED-ONLY: raw bundle absent from repo; byte re-audit required** |
| ARK-020 V1→V4 | five + four + seven blockers repaired across versions; 39/39 tests; A1 durability amendments | **IMPLEMENTED / NOT EXECUTED in-repo; external Drive results inaccessible** |

Failures are not waste; they remove bad explanations. But a failed experiment with a confound does not justify a universal negative claim.

---

# PART VII — WHAT TO BUILD NEXT, IN ORDER

## Gate A — run the R1C softmax-competition dissection (supersedes the old compact-exposure closure)

The old Gate A (extend the compact bridge to full exposure) was overtaken by events: R1 and R1B already mapped the class-space response curve with fresh seeds. The open question is now causal, and CYR-GPU-014-R1C is the designed discriminator:

```text
R1C MASK_4096 rescues formation on the fixed 24,576 matrix
    → softmax-partition/competition is causal
    → production-representation change candidates: inactive-row exclusion,
      output regularization, or controlled untied low-rank output
R1C MASK arms fail like FULL_24576
    → the effect lives in tied initialization/optimization geometry
    → design the next dissection there
```

**Blocking defect first:** rebind the R1C launcher to a commit that exists on origin (live `f2c27a6` carries byte-identical repair content and the identical RUN_READINESS_V4 blob); the current launcher checkout of `6653b4ce` cannot succeed from a fresh clone.

## Gate B — only then touch the production tokenizer

Any production vocabulary/representation change must wait for R1C's mechanism verdict plus a replication class (R1B gave directional R1 replication; R1C adds 4 matched seeds per arm). Do not infer a general-language tokenizer rule from arithmetic alone.

## Gate C — mechanism credit: ANSWERED (ARK-017 V2)

ARK-017 V2 executed: **BOTH_LEVERS_SUFFICIENT** — lowered applied update magnitude (CAP1X) and sparse treatment-exact invariant-support replay each protected 6/6, as did their combination and augmented-HIGH; unprotected NARROW_HIGH failed 4/6; "retention = small total parameter movement" is falsified. Remaining open: which lever is *necessary* in which regime, dose universality (secondary screens were single-order), and scale transfer.

## Gate D — real-text substrate: ANSWERED at first order (ARK-018 V4)

ARK-018 V4 executed and audited: a ~20–25M model trained on peS2o with periodic Birth-Book exposure assimilated the Birth distribution strongly but did **not** meet the preregistered Birth-content internalization threshold (+0.000/+0.067 vs required +0.10), cost ~3.7–3.9% SEALED science NLL vs matched science replay, and slowed later temporary-binding acquisition (1200/>1500 vs 300/300 steps). Birth content learning is not evidence of identity, consciousness, reasoning or AGI. Its SCIENCE_ONLY checkpoints are now the standing real-text parents.

## Gate E — continual-learning controller: formation-gated, evidence mixed

ARK-019 V3.1 executed: `CONTROLLER_NOT_SUPPORTED` because the new skill never formed in any arm (4 binding slots/update was underpowered); the controller *recovered* old-skill health via replay escalation (~0.963) but did not demonstrate prevention. ARK-019 V4 (dose-qualified, science-preserving parents) carries only a **transcribed external** `GUARDIAN_CONTINUAL_PROXY_CANDIDATE`. Do not treat either as a production controller recommendation. The honest sequence is: commit + byte-audit the V4 bundle (or rerun), then ARK-020's four-skill efficiency-gated battery.

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

Citadel's production-path audit (at pin `28bf57a`, 2026-09-06) found the corpus and top-level entry MISSING and tokenizer/schedule/evaluation AMBIGUOUS; cymek-500m-readiness has advanced substantially since, but the data gate is still external and no scientific result yet authorizes scale.

---

# PART VIII — CURRENT BEST AGI RESEARCH HYPOTHESES

## H1 — representation/class-space can change whether structural capability emerges

**Status: SUPPORTED with the effect partially isolated; mechanism untested.** CYR-011 demonstrated the condition-level divergence; CYR-012 demonstrated class-space size alone (active IDs fixed) can move held-out formation dramatically; CYR-013 replicated the non-monotonic intermediate-regime ordering across two fresh seeds. The remaining mechanism question (training-time inactive softmax competition vs tied-matrix geometry) is exactly what R1C tests.

## H2 — capability emergence can be delayed far beyond memorization

**Status: DEMONSTRATED at Micro symbolic scale.** ARK-002B replicated the qualitative memorize-first → delayed-generalize transition with large seed variance (rho 0.00 between M99 and G90 timing). This means stopping immediately after train saturation can miss a later structural transition.

## H3 — acquired capability can narrow without canonical accuracy falling

**Status: DEMONSTRATED at controlled Micro non-arithmetic scale.** ARK-015 retained canonical exact at 1.0 while order/query-order invariance eroded under narrow high-plasticity continuation.

## H4 — retention is an interaction between plasticity and ongoing support

**Status: STRONGEST CURRENT MECHANISM PICTURE; levers shown independently sufficient (ARK-017 V2: `BOTH_LEVERS_SUFFICIENT`); single necessary-mechanism attribution still open.** LOW protected; CAP1X protected; sparse treatment-exact replay protected; total parameter movement does not explain outcomes. Dose universality and scale transfer remain unresolved.

## H5 — acquisition/recovery and retention need different control regimes

**Status: SUPPORTED at Micro T2.** HIGH helps recover absent capability (8/9), LOW protects present capability (0/12), and HIGH→LOW after recovery reduces recurrence (0/6). Exact state thresholds were not identifiable (ARK-012) and scale transfer is unresolved.

## H6 — query-conditioned addressing is a real missing operation, but current weak-substrate evidence cannot justify a learned self-model

**Status: SUPPORTED as a bottleneck hypothesis / self-model NOT_DEMONSTRATED.** Triquetra measured chance-level query ranking and strong recency effects (replicated), BRAMASTRA independently showed aggregate binding accuracy is fully explained by query-blind copying, and the X1 self-model PASS was invalidated by a trivial-baseline audit.

## H7 — primitives do not automatically compose

**Status: SUPPORTED.** T1D teacher primitives learned to ~51.5% held-out while full arithmetic composition stayed near floor.

## H8 — better data/measurement should be tested before exotic architecture

**Status: STRONG INFERENCE.** No current evidence shows MoE, recurrence, SSM, latent thought or neural long-term memory is the binding bottleneck. Adding them before resolving representation mechanism, objective, data support and capability formation would reduce interpretability.

## H9 — new-skill formation gates every continual-learning verdict

**Status: DEMONSTRATED (formation-first lesson).** ARK-013 (T3 never acquired) and ARK-019 V3.1 (SKILL_B never formed, so the official controller verdict was `CONTROLLER_NOT_SUPPORTED` interpreted as a formation bottleneck) both show that controller comparisons are meaningless before the unprotected reference can acquire. V4/ARK-020 encode this as prospective dose selection and formation-first gates.

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

The project's strongest result is **not "we have AGI."** The strongest positions after the 2026-09-13 audit are: a rigorous experimental framework that repeatedly catches its own false greens; several real narrow discoveries (delayed generalization after memorization; state-dependent acquisition/recovery/retention behavior; non-arithmetic invariance narrowing; two independently sufficient retention levers; a non-monotonic class-space formation effect); an executed real-text substrate result (ARK-018); and an executed controller result that is honestly negative at formation level (ARK-019 V3.1).

The next breakthrough is most likely to come from **the R1C softmax-competition dissection**, which converts the class-space discovery from correlation to mechanism — provided its launcher provenance defect is repaired first. The second-most valuable action is committing and byte-auditing the ARK-019 V4 raw bundle (or rerunning it), because the entire Guardian question currently rests on a transcription.
