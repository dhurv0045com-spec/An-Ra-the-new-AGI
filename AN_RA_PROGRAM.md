# An-Ra Research Program — Complete Guide

## What An-Ra Is

An-Ra is a research program to build a small intelligence that **notices its
own failures, diagnoses them with controlled experiments, makes the smallest
corrective change, and proves on sealed tests that it gained capability
without losing what it had.**

The ultimate goal is **causal capability accumulation**: a closed loop where
the system repeatedly acquires cognitive primitives, where each acquisition
was justified by evidence, produced by the smallest intervention, transferred
to unseen tasks, and did not destroy existing capabilities.

## The Research Arc (what we proved, step by step)

### 1. The substrate had zero context binding (measured)

V4 180M model, steps 5k/20k/30.4k: nonce-fact-to-answer binding was **0/5
at every checkpoint, in both natural-language and structured protocols**.
The model could not use information placed in its context at all.

### 2. Targeted SFT works — capability is learnable

~940 corrective examples + 9 GPU-minutes gave single-fact binding that
transfers across 5 untrained protocols (4/15 → 13/15). **Context binding
is learnable at 180M.**

### 3. Narrow training steals capability

A prose-only follow-up gained its target but destroyed protocol transfer
(13/15 → 6/15). Gradient analysis showed cos(target, retention) = +0.17:
the forgetting was **starvation** (14.3% rehearsal, single format), not
active interference.

### 4. Balanced replay preserves capability — replicably

The accumulation child (balanced multi-format replay) is ≥ its parent on
every axis of two independent sealed suites. **Retention transfer is real.**

### 5. Selective binding did NOT survive independent replication

OOD-4: paired-counterfactual selective binding 0/26 on both models.
The OOD-3 gain was format familiarity, not robust selection.
**Honest negative preserved.**

### 6. The bottleneck was hypothesized as query-conditioned addressing (DOWNGRADED)

Causal decomposition on 61 real failures with orthogonal intervention axes:

| Bottleneck | Evidence | Status |
|---|---|---|
| **Routing/Addressing** | +31pp from relevant-fact selection; +0.977 nats gold logprob | STRONG DEVELOPMENT CLUE — the selection intervention also removed distractors, shortened context, moved the fact, and explicitly selected it. Does NOT cleanly isolate addressing. |
| **Selection→Realization gap** | 87% copy → 31% generate | DEVELOPMENT observation, retained |
| **Copy/Realization** | 86.89% — mostly intact | DEVELOPMENT observation, retained |

Superseded by the entity×value factorial (§7b): the information responds
most to correct-value recency, not to entity addressing.

### 7. Target duplication contrast (controlled factorial, historical)

120 tasks, 92 failures, 7 conditions with matched controls:
- Target-duplicate near Answer: **19.57%** vs random-duplicate 4.35% = **+15.22pp**
- Entity-only marking: **0%** — INVALID as entity-addressing evidence: the
  intervention replaced the fact with `>>> {Entity} (queried)`, deleting the
  value code, and the `p3_entity_dup`/`p3_fact_dup` arrays were never
  populated (helper returned 0 for empty arrays). Those runs never occurred.
- Interference curve (D0–D4): EXPLORATORY — D0 margin definition differs,
  loads contain different eligible populations, distractors resampled per
  load. Clean nested replication (competitive_binding_dev.json) shows floor
  accuracy and no competing-specific harm beyond length. Threshold claim withdrawn.

**Defensible interpretation: query-matched answer-bearing fact
localization/repetition materially changes behavior under selective binding
failures. Mechanism unresolved by this experiment alone.**

### 7b. Entity×value factorial isolates the mechanism (NEW, DEV_REPLICATED)

Same-failure, full-context, fixed-insertion factorial C0–C11 on the locally
available step-30400 pretraining checkpoint (legacy SFT child file absent;
effect sizes not comparable to the +15.22pp history):
- DEV (seed 41414, 107 failures): C0 3.7, C1 entity-only 0.0,
  C2 value-only **46.7**, C3 pair 26.2, C4 wrong-entity+value 28.0,
  C5 entity+wrong-value 0.9, C6 valid-distractor pair 0.0,
  C7 full-target 20.6, C8 full-distractor 0.0 (all % repair on failures).
- Paired: C2−C0 **+43.0pp** [32.7,52.3] p≈0; C1−C0 −3.7pp p=0.125;
  C3−C2 **−20.6pp** p=0.0001; C3−C4 −1.9pp p=0.845;
  C7−C8 **+20.6pp** p≈0 (direction replicated, cause reattributed).
- Mechanism likelihood: gold-LP deltas C2 +13.7, C3 +10.6, C7 +6.0,
  C1 +5.4 (LP-only nudge, 0% behavior); margins C2 +7.8, C3 +6.0, C7 +3.1.
- DEV replication (seed 51515, 104 failures): C2 46.2, C3 33.7, C4 27.9,
  C1 1.9, C7 13.5 vs C8 2.9; C2−C0 +33.7pp p≈0; C3−C2 −12.5pp p=0.019;
  C3−C4 +5.8pp p=0.34. Same direction, compatible magnitude.

**Classification: VALUE_RECENCY_DOMINANT (CLEAN_DEV_EFFECT, DEV_REPLICATED).**
Bare correct-value repetition suffices and beats pair/full-fact; entity
identity is irrelevant given the correct value; wrong value kills; entity
alone never repairs behaviorally. The historical target>distractor contrast
replicates in direction but is driven by value recency, not proven routing.
Evidence: `output/entity_value_factorial_dev.json`,
`output/entity_value_factorial_dev_rep.json`,
protocol `x_factor/protocols/entity_value_factorial_dev_v1.json`.

### 7c. Competitive binding load (NEW, negative DEV evidence)

Nested same-task L0–L4 with token-matched filler controls (80 tasks, k=5):
accuracy at floor (L0 7.5%, L1c 12.5% vs L1f 0%, L2+ ~0%); gold LP degrades
with load in both arms (L0 −14.2 → L4 ~−16.5) with tiny unsystematic
competing−filler gaps. CBL-as-beyond-length NOT earned.
Evidence: `output/competitive_binding_dev.json`.

### 8. Query-conditioning SFT claim (UNVERIFIED HISTORICAL CLAIM)

AN_RA_PROGRAM previously stated group-structured query-swap SFT gave
**+0.669 nats, p=0.018** citing `x1_real_receipt.json`. That receipt is the
sparse X1-REAL-0 matrix (45 failures, oracle 13.6%, prospective 95.45% from
an almost-always-negative predictor) and does NOT contain the +0.669 result.
X1-REAL-0 is INVALID for predictive self-modeling. The +0.669 claim is
UNVERIFIED until its exact artifact, checkpoint, commit, protocol, sample,
metric, and test are produced. Do not repeat it as fact.

## The Architecture

```
anra.run(task)                    ← the one reference loop
  ├─ anra_core/                   Core: executor, state, tokenizer, model
  ├─ connector/runtime.py         task → attempt → verify → intervene → repair
  ├─ connector/experiments/       probes, batteries, causal decomposition
  ├─ x_factor/                    THIS PACKAGE: the causal self-modeling program
  │   ├── contracts.py            leakage law, intervention registry
  │   ├── world.py                deterministic latent-factor physics
  │   ├── geometry.py             7 falsification worlds, neutral mixing
  │   ├── evaluation.py           policies, metrics, learner
  │   ├── ladder.py               preregistered X0–X7
  │   ├── ibq.py                  basis qualification engine v1
  │   ├── ibq_v2.py               v2 basis, tie-safe metrics, sequential policy
  │   ├── binding_factorial.py    the decisive factorial experiment
  │   ├── causal_decomposition.py orthogonal elicitation axes
  │   ├── capability_bank.py      development/replay data infrastructure
  │   └── SPEC.md                 canonical specification
  └─ training/                    SFT trainers (causal eligibility, retention floors)
```

## Non-Negotiable Rules

1. **The verifier is the only source of success.** Completers return raw text.
2. **Hidden labels never touch diagnosis or curriculum generation.**
3. **Sealed OOD suites are frozen before training, never imported by it.**
4. **Counterfactual pairs are byte-exact** — the anti-self-deception metric.
5. **Training loss is not behavior.** Every major bug was invisible in loss.
6. **Checkpoint selection is multi-objective**: target gain AND parent-relative
   retention floors.
7. **Promotion is always scoped** — never a bare "PASS."
8. **Every result carries a receipt** with full provenance.
9. **Minimal intervention ladder**: no change → runtime → memory → policy →
   small training → large training. Learn when NOT to train.
10. **Negative results are preserved**, never deleted.

## Branch Structure

- `core-vnext` — stable reference architecture
- `core-exp` — experimental research (SFT children, self-model v7-v11)
- `triquetra` — causal self-modeling + binding factorial (this branch)
- `esoes` — V5 blueprint/execution readiness

## Key Results Summary

| Experiment | Result | Evidence |
|---|---|---|
| Baseline binding | 0/5 all steps | probe_v2_*.json |
| Context-binding SFT | 5/5 both protocols | probe_v2_sft.json |
| Protocol transfer | 13/15 | ood_child_sft.json |
| Interference from narrowing | E: 13→6/15 | ood_grandchild_sft2.json |
| Replay preserves retention | ≥ parent | ood2_*.json |
| Selective binding NOT replicated | 0/26 both | ood4_*.json |
| Gradient conflict | cos=+0.17 (H1) | grad_conflict.json |
| Query lift | UNVERIFIED (see §8) | — do NOT cite x1_real_receipt.json |
| X1-REAL-0 | INVALID for self-model (imbalanced) | output/x1_real_receipt.json |
| Causal decomposition | clue, not isolation | output/causal_decomposition.json |
| Target-dup contrast | +15.22pp direction only; cause reattributed | output/binding_factorial.json |
| Entity×value factorial | VALUE_RECENCY DEV_REPLICATED | output/entity_value_factorial_dev.json + _rep |
| Competitive binding | CBL-specific effect NOT supported (floor) | output/competitive_binding_dev.json |

## What's Next

1. **Structural-OOD DEV**: new lexicon/code-family/format repl of entity×value
   factorial (value-recency vs pair under distribution shift).
2. **Checkpoint comparison**: SAME entity×value tasks across compatible
   lineage checkpoints (raw accuracy + assistance-dependence vector).
3. **Internalization test**: smallest training targeting value-recency
   dependence; success = raw up AND intervention lift down AND transfer.
4. **X0/X1**: response-matrix structure → prospective intervention-response
   prediction with committed PredictionBeforeInterventionRecords.
5. **Cognitive Bottleneck Atlas**: ONE next family only after binding
   methodology earns it.
