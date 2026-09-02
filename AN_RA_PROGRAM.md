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

### 6. The bottleneck is query-conditioned addressing

Causal decomposition on 61 real failures with orthogonal intervention axes:

| Bottleneck | Evidence |
|---|---|
| **Routing/Addressing** | +31pp from relevant-fact selection; +0.977 nats gold logprob |
| **Selection→Realization gap** | 87% copy → 31% generate |
| **Copy/Realization** | 86.89% — mostly intact |

The information IS in the model. The routing to it is broken.

### 7. Target duplication confirms addressing (controlled factorial)

120 tasks, 92 failures, 7 conditions with matched controls:
- Target-duplicate near Answer: **19.57%** vs random-duplicate 4.35% = **+15.22pp**
- Entity-only marking: **0%** — the model needs the code token, not just the entity
- Interference curve is non-monotonic: some distractors help, too many collapse

**Conclusion: the model cannot internally route from query entity to the
corresponding fact's value under competitive binding load. External fact
duplication bypasses this routing failure.**

### 8. First significant query conditioning achieved

Group-structured query-swap SFT produced the first statistically significant
query-conditioned preference: **lift +0.669 nats, p = 0.018** (was ~0).
The computation is learnable; breadth is the remaining gap.

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
| Query lift | +0.669, p=0.018 | x1_real_receipt.json |
| Causal decomposition | 3 bottlenecks | causal_decomposition.json |
| Target-dup contrast | +15.22pp | binding_factorial.json |

## What's Next

1. **X2**: fresh replication of the binding factorial on a new seed
2. **Checkpoint comparison**: parent vs children on the binding factorial
3. **Internalization test**: does group-structured SFT reduce addressing dependence?
4. **Scale to P35**: the group-structured curriculum at P35 scale
5. **Cognitive Bottleneck Atlas**: expand to state tracking, retrieval, composition
