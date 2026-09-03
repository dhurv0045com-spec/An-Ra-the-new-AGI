# An-Ra Research Program — Evidence Ledger

> **Substrate framing (binding): the current V4 checkpoint
> (`checkpoints/anra-v4-current-full-resume.pt`, step ~30400) is a WEAK
> SUBSTRATE / HISTORICAL DIAGNOSTIC TARGET.** Its failures must NOT drive
> architecture decisions, rescue training, or claims about what An-Ra Cores
> can in principle do. Triquetra on this checkpoint builds instruments and
> preserves negative controls; the research subject will be a stronger
> checkpoint evaluated through the readiness gate
> (`python x_factor/qualify_checkpoint.py --checkpoint PATH`).
> Readiness regimes: PARTIAL = researchable; FLOOR/CEILING = not identifiable.

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

### 7d. Query-value evidence matrix (NEW, DEV + DEV_REPLICATED + checkpoint comparison)

No gold inserted; S[i,j] = log P(value_j | facts, query_i), k=4, chance 25%:
- DEV seed 71717 (80 sets, 320 queries, step-30400): raw rank1 **25.0%**
  (=chance), normalized 30.6% (+5.6pp p=0.13 ns); QCS +0.053 CI
  [−0.008,+0.114] incl 0; diagonal +0.05 nats; position std +0.161 vs
  query-match +0.009 (19×); permutation raw 26.3/norm 32.5%.
- Ladder (160): E0 raw 15.0, E1 raw-rank 24.4, E2 norm 27.5 (ns),
  **E5 visible-query dup 23.1 vs sham 7.5 (+15.6pp p≈0)**,
  E6 mark 13.8, E7 selection 15.0 (=E0, null), E8 oracle 51.9%.
- REP seed 81818: raw 26.9, norm 32.5 (+5.63pp, identical), QCS +0.046 ns,
  position 35×, E5−sham +19.4pp p≈0, E7 null, oracle 45.0. Pattern holds.
- Checkpoint comparison, SAME tasks: step-22517 gen **0.9%** vs step-30400
  12.2%; raw rank chance both; QCS ~0 both; **E5dup 0.6% vs 23.1%**
  (duplication-elicitability emerged); oracle 36.3% vs 51.9%.
- Training 22517→30400 built copy/readout machinery, NOT query control.
  Connector gap small (+8–12pp), oracle gap large (+24–29pp):
  ORACLE-leaning, CONNECTOR-weak. GATE1 = marginal/NO for latent-signal
  routing; X0/X1/active-diagnosis NOT earned; training NOT justified.
- Decomposition (DEV 320 queries): raw-rank1 80, normalized 98, generation
  39; rank1-but-genfail 60/80 (conditional realization gap atop chance-level
  ranking — not evidence of latent knowledge).
- Firewall `x_factor/observed.py` (VisibleTask-only E1/E2/E5/E6/E7; E8
  oracle intentionally fails guard; 5 CI tests pass).
Evidence: `output/query_value_evidence_dev.json`,
`output/query_value_evidence_dev_rep.json`,
`output/query_value_ckpt22517.json`,
protocols `query_value_evidence_dev_v1.json`,
`query_value_evidence_dev_rep_v1.json`.

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
| QV evidence matrix | latent query signal ~absent; E5-by-recency only | output/query_value_evidence_dev.json + _rep |
| Checkpoint 22517→30400 | copy/readout grew, query control flat | output/query_value_ckpt22517.json |
| E5 structural-OOD | FAILED: 0.0 vs 0.0, template-bound; E5 line closed | output/structural_ood_e5.json |
| Readiness pilot 30400 | B3 ADEQUATE pocket; B0/B2 limited; gate READY-scoped-to-B3 | output/readiness_pilot_30400.json |

## Substrate-Adequacy Audit (old V4 checkpoint; historical, not architectural)

| Experiment | Raw | Oracle/legal best | Adequacy |
|---|---|---|---|
| X1-REAL-0 | 25% | oracle repair 13.6% | ORACLE_LIMITED + INVALID (prediction) |
| IBQ legacy / v2-DEV | — | coverage ~0–8.8% | INTERVENTION_SPARSE, NOT QUALIFIED |
| Causal decomposition | 24% | selection 31% (confounded) | MARGINAL (clue only) |
| Binding factorial | 23% | target-dup 19.6% | MARGINAL |
| Entity×value DEV/REP | ~11–13% | oracle C2 ~46% | ADEQUATE (DEV surfaces only) |
| Competitive binding | ~0–12% | — | FLOOR_LIMITED |
| QV matrix DEV | gen 12–15% | oracle ~45–52% | MARGINAL (behavioral); score-level sparse |
| Structural OOD E5 | 0% | oracle ~24% (≈chance) | FLOOR_LIMITED + ORACLE_LIMITED |
| ckpt-22517 QV | 0.9% | — | FLOOR_LIMITED |

Instruments (no old-ckpt science claims attached):
`x_factor/readiness/` (B0–B7 ladder, floor/ceiling auto-detect, McNemar
power, gate runner), `x_factor/qualify_checkpoint.py` entry point,
`x_factor/checkpoint_identity.py` (strict, no silent fallback),
`x_factor/registry/checkpoints.json`, prediction/response schemas
(SOFTWARE_DEMONSTRATED via 15 CPU CI tests, NOT MODEL_DEMONSTRATED).

## What's Next

1. **E5 line CLOSED (structural-OOD FAILED, template-bound).** Fresh
   lexicon/codes/grammar/queries (60 sets×4): E0 raw 0.0%, E5dup 0.0% vs
   sham 0.0% (effect 0, p=1.0). Duplication-assist does not generalize.
   Do NOT train internalization off E5. Evidence:
   `output/structural_ood_e5.json`, protocol
   `x_factor/protocols/structural_ood_e5_v1.json`.
   Note: E2-normalized reached 20.8% (below 25% chance) with oracle at
   24.2% — OOD surfaces push all arms to chance/floor, so the latent-signal
   question is UNRESOLVED there, not answered.
2. **Candidate next mechanism (uncommitted):** value-prior/position
   decomposition — position dominates query-match 19–35× on DEV. If a
   position-debiased selection rule beats chance on DEV *and* OOD, it would
   reopen GATE1. No protocol frozen yet.
3. **X0/X1**: response-matrix prediction ONLY over legal interventions,
   with committed PredictionBeforeInterventionRecords — only if (2) earns it.
4. **Cognitive Bottleneck Atlas**: ONE next family only after binding
   methodology earns it.
