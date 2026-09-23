# ROLE-TRANSFER-001 design

Status: **PREREGISTERED DESIGN — EXECUTION BLOCKED**

## Decision

Do not multiply the current experiment by an arbitrary 10,000×. The preserved run is floor-bound and its primary proxy is too narrow:

- `CS-MECH-002`: 21/24 official arms ended at identity success `0.0`; the largest development AUC contrast was `0.008036`, below the preregistered `0.05` threshold.
- `REP-FORM-003A`: all eight official arms ended at identity success `0.0`; this is a clean development null, not an equivalence result.
- Every recorded `CS-MECH-002` arm clipped on every update, so gradient role and global update magnitude were not isolated.
- The TIE-ROLE frontier has only two canonical controls; it has no treatment contrast or transfer result.
- The current endpoint is identity formation on a narrow synthetic surface, not held-out downstream task success.

A 10,000× brute-force budget would cost much more without fixing those validity problems. The new design spends additional remote compute on independent replication, full-preclip norm controls, a clean-room evaluator, and capability calibration.

## Relationship to the frozen run

`ROLE-TRANSFER-001` is a new campaign. It does not modify, resume, reinterpret, or merge into `FORMATION-MUX-001`.

It must not reuse upstream:

- training, development, or sealed rows;
- entity/rule namespaces;
- checkpoints;
- seeds `73011–73014`;
- sealed commitments;
- output directories or mutable coordinator state.

Recovery and completion of the upstream frontier remain prerequisites. This design may be reviewed and implemented now, but official execution remains blocked.

## Primary causal question

At fixed forward function, initialization, data stream, optimizer, clipping policy, and token exposure:

> Does balancing input-embedding and output-projection gradient roles, while matching the full pre-clip trainable-gradient norm, causally improve independently generated held-out production-BPE downstream task success within the new synthetic ontology?

Primary estimand:

```text
mean over 12 frozen matched seed blocks of
  downstream_macro_success(T3_NORM) - downstream_macro_success(T0_CANONICAL)
```

This is a **within-ontology transfer** claim. It is not an external benchmark claim.

## Arms

### Confirmatory four-arm set

- `T0_CANONICAL`: tied embedding, canonical input/output gradient scales `(1, 1)`; reference trajectory for `T0/T3_NORM` magnitude matching.
- `T3_RAW`: role scales `(4, 0.25)` without norm matching; reference trajectory for `P_NORM/T3_RAW` magnitude matching.
- `T3_NORM`: role scales `(4, 0.25)`, rescaled on every update to the current `T0_CANONICAL` trajectory's full pre-clip norm.
- `P_NORM`: canonical `(1, 1)` role scales, globally rescaled on every update to the current `T3_RAW` trajectory's full pre-clip norm.

Primary efficacy is `T3_NORM - T0_CANONICAL`.

Role specificity is `T3_RAW - P_NORM`, evaluated on the same 12 blocks and same 2M-token exposure as the primary contrast.

Magnitude placebo is `P_NORM - T0_CANONICAL`. If the placebo reproduces the primary gain, the result is not role-specific.

The four arms run in synchronized seed blocks and data order. Each norm-matched arm follows the current full pre-clip norm of its pair-specific reference trajectory while its own model state evolves.

The global clip norm is fixed at `1.0`. Norm matches must be within relative tolerance `1e-6`. `T0/T3_NORM` and `P_NORM/T3_RAW` must have identical per-update clip decisions.

Every update records:

- `T0` reference full pre-clip norm;
- `T3_RAW` reference full pre-clip norm;
- matched full pre-clip norm;
- canonical and raw full post-clip norms;
- clip decision;
- parameter-displacement L2 norm.

Pre-clip norm matching does not guarantee identical AdamW displacement; displacement is therefore recorded, not claimed equal. The treatment is described as **role-reweighted**, not assumed balanced, unless the preregistered diagnostic gate passes.

Every update must emit a hash-chained `anra.role-transfer-norm-receipt/v1` record containing the seed block, update, processed tokens, reference arm, target/observed norms, relative error, clip decision, parameter displacement, and previous receipt hash. The final checkpoint receipt must index every update receipt.

Role engagement is measured on the `T0` reference state/batch by decomposing the tied gradient into input-only and output-only L2 contributions before routing. The baseline statistic is `||g_out|| / ||g_in||`. Per seed, the manipulation effect is the median across updates of `1 - raw_ratio / canonical_ratio`. Every confirmatory seed must show at least 25% reduction, and the stratified interval lower bound must exceed zero.

### Exploratory mechanism arms

- `T1_INPUT_X4`: `(4, 1)`.
- `T2_OUTPUT_X025`: `(1, 0.25)`.
- Optional `U_UNTIED_OUTPUT`: separate-parameter diagnostic, never the primary tying result.

These arms cannot change the confirmatory protocol after outcomes are observed.

## Replication and data

Two independent confirmatory replications:

- Replication A: seed blocks `74001–74006`.
- Replication B: seed blocks `74007–74012`.

The seed block is the inferential unit. It binds a fresh initialization, data permutation, stochastic task stream, and matched four-arm set.

Per replication:

- 120,000 training rows: 20,000 in each of six domains.
- 6,000 development rows: 1,000 per domain.
- 12,000 sealed tasks: 2,000 per domain.
- New rule graph, entity namespace, relation compositions, and tool schemas.
- Production BPE is the primary rendering; latent IDs are diagnostic only.

Seed roles are disjoint:

- calibration: `74101–74104`;
- mechanism engineering: `74201–74202`;
- confirmatory: `74001–74012`;
- optional model shift: `74301–74304`;
- excluded upstream seeds: `73011–73014`.

## Fixed model and exposure

Primary model:

- 8 layers, width 256;
- 4 query heads, 2 key-value heads, head dimension 64;
- FFN width 1,024;
- context length 1,024;
- physical vocabulary 24,576;
- tied input/output embeddings;
- no dropout.

Per arm:

- fixed 2,000,000 processed non-padding tokens;
- batch size 32;
- at most 3,000 updates and 96,000 row presentations;
- checkpoint every 100,000 tokens;
- exposure mismatch no greater than 0.1%;
- no outcome-dependent dose selection;
- no early stopping.

This is four times the current 500,000-token transfer exposure, not 10,000×. The mandatory campaign pre-training ledger is exactly 120,000,000 processed tokens before sealed evaluation.

## Clean-room evaluator

Six equal-weight domains:

1. unseen composition;
2. state/workflow reasoning;
3. deterministic tool execution;
4. long-context binding;
5. missing-information abstention;
6. structured termination with valid EOS.

Tool tasks are graded by final environment state. No model-generated judge is permitted. Evaluator controls must show:

- query-blind performance at the declared null level;
- shuffled-label performance at the declared null level;
- no train/development/sealed task overlap;
- valid schema, EOS, and abstention;
- zero unauthorized actions;
- no material identity-retention regression.

The evaluator, task list, scoring code, and commitments are frozen before official treatment outcomes.

## Fixed substrate calibration

Before causal treatment arms:

- run `T0_CANONICAL` on four fresh calibration seeds;
- use the fixed 2,000,000-token dose;
- lower-dose runs may be recorded as diagnostics but cannot select the confirmatory dose;
- require control success between 30% and 70%;
- require global-clip fraction at most 5% on every calibration seed.

A floor, ceiling, or clipping-dominated calibration produces `INCONCLUSIVE_SUBSTRATE`; it is not a mechanism null.

Every official primary/mechanism seed-arm must have clip fraction at most 5%, and norm-matched pairs must have identical clip decisions. Any violation produces `CLIPPING_CONFOUNDED`, not a mechanism null.

The mechanism manipulation must reduce the output-to-input tied-gradient norm ratio by at least 25% in every confirmatory seed. This supports a role-reweighting manipulation; it does not assume the raw `(4, 0.25)` arm is perfectly balanced.

## Decision thresholds

A primary `SUPPORTED_TRANSFER` result requires all of:

- at least +8 percentage points held-out downstream macro-success;
- exact stratified two-sided paired sign-flip `p < 0.05` over all 4,096 sign vectors;
- stratified paired 95% bootstrap interval excluding zero;
- positive effect in at least 9/12 seed blocks;
- at least 4/6 positive blocks and a positive mean in each replication;
- at least +5 points on unseen composition and tool execution;
- no more than 3 points regression on termination, abstention, or identity-retention guards;
- zero unauthorized actions;
- `T3_RAW - P_NORM >= 5` points with a paired 95% interval lower bound above zero;
- the `P_NORM - T0_CANONICAL` paired 95% interval upper bound is below 5 points.

The bootstrap uses 100,000 paired resamples stratified by replication with fixed analysis seed `20260924`.

`LATENT_ONLY` uses one frozen secondary endpoint: latent-ID identity-formation AUC on `LATENT_DEVELOPMENT_MANIFEST_V1.json`. The manifest and evaluator SHA-256 receipts must freeze before Stage 1. It requires a paired gain of at least 5 points with a 95% interval lower bound above zero. It can never support the primary production-BPE claim.

## Ordered decisions

Decisions are mutually exclusive in this order:

1. `INCONCLUSIVE_PROTOCOL`: custody, evaluator, missing-seed, exposure, sealed-firewall, or analysis-contract failure.
2. `INCONCLUSIVE_SUBSTRATE`: valid protocol but control-only floor, ceiling, or calibration failure.
3. `CLIPPING_CONFOUNDED`: any official seed-arm exceeds 5% clipping or either norm-matched pair has different clip decisions.
4. `MECHANISM_NOT_ENGAGED`: output-to-input norm ratio fails the 25% reduction gate in any seed.
5. `SUPPORTED_TRANSFER`: primary core, replication, composition/tool transfer, guards, role specificity, and placebo-equivalence flags all pass.
6. `PRIMARY_NONTRANSFER`: primary core and replication pass, but composition/tool transfer or guards fail.
7. `PRIMARY_NONREPLICATED`: primary core passes, but either replication mean or four-of-six rule fails.
8. `MAGNITUDE_EFFECT`: primary core, replication, transfer, and guards pass, and the magnitude placebo has a qualifying gain.
9. `MECHANISM_UNRESOLVED`: primary core, replication, transfer, and guards pass, but neither supported-transfer nor magnitude criteria pass.
10. `MAGNITUDE_ONLY`: primary core fails while the magnitude placebo independently passes.
11. `LATENT_ONLY`: primary core fails the magnitude-only branch and the frozen latent diagnostic passes.
12. `NULL_OR_REVERSE`: none of the preceding branches applies.

## Statistical analysis

- Sampling population: the 12 frozen seed blocks, not an unlimited stream of seeds.
- Primary test: exact stratified two-sided paired sign-flip test on the mean block effect.
- Report the paired mean effect, stratified paired 95% interval, and both replication estimates.
- Sensitivity: mixed-effects logistic regression with treatment and task family fixed effects plus a seed-block random intercept.
- Do not treat task rows as independent experimental replications.
- Do not impute failed or missing confirmatory seeds.

## Remote-only staged plan

- Stage 0: zero-GPU custody, forward equivalence, norm-matching, source, leakage, and sealed-firewall checks.
- Stage 1: four control-only calibration blocks at fixed 2M tokens; lower doses are diagnostic only.
- Stage 2: two mechanism-engineering blocks with all four arms at fixed 2M tokens; only manipulation and norm receipts decide progression.
- Stage 3: twelve confirmatory blocks with all four arms at fixed 2M tokens.
- Stage 4: one clean-room sealed evaluation after all development artifacts and evaluator hashes freeze.
- Optional Stage 5: four fresh `T0`/`T3_NORM` blocks at 12 layers, width 384, six query heads, three key-value heads, and head dimension 64.

Treatment outcomes cannot change arms, dose, thresholds, data, evaluator, or progression rules.

Mandatory processed-token exposure is 120,000,000:

- Stage 1: 8,000,000;
- Stage 2: 16,000,000;
- Stage 3: 96,000,000.

Kaggle T4×2 wall-time remains `PENDING_PRE_OUTCOME_T4X2_ENGINEERING_CALIBRATION`; no unsupported duration claim is made.

## Execution blockers

Official execution remains blocked until all of these are satisfied:

1. upstream checkpoint-bearing Output is recovered and the frozen frontier is completed;
2. new generator, four-arm trainer, full-preclip norm controls, and exact-resume checkpoints are implemented;
3. clean-room evaluator and sealed commitments are frozen;
4. no upstream row/checkpoint/seed overlap is proven;
5. all new seed roles are proven disjoint;
6. protocol and source hashes are canonicalized in remote CI;
7. CPU/static and Kaggle T4×2 engineering qualifications pass.

## Claim ceiling

The strongest possible positive claim is:

> Under this preregistered synthetic ontology, a forward-equivalent, full-preclip-norm-matched role-reweighted tied-gradient treatment improved independently generated held-out production-BPE downstream macro-success by at least eight percentage points at the specified model and exposure budget, with consistent replication and no material guard regression.

It cannot establish external benchmark validity, AGI, cognition, consciousness, human-level reasoning, real-world tool competence, production architecture superiority, tokenizer optimality, large-scale authorization, or that the upstream S5 null was wrong.
