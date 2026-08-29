# An-Ra V5 Master Blueprint

Status: **GROUND BLUEPRINT v0.4 — E0 SHORTCUT-RESISTANT BENCHMARK CONTRACT**
Date: 2026-08-30
Branch: `esoes`
Training authorization: **NO**

This document is the canonical V5 research blueprint. It is intellectually independent of VNext implementation. The evidence base is `EVIDENCE_BASE.md`, the four-round attack record is `ITERATIONS.md`, and change control is `DECISIONS.md`. A value marked **[EXPERIMENT REQUIRED]** is not permission to encode it silently into a trainer.

[`../../blueprints/IMPLEMENTATION_BLUEPRINT.md`](../../blueprints/IMPLEMENTATION_BLUEPRINT.md) is the canonical code/infrastructure companion. Executable arithmetic and artifact schemas live in `v5_contracts/`; E0 and E1 research interfaces live in `e0_cognition/` and `e1_tokenizer/`. None of these packages is a production trainer.

Ground Blueprint v0.4 freezes the scientific boundaries and experiment order, not the exact training recipe:

| Area | State |
|---|---|
| Cognition contract, layer ownership, evidence/provenance rules | **[FROZEN]** |
| Dense Transformer baseline and no architecture soup | **[FROZEN]** |
| 250.22M/5B center, AdamW/BF16 family, checkpoint cadence | **[PROVISIONAL]** |
| Shape, tokenizer, attention details, data fraction, auxiliary loss, curriculum, LR/batch | **[EXPERIMENT REQUIRED]** |
| Recurrence/memory/MoE/BLT, >4k context, >400M scale | **[OPEN — DEFERRED]** |

## 1. Executive decision

V5 should not be a collection of speculative cognitive modules or jump to billion-scale training. The current implementation center is a conventional, inspectable 250.22M dense decoder with 26 layers, full 4k attention, a tested tokenizer, roughly 5B auditable tokens, and one narrow candidate auxiliary objective: query-swap contrast. The 250M target is a user-directed provisional envelope, not evidence that 195M was capacity-limited; E2/E5 still decide whether the extra capacity earns its compute.

Why this is the best bet:

- V4's certified continuation covered only 329,908,224 tokens; its lifetime total is not certifiable. That is insufficient evidence that 180M parameters were exhausted.
- V4 improved LM loss while binding, contextual selection, copying, and composition remained weak. Lower loss is therefore not the program's success criterion.
- SFT6 showed that query-conditioned data could expose a selection signal. SFT7 did not solve its intended rank-1 selection bottleneck, although it produced smaller lift/realization effects; the final continuation checkpoint also regressed behaviorally. Objective shape and checkpoint selection matter.
- Public scaling work supports spending substantially more trustworthy tokens on a well-sized dense model before buying parameters. Data quality and mixtures can be screened at smaller scale with useful rank correlation.
- The proposed query-swap objective directly targets the causal variable V4 failed to use: changing the query while holding the fact context fixed.

## 2. Epistemic labels

Every major choice uses one label:

1. **EVIDENCE-BACKED** — supported by direct An-Ra evidence and/or replicated public primary sources.
2. **STRONG INFERENCE** — follows from evidence and constraints, but has not been tested in the exact An-Ra setting.
3. **[EXPERIMENT REQUIRED]** — plausible alternatives remain and a bounded experiment can decide.
4. **[OPEN]** — evidence is presently inadequate or the question is deliberately deferred.

These labels apply to the decision, not to every implementation detail beneath it.

## 3. Design philosophy

### 3.1 Optimize useful cognition per total program compute

The target is not the lowest held-out next-token loss. The target is a Core that uses context to represent alternatives, selects the answer implied by the current query and state, realizes that selection reliably, and transfers these operations beyond training vocabulary and templates. **EVIDENCE-BACKED**

### 3.2 Keep the Core conventional enough to falsify the data hypothesis

Novel recurrence, memory modules, MoE routing, state-space blocks, latent-thought machinery, and multiple auxiliary losses would make a failure uninterpretable. The first V5 baseline must isolate whether better data and causal supervision can teach the missing computations. **STRONG INFERENCE**

### 3.3 Separate representation, selection, and realization

A model may contain the correct value among candidate likelihoods, fail to select it under the query, or select it internally and fail to emit it. Aggregate exact match hides these different failure modes. All V5 evaluations and any auxiliary objective must preserve this separation. **EVIDENCE-BACKED**

### 3.4 Treat every synthetic example as an experiment

Synthetic data is valuable when the answer, latent dependency graph, counterfactual, and difficulty are mechanically verifiable. It is dangerous when it is merely model-generated prose recursively imitating other model outputs. Generator version, seed, causal structure, and split membership are part of the datum. **EVIDENCE-BACKED**

### 3.5 Promote checkpoints behaviorally, not chronologically

The final checkpoint is not automatically the best checkpoint. Loss, cognition families, OOD transfer, integrity, and fresh replication jointly determine promotion. **EVIDENCE-BACKED**

## 4. Lessons from previous An-Ra generations

### V4 / continuation

- V4 is approximately 180.1M parameters: 18 layers, width 896, 14 query heads, 2 KV heads, FFN 2432, vocabulary 32,768, context 2,048, tied embeddings, RoPE, RMSNorm, QK norm, and hybrid local/full attention.
- The only certified continuation amount is 329,908,224 tokens. Claims of a 2.7B lifetime total are unsupported by the preserved ledger.
- Parent step 20,000 to final step 22,517 improved recorded LM loss from 2.1884 to 1.9710, yet the preserved PGE audit found copying, context use, binding, and composition near chance or zero.
- Step 21,800 behaved better than the final checkpoint. A final-equals-best policy is invalid.

Conclusion: V4 does not prove that a 180M dense model is too small. It proves that ordinary loss and final-step selection are insufficient. **EVIDENCE-BACKED**

### SFT6 / SFT7

- SFT6 improved some copy and context cases, but its selection result remained weak (13/48 in the preserved evaluation).
- SFT7's same-query margin intervention did not improve its intended independent-fixture rank-1 selection mechanism, although corrected receipts show a small positive query-lift delta and a replicated +5/119 greedy side effect. “It did nothing” is not an accurate summary.
- Counterfactual normalization worked on its native query-intervention matrix but failed to transfer cleanly to the fresh PGE battery.

Conclusion: contrast should manipulate the causal query while holding facts fixed. Same-query margin may affect realization/commitment, but is not adequate as the primary query-conditioned selection objective. **EVIDENCE-BACKED**

### EXP / Connector

- The frozen v7 three-action observed-only policy cleanly beat tested fixed policies, replicated on a second fresh fixture, and transferred from SFT6 to SFT7. This is evidence for repair-success routing from observable score geometry.
- Expanded v10/v11 pair actions are contaminated: stale previous-loop `candidates` state controls pair applicability; fixed pair baselines are incomplete; `best_const` is not a true fixed-family baseline; and some trainer artifacts are not reproducible from committed source. Their composition and later-promotion headlines are excluded from V5 design evidence.
- The 166 historical VIE count is invalid as a qualified causal bank because cases violate the one-variable contract and the harvest used outcome-conditioned deletion.
- External repair can diagnose a failure without proving the Core learned the missing computation.

Conclusion: the Connector should remain an experimenter and runtime controller, while success authority remains in an independent evaluator. Query-conditioned discrimination and faithful realization remain candidate native primitives; pair-action composition does not. **EVIDENCE-BACKED**

## 5. Provisional V5 architecture

### 5.1 V5-A candidate

| Component | Provisional value | Status | Rationale |
|---|---:|---|---|
| Decoder | dense causal Transformer | EVIDENCE-BACKED | Strong baselines and easiest causal attribution |
| Parameters | 250,216,960 | EXPERIMENT REQUIRED | Exact executable receipt inside the requested 250M envelope |
| Vocabulary | 24,576 | EXPERIMENT REQUIRED | Candidate tradeoff between embeddings, sequence inflation, code, and nonce copying |
| Width | 896 | EXPERIMENT REQUIRED | Retains V4 width while adding depth and capacity |
| Layers | 26 | EXPERIMENT REQUIRED | Eight more sequential blocks than V4; depth benefit must be measured |
| Q heads | 14 × 64 | STRONG INFERENCE | Exact width coverage and standard head dimension |
| KV heads | 7 | EXPERIMENT REQUIRED | Two Q heads per KV head; much less compression than V4's 2 KV heads |
| FFN | 2,368 SwiGLU | STRONG INFERENCE | Lands the complete architecture at 250.22M with a conventional ratio |
| Context | 4,096 native | EXPERIMENT REQUIRED | Doubles V4 while retaining feasible full attention |
| Attention | full causal in every layer | EXPERIMENT REQUIRED | Avoids locality as a confound in binding/state tests |
| Position | RoPE, base 10,000, no extrapolation claim | EVIDENCE-BACKED | Mature relative-position method; evaluate only native length |
| Norm | pre-RMSNorm, epsilon 1e-5 | EVIDENCE-BACKED | Stable and computationally simple |
| QK norm | enabled candidate with affine Q/K scales | EXPERIMENT REQUIRED | Stability evidence exists; discrimination effect is unknown |
| Embedding/output | tied | STRONG INFERENCE | Parameter efficient; no An-Ra evidence to untie |
| Bias/dropout | bias-free, dropout 0 | STRONG INFERENCE | Conventional at this scale and data volume |
| Initialization | normal std 0.02; residual output projections scaled by about `1/sqrt(2L)` | STRONG INFERENCE | Reduces depth-dependent residual growth |
| Precision | BF16 compute, FP32 reductions and optimizer state | EVIDENCE-BACKED | Appropriate TPU stability path |

Parameter calculation under the current An-Ra block conventions:

```text
embedding = 24,576 × 896 = 22,020,096
per layer = attention 2,408,448 + SwiGLU 6,365,184 + block norms 1,792 + QK norm scales 1,344 = 8,776,768
26 layers = 228,195,968; final RMSNorm = 896
total tied dense parameters = 250,216,960
```

The executable model constructor is authoritative; the freeze gate requires its exact count and a layer-by-layer receipt. **EXPERIMENT REQUIRED**

The framework-independent configuration/run receipt is executable now at `artifacts/v5/implementation_contract.json`, model-spec SHA-256 `d6c143e50d222def03cd46fb1b140248a58b3b06f56ac084b7bbf91ded43f2df`. It certifies arithmetic and schema invariants only; it does not instantiate a neural model or authorize training.

### 5.2 Explicit rejections for V5-A

Do not add MoE, recurrent memory, SSM blocks, latent-thought heads, learned routers, multi-token prediction, untied output heads, or dormant cognition modules to the baseline. They may become later ablations only after V5-A establishes a strong conventional control. **STRONG INFERENCE**

### 5.3 Scaling ladder

| Stage | Candidate shape | Approx. parameters | Purpose |
|---|---|---:|---|
| P35 | 16×384, 6Q/2KV, FFN 1024, vocab 24,576 | 34.6M | High-throughput rank ordering |
| M102 | 20×640, 10Q/2KV, FFN 1728, vocab 24,576 | 101.8M | Replication and scale-transfer gate |
| V5-A | 26×896, 14Q/7KV, FFN 2368, vocab 24,576 | 250.22M | First major run after freeze |
| V5-B | approximately 400M | OPEN / UNKNOWN | Only after V5-A passes and ≥8B clean tokens exist |
| 1B / 3B | unspecified | OPEN / UNKNOWN | Not justified by current evidence |

## 6. Tokenizer specification

### Provisional candidate

- Byte-fallback BPE or Unigram with exactly 24,576 entries.
- Identity-preserving preprocessing: no destructive Unicode normalization, case folding, accent stripping, whitespace collapse, or digit rewriting.
- All bytes representable; unknown-token rate must be zero.
- A minimal reserved set for BOS/EOS/PAD plus structural boundaries. No task-family, difficulty, answer-index, or latent-graph tokens.
- Train on the actual frozen data mixture, with source-balanced sampling so dominant web text does not erase code, mathematics, identifiers, and multilingual bytes.
- Serialize the tokenizer, trainer configuration, corpus sample manifest, normalization rules, special-token map, and SHA-256.

Status: **EXPERIMENT REQUIRED**. Token compression alone is not the objective. The tournament must measure raw-byte-normalized loss, tokens per byte by domain, exact nonce copying, identifier fragmentation, equation/code handling, and downstream cognition at matched raw bytes and FLOPs.

Required tournament: 16,384 versus 24,576 versus 32,768. A byte-level/BLT arm is research-only unless it wins a separate architecture-cost study; it changes too many variables for V5-A. **STRONG INFERENCE**

## 7. Data strategy

### 7.1 Main-run target

Target **5.0B unique, auditable training tokens** for V5-A, approximately 19.99 tokens per parameter. This is a reference budget, not a universal law. It must be counted with the frozen tokenizer after filtering and deduplication. **STRONG INFERENCE**

Provisional composition:

| Family | Share | Tokens | Status |
|---|---:|---:|---|
| High-quality natural substrate | 65% | 3.25B | EXPERIMENT REQUIRED |
| Code, math, formal and structured natural data | 20% | 1.00B | EXPERIMENT REQUIRED |
| Mechanically verified cognition data | 15% | 0.75B | EXPERIMENT REQUIRED |

The actual cognition share will be selected by a 5% / 15% / 30% proxy experiment. Generic web volume is not a substitute for provenance or quality.

### 7.2 Quality contract

Every source must have an origin, acquisition date, license/terms category, filtering version, deduplication receipt, tokenizer hash, and exact post-filter token count. The freeze requires:

- licensed, public-domain, or otherwise legitimately usable sources only;
- document- and span-level deduplication within and across families;
- benchmark contamination search before tokenization and again after canonicalization;
- explicit removal of secrets, private data, malware instructions, template spam, and low-information repetition;
- held-out natural domains and sources, not merely held-out random rows;
- exact mixture sampler and seed captured in checkpoints.

Model-based filtering may help rank quality, but filters must be audited for stylistic monoculture and must not silently turn the corpus into another model's distribution. **EVIDENCE-BACKED**

### 7.3 Synthetic cognition contract

Preferred generators are programs, simulators, solvers, databases, or formal transforms with executable truth. LLM-generated paraphrases are quarantined, traceable, mechanically semantics-checked, and capped at 5% of total tokens unless a later experiment justifies more.

Each example stores a hidden evaluator record containing its dependency graph, correct answer, plausible distractors, counterfactual queries, intervention set, difficulty, generator version, and seed. Those fields are never leaked as input shortcuts.

Splits must be disjoint across:

- generator seeds and template families;
- entity/token alphabets;
- graph topologies and relation compositions;
- cardinality and hop count;
- context position and distractor density;
- surface style, domain, and answer format.

Recursive synthetic imitation without access to original human distributions is prohibited because it can amplify errors and erase low-probability structure. **EVIDENCE-BACKED**

## 8. Cognition curriculum

V5 should learn a small set of transferable operations rather than task names:

1. exact copy and localized retrieval;
2. query-conditioned entity/value binding;
3. mutable state and overwrite precedence;
4. retrieval under distractors and adversarial position;
5. relational composition at one, two, then three hops;
6. counterfactual query and state sensitivity;
7. rule induction and application on novel symbols;
8. missing-information recognition;
9. realization of a correctly selected value in natural and constrained forms.

Curriculum is **[EXPERIMENT REQUIRED]** because broad language-model curriculum evidence is mixed. Compare uniform interleaving with one staged schedule: establish copy/binding/state, add composition, then raise cardinality and distractors while preserving at least 30% replay of earlier families. Advancement is determined by preregistered competence gates, not subjective inspection. If staging does not improve worst-family fresh-OOD performance at equal tokens, use uniform mixing.

The provisional sequence-length mix, measured by tokens rather than examples, is 50% at approximately 512–1,024, 30% at 2,048, and 20% at 4,096. The 4k share is enriched for position, distractor, state, and retrieval controls; relevant spans are position-randomized. This preserves native 4k training without charging quadratic attention cost to every datum. **[EXPERIMENT REQUIRED]**

Natural transfer examples must accompany each synthetic family. A synthetic-only gain is not sufficient for promotion.

## 9. Training objective

### 9.1 Base objective

Standard causal next-token likelihood remains the universal objective. **EVIDENCE-BACKED**

### 9.2 Query-swap auxiliary objective

The only auxiliary objective allowed into the decisive experiment is query-swap contrast. For fixed fact context `F`, correct query `q`, counterfactual query `q'`, gold value `v*`, and plausible wrong values `v-`:

```text
lift(v, q) = log P(v | F, q) - mean_q' log P(v | F, q')
L_query = log(1 + sum_v- exp(m + lift(v-,q) - lift(v*,q)))
L_total = L_LM + lambda * L_query
```

The counterfactual changes the causal question while facts remain fixed. Candidate lengths must be normalized, candidates must be position-balanced, and wrong candidates must be plausible values from the same context. Apply this only to mechanically verified cognition examples, not arbitrary natural text.

Candidate sweep: `lambda ∈ {0, 0.05, 0.15}`; a broader `{0, 0.1, 0.3}` sweep is allowed only if the first range is underpowered. The objective is admitted only if it improves fresh query-swap selection and natural transfer without more than a 3% substrate-loss regression. **EXPERIMENT REQUIRED**

Do not use the SFT7 same-query margin as the primary selection objective: it did not improve the intended independent-fixture rank-1 selection result, despite smaller lift and greedy-realization effects. Do not create a separate realization head: train realization with ordinary answer-span likelihood and evaluate it conditionally on correct selection. **EVIDENCE-BACKED**

## 10. Optimization and schedule

| Item | Provisional choice | Status |
|---|---:|---|
| Optimizer | AdamW | EVIDENCE-BACKED |
| Betas | 0.9, 0.95 | STRONG INFERENCE |
| Epsilon | 1e-8 | STRONG INFERENCE |
| Weight decay | 0.1 on weight matrices; none on norm/bias | STRONG INFERENCE |
| Peak LR | 3e-4 | EXPERIMENT REQUIRED |
| LR candidates | 2e-4, 3e-4, 4e-4 | EXPERIMENT REQUIRED |
| Global tokens/update | 131,072 center | EXPERIMENT REQUIRED |
| Batch candidates | 131,072 / 262,144 | EXPERIMENT REQUIRED |
| Schedule | WSD: 1% warmup, 89% stable, 10% decay to 0.1× peak | EXPERIMENT REQUIRED |
| Gradient clip | global norm 1.0 | EVIDENCE-BACKED |
| Precision | BF16; FP32 reductions and optimizer | EVIDENCE-BACKED |

At 5B tokens, the center batch yields about 38,147 optimizer updates; 262,144 yields about 19,073. The smaller center is preferred until E4 shows that the larger batch's throughput compensates for lower update resolution. Do not rewarm LR merely because a data pack or session changes. A continuation resumes the global schedule and optimizer state. Batch changes require an explicit migration experiment and receipt.

Every canary verifies that a real optimizer update occurred: parameter SHA changes, optimizer maximum step increments exactly, moments change, nonzero finite gradients exist, data cursor advances, and all live parameters belong to the optimizer.

## 11. Checkpoint and evaluation system

### Checkpoints

- Mutable full-resume recovery checkpoint every 10M tokens; retain at least two and copy durably off worker.
- Immutable full-resume milestone every 100M tokens, every 50M over the final 500M, and at every data/curriculum boundary.
- Every checkpoint contains model, optimizer, scaler if any, sampler/cursor, RNG states, global step, exact cumulative tokens from zero, tokens by family/source, config, tokenizer/data/source commit hashes, parent hash, and parameter hash.
- Resume equivalence test: uninterrupted `N+K` updates must match `N → save → resume → K` within a defined BF16 tolerance.

Status: **EVIDENCE-BACKED** for full-resume integrity and immutable milestones; cadence is **STRONG INFERENCE** pending storage/latency measurement.

### Evaluation cadence

- Tier 0 canaries every 25M tokens: 32 cases per family, integrity and gross regression detection.
- Tier 1 every 100M tokens, asynchronous: 512 cases per family plus substrate/code/math loss.
- Tier 2 sealed promotion suite: 1,024 cases per family, used only at preregistered boundaries.
- Tier 3 fresh replication: newly generated fixtures and held-out natural domains before promotion.

Training must not wait on routine behavioral evaluation. A separate evaluator consumes immutable checkpoints.

## 12. Cognition benchmark and promotion gates

### 12.1 Measurement cube

Every family is measured at three levels:

- **Representation:** gold candidate NLL/rank and query-conditioned lift relative to matched counterfactual queries.
- **Selection:** raw argmax accuracy, calibrated margin, query-flip correctness, and abstention where information is absent.
- **Realization:** free generation exact/semantic correctness, also reported conditional on the gold candidate already being selected.

This prevents a generation failure from being mislabeled as a representation failure, or an assisted candidate score from being claimed as native free-generation ability. **EVIDENCE-BACKED**

### 12.2 Anti-memorization / OOD design

Report independent and joint shifts in symbols, lexical aliases, templates, domains, answer formats, context positions, distractor density, graph topology, cardinality, hop count, and sequence length. Compare composition against direct-retrieval controls with identical vocabulary and surface statistics. Use adversarial irrelevant facts and query swaps. Keep sealed fixtures unavailable to training/data filtering agents.

### 12.3 Provisional promotion gates

Exact thresholds will be calibrated in E0, but the intended gates are:

- integrity and contamination checks pass with no exception;
- lower 95% confidence bound for fresh-OOD selection is at least chance + 10 percentage points;
- query swaps move selection in the correct direction on at least 80% of cases;
- state/overwrite OOD accuracy at least 70%;
- two-hop composition OOD at least 60% and materially above matched retrieval control; three-hop shows positive scaling rather than collapse;
- realization conditional on correct selection at least 80%;
- natural substrate loss no worse than 3% versus matched LM-only control and code/math loss no worse than 5%;
- no cognition family regresses by more than 5 percentage points;
- decisive gains reproduce across two seeds at ~102M and once on a fresh fixture/domain.

Use worst-family gates, not a compensating average. Report unassisted and assisted scores separately. **STRONG INFERENCE**, thresholds **EXPERIMENT REQUIRED**.

## 13. Decisive experiments before freeze

The original 100+ questions collapse into six decisions. Optimize information gain per accelerator-hour using matched tokens/FLOPs, paired fixtures, early pruning only at preregistered checkpoints, and replication of finalists.

### E0 — Benchmark and generator certification (no TPU required)

**Development status in v0.4:** an executable evaluation-only generator emits 368 deterministic cases and 112 mechanically checked causal pairs at the certification setting. State cases randomize serialization independently of semantic time and cover latest, intermediate, rollback, and precedence queries over interleaved variables. Rule induction covers eight latent operand structures in development, with disjoint structures in sealed/fresh. Training generators use a separate template namespace. Candidate order, context position, answer formats, state-query axes, rule structures, and difficulty axes are explicit audited dimensions; hidden truth is excluded from model views; development/sealed/fresh graph/template/domain/rule-structure namespaces are mechanically disjoint. Representation, selection, query-lift, realization, sensitivity, invariance, assistance, and intervention-dependence metrics are separate APIs.

The committed development receipt is `artifacts/e0/development_certificate.json`, suite SHA-256 `2b0204a37cde1762d47cdb6088542a7eb4307b3d8904d5f1ffad496c312d7d7e`. It reports random/first/last/lexical/position/bag-of-words, broken-state, fixed-rule, direct-retrieval, and full-oracle controls. The state positional red-team and rule shortcut red-team pool eight independent seeds; positional baselines are judged against an exact permutation-calibrated null (rule baselines use uniform-candidate chance) plus 10 percentage points. An independent surface parser/solver agrees on every case, including a 20-seed property sweep. Uniform-candidate chance, Wilson intervals, difficulty curves, pair-effect counts, and approximate sample-size planning are explicit. **This certifies infrastructure invariants, not benchmark difficulty and not model cognition.**

**Still required for E0 exit:** a source-disjoint natural set and a real T2 seed/fixture held externally with only its commitment hash in Git. Paired exact binary comparisons, 10,000-resample paired bootstrap score deltas, Wilson intervals, worst-family gates, and sealed-consumption semantics are now machine-preregistered. Until external custody closes, E0 is **DEVELOPMENT PASS / PROMOTION NOT FROZEN** and E1 model training remains unauthorized.

Abort if shortcuts, duplicate graphs, token-label leakage, scorer ambiguity, or custody leakage remain.

### E1 — Tokenizer tournament

Compare 16k, 24k, and 32k on static corpus metrics and matched P35 training (100–200M tokens per serious arm). Match raw bytes and approximate FLOPs, not token count alone. Winner must be Pareto-competitive on byte-normalized substrate loss, sequence inflation, nonce copy, identifiers, code/math, and cognition.

The artifact-bound static audit, Pareto harness, and matched-budget tournament plan are implemented in `e1_tokenizer/`. A reproducible local development run independently trained exact 16k/24k/32k byte-BPE artifacts on 8.56 MB and evaluated 1.06 MB of content-hash-held-out records: compressed artifacts reload, all round trips are exact, unexpected unknowns are zero, and a repeated 24k build is byte-identical. Tokens/byte are 0.23826/0.23217/0.23022, placing 24k only 0.85% behind 32k for 7.34M fewer embeddings at width 896. This retains 24k as the planning center but does not select it: the fixed legacy/local corpus is not representative and no matched model loss or cognition exists. `e1_tokenizer.tournament` still fails closed until an external corpus manifest and equal raw-byte/FLOP budgets are bound.

### E2 — Architecture screen

At ~35M parameters compare parameter-matched deep/narrow, middle, and wide/shallow shapes; screen 4-KV GQA versus MHA and QK norm on/off using a fractional design. Single-seed successive halving may eliminate clearly weak arms; top two shapes receive three seeds.

The purpose is to decide depth, attention topology, and QK/GQA—not to crown a benchmark winner from one seed. A replicated local CUDA kernel probe already discovered that this exact PyTorch/Windows build routes native GQA through the math backend: it is 5.20× MHA latency and 13.86× peak allocation, while explicit repeated K/V is 1.09× latency and 1.45× memory. Native 4k is 3.59× native-2k latency and 3.80× memory. This is **EVIDENCE-BACKED for implementation selection on the measured stack**, not evidence against GQA or 4k on TPU; the target framework must run the same backend canary before E2.

### E3 — Data/objective screen

At the winning P35 architecture compare 5%, 15%, and 30% cognition mixtures under LM-only. Add query-swap contrast to the best mixture and one neighboring mixture. Compare with matched token and raw-byte budgets. Replicate finalists across three seeds and fresh generators.

This resolves whether gains come from structured contrasts in the data or the explicit objective.

### E4 — Curriculum and optimization

Compare uniform interleaving against one competence-staged curriculum with replay. Only after recipe selection, test LR `2e-4/3e-4/4e-4`, effective batch `131k/262k/524k`, and WSD versus the simplest competitive schedule if curves disagree. Do not run a combinatorial sweep.

### E5 — Mid-scale replication

Train the ~102M recipe and a strong LM-only control for 600M–1B tokens, two seeds for the winning recipe. Require sealed and fresh-OOD gains, natural transfer, stable training, and checkpoint-resume equivalence. If the effect does not survive scale, do not launch V5-A.

### E6 — Freeze review

Resolve all checklist items; build executable parameter/compute/data receipts; freeze `V5_TRAINING_SPEC_v1.0.md`; run target-TPU canaries and remote checkpoint restore. Only then authorize the 250M/5B run.

## 14. Compute and token estimates

For the provisional 250,216,960-parameter model and 5.0B tokens, the common dense-transformer estimate `6ND` is:

```text
6 × 250,216,960 × 5,000,000,000 = 7.50651e18 FLOPs ≈ 7.51 EFLOP
```

Allow roughly 10–25% additional practical cost for attention, evaluation, checkpointing, padding, and pipeline inefficiency: about **8.3–9.4 EFLOP** for the main run. Pure compute time is approximately 41.7 / 20.8 / 10.4 hours at sustained 50 / 100 / 200 TFLOP/s; realistic wall time is provisionally **14–55 hours** after input and checkpoint overhead. These are device-neutral estimates, not a Kaggle promise. **STRONG INFERENCE**

The pre-freeze program should be capped near **1.5–2.5 EFLOP**, with most weak P35 arms stopped early. This is cheap relative to an uninterpretable main run. **STRONG INFERENCE**

Storage must be measured from an actual schema-3 full-resume checkpoint. Budget immutable milestones separately from two rotating recovery checkpoints.

## 15. Failure and abort criteria

Abort or pause an arm when any of the following occurs:

- non-finite loss/gradients, optimizer/live-parameter mismatch, token ledger inconsistency, or failed resume equivalence;
- source/license/contamination/hash violation—results from that data are invalid;
- query objective fails to beat LM-only on fresh query-swap selection with its preregistered confidence interval, or harms substrate loss by more than 3%;
- synthetic gains fail on fresh generators and held-out natural analogues;
- two consecutive Tier 1 evaluations show a >5-point cognition decline while LM loss improves; preserve and compare the earlier checkpoint;
- no replicated cognition gain at ~102M; do not scale to 250M;
- target-hardware throughput, durable checkpoint upload, and restore canaries do not pass;
- benchmark artifacts or direct-retrieval shortcuts explain the apparent composition gain.

An inconclusive result is not a license to average arms or select the prettiest curve. Increase power only when the decision value exceeds the compute cost.

## 16. Scaling roadmap

1. Certify benchmark and data-generators.
2. Run P35 tokenizer, shape, mixture, objective, and minimal optimization screens.
3. Replicate finalists, discard unreplicated mechanisms, and publish negative results.
4. Train M102 control and winning recipe; require transfer and resume integrity.
5. Freeze V5-A exact spec, manifests, seeds, gates, and commands.
6. Train one 250M/5B main run with immutable behavioral checkpoints; do not call the final checkpoint best automatically.
7. Conduct sealed and fresh post-training evaluation, mechanistic probes, contamination audit, and independent rerun of decisive claims.
8. Consider ~400M only if V5-A is capacity-limited by controlled evidence, at least 8B clean tokens are ready, and scaling predicts a useful gain per compute.
9. Consider 1B/3B only after the recipe transfers twice and the data/provenance program can support it.

## 17. Core versus Connector

### Train natively into Core

- contextual retrieval and exact copying;
- query-conditioned binding;
- local working-state update and overwrite precedence;
- short relational composition;
- counterfactual sensitivity;
- internal candidate discrimination;
- reliable realization;
- evidence-sensitive missing-information behavior.

These are repeated, local computations required across domains. **STRONG INFERENCE**

### Keep in Connector/runtime

- tool execution and API schemas;
- retrieval orchestration and durable external memory;
- long-horizon search, branching, budgeting, and planning;
- verification, causal ablation, and credit assignment;
- adaptive routing, permissions, safety/risk policy, and recovery;
- counterfactual normalization as a diagnostic or fallback until native transfer is demonstrated.

These change with tools, environment, risk, or horizon and are easier to audit externally. **STRONG INFERENCE**

Do not train transient tool syntax or one deployment's routing policy into the foundation unless it demonstrates broad, stable transfer.

### Keep in the Outer layer or independent evaluator

- permissions, irreversible-action approval, user authority, safety/risk policy, and deployment budgets;
- ground-truth success, sealed benchmark custody, promotion authority, and evidence-signing.

Core cannot grade itself and Connector intervention output cannot be its own success label. **[FROZEN]**

## 18. Unresolved decisions

| Decision | Current state | Resolver |
|---|---|---|
| 16k vs 24k vs 32k vocabulary | EXPERIMENT REQUIRED | E1 |
| 26×896 versus another ~250M depth/width shape | EXPERIMENT REQUIRED | E2 |
| 4-KV GQA versus MHA | EXPERIMENT REQUIRED | E2 |
| QK norm effect on discrimination | EXPERIMENT REQUIRED | E2 |
| Full attention benefit at 4k | EXPERIMENT REQUIRED | E2 plus long-context suite |
| 5/15/30% cognition share | EXPERIMENT REQUIRED | E3 |
| LM-only versus query-swap auxiliary | EXPERIMENT REQUIRED | E3 |
| Uniform versus staged curriculum | EXPERIMENT REQUIRED | E4 |
| Exact LR/batch/WSD schedule | EXPERIMENT REQUIRED | E4 |
| Whether 250M can robustly learn 3-hop composition | OPEN / UNKNOWN | E5/V5-A evidence |
| Whether synthetic causal gains transfer broadly to natural reasoning | OPEN / UNKNOWN | E3/E5 fresh natural domains |
| Whether ~400M materially outperforms better data at equal compute | OPEN / UNKNOWN | post-V5 scaling study |

## 19. Exact sequence from today

1. Treat the v10/v11 EXP expansion as contaminated; do not derive a V5 composition objective from it.
2. Finish E0 shortcut balancing, power calibration, and natural-source custody checks.
3. Freeze Tier 2 fixtures under external custody, with split rules, metrics, chance controls, and only commitment hashes in Git.
4. Produce the auditable candidate corpus and tokenizer training sample.
5. Execute E1 and E2 at P35 with matched budgets and preregistered stopping.
6. Execute E3; replicate finalists before reading sealed Tier 2.
7. Execute E4 only for remaining decisions; avoid a broad hyperparameter search.
8. Execute E5 at M102 with a strong LM-only control and exact-resume tests.
9. Hold a freeze review; resolve the checklist and issue `V5_TRAINING_SPEC_v1.0.md` with executable receipts.
10. Run target-TPU one-step, multi-step, save/resume, upload/download, and evaluator canaries.
11. Authorize and launch V5-A only if every gate passes.
12. Evaluate intermediate and final checkpoints; promote the evidence-backed checkpoint, not necessarily the last one.

## 20. Verdict

**V5 CORE WE SHOULD PROBABLY BUILD:** A 250.22M dense 26-layer × 896-width causal Transformer, 14 Q / 7 KV heads, SwiGLU 2368, full 4k attention, affine QK norm, RoPE, pre-RMSNorm, tied byte-fallback tokenizer embeddings, trained on 5B audited tokens.
**WHY:** V4 is not proven capacity-limited; it is under-evidenced, under-tokened, and behaviorally mis-selected. A clean dense model makes the data/objective hypothesis falsifiable.
**BIGGEST CHANGE FROM V4:** A clean-sheet foundation with more depth and native full 4k context at nearly the same scale, selected by causal proxy experiments rather than inherited code or intuition.
**BIGGEST DATA CHANGE:** A provenance-complete 5B-token corpus with a tested ~15% mechanically verified causal cognition component.
**BIGGEST TRAINING CHANGE:** Query-swap contrast as the sole candidate auxiliary objective, plus immutable behavioral checkpoint promotion.
**BIGGEST COGNITION CHANGE:** Train and measure representation, selection, and realization separately under causal counterfactual and OOD splits.
**MOST IMPORTANT UNKNOWN:** Whether query-swap gains survive fresh generators, natural domains, and scale without harming the language substrate.
**EXPERIMENTS BEFORE FREEZE:** E0 benchmark certification; E1 tokenizer; E2 architecture; E3 data/objective; E4 minimal curriculum/optimizer; E5 102M replication.
**ESTIMATED COMPUTE:** 1.5–2.5 EFLOP pre-freeze; 7.51 EFLOP idealized and roughly 8.3–9.4 EFLOP practical for V5-A.
**CONFIDENCE:** 0.72 in the program direction; 0.45 in the exact provisional shape and mixture.
**READY TO FREEZE:** **NO**.
**NEXT ACTION:** Obtain external E0 custody and real 16k/24k/32k tokenizer artifacts; run the implemented E1 static audits, then authorize only bounded P35 comparisons after E0 exits.

## 21. Research basis

The internal claim ledger, exact previous-branch references, public primary-source bibliography, contrary evidence, and invalidated claims are consolidated in [`EVIDENCE_BASE.md`](EVIDENCE_BASE.md). The four rounds that transformed those inputs into this design are in [`ITERATIONS.md`](ITERATIONS.md).
