# ESOES Evidence and Context Base

This document records the evidence that should shape the next An-Ra Core. It is not a mythology file. Claims are separated into demonstrated, promising, contaminated, and unknown.

---

## 1. Canonical V4 starting point

Historical V4 dense Core:

- vocabulary: 32,768
- width (`d_model`): 896
- layers: 18
- query heads: 14
- KV heads: 2
- head dimension: 64
- SwiGLU hidden width: 2432
- context: 2048
- dense executable parameters: about 180.1M
- tied embeddings/output
- RMSNorm
- RoPE
- QK norm
- GQA
- hybrid sliding/full causal attention

The architecture is a valid baseline, not a sacred design.

The old broader system had significant runtime/cognition infrastructure, but much of its “self-understanding” was implemented as hard-coded categories and control scaffolding rather than learned latent cognitive structure.

---

## 2. Core scientific direction discovered in EXP

The most valuable conceptual shift was from generic “self-improvement” toward **cognitive credit assignment**.

After a failure, the system should eventually be able to answer:

> I failed. Which internal/runtime variable caused the failure? What intervention should repair it? What evidence shows that diagnosis was correct? Will the same repair predictably work again?

The useful decomposition that emerged is roughly:

- knowledge / representation
- query-conditioned selection
- retrieval
- planning / composition
- memory / state
- realization / decoding
- tool / execution failure

These labels are hypotheses and measurement axes, not permanent ontology.

The strongest scientific criterion is predictive:

> Does the learned diagnosis predict which intervention will repair a new failure better than baselines that do not possess that diagnosis?

---

## 3. Selective binding and SFT history

Early selective-binding work showed that replay could improve retention without robustly transferring selective query binding.

Important primitive findings:

- models could sometimes extract a fact while failing to condition the selected value on the actual query;
- query swaps often failed to move the output appropriately;
- improving surface query recognition did not automatically create correct value selection.

This led to direct measurement of query-conditioned candidate probability rather than relying only on free generation.

### SFT6

SFT6 was the strongest clear training result in this line.

Parent checkpoint parameter SHA:

`36c87e5be671ff37951b4b433e21ec14bb5e59f6f0770d7f81e50e99ead9e001`

On QIM-v3, relative to its parent, SFT6 produced a large increase in query-conditioned lift and improved rank-1 candidate selection. It showed that targeted training could create a real query-conditioned preference signal.

Generation decomposition also showed two distinct problems:

1. **selection failure** — the model prefers the wrong candidate;
2. **realization failure** — the model's candidate ranking contains the correct answer, but free generation does not emit it.

This distinction is now foundational and should be present in all V5 evaluation.

### SFT7

SFT7 used a same-query candidate-margin objective. It did **not** improve the preregistered selection hypothesis, though free/greedy behavior improved somewhat.

Interpretation:

> SFT7 behaved more like a decode-commitment/realization specialist than a solution to robust selective binding.

This is useful evidence that different training objectives can move different failure mechanisms.

---

## 4. Counterfactual query normalization

One of the strongest discoveries in the project is counterfactual query normalization.

For candidate/value `v_i` and actual query `q_k`:

`RAW_i = log P(v_i | facts, q_k)`

`BASELINE_i = mean_j!=k log P(v_i | facts, q_j)`

`ADJUSTED_i = RAW_i - BASELINE_i`

Select the candidate with maximum `ADJUSTED_i`.

Interpretation:

- RAW score contains candidate/context intrinsic preference;
- counterfactual queries estimate that prior;
- subtraction exposes the probability change specifically associated with the actual query.

### QIM-v5

Frozen SFT6 checkpoint on a fresh 149-target fixture:

- FREE: 35/149
- RAW: 70/149
- CONSTRAINED: 73/149
- NORMALIZED: 106/149
- NORM_EXACT: 106/149

Normalization improved by +36 over RAW and repaired 53/79 RAW selection failures, though it also caused 17 RAW pass→fail regressions.

The defensible mechanism statement is:

> **Candidate-specific priors can hide a real query-conditioned signal; counterfactual query normalization can expose that signal.**

It should not be interpreted as universally dominant.

---

## 5. Observed-only intervention routing

An early self-model experiment accidentally used evaluator/ground-truth-derived fields and was invalid as observed-only evidence. That was corrected.

A later observed-state policy used runtime-visible features only, such as:

- candidate count
- output arity / format
- RAW and normalized score geometry
- margins / spreads
- action agreement
- free-output agreement
- query-lift-related statistics

### Strong 3-action evidence

A frozen policy routing among:

- NO_CHANGE
- CONSTRAINED
- NORMALIZED

showed meaningful prospective performance.

MC-v7:

- adaptive: 310/480
- best tested fixed baseline: 274/480

MC-v8, fresh second draw using the exact same frozen policy:

- adaptive: 291/480
- best tested fixed baseline: 248/480

SFT6→SFT7 checkpoint transfer:

- adaptive: 320/480
- always-normalized: 280/480

Defensible statement:

> A frozen observed-only policy repeatedly routed among three runtime interventions better than the tested fixed/simple baselines on fresh mixed-failure fixtures, and the signal transferred to one nearby checkpoint in the same architecture/lineage.

Important limitation:

A strong structure-only router was not fully established as a baseline, so the extent to which the policy models internal failure state versus external task structure remains an open scientific question.

---

## 6. Expanded pair-action evidence: promising but contaminated

Later policies added:

- EXACT_PAIR
- SLOT_PAIR

These were intended to repair ordered multi-emission/composition cases.

However MC-v10/v11 runners contained a stale-local bug: pair-action applicability used `candidates` before assigning the current task's candidate list. Pair availability therefore depended on the previous shuffled task.

This contaminated expanded-action composition claims.

Additional weaknesses:

- fixed pair-action baselines were missing;
- per-family `best_const` was actually a per-case action oracle, not a constant;
- the exact learning process for later frozen policy artifacts was not reproducible from the committed trainer;
- development harvesting included outcome-conditioned row deletion.

Therefore:

> The 4/5-action results are promising research leads, not clean promotion evidence.

V5 should preserve the lesson — composition likely requires distinct mechanisms — without inheriting the contaminated headline.

---

## 7. Verified Intervention Experience (VIE) caution

The project accumulated 166 claimed VIEs, but the bank did not satisfy its own strict causal contract.

A valid VIE is supposed to hold everything constant except one controlled variable and observe baseline failure → intervention success.

In practice, comparisons such as FREE → NORMALIZED often changed both:

- selection mechanism;
- realization/decode mechanism.

Some metadata also claimed decode configuration was held constant when decoding itself was the intervention.

Therefore the 166 count should be treated as provisional/not qualified.

Lesson for V5:

> causal records must be generated from structured before/after configurations whose diff can be mechanically checked to contain exactly the intended variable change.

---

## 8. Minimal-intervention objective not yet demonstrated

Later routing policies stored intervention costs but used `lambda = 0`, meaning costs did not influence the decision.

Thus the demonstrated result is closer to:

> learned repair-success routing

than:

> learned smallest/cheapest effective intervention selection.

Future self-improvement work should explicitly model success probability, intervention cost, risk, and unnecessary-change rate.

---

## 9. Critical VNEXT training failure and repair

A historical `core-vnext` continuation appeared to advance training metadata while the persisted model did not actually update:

- 203/203 tensors were bitwise identical to the parent;
- Adam moments were unchanged;
- Adam step remained 20,000;
- generations were identical.

The likely cause was optimizer/model parameter identity drift after moving the model to XLA: the optimizer retained stale CPU Parameter objects while the live XLA model received gradients.

This established mandatory invariants for future training:

- optimizer parameter objects must exactly match live model parameters;
- optimizer/model devices must agree;
- gradients must exist and be finite;
- Adam step must advance;
- parameter SHA must change after real updates;
- optimizer moments must change.

Current `core-vnext` includes explicit optimizer rebinding and first-update checks for this failure class.

---

## 10. Current PGE/Core audit

Current audited `core-vnext` head at the time ESOES was created:

`054619f20851317e9b1c49b6f31599f6444a8280`

The step-22,517 checkpoint is real:

- final parameter SHA differs from parent;
- Adam first and second moments differ;
- optimizer/global/trainer steps are consistent;
- strict final tokenizer contract loads;
- the continuation pack completed.

### Token provenance

The often-repeated ~2.7B/~2.95B lifetime token interpretation is **not certified**.

What is certified:

- 2,517 continuation optimizer steps
- 131,072 tokens/optimizer step
- 329,908,224 continuation tokens

Historical parent/campaign metadata conflict, so lifetime token count is not currently trustworthy.

This is a governance lesson: future V5 training should use one unambiguous cumulative token ledger from step zero.

### PGE behavioral result

Same-probe audit:

| Capability | Parent 20k | Step 21.8k | Final 22.517k | SFT6 | SFT7 |
|---|---:|---:|---:|---:|---:|
| Held-out loss ↓ | 2.1884 | 2.0156 | **1.9710** | not run | not run |
| Exact copy RAW | 0/6 | 0/6 | 0/6 | **5/6** | 4/6 |
| Nonce context RAW | 0/8 | 0/8 | 0/8 | **7/8** | **7/8** |
| Multi-fact FREE | 0/48 | 0/48 | 0/48 | 3/48 | 2/48 |
| Candidate selection RAW | 12/48 | 11/48 | 12/48 | 13/48 | 13/48 |
| Counterfactual normalization | 12/48 | 12/48 | 10/48 | 9/48 | 9/48 |
| Composition FREE | 0/12 | 0/12 | 0/12 | 0/12 | 0/12 |
| Raw degenerate continuations | 3/4 | 1/4 | 3/4 | 0/4 | 0/4 |

With four candidates, chance is 12/48.

Therefore:

- PGE strongly improved ordinary next-token modeling;
- it did not produce robust query-conditioned candidate selection on this battery;
- it did not natively recover SFT6's exact copy/context-use behavior;
- composition was not demonstrated;
- final loss improved while some generation behavior regressed relative to the intermediate checkpoint.

The most important lesson is not “PGE is useless.” It is:

> **Generic next-token improvement and targeted cognitive capability are not the same learning objective at this scale/recipe.**

### Audit limitations

The current PGE audit itself should not be overinterpreted:

- copy/context sample counts are small;
- free-generation tests mix representation with realization;
- the “constrained” arm in the audit is deterministic external emission of the model-selected candidate, not another constrained model forward pass;
- degeneration uses only four creative prompts;
- the branch CI validates infrastructure but does not independently rerun all GPU behavioral receipts.

Thus the final-vs-21.8k behavioral winner is not fully settled.

---

## 11. Current architectural philosophy

An-Ra is conceptually separated into:

1. **Core** — learned neural substrate; expensive and relatively stable.
2. **Connector** — planning, memory strategy, retrieval, verification, self-modeling, experiment design, intervention routing.
3. **Outer Layer** — tools, interfaces, actions, environment interaction.

Useful shorthand:

> The Core computes. The Connector decides/experiments. The Outer Layer acts.

For V5, this boundary becomes a learning loop:

1. Core fails.
2. Connector decomposes/diagnoses the failure.
3. A controlled intervention proves or rejects a repair.
4. Repeated verified repairs become curriculum evidence.
5. Future Core training attempts to internalize the useful computation.
6. Connector intervention frequency should fall if internalization succeeds.

The Connector should not become a permanent prosthetic for capabilities that should exist in the foundation.

---

## 12. Scientific status entering ESOES

### Strong / useful evidence

- real V4 Core baseline exists;
- targeted training can create query-conditioned signal;
- selection and realization are separable failure mechanisms;
- counterfactual normalization can expose hidden query-conditioned signal;
- a frozen observed-state 3-action policy has prospective same-policy replication on its native mixed-failure setting;
- that routing signal transferred to one closely related checkpoint;
- current repaired TPU continuation performs real optimizer/model updates;
- generic PGE improves LM loss substantially.

### Not established

- robust selective binding across broad distributions;
- robust multi-hop composition;
- architecture-general self-modeling;
- universal intervention routing;
- learned minimal-intervention selection;
- clean causal VIE bank;
- full cognitive credit assignment;
- AGI.

### Current training decision

Do **not** respond to the weak cognitive foundation by blindly adding more generic tokens.

The next major training campaign should happen only after a cognition-first training path is designed and smaller discriminating experiments justify its choices.

---

## 13. Primary lesson carried into V5

The previous pipeline effectively asked:

> Can a competent small language model later be patched into stronger cognition?

ESOES should instead ask:

> **What foundation training process causes the Core itself to acquire the cognitive primitives that later self-improvement depends on?**

That is the design problem for the next Core.