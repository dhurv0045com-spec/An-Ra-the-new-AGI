# An-Ra V5 — Cognition-First Foundation Blueprint (Working Draft)

Status: **SUPERSEDED WORKING DRAFT**

The authoritative implementation constants are now in
[`../../blueprint/V5_TRAINING_SPEC_v1.0.md`](../../blueprint/V5_TRAINING_SPEC_v1.0.md). This file is retained as
research history; its older mixture ranges and unset constants must not be used
by code or a training run.

This document defines how the next An-Ra Core should be designed and how the design itself should be validated before a large training run.

The central rule is:

> **Choose architecture, tokenizer, data, optimization, and scale according to cognitive learning behavior — not according to tradition or loss alone.**

---

# 1. Objective

Build a small but unusually strong neural foundation that can acquire and retain cognitive primitives needed for later self-improvement.

The immediate target is not “AGI in one training run.”

The target is a Core that is materially stronger than V4 in:

1. query-conditioned binding;
2. use of information supplied only in current context;
3. working-state tracking and overwrite;
4. relational and multi-hop composition;
5. counterfactual sensitivity;
6. distractor-resistant selective retrieval;
7. internal candidate discrimination;
8. reliable realization of internally selected answers;
9. generalization to unseen symbols/templates;
10. learning efficiency on new cognitive tasks.

These should emerge as measurable training curves, not as post-hoc anecdotes.

---

# 2. Non-negotiable design principles

## 2.1 Cognition must be measured during foundation training

Every important checkpoint is evaluated on both:

### Language / knowledge substrate

- held-out loss
- perplexity by domain
- code/math/science performance
- generation quality/diversity

### Cognitive substrate

- binding
- context use
- state update
- composition
- counterfactual response
- retrieval under distractors
- missing-information behavior
- realization

A model is not promoted only because loss is lower.

## 2.2 Representation must be separated from realization

For cognitive tasks measure at least:

- `FREE`: normal generation
- `RANKED`: whether the correct candidate/state has highest model probability
- `CONTROLLED_REALIZATION`: whether output succeeds when semantic selection is held fixed

Where useful also measure:

- candidate margin
- query-swap score change
- counterfactual normalization
- calibration / entropy

This prevents a decoder failure from being mistaken for a knowledge failure and vice versa.

## 2.3 Controlled contrasts are first-class training examples

Training should contain related examples that differ in exactly one causally relevant variable.

Examples:

- same facts, different query;
- same query, one changed fact;
- same entities, changed relation;
- same state history, one overwrite;
- same target, changed distractor;
- same graph, one changed edge;
- same rule, different symbols;
- same surface form, different latent structure.

The goal is to force the model to represent conditional structure rather than merely memorize common continuations.

## 2.4 Generalization must defeat template memorization

Every synthetic cognitive family must have explicit splits over multiple dimensions:

- entity vocabulary
- value vocabulary
- template/paraphrase
- ordering
- graph topology
- number of distractors
- hop count
- sequence length
- surface domain

A capability is not counted as learned if it only survives within-template splits.

## 2.5 Scale only after evidence

Before a long expensive run, use smaller models and shorter token budgets to answer architecture/data questions.

Large training is an execution step after design evidence, not a substitute for design evidence.

---

# 3. V5 cognition specification

## C1 — Exact contextual retrieval

Given information present only in the prompt, recover the requested item reliably.

Tests:

- nonce strings
- arbitrary identifiers
- uncommon symbols
- key/value records
- long distractor blocks

Measure FREE and candidate/logprob recognition separately.

## C2 — Query-conditioned binding

Given several simultaneous facts, choose the value belonging to the requested entity/relation.

Difficulty ladder:

- 2 bindings
- 4 bindings
- 8 bindings
- 16 bindings
- distractors
- paraphrases
- shuffled order
- novel symbols

Critical diagnostic:

> same facts + changed query should predictably move candidate preference.

## C3 — Working-state update

Track a variable/entity across updates and use the current state rather than the first or most frequent state.

Examples:

- value overwrite
- location transitions
- ownership transfer
- finite-state machine
- inventory update
- event sequence

## C4 — Relational composition

Combine multiple relations rather than retrieve one directly stated answer.

Difficulty:

- 1-hop control
- 2-hop
- 3-hop
- variable-length path
- multiple competing paths
- order-sensitive composition

## C5 — Counterfactual sensitivity

If a relevant premise changes, prediction should change. If an irrelevant premise changes, prediction should remain stable.

This should be evaluated using paired examples and score deltas, not only accuracy.

## C6 — Selective retrieval under interference

Retrieve useful information while resisting plausible but irrelevant facts.

Manipulate:

- distractor count
- semantic similarity
- recency
- frequency
- candidate prior strength

## C7 — Missing-information recognition

Distinguish:

- answer is present;
- answer is inferable;
- answer is absent.

This is foundational for later tool use and self-monitoring.

## C8 — Reliable realization

If an answer is strongly preferred internally, produce it without degeneration, substitution, or irrelevant continuation.

Measure conditional realization rate:

`P(free output correct | ranked answer correct)`

## C9 — Rule induction / transformation

Infer a local transformation from examples and apply it to unseen inputs.

Use symbolic, linguistic, numerical, and structural variants.

## C10 — Fast adaptation

After small targeted updates, measure how quickly a model acquires a new cognitive family and how much unrelated capability it forgets.

This is important because a self-improving system needs a substrate that can learn efficiently without catastrophic collateral damage.

---

# 4. Data architecture

Initial hypothesis — **not frozen**:

### A. Broad high-quality substrate: ~55–70%

- educational web/text
- science/technical
- math
- high-quality code
- explanatory prose
- structured reference material

Purpose:

language, world structure, broad concepts, syntax, semantics, domain substrate.

### B. Cognitive curriculum: ~20–35%

Generated and curated tasks targeting C1–C10.

This is not ordinary chat/instruction tuning. It should be integrated into autoregressive foundation training with enough structural diversity that the model learns the underlying relation.

Families include:

- key/value bindings
- query swaps
- fact swaps
- state updates
- relational graphs
- composition
- ordered multi-output
- sorting/grouping
- rule induction
- sequence transformations
- counterfactual worlds
- missing-information cases
- distractor-resistant lookup
- simple planning/state trajectories

### C. High-density structural reasoning: ~10–20%

- math derivations
- algorithm traces
- programs
- formal/structured reasoning
- scientific causal explanations
- tables/graphs converted to structured sequences

Exact mixture should be selected experimentally.

---

# 5. Curriculum design

Do not present maximum difficulty from token one.

A working curriculum could progress along axes such as:

### Phase 0 — representation stability

- exact copy
- one binding
- short context
- simple state identity

### Phase 1 — controlled selection

- 2–4 bindings
- query swaps
- distractors
- state overwrite
- simple transformations

### Phase 2 — composition

- 2-hop graphs
- ordered pair output
- combined state + relation
- counterfactual premise changes

### Phase 3 — interference/generalization

- 8–16 bindings
- longer contexts
- semantically similar distractors
- template/OOD splits
- unseen symbols
- multi-domain presentation

### Phase 4 — mixed cognitive load

Tasks require multiple primitives simultaneously:

- retain state + select + compose;
- retrieve + apply rule;
- distinguish absent vs inferable;
- multi-step relation with distractors.

Curriculum advancement should depend on capability metrics, not only token count.

---

# 6. Architecture search before freeze

Do not assume V4 is optimal.

The first architecture campaign should compare a small set of controlled variants at equal or near-equal parameter/token/compute budgets.

## 6.1 Depth vs width

Hypothesis:

> More depth at equal parameter count may improve sequential/compositional computation, while more width may improve capacity/optimization but not necessarily multi-step cognition.

Test at least three shapes around one small research scale.

Example search region only:

- deep/narrow
- V4-like middle
- wide/shallow

Keep tokenizer, dataset, optimizer, and token budget fixed.

Primary comparison should include cognition-per-FLOP, not just validation loss.

## 6.2 Vocabulary

Compare approximately:

- 16k
- 24k
- 32k

Measure:

- sequence inflation
- embedding parameter cost
- normal LM loss
- code/math tokenization
- exact copying
- unseen identifier handling
- binding
- morphology/compositionality

Do not decide vocab size by habit.

## 6.3 Context length

Initial candidates:

- 2k research baseline
- 4k likely V5 minimum target
- potentially 8k after stable 4k behavior

Long context is valuable only if the model can actually select relevant state within it.

Therefore context scaling should be accompanied by distractor/binding scaling tests.

## 6.4 Attention topology

Start from the cleanest interpretable baseline.

Compare only when justified:

- full causal attention
- hybrid local/full attention
- other sparse approaches

For a ~300M-class model and 4k context, full attention may be worth keeping as the scientific baseline even if a sparse design is eventually needed for efficiency.

## 6.5 GQA / head structure

Search Q/KV head choices with attention quality and memory cost in mind.

V4's 14Q/2KV configuration is not automatically inherited.

Possible region:

- head_dim 64
- 12–16 Q heads depending on width
- 2–4 KV heads

## 6.6 FFN ratio

Test modest variants rather than blindly maximizing FFN size.

Goal:

find compute allocation that best supports cognitive metrics at fixed total parameters/FLOPs.

## 6.7 Recurrence / explicit state

Do **not** add recurrent/state-space mechanisms to the first baseline simply because cognition is the goal.

First establish whether a strong conventional decoder with cognition-first data can learn the primitives.

Only test recurrence/state mechanisms if a specific failure persists and the experiment can discriminate the mechanism.

## 6.8 MoE

Do not use MoE in the initial cognition-first foundation.

Reason:

it adds routing, training, and interpretation complexity before the dense baseline has answered the scientific questions.

---

# 7. Provisional scale strategy

Do not jump directly to 3B.

### Stage A — tiny mechanism models

Approximate region: 30M–80M.

Purpose:

- debug data/curriculum;
- compare tokenizer options;
- test architecture direction;
- verify metrics move at all.

### Stage B — research models

Approximate region: 80M–180M.

Purpose:

- reproduce effects at meaningful capacity;
- compare depth/width;
- test curriculum ratios;
- measure generalization.

### Stage C — confirmation model

Approximate region: 250M–400M.

Purpose:

- test whether the winning recipe scales;
- establish the first serious V5 candidate.

Only after Stage C cleanly outperforms V4 on both substrate and cognitive metrics should a 1B–3B expansion be considered.

A provisional serious V5 region to investigate is roughly **280M–350M dense**, but this number is a hypothesis, not the starting command.

---

# 8. Provisional architecture region for the serious V5 candidate

This is a search target, not frozen configuration.

Possible region:

- dense decoder-only Transformer
- parameters: ~280M–350M
- vocab: 16k–24k if experiments support it
- width: around 1024
- layers: around 22–26
- Q heads: around 16
- KV heads: around 4
- head_dim: 64
- SwiGLU FFN: roughly 2.7k–3.1k
- context: 4096 initially
- RMSNorm
- RoPE
- QK norm
- tied embedding/output
- no MoE
- no experimental dormant modules
- clean causal attention baseline

This region deliberately allocates more depth than V4 and more context, but those choices must survive the controlled search.

---

# 9. Optimization contract

The exact optimizer recipe is not frozen yet, but the training system must satisfy these invariants.

## 9.1 Exact accounting

Every checkpoint records:

- global optimizer step
- pack/curriculum step
- exact cumulative tokens processed
- tokens in each data family
- effective batch tokens
- source commit
- dataset manifests/hashes
- tokenizer SHA/contract
- model parameter SHA
- optimizer moment hashes or verifiable state identity
- LR schedule state
- sampler/data cursor

There must be one canonical lifetime token counter from step zero.

## 9.2 Real-update invariant

At startup and periodically:

- optimizer owns exact live model Parameter objects;
- gradients are present and finite;
- optimizer state step increments;
- parameter SHA changes after a canary update;
- optimizer moments change;
- loss/logit probe changes consistently.

If any fails, training fails closed.

## 9.3 Immutable milestone checkpoints

Do not keep only a mutable latest checkpoint.

Preserve immutable milestones at a cadence chosen to make behavioral regression detectable.

Each milestone must be full-resume and externally durable.

## 9.4 Checkpoint promotion

A checkpoint can be labeled “best” only using a preregistered multi-objective rule.

Possible dimensions:

- held-out LM loss
- cognition aggregate
- worst-family cognition score
- degeneration
- retention
- training stability

Never automatically equate final with best.

---

# 10. Evaluation architecture

## 10.1 Frozen benchmark tiers

### Tier 0 — canaries

Fast tests every small interval.

Examples:

- exact copy
- 2/4 binding
- simple state update
- one 2-hop family
- generation degeneration

### Tier 1 — development cognition battery

Run frequently enough to guide research.

Can be consumed/tuned against.

### Tier 2 — sealed promotion battery

Created/frozen before the candidate model/config is chosen.

Never used for feature/data tuning.

### Tier 3 — fresh replication

New independently generated fixture after a successful promotion.

Used to test whether the claimed mechanism generalizes.

## 10.2 OOD matrix

Every cognitive family should support several OOD axes:

- unseen entities
- unseen values
- unseen templates
- longer cardinality
- longer hop count
- changed surface domain
- changed order
- changed distractor statistics

## 10.3 Internal-state diagnostics

For tasks with finite candidate sets, store:

- full candidate score vector
- margins
- entropy/spread
- RAW choice
- normalized choice
- free output
- query-swap response
- intervention outcome

This allows later cognitive-credit experiments without contaminating evaluation.

---

# 11. Data quality requirements

“High quality” should be measurable.

For natural data:

- deduplicate aggressively;
- identify source/domain;
- remove corrupted/token-garbage text;
- control low-quality boilerplate;
- document licenses/permissions as applicable;
- preserve validation contamination boundaries;
- measure domain balance.

For synthetic cognition data:

- generator code is versioned;
- latent structure is explicit;
- train/dev/test seeds are separated;
- vocab/template/topology splits are auditable;
- task difficulty metadata is stored;
- no evaluator answer fields are exposed to runtime policy features;
- examples are checked for accidental lexical shortcuts.

---

# 12. Training objectives

Baseline remains autoregressive next-token prediction because it is simple, scalable, and compatible with broad data.

But the research program should test whether controlled auxiliary/contrastive objectives improve cognitive structure.

Candidates for controlled experiments include:

- standard LM-only on contrast-rich data;
- explicit query-conditioned candidate-margin objective;
- paired contrastive likelihood objective;
- representation-level consistency/sensitivity objective;
- curriculum-weighted LM objective.

Do not combine several novel losses in the first experiment.

Question first:

> Can data structure alone make standard LM training acquire the desired cognitive primitive?

Then add objectives only when needed.

---

# 13. Internalization loop from EXP to Core

One long-term purpose of Connector/EXP is to discover computations that should migrate into the Core.

For a failure family:

1. detect recurring failure;
2. characterize internal signal;
3. identify an intervention that reliably repairs it;
4. prove mechanism on fresh cases;
5. convert the causal contrast into curriculum/training examples;
6. train a new Core candidate;
7. test whether raw Core performance increases;
8. test whether intervention frequency falls;
9. preserve the intervention as a probe until internalization is demonstrated.

Success criterion:

> A later Core performs natively what an earlier Core required an external intervention to accomplish.

Counterfactual normalization is the clearest existing candidate for this internalization program.

---

# 14. What should NOT happen

Do not:

- pick parameter count before defining the cognition benchmark;
- choose 32k vocab only because V4 used it;
- spend hundreds of TPU hours to answer a question a 50M model can answer;
- optimize only validation loss;
- treat free-generation failure as proof that representation is absent;
- let agents silently change architecture during execution;
- add multiple fancy modules at once;
- use future sealed tests to tune the model;
- hide negative results;
- call a runtime scaffold an internal learned capability;
- call a token count verified unless provenance proves it;
- automatically use the final checkpoint;
- use the Connector as a permanent patch for a repeatedly trainable Core capability.

---

# 15. Blueprint freeze process

The final V5 training path should be frozen in three passes.

## Pass A — design

Specify:

- cognition objectives
- benchmark
- data families
- tokenizer candidates
- architecture candidates
- optimizer candidates
- scale ladder
- token budgets
- checkpoint cadence
- promotion/falsification rules

## Pass B — attack

Use an independent agent/reviewer to search for:

- hidden assumptions
- compute infeasibility
- benchmark leakage
- shortcuts
- conflicting constraints
- bad token accounting
- missing baselines
- irreproducible steps
- unsafe checkpoint paths

The reviewer should not redesign everything; it should expose weaknesses.

## Pass C — freeze v1.0

Resolve the critique and lock:

- exact config
- exact tokenizer
- exact dataset manifests/mixture
- exact curriculum
- exact optimizer/LR/batch
- exact token budget
- exact evaluation fixtures
- exact promotion rule
- exact checkpoint schedule
- exact provenance contract

After freeze, execution agents implement the specification and deviations require an explicit version bump.

---

# 16. First experimental campaign before V5 freeze

A high-value first campaign should answer a small number of questions rather than train “V5” immediately.

Suggested order:

1. Build the cognition benchmark/generators and prove no obvious shortcut.
2. Train tiny models to test whether cognition-rich data moves binding/state/composition curves.
3. Test 16k/24k/32k tokenizer effects at small scale.
4. Test depth vs width at matched parameter/compute scale.
5. Test cognitive curriculum mixture ratios.
6. Test standard LM objective vs one query-conditioned contrastive objective.
7. Reproduce winning recipe at ~150M.
8. Only then freeze the ~300M-class V5 candidate.

---

# 17. Success criteria for the first serious V5 Core

A serious V5 candidate should not be promoted merely for modest benchmark gains.

It should demonstrate, on sealed and then fresh tests:

- substantially better raw contextual retrieval than V4 PGE;
- binding materially above chance across unseen symbols/templates;
- query swaps reliably move internal preference;
- state overwrite works beyond training templates;
- meaningful 2-hop composition and a positive scaling curve toward 3-hop;
- lower conditional realization failure;
- cognitive gains persist without catastrophic loss regression;
- cognition gains appear before specialized post-training;
- interventions such as normalization are needed less often, or their useful signal is more strongly internalized;
- checkpoints/provenance are fully reproducible.

---

# 18. Long-term direction

The desired trajectory is:

**V4:** generic Core + substantial external repair

→ **V5:** cognition-aware foundation + external diagnosis

→ **later Core:** repeated verified external repairs become internalized training structure

→ **self-improving system:** causal self-model predicts useful intervention, collects verified evidence, and proposes the smallest justified change to future training/runtime behavior.

The branch should keep returning to this question:

> **Are we making the Core genuinely better at the computation, or merely building a smarter wrapper around the same weak computation?**
