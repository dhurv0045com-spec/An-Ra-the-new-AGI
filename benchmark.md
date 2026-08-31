# ESOES Cognitive Foundation Benchmark

**Status:** Ground Blueprint benchmark specification v0.1
**Branch:** `esoes`
**Purpose:** Decide whether a Core design, training recipe, or checkpoint is genuinely more useful for future cognition—not merely a better next-token predictor.
**Main-training authority:** **NONE.** This specification defines evidence requirements; it does not authorize the V5-A run.

---

## 1. North-star question

The benchmark exists to answer one question:

> **Did this design make the neural Core measurably better at learning, manipulating, and transferring reusable cognitive operations under controlled out-of-distribution conditions?**

This is deliberately different from asking whether the model is “AGI,” whether it has lower validation loss, or whether it scores higher on one reasoning benchmark.

A cognition-supportive Core should increasingly support reusable computations such as binding arbitrary entities to values, selecting the binding relevant to a changing query, tracking state independently of textual order, composing relations, inferring a rule from demonstrations, responding to causal changes while ignoring irrelevant changes, recognizing missing information, and faithfully realizing an internally preferred answer.

The canonical decomposition is:

```text
REPRESENT → ADDRESS → TRANSFORM → CHOOSE → REALIZE
```

The benchmark must measure these stages separately whenever possible.

---

## 2. What this benchmark is not

It is **not**:

- a single “AGI score”;
- a replacement for held-out language-model loss;
- an instruction-following benchmark alone;
- a collection of synthetic templates the model can memorize;
- a reward for Connector-assisted success;
- a test where the final checkpoint automatically wins;
- evidence that a specific parameter count, head count, or architecture is cognitive merely because it is modern or larger.

A design choice earns the label **cognition-helpful** only through controlled evidence.

---

## 3. Scientific laws of the benchmark

### 3.1 Causal contrast, not correlation

Important cases must have paired interventions where exactly one intended variable changes.

Examples:

```text
same facts + different query
same query + changed relevant fact
same query + changed irrelevant fact
same semantic state + different serialization order
same problem + changed counterfactual premise
```

The generator must mechanically assert the intervention contract after serialization. “The template was intended to change only one thing” is not sufficient.

### 3.2 Sensitivity and invariance are both required

A useful cognitive representation should:

- **change** when a causally relevant variable changes;
- **stay stable** when an irrelevant variable changes.

A benchmark that checks only correctness but not these paired behaviors can reward shortcuts.

### 3.3 Structure must generalize beyond surface form

Success on training-like templates is weak evidence. Evaluation must hold out combinations of:

- symbols/entities;
- relation names;
- surface templates;
- graph topology;
- rule structure;
- cardinality;
- hop count;
- distractor count;
- context position;
- answer format;
- domain/style;
- generator family where feasible.

### 3.4 Synthetic success requires transfer

Synthetic tasks are useful because their causal truth can be executable. They are not sufficient by themselves.

Important primitive gains must be checked on structurally analogous natural or semi-natural tasks whose source, wording, entities, and surface form were not used to build the synthetic generator.

### 3.5 Raw Core and assisted system are different measurements

Always report separately:

```text
RAW CORE
CONTROLLED / CONSTRAINED REALIZATION
CONNECTOR-ASSISTED
ORACLE-ASSISTED
```

A Connector repair is valuable diagnostic evidence, but it cannot satisfy a native-Core promotion gate.

### 3.6 No single aggregate score

The primary output is a **Cognitive Foundation Profile**, not one scalar.

Averages can hide catastrophic primitive failures. Worst-family behavior, difficulty curves, OOD transfer, and causal-pair consistency matter more than a pretty leaderboard number.

---

# 4. Core benchmark families

## CF-1 — Exact representation and identity preservation

**Question:** Can the Core preserve and use information that exists only in the current context?

Example:

```text
Marker DV-vek-19 reads exactly [QX71-A].

Copy only the characters inside the brackets.
```

Measure:

- gold answer log-probability;
- exact free generation;
- byte/token exactness;
- performance on novel identifiers;
- position robustness;
- degradation with identifier length and context length.

Important rule: one-candidate copy tasks are **realization/representation probes**, not selection evidence.

---

## CF-2 — Query-conditioned variable binding

**Question:** Can the Core bind arbitrary entities to arbitrary values and let the current query select the relevant binding?

Base:

```text
Ranu = VX17
Teka = PQ82
Lomu = HK31
Siva = ND44

Question: What belongs to Lomu?
Answer: HK31
```

### Required causal variants

**Query swap**

```text
Facts unchanged.
Question: What belongs to Teka?
Answer must move to PQ82.
```

**Relevant-fact swap**

```text
Question unchanged.
Lomu = AZ90 replaces Lomu = HK31.
Answer must move to AZ90.
```

**Irrelevant-fact swap**

```text
Question unchanged.
Siva's value changes.
Answer must remain HK31.
```

**Order permutation**

Facts are reordered without changing semantics. Answer must remain stable.

### Difficulty ladder

```text
2 bindings
4 bindings
8 bindings
16 bindings
32 bindings (later / scale permitting)
```

Measure not just accuracy but the slope of degradation.

### Key metrics

- candidate rank-1 accuracy;
- gold-vs-best-alternative margin;
- query-conditioning lift;
- query-swap direction correctness;
- irrelevant-change invariance;
- permutation invariance;
- free realization conditional on correct selection.

This is one of the highest-priority V5 primitives.

---

## CF-3 — State tracking and semantic precedence

**Question:** Can the Core determine current or historical state from semantic ordering rather than textual position?

A valid state benchmark must **not** serialize events in the same order as their logical time.

Example:

```text
At time 14, X became C.
At time 3, X became A.
At time 20, X became D.
At time 8, X became B.

Question: What is the newest state of X?
Answer: D
```

### Required variants

- events shown out of chronological order;
- query for newest state;
- query for state at an intermediate time;
- multiple interleaved variables;
- rollback / superseding update semantics;
- irrelevant later textual statements;
- changed relevant update;
- changed irrelevant update;
- same state history under different serialization order.

The query cutoff must lie **between** events or **after** the history; it must
never equal an event timestamp. Each case must expose enough competing values
for the queried entity that entity matching alone is no better than the
casewise candidate null. Rollbacks point to an earlier semantic state and
same-time conflicts require explicit priority comparison. These are generator
invariants, not optional difficulty settings.

### Mandatory shortcut controls

The following heuristics must **not** solve the family:

- last textual fact;
- nearest candidate mention;
- first/last candidate;
- lexical overlap;
- bag of words / entity-only retrieval;
- exact timestamp lookup.

Certification pools eight independent generator seeds and fails closed if any
named heuristic exceeds its registered null by more than 10 percentage points.
`latest_fact` and `nearest_position` use the analytic random-serialization null;
all other listed controls use casewise uniform-candidate chance. If any shortcut
fails, the state family is invalid regardless of the overall certificate.

---

## CF-4 — Selective retrieval under interference

**Question:** Can the Core retrieve the relevant information while resisting distractors that are lexically or positionally attractive?

Manipulate independently:

- number of distractors;
- lexical similarity of distractors;
- answer position;
- query position;
- repeated entities;
- decoy values;
- local versus distant relevant facts.

Difficulty curve:

```text
0 distractors
2
4
8
16
32
```

A useful primitive should degrade smoothly rather than collapse as soon as a distractor pattern changes.

---

## CF-5 — Relational composition

**Question:** Can the Core compose multiple relations rather than retrieve a directly stated endpoint?

Example:

```text
A routes to B.
B maps to C.
C feeds D.

Question: Starting at A and following routes → maps → feeds, where do we arrive?
Answer: D
```

Every composition case needs a **matched direct-retrieval control** with comparable entities, answer candidates, formatting, and context length.

### Difficulty ladder

```text
1 hop
2 hops
3 hops
4 hops
5+ hops later if useful
```

### Promotion evidence

A composition gain is meaningful only if:

- matched retrieval remains controlled;
- simple lexical/position heuristics do not explain the gain;
- performance transfers to new graph topologies and relation vocabularies;
- the model shows a nontrivial 2→3-hop curve rather than a one-template discontinuity;
- at least one natural analogue improves.

---

## CF-6 — Rule induction and novel application

**Question:** Can the Core infer an operation from demonstrations and apply it to unseen inputs?

Do **not** use one permanent latent rule such as “reverse the pair.”

The generator must sample from multiple rule families, for example:

```text
reverse pair
select left
select right
rotate triple
interleave two sequences
apply learned substitution map
conditional branch
compose two primitive transformations
```

Some rule structures—not only symbols—must be held out from development/training and reserved for fresh evaluation.

Measure:

- in-family novel-symbol generalization;
- held-out-rule-structure generalization;
- demonstration-count curve;
- candidate selection;
- free realization;
- sensitivity to changed demonstrations;
- invariance to irrelevant demonstrations.

A bag-of-words or answer-position heuristic performing strongly on this family is a benchmark failure signal.

---

## CF-7 — Counterfactual and causal sensitivity

**Question:** Does the Core follow the supplied local world rather than its prior or superficial continuation pattern?

Example:

```text
Every Zor is blue.
Kira is a Zor.
Is Kira blue?
→ YES
```

Paired intervention:

```text
No Zor is blue.
Kira is a Zor.
Is Kira blue?
→ NO
```

Then change an irrelevant premise and require the answer to stay stable.

Metrics:

- answer-switch correctness;
- candidate probability movement;
- relevant-premise sensitivity;
- irrelevant-premise invariance;
- calibration under contradictory or insufficient premises;
- transfer across symbolic and natural domains.

---

## CF-8 — Missing information and epistemic restraint

**Question:** Can the Core distinguish “inferable from this context” from “not specified here”?

Construct cases where:

- one queried fact is genuinely absent;
- a plausible answer exists among distractors;
- prior-world knowledge would tempt a guess;
- the answer becomes inferable after exactly one relevant fact is added.

Paired test:

```text
missing context → <MISSING>
add necessary fact → concrete answer
```

Measure both detection and probability movement.

The model should not be rewarded merely for learning that a particular template usually expects `<MISSING>`.

---

## CF-9 — Faithful realization

**Question:** If the correct content is internally preferred, can the Core reliably emit it?

Separate:

```text
SELECTION: is the gold candidate highest-scoring?
REALIZATION: does free generation emit that content?
CONTROLLED REALIZATION: can constrained decoding express it exactly?
```

Report the **selection–realization gap**:

```text
selection accuracy − free-generation accuracy
```

Also report realization conditional on correct internal selection.

A future Core should reduce this gap without requiring permanent external decoding repair.

---

## CF-10 — Long-context interference and usable memory

**Question:** Does increasing context preserve useful addressing and composition, or merely increase the nominal window?

This family is secondary until 4k-context V5 is validated, but the benchmark contract should support:

- relevant fact at beginning/middle/end;
- distractor density scaling;
- repeated entities across distant spans;
- state updates separated by long irrelevant sections;
- composition across distant facts.

Context length is useful only if capability remains usable throughout the window.

---

# 5. Measurement stack

Each compatible case should produce a structured record rather than one correctness bit.

## 5.1 Representation

Possible measures:

- gold answer NLL;
- candidate score vector;
- rank of gold;
- score stability under irrelevant interventions.

Representation is evidence that the needed information is present in the model's output geometry. It is not by itself proof of correct selection.

## 5.2 Addressing

Measure whether the query or requested state actually controls which representation is favored.

Primary measures:

- query-swap candidate movement;
- query-conditioning lift;
- relevant-fact swap movement;
- irrelevant-fact invariance.

## 5.3 Transformation

Measure operations that cannot be solved by matched direct retrieval:

- relation composition;
- rule application;
- state precedence;
- counterfactual inference.

Always compare against matched non-transformational controls.

## 5.4 Choice

Report:

- rank-1 accuracy;
- gold margin;
- calibration where meaningful;
- pass→fail and fail→pass transitions under interventions;
- worst-family accuracy.

## 5.5 Realization

Report separately:

- free exact / normalized exact;
- constrained exact;
- conditional realization given correct selection;
- malformed-output rate;
- degeneration/repetition rate where relevant.

---

# 6. Difficulty curves are first-class evidence

A foundation benchmark should reveal a capacity frontier, not just a pass/fail label.

Examples:

```text
Binding:     2 → 4 → 8 → 16 entities
Composition: 1 → 2 → 3 → 4 hops
Distractors: 0 → 2 → 4 → 8 → 16 → 32
Context:     512 → 1k → 2k → 4k
Rule demos:  1 → 2 → 4 → 8 examples
State:       1 → 2 → 4 variables, increasing updates
```

For each curve report:

- absolute performance;
- confidence interval;
- slope / degradation;
- point of sharp failure;
- comparison to matched baseline/control.

A smooth degradation with increasing difficulty is stronger evidence of a learned reusable mechanism than a brittle template-specific cliff.

---

# 7. Split and custody protocol

## Tier 0 — Generator/property tests

Purpose: prove causal contracts and serialization logic.

No scientific capability claim.

## Tier 1 — Development

Visible to researchers. Used to debug generators, metrics, adapters, and obvious heuristics.

Once a development outcome materially influences a design choice, it is considered consumed for unbiased confirmation.

## Tier 2 — Sealed

Generated under independent/external seed custody. The repository contains only a commitment/hash before evaluation.

Used for preregistered promotion tests.

The seed, cases, answers, and hidden truth must not enter training, architecture selection, or prompt tuning.

## Tier 3 — Fresh replication

New generator draw or held-out generator family produced only after the first sealed result is known.

Used to test whether the claimed mechanism survives another prospective sample.

## Natural-transfer tier

Source/domain-disjoint natural or semi-natural tasks. Must not merely paraphrase the synthetic development template.

---

# 8. Shortcut and adversarial baselines

Every family should be attacked by simple systems before it is trusted.

Required classes include where applicable:

- random candidate;
- first/last candidate;
- first/last fact;
- nearest mention;
- lexical overlap;
- bag-of-words retrieval;
- latest textual state;
- most frequent value;
- answer-length/format heuristic;
- matched direct retrieval;
- deliberately broken symbolic solver;
- full truth oracle.

A benchmark family fails certification if an unintended cheap heuristic reaches performance that would make a model result ambiguous.

The goal is not to make every task hard. The goal is to make success causally interpretable.

---

# 9. Statistical protocol

Before a serious comparison, preregister:

- primary unit of analysis;
- primary metric;
- baseline/control;
- sample size or power rationale;
- confidence interval method;
- paired-test method where paired interventions exist;
- minimum practically meaningful effect;
- multiple-comparison handling if many families are promoted simultaneously;
- abort/failure criteria.

Do not select a statistical method after seeing which one produces significance.

For paired causal fixtures, preserve group/pair structure in bootstrap or exact tests. Do not pseudo-replicate multiple candidates from one fact group as independent observations.

Report uncertainty even when the result is negative.

## 9.1 Executable E0 calibration contract

The broad standard above is instantiated by `e0_cognition/`. Its canonical
development receipt is `artifacts/e0/development_certificate.json`. The owner is
an independent evaluator—not Core, Trainer, or Connector—and the present status
is **development infrastructure PASS / sealed promotion NOT FROZEN**.

Before any model comparison, run the same mathematical pre-mortem in the design,
generator, and signed receipt:

1. Write the latent causal graph and the exact variable that should change the
   answer.
2. Enumerate position, frequency, candidate-order, lexical, fixed-rule,
   template, tokenization, and answer-format adversaries.
3. Compute the casewise uniform-candidate null (p_{0,i}=|C_i|^{-1}) and the
   suite-weighted null \(\bar p_0=n^{-1}\sum_i p_{0,i}\).
4. For a heuristic mechanically coupled to serialization—especially
   `latest_fact` and `nearest_position`—replace the uniform null with its exact
   permutation-calibrated accuracy over all fact orders.
5. Pool independent generator seeds and measure every adversary before reading
   model results.
6. Freeze generator, fixture, protocol, tokenizer, split, model, and evaluator
   identities before the decision boundary.
7. Attempt falsification in raw-Core, constrained, assisted, OOD, fresh, and
   natural-analogue conditions.

The E0 shortcut gate is:

\[
\max_h\left(\operatorname{Acc}(h)-\bar p_{0,h}\right) \le 0.10,
\]

where \(\bar p_{0,h}\) is the declared uniform or exact-permutation null for
heuristic \(h\). Development certification pools eight independent seeds; a
single convenient seed cannot establish shortcut resistance.

### E0 information firewall

`CausalCase.model_view()` contains only model-facing context, query, and prompt.
Gold answer, candidates, graph, relevant/distractor indices, operation trace,
difficulty labels, axes, seed, and provenance remain evaluator-only. Every
intervention pair keeps its candidate set fixed.

- **Sensitivity pairs:** query swap, relevant-fact swap, and state swap. Both
  members must be correct and the answer/prediction must change.
- **Invariance pairs:** irrelevant-fact swap and serialization permutation. Both
  members must be correct and the answer/prediction must remain fixed.

Never average sensitivity and invariance into one robustness score.

### State and rule gates

State cases build semantic event graphs before independently shuffling textual
serialization. At least two variables are interleaved. Queries cover latest,
intermediate-time, rollback, and same-time precedence; timestamps and priority,
never list position, determine truth. Certification fails if `latest_fact` or
`nearest_position` exceeds its exact permutation null by more than 10 percentage
points on pooled state cases.

Rule-induction cases provide two to four demonstrations and query an unseen
input. Development contains eight latent operand-index structures; sealed and
fresh use disjoint structure sets, not merely new symbols. Fixed reverse,
identity, repeated-left, repeated-right, and bag-of-words baselines must remain
within their declared chance + 10-point gates. The independent solver must infer
the structure from surface demonstrations and may not read hidden metadata.

One-candidate copy cases are excluded from candidate-selection denominators.
They measure identity representation and realization only.

### Required metric separation

For candidate score \(s(c\mid F,q)\), report gold rank and margin

\[
m=s(c^*\mid F,q)-\max_{c\ne c^*}s(c\mid F,q).
\]

For a counterfactual query \(q'\) over the same facts, report addressing lift

\[
\Delta_q=s(c^*\mid F,q)-s(c^*\mid F,q').
\]

For raw output \(r\), assisted output \(a\), and truth \(y\), report separately

\[
D=\mathbb{1}[a=y\land r\ne y], \qquad
H=\mathbb{1}[r=y\land a\ne y],
\]

where \(D\) is intervention dependence and \(H\) is assistance harm. Raw
selection, raw realization, constrained realization, assisted realization,
conditional realization, \(D\), and \(H\) require distinct denominators.

### E0 development certification checklist

The executable certificate must prove all of the following together:

1. deterministic regeneration and unique canonical cases;
2. independent surface-solver agreement;
3. hidden-truth/model-view isolation;
4. fixed-candidate sensitivity and invariance pairs;
5. semantic-time state coverage with shuffled serialization;
6. pooled positional-heuristic failure under calibrated nulls;
7. multiple rule structures, structural split holdout, and pooled fixed-rule /
   bag-of-words failure;
8. matched direct-retrieval controls that do not solve composition;
9. copy exclusion from selection metrics;
10. explicit difficulty axes and naturalistic analogues;
11. disjoint training/evaluation template namespaces;
12. frozen statistical identity and fail-closed sealed custody.

The current 368-case / 112-pair receipt and a 20-seed property sweep pass these
development invariants. That certifies generator/evaluator infrastructure—not
benchmark difficulty, model cognition, external natural transfer, or sealed
promotion. A real externally custodied T2 fixture and source-disjoint natural set
remain mandatory.

---

# 10. Cognitive Foundation Profile

Every promoted checkpoint should produce a profile approximately like:

```text
SUBSTRATE
  held-out language loss
  code/math loss or capability
  science/technical loss or capability

REPRESENT
  exact contextual representation
  nonce identity preservation

ADDRESS
  binding rank-1
  query-swap sensitivity
  irrelevant-change invariance

TRANSFORM
  semantic state tracking
  2-hop / 3-hop composition
  rule induction
  counterfactual sensitivity

CHOOSE
  raw selection accuracy
  gold margin
  missing-information calibration

REALIZE
  free accuracy
  constrained accuracy
  conditional realization
  selection-realization gap

TRANSFER
  unseen symbols
  unseen templates
  unseen topology/rules
  fresh-generator replication
  natural-domain transfer

ROBUSTNESS
  difficulty curves
  distractor curve
  context-position curve
  worst-family lower confidence bound

INTERVENTION DEPENDENCE
  no-change/raw
  normalization or diagnostic repair
  constrained-realization repair
  Connector-assisted oracle gap
```

No single scalar is authoritative.

---

# 11. Intervention-dependence as a longitudinal metric

EXP showed that external interventions can sometimes reveal or repair latent capability. For V5, that should become a diagnostic of **what the Core has not yet internalized**.

Example progression:

```text
Version        RAW binding   assisted binding   repair gap
V5.0              55%             80%              25 pp
V5.1              72%             82%              10 pp
V5.2              84%             86%               2 pp
```

If raw capability rises while the best external repair advantage shrinks on fresh data, that is evidence that the Core is internalizing a previously external computation.

The same principle applies to realization:

```text
correctly selected but not freely realized
30% → 13% → 4%
```

Do not optimize the Core merely to imitate a specific runtime intervention. Use the intervention gap as a mechanistic diagnostic.

---

# 12. When is a design change “cognition-helpful”?

A tokenizer, architecture change, data mixture, objective, curriculum, optimizer choice, or scale increase may be called **cognition-helpful** only if all applicable gates hold:

### Gate A — Controlled improvement

It improves a preregistered cognitive primitive against the matched control under approximately matched data, parameters/compute, and evaluation.

### Gate B — Causal validity

The improvement survives relevant-variable interventions and invariance controls. It is not explained by answer position, lexical overlap, recency, template identity, or another tested shortcut.

### Gate C — OOD survival

The effect survives at least one meaningful held-out axis beyond symbols alone—for example new templates, topology, rule structure, cardinality, or generator family.

### Gate D — Prospective confirmation

The direction replicates on sealed/fresh data that did not influence the decision.

### Gate E — Transfer

For a major foundation claim, there is evidence on more than one surface form and at least one natural/semi-natural analogue, unless the capability is intrinsically format-specific and explicitly labeled so.

### Gate F — Substrate retention

The change does not buy a narrow synthetic specialist by causing unacceptable regression in held-out language, code, math, science, or other frozen substrate metrics.

### Gate G — Native capability

The promoted result is visible in the raw Core. Connector or oracle assistance may be reported but cannot satisfy this gate.

A change that only lowers LM loss is **LM-helpful**, not yet cognition-helpful.

A change that only improves a development template is **benchmark-helpful**, not yet cognition-helpful.

A change that improves controlled, fresh, transferable cognitive operations is **cognition-helpful**.

---

# 13. Evidence maturity levels

Use these labels instead of prematurely saying “capability acquired.”

## Level 0 — Signal

Some internal score movement or development-set improvement exists.

No capability claim.

## Level 1 — Controlled primitive

The model beats relevant shortcut baselines on the intended primitive and passes the causal intervention contract on development data.

## Level 2 — OOD primitive

The effect survives held-out entities/templates/structures and a sealed evaluation.

## Level 3 — Transferable primitive

The effect replicates prospectively and transfers to different surface forms or natural analogues.

## Level 4 — Foundation-grade primitive

Evidence includes:

- controlled causal behavior;
- sealed + fresh replication;
- meaningful structural OOD;
- natural-domain or cross-format transfer;
- stable difficulty scaling;
- substrate retention;
- raw-Core success;
- reduced dependence on external repair relative to earlier Core versions where such a repair exists.

Level 4 is the standard for saying a V5 design has created a genuinely useful cognitive foundation primitive.

---

# 14. Architecture/data experiment decision rule

Suppose two candidates A and B have similar compute and B has slightly lower LM loss. That alone does not decide the architecture.

Prefer B as a cognition-supportive design only when its profile demonstrates a reproducible improvement such as:

```text
                          A       B
LM loss                 2.31    2.29
8-way binding            62%     78%
query-swap sensitivity   58%     82%
irrelevant invariance    71%     90%
semantic state tracking  53%     76%
2-hop composition        42%     64%
3-hop composition        21%     41%
OOD rule induction       29%     50%
free realization         67%     81%
natural transfer         39%     57%
```

The numbers above are illustrative, not frozen thresholds.

The actual promotion thresholds must be preregistered per experiment and powered appropriately.

---

# 15. Blueprint execution gate

The V5 blueprint becomes ready for the serious main run when the remaining uncertainty is intentionally delegated to the main run rather than caused by unresolved design negligence.

Required evidence:

| Question | Required evidence before main run |
|---|---|
| Can we measure cognition honestly? | E0 causal/shortcut certification + sealed/fresh contract |
| Which tokenizer? | E1 winner with artifact and corpus hashes |
| Which depth/width/attention shape? | E2 matched-compute result + replication |
| Does cognition data help OOD rather than only templates? | E3 fresh + natural transfer |
| Does query-swap auxiliary help? | E3 controlled ablation; otherwise remove it |
| Which curriculum / LR / batch / schedule? | E4 bounded stability and capability evidence |
| Does the recipe survive scale increase? | M102/E5 replication |
| Is the target scale justified? | scaling curve / compute-data argument, not parameter aesthetics |
| Are parameter/update/token receipts trustworthy? | real-update + exact accounting canaries |
| Can training resume exactly? | uninterrupted vs restored equivalence test |
| Can outputs survive disposable compute? | remote upload + clean re-download + restore canary |
| Which checkpoint wins? | behavioral promotion contract; final step has no privilege |
| What causes abort? | preregistered integrity/substrate/cognition thresholds |

When these are satisfied, the blueprint does **not** need to be perfect. It needs to be decision-complete enough that the 250M-class run itself answers a meaningful scientific question.

---

# 16. Final criterion

The V5 program should never claim success because:

```text
250M parameters
+ 5B tokens
+ lower validation loss
= cognition
```

The intended claim is much stricter:

> **A V5 design is valuable for future AGI only when it produces reusable neural operations whose behavior is causally controlled, structurally out-of-distribution, prospectively replicated, transferable across surface forms/domains, robust under increasing difficulty, and increasingly native to the Core rather than dependent on external repair.**

That is the benchmark standard ESOES should use when deciding what architecture, data, objective, checkpoint, or future scale deserves promotion.
