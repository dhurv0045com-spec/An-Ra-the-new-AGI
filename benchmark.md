# ESOES E0 Benchmark Contract

Status: **v0.4 executable development contract; sealed evaluation not yet frozen**
Owner: independent evaluator, not Core, Trainer, or Connector
Canonical code: `e0_cognition/`
Canonical receipt: `artifacts/e0/development_certificate.json`

This file is the operational definition of what an An-Ra cognition claim must
measure. It is deliberately stricter than a task list: every result must identify
the causal variable, the shortcut adversaries, the metric denominator, the split
identity, and the intervention condition.

## 1. Scientific target

The benchmark asks whether a raw Core can represent information, address it with
the current query, transform it, select the correct answer, and realize that
answer. It does not call tool use, retrieval, long-horizon planning, or a
Connector-assisted repair “native cognition.” Those are separate runtime
conditions.

The measurement order is:

```text
representation → addressing → transformation → selection → realization
                         ↘ intervention dependence (assistance, separately)
```

Copy controls are realization/representation controls only. They never enter a
candidate-selection denominator. A single aggregate exact-match number is not a
promotion metric.

## 2. Mathematical pre-mortem loop

Before any model comparison, repeat this loop on paper, in code, and in the
receipt:

1. **Causal graph:** write the latent dependency that should change the answer.
2. **Adversary set:** enumerate position, frequency, candidate order, lexical,
   fixed-rule, template, tokenization, and answer-format shortcuts.
3. **Null calculation:** compute the per-case uniform-candidate chance
   \(p_0 = |C|^{-1}\), then the suite-weighted \(\bar p_0 = n^{-1}\sum_i p_{0,i}\).
4. **Permutation simulation:** generate independent seeds and measure every
   adversary before looking at model results.
5. **Power check:** for a binary gate, use paired exact sign tests; for a score
   delta, use the preregistered 10,000-resample paired bootstrap. Increase power
   only if the decision is worth the compute.
6. **Freeze:** hash generator, fixtures, protocol, tokenizer, and split manifests.
7. **Falsify:** run raw-Core, constrained, assisted, fresh, and natural analogue
   conditions. A gain that disappears under any required shift is not promoted.

The primary shortcut rule is:

\[
\max_h(\mathrm{Acc}(h)-\bar p_0) \le 0.10,
\]

where \(h\) ranges over every named heuristic in the family. The development
certificate pools eight independent seeds before applying this bound; a single
seed is never sufficient evidence that a shortcut is absent.

## 3. Case schema and information firewall

`CausalCase` contains model-facing `context`, `query`, and `prompt`, plus
evaluator-only answer, candidates, graph, relevant/distractor indices, operation
trace, difficulty, axes, seed, and provenance. `model_view()` must not contain
answer, candidates, hidden truth, or surface-axis metadata.

Every pair keeps its candidate set fixed. Pair kinds are classified as:

- **Sensitivity:** query swap, relevant-fact swap, state swap. The answer should
  change and the prediction should follow it.
- **Invariance:** irrelevant-fact swap, order permutation. The answer should
  stay fixed and the prediction should remain stable.

Pair success requires both members to be correct. Report sensitivity and
invariance separately; never average them into one “robustness” score.

## 4. Required task families

The development generator covers exact copy, nonce retrieval, entity/value
binding, mutable state, one/two/three-hop composition, matched direct retrieval,
missing information, counterfactual premises, rule induction, and naturalistic
binding/state/composition analogues. Each family reports representation,
selection, and realization where applicable.

### 4.1 State tracking: semantic time is not position

State cases must satisfy all of the following:

- serialization order is independently shuffled after the semantic event graph
  is built;
- at least two variables are interleaved, with distractor updates for both;
- queries include latest, intermediate-time, rollback, and same-time precedence;
- timestamps/priority define semantics, never list position;
- the state swap changes one causal event while preserving query and candidates;
- rollback references an earlier semantic time without copying the answer into the
  query;
- development, sealed, and fresh splits use disjoint symbols, templates, domains,
  and rule structures.

The baseline battery includes `latest_fact`, `nearest_position`, and a deliberately
broken state tracker. Certification fails if either positional baseline exceeds
chance plus 10 percentage points on the pooled state families.

### 4.2 Rule induction: infer structure, do not memorize reverse pairs

Each case provides two to four demonstrations and asks for an unseen pair. The
latent rule is an operand-index sequence, such as `(1,0)`, `(0,1,0)`, or
`(0,0,1)`. The development split has eight structures; sealed and fresh have
different eight-structure sets. Thus holding out symbols alone is insufficient:
the transformation itself is OOD.

The battery includes fixed reverse, fixed identity, fixed repeated-left, fixed
repeated-right, and bag-of-words baselines. All must remain within chance + 10pp
on pooled rule cases. The independent surface solver infers the structure from
demonstrations and is not allowed to read hidden metadata.

### 4.3 Copy and realization separation

Exact-copy cases intentionally have one candidate. They measure identity
representation and free realization. Candidate-selection accuracy excludes them;
conditional realization is reported only for cases where a multi-candidate raw
selection was correct. Constrained decoding, normalization, or Connector repair
is an assisted column, never a raw-Core result.

## 5. Metrics

For candidate scores \(s(c\mid F,q)\), selection rank is the rank of the gold
candidate and margin is:

\[
m = s(c^*\mid F,q)-\max_{c\ne c^*}s(c\mid F,q).
\]

Query addressing uses a fixed context and a counterfactual query:

\[
\Delta_q = s(c^*\mid F,q)-s(c^*\mid F,q').
\]

For a pair prediction \(\hat y_b,\hat y_c\):

\[
S = \mathbb{1}[\hat y_b=y_b,\hat y_c=y_c,\hat y_b\ne\hat y_c],
\]
\[
I = \mathbb{1}[\hat y_b=y_b,\hat y_c=y_c,\hat y_b=\hat y_c].
\]

The evaluator reports the mean of \(S\) over sensitivity pairs and \(I\) over
invariance pairs, plus both-correct rates. A prediction change without two
correct answers is not a successful sensitivity result.

For raw output \(r\) and assisted output \(a\):

\[
D = \mathbb{1}[a=y \land r\ne y] \quad\text{(intervention dependence)},
\]
\[
H = \mathbb{1}[r=y \land a\ne y] \quad\text{(assistance harm)}.
\]

Report raw selection, raw realization, constrained realization, assisted
realization, \(D\), and \(H\) with their own denominators. The executable
`EvaluationRun` / `ReplicationBundle` contracts enforce these separations.

## 6. Difficulty curves and OOD axes

Every result is stratified by cardinality, hop count, distractor count, event
count, variable count, demonstration count, context length, relevant position,
answer format, and state-query type. Curves are reported rather than collapsed:

```text
accuracy(axis=value), margin(axis=value), raw realization(axis=value)
```

Required shifts include:

- new entity alphabets and nonce identifiers;
- held-out templates, prose styles, domains, and graph topologies;
- randomized relevant position and distractor density;
- held-out rule structures, not only held-out symbols;
- intermediate, rollback, precedence, and answer-absent queries;
- naturalistic analogues and genuinely source-disjoint natural evaluation;
- fresh generator implementations and a sealed fixture.

The generator reports `surface_axis_histograms()` and
`difficulty_axis_histograms()` in the development receipt. A model may only be
called robust if its worst-family fresh-OOD result survives these strata.

## 7. Controls and certification gates

The development certificate must prove:

1. deterministic regeneration and unique canonical cases;
2. independent surface-solver agreement;
3. hidden-truth/model-view isolation;
4. sensitivity and invariance pair presence and fixed candidates;
5. state semantic-query coverage and shuffled serialization;
6. pooled positional-heuristic failure on state families;
7. multiple rule structures, cross-split structural holdout, and pooled fixed-rule
   / bag-of-words failure;
8. direct-retrieval controls do not solve multi-hop composition;
9. copy cases are excluded from selection metrics;
10. difficulty axes and naturalistic analogues are present;
11. training and evaluation template namespaces are disjoint;
12. statistical protocol identity is fixed and sealed custody fails closed.

The current executable receipt is a **development infrastructure PASS**, not a
model-quality result. It is intentionally not a sealed exit because the real T2
fixture and source-disjoint natural set must be generated outside the repository.

## 8. Split, custody, and replication

T0 is a fast canary, T1 is development screening, T2 is a 1,024-per-family
sealed promotion suite, and T3 is newly generated fresh replication. T2 seeds,
answers, and fixtures are externally held; Git contains only a commitment hash.
Any outcome-guided access consumes the sealed fixture and requires a successor.

`ReplicationBundle` requires one checkpoint and evaluator identity across distinct
development, sealed, and fresh suite hashes. Raw-Core and assisted outcomes are
stored together only as separate condition records. No evaluator may tune a
model, tokenizer, data filter, or promotion threshold on T2.

## 9. Promotion intent (thresholds remain E0-calibrated)

Promotion is conjunctive, worst-family, and raw-Core first. Intended gates are
fresh-OOD selection above chance + 10pp, query-sensitivity above 80%, state/OOD
accuracy above 70%, two-hop composition above 60% versus matched retrieval,
conditional realization above 80%, substrate regression within 3%, no family
regression above 5pp, and replication at M102. Exact thresholds and confidence
intervals are frozen only after external T2 custody and power review.

## 10. Next experiment: E1 tokenizer tournament

E1 is prepared by `e1_tokenizer.tournament` and tests exactly 16,384, 24,576, and
32,768 identity-preserving byte-fallback candidates. Arms share the same external
corpus manifest, raw-byte budget, measured training-FLOP budget, committed probe
hash, and P35 comparison protocol. Static audits require zero unknowns and exact
Unicode/byte round-trip; matched training then measures byte-normalized loss,
nonce copy, identifiers, code/math, and E0 cognition.

No tokenizer winner is claimed until real artifacts and an external corpus
manifest exist. The planning center remains the provisional 250M / 26×896 / 5B
contract, but E1/E2/E3/E5 evidence can change it.

## 11. Current verdict

- **Scientific state:** local E0 generator and shortcut red-team are certified;
  sealed evaluation is not yet ready.
- **Largest repaired weakness:** semantic state and rule structure are now
  independent of serialization position and split symbols.
- **Highest-information next action:** obtain external T2/natural custody and run
  the matched three-arm E1 tokenizer tournament.
- **Main V5 run:** not authorized.
