# Ground Blueprint v0.1 — Decisive Experiment Program

Order is frozen: **E0 → E1/E2 → E3 → E4 → E5 → freeze review**. No production trainer or V5-A main run precedes these gates.

## Ranking by expected information gain per accelerator-hour

| Rank | Experiment | Decision value | Budget ceiling |
|---:|---|---|---:|
| 1 | E0 benchmark/generator certification | determines whether every later result is meaningful | CPU only |
| 2 | E1 tokenizer tournament | affects parameter allocation, sequence compute, copy, numbers, code | 3 serious P35 arms × 100–200M tokens |
| 3 | E3 data/objective screen | tests the central cognition-learning hypothesis | 5–7 P35 arms with early pruning |
| 4 | E2 architecture screen | selects depth/attention representation at fixed budget | fractional P35 screen; replicate top 2 |
| 5 | E5 M102 replication | prevents false small-scale transfer | 600M–1B tokens per main arm |
| 6 | E4 curriculum/optimization | improves reliability after recipe exists | ≤4 targeted arms; no broad sweep |

E1 and E2 may share baseline runs after E0, but scientific comparisons retain isolated variables.

## E0 — Benchmark and generator certification

This is the next action. It is specified here but not implemented in the design phase.

### Purpose

Prove that the benchmark measures causal cognitive operations rather than templates, candidate priors, tokenization artifacts, or evaluator bugs.

### Nine families

1. exact copy/local retrieval;
2. query-conditioned binding;
3. mutable state and overwrite precedence;
4. distractor/position-resistant retrieval;
5. relational composition;
6. counterfactual query/state sensitivity;
7. rule induction on novel symbols;
8. missing-information/abstention;
9. realization of a known selected value.

### Measurement cube

Each applicable family produces:

- representation: candidate NLL, rank, calibration;
- address: fact-fixed query-swap lift and flip direction;
- transform: state/relation result versus matched direct-retrieval control;
- selection: raw argmax and margin;
- realization: free output and output conditional on correct selection;
- assistance: constrained/normalized result reported separately.

### OOD axes

Entity alphabet, tokenizer fragmentation, template, prose/table/code style, domain, graph topology, relation labels, cardinality, hop count, context position, distractor density, sequence length, answer form, and generator implementation. Tier 2 uses joint shifts, not one easy axis at a time.

### Controls

- random and frequency-prior baselines;
- direct-retrieval control matched to composition surface statistics;
- structure-only policy baseline for intervention routing;
- query-blind, position-only, candidate-order, and template-ID heuristics;
- oracle scorer and deliberately broken scorer/model;
- byte-exact assertion that counterfactual pairs change only the intended field;
- duplicate graph and canonicalized-text collision scan;
- candidate-length and position balancing;
- answer-absent and multiple-valid-answer cases handled explicitly.

### Tier sizes

| Tier | Size per family | Role |
|---|---:|---|
| T0 | 32 | fast canary, never promotion |
| T1 | 512 | development curves and checkpoint screening |
| T2 | 1,024 | sealed promotion suite |
| T3 | newly generated | fresh replication and natural transfer |

Sizes may increase if E0 power analysis shows wide intervals. They may not shrink for convenience after seeing model results.

### Certification tests

- deterministic regeneration from version + seed;
- independent solver agreement and property tests;
- exact expected chance rate for every metric;
- heuristic baselines fail the intended causal gate;
- oracle passes; deliberately broken systems fail in the predicted dimension;
- selection and realization defects produce different measurement signatures;
- direct retrieval cannot satisfy composition gate;
- all split dimensions and hashes are frozen before any model sees T2;
- no training or data-filter component can import T2/T3 content;
- bootstrap/binomial confidence procedure is preregistered;
- natural analogues are source-disjoint from training candidates.

### E0 exit gate

E0 passes only when every family has an unambiguous scorer, chance/control behavior, no discovered shortcut, a frozen T2 hash, and a documented fresh-generation route. Failure keeps E1–E5 unauthorized.

## E1 — Tokenizer tournament

Candidates: 16,384 / 24,576 / 32,768 identity-preserving byte fallback. Evaluate tokens per raw byte by domain, zero unknowns, context-answer consistency, Unicode round-trip, number direction/granularity, nonce copying, identifier/code fragmentation, byte-normalized loss, memory/throughput, and E0 cognition. Match raw bytes and approximate training FLOPs.

Select a Pareto winner, not a weighted-score winner hidden by an arbitrary aggregate.

## E2 — Architecture attack

At ~35M compare deep/narrow, middle, and wide/shallow iso-parameter shapes. Use a fractional design for 4-KV GQA versus MHA, QK norm on/off, and 2k/full versus 4k/mixed-length full attention. Single-seed screening may eliminate clearly dominated arms at preregistered token boundaries; top two shapes receive three seeds.

Primary outcome: worst-family fresh-OOD cognition per measured training FLOP, with substrate loss and throughput constraints.

## E3 — Data and objective

Compare 5/15/30% verified cognition mixtures under CE. Add query-swap contrast at the best mixture and one neighbor, λ 0.05/0.15. Include one bounded intermediate-trace arm only if composition remains the decisive gap. Match tokens and raw bytes; replicate finalists on new generators and natural analogues.

Reject any auxiliary that improves candidate scores but not candidate-free output, causes >3% substrate-loss regression, or fails fresh transfer.

## E4 — Learning dynamics

Compare uniform mixing with one staged/replay schedule. Test 131k versus 262k tokens/update with paired LR range 2e-4/3e-4/4e-4 only as needed. Compare WSD against a simpler schedule only if the learning curves create a decision. Do not run a full grid.

## E5 — M102 scale-transfer gate

Train ~102M winner and a strong CE/general-data control for 600M–1B tokens. Use two winner seeds. Require exact resume, sealed/fresh cognition gains, natural transfer, substrate retention, and compatible effect direction from P35.

If E5 fails, do not implement or train V5-A. Return to the failed causal hypothesis, not an unplanned bigger model.

## Freeze review

Produce `V5_TRAINING_SPEC_v1.0.md` only after E0–E5. It contains exact model/tokenizer/data/objective/optimizer/checkpoint/evaluation values and hashes. Target-device real-update, exact-resume, durable-upload, and clean-download restore canaries are then mandatory.

## Program abort rules

- evidence/contamination/provenance failure invalidates the affected result;
- non-finite or fake-update behavior stops the run;
- synthetic-only gain rejects the recipe;
- auxiliary objective without fresh transfer rejects the objective;
- M102 non-replication blocks V5-A;
- measured corpus below the required unique quality budget triggers model/token recalculation;
- two consecutive cognition regressions >5 points while LM loss improves pause training and preserve the earlier milestone;
- production implementation remains out of scope until the freeze review authorizes it.
