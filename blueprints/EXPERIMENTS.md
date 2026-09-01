# Ground Blueprint v0.4 — Decisive Experiment Program

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

Development infrastructure is implemented in `e0_cognition/` and has a deterministic development certificate. It has **not** frozen or consumed a real sealed suite.

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

Current result: **DEVELOPMENT INFRASTRUCTURE PASS; FULL E0 EXIT NOT YET PASSED.** The current 368-case/112-pair receipt proves generator determinism, causal-pair contracts, split namespace isolation, model-view secrecy, template separation, semantic-time state tasks with shuffled serialization, eight split-held-out rule structures, pooled positional/fixed-rule/bag-of-words red-team gates, independent surface-solver agreement over a 20-seed property sweep, candidate/context position, difficulty, and answer-format coverage, and explicit chance/power calibration. Raw-Core versus assisted result contracts, intervention-dependence measurements, sensitivity/invariance summaries, and fresh/sealed replication contracts are implemented. Paired exact/10,000-resample bootstrap procedures are machine-preregistered. Source-disjoint natural custody and the externally held T2 commitment remain.

## E1 — Tokenizer tournament

Candidates: 16,384 / 24,576 / 32,768 identity-preserving byte fallback. Evaluate tokens per raw byte by domain, zero unknowns, context-answer consistency, Unicode round-trip, number direction/granularity, nonce copying, identifier/code fragmentation, byte-normalized loss, memory/throughput, and E0 cognition. Match raw bytes and approximate training FLOPs.

Select a Pareto winner, not a weighted-score winner hidden by an arbitrary aggregate.

Current result: the artifact-bound E1 static audit, Pareto harness, and matched-budget tournament plan exist in `e1_tokenizer/`. A local development tournament now independently trained exact 16k/24k/32k byte-BPE artifacts on 8,561,653 training bytes and evaluated 1,057,977 hash-held-out bytes across 14,757 records. All compressed artifacts reload, all arms have exact round trip and zero unexpected unknowns, and a repeated 24k build is byte-identical. Held-out tokens/byte are 0.23826 / 0.23217 / 0.23022, so 24k costs only +0.85% tokens versus 32k while saving 7.34M embeddings at width 896; 16k costs +3.50%. This is stronger than prefix truncation but remains `DEVELOPMENT_STATIC_PASS`: the fixed legacy/local corpus is not representative, and no model loss or cognition was measured. The next promotion input remains a hash-bound external corpus and matched P35 runs.
An additional deterministic perturbation sweep (64 cases each for numbers, identifiers, nonces, Unicode, spacing, and formal expressions) has zero round-trip failures and zero unknowns for all three candidates. Overall tokens/byte are 0.63834 (16k), 0.60568 (24k), and 0.59293 (32k); 24k is intermediate on every family, including nonce `0.56756` and number `0.53815`. This improves coverage of the cognition-relevant surfaces but remains static local evidence, not an E1 promotion.

## E2 — Architecture attack

At ~35M compare deep/narrow, middle, and wide/shallow iso-parameter shapes. Use a fractional design for the scale-consistent 2:1 GQA topology versus MHA, QK norm on/off, and 2k/full versus 4k/mixed-length full attention. Single-seed screening may eliminate clearly dominated arms at preregistered token boundaries; top two shapes receive three seeds. Earlier 3:1 kernel canaries remain mechanism evidence, not the recipe arm.

Primary outcome: worst-family fresh-OOD cognition per measured training FLOP, with substrate loss and throughput constraints.

Current executable result: `e2_architecture.plan` fixes 200M-token screening boundaries at 50M/100M/200M and three-seed finalist replication. Shape arms are 24×320/FFN768 (35.420M), 16×384/FFN896 (35.414M), and 8×512/FFN1152 (35.144M), under 0.8% parameter spread with MHA/QK/2k held constant. Separate fractional arms isolate MHA versus 3:1 GQA, QK norm, and 2k versus 4k context. Static receipts expose parameter, idealized 6ND, full-sequence forward, and KV-cache proxies. A three-seed RTX 4050 bf16 SDPA microbenchmark found a backend-specific trap: native 3:1 GQA supports only the math backend on this PyTorch/Windows build and is 5.20× MHA training-kernel latency with 13.86× peak allocated memory. Explicit repeat-K/V restores the faster path but is 1.09× MHA latency and 1.45× memory. Native 4k versus 2k is 3.59× latency and 3.80× memory; QK normalization adds 1.09× latency. GQA semantics agree with repeated K/V to max absolute error 0.0078125. This constrains implementation selection on this stack, not GQA quality or TPU behavior. Plan status remains `BLOCKED_E1_INPUTS`; no P35 run is authorized.

The exact full-stack execution canary now instantiates each P35 shape with tied embeddings/logits, all blocks, RoPE, affine QK norm, MHA, SwiGLU, cross-entropy, and backward—but no optimizer update. Three randomized-order CUDA seeds at sequence 512/1024/2048/4096 pass exact parameter counts and finite-gradient checks. Wide/shallow is 3.17× deep and 2.05× middle latency at 512; at 4k the advantages narrow to 1.91× and 1.44×. Peak 4k allocation is 2.57 GB wide, 3.14 GB middle, and 3.54 GB deep. A single bounded CPU sequence-64 check has the same ordering (wide 308 ms, middle 385 ms, deep 423 ms). This is evidence that layer count/kernel-launch cost matters on the measured stacks; it cannot select the shape without matched cognition per measured FLOP.

A paired initialization canary now attacks the provisional `1/sqrt(2L)` residual-output scaling directly. Five short-context CUDA seeds and three CPU seeds compare identical `normal(0,0.02)` draws with only attention-output and FFN-down matrices rescaled. Short-context CUDA scaled/unscaled end-to-end residual-growth ratios are 0.122/0.156/0.230 for deep/middle/wide, with gradient-spread ratios 0.604/0.650/0.731; CPU reproduces both directions. A separate three-seed CUDA run at the intended 4,096-token context passes with even stronger ratios: 0.115/0.144/0.217 growth and 0.542/0.510/0.654 gradient spread. Classification is `SUPPORTED_LOCAL_SIGNAL_PROPAGATION`, with exact counts and finite/nonzero gradients throughout. This promotes residual-output scaling from convention to a local implementation prior—not to a learning or cognition result. Target-TPU constructor replication and bounded real-update E4 checks remain mandatory.

The QK-norm hypothesis now has a paired mechanistic test rather than only a convention citation. With the same hidden states/projections/queries, QK norm makes attention logits and entropy invariant to a 0.25×/1×/4× Q/K projection-scale stress; the unnormalized control exposes the expected 256× logit-RMS change and a 0.365–0.416 normalized-entropy span on CUDA. Five CUDA seeds through 4k and three CPU seeds agree, and every proxy backward is finite/nonzero. At 4k base scale, normalized attention covers an effective 0.616 of available keys versus 0.988 without QK norm, showing that the unnormalized `std=0.02` initialization is almost uniform. Classification is `SUPPORTED_QK_SCALE_CONTROL`, not “QK norm improves cognition”: E2 must still compare fresh-OOD learning, and learned affine scales plus optimizer dynamics remain open.

The provisional BF16 path now has exact-stack numerical evidence. Identical FP32/BF16 scaled-residual weights and random-token batches were compared over three seeds for all P35 shapes on CUDA (sequence 256) and CPU (sequence 64). All cases pass fixed loss/logit/gradient limits. A further three-seed CUDA replication at native 2,048 context also passes every shape, with worst logit cosine 0.999917, logit RMS error 1.289%, representative-gradient cosine 0.999631, and gradient RMS error 2.717%. Across all receipts, worst loss relative error is 0.000118. Classification is `SUPPORTED_LOCAL_BF16_PARITY`. This authorizes BF16 for future local canaries through P35 native context, not a main run: 4k V5-A, target TPU/XLA, real-update, optimizer-state, long-duration, and data-tail checks remain open.

The RoPE conformance canary now checks the extracted P35 implementation against an independent float64 oracle at native 4k. Fresh five-seed CUDA and three-seed CPU receipts pass for FP32 and BF16, including norm preservation and relative-shift invariance. The first calibration correctly failed its overly strict float32 round-off limits; those limits were analytically revised and the fresh receipts pass. This is implementation evidence only; RoPE base selection, extrapolation, TPU/XLA, and learned cognition remain open.

The real-update canary now runs three deterministic AdamW updates plus a save/load continuation on both CUDA and CPU. All live parameters are optimizer-owned, parameter hashes change, Adam steps reach 3, moments become nonzero, post-clip norms stay at or below 1.0, and both resumed parameters and optimizer states are exactly equal to uninterrupted execution within each storage dtype. The corrected BF16 path keeps FP32 master parameters under BF16 autocast and passes; the native-BF16-parameter control is retained as a fail-closed negative receipt because its Adam moments are BF16 and its post-clip norm overshoots 1.0 by about 0.3%. This certifies update wiring and resume continuity only; long-run optimization, distributed resume, and TPU/XLA remain open.
The canary now accepts every E2 shape arm and explicitly releases full optimizer state before constructing the resumed copy; this prevents a local host-RAM artifact from masquerading as a training failure. Fresh deep-narrow and wide-shallow 3-update CUDA/CPU receipts pass the same wiring, clipping, FP32-state, and tolerance-resume gates as the middle arm.
A fresh-seed 10-update replication of the corrected path also passes on CUDA (sequence 32) and CPU (sequence 16): all losses and gradients remain finite, FP32 moments remain active, and optimizer-state resume equivalence is exact. This is still a bounded stability signal, not evidence for the final learning rate or cognition.
At a more realistic context, CUDA sequence 1,024 and CPU sequence 128 pass the calibrated tolerance-based resume gate for both parameters and optimizer state. A strict CUDA run may pass on a deterministic local runtime or fail from reduction-order drift; neither outcome is portable. The receipt therefore records strict equality separately from the registered tolerance result, which is the production gate.
The companion cursor canary consumes 9×47 tokens across shuffled content-addressed shard/sequence boundaries, round-trips the cursor through JSON, and reproduces the uninterrupted tail exactly on CPU and CUDA. Manifest and offset tampering are rejected, and host/device token digests agree. This closes local single-process cursor continuity only; distributed sampler partitioning, real pack I/O, and durable remote restore remain open.

## E3 — Data and objective

Compare 5/15/30% verified cognition mixtures under CE. Add query-swap contrast at the best mixture and one neighbor, λ 0.05/0.15. Include one bounded intermediate-trace arm only if composition remains the decisive gap. Match tokens and raw bytes; replicate finalists on new generators and natural analogues.

Reject any auxiliary that improves candidate scores but not candidate-free output, causes >3% substrate-loss regression, or fails fresh transfer.

Current executable result: `e3_data_objective.plan` fixes a 200M-token Phase A with exact 5/15/30% cognition allocations while preserving the 65:20 natural/code ratio in all remaining tokens. Phase B cannot instantiate until Phase A identifies a winner and adjacent mixture; its only arms are CE and query-swap λ 0.05/0.15 on mechanically verified pairs. The trace arm is disabled unless a hashed composition-transfer failure triggers at most 25% exposure, followed by trace-free evaluation. Natural transfer, candidate-free/raw-Core improvement, and ≤3% substrate-loss regression are conjunctive gates. Status remains `BLOCKED_UPSTREAM_INPUTS`; no run is authorized.

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
