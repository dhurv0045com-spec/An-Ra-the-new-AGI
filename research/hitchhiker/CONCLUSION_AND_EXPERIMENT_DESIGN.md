# Hitchhiker: conclusion and mechanism-identification experiment

Date: 2026-11-14. Author: Codex. Status: repository audit and proposed protocol; no implementation or new model results.

## Conclusion first

Proceed with a controlled grounding-and-binding investigation. The strongest improvement to the supplied proposal is to separate **task interpretation, evidence extraction, action legality, variable binding, and composition** before combining architectural mechanisms. Building A–H together would make a positive result difficult to explain and a negative result almost impossible to debug.

The V7.17 evidence supports optimization sensitivity and severe transfer failures. It does **not** establish a memorization basin, an absence of reasoning circuits, or a capacity ceiling. Lower learning rate improved held-out value accuracy and several evidence-intervention panels as well as confirmation. That is more informative than “better memorization.” Direct lookup, surface dependence, and action-selection failures remain competing explanations.

My recommendation is: frozen-checkpoint diagnosis; a matched static-versus-fresh-data and relabeling experiment; targeted availability training; then slot, recurrent, and associative-memory comparisons only where the preceding evidence warrants them. Keep external canonicalization and verification as explicitly engineered controls. A useful compact solver is a valid outcome; learned representation formation requires additional evidence.

This document is the requested first deliverable. Implementation, GPU training, and checkpoint inference are pending your next instruction.

## 1. Source and evidence boundaries

- Reviewed source: `0c9cb284d2c8c949e17ac342959c6f2bf351bd30`, clean tracked checkout at `C:/Users/ankit/Downloads/An-Ra-Elendril-V7` when inspected.
- Recorded V7.17 optimization run: `f443d3f4141b4163bbc130d46ebd7395`; actual run source `724af4cb508173e0eb3f4397d5dcb4a42a66712f`. The reviewed HEAD contains the later analysis, not the original training source identity.
- `hitchhiker` starts at `2affffb806b7f73b00bc21aa90538019f37ce904`, an ancestor of that reviewed HEAD. This document lives in its own worktree at `C:/Users/ankit/Downloads/An-Ra-hitchhiker`. The branch does not yet contain the subsequent V7 implementation. Before building, advance/merge the reviewed implementation with an explicit source receipt; do not unknowingly implement against the older V6 baseline.
- The supplied numbers agree with the committed owner report. That report records replay of 44 evaluation reports and audit of 32 diagnostic milestones. I read the report and relevant implementation; I did not independently rerun its archive analyzer in this review.
- The report says the supplied analysis ZIP lacked weights, the FULL ZIP, and final delivery certification. This review has not established availability of those external checkpoints. Raw runner-up logits and attention maps cannot be reconstructed from aggregate scores or a selected-action confidence.
- No V8/V8.1 design document was found in the searched tracked research/docs trees of the reviewed V7 checkout or the Elendril base. V8 compatibility below is therefore a proposed contract based on the existing retention interface and the supplied brief, not verification of an unseen V8 design.

## 2. What the code changes about the hypotheses

| Question | Finding | Experimental consequence |
|---|---|---|
| Are worlds static? | `v6_data.make_dataset` materializes cases/examples/samples. V7.17 repeats 9,472 examples for four epochs: 2,368 batch-16 updates. | A is a meaningful freshness intervention, but it must separate new worlds from new renderings and relabelings. |
| Are entity identities fixed associations? | `_case` already samples A–P entities and relation endpoints independently across worlds. | The training set is finite, but there is no generator-wide rule A→B to memorize. Whole-prompt, template, position, and task shortcuts remain possible. |
| Is confirmation leaking exact worlds? | `_state_key`, `_partition`, `audit_dataset`, and `assert_disjoint` enforce normalized observation-state/world/prompt separation. Confirmation uses a separate ledger constructor with related semantics. | No direct leak is demonstrated by this review. Shared labels/templates/distributions are in-distribution overlap, not proof of leakage. Audit symmetry-equivalent families separately. |
| Are novel aliases new token embeddings? | `representation.UTF8` uses 256 byte tokens plus four specials. `node-A` and `unit/A` are byte sequences; the answer stays A. | There are no newly allocated whole-entity rows. Test byte exposure, sequence length, positions, formatting, and instruction interpretation. Do not assume every alias byte is equally trained. |
| Is there a dedicated action head? | `v5_model.core` projects hidden states with the same embedding weight used on input. Actions are output bytes A–P, Q, R, !, or ~ plus a letter. | Random input embedding replacement can break the tied output geometry. Begin with consistent entity relabeling, not arbitrary vector injection. |
| What receives loss? | `Tokenizer.frame` masks prompt loss and supervises the answer and EOS. V7.17 uses CE. `public_action` computes the teacher answer but does not expose its internal traversal. | Endpoint supervision permits any predictive shortcut. There is no explicit intermediate-hop objective in this recipe. |
| What are the six representation jobs? | `v717_protocol.jobs('representation')`: three original baselines and three row-order arms, all at **3e-4**. `prepare_variant` deterministically reverses rows in half the worlds. | These are not binding/slot training and not A–D. Do not silently relabel an amended 1e-4 experiment as the original preregistration. |
| Has action masking already been tried? | `GrammarPolicy` filters legal first actions and forces grammar/EOS. The report records lower-LR grammar three-hop **0/48**. | A separate availability-only mask remains useful for attribution, but a large rescue is not established and full grammar has already failed on this panel. |
| Is S9 defined? | Registry profiles are S3, S8, S20; S20 has 19,871,616 parameters, seven blocks, width 512. | Use registered S8 or define a new S9 with an exact parameter receipt. Do not silently rename S8. |

### Why 84.52% does not establish generalizable one-hop lookup

Confirmation is a mixture of known, missing, conflict, null, unavailable, route, and memory cases. The recorded lower-LR models perform well on query/retrieve decisions and many returned-evidence states, while direct known-value and route states remain weak. Aggregate success can rise through these easier or better-supported pathways without solving the novel chain task.

The training and novel chain distributions also differ in several concrete ways:

- Base generation uses ticks 2–799, with nearby offsets; `_chain` samples its initial tick from 1–9999.
- Base route facts share a tick, include three distractor rows, and are shuffled. Novel chains contain only path edges, serialized in traversal order with increasing ticks.
- Base routes train two-hop questions. The diagnostic introduces one-, three-, and four-hop requests; “Current value of A?” and “Follow 1 links from A” are different question forms even when constructed to have the same answer.
- Base templates alternate; chain surfaces use the default template. Alias prefixes add an explicit output-instruction clause as well as bytes and length.

These are measurable shifts, not established causal explanations. The first experiment must vary them independently. A model that fails one-hop under a new request form has not thereby been shown incapable of all one-hop computations.

The old split key retains entity identities and is not a full graph-isomorphism partition. A renamed graph family can fall on another side of the hash partition. Fresh-data/relabeling experiments must prevent evaluation-family crossing. Conversely, requiring every simple one-hop topology to be topology-disjoint is impossible: there are too few meaningful one-edge structures. Define separate identity, surface, and structural-composition tests with honest claims for each.

## 3. Stage 0: diagnose frozen checkpoints before training

### Artifact admission

Restore the three 1e-4 S20 final checkpoints for seeds **7, 19, 41** if available. Verify source/protocol/data hashes, architecture/tokenizer, tensor fingerprints, update count, and original report binding. Replay a preregistered subset followed by the original cohorts. Preserve discrepant receipts. Missing weights mean **checkpoint audit pending**, not permission to fabricate logits or describe retraining as a replay.

### Matched public-world suite

Start from at least 256 independently sampled latent worlds per hop for development, with separate held-out confirmation worlds. Publish the count and compute forecast before execution. Keep each world's variants together. Include the old 16-world panels solely for continuity; their repeated evaluation across seeds is not 48 independent worlds.

For each applicable world, independently manipulate:

1. One-hop “current value” versus “follow one link” wording, with identical visible facts and verified equivalent semantics.
2. In-range versus larger timestamps, with relative order and truth fixed.
3. Original, reversed, and random row order; path-position shortcuts must fail under permutation.
4. No distractors versus matched-count distractors; vary row relevance separately from total length.
5. A–P identity permutation with the question, all visible/hidden evidence, and targets transformed consistently.
6. Alias prefix versus length-matched neutral formatting; instruction-clause presence as a separate factor.
7. Answer-changing edge counterfactuals with matched answer marginals; also query-subject changes on the same graph.

Use controlled generation to prevent accidental extra paths, conflicting equal-time facts, or changes in availability. A verifier checks counterfactual semantics before model execution. Actor inputs contain public observations only; gold joins during scoring.

### Three separate decoder conditions

- **Raw:** unchanged byte generation and parsing.
- **Availability-only:** at the first action byte, set R to negative infinity when memory is unavailable and Q when a query is unavailable. Leave all answer letters, !, ~, other bytes, and subsequent EOS behavior unchanged. Do not silently apply a full syntax mask.
- **Existing grammar control:** reproduce the current `GrammarPolicy`, including its forced continuation rules, under a distinct identity.

Store the complete 260-way first-byte logits, raw probabilities, top alternatives, gold-answer rank, illegal-action mass, and EOS continuation. Report `z_R - max(z_A,…,z_P)` where relevant and the total probability of all answer letters; the highest individual byte alone mixes action type and value choice. Complete-action scores may include EOS probability but must be labeled separately from first-byte scores. Preserve raw and masked probabilities independently.

An answer-only forced-choice ranking can be an additional diagnostic, but it forces answering and is not an autonomous policy success. ABSTAIN must remain available in the primary availability-mask condition.

Interpretation rules:

- Mask improves legal actions but not correct answers: controller restriction helped legality; evidence-to-answer mapping remains unresolved.
- Mask improves paired correct answers: action selection contributed under that intervention; this does not prove a specific internal binding circuit.
- Wording/timestamp/row changes rescue otherwise identical worlds: prioritize that distribution mismatch before changing architecture.
- Correctness changes with an irrelevant entity permutation or distractor: behavior is not adequately invariant.
- Correct answers follow answer-changing edge interventions: stronger behavioral grounding evidence than ordinary accuracy alone.

Attention entropy is optional instrumentation, not the first gate. Measure a specified head/layer/query position, normalize by log(number of allowed keys), and match lengths. Verify captured attention preserves forward outputs for the chosen backend. High entropy is not proof of confusion; low entropy is not proof of correct evidence use. Causal evidence edits and activation ablations are more informative than a heatmap alone.

## 4. Refined mechanism designs and falsifiers

### A — Fresh worlds, not an unbounded training promise

Use a finite, reproducible procedural stream indexed by `(generator_version, data_seed, update, batch_slot)`. Generate on CPU with caching/prefetch if beneficial, and record accepted/rejected candidate counts and generation time. There is no justified “zero overhead” claim.

First compare a static pool against fresh worlds under the **same distribution**, task mixture, label marginals, update count, batch size, supervised-token budget, and initialization/order pairing. Measure input positions and actual work. Adding one-/three-hop training or balancing availability is a separate intervention; otherwise a gain could be better task coverage rather than freshness.

Preserve split-family identities before augmentation. Rejection-sampling, distributed worker scheduling, and resume must reproduce the same stream. Repeated microbatches within an accumulated update must not accidentally resample the semantic example.

**Falsifier:** fresh worlds give no reliable paired transfer benefit at matched exposure, or gains disappear when task coverage and compute are controlled. Freshness makes exact example reuse harder; it does not mathematically force rule learning.

### B — Entity equivariance, before random embeddings

Apply a bijection π over A–P to semantic entity fields, then rerender and transform ANSWER/DELIBERATE arguments. Preserve action tokens, prose, sources, and timestamps. Raw string replacement would also corrupt instructions. Store the mapping and inverse. Use the same mapping throughout an episode and its acquired evidence.

The desired property is `p(π(y) | π(x)) ≈ p(y | x)` for entity answers; Q/R/! remain fixed. Measure argmax equivariance and inverse-mapped distribution divergence. Accuracy must accompany consistency: a consistently wrong or uniformly random model can be equivariant.

A consistency loss between matched views is a later sub-arm; charge its extra forwards and compare to an equal-work CE control. Do not fold it invisibly into the relabeling treatment. Arbitrary orthogonal vector perturbation is deferred because it changes both representation and input/output alignment in this tied-byte model.

**Falsifier:** performance survives training permutations but fails withheld alias/format or edge interventions. That demonstrates a narrower invariance, not broad symbolic grounding.

### C — Separate availability training from an engineered mask

Train matched examples crossing evidence completeness and action availability: complete context with memory on/off; missing bridge with relevant memory; missing bridge with no external evidence; available but irrelevant memory; returned evidence requiring revision. Compute labels with the public teacher under the actual state semantics. Context dropout must recompute solvability/labels rather than retaining an invalid old answer.

A base CE contrast tests supervision coverage. Only afterward test an auxiliary term such as `L = L_CE + λ[-log(1 - p_illegal)]`, with a numerically stable implementation and illegal mass defined at the first action byte from public availability. CE already discourages wrong actions, so additional benefit cannot be assumed. Freeze λ on development data.

**Falsifier:** illegal actions disappear but wrong answers/abstention replace them, or necessary retrieval recall falls. Report both unnecessary retrieval and failure to retrieve when needed.

### D — Slot indirection as an attributable control

First build a public-only canonicalizer mapping entity mentions to episode-local slots and dereferencing outputs. It must not compute paths, remove distractors, resolve conflicts, or peek at the answer. Test random slot assignments as well as first-mention order, since canonical ordering can create new positional shortcuts. Charge parsing, extra tokens, dictionary limits, and output adaptation.

External slots can make entity-renamed inputs equivalent only under the defined mapping and renderer; they do not automatically equalize arbitrary aliases, syntax, or ambiguity. The present event/goal contracts accept only A–P and the output parser accepts letters. Arbitrary names, pointer outputs, or new slot tokens require a versioned interface and evaluator, not a silent token replacement.

Keep **external canonicalization** and **learned slot formation from raw bytes** as different arms. For a learned binder, measure mention-to-slot assignment, cross-mention consistency, pointer correctness, and reset between episodes. Any assignment supervision is additional supervision and must be declared.

**Falsifier:** external slots work but the raw-input learned binder fails. This localizes an interface/formation limitation and establishes an engineered solver, not autonomous representation formation.

### E–H — Conditional second-wave architecture

| Mechanism | Cleaner version | Required comparison / failure evidence |
|---|---|---|
| E: recurrent reasoning | Shared transition block over a working state; fixed 1/2/3/4 iterations first, learned stopping later. | Compare equal-work feed-forward computation and repeated inert/no-op state updates. The current model has seven blocks per output token and repeated environment decisions; it has no explicit one-recurrent-step-per-hop mechanism. Depth limitation is not established by a one-hop failure. Measure longer unseen paths and stepwise causal interventions. |
| F: falsification | Candidate-conditioned evidence verifier with hard negatives from wrong edges, stale facts, and missing bridges. | Reverse traversal need not be unique in directed many-to-one graphs. Verify actual forward support rather than assuming an inverse function. Compare to equal-cost candidate scoring and random negatives. A verifier sharing the same shortcut can agree while wrong. Training-only consistency adds no inference head but cannot provide an inference-time rejection guarantee. |
| G: entropy gate | Calibrated routing to extra computation or context processing, based on specified output uncertainty and public availability. | Compare fixed routing and availability-only routing at equal cost. The recorded invalid RETRIEVE confidence near 0.9994 already warns that error need not be uncertain. Generic hidden vectors have no intrinsic Shannon entropy without defining a distribution. |
| H: associative scratchpad | Episode-local learned keys/values with explicit reset, update semantics, collision measurement, and iterative reads. | A sum of key/value outer products is not automatically orthogonal or interference-free. Compare ordinary attention and slots at matched work, test key collisions, slot capacity, stale overwrites, conflicting sources, and cross-episode contamination. Learned write/read ablations must affect behavior if the mechanism is doing useful work. |

For E/H, use intervention tests: corrupt or exchange an intermediate state on paired worlds and check predicted downstream changes against matched sham perturbations. Probe decodability alone shows information is present, not that the model uses it. Intermediate supervision is an explicit extra treatment.

## 5. Training matrix that identifies effects

Keep seeds 7/19/41 fixed. Treat the LR of 1e-4 as the primary S20 comparison recipe, not a proven optimum at every size. Any per-scale tuning receives the same declared search budget and uses only development data. Preserve historical 3e-4 results as labeled historical comparators.

**First S3 screen: 15 runs.** Use five configurations × three seeds:

| Arm | Purpose |
|---|---|
| B0 | Static original-distribution CE baseline at 1e-4. |
| A | Fresh worlds; same task distribution and exposure as B0. |
| B | Static worlds with repeated semantic entity relabeling. |
| AB | Fresh worlds plus relabeling; identifies A×B interaction. |
| C | Static matched mixture with availability/task-coverage contrast supervision; no auxiliary penalty initially. |

This uses the same 15-run budget as the proposed cumulative ladder while providing direct A, B, and interaction contrasts. C estimates a declared supervision-package effect, not an isolated penalty effect. D is excluded from this first grid because it changes the representation boundary.

**Conditional follow-ups, explicitly budgeted:** AB+C (three runs) if complementary gains warrant combination; C+penalty (three runs) if availability errors remain; external D versus its matched raw recipe (at least three new runs if that comparator already exists under identical conditions); learned D/E/H only after their input/output contracts are frozen. A negative S3 result does not disprove a mechanism at S20: allow one preregistered S20 three-seed contrast when capacity or optimization remains a plausible interaction.

**Scale confirmation:** the winning recipe on S8 and S20 across three seeds is **six treatment runs**, not nine. Add six same-protocol baselines if they are not already available: twelve runs total. Nine describes all three scales' treatment runs only if counting the completed S3 treatment. Historical S20 baselines cannot substitute when data or training contracts differ.

No extrapolation of 111.48 controller minutes to this campaign: the report also gives 218.14 summed worker GPU-minutes, and new generation, sequence lengths, instrumentation, and recurrent computation change cost. Time representative batches and complete diagnostic passes before fixing concurrency and launch budgets.

## 6. Evidence, uncertainty, and promotion criteria

Freeze metrics, splits, counts, and candidate selection before any new training. The historical confirmation set has already informed this design; retain it as a regression set and use a new untouched set for final confirmation.

Primary endpoint: raw exact terminal correctness on the matched renamed, surface-shifted, context-complete one-hop panel. Two-/three-hop, illegal actions, and paired answer-changing interventions are prespecified secondary endpoints. Separate interpolation (train and test each hop) from extrapolation (train 1–2, test 3–4). Training on three-hop and then passing three-hop is not evidence of unseen-depth transfer.

Report per seed and per cohort, not just a pooled number. Bootstrap paired **latent worlds**, keeping all aliases and counterfactuals clustered; report seed effects separately. Three seeds give limited evidence about optimizer variability. For planned A/B/AB contrasts, declare multiplicity handling, such as Holm adjustment for their paired tests, before execution. Select on development data and confirm the frozen winner once.

Proposed screening gate, to freeze before runs: ≥10 percentage-point mean primary gain, positive effect in all three paired seeds, no >5-point decline in any original task kind, and confidence bounds compatible with a positive world-level effect. Treat a smaller or imprecise effect as inconclusive, not proof of no effect. Above-zero success on 16 cases is not a meaningful formation gate.

Proposed formation gate: ≥90% raw accuracy on each intended hop cohort, ≥80% on every declared stress subgroup, ≥90% both-correct answer-changing pairs, ≥95% renaming consistency accompanied by accuracy, and ≤1% invalid actions. These are proposed engineering/research acceptance thresholds, not a definition of cognition or existing measured achievements. Use at least 512 independent confirmation worlds per primary cohort, and size further after a preregistered power/precision analysis. No finite zero-failure sample establishes a zero true error rate.

Health probes must respect this repository's semantics:

- **Distractors:** include disconnected chains, misleading row positions, and overlapping entities while retaining a unique valid answer. `source` is provenance, not a relation type. Different-relation trap graphs require a new relation-aware event/goal contract and independent verifier.
- **Revision:** newer facts from the same source supersede older facts; disagreement among the latest facts of different sources requires resolution. The current `belief` is not global “last text wins.” Test both cases and irrelevant updates.
- **Missing bridge:** score safe ABSTAIN when no evidence can be acquired, plus answer coverage when solvable and useful QUERY/RETRIEVE when evidence exists. An always-abstain system must fail the combined gate.
- **Preservation:** independently acquire task A, acquire B, retest A, then compare C learning curves against scratch and equal-total-work controls. Reset episode memory; measure retained weights and episodic facts separately. The current `retention_panel` already refuses preservation claims without independent acquisition and matched cumulative work.

## 7. Concrete implementation handoff prompt

> Implement Hitchhiker against the reviewed V7.17 source, preserving historical protocol/report identities. First resolve the documented branch ancestry and create an explicit new protocol namespace. Build artifact admission and read-only checkpoint diagnostics before changing training. Implement paired public-world counterfactuals, raw/availability-only/grammar decoder receipts, full first-byte distributions, and independent scoring. Verify that masking uses only public availability and that transformations preserve or deliberately change truth as declared.
>
> Next implement deterministic static/fresh/relabeling data modes with resume-stable identities and family-level split custody; implement the five-arm S3 registry and complete per-run cost accounting. Keep C's coverage intervention separate from its optional auxiliary loss. Add D as an externally attributed canonicalization control before any learned binder. Do not introduce E–H into the first comparison. Add them as independently registered experiments after the diagnostic gates identify a reason.
>
> Required engineering checks: bijection/inverse output consistency; public-only actor inputs; counterfactual verifier parity; all-stage split separation; unmodified raw decoding; availability-only mask scope; finite loss and exact accumulation denominator; stream/resume equivalence; interrupted-run recovery; immutable selection; complete negative/failure receipts. Use tiny models for mechanical checks and registered full profiles for research evidence. Tests passing is not model formation.
>
> Publish an experiment manifest, paired episode-level results, confidence intervals, per-seed contrasts, model/data/source hashes, checkpoint restoration receipts, resource measurements, and a mechanism decision table. Mark every claim as implemented, observed, inferred, inconclusive, or pending. Never turn a pending run into a negative result or credit external parsing/masking as learned cognition.

## 8. Evidence map for the receiving builder

Paths below are relative to the audited source commit, not necessarily present on the current older hitchhiker base:

| Source | Relevant evidence |
|---|---|
| `anra_v5/elendril/v6_data.py` | `_case`, `_confirmation_case`, `render_public`, `public_action`, `_state_key`, `make_dataset`, `assert_disjoint`. |
| `anra_v5/elendril/representation.py` | `Tokenizer.encode`, `Tokenizer.frame`, UTF8 vocabulary and loss mask. |
| `v5_model/core.py` | `forward_hidden`, tied embedding output projection. |
| `anra_v5/elendril/v6_registry.py` | S3/S8/S20 profiles and existing formation/retention gates. |
| `anra_v5/elendril/v7_training.py` | Blocked first-epoch sampling and fixed epoch permutations. |
| `anra_v5/elendril/v717_protocol.py`, `v717_training.py` | Exact optimization/representation jobs, learning rates, row intervention and CE recipe. |
| `anra_v5/elendril/v71_data.py` | `_chain`, `_clamped`, existing route-training generator. |
| `anra_v5/elendril/v6_evaluation.py`, `v71_policy.py` | Raw parsing/generation, grammar control, independent replay and retention qualification. |
| `anra_v5/elendril/backend.py` | `logits` and unmasked `next_distribution`; full-prefix inference cost. |
| `anra_v5/elendril/cognition.py` | Entity contracts, per-source latest-time belief, episode state and budgets. |
| `Experiment Result/V7.17/f443d3f4141b4163bbc130d46ebd7395/RESULTS.md` | Recorded numerical outcomes, grammar failure, first-action traces, archive limitations and costs. |

Final decision: **build the evidence-producing experiment first, then promote mechanisms on their measured causal contribution.** The strongest near-term opportunity is improving data/task coverage and invariance while preserving an honest distinction between learned behavior and engineered assistance. No new capability result is claimed by this document.
