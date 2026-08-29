# ESOES Design Iterations

Exactly four major design iterations produced Ground Blueprint v0.1. This document closes open-ended architecture search after Iteration 4; remaining uncertainty is routed to experiments or deferred.

## Iteration 1 — Foundation reconstruction

### Starting assumptions attacked

- A better language model will automatically become a better cognitive substrate.
- “Reasoning” is one capability.
- Everything useful discovered by EXP should be trained into Core.
- Architectural memory is required before a Transformer can bind variables.

### First-principles cognition contract

A useful neural Core must repeatedly perform five operations inside a bounded context:

1. **Represent:** encode entities, values, roles, relations, order, and current state without collapsing alternatives.
2. **Address:** allow the current query to select which represented facts or states matter.
3. **Transform:** update state and compose a small number of relations/rules.
4. **Choose:** make the correct latent candidate dominate plausible alternatives, including under counterfactual queries.
5. **Realize:** emit the chosen content faithfully in the required surface form.

These operations must generalize to unseen symbols, surface forms, graph structures, and context positions. Knowledge breadth and fluent prose are substrate requirements, not substitutes for the operations.

### Primitive classification

| Ability | Owner | Reason |
|---|---|---|
| local copy/retrieval | Core foundation | required by every higher operation |
| entity-role/value binding | Core foundation | repeated query-addressing primitive |
| state update and overwrite | Core foundation | local mutable-context semantics |
| counterfactual/query sensitivity | Core foundation | distinguishes causal use from priors |
| short relational composition | Core foundation | reusable local transformation |
| uncertainty from missing evidence | Core foundation + evaluator | Core represents lack; evaluator verifies calibration |
| long search/planning | Connector | variable horizon, auditable branching, cost control |
| tool/API execution | Runtime | environment-specific and deterministic where possible |
| durable memory/retrieval | Connector/runtime | changes outside weights and needs provenance |
| verification/credit assignment | Evaluator/Connector | must stay structurally independent of generation |
| permissions/risk policy | Outer layer | deployment-specific authority |

### Research and alternatives considered

- Mechanistic variable-binding work shows a standard Transformer can learn residual-stream addressable memory and attention routing without explicit memory modules.
- EXP shows runtime normalization and observed-state routing are valuable, but assisted success remains distinct from native cognition.
- Explicit symbolic modules would improve narrow reliability but would change the scientific question from learned neural primitives to a hybrid executor.

### Attack against the result

“Represent/address/transform/choose/realize” could merely rename common failures. To make it scientific, every level receives an independent observable: candidate NLL/lift, query-conditioned rank, controlled transformation accuracy, free emission, and intervention delta. If a level cannot be measured without the answer leaking into Core, it is not yet a valid benchmark.

### Decisions changed

- Replaced task-list cognition with a five-operation contract.
- Added the Outer layer explicitly; not all non-Core behavior belongs in Connector.
- Rejected native long-term memory and self-diagnosis as baseline requirements.

### Decisions unchanged

- Dense autoregressive modeling remains the simplest universal substrate.
- Representation, selection, and realization remain separate; “transform” and “address” sharpen the causal middle.

### Confidence change

- Core/Connector/Outer boundary: 0.65 → 0.82.
- Need for explicit memory in V5-A: 0.35 → 0.15.

### Unresolved

- Can ~200M parameters acquire robust two/three-hop transformation on fresh natural forms?
- Does missing-information behavior emerge from negative evidence or require post-training calibration?

### Resulting blueprint

**Ground candidate 0.1-I1:** a clean dense Core optimized and evaluated for represent → address → transform → choose → realize; external control remains external.

### Next iteration target

Attack the representation machinery and parameter allocation.

---

## Iteration 2 — Architecture and representation attack

### Starting assumptions attacked

- V4's 18×896 shape is approximately optimal.
- A 300M+ jump is needed for cognition.
- Full attention is automatically wasteful.
- 32k vocabulary and 2 KV heads are harmless inherited defaults.
- Recurrent/state-space models are automatically more cognitive because they carry state.

### Alternatives considered

1. Dense global-attention Transformer.
2. V4-like hybrid local/full Transformer.
3. Mamba-style selective SSM.
4. Griffin-style recurrence plus local attention.
5. Titans-style test-time neural memory.
6. MoE Transformer.
7. Explicit key/value memory module.
8. Byte-level/BLT representation.

### Architecture attacks

**Recurrent and SSM alternatives.** They offer compelling long-context efficiency, but V5-A targets 4k rather than million-token context. Their state compression can become a binding bottleneck, their kernels/toolchains expand implementation risk, and no An-Ra-scale evidence shows superior query-addressed binding per TPU-hour. They remain a future architecture-validation arm, not the baseline.

**MoE.** It buys parameters without proportional FLOPs, but routing stability, expert imbalance, checkpoint size, and attribution burden are unjustified before a dense baseline succeeds.

**Explicit memory.** It may hide a weak contextual Core and shifts verification to memory semantics. It belongs in Connector until a controlled dense-vs-memory experiment at longer context has decision value.

**Full attention.** At 4k, its quadratic component is real but manageable for a ~195M model; it removes the sliding-window bottleneck from binding experiments. Sequence-length mixing controls cost.

**Depth.** More layers provide additional sequential transformations, but extreme narrowness can starve feature width and throughput. The candidate moves from V4's 18×896 to 28×768, subject to an iso-parameter tournament.

### Representation attack

Vocabulary cannot be selected by compression alone. Smaller vocabularies save embeddings and improve atomicity but inflate sequences; large vocabularies can fuse identifiers, numbers, and context answers inconsistently. A 24,576 byte-fallback vocabulary is the center candidate, with explicit 16k/32k challengers. Number segmentation and exact context/answer token consistency are benchmark dimensions.

### Provisional architecture family

| Role | Shape | Approx. parameters | Purpose |
|---|---|---:|---|
| Micro | 12×256, FFN 704, 4Q/2KV | 15–20M | pipeline and generator sanity, not scaling claims |
| P35 | 16×384, FFN 1024, 6Q/2KV | 34.6M | tokenizer/data/objective ranking |
| M102 | 20×640, FFN 1728, 10Q/2KV | 101.8M | scale-transfer replication |
| V5-A | 28×768, FFN 2048, 12Q/4KV | 195.08M | first serious run after gates |

Common candidate: dense causal decoder, pre-RMSNorm, RoPE, SwiGLU, tied embeddings, bias-free, BF16, full attention. QK norm, GQA/MHA, exact shape, vocabulary, and context economics remain experiment-gated.

### Attack against the result

The 195M choice could be conservative anchoring on V4. The defense is not that 195M is intrinsically optimal: it is the largest candidate naturally supported by a 4B-token auditable corpus near the public 20-token/parameter reference. If 6B+ equally good tokens become available or M102 shows a capacity wall, reopen scale before training. Otherwise parameter growth spends compute while leaving data and causality unresolved.

### Decisions changed

- Rejected immediate 300M–3B.
- Reduced center vocabulary from 32k to provisional 24k.
- Increased provisional depth, KV heads, native context, and global attention.
- Added a micro model solely for pipeline validation; it cannot decide emergent cognition.

### Decisions unchanged

- Dense decoder baseline.
- Tied embeddings, RMSNorm, RoPE, SwiGLU, and no architecture soup.

### Confidence change

- Dense Transformer V5-A baseline: 0.72 → 0.84.
- Exact 28×768 shape: 0.45 → 0.50.
- 24,576 vocabulary: 0.42 → 0.45.
- Need for full 4k attention: 0.55 → 0.64.

### Unresolved

- Depth/width, QK norm, GQA versus MHA, 2k versus 4k economics, and tokenizer.

### Resulting blueprint

**Ground candidate 0.1-I2:** an experimentally selected member of a dense family, centered on 195M/28×768 with full native 4k attention and 24k byte fallback.

### Next iteration target

Design experiences that cause the five cognitive operations to emerge.

---

## Iteration 3 — Data and learning dynamics

### Starting assumptions attacked

- High-quality web text alone will produce the target operations.
- Synthetic cognition should occupy 20–35% because it is controllable.
- Pure next-token CE is necessarily sufficient—or necessarily insufficient.
- A staged curriculum is inherently better than interleaving.
- 4B tokens and 262k tokens/update are obviously correct.

### Learning mechanism

Natural text supplies language, knowledge, styles, and implicit procedures. Mechanically generated causal families supply controlled interventions that natural corpora rarely balance. Each cognitive datum should include matched variants:

- same facts, different query;
- same query, changed relevant fact;
- same query, changed irrelevant fact;
- permuted order with unchanged semantics;
- state overwritten versus not overwritten;
- distractor inserted or removed;
- one-hop versus matched two/three-hop graph;
- enough information versus a minimal missing edge;
- direct answer versus intermediate trace.

The generator's latent graph belongs only to the evaluator/objective builder. No task label, answer index, difficulty token, or hidden cause enters the model input.

### Data design

Center candidate remains 4.0B frozen-tokenizer tokens:

- 65% high-quality natural substrate;
- 20% code, math, formal, and structured natural data;
- 15% mechanically verified cognitive contrasts.

The cognition share is bounded to 5/15/30% until E3. Unverified LLM paraphrases are capped at 5% total, provenance-tagged, and accepted only when a solver verifies semantic equivalence.

### Objective design

CE remains universal. Query-swap contrast is the only auxiliary candidate because it is causally aligned with the strongest internal evidence. Same-query margin is not admitted as the primary selection objective.

For fixed facts `F`, factual query `q`, counterfactual query `q'`, and candidate `v`:

```text
lift(v,q) = log P(v | F,q) - mean_q' log P(v | F,q')
```

The auxiliary compares gold lift against plausible same-context competitors. It is applied only to verified examples and must beat CE-only on fresh query/domain transfer without damaging substrate loss beyond the preregistered bound.

### Curriculum and sequence length

Compare uniform interleaving against one staged plan with replay. Staging may advance copy/binding/state before composition and raise cardinality/distractors only after competence, but at least 30% of cognition sampling replays earlier families.

Use a length mixture rather than padding all data to 4k:

- 50% packed sequences around 512–1,024 tokens;
- 30% around 2,048 tokens;
- 20% at 4,096 tokens, enriched for position/distractor/state tests.

Ratios are provisional and counted by tokens, not sequences. Position randomization prevents “answer near the end” shortcuts.

### Optimization attack

At 4B tokens, 262,144 tokens/update yields only about 15,259 updates. A 131,072-token update yields about 30,518 updates and may give better optimization/curriculum resolution at modest extra synchronization. Therefore 131k becomes the center candidate; 262k remains the throughput challenger. Batch and LR are paired decisions.

Center optimizer candidate: AdamW, betas 0.9/0.95, epsilon 1e-8, weight decay 0.1 on weight matrices, BF16 compute with FP32 reductions/state, global clip 1.0, WSD schedule. Peak LR 3e-4 is provisional and tested with 2e-4/4e-4.

### Attack against the result

Synthetic benchmarks can be learned by templates, intermediate traces can teach serialization rather than reasoning, and a contrastive loss can distort calibration. The design therefore requires graph/template/vocabulary/domain disjointness, direct-retrieval controls, unexposed natural analogues, CE-only controls, and representation/selection/realization reporting. A gain that exists only on generator siblings is rejected.

### Decisions changed

- Lowered center tokens/update from 262k to 131k pending E4.
- Added explicit length mixture and position balancing.
- Limited LLM-generated paraphrase more strictly than programmatic synthetic data.
- Added intermediate-state exposure as an experiment, not a universal training format.

### Decisions unchanged

- 4B center token budget.
- 65/20/15 center mixture.
- CE base plus one query-swap candidate.
- Programmatic verification and provenance are mandatory.

### Confidence change

- Need for causal contrast families: 0.78 → 0.90.
- Exact 15% cognition share: 0.43 → 0.46.
- Pure CE sufficient: 0.50 → 0.40.
- Query-swap auxiliary useful: 0.55 → 0.62.

### Unresolved

- Natural transfer, exact mixture, CE versus auxiliary, staged versus uniform, batch/LR, and whether intermediate traces help OOD composition.

### Resulting blueprint

**Ground candidate 0.1-I3:** 4B provenance-complete tokens with controlled causal families, CE baseline, one query-swap challenger, length mixing, replay, and behavior-aware optimization.

### Next iteration target

Try to destroy the integrated program and close the search.

---

## Iteration 4 — Red team and system integration

### Proposal under attack

A ~195M dense 28×768 decoder with full 4k attention and a 24k byte-fallback tokenizer, trained for 4B tokens on a 65/20/15 mixture using CE plus optional query-swap contrast, promoted by a causal OOD benchmark.

### Failure attacks and disposition

| Attack | Consequence | Disposition |
|---|---|---|
| 195M is too small for composition | main run could be predictably capped | M102 scale-transfer curves and V5-A two-hop gates; do not claim arbitrary-depth reasoning |
| 195M is too large for available data | wasted compute/overfit | exact 4B unique-token manifest required; shrink model if corpus fails |
| synthetic templates masquerade as cognition | false success | disjoint generators, topologies, alphabets, styles, natural analogues, direct-retrieval controls |
| query objective games candidate set | good rank, bad generation | candidate-free natural tests and conditional realization gate |
| full 4k attention wastes compute | fewer useful tokens | length mixture; E2 2k/4k throughput-quality accounting |
| 24k tokenizer damages numbers/code | representation pathology | E1 number direction, identifier fragmentation, code/math, and context-answer consistency |
| deep/narrow starves capacity | weak representations | iso-parameter deep/mid/wide P35 tournament, finalists replicated |
| QK norm/GQA blunt sharp selection | candidate geometry harmed | E2 factorial screen versus MHA and no-QK controls |
| curriculum forgets earlier primitives | specialization | fixed replay floor, per-family curves, worst-family abort |
| natural substrate degrades | narrow toy model | CE-only matched control and ≤3% substrate-loss gate |
| final checkpoint regresses | repeated V4 mistake | immutable behavioral milestones; final never auto-promoted |
| resume/checkpoint fails | disposable TPU work lost | exact-resume equivalence, remote restore canary, two recovery generations |
| optimizer does not update intended weights | fake training | live parameter IDs, parameter SHA, Adam step/moments, cursor/tokens verified |
| benchmark leaks into data/filtering | invalid science | sealed fixture isolation, hashes, successor suite after any exposure |
| Connector hides Core weakness | assisted score misclaimed | raw and assisted leaderboards, Core gates use raw scores |
| compute estimate is fantasy | run cannot finish | measured micro/M102 throughput before authorization; device-neutral budget |
| architecture alternatives were dismissed too early | local optimum | record Mamba/Griffin/Titans/BLT as deferred; reopen only if 4k dense baseline hits measured bottleneck |

### Integration decisions

- Freeze the cognition contract, evidence taxonomy, layer boundary, dense-baseline family, causal evaluation contract, provenance rules, and six-experiment order.
- Keep exact tokenizer, model shape, QK/GQA, objective weight, mixture, curriculum, batch/LR, and schedule provisional behind named experiments.
- Defer recurrence, neural memory, MoE, byte-latent architecture, long context beyond 4k, and 300M+ scale.
- Do not implement a production trainer before the benchmark and architecture/data experiments provide the missing decisions.

### Confidence after red team

- Program direction: 0.76.
- Dense baseline family: 0.84.
- 195M/4B center: 0.66.
- Exact 28×768/24k/15% recipe: 0.46.
- Query-swap auxiliary transfer: 0.55.
- Robust natural three-hop composition at this scale: 0.25.

### Ground Blueprint v0.1

The integrated specification is `V5_MASTER_BLUEPRINT.md`. It is strong enough to define experiments and software boundaries, but not to authorize production implementation or main training.

### Search closure

No Iteration 5 is authorized. Remaining issues are categorized as:

- **A — frozen:** cognition contract, layer boundary, dense baseline, evidence/provenance/evaluation rules.
- **B — cheap experiment required:** tokenizer, shape, attention details, data fraction/objective, curriculum/optimization.
- **C — deferred:** recurrent/SSM/memory/MoE/BLT, >4k native context, >300M scale.
- **D — low-impact unknown:** cosmetic divisibility and minor implementation choices resolved after scientific decisions.

### Next phase target

Use the post-Iteration-4 blueprints to implement E0 only. E0 must certify the benchmark/generator contract before any training infrastructure is built.
