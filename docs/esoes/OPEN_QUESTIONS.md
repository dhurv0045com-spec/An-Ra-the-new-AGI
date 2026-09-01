# ESOES High-Impact Open Questions

Ground Blueprint v0.4 intentionally limits the open set. A question belongs here only if its answer can materially change architecture, data, training, evaluation, or system boundaries.

`V5_TRAINING_SPEC_v1.0.md` supplies the implementation default while these
questions remain open. A winning experiment changes that default only through a
new spec version; uncertainty is no longer permission for code to guess.

## Representation and tokenizer

### Q1 — Which vocabulary gives the best cognition per raw byte and FLOP?

**WHY IT MATTERS:** 16k saves embeddings and may preserve atomic symbols; 32k shortens sequences; either can damage numbers, identifiers, and exact copying.
**CURRENT BEST HYPOTHESIS:** 24,576-entry byte-fallback subword model is the Pareto center.
**WHAT EVIDENCE EXISTS:** compression alone is not predictive; numerical and context/answer tokenization create measurable inductive biases.
**CHEAPEST DECISIVE TEST:** E1 static corpus audit plus matched P35 16k/24k/32k runs normalized by raw bytes and FLOPs.
**DECISION DEADLINE:** before any architecture finalist exceeds 200M training tokens.

### Q2 — Should V5 enforce special numerical atomization?

**WHY IT MATTERS:** number segmentation can dominate arithmetic behavior, but special tokens may reduce generality or inflate vocabulary.
**CURRENT BEST HYPOTHESIS:** preserve digits and separators without a novel numeric embedding in V5-A.
**WHAT EVIDENCE EXISTS:** public work shows direction and granularity effects; no An-Ra tokenizer experiment exists.
**CHEAPEST DECISIVE TEST:** E1 arithmetic, counting, copy, and unseen-length transfer under candidate tokenizers.
**DECISION DEADLINE:** tokenizer freeze.

## Architecture

### Q3 — Is 26×896 better than other ~250M shapes at equal parameters and training FLOPs?

**WHY IT MATTERS:** composition may need sequential depth, while representation may need width.
**CURRENT BEST HYPOTHESIS:** 26×896 balances sequential depth with stable width; 14Q/7KV avoids severe KV compression.
**WHAT EVIDENCE EXISTS:** directional depth benefit in controlled composition; no An-Ra iso-budget result.
**CHEAPEST DECISIVE TEST:** E2 P35 deep/mid/wide tournament, replicate top two.
**DECISION DEADLINE:** before M102.

### Q4 — Does 2:1 GQA or QK norm damage query-conditioned discrimination?

**WHY IT MATTERS:** throughput/stability gains may compress the score geometry V5 needs.
**CURRENT BEST HYPOTHESIS:** use the consistent 2:1 scale family: 6Q/3KV at P35, 10Q/5KV at M102, and 14Q/7KV at V5-A, with QK norm enabled. V4's 2 KV heads should not be inherited.
**WHAT EVIDENCE EXISTS:** general GQA and stability evidence, but no nonce-binding evidence.
**CHEAPEST DECISIVE TEST:** E2 factorial 2:1 GQA comparison against MHA and 2:1/no-QK variants on identical data; earlier 3:1 kernel evidence is not a V5 topology result.
**DECISION DEADLINE:** architecture freeze.

### Q5 — Is native 4k full attention worth its training cost?

**WHY IT MATTERS:** it removes locality confounds but may buy fewer high-quality tokens per TPU-hour.
**CURRENT BEST HYPOTHESIS:** yes with mixed sequence lengths; not every sequence should be 4k.
**WHAT EVIDENCE EXISTS:** V4 hybrid failures are confounded; public work shows nominal context is not usable context.
**CHEAPEST DECISIVE TEST:** E2 matched-FLOP 2k/full versus 4k/mixed-length, evaluated at adversarial positions.
**DECISION DEADLINE:** architecture and pack-format freeze.

## Cognition and data

### Q6 — What share of verified cognition data improves primitives without narrowing the substrate?

**WHY IT MATTERS:** too little provides no causal pressure; too much creates a synthetic specialist.
**CURRENT BEST HYPOTHESIS:** 15% by tokens.
**WHAT EVIDENCE EXISTS:** targeted An-Ra SFT works but can starve formats; natural transfer remains limited.
**CHEAPEST DECISIVE TEST:** E3 5/15/30% at matched tokens, including natural analogues and substrate loss.
**DECISION DEADLINE:** before M102.

### Q7 — Do intermediate traces teach composition or merely a serialization format?

**WHY IT MATTERS:** traces can add recurrent compute but can leak solution structure and hurt direct realization.
**CURRENT BEST HYPOTHESIS:** sparse, mechanically verified state traces help two/three-hop transfer when mixed with direct answers.
**WHAT EVIDENCE EXISTS:** controlled public synthetic evidence supports intermediate outputs; An-Ra has no transfer result.
**CHEAPEST DECISIVE TEST:** one E3 sub-arm comparing direct-only with 25% trace exposure, tested trace-free.
**DECISION DEADLINE:** data freeze.

### Q8 — Does the cognitive recipe transfer to natural domains?

**WHY IT MATTERS:** synthetic-only success does not satisfy the program.
**CURRENT BEST HYPOTHESIS:** binding/state gains transfer more readily than multi-hop composition.
**WHAT EVIDENCE EXISTS:** SFT6 cross-vocabulary lift; weak broader transfer; public counterfactual fragility.
**CHEAPEST DECISIVE TEST:** fresh natural-document analogues whose sources, style, and entities are absent from generator development.
**DECISION DEADLINE:** E3 promotion and again E5.

## Objective and optimization

### Q9 — Is structured CE sufficient, or does query-swap contrast add causal transfer?

**WHY IT MATTERS:** the auxiliary may directly teach query addressing or merely game candidate likelihoods.
**CURRENT BEST HYPOTHESIS:** small query-swap weight improves selection; CE remains dominant.
**WHAT EVIDENCE EXISTS:** SFT6 query lift; SFT7 margin failed its intended selection hypothesis; no foundation-scale ablation.
**CHEAPEST DECISIVE TEST:** E3 CE-only versus identical data with λ 0.05/0.15, including candidate-free generation.
**DECISION DEADLINE:** before M102.

### Q10 — Uniform interleaving or competence-staged curriculum?

**WHY IT MATTERS:** prerequisites may accelerate learning; stages may induce forgetting or recognizable progression shortcuts.
**CURRENT BEST HYPOTHESIS:** uniform is the default; staging wins only if it improves worst-family OOD with fixed replay.
**WHAT EVIDENCE EXISTS:** mixed public curriculum results and An-Ra starvation/forgetting.
**CHEAPEST DECISIVE TEST:** E4 one uniform and one staged/replay arm.
**DECISION DEADLINE:** before M102.

### Q11 — Which batch/LR pair gives enough updates without sacrificing throughput?

**WHY IT MATTERS:** 262k tokens/update produces only ~19.1k updates over 5B tokens; 131k doubles update count.
**CURRENT BEST HYPOTHESIS:** 131k tokens/update with peak LR near 3e-4.
**WHAT EVIDENCE EXISTS:** conventional optimizer reports; no V5 gradient-noise measurement.
**CHEAPEST DECISIVE TEST:** E4 short LR-range test at 131k/262k, then extend only the stable Pareto arms.
**DECISION DEADLINE:** M102 recipe freeze.

## Scale and evaluation

### Q12 — Does the winning P35 recipe survive at ~102M?

**WHY IT MATTERS:** small-model rankings may fail at a capability threshold.
**CURRENT BEST HYPOTHESIS:** data/tokenizer ranking transfers; absolute composition ability may not.
**WHAT EVIDENCE EXISTS:** DataComp-LM proxy correlations; no An-Ra scale transfer.
**CHEAPEST DECISIVE TEST:** E5 matched M102 recipe and CE/general-data control, two winner seeds.
**DECISION DEADLINE:** before any 250M implementation freeze.

### Q13 — Is ~250M capacity-limited after 5B high-quality tokens?

**WHY IT MATTERS:** this is the only valid reason to reopen ~400M+.
**CURRENT BEST HYPOTHESIS:** two-hop improves; robust natural three-hop may remain capacity- or computation-limited.
**WHAT EVIDENCE EXISTS:** V4 is too confounded; public composition work shows OOD limits.
**CHEAPEST DECISIVE TEST:** scaling curves across P35/M102/V5-A on matched primitive families and loss.
**DECISION DEADLINE:** after V5-A evaluation, not before.

### Q14 — Which benchmark thresholds predict useful downstream cognition?

**WHY IT MATTERS:** arbitrary gates can reject good recipes or bless toy specialists.
**CURRENT BEST HYPOTHESIS:** worst-family confidence bounds plus fresh natural transfer are more predictive than aggregates.
**WHAT EVIDENCE EXISTS:** An-Ra loss/behavior divergence and consumed-fixture failures.
**CHEAPEST DECISIVE TEST:** E0 calibration with known trivial, heuristic, oracle, and deliberately broken systems; preregister thresholds before training.
**DECISION DEADLINE:** before E1.

**V0.4 UPDATE:** a second audit overturned the first repair's false green: bag-of-words and lexical overlap still solved 81.77% and 71.09% of pooled state cases. Generator 0.4.0 now uses between/after-event cutoffs and eight competing target histories; the same controls fall to 4.17% and 10.16% against a 10.37% null, and six state heuristics are fail-closed gates. This earns shortcut-resistant development infrastructure only. Full E0 exit still requires a certified model-scoring adapter, source-disjoint natural custody, an externally held sealed commitment/result, and fresh replication.

**SCORER UPDATE:** deterministic scorer plumbing and exact local P35 CPU/CUDA parity now pass, but no aggregation mode is authorized. Across 486 paired scores there were zero prediction mismatches (relative RMS error 1.286e-7), while the real-tokenizer random-weight null still selected fewest-token candidates 100% under sum, 83.33% under byte normalization, and 50%--66.67% under token normalization. The next decisive work is a preregistered policy that removes or explicitly models this null bias, followed by target-TPU parity; trained-model results must not choose the policy retrospectively.

## Core/Connector boundary

### Q15 — Which EXP repairs should become native training signals?

**WHY IT MATTERS:** permanent runtime repair can mask weak Core, but over-internalization bakes deployment policy into weights.
**CURRENT BEST HYPOTHESIS:** query-conditioned discrimination and faithful realization belong in Core; intervention selection and verification remain Connector/evaluator responsibilities.
**WHAT EVIDENCE EXISTS:** normalization, constrained decode, and replicated observed-policy routing.
**CHEAPEST DECISIVE TEST:** after E3/E5, measure whether raw repair frequency falls on fresh domains while assisted oracle advantage remains reportable.
**DECISION DEADLINE:** post-E5 freeze review.

## Training-system boundary

### Q16 — Does the atomic state contract survive the real distributed target stack?

**WHY IT MATTERS:** local model resume, cursor resume, and transaction semantics can each pass while their integrated distributed checkpoint loses tokens, rank RNG, optimizer shards, or the committed parent.
**CURRENT BEST HYPOTHESIS:** one coordinator should publish a content-addressed generation after a collective completed-update barrier, with exact partial-final-update semantics and an object-store compare-and-swap pointer.
**WHAT EVIDENCE EXISTS:** the framework-neutral local canary passes identity binding, exact `4+4+2` accounting, clean-copy resume, corruption/missing inventory rejection, writer fencing, and injected crash boundaries. The exact middle-P35 canary now joins model/AdamW/scheduler/RNG/cursor/ledger state through that transaction and reproduces the next update with zero error; the local immutable CAS canary passes upload/redownload equality and corruption rejection; `v5_training.distributed` rejects incomplete/duplicate/misaligned ranks, mismatched collective barriers, shard reuse, and token-ledger drift.
**CHEAPEST DECISIVE TEST:** repeat the joined canary on the chosen TPU topology with real per-rank RNG/shard payloads and object-store upload→redownload→clean restore; local schemas cannot substitute for target collective and durability evidence.
**DECISION DEADLINE:** before M102 and target training freeze.

## Deferred—not active search

- Recurrent/SSM/Titans memory architecture: reopen only for a measured dense long-context bottleneck.
- MoE: reopen only when dense data/compute scaling is established.
- Byte-latent architecture: independent later program, not a tokenizer arm.
- Native context beyond 4k: reopen only when tasks and data require it.
- 400M/1B/3B: reopen only from measured scaling and adequate clean-token inventory.
