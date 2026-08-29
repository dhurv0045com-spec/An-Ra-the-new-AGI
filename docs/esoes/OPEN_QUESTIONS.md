# ESOES High-Impact Open Questions

Ground Blueprint v0.1 intentionally limits the open set. A question belongs here only if its answer can materially change architecture, data, training, evaluation, or system boundaries.

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

### Q3 — Is 28×768 better than wider/shallower shapes at equal parameters and training FLOPs?

**WHY IT MATTERS:** composition may need sequential depth, while representation may need width.
**CURRENT BEST HYPOTHESIS:** moderately deep 28×768 is better than V4-like 18×896, but extreme depth is not.
**WHAT EVIDENCE EXISTS:** directional depth benefit in controlled composition; no An-Ra iso-budget result.
**CHEAPEST DECISIVE TEST:** E2 P35 deep/mid/wide tournament, replicate top two.
**DECISION DEADLINE:** before M102.

### Q4 — Does 4-KV GQA or QK norm damage query-conditioned discrimination?

**WHY IT MATTERS:** throughput/stability gains may compress the score geometry V5 needs.
**CURRENT BEST HYPOTHESIS:** 4-KV GQA with QK norm is adequate; V4's 2 KV heads should not be inherited.
**WHAT EVIDENCE EXISTS:** general GQA and stability evidence, but no nonce-binding evidence.
**CHEAPEST DECISIVE TEST:** E2 factorial comparison against MHA and no-QK variants on identical data.
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

**WHY IT MATTERS:** 262k tokens/update produces only ~15.3k updates over 4B tokens; 131k doubles update count.
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
**DECISION DEADLINE:** before any 195M implementation freeze.

### Q13 — Is ~195M capacity-limited after 4B high-quality tokens?

**WHY IT MATTERS:** this is the only valid reason to reopen 300M+.
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

## Core/Connector boundary

### Q15 — Which EXP repairs should become native training signals?

**WHY IT MATTERS:** permanent runtime repair can mask weak Core, but over-internalization bakes deployment policy into weights.
**CURRENT BEST HYPOTHESIS:** query-conditioned discrimination and faithful realization belong in Core; intervention selection and verification remain Connector/evaluator responsibilities.
**WHAT EVIDENCE EXISTS:** normalization, constrained decode, and replicated observed-policy routing.
**CHEAPEST DECISIVE TEST:** after E3/E5, measure whether raw repair frequency falls on fresh domains while assisted oracle advantage remains reportable.
**DECISION DEADLINE:** post-E5 freeze review.

## Deferred—not active search

- Recurrent/SSM/Titans memory architecture: reopen only for a measured dense long-context bottleneck.
- MoE: reopen only when dense data/compute scaling is established.
- Byte-latent architecture: independent later program, not a tokenizer arm.
- Native context beyond 4k: reopen only when tasks and data require it.
- 300M/1B/3B: reopen only from measured scaling and adequate clean-token inventory.
