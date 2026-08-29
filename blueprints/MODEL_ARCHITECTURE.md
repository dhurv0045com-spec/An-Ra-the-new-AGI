# Ground Blueprint v0.4 — Model Architecture

This is an engineering specification, not a neural-model implementation. Bracketed states are governed by `docs/esoes/DECISIONS.md`; `v5_contracts/model_spec.py` is the executable arithmetic/configuration authority.

## Cognitive information flow

```mermaid
flowchart TB
    raw[Raw UTF-8 text] --> tok[Identity-preserving byte-fallback tokenizer<br/>24,576 center candidate]
    tok --> emb[Tied token embedding<br/>V × 896]
    emb --> rope[RoPE positions<br/>native 4,096]
    rope --> stack[26 × dense causal processing blocks]
    stack --> norm[Final RMSNorm]
    norm --> head[Tied vocabulary projection]
    head --> ce[Causal next-token CE]
    head --> qs[Verified query-swap contrast<br/>experimental examples only]

    subgraph block[One candidate block]
      x[Residual stream 896] --> n1[RMSNorm]
      n1 --> qkv[14 Q / 7 KV × 64<br/>QK norm candidate]
      qkv --> attn[Full causal attention]
      attn --> add1[Scaled residual add]
      add1 --> n2[RMSNorm]
      n2 --> ffn[SwiGLU FFN 2,368]
      ffn --> add2[Scaled residual add]
    end
```

The architecture contains no explicit task labels, symbolic slots, state registers, memory modules, routers, or cognition heads. Learned attention/MLP circuits must earn query-conditioned binding and transformation under controlled training.

## Candidate V5-A configuration

| Field | Value | State |
|---|---:|---|
| family | dense decoder-only Transformer | [FROZEN] baseline |
| vocabulary | 24,576 | [EXPERIMENT REQUIRED: E1] |
| width | 896 | [EXPERIMENT REQUIRED: E2] |
| layers | 26 | [EXPERIMENT REQUIRED: E2] |
| query heads | 14 | [PROVISIONAL] |
| KV heads | 7 | [EXPERIMENT REQUIRED: E2] |
| head dimension | 64 | [PROVISIONAL] |
| FFN | 2,368 SwiGLU | [EXPERIMENT REQUIRED: E2] |
| attention | full causal, every layer | [EXPERIMENT REQUIRED: E2] |
| context | 4,096 native | [EXPERIMENT REQUIRED: E2] |
| position | RoPE, base 10,000; no extrapolation claim | [PROVISIONAL] |
| norm | pre-RMSNorm, epsilon 1e-5 | [PROVISIONAL] |
| QK norm | on candidate, affine Q/K scales initialized to 1 | [LOCAL SCALE-CONTROL EVIDENCE; LEARNING EXPERIMENT REQUIRED] |
| embeddings/head | tied | [PROVISIONAL] |
| linear bias | none | [PROVISIONAL] |
| dropout | zero | [PROVISIONAL] |
| initialization | normal 0.02; attention-output and FFN-down weights scaled by `1/sqrt(2L)` | [LOCAL SIGNAL EVIDENCE; TARGET CANARY REQUIRED] |
| compute precision | BF16 with FP32 reductions/optimizer | [PROVISIONAL] |

## Parameter receipt

Assuming bias-free projections, GQA keys/values of width 448, two per-block RMSNorm vectors, tied embeddings, and one final RMSNorm:

```text
embedding        24,576 × 896                           = 22,020,096
Q projection     896 × 896                              =    802,816
K + V            2 × 896 × 448                          =    802,816
O projection     896 × 896                              =    802,816
SwiGLU           3 × 896 × 2,368                        =  6,365,184
block norms      2 × 896                                =      1,792
QK norm scales   896 + 448                              =      1,344
per block                                                =  8,776,768
26 blocks        26 × 8,776,768                         =228,195,968
final norm                                               =        896
total                                                    =250,216,960
```

The future executable constructor must independently reproduce this count. If E1/E2 changes vocabulary or shape, this receipt becomes historical and `DECISIONS.md` must be reopened.

## Scale family

| Model | Center shape | Approx. parameters | Scientific authority |
|---|---|---:|---|
| Micro | 12×256, FFN 704 | 15–20M | pipeline/generator canary only |
| P35 | 16×384, FFN 1024 | 34.6M | rank experimental choices |
| M102 | 20×640, FFN 1728 | 101.8M | replicate recipe/scale transfer |
| V5-A | 26×896, FFN 2368 | 250.22M | first serious run after freeze |
| Later | unselected | open | only measured capacity/data evidence |

P35 results cannot prove an emergent capability. M102 must reproduce the winning direction before V5-A.

### Initialization canary

An exact-stack, paired-draw probe compared unscaled `normal(0, 0.02)` with scaling only the attention-output and FFN-down matrices by `1/sqrt(2L)`. Five randomized CUDA seeds at sequence 256 and three CPU seeds at sequence 64 used every P35 block, cross-entropy, and one backward pass without an optimizer update. On short-context CUDA, scaled/unscaled final-RMS-growth ratios were 0.122 / 0.156 / 0.230 for deep/middle/wide, while gradient-spread ratios were 0.604 / 0.650 / 0.731. CPU reproduced the direction at 0.125 / 0.160 / 0.240 and 0.647 / 0.776 / 0.820. A separate three-seed CUDA replication at the intended 4,096-token context strengthens the result: growth ratios are 0.115 / 0.144 / 0.217 and gradient-spread ratios are 0.542 / 0.510 / 0.654. All parameter, hook, finite, and nonzero-gradient checks passed. This supports the scaled policy as the implementation default; it does not select depth, LR, optimizer, or cognition quality. The target TPU/XLA constructor must still repeat the canary and E4 must test real-update stability before freeze.

### QK-normalization canary

A paired MHA probe holds normalized hidden states, projection draws, values, targets, RoPE, and causal query positions fixed while multiplying Q/K projection weights by 0.25×, 1×, or 4×. Five CUDA seeds cover 512/2,048/4,096 context and three CPU seeds cover 128/512. At 4k, QK norm holds attention-logit RMS within a 1.000045 max/min ratio and normalized entropy within 0.000004 across the scale stress; without QK norm, logit RMS changes exactly 256× and normalized entropy spans 0.365. At the base scale, QK norm produces logit RMS 0.998 and effective attended fraction 0.616, versus 0.155 and 0.988 without it. This proves that the implementation controls attention-distribution sensitivity to Q/K norm drift and avoids an almost-uniform initialization on this probe. It does not prove that selectivity improves learning or cognition. Parameter-gradient RMS changes inversely with projection scale even under QK norm, so optimizer dynamics are not scale-invariant; E2 must retain the learned-quality ablation and target TPU/XLA canary.

## Rejected from V5-A baseline

- Mamba/SSM, Griffin/recurrent blocks, Titans/test-time memory;
- MoE or learned routing;
- explicit key/value or differentiable long-term memory;
- multi-token prediction, latent-thought heads, or separate realization head;
- untied output head without an ablation;
- local/sliding attention inherited from V4;
- learned task, family, difficulty, answer-index, or latent-graph tokens.

These are deferred, not declared impossible. They reopen only when a measured bottleneck gives the experiment decision value.

## Cognitive measurements mapped to model outputs

| Operation | Observable |
|---|---|
| represent | gold/distractor candidate NLL, rank, layerwise probe used diagnostically |
| address | likelihood change under fact-fixed query swaps |
| transform | state-swap and matched multi-hop versus retrieval-control accuracy |
| choose | raw candidate argmax, margin, query-flip direction |
| realize | free exact/semantic output, conditional on known correct selection |

Probes do not become training labels by default. A linear probe detecting information does not prove the model uses it.
