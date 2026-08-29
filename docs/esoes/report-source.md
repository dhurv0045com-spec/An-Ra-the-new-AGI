# V5 research source and claim ledger

Internal research artifact for `V5_MASTER_BLUEPRINT.md`. The master blueprint is the canonical decision document; this file records the evidence audit behind it.

Date: 2026-08-29

## Scope and direct answer

Question: what foundation should An-Ra build to maximize useful cognition per parameter, token, and unit of compute?

Direct answer: do not scale V4 directly to 300M–3B. First build a roughly 195M dense, deeper decoder trained on about 4B auditable tokens, of which a provisional 15% are mechanically verified causal-contrast examples. Add at most one auxiliary objective: query-swap contrastive likelihood derived from the successful SFT6/normalization mechanism. Freeze that design only after a 35M screening campaign and a 102M replication demonstrate OOD transfer beyond synthetic templates.

## Claim/gap matrix

| Consequential claim | Evidence | Classification | Contradiction or limitation | Resolution |
|---|---|---|---|---|
| V4 was undertrained for its size | Chinchilla/DCLM use roughly 20 tokens/parameter as a compute-optimal reference; V4 has only 329.9M certified continuation tokens and uncertified lifetime totals | EVIDENCE-BACKED for token insufficiency; STRONG INFERENCE for exact V5 budget | Scaling laws optimize average loss, not cognition | Use 20 tokens/parameter as a floor, then evaluate cognition curves independently |
| Better data can beat merely more data | DCLM found model-based filtering decisive and small-scale data rankings correlated with larger-scale rankings | EVIDENCE-BACKED | DCLM evaluates broad downstream quality, not An-Ra cognition | Use small proxy models to choose data, but require An-Ra-specific gates |
| Targeted query contrasts are the highest-value native objective candidate | An-Ra SFT6 created query-conditioned signal; same-query SFT7 margin did not; normalization sometimes exposed hidden signal | EVIDENCE-BACKED inside An-Ra | Normalization failed to transfer robustly to the latest PGE battery | Test a query-swap objective against LM-only; do not assume success |
| A modestly deeper model may help composition | Parameter-matched depth research finds deeper models improve compositional generalization with diminishing returns | EVIDENCE-BACKED directionally | Exact optimum depends on task, latency, and training scale | Test three iso-parameter shapes; provisional 28×768 only |
| Full attention at 4k is the clean cognition baseline | Long-context studies show nominal windows do not imply reliable retrieval/composition; sparse attention adds another bottleneck | STRONG INFERENCE | No direct iso-FLOP An-Ra full-vs-hybrid result | Use full attention provisionally; compare hybrid only if throughput blocks training |
| GQA is acceptable but 2 KV heads are too aggressive to inherit blindly | GQA approaches MHA quality in published tasks while improving cache efficiency | EVIDENCE-BACKED generally | Fine-grained nonce binding was not the target; V4 used only 2 KV heads | Provisional 4 KV heads; run 4-KV vs MHA binding test |
| QK normalization should remain provisional | Small-scale proxy work and OLMo 2 support stability benefits | EVIDENCE-BACKED for stability | Its effect on candidate-score geometry is unknown | Run QK-norm on/off in the architecture screen |
| 24,576-token byte-fallback BPE is a sensible default, not a conclusion | Tokenizer choice affects performance/cost; compression alone is not predictive; byte models improve robustness but add architecture complexity | STRONG INFERENCE for the default | No An-Ra nonce/copy/code tournament exists | Compare 16,384/24,576/32,768 under raw-text and FLOP controls |
| Synthetic cognition should be bounded and verified | Phi shows high-quality synthetic data can be efficient; recursive indiscriminate model-generated data can collapse tails | EVIDENCE-BACKED | Programmatic causal data differs from free-form model imitation | Prefer executable generators; cap unverified LLM prose at 5% of total tokens |
| Curriculum order may help, but should not be adaptive by default | Skill-It finds prerequisite ordering can improve efficiency; other LM curriculum work finds no compelling general benefit | OPEN / MIXED | Results are domain- and metric-dependent | Test uniform interleaving vs staged difficulty; do not deploy online adaptation before evidence |
| WSD is operationally attractive | MiniCPM reports WSD supports continued training; An-Ra already has pack-bound WSD infrastructure | EVIDENCE-BACKED operationally | No evidence WSD is best for cognition; V4 behavior peaked before final | Keep WSD provisional and promote checkpoints by behavior, not final step |
| AdamW/BF16/clip-1 are the lowest-risk baseline | Chinchilla, OLMo and modern open reports converge on AdamW-style training and global clipping | EVIDENCE-BACKED as a stable baseline | Exact LR/batch/decay remain scale-specific | Freeze family now; sweep peak LR and tokens/update |
| Small proxies can reduce expensive mistakes | DCLM data rankings transfer across scale; instability work reproduces large-model pathologies in proxies | EVIDENCE-BACKED | Cognitive capability thresholds may not transfer smoothly | Require a 102M replication before the 195M main run |

## Primary sources consulted

- Hoffmann et al., 2022, [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556).
- Li et al., 2024, [DataComp-LM](https://arxiv.org/abs/2406.11794).
- Grattafiori et al., 2024, [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783).
- OLMo Team et al., 2024, [2 OLMo 2 Furious](https://arxiv.org/abs/2501.00656).
- Ainslie et al., 2023, [GQA](https://arxiv.org/abs/2305.13245).
- Su et al., 2021, [RoFormer / RoPE](https://arxiv.org/abs/2104.09864).
- Zhang and Sennrich, 2019, [RMSNorm](https://arxiv.org/abs/1910.07467).
- Shazeer, 2020, [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202).
- Press and Wolf, 2016, [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859).
- Hu et al., 2024, [MiniCPM and WSD](https://arxiv.org/abs/2404.06395).
- Wortsman et al., 2023, [Small-scale proxies for Transformer instabilities](https://arxiv.org/abs/2309.14322).
- Mueller et al., 2024, [The Impact of Depth on Compositional Generalization](https://arxiv.org/abs/2310.19956).
- Schmidt et al., 2024, [Tokenization Is More Than Compression](https://arxiv.org/abs/2402.18376).
- Pagnoni et al., 2024, [Byte Latent Transformer](https://arxiv.org/abs/2412.09871).
- Gunasekar et al., 2023, [Textbooks Are All You Need](https://arxiv.org/abs/2306.11644).
- Shumailov et al., 2024, [AI models collapse when trained on recursively generated data](https://doi.org/10.1038/s41586-024-07566-y).
- Liu et al., 2023, [Lost in the Middle](https://arxiv.org/abs/2307.03172).
- Hsieh et al., 2024, [RULER](https://arxiv.org/abs/2404.06654).
- Mirzadeh et al., 2024, [GSM-Symbolic](https://openreview.net/forum?id=AjXkRZIvjB).
- Chen et al., 2023, [Skill-It](https://arxiv.org/abs/2307.14430).
- Xie et al., 2023, [DoReMi](https://arxiv.org/abs/2305.10429).
- Kaplan et al., 2020, [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361).
- Dey et al., 2023, [Cerebras-GPT](https://arxiv.org/abs/2304.03208).

## Research stop condition

Discovery stopped when every major blueprint slot had either primary evidence, an An-Ra receipt, or an explicit experiment; remaining disagreements are empirical choices that another literature search cannot settle for this model and dataset. No private leaks, stolen corpora, or unverifiable architecture claims were used.
