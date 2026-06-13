# An-Ra Best Research for Intelligence and Efficiency

Date: 2026-06-08

**Research context updated 2026-06-13:** this is a candidate-method map. A technique becomes part of AN-RA only through the standard candidate, evidence, promotion, and rollback lifecycle. The current canonical owners are listed in [`../ARCHITECTURE.md`](../ARCHITECTURE.md).

Three labels should be used when reading the recommendations:

- **Candidate:** technically plausible and worth an experiment.
- **Measured:** tested against a named baseline with artifacts.
- **Promoted:** passed protected capability, identity, safety, owner, and deployment gates.

Purpose: identify the best research and technologies for making An-Ra more intelligent, faster, cheaper to train, and more efficient while preserving the original An-Ra vision: sovereign, owner-shaped, memory-rich, verifier-grounded intelligence.

This is not a hype list. It is a fit analysis. A method is valuable only if it increases An-Ra's ability to run more verified learning loops, remember more faithfully, reason with fewer wasted tokens, train under constrained hardware, or preserve identity while improving.

## Vision Lock

An-Ra should not become a generic benchmark-chasing chatbot. Every recommendation must preserve these constraints:

- Owner-shaped data remains the center of gravity.
- CIV, ESV, HAL, and sovereignty gates protect identity and drift boundaries.
- Symbolic and verifier-backed tasks should be checked, not merely narrated.
- Memory, replay, and falsification are compounding intelligence, not decoration.
- Every serious subsystem must be registered, switchable, measurable, reportable, and testable.
- Efficiency is not the final goal. Efficiency buys more experiments, longer memory, better evals, and more owner-aligned intelligence per unit compute.

## Executive Ranking

| Rank | Technology family | Why it matters for An-Ra | First An-Ra action |
|---:|---|---|---|
| 1 | Golden eval + verifier replay | Prevents random "improvements" from corrupting the system | Create an eval baseline before training changes |
| 2 | SparseLoRA / efficient adapters | Faster owner-data adaptation without full retrain | Add an experiment mode beside LoRA/QLoRA |
| 3 | Muon / SCALE / GaLore optimizer family | More training progress per GB and per step | Benchmark on An-Ra small models first |
| 4 | RLVR, DAPO, GRPO variants | Better reasoning from verifiable rewards | Upgrade `training/rlvr.py` with safer variants |
| 5 | GEPA-style reflective improvement | Improves prompts/tools with far fewer rollouts than RL | Add self-improvement trace optimizer |
| 6 | TurboQuant / KVarN KV compression | Longer context and cheaper inference | Audit `core/turboquant.py` against KVarN |
| 7 | TurboVec-style vector compression | Larger local memory/RAG index | Prototype alternative to FAISS fallback path |
| 8 | LMCache / prefix caching | Avoid recomputing repeated agent context | Add cache benchmark before implementation |
| 9 | EAGLE-3 / speculative decoding | Faster generation and rollout collection | Feasibility study for `generate.py` |
| 10 | Titans / neural test-time memory | Long-horizon cognition research path | Keep as research prototype, not core rewrite |

## Evidence Tiers

- A: proven foundational method with broad adoption.
- B: strong paper and/or open-source system, good candidate for implementation.
- C: frontier method with promising results but limited independent reproduction.
- D: watchlist, useful idea but too risky or too large for immediate integration.

## Research Map

| # | Paper or technology | Evidence | Primary benefit | An-Ra fit | Priority |
|---:|---|---|---|---|---|
| 1 | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | A | Transformer base architecture | Already core | Foundation |
| 2 | [RoPE](https://arxiv.org/abs/2104.09864) | A | Position encoding for long context | Already compatible with brain | Foundation |
| 3 | [YaRN](https://arxiv.org/abs/2309.00071) | B | Context extension with RoPE scaling | Good for long-context training | P1 |
| 4 | [FlashAttention](https://arxiv.org/abs/2205.14135) | A | Faster exact attention | Essential GPU training/inference | P0 if not already active |
| 5 | [FlashAttention-2](https://arxiv.org/abs/2307.08691) | A | Better GPU parallelism | Training speed path | P1 |
| 6 | [LoRA](https://arxiv.org/abs/2106.09685) | A | Cheap adapter fine-tuning | Already philosophically aligned | Foundation |
| 7 | [QLoRA](https://arxiv.org/abs/2305.14314) | A | Fine-tune quantized models | Best constrained-GPU default | P0 |
| 8 | [DoRA](https://arxiv.org/abs/2402.09353) | B | Better low-rank adaptation by separating magnitude and direction | Strong few-data fit | P1 |
| 9 | [SparseLoRA](https://arxiv.org/abs/2506.16500) | B | Up to 2.2x compute reduction and 1.6x measured speedup during LoRA-style tuning | High fit for owner-data fine-tuning | P0 |
| 10 | [LowRA](https://arxiv.org/abs/2502.08141) | C | LoRA under very low bit budgets | Useful for constrained experiments | P2 |
| 11 | [GaLore](https://arxiv.org/abs/2403.03507) | B | Reduces optimizer-state memory, reports up to 65.5% optimizer-state reduction | Good for small/full training experiments | P1 |
| 12 | [Q-GaLore](https://arxiv.org/abs/2407.08296) | B | Quantized low-rank gradient optimizer | Good under VRAM limits | P1 |
| 13 | [Natural GaLore](https://arxiv.org/abs/2410.16029) | C | Faster/more stable GaLore variant | Benchmark only | P2 |
| 14 | [Muon is Scalable for LLM Training](https://huggingface.co/papers/2502.16982) | B | Orthogonalized optimizer, strong large-model training results | Good optimizer bake-off candidate | P0 |
| 15 | [SCALE optimizer](https://arxiv.org/abs/2506.16659) | B | Adam-quality results using 35-45% total memory in reported LLaMA-scale tests | High fit for memory-efficient pretraining | P0 |
| 16 | [FP8-LM](https://arxiv.org/abs/2310.18313) | B | FP8 training and low-bit optimizer memory | Useful only with proper hardware | P2 |
| 17 | [Switch Transformer](https://arxiv.org/abs/2101.03961) | A | Sparse MoE scaling | Long-term architecture idea | P2 |
| 18 | [Mixtral 8x7B](https://arxiv.org/abs/2401.04088) | A | Practical sparse MoE recipe | Distillation/teacher signal more realistic than rewrite | P2 |
| 19 | [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | A/B | RLVR can induce reasoning behaviors; distillation matters | Directly supports An-Ra verifier loop | P0 |
| 20 | [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300) | A/B | Group relative rewards without a critic | Already reflected in `training/rlvr.py` | Foundation/P0 |
| 21 | [DAPO](https://openreview.net/forum?id=2a36EMSSTp) | B | Open RL system improving GRPO stability and efficiency | Strong upgrade candidate for RLVR | P0 |
| 22 | [Dr. GRPO / GRPO Done Right](https://arxiv.org/abs/2503.20783) | B | Bias fixes for GRPO-style training | Add as loss/eval comparison | P1 |
| 23 | [GRPO-lambda](https://arxiv.org/abs/2510.00194) | C | Better token-level credit assignment | Strong reasoning-training experiment | P1 |
| 24 | [Memory-constrained RL reasoning](https://arxiv.org/abs/2504.20834) | B | S-GRPO/T-SPMO improves small LLM reasoning under limited GPU | Very high fit for An-Ra resources | P0 |
| 25 | [iGRPO](https://arxiv.org/abs/2602.09000) | C | Iterative self-feedback extension of GRPO | Worth watching for self-conditioning | P2 |
| 26 | [STaR](https://arxiv.org/abs/2203.14465) | A | Self-taught reasoning with rationales | Already aligned with replay pipeline | P0 |
| 27 | [DPO](https://arxiv.org/abs/2305.18290) | A | Simpler preference optimization | Useful for owner preference data | P1 |
| 28 | [GEPA](https://arxiv.org/abs/2507.19457) | B | Reflective prompt evolution; reports up to 35x fewer rollouts than GRPO | Excellent self-improvement fit | P0 |
| 29 | [ReAct](https://arxiv.org/abs/2210.03629) | A | Reasoning plus tool use | Supports agent loop design | Foundation |
| 30 | [Reflexion](https://arxiv.org/abs/2303.11366) | A/B | Verbal self-reflection improves agents | Strong for failure replay and tool correction | P1 |
| 31 | [Toolformer](https://arxiv.org/abs/2302.04761) | B | Models learn when to call tools | Future natural tool routing training | P1 |
| 32 | [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401) | A | External memory grounding | Already central to memory vision | Foundation |
| 33 | [RETRO](https://arxiv.org/abs/2112.04426) | B | Retrieval-enhanced language modeling | Strong long-term brain/memory bridge | P2 |
| 34 | [LMCache](https://arxiv.org/abs/2510.09665) | B | Shared/offloaded KV cache, reports up to 15x throughput with vLLM in workloads | Great for agent sessions and repeated prefixes | P1 |
| 35 | [TurboQuant](https://arxiv.org/abs/2504.19874) | B/C | KV-cache and vector quantization, reports about 6x KV memory reduction in paper settings | Already partly present in `core/turboquant.py` | P0 audit |
| 36 | [KVarN](https://arxiv.org/abs/2606.03458) | C | Calibration-free KV quantization focused on autoregressive reasoning error accumulation | Compare against TurboQuant before deeper integration | P0 audit |
| 37 | [TurboVec](https://pypi.org/project/turbovec/) | C | Rust/Python vector index using TurboQuant-style compression | Memory/RAG index candidate, not training tech | P1 |
| 38 | [VecInfer](https://arxiv.org/abs/2510.06175) | C | Low-bit KV cache with outlier suppression | Inference benchmark candidate | P2 |
| 39 | [KVTC](https://arxiv.org/abs/2511.01815) | C | Transform coding for reusable KV caches, claims high compression | Useful for offloaded/stale cache storage | P2 |
| 40 | [InnerQ](https://arxiv.org/abs/2602.23200) | C | Hardware-aware tuning-free KV quantization | Benchmark-only candidate | P2 |
| 41 | [vLLM / PagedAttention](https://arxiv.org/abs/2309.06180) | A | Efficient serving and KV memory management | Good external serving reference | P1 |
| 42 | [EAGLE-3](https://arxiv.org/abs/2503.01840) | B | Speculative decoding with reported speedups up to 6.5x in paper settings | Good for generation and RL rollout speed | P1 |
| 43 | [SpecForge](https://arxiv.org/abs/2603.18567) | C | Open framework for EAGLE-3 draft models, reports up to 4.48x on SGLang | Good if speculative decoding becomes P1 | P1/P2 |
| 44 | [Speculative Vocabulary](https://arxiv.org/abs/2602.13836) | C | Improves speculative acceptance length | Watchlist for speculative decoding | P2 |
| 45 | [Mamba](https://arxiv.org/abs/2312.00752) | A/B | Linear-time sequence modeling; reports fast inference and long sequence scaling | Architecture research path | P2 |
| 46 | [Mamba-2 / SSD](https://arxiv.org/abs/2405.21060) | B | Structured state-space duality, more transformer-compatible SSM | Architecture research path | P2 |
| 47 | [Gated DeltaNet](https://arxiv.org/abs/2412.06464) | B | Linear attention with better retrieval/long-context behavior | Possible future hybrid block | P2 |
| 48 | [Log-Linear Attention](https://arxiv.org/abs/2506.04761) | C | Hybrid between softmax expressiveness and linear efficiency | Watchlist | Watchlist |
| 49 | [Titans](https://arxiv.org/abs/2501.00663) | B/C | Test-time neural memory, reports very long-context behavior | High-vision fit, high implementation risk | P2/watchlist |
| 50 | [Momentum DeltaNet](https://huggingface.co/papers/2605.05838) | C | Parallelizable Delta linear attention with momentum | Watchlist | Watchlist |

## What To Build First

### 1. Evaluation before mutation

The most important upgrade is not a paper. It is discipline. Before changing training, optimizer, memory, or generation, An-Ra needs a golden eval baseline that covers:

- Identity drift and CIV/ESV stability.
- Owner-style response quality.
- Symbolic math/code correctness.
- Long-context recall.
- Tool-use success.
- Latency, tokens, and memory footprint.

This is the only way to separate real improvement from noise. It directly supports `P0-02` and should happen before any serious training change.

### 2. Training efficiency without vision drift

The best immediate path is not a full architecture rewrite. It is faster adaptation over the owner-first data law:

- Use QLoRA/DoRA as the stable adapter baseline.
- Add SparseLoRA-style contextual sparsity to reduce fine-tuning compute.
- Compare Muon, SCALE, GaLore, and AdamW on the same small An-Ra model and data slice.
- Report tokens/sec, max VRAM/RAM, loss curve, identity probes, and eval delta.

Success means more useful learning sessions on the same hardware, not just lower loss.

### 3. Reasoning via verifiers, replay, and reflection

An-Ra already has `training/rlvr.py` and `training/replay_pipeline.py`. The best research direction is to evolve these into a measured loop:

- Use GRPO as baseline.
- Add DAPO-style overlong reward shaping and token-level policy loss experiments.
- Add S-GRPO/T-SPMO variants for memory-limited reasoning.
- Add GRPO-lambda only after stable baseline curves exist.
- Feed verified successes and hard failures into replay with provenance.
- Use GEPA-style reflection for prompts, tool policies, and self-improvement rules before weight updates.

This preserves the An-Ra idea that intelligence compounds from verified attempts.

### 4. Memory and RAG efficiency

TurboVec is relevant, but it belongs in memory/RAG, not in the trainer. Current `memory/faiss_store.py` has FAISS and a NumPy fallback. The safest path is:

- Keep FAISS/fallback as baseline.
- Add a benchmark harness for recall@k, memory bytes, build time, and query latency.
- Test TurboVec-like compressed search only behind an optional dependency and feature flag.
- Never replace memory routing until recall and identity-critical retrieval are proven.

### 5. Inference efficiency

`core/turboquant.py` is already present. The next step is not to duplicate TurboQuant; it is to audit it.

- Compare implementation claims to TurboQuant paper details.
- Add tests for compression ratio, attention-score error, generation quality, and long-horizon decoding.
- Compare KVarN because it specifically targets autoregressive reasoning error accumulation, the exact regime An-Ra cares about.
- Evaluate speculative decoding only after baseline generation and KV tests are stable.

### 6. Architecture research, not rewrite

Mamba, Gated DeltaNet, Log-Linear Attention, Titans, and Momentum DeltaNet are important but risky. They should be studied as future modules or small experimental blocks, not dropped into the brain without evidence.

The right An-Ra approach:

- Prototype at small scale.
- Compare against current transformer on identical tokenizer/data/eval.
- Require identity and symbolic benchmarks to pass before promotion.
- Treat architecture changes as milestone train work with sovereignty gating.

## Anti-Hype Rules

1. Do not accept "5x faster" without asking: faster where? training, attention logits, end-to-end inference, vector search, or rollout collection?
2. Do not combine claims from different settings. TurboQuant memory compression and attention speed numbers are not automatically end-to-end speedups.
3. Do not optimize one metric while losing identity, recall, verifier accuracy, or owner style.
4. Do not rewrite architecture before cheaper training, replay, and eval loops are mature.
5. Do not treat early libraries as production dependencies until tested under An-Ra workloads.

## Best Fit Summary

If An-Ra follows only one sentence from this paper, it should be:

Build a measured loop where owner data, verifier rewards, hard-failure replay, efficient adapters, and memory compression allow more high-quality learning cycles without compromising identity.

That is the path from "latest research" to An-Ra's actual vision.
