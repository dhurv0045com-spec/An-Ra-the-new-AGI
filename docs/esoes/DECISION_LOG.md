# ESOES Decision Log

This file prevents silent redesign.

## 2026-08-29 — Branch created

Base: `core-vnext` at `054619f20851317e9b1c49b6f31599f6444a8280`.

Decision:

- create a separate research/design branch rather than modifying the active V4 training branch;
- treat V4, EXP, and VNEXT as evidence, not immutable architecture;
- do not launch a major V5 training run until the cognition-first blueprint is stress-tested and frozen;
- preserve open questions explicitly rather than allowing execution agents to answer them implicitly in code.

Current status: **V5 design open; no final architecture/tokenizer/data/optimizer/scale decision frozen.**

## 2026-08-29 — STEP 2 research synthesis

Evidence basis: repository receipts summarized in `EVIDENCE_AND_CONTEXT.md` plus the primary-source claim ledger in `report-source.md`.

### Decisions made

1. **Do not jump directly to 300M–3B.** The first major V5 candidate remains near V4 scale at approximately 195M parameters. V4 has only 329,908,224 certified continuation tokens and does not establish a capacity ceiling. **STRONG INFERENCE**
2. **Keep V5-A dense and architecturally conservative.** No MoE, recurrence, explicit memory, latent-thought, SSM, or multi-objective cocktail in the baseline. This preserves causal attribution. **STRONG INFERENCE**
3. **Adopt a provisional 28×768, 12Q/4KV, FFN-2048, full-attention 4k candidate.** Exact shape, QK norm, and GQA remain subject to E2. **EXPERIMENT REQUIRED**
4. **Use a provisional 24,576-entry identity-preserving byte-fallback tokenizer.** Freeze only after the 16k/24k/32k tournament. **EXPERIMENT REQUIRED**
5. **Target 4.0B audited tokens, not another short continuation.** Provisional mix is 65% high-quality natural, 20% code/math/formal, and 15% verified cognition; E3 must select the cognition fraction. **STRONG INFERENCE / EXPERIMENT REQUIRED**
6. **Standard LM is the base objective.** The only candidate auxiliary is true query-swap contrast on mechanically verified examples. The failed SFT7 same-query margin objective is rejected. **EVIDENCE-BACKED / EXPERIMENT REQUIRED**
7. **Separate representation, selection, and realization.** Promotion uses worst-family OOD gates and fresh replication; the final checkpoint is never promoted automatically. **EVIDENCE-BACKED**
8. **Keep tool execution, durable memory, long-horizon planning, risk policy, verification, and credit assignment in the Connector.** Internalize only repeated local cognitive primitives with replicated transfer. **STRONG INFERENCE**
9. **Collapse pre-freeze research into E0–E6.** The 35M screens and 102M replication are mandatory before V5-A. **STRONG INFERENCE**

### Rejected assumptions

- Lower LM loss is enough to select the parent checkpoint.
- More parameters are currently the highest-confidence use of compute.
- Token compression alone selects the best tokenizer.
- More synthetic data is automatically better.
- A runtime normalization success proves native Core cognition.
- Broad adaptive curriculum or many auxiliary losses should be added before a strong control exists.

### Current status

The research direction is decided, but the numerical training spec is **not frozen**. E0 development certification has begun; full sealed certification is the next gate. Major V5 training remains unauthorized.

## 2026-08-29 — Ground Blueprint v0.2 evidence phase

- Reclassified EXP v10/v11 pair/composition promotions as contaminated by stale candidate applicability, incomplete controls, and reproduction gaps; retained v7/v8/v9 only for three-action repair-success routing.
- Rejected the 166 historical VIE count as a qualified causal bank.
- Physically removed inherited V4/VNext model, trainer, notebooks, outputs, tests, and launch infrastructure from ESOES while retaining immutable branch references.
- Implemented the E0 development package and deterministic certificate. Its pass certifies infrastructure invariants only; a real sealed suite and full shortcut red-team remain prerequisites for E1.
- Kept the 195M dense family provisional. No architecture number was upgraded from hypothesis to fact.

Major V5 training and production-stack implementation remain unauthorized.

## 2026-08-29 — Ground Blueprint v0.3 code/infrastructure phase

- Reopened the provisional scale center to the requested 250M envelope and derived a coherent 26×896, 14Q/7KV, FFN-2368 configuration with exactly 250,216,960 parameters including affine QK-norm scales.
- Increased the center token budget to 5B to preserve approximately 20 tokens/parameter; recalculated data allocation, optimizer-update count, FLOPs, storage, and wall-time planning.
- Added framework-independent model/run/checkpoint/promotion contracts and a reproducible implementation receipt.
- Completed E0 context-position/output-format axes and machine-preregistered paired statistical procedures.
- Added a sealed commitment tool that refuses fixtures inside Git and emits no seed/cases/answers.
- Added the E1 artifact-bound identity/compression audit and Pareto harness. No tokenizer winner is claimed without real artifacts and matched training.

## 2026-08-30 — Ground Blueprint v0.4 benchmark repair

- Added the canonical [`benchmark.md`](../../benchmark.md) contract with an explicit mathematical pre-mortem loop, null/chance calculations, power policy, and raw-versus-assisted measurement rules.
- Replaced position-solvable state cases with shuffled semantic-time logs covering latest, intermediate, rollback, precedence, and interleaved variables.
- Replaced the permanent reverse-pair rule with eight development structures and disjoint sealed/fresh structure sets; fixed-rule and bag-of-words controls are now pooled certification gates.
- Added difficulty axes, sensitivity/invariance pair summaries, raw-Core/assisted/intervention-dependence result contracts, and fresh/sealed replication identity checks.
- Added the matched-budget E1 tokenizer tournament plan. No E1 artifact or main V5 training run is authorized.
- Created `blueprints/IMPLEMENTATION_BLUEPRINT.md` as the authoritative package/interface/command/checkpoint/CI/milestone design.

## 2026-08-30 — Local device evidence pass

- Added a bounded `e0_cognition.device_benchmark` probe that reports CPU E0 throughput and runs a matched CPU/CUDA matrix smoke test when PyTorch CUDA is available.
- Measured 368 E0 cases and 112 pairs at approximately 6.6k cases/sec on the local 16-thread AMD host. The default Codex Python lacked PyTorch, but a later workspace audit found the repository's existing `.venv-cuda`: PyTorch 2.11.0+cu128 sees the RTX 4050 Laptop (6,141 MiB) and supports bf16. The bounded 2048² FP32 matmul measured 4.85 TFLOP/s CUDA versus 0.211 TFLOP/s CPU. These are device-path measurements, not model quality or TPU throughput.
- An attempted installation into the bundled runtime was rejected at the usage-limit approval gate; no installation was actually needed once the repository CUDA environment was located. Runtime discovery must therefore precede dependency installation in future preflight logic.
- Recalibrated positional shortcut gates against an exact fact-permutation null after multi-seed tests exposed that uniform-candidate chance was mathematically invalid for these heuristics. All 9 tested seeds and the current 41 regression tests pass.
- Ran a matched-parameter NumPy FFN proxy at sequence length 512. The stabilized CPU run estimated matmul-only forward latency of roughly 345 ms for 34×768, 310 ms for the 26×896 center, and 278 ms for 17×1152 (before attention, norms, communication, or framework overhead). This is a performance trade-off signal, not a cognition result; E2 must measure the same shapes at matched FLOPs on the target accelerator and E0/OOD gates.
- Recomputed the executable 250M run receipt: 250,216,960 parameters, 5B tokens, 38,146 optimizer updates, 7.5065e18 idealized 6ND FLOPs, and 3.503 GB full-resume storage without gradients (4.003 GB with bf16 gradients). A 6 GB laptop GPU therefore leaves limited room for activations and requires memory-efficient attention; this is a feasibility constraint, not evidence to move the main run onto the laptop.
- Repeated the full E0 certificate over 20 independent development seeds (100–119): 20/20 PASS in 117.3 seconds. The position heuristic stayed in [0.1719, 0.2344] against the exact permutation-calibrated null 0.2135; this is stronger stability evidence than the single canonical receipt, while still not a model-quality result.
- Audited the existing V4 native 32k tokenizer as an E1 baseline against all 12 committed canaries. Artifact SHA-256 `1a0140661c9d16c830f8dd8292699946e92db0be6f7aec92fb272d13cc1c745b` passes byte-exact round trip, zero unknowns, and ID-range checks; it encodes 476 UTF-8 bytes as 313 tokens (0.6576 token/byte), with cognition 0.6279 and Unicode/formal 0.9–1.0. This is a baseline to beat, not a 32k promotion; serious corpus and matched P35 evidence remain required.
- The V4 package initializer imports PyTorch even for tokenizer-only use. The audit isolated the tokenizer module to avoid that unrelated dependency, confirming that the future E1 adapter should remain library-independent and artifact-bound.
- Ran an explicitly non-candidate prefix-truncation ablation of the V4 vocabulary. On 536,826 bytes from 71 local code/documentation files, 24k costs +1.56% tokens versus 32k while saving 7,340,032 embedding parameters at width 896; 16k costs +4.70% while saving 14,680,064. All canaries round-trip with byte fallback at every size. This supports retaining 24,576 as the planning center, but cannot promote it without independently trained artifacts, an external corpus, matched FLOPs, and cognition results.
- Implemented the fail-closed E2 static plan. The deep/middle/wide P35 shape arms are 35.420M/35.414M/35.144M parameters (<0.8% spread) with MHA, QK norm, 2k context, vocabulary, data order, and evaluation fixed. Separate fractional arms isolate 3:1 GQA, QK norm, and 4k mixed context. The GQA 2k KV-cache proxy is 16.8 MB versus 50.3 MB for MHA; the 4k full-sequence forward proxy is 599.1G versus 196.5G FLOPs at 2k. These are planning estimates, not accelerator or cognition results.
- E2 preregisters 200M screen tokens with 50M/100M/200M evaluations, one screening seed, and three-seed replication of finalists. It fails closed as `BLOCKED_E1_INPUTS` until tokenizer/corpus hashes exist, then as `BLOCKED_MODEL_IMPLEMENTATION` until an executable constructor hash exists.
- Implemented the staged E3 data/objective plan. Phase A allocates exactly 200M tokens per CE-only arm: 5% cognition = 145,294,118 natural / 44,705,882 code-formal / 10,000,000 cognition; 15% = 130M / 40M / 30M; 30% = 107,058,824 / 32,941,176 / 60M. The natural:code ratio remains 65:20 instead of drifting with cognition share.
- Phase B is mechanically blocked until Phase A supplies a winner and one adjacent mixture. It then permits only CE and verified query-swap λ 0.05/0.15. A trace arm remains disabled without a hashed composition-transfer failure and must be evaluated trace-free. Promotion requires natural transfer, candidate-free and raw-Core gains, and no more than 3% substrate-loss regression; synthetic-only or assisted-only gains are rejected.
- Independently trained exact 16k/24k/32k byte-BPE candidates on a hash-bound local development corpus. The fixed split contains 8,561,653 training bytes and 1,057,977 held-out bytes / 14,757 records; content hashing prevents duplicate text from crossing the split. All compressed artifacts reload, all three candidates have exact round trip and zero unexpected unknowns, and a repeated 24k training run is byte-identical. Held-out tokens/byte are 0.23826 / 0.23217 / 0.23022. Thus 24k is +0.85% versus 32k while saving 7,340,032 embedding parameters at width 896; 16k is +3.50% while saving 14,680,064. Classification: **STRONGER DEVELOPMENT EVIDENCE, NOT E1 PROMOTION**, because source composition is legacy/local and no matched model loss/cognition exists.
- Replicated the isolated bf16 causal-SDPA benchmark over seeds 31001/31002/31003 on the RTX 4050. Native 3:1 GQA is restricted to the math backend on this build and measures 5.20× MHA forward/backward latency with 13.86× peak allocated memory. Explicit repeated-K/V GQA restores the fused path but measures 1.09× MHA latency and 1.45× memory. Native 4k/2k is 3.59× latency and 3.80× memory; QK normalization is 1.09× latency. Native GQA agrees with repeated K/V to maximum absolute error 0.0078125. Classification: **EVIDENCE-BACKED backend rejection for this Windows/PyTorch stack; OPEN for TPU/XLA and cognition**.
- Built an exact full-stack execution canary for the three P35 shape arms. It includes the 24,576-entry tied embedding/output, every block, RoPE, affine QK norm, MHA, SwiGLU, cross-entropy, and backward, while deliberately performing no optimizer update. Execution order is independently shuffled by seed and results are emitted canonically to prevent warm-up/order confounding. Seeds 32001/32002/32003 pass exact parameter counts and finite gradients at 512/1024/2048/4096 tokens. At 512, deep/middle are 3.17×/2.05× wide latency; at 4k the ratios narrow to 1.91×/1.44×. Peak 4k allocation is 3.54/3.14/2.57 GB. A CPU sequence-64 canary gives deep/middle/wide 423/385/308 ms. Classification: **EVIDENCE-BACKED execution prior favoring fewer wider blocks locally; EXPERIMENT REQUIRED for cognition and TPU/XLA**.
- Isolated residual-output initialization using identical random draws and exact P35 stacks. Five CUDA seeds show `1/sqrt(2L)` scaling cuts final residual-stream growth versus unscaled 0.02 to 0.122×/0.156×/0.230× for deep/middle/wide and cuts gradient max/min spread to 0.604×/0.650×/0.731×. Three CPU seeds reproduce both directions at 0.125×–0.240× growth and 0.647×–0.820× gradient spread. A further three-seed CUDA run at native 4k context strengthens the result to 0.115×/0.144×/0.217× growth and 0.542×/0.510×/0.654× gradient spread. Every run passes exact-count, hook-count, finite, and nonzero-gradient checks. Classification: **EVIDENCE-BACKED local initialization prior at intended context; EXPERIMENT REQUIRED for target TPU/XLA and learning/cognition**. The probe performed no optimizer update.
- Attacked QK normalization with paired 0.25×/1×/4× projection-scale perturbations. Across five CUDA seeds at 512/2k/4k, QK-normalized logit RMS stays within 1.00008× and entropy within 0.000008; the unnormalized control changes logit RMS exactly 256× and entropy by 0.365–0.416. Three CPU seeds reproduce the mechanism. At 4k base scale, QK norm has logit RMS 0.998/effective attended fraction 0.616 versus 0.155/0.988 without it. Proxy gradients are finite but not scale-invariant. Classification: **EVIDENCE-BACKED QK scale control; EXPERIMENT REQUIRED for learned cognition, affine-scale dynamics, and TPU/XLA**. No optimizer update occurred.

The 250M production model and trainer are not implemented or authorized; the current code is contract and research infrastructure.
