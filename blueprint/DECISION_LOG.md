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

## 2026-08-31 — Repository truth and E0 false-green repair

- Exact-head CI at `03c6b1e` was red on Linux although 81 local tests passed. Aggregate receipts hashed raw Windows CRLF bytes, while Git checked out LF bytes. Both aggregate formats now hash canonical JSON semantics; CRLF/LF regression tests cover the boundary. Commit `3f8f80a` passed exact-head GitHub Actions run `33379827996`. **EVIDENCE-BACKED**
- The prior E0 certificate was not shortcut-resistant. Across its exact eight-seed state pool, bag-of-words scored 81.77% and lexical overlap 71.09%, versus casewise chance 13.89%; certification gated only two positional controls. The apparent development pass was revoked as a scientific claim. **EVIDENCE-BACKED NEGATIVE RESULT**
- Generator `e0-eval/0.4.0` forbids a query cutoff equal to an event timestamp, uses between/after-event queries, and supplies eight competing target histories with interleaved variables, rollback, and priority cases. Certificate schema 2 gates first/last candidate, latest-fact, nearest-position, lexical-overlap, and bag-of-words over eight seeds. **EVIDENCE-BACKED IMPLEMENTATION CONTRACT**
- On the repaired canonical receipt, bag-of-words is 4.17% and lexical overlap 10.16% against casewise null 10.37%; latest/nearest are 13.80% against analytic random-serialization null 12.95%. All named controls pass null + 10 points, and independent surface solving still agrees. **EVIDENCE-BACKED DEVELOPMENT RESULT**
- E0 is still not closed: source-disjoint natural evidence, a real external sealed commitment/result, a fresh replication, and a certified model-scoring adapter do not exist. No learned cognition claim or training authorization follows from the development receipt. **OPEN / BLOCKING**
- Highest-information next experiment is scoring-adapter certification across deterministic oracle/broken controls and random-weight P35 × 16k/24k/32k. This precedes learned P35 comparisons because candidate-length/tokenization bias could invalidate E1–E3. **STRONG INFERENCE**

## 2026-08-31 — Scoring firewall and atomic training-state contract

- Added a suffix-only candidate-logprob adapter contract with sum, token-normalized, and byte-normalized aggregation; 96 balanced candidate rotations; oracle/random controls; and deliberately broken position, length, tokenization, and first-token controls. All 13 deterministic contract checks pass. **EVIDENCE-BACKED INFRASTRUCTURE**
- Random fake logits exposed material evaluator bias: summed likelihood selected the fewest-token candidate 65.625% of the time and byte normalization selected the constructed first-token/token-density pattern 84.375%. Token normalization was less biased but is not selected. `production_scoring_mode=null` remains fail closed until exact random-weight P35 × real tokenizer/device evidence exists. **EVIDENCE-BACKED NEGATIVE RESULT / OPEN DECISION**
- Run-spec schema 2 resolves the non-divisible 5B budget: 38,146 full 131,072-token updates plus one 127,488-token partial update, exactly 5,000,000,000 tokens with no accounting overshoot. **FROZEN CONTRACT; LR/BATCH STILL EXPERIMENT REQUIRED**
- Added framework-neutral `TrainingState` and content-addressed checkpoint-transaction contracts. A bounded canary starts every lifetime counter at zero, schedules by cumulative tokens, binds source/model/tokenizer/data/pack/run/optimizer/schedule/curriculum identities, enforces parent fencing, verifies component inventory/hash/size, and matches uninterrupted versus clean-copy resume across `4+4+2` tokens. Missing/corrupt components and crashes after stage, publication, or pointer update fail safely. **EVIDENCE-BACKED LOCAL TRANSACTION SEMANTICS**
- This does not yet prove one atomic real-P35 checkpoint, remote durability, distributed rank state, object-store CAS, or TPU/XLA restore. Local fsync is not durable storage, and final checkpoint promotion remains unauthorized. **OPEN / BLOCKING BEFORE FREEZE**

## 2026-08-29 — STEP 2 research synthesis

Evidence basis: repository receipts summarized in `EVIDENCE_AND_CONTEXT.md` plus the primary-source claim ledger in `report-source.md`.

### Decisions made

1. **Do not jump directly to 300M–3B.** The first major V5 candidate remains near V4 scale at approximately 195M parameters. V4 has only 329,908,224 certified continuation tokens and does not establish a capacity ceiling. **STRONG INFERENCE**
2. **Keep V5-A dense and architecturally conservative.** No MoE, recurrence, explicit memory, latent-thought, SSM, or multi-objective cocktail in the baseline. This preserves causal attribution. **STRONG INFERENCE**
3. **Adopt a provisional 28×768, 12Q/4KV, FFN-2048, full-attention 4k candidate.** Exact shape, QK norm, and GQA remain subject to E2. **EXPERIMENT REQUIRED**
4. **Use a provisional 24,576-entry identity-preserving byte-fallback tokenizer.** Freeze only after the 16k/24k/32k tournament. **EXPERIMENT REQUIRED**
5. **Target 5.0B audited tokens, not another short continuation.** Provisional mix is 65% high-quality natural, 20% code/math/formal, and 15% verified cognition; E3 must select the cognition fraction. **STRONG INFERENCE / EXPERIMENT REQUIRED**
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

- Added the canonical [`BENCHMARK.md`](BENCHMARK.md) contract with an explicit mathematical pre-mortem loop, null/chance calculations, power policy, and raw-versus-assisted measurement rules.
- Replaced position-solvable state cases with shuffled semantic-time logs covering latest, intermediate, rollback, precedence, and interleaved variables.
- Replaced the permanent reverse-pair rule with eight development structures and disjoint sealed/fresh structure sets; fixed-rule and bag-of-words controls are now pooled certification gates.
- Added difficulty axes, sensitivity/invariance pair summaries, raw-Core/assisted/intervention-dependence result contracts, and fresh/sealed replication identity checks.
- Added the matched-budget E1 tokenizer tournament plan. No E1 artifact or main V5 training run is authorized.
- Created `blueprint/IMPLEMENTATION_BLUEPRINT.md` as the authoritative package/interface/command/checkpoint/CI/milestone design.

## 2026-08-30 — Local device evidence pass

- Added a bounded `e0_cognition.device_benchmark` probe that reports CPU E0 throughput and runs a matched CPU/CUDA matrix smoke test when PyTorch CUDA is available.
- Measured 368 E0 cases and 112 pairs at approximately 6.6k cases/sec on the local 16-thread AMD host. The default Codex Python lacked PyTorch, but a later workspace audit found the repository's existing `.venv-cuda`: PyTorch 2.11.0+cu128 sees the RTX 4050 Laptop (6,141 MiB) and supports bf16. The bounded 2048² FP32 matmul measured 4.85 TFLOP/s CUDA versus 0.211 TFLOP/s CPU. These are device-path measurements, not model quality or TPU throughput.
- An attempted installation into the bundled runtime was rejected at the usage-limit approval gate; no installation was actually needed once the repository CUDA environment was located. Runtime discovery must therefore precede dependency installation in future preflight logic.
- Historical v0.4 result: recalibrated positional shortcut gates against an exact fact-permutation null after multi-seed tests exposed that uniform-candidate chance was mathematically invalid for these heuristics. All 9 tested seeds and the then-current 41 regression tests passed. **SUPERSEDED 2026-08-31:** lexical/bag-of-words state shortcuts were not gated and invalidated the broader shortcut-resistance claim.
- Ran a matched-parameter NumPy FFN proxy at sequence length 512. The stabilized CPU run estimated matmul-only forward latency of roughly 345 ms for 34×768, 310 ms for the 26×896 center, and 278 ms for 17×1152 (before attention, norms, communication, or framework overhead). This is a performance trade-off signal, not a cognition result; E2 must measure the same shapes at matched FLOPs on the target accelerator and E0/OOD gates.
- Historical v0.4 receipt reported 38,146 full-size updates for the 250,216,960-parameter / 5B-token center, plus 7.5065e18 idealized 6ND FLOPs and 3.503 GB full-resume storage without gradients. **CORRECTED 2026-08-31:** 38,146 is only the quotient; run-spec v2 adds the required 127,488-token partial update for 38,147 total. The storage/compute feasibility observation remains valid.
- Historical generator-0.3.0 result: 20 development seeds (100–119) passed the incomplete position-only certificate. The position heuristic stayed in [0.1719, 0.2344] against its permutation null. **SUPERSEDED 2026-08-31:** this sweep did not test the lexical/bag-of-words failure and cannot support current E0 certification.
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
- Compared identical exact-stack FP32/BF16 weights, tokens, logits, FP32 cross-entropy, and representative gradients over three CUDA and three CPU seeds for every P35 shape. All 18 short-context cases pass preregistered limits. A separate nine-case CUDA replication at native 2k also passes, with worst logit cosine 0.999917, logit RMS error 1.289%, sampled-gradient cosine 0.999631, and gradient RMS error 2.717%. Across all receipts, worst loss relative error is 0.000118. An initial float32 cosine accumulator was rejected after producing values slightly above one; receipts were regenerated with float64 accumulation. Classification: **EVIDENCE-BACKED local BF16 forward/backward parity through P35 native context; EXPERIMENT REQUIRED for 4k V5-A, TPU/XLA, real updates, optimizer state, and long-run tails**. No optimizer update occurred.

The 250M production model and trainer are not implemented or authorized; the current code is contract and research infrastructure.

- Added a native-4k RoPE conformance probe using the exact P35 `RotaryEmbedding` and an independent float64 oracle. Fresh seeds 36101–36105 (CUDA) and 36101–36103 (CPU) pass FP32/BF16 reference, norm, and relative-shift checks. The first calibration exposed overly strict float32 limits (`2e-6`/`3e-6`) because executable phases are computed in float32; that failed receipt is preserved, limits were analytically revised to `5e-5`, and fresh seeds pass. Classification: **EVIDENCE-BACKED local RoPE implementation conformance; EXPERIMENT REQUIRED for base choice, extrapolation, TPU/XLA, and learned cognition**. No attention or optimizer update occurred.
- Ran the real-update canary on CUDA (sequence 32) and CPU (sequence 16) for three deterministic AdamW updates plus save/load continuation. The corrected BF16 path uses FP32 master parameters under BF16 autocast: parameter hashes change, optimizer owns all live parameters, moments are nonzero and FP32, post-clip norms stay within 1.0, steps reach 3, and resumed parameters exactly match uninterrupted execution. Native BF16 parameter storage is retained as a negative control because its moments are BF16 and its post-clip norm overshoots 1.0 by about 0.3%, failing both integrity gates. Classification: **EVIDENCE-BACKED local update wiring/resume; EXPERIMENT REQUIRED for long-run, distributed, and TPU/XLA behavior**.
- Repeated the corrected real-update canary for 10 updates with a fresh seed on both CUDA (sequence 32) and CPU (sequence 16). Losses and gradients stayed finite, FP32 moments remained active, and optimizer-state resume equivalence remained exact. This strengthens bounded stability evidence but does not freeze the LR or schedule.
- Extended the corrected canary to CUDA sequence 1,024 and CPU sequence 128 for two updates. Both pass registered tolerance-based parameter and optimizer-state resume checks. Strict bitwise equality is recorded separately because deterministic runtimes may pass while other kernels show small reduction-order drift (historically FP32 max parameter error `2.70e-6`; BF16 `1.15e-3`); future contracts gate on tolerance, not strict cross-backend identity.
- Added a deterministic E1 tokenizer perturbation sweep with 64 cases each for numbers, identifiers, nonces, Unicode, spacing, and formal expressions. All local 16k/24k/32k artifacts round-trip exactly with zero unknowns; overall tokens/byte are 0.63834/0.60568/0.59293 and 24k is intermediate on every family. This is broader local development evidence only; external corpus custody and matched P35 learning remain required.
- Extended the real-update/save-resume canary to all three P35 shape arms and refreshed source-bound CPU/CUDA receipts. Deep-narrow, middle, and wide-shallow all pass FP32-master BF16 update, clipping, optimizer ownership, and tolerance-resume checks. Added explicit model-state lifetime cleanup after serialization after local probes demonstrated multi-GB RSS could otherwise look like a training failure. **EVIDENCE-BACKED infrastructure invariant; EXPERIMENT REQUIRED for TPU/XLA, distributed collectives, and long-run optimization.**
- Reclassified strict bitwise equality at 1k context as an observation rather than a portable gate: the current deterministic CUDA runtime passes, while prior kernels showed small reduction-order drift. Production promotion gates on registered dtype/backend tolerances and records strict equality separately.
- Added a content-addressed sampler/cursor canary. CPU and CUDA both consume 9×47 tokens across shuffled shard/sequence boundaries, round-trip the cursor through JSON, reproduce the uninterrupted tail exactly, agree on device/host digests, and reject manifest or offset tampering. **EVIDENCE-BACKED local single-process continuity; EXPERIMENT REQUIRED for distributed partitioning, real pack I/O, and durable remote restore.**

## 2026-08-31 — Exact local scoring and promotion-integrity pass

- Ran suffix-only teacher-forced candidate scoring with random exact middle-P35 weights and the actual local 16k/24k/32k tokenizers on CPU and RTX 4050 CUDA. Full-sequence tokenization is split only at a verified non-crossing prompt/candidate boundary. All candidate rotations were stable; 486 paired scores produced zero prediction mismatches with relative RMS error 1.286e-7.
- Rejected all three naive aggregation modes as production policy. Sum chose a fewest-token candidate in 100% of null groups, byte normalization in 83.33%, and token normalization in 50%--66.67%. Device parity proves implementation agreement, not measurement validity.
- Split evaluation, durable upload/redownload restore, and promotion into separately hash-bound contracts. Promotion now fails closed on chronology-only selection, assisted/native substitution, absent fresh replication, mutable artifacts, failed gates, or missing independent signature.

## 2026-08-31 — Exact P35 transaction join

- Joined the exact middle-P35 model, AdamW optimizer, constant scheduler, RNG payload, cursor, source ledger, and `TrainingState` to `CheckpointStore` in a bounded two-update canary.
- The first update publishes an immutable content-addressed generation; a clean local copy restores it and reproduces the uninterrupted second update with zero parameter and optimizer-state error. The final state reaches update 2 / 16 synthetic tokens, with optimizer step 2.
- This closes local integration only. Distributed rank state, remote object-store custody, TPU/XLA behavior, and long-run training remain open; no model checkpoint tensors are committed.

## 2026-08-31 — Runner failure-state fence

- Added `v5_training.runner`: an in-flight update cannot advance the durable parent, completion requires a committed target update, and worker/upload failure preserves the last good checkpoint for explicit recovery.
- Canonical JSON round-trip and pending-update/terminal-state tests pass. Distributed supervision and target failure injection remain open.

## 2026-08-31 — Distributed rank checkpoint boundary

- Added `v5_training.distributed` to bind each rank's RNG, optimizer shard, cursor, data-shard identity, token contribution, and collective barrier to one canonical checkpoint.
- The coordinator contract rejects incomplete or duplicate rank sets, shard reuse, mismatched barriers, world-size drift, and global-token reconciliation errors. Target TPU/XLA collective behavior remains unmeasured.

## 2026-08-31 — Target TPU/XLA preflight contract

- Added `v5_training.target_preflight`, a target-only smoke command for XLA device identity, world-size/ordinal reconciliation, BF16 matmul, all-reduce, and device RNG progression.
- The command returns `BLOCKED_TORCH_XLA` with explicit missing dependencies when executed off-target, so a CPU developer machine cannot accidentally certify TPU readiness. No training is performed.

## 2026-09-01 — Scoring-policy preregistration

- Rejected a proposed six-group calibration run as underpowered before executing it. The existing six-group result remains consumed pilot evidence only.
- Froze a 256-triplet, five-seed, three-tokenizer tournament with DC-PMI primary, contextual calibration secondary, two neutral panels, equivalence/decoy/sensitivity/device gates, and separate development→immutable selection→fresh sequencing. The receipt contains no model outcomes.
- Independent audit found fixture v1's hidden label was exactly recoverable from surface family. It was invalidated before powered execution; schema 2 crosses all 6×3 family/role cells to within one count and gates the contingency.
- Added a resumable exact-P35 development runner with one immutable device×tokenizer×seed shard per cell, common trace reuse, exact full-rank panel and four-candidate decoy recomputation, valid trace-level interventions, five-seed Student-t TOST, and Holm correction across the frozen 132-hypothesis promotion family. A one-group CUDA smoke passed and was discarded as non-scientific evidence.
- Executed all 15 CUDA development cells for 0.07574 GPU-hours. DC-PMI and contextual calibration both selected the fewest-token role 100% across every tokenizer while crossed hidden-label selection stayed at chance. Both are rejected; CPU and fresh stages were stopped as logically unable to rescue a failed primary bias gate.

## 2026-09-01 — V5-A implementation candidate v1.0

- Froze one exact code-facing candidate in `V5_TRAINING_SPEC_v1.0.md` and `v5_contracts.training_spec`; unset external identities and experimental gates keep the main run unauthorized.
- Resolved the scale-family GQA contradiction with 2:1 P35/M102/V5 recipes, exposed QK epsilon `1e-6`, fixed the intended AdamW decay/no-decay rule, selected a single FP32 persistent parameter layout, and specified exact WSD, accumulation, final-partial-update, sequence-supercycle, checkpoint, and decoding behavior.
- Implemented the canonical AdamW constructor with identity-checked, deterministic decay/no-decay groups and a hashable ownership receipt. The exact CUDA P35 transaction canary now covers every optimizer group after clean-copy restore; parameter and optimizer-state continuation errors are both zero. Freeze review also caught and corrected a stale P35 count: affine QK scales add 3,072 parameters, making the 16×384 recipe exactly 35,414,400 parameters. **EVIDENCE-BACKED local implementation; TPU/XLA REQUIRED**
- Chose CE-only for the launch candidate. Query-swap and trace losses remain zero until a FLOP-matched, equal-suffix-token E3 challenger passes fresh candidate-free and natural transfer.
- Fixed the 750M-token cognition slice across nine native operations and uniform difficulty/family interleaving; runtime tools, memory, search, and intervention routing remain outside Core.
- Repaired conditional realization so its denominator is correct unassisted selection, not constrained-output correctness.
- Versioned runner state now separates volatile completed updates from durable committed updates, supports the ~76-update gap implied by 10M-token recovery cadence, and rolls volatile progress back to the durable parent on recovery.

## 2026-09-02 — Canonical execution package and fail-closed launch gate

- Consolidated every authoritative architecture, training, cognition, benchmark, experiment, governance, and operator document under `blueprint/`; historical evidence remains outside it and cannot override it.
- Added `blueprint/LAUNCH_GATES.json` and `v5_contracts.launch_readiness`. The checker verifies the package and evidence inventory, not scientific truth: E1–E6 `PASS`, existing receipts with matching SHA-256, and six external identities advance only to independent freeze review. A production launcher and signed-launch verifier remain unimplemented; this checker never authorizes main training.
- Added CI reproduction of `artifacts/v5/launch_readiness.json`. The repository template intentionally reports `READY_FOR_PRELAUNCH_EXPERIMENTS` and `main_training_authorized=false`; editing a status to `PASS` without a real receipt fails closed.
- Added Windows/Linux newline-invariant document hashes and regression tests proving a complete self-declared evidence inventory cannot authorize training.
