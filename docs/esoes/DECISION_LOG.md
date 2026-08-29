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
