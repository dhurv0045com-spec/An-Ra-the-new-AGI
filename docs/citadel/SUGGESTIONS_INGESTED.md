# SUGGESTIONS FROM BRAMASTRA + ARKENSTONE — ingested into Citadel design

Date: 2026-09-06. Sources: `origin/BRAMASTRA` (0235001) RESULTS.md +
CITADEL_100X_ROADMAP.md; `origin/Arkenstone` (4cdd8f0) PROGRESS.md +
ARK-004A/006/007 experiment receipts. All findings below are from real
executed experiments on the sibling branches, not speculation.

---

## S1. EOS supervision: PROVEN to fix the stopping problem

**BRAMASTRA result (2 seeds, controlled):**

| Condition | Correct answer prefix | Complete answers (with stop) |
|---|---|---|
| Without EOS supervision | 32/32 | **0/32** |
| With EOS supervision | 32/32 | **32/32** |

Both no-EOS arms hit MAX_TOKENS on all queries. Both EOS arms stopped
correctly on all queries. The effect is deterministic across seeds.

**Citadel action (already in T1E PLAN E1):** EOS is supervised in the
eligible mask, same segment, production `causal_lm_loss`. The T1E helper
`citadel_tpu/t1e_helpers.py` implements the contract and is unit-tested.
**Status: IMPLEMENTED AND TESTED.**

## S2. LR threshold for retention: below 1e-5 is safe

**Arkenstone result (ARK-006 dose-response + ARK-007 5-seed replication):**
- Retention is perfect below LR 1e-5; collapse occurs above.
- Post-G90 decay is STOCHASTIC (data-order-dependent), not deterministic.
- Low-LR safety confirmed universally (5/5 seeds).
- High-LR decay is a seed-DEPENDENT risk factor, not deterministic.

**Citadel action:** the 500M campaign uses LR 3e-4, which is ABOVE the
retention-safe threshold. This means the campaign MUST include:
1. Retention probes at every milestone (already in T1E PLAN E8 + campaign
   spec retention cadence).
2. An LR-decay design consideration: consider decaying LR below 1e-5 near
   the end of training (or between milestones) to preserve acquired
   capabilities.
3. The go/no-go gates at 50M/100M/200M should include a retention check:
   if capability drops significantly, flag REVIEW with LR reduction as a
   candidate remedy.

**Status: RECORDED in 500M campaign spec retention policy + go/no-go gate
criteria. NOT a blocker (diagnostic per §22).**

## S3. Query control: models ignore the query and copy a default answer

**BRAMASTRA result:** in 62/64 fresh worlds, the model returned the same
answer despite the changed query. Trivial copy-policies achieve 50% on
balanced tasks while also scoring zero on both-correct pairs. More
training variety improved fresh-world accuracy (17→48%) but left query
control at zero.

**Citadel action:** this is directly relevant to the cognition goal. The
finding means:
1. T1E's query-conditioned metrics are the right measurement (already
   preregistered).
2. The 500M campaign's evaluation MUST include query-swap sensitivity
   (change the query, check the answer changes) — not just static
   accuracy. This is already in the E0 benchmark design (causal contrast
   cases with mechanical assertions) but should be explicitly added to
   the 500M evaluation token points.
3. The 15% cognition slice in the production mixture needs
   query-swap/causal-contrast pairs, not just Q/A format examples.

**Status: RECORDED. The 500M evaluation cadence should include
query-swap sensitivity as a measured metric at each eval point.**

## S4. TPU vs GPU for small models: torch_xla crashes on tiny models

**Arkenstone result:** "TPU crashes from torch_xla graph compilation RAM
overhead on tiny models". T4 GPU is the required runtime for sub-10M
parameter experiments.

**Citadel action:** T1D/T1E use MID (3.7M) and SCALE2 (7.4M) models.
These are in the "tiny" range that caused Arkenstone's TPU crashes.
However, Citadel's T1D ran successfully on TPU — the difference may be
the specific torch_xla version or the graph size. For the 500M campaign:
1. The 250M V5A model is large enough that TPU should work (Arkenstone's
   crashes were on sub-1M models).
2. T1E should note the TPU risk and provide a GPU fallback option.
3. The PRE500M canary should verify TPU stability at the actual campaign
   model size before committing.

**Status: RECORDED. T1E PLAN already notes the TPU-vs-CPU tradeoff.**

## S5. Learn-to-train ≠ learn-to-transfer (replicated)

**BRAMASTRA result:** models learn training sets to high accuracy but
fail on fresh worlds with different rendering. More variety helped
fresh-world accuracy (17%→48%) but query control stayed at zero.

**Citadel action:** this replicates the T1D finding (all arms FAIL,
train ≤ 11%, test ≤ 6.6%) in a different model/training setup. The
lesson is consistent: **the gap is between fitting and generalizing,
not between optimizing and fitting**. The 500M campaign's 15%
cognition slice must include held-out structural variants, not just
held-out lexical variants.

**Status: RECORDED. T1E PLAN already includes structural holdout
requirement (T2+ tiers).**

## S6. Capacity accounting correction

**BRAMASTRA finding:** `build_data_status` sums receipt-local unique
counts without cross-receipt union/deduplication — aggregate totals
overcount the actual unique supply.

**Citadel action:** already recorded in CROSS_BRANCH_INGESTION.md as a
post-PRE50M observation. The `pre500m.data_readiness` gate requires
per-source supply and computes the shortfall correctly; it does NOT
trust aggregate sums. The 500M campaign must verify the UNION before
counting supply.

**Status: IMPLEMENTED in pre500m data gate.**

## S7. Fresh-runtime resume must be proven on the production model

**BRAMASTRA/Arkenstone both require this.** Citadel's PRE500M certifies
it (exact-resume + fresh-runtime-resume are hard requirements in
`build_next_500m_decision`).

**Status: IMPLEMENTED in pre500m decision builder.**

---

## Priority order for the 500M campaign

1. **EOS supervision** — proven, implemented in T1E helpers
2. **Retention probes at milestones** — designed, in campaign spec
3. **Query-swap sensitivity in evaluation** — add to 500M eval metrics
4. **LR decay consideration near milestones** — go/no-go gate criterion
5. **Data materialization** — BLOCKING (B1), needs operator decision on
   data source
6. **Production entry point** — BLOCKING (B2), implemented on
   cymek-500m-readiness branch
7. **Tokenizer freeze** — BLOCKING (B3), implemented on
   cymek-500m-readiness branch
