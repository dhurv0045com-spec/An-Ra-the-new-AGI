# T1D PRE50M ADDENDUM — final validation before the 50M-token milestone

Status: **PREREGISTERED — NO RESULTS**. Amends nothing in T1D/PLAN.md (all
T1D arms, gates, budgets unchanged); ADDS the PRE50M systems-certification
phase to the same session. No results exist anywhere.

## A0. What "50 million checkpoint" means (resolved from Cymek, not guessed)

```text
PRE50M_TARGET_TYPE  = training-tokens milestone checkpoint
PRE50M_TARGET_VALUE = 50,000,000 tokens consumed
CYMEK_SOURCE_PATH   = v5_contracts/training_spec.py:223-224
                      ("final_500m_milestone_threshold_tokens": 50_000_000;
                      milestones every 100M tokens, every 50M in the final 500M)
                      + blueprint/DECISIONS.md:142
CYMEK_SHA           = 28bf57a
```

Cymek defines NO ~50M-parameter model spec (ladder is 35M P35 → 102M →
250M). Therefore: NO new model is built for PRE50M. The smoke certifies the
path on the largest available Citadel scale (SCALE2 7.4M) and produces the
throughput curve that sizes the 50M-token run. If a 50M-param spec appears
later, this addendum does not cover it.

## A1. Session order (§24 priority)

preflight → calibration → manifest → T1D arms A–E → PRE50M certification →
cross-arm summary (+ PRE50M verdicts) → one bundle. T1D science runs
COMPLETELY and unchanged; PRE50M never replaces arms.

## A2. PRE50M smoke (SCALE2, bounded, separate smoke input)

3–10 updates on a deterministic T0/T1-TRAIN smoke slice (never TEST), then:
finite logits/shapes → finite loss/grads (nonzero) → one optimizer update with
param-hash mutation → checkpoint save → reload into fresh model with output-hash
identity → save+reload optimizer/step state → continue one more update
(resume; strongest ACTUAL invariant recorded, none fabricated). Numerical
health throughout (first/last loss, grad min/mean/max, nonfinite count).

## A3. Data interface + packing certification (small synthetic shard)

Token/segment/position/mask/eligible construction, static buckets, host→XLA
transfer, batching, exact ledger (capacity/real/loss-bearing/unique/scheduled/
replayed, padding excluded). Packing adversarial matrix (single row, exact fit,
1-token remainder, max-length row, many tiny rows, mixed lengths, padding
tail): efficiency, no cross-segment attention/loss, segment isolation,
mask correctness, boundary correctness. All adversarial cases are LOCAL unit
tests; the session re-verifies efficiency + isolation live.

## A4. Buckets, compile audit, throughput curve

Calibration extended: candidates (64,64)/(128,64)/(256,64)/(512,64)/(1024,64)
(all static); per candidate: correctness, capacity/real/loss tok/s, first-step
vs second-step vs steady-median timing, UNEXPECTED_RECOMPILE flag (step-2 >
2× steady), observed memory if obtainable, rejection reasons. OOM policy:
try candidates in descending throughput order, keep the first that passes;
record selected + rejected (no operator choice needed). Throughput curve per
scale (MID, SCALE2): full tok/s triple + planning estimates for 10M/50M/100M/1B
tokens (labeled estimates, not measurements).

## A5. Memory + grad-accumulation + checkpoint-compat verdicts

Memory: exact parameter/optimizer/gradient/checkpoint bytes from receipts +
runtime-observed where available → FIT/MARGINAL/DOES_NOT_FIT per scale, with
the next lever named on failure (batch/seq/accum/optimizer — never a Citadel
architecture change). Grad accumulation: NOT required at these scales
(batch ≤1024×64 fits by arithmetic + calibration proof) → recorded
NOT_REQUIRED, no dead machinery built. Checkpoint compat: Citadel smoke
checkpoints stay lightweight-torch format; compat CHECK asserts exact
parameter inventory + spec SHA + both codebase SHAs in metadata (a full
Cymek-transaction-format migration is documented as future work, not claimed).

## A6. Arm diagnostics extension (same session, no extra runs)

Per arm add: first-answer-position top-5/digit-mass/entropy (string-derived
where possible without logits Guides: per-position accuracy, length
distribution, most-common strings/digits, target-vs-generated digit
histograms), teacher-task held-out eval for arm C (digadd/digsub/singlemul/
divmicro exact), easy-tier TRAIN memorization lens (T0/T1/T2 train samples —
already collected; surfaced explicitly), digit-level accuracy everywhere.
Primary metric stays exact-match; these are diagnostic lenses.

## A7. NEXT_50M_DECISION.json (machine-built in session)

Fields per spec §29 (target understood/type/count, fits, recommended
batch/seq, grad-accum required, tok/s estimate, save/reload, resume,
data/packing pass, ready flag, blocking reasons). READY_FOR_50M gate per §30
(system conditions only; arithmetic success NOT required; T1D_SCIENTIFIC_VERDICT
and PRE50M_SYSTEM_VERDICT recorded separately).

## A8. Bundle + budget

Bundle adds: PRE50M_TARGET.json, PRE50M_FEASIBILITY.json,
PRE50M_THROUGHPUT.json, PRE50M_CHECKPOINT_SMOKE.json,
PRE50M_DATA_INTERFACE.json, PRE50M_PACKING.json, DIAGNOSTICS.json (pointer to
in-arm diagnostics), NEXT_50M_DECISION.json. Session ceiling still <2 TPU-h;
PRE50M bounded to ~15–20 min by construction (bounded updates, small shard,
no new corpus).
