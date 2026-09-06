# CYMEK 500M CAMPAIGN PLAN

Status: **SPEC_ONLY — NOT_AUTHORIZED_TO_TRAIN.** Machine-readable identity:
[`CYMEK_500M_CAMPAIGN.json`](CYMEK_500M_CAMPAIGN.json). Cymek is the
production authority; Citadel validates.

## Campaign target

**500,000,000 consumed training tokens** (Cymek
`TrainingState.cumulative_tokens`), milestone ladder
**50M / 100M / 200M / 350M / 500M** — the 50M milestone from PRE50M is
preserved as the first observation point, not erased.

## Milestone crossing (§2)

Pure helper `citadel_tpu.milestones.crossed_milestones(previous, new)`:
token-based, boundary-tested, replay-safe (deterministic re-derivation from
consecutive ledger states; callers persist published milestones so resume
never republishes).

## Production model identity (§12)

**V5A_250M** — 250,216,960 params, 26 layers × width 896, 14Q/7KV GQA,
head_dim 64, FFN 2368, vocab 24,576, ctx 4096 (Cymek
`v5_contracts/model_spec.V5A_250M` at the pinned lineage; exact receipt
enforced by `assert_receipt`). T1D's MID/SCALE2 were experiment models, not
this production target.

## LR schedule (§19)

Cymek's schedule is already defined in token space and is 500M-compatible —
retained unchanged: warmup 0 → 50M linear to 3e-4, stable 3e-4 through 4.5B
(`v5_contracts/training_spec.py`). Final LR at 500M = 3e-4. Table:

| tokens | LR |
|---|---|
| 0 | 0.0 |
| 1M | 6.0e-6 |
| 5M | 3.0e-5 |
| 10M | 6.0e-5 |
| 25M | 1.5e-4 |
| 50M | 3.0e-4 |
| 100M–500M | 3.0e-4 |

## Production path audit (§4)

See [`PRODUCTION_PATH_AUDIT.md`](PRODUCTION_PATH_AUDIT.md) — per-component
CONNECTED/BYPASSED/AMBIGUOUS/MISSING classification. Headline: components
are individually CONNECTED and certified (real-update backend, checkpoint
transactions, packing/cursor, PRE50M smoke on TPU), but the **top-level
production entry point that wires real corpus → tokenizer → packing →
trainer is MISSING**, and the **production corpus is not MATERIALIZED**.

## Data gate (§5–§9) — currently DATA_NOT_READY

- Readiness states: DECLARED → MATERIALIZED → VERIFIED → QUALIFIED →
  RUNNABLE (BRAMASTRA ladder, adopted).
- The production mixture (65/20/15 natural/code/cognition) is DECLARED only:
  no manifest-bound bytes exist. PRE500M therefore reports **DATA_NOT_READY**
  until Cymek materializes + verifies + qualifies the corpus subset.
- Unique-supply arithmetic: 500M consumed tokens must be covered by per-source
  unique runnable tokens; under-supplied sources report their replay factor
  explicitly (`pre500m.data_readiness`). Pathological replay blocks.
- Cross-source union dedup required (receipt-local unique sums don't count).

## Checkpoints (§16–§18)

- **Recovery**: every ~30 minutes, rotating (4).
- **Scientific**: immutable at 50M/100M/200M/350M/500M.
- Storage estimate (fp32 master + AdamW): ~3.3 GB/checkpoint →
  ~16.3 GB recovery rotation + ~16.5 GB milestones. Colab ephemeral disk
  requires the operator-visible persistence strategy before training.

## Multi-session (§14–§15)

Campaign state (recovery checkpoint + campaign receipt) persists in the
session directory; a fresh runtime resumes from the last committed
generation. Exact-resume requirement: tokens, source ledgers, cursor,
sampler order, scheduler, and RNG-dependent behavior identical; parameter
equality per the declared backend-determinism criterion.

## Go/no-go gates (§23)

Machine decisions (CONTINUE / REVIEW / STOP) at 50M / 100M / 200M for: loss
not improving, nonfinite instability, data failure, evaluation regression,
capability below the preregistered floor, excessive replay, throughput
infeasibility, checkpoint/recovery failure. Thresholds frozen before the run.

## Evaluation (§21–§22)

Immutable evaluation points at 0/10M/25M/50M/100M/200M/350M/500M: loss,
capability slices, memorization vs held-out split, generation samples,
termination behavior. Retention probes at ladder milestones (diagnostic,
not blocker — Arkenstone ARK-005 lesson).
