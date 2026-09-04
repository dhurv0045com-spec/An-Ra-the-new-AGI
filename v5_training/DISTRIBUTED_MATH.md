# Distributed training mathematics (M36): reference, not a claim.

No multi-device execution has happened on Cymek. This document pins the
exact mathematics a future distributed implementation must satisfy so that
single-replica behavior transfers without silent redefinition. A target run
that deviates from any rule below fails closed.

## 1. Token accounting

- The atomic unit is the **eligible real token**: non-padding, supervised
  (BOS/PAD excluded, segment transitions excluded, budget-eligible).
- `cumulative_real_tokens` advances by exactly the update's eligible count.
  Microbatch sizes are an implementation detail; the ledger sees only sums.
- Padding never enters any ledger, denominator, or schedule index.

## 2. Replica-global CE denominator

- Loss = (sum over replicas of replica loss-sums) / (sum over replicas of
  replica eligible counts). Never a mean of per-replica means.
- A replica contributing zero eligible tokens contributes sum 0 and count 0;
  it must still participate in the collective with explicit zeros (no silent
  rank dropout).

## 3. Gradient accumulation

- Microbatch gradients accumulate as token-weighted sums; the single
  optimizer update divides once by the replica-global eligible count.
- `optimizer.step()` fires exactly once per update boundary, never per
  microbatch.

## 4. All-reduce placement

- One collective per update at the accumulation boundary: reduce summed
  gradients AND the eligible-count scalar together.
- No collective inside microbatch forward/backward. No second collective
  between clipping and the optimizer update (clip uses the reduced norm).

## 5. Global norm and clipping

- Global L2 norm over the reduced gradients in FP32.
- Clip to 1.0 BEFORE `optimizer.step()`; record pre- and post-clip norms.
- A nonfinite norm aborts the update and the run advances nothing.

## 6. Replica token ledger

- Per-replica eligible counts sum exactly to the update budget.
- The committed `tokens_by_source` ledger is the sum over replicas.
- Any replica/source mismatch aborts before the optimizer update.

## 7. Sampler partition

- Rank `r` of `W` consumes sampler positions `p` with `p % W == r` within
  each epoch, derived from the same `(pack_manifest, seed, epoch)` order as
  single-replica execution. Partitioning never changes per-rank content
  versus the sharded equivalent: the union over ranks, in sampler order,
  equals the single-replica stream exactly.
- Rank RNG streams are disjoint by construction (rank-salted), recorded per
  rank in the checkpoint.

## 8. Checkpoint writer

- Exactly one logical writer advances the lineage per update (collective
  snapshot, single coordinator commit), as in the local transaction.
- Every rank's RNG state, optimizer shard, cursor partition, and token
  contribution is explicit in the manifest (see `distributed.py`).

## 9. What is NOT demonstrated

Single-GPU execution, this document, and the metadata contracts do not
constitute distributed proof. Required before any claim: a multi-device run
of the target canary (`v5_training/target_canary.py`) with per-rank
receipts showing identical math to this spec.
