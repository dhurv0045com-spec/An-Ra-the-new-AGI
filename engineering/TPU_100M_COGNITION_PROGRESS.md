# 100M TPU cognition and training continuation

## K8 owner experiment context

The separate FINAL-K8 build remains the current two-T4 owner experiment; its
latest report and exact source identity are in **Current evidence** below.
E2's production workspace-policy path uses the typed cognitive evidence ledger
and records conflict/supersession state. The detailed K8 cursor is
[`FINAL_K8_PROGRESS.md`](FINAL_K8_PROGRESS.md). The central limitation remains:
no dedicated Kaggle TPU campaign consumer or real TPU qualification exists.

**Date:** 2026-09-23

**Branch:** `Gandiva`
**Purpose:** durable engineering cursor for the Kaggle TPU 100M path. This
document distinguishes code that exists from the still-unbuilt Kaggle run
path; it does not claim AGI, a successful TPU launch, or learned results.

## Current evidence

The current zero-update FINAL-K8 verification is
[`gandiva-cognition-tpu-r6-20260923/build_verification.json`](reports/FINAL_K8/gandiva-cognition-tpu-r6-20260923/build_verification.json).
It reports all F01–F24 passing for the existing 6,493,952-parameter K8
campaign, zero local optimizer updates, and source-closure SHA-256
`3fa4bc14f649741a80b34e7931e83228e232077aadf014746d450db89b3b50a1` at
`f7d10985eaee0dcaadbe1db5480e46f683cfe8ba`. The report records
`dirty=true` because the user's notebook and K8 test edits were preserved in
the worktree and excluded from the implementation commit. Runtime checks
G01–G04 remain pending. This verifier does not qualify the 100M TPU profile.

The current focused regression command passed **106 tests and 16 subtests**:

```powershell
python -m pytest -p no:cacheprovider tests/test_research_tpu.py tests/test_research_cognition.py tests/test_research_cognition_runtime.py tests/test_research_gandiva_rsi_cognition.py tests/test_research_k8_operational.py -q --maxfail=1
```

No local optimizer update was made, and no 100M model was instantiated.

Earlier focused checks on the TPU runtime and cognition changes passed 152
tests, one skip, and 16 subtests using a different test selection. The current
revision's exact command and result are recorded above; test totals are not
additive across the two selections.

## Cognition changes now in the branch

Finite-support belief revision now validates normalized support and exact
hypothesis alignment, refuses invalid likelihoods, and records a model
mismatch instead of silently resetting a posterior. Evidence ancestry is
resolved transitively so replaying a source or a derived copy is idempotent.
One revision accepts one new independent root, keeping each observation's
reliability auditable. Reliability tempers the likelihood as
`L_eff(h) = r * L(h) + (1-r) * mean_h(L(h))`; reliability zero records the
observation but leaves the posterior unchanged. Evidence and belief identities
include reliability, evidence status, and conflict lineage.

The rendered cognitive workspace now includes belief evidence/conflict links
and pending commitments. It rejects nested private provenance (including split,
source, family, and task-semantic identifiers). Its declared budget is enforced
against the serialized JSON-byte view: whole low-priority/old records are
removed in a deterministic order and counted in an omission receipt; an
irreducibly oversized goal frame is refused. Evidence is not silently cut in
the middle of a record.

These belief updates are still a host-side reference mechanism, not a learned
belief updater. The existing training repository also has a training-only,
exact depth-two information-gain teacher in
[`learning/inquiry/teaching.py`](../bramastra_lab/research/learning/inquiry/teaching.py).
It produces candidate query targets; the current branch has not established
that the learner acquires a transferable inquiry policy.

Each belief revision now also has an append-only audit entry containing its
prior, submitted and reliability-adjusted likelihoods, posterior, source
aliases, and outcome. Applied updates, duplicate/correlated evidence replays,
and finite-support model mismatches have distinct outcomes. Replays record a
no-op rather than multiplying evidence twice; model mismatches retain the
prior and the admitted evidence. The durable snapshot preserves private
ancestry; the bounded rendered view exposes only episode-local evidence
aliases. Restore validates contiguous revision indices and all referenced
belief/evidence IDs. This makes the reference reasoning path inspectable and
replayable; it does not cause the model to perform learned belief revision.

## TPU path implemented so far

`BuildConfig` contains a **configuration-only** `tpu_100m` profile with exact
analytic count 100,334,720 (decoder plus action/value heads), width 640, 15
layers, 10 heads, FFN 2,624, vocabulary 260, and context 512. The profile's
existence does not imply memory fit or usable campaign integration.

The new [`runtime/tpu.py`](../bramastra_lab/research/runtime/tpu.py) provides
fail-closed PJRT/topology checks, an eight-replica XLA backend, a
`torch_xla.launch` guard, a rank-specific input-loader builder, and explicit
master-parameter broadcast plus per-rank initialization receipts. After
broadcast, each worker can atomically write its rank, config identity, exact
state-dict SHA-256, and parameter count; the verifier refuses missing,
duplicate, malformed, or divergent rank receipts. This closes the
initialization-contract helper but is not yet invoked by a production TPU
campaign worker. `ProductionOps` now maps an XLA device to the BF16 autocast
mode required by `K8Trainer`; the old generic non-CUDA routing incorrectly
passed FP32 and failed during XLA trainer construction. A unit regression
covers XLA, CUDA, and CPU routing. Its
`DistributedSampler(drop_last=True)` avoids padding duplicates and keeps
replica shards equal; callers must call `sampler.set_epoch(epoch)`. The host
DataLoader defaults to retaining each rank's final partial batch. If the
dataset size is not divisible by replica count, the sampler omits the shuffled
remainder for that epoch; it must be accounted for in exposure reports.
`MpDeviceLoader` is treated as a prefetch/transfer wrapper, not as evidence of
input sharding.

The K8 trainer's replica path now:

1. all-reduces each objective's eligible-unit count;
2. scales each replica's local unnormalized objective sum by
   `replica_count / global_count`, so the mean gradient reduction equals one
   global sum divided by its correct denominator;
3. handles locally empty action/value heads only when those heads are active
   on another shard, while leaving globally inactive heads at `grad=None`;
4. computes the counterfactual pair objective from real own/swapped rows. The
   base objectives are backwarded and flushed first, then the pair forwards
   run, limiting simultaneous activation memory. Pair gradients are added
   before one replica gradient reduction, global clip, and optimizer boundary.

The pair-loss scaling is checked against an unequal-shard global-mean
reference; CPU doubles also test remote-only pair eligibility without calling
`optimizer.step`. These tests establish math/contracts, not XLA execution.
PyTorch/XLA documents `MpDeviceLoader` as a prefetch wrapper, PJRT worker
launch, XLA replica gradient reduction, and BF16 autocast. See the primary
references: [PyTorch/XLA migration guide](https://docs.pytorch.org/xla/master/learn/migration-to-xla-on-tpus.html),
[XLA AMP guide](https://docs.pytorch.org/xla/master/perf/amp.html), and
[XLA API guide](https://docs.pytorch.org/xla/master/learn/api-guide.html).

## Not ready for the 100M Kaggle run yet

The checked-in `notebooks/bramastra_k8.ipynb` is the existing two-T4 K8
campaign notebook. It does not launch a TPU worker or select `tpu_100m`.
The E1–E6 campaign ops also instantiate the frozen K8 profile, not the new
100M profile. The TPU sampler/backend helpers have contract tests but are not
yet connected to a production campaign dataset consumer. No Kaggle TPU session
has exercised the code, and no 100M forward/backward, memory fit, checkpoint
restore, or throughput result exists.

The current TPU backend deliberately requires construction inside
`launch_tpu_workers`. A worker must still call the new broadcast/receipt
helpers, construct the 100M trainer, build a cognitively meaningful sharded
training stream, bind the allocation, and publish a single coherent
checkpoint/result set from the replica group. The production data consumer
must shard complete optimizer windows, not merely wrap an unpartitioned
loader. The owner-facing Kaggle path must first run a no-update eight-core
preflight that measures peak memory, fixed-shape compilation, actual eligible
examples per update, checkpoint round-trip, and sustained throughput. Only
then can a measured update budget be chosen. Kaggle currently documents TPU
v3-8 and a nine-hour notebook session limit; schedule the preflight/training
and artifact export inside that cap, with export time reserved. See
[Kaggle TPU documentation](https://www.kaggle.com/docs/tpu) and
[Kaggle Notebooks documentation](https://www.kaggle.com/docs/notebooks).

The next engineering slice is therefore to wire a dedicated Kaggle TPU
preflight/launch path into a production consumer, then qualify it on Kaggle
with the registered 100M configuration before admitting optimizer updates.
Until that is done, the existing K8 build is verified, but the 100M TPU run is
**not ready**.
