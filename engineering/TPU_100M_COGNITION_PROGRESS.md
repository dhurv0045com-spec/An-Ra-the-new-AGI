# 100M TPU cognition and training continuation

## 23 September K8 cognition integration update

The separate FINAL-K8 build remains the current owner experiment. Its latest
zero-update report is
[`gandiva-cognition-runtime-r2-20260923/build_verification.json`](reports/FINAL_K8/gandiva-cognition-runtime-r2-20260923/build_verification.json):
F01–F24 pass for K8 with source closure
`cd4b02d77bbde04772989b6c8e8533a1d1d0c076a4d897e4d6289351aa69c8e6`, data
identity `6fb94b7018406632b0e62dcd23ca777046ae5d881363bb8e8c78d74139785fd6`,
zero local optimizer updates, and G01–G04 pending the actual two-T4 E0 gate.
E2's production workspace-policy path now uses the typed cognitive evidence
ledger and records conflict/supersession state. The detailed cursor is in
[`FINAL_K8_PROGRESS.md`](FINAL_K8_PROGRESS.md).

This does not change the limitation below: the `tpu_100m` profile and XLA
helpers still lack a dedicated Kaggle TPU campaign consumer and real TPU
qualification.

**Date:** 2026-09-23

**Branch:** `Gandiva`
**Purpose:** durable engineering cursor for the Kaggle TPU 100M path. This
document distinguishes code that exists from the still-unbuilt Kaggle run
path; it does not claim AGI, a successful TPU launch, or learned results.

## Current evidence

The latest zero-update FINAL-K8 verification is
[`gandiva-cognition-tpu-r3-20260923/build_verification.json`](reports/FINAL_K8/gandiva-cognition-tpu-r3-20260923/build_verification.json).
It reports all F01–F24 passing for the existing 6,493,952-parameter K8
campaign, zero local optimizer updates, and source-closure SHA-256
`4e32ad4460e11d6696136a714f25bccb3dc534bede39a4f2e5571183bd954d0d`.
Runtime checks G01–G04 are still pending. That verifier does not qualify the
100M TPU profile.

The focused regression command passed **152 tests, 1 skipped, and 16
subtests**:

```powershell
python -m pytest -p no:cacheprovider tests/test_research_tpu.py tests/test_research_cognition.py tests/test_research_config.py tests/test_research_k8.py tests/test_research_k8_operational.py tests/test_research_master_m05_m06_m12.py tests/test_research_gandiva_rsi_cognition.py -q --maxfail=1
```

The current development host reports 16 logical CPUs, 15.32 GiB total RAM,
2.87 GiB available RAM, no CUDA, and no installed PyTorch/XLA runtime. The
100M model was not instantiated or trained locally. No optimizer update was
made.

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

## TPU path implemented so far

`BuildConfig` contains a **configuration-only** `tpu_100m` profile with exact
analytic count 100,334,720 (decoder plus action/value heads), width 640, 15
layers, 10 heads, FFN 2,624, vocabulary 260, and context 512. The profile's
existence does not imply memory fit or usable campaign integration.

The new [`runtime/tpu.py`](../bramastra_lab/research/runtime/tpu.py) provides
fail-closed PJRT/topology checks, an eight-replica XLA backend, a
`torch_xla.launch` guard, and a rank-specific input-loader builder. Its
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
`launch_tpu_workers`; before a real run, the worker entry point still needs to
seed and verify identical initial parameters across replicas, construct the
100M trainer, build a cognitively meaningful sharded training stream, bind the
allocation, and publish a single coherent checkpoint/result set from the
replica group. The production data consumer must shard complete optimizer
windows, not merely wrap an unpartitioned loader. The owner-facing Kaggle path
must first run a no-update eight-core preflight that measures peak memory,
fixed-shape compilation, actual eligible examples per update, checkpoint
round-trip, and sustained throughput. Only then can a measured update budget
be chosen.

The next engineering slice is therefore to wire a dedicated Kaggle TPU
preflight/launch path into a production consumer, then qualify it on Kaggle
with the registered 100M configuration before admitting optimizer updates.
Until that is done, the existing K8 build is verified, but the 100M TPU run is
**not ready**.
