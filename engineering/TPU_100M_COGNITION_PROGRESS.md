# 100M TPU cognition and training continuation

## 2026-09-23 engineering update

Gandiva now has a dedicated [`bramastra_tpu_100m_preflight.ipynb`](../notebooks/bramastra_tpu_100m_preflight.ipynb) and CPU-safe preflight implementation in [`campaigns/tpu_100m.py`](../bramastra_lab/research/campaigns/tpu_100m.py). On Kaggle it checks for an eight-replica PJRT TPU, validates the prepared bundle, deterministically assigns eight exact-training examples, verifies identical broadcast model state, and runs a real BF16 B-arm forward/backward with finite-gradient checks. The worker does not call `finalize_update`; it records zero optimizer updates and attempts. The notebook can clone Gandiva and prepare the full K8 bundle when no valid input bundle is attached, then packages the run directory for download.

See the [2026-09-23 implementation handoff](reports/GANDIVA_COGNITION_TPU_PREFLIGHT_20260923/HANDOFF.md) for the exact source revision, test evidence, limitations, and next Kaggle action.

CPU checks compiled eight examples from the existing validated 49,152-row bundle. The first integration pass caught and fixed a pair-denominator mismatch: each distinct-answer source pair yields two ordered comparisons, so the XLA trainer now receives a denominator of two. These checks do **not** instantiate the 100M model on this machine and do not qualify TPU memory or execution.

## Cognition input and memory update (2026-09-23)

The current Gandiva worktree adds a shared public-state v3 codec for training
and inference. Exact `inspect(variable)` / `observation(variable, value)`
pairs use a typed, reversible compact form. Other history shapes use the
general versioned codec, which preserves unknown fields. The instruction tag
and memory-event schema are included in the public-state identity, so
checkpoints and compiled-row sidecars cannot silently reuse an older prompt
protocol.

Memory retrieval budgets exact serialized events, skips records that cannot
fit, continues to lower-ranked candidates when they fit, and retains the
highest-ranked item when optional context must be omitted. Trace counts now
reflect records actually rendered rather than retrieval candidates. These are
host-side reference mechanics; they do not demonstrate learned long-term
memory or transfer.

The existing 49,152-row training bundle has been checked on CPU against the
512-token decision limit: zero compile errors, zero complete examples over
limit, maximum 464 tokens, median 336, and p95 464. All three families have
zero over-limit rows. The check performs no optimizer update or local model
training and does not qualify a Kaggle TPU run. The focused cognition/TPU
suite passes 141 tests and 24 subtests; exact scope and limits are in the
[public-state v3 handoff](reports/GANDIVA_COGNITION_PUBLIC_STATE_V3_20260923/HANDOFF.md).

The live cognition planner now conditions its depth-two world prediction on a separate hypothetical successor history containing the first action and predicted feedback. That synthetic history never enters the real episode trace. Search shares the remaining node/call budget, rotates partially covered root sets, and allocates second-depth nodes round-robin across roots. E2 calibration joins the executed first action to its depth-one prediction, never to a hypothetical child. `ModelWorldModel` reports input/output token counts; planner call and token use now appear in episode event metadata and budget accounting. Added behavioral tests cover history-conditioned prompts, simulated successor states, fair child expansion, remaining-budget behavior, call/token accounting, and selected-action calibration.

## Acceptance boundary

The changes above repair selected cognition and launch-path defects. They do not complete the broader cognition-foundation F1–F6 acceptance contract, establish that the random-weight model has useful cognition, or demonstrate recursive self-improvement. The 100M profile remains configuration-only until a real Kaggle TPU run verifies the registered architecture and this no-update backward path. No optimizer update, checkpoint round-trip, optimizer-state memory measurement, sustained throughput test, or multi-hour training campaign has occurred.

The owner should select **TPU v3-8** in Kaggle and enable Internet before running the dedicated notebook. The notebook runs a backward-only preflight, not full training. Kaggle's current [TPU documentation](https://www.kaggle.com/docs/tpu) and [Notebook documentation](https://www.kaggle.com/docs/notebooks) should be checked again before scheduling; the documented notebook session limit is nine hours, so leave time for packaging and download.

## K8 owner experiment context

The separate FINAL-K8 build remains the current two-T4 owner experiment; its
latest report and exact source identity are in **Current evidence** below.
E2's production workspace-policy path uses the typed cognitive evidence ledger
and records conflict/supersession state. The detailed K8 cursor is
[`FINAL_K8_PROGRESS.md`](FINAL_K8_PROGRESS.md). The central limitation remains:
a dedicated Kaggle zero-update backward-preflight consumer exists, but it has
not been run on Kaggle; no TPU qualification or optimizer-training campaign
exists.

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
duplicate, malformed, or divergent rank receipts. The dedicated
zero-update preflight exercises these helpers; a production optimizer-training
campaign worker does not yet use them. `ProductionOps` now maps an XLA device to the BF16 autocast
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

## Kaggle readiness boundary

The checked-in `notebooks/bramastra_k8.ipynb` is the existing two-T4 K8
campaign notebook. It does not launch a TPU worker or select `tpu_100m`.
The dedicated [`bramastra_tpu_100m_preflight.ipynb`](../notebooks/bramastra_tpu_100m_preflight.ipynb)
is the new Kaggle consumer for a **zero-update backward preflight**; it is
separate from the normal K8 notebook. No Kaggle TPU session has exercised it.
Therefore no 100M forward/backward, peak-memory fit, committed update,
checkpoint restore, or sustained-throughput result has been measured yet.

The preflight worker calls the replica parameter-broadcast and state-receipt
helpers, constructs the registered 100M model inside the eight-worker launch,
and runs one compiled B-arm forward/backward per replica. It does not exercise
an optimizer update or prove optimizer-state memory fit. The training campaign
still needs a cognitively meaningful sharded stream, complete optimizer-window
partitioning, allocation binding, atomic checkpoint/resume, and one coherent
replica result set. Before authorizing an update, separately measure peak
memory including optimizer state, fixed-shape compilation, eligible examples
per global update, checkpoint round-trip, and sustained throughput.

Kaggle documents TPU v3-8 and a nine-hour notebook-session limit. Reserve time
for artifact packaging and download within that cap. Check the live
[Kaggle TPU documentation](https://www.kaggle.com/docs/tpu) and
[Kaggle Notebooks documentation](https://www.kaggle.com/docs/notebooks)
before launching because quotas and runtime availability can change.

Next action: run the dedicated notebook on an attached Kaggle TPU v3-8 with
Internet enabled, preserve and download its ZIP plus SHA-256 receipt, and
review all eight worker reports. A pass qualifies only this no-update backward
path. The 100M training campaign remains **not ready** until memory,
optimizer-update, checkpoint/resume, data-sharding, and sustained-run gates
are implemented and evidenced. The existing two-T4 K8 experiment is a separate
owner run and is unaffected by this notebook.
