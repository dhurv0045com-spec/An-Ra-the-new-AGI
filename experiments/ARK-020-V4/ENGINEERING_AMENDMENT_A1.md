# ARK-020 V4 — Durability Engineering Amendment A1

**Status:** IMPLEMENTED, validation pending  
**Date:** 2026-09-12  
**Amendment ID:** `ARK-020-V4-DURABILITY-A1`

## Scope

This is an engineering/durability amendment around the frozen ARK-020 V4 scientific
protocol. It does **not** change capability tasks, train/control/validation/sealed
splits, seeds, phase lengths, task-slot doses, qualification thresholds, Guardian
state semantics, replay treatment semantics, arms, verdict rules, or claim ceiling.

The frozen V4 scientific runner remains preserved. Operator execution is routed
through `run_ark020_v4_hardened.py`, which installs a narrow durability overlay.

## Why A1 exists

A fresh audit of live `Arkenstone` found that the repository-level readiness claim
was stronger than the executable evidence.

### A1-01 — exact phase-boundary resume defect

In the frozen runner, a checkpoint can be written at the final update of a phase
before the post-phase Guardian transition. On resume, `start_step == horizon`,
the inner phase loop is empty, and the post-phase code references loop-local
`gstep`. Guardian arms can therefore fail with an unbound variable at exact
B=2000 or C=1500 boundaries.

**Repair:** before entering the frozen runner, A1 verifies checkpoint scientific
identity, applies exactly the missing post-phase sparse-floor transition at the
deterministic phase-end global step, and advances the checkpoint to the next phase
(or finalization). The migration is atomic and writes `BOUNDARY_RESUME_A1.json`.

Prospective regression cases:

- B 1999: no migration; update 2000 remains to execute.
- B 2000: migrate once to C step 0.
- C 1499: no migration.
- C 1500: migrate once to D step 0.
- D 1499: no migration.
- D 1500: migrate once to finalization.

### A1-02 — partial identity was treated as resumable

The frozen scan classified missing receipt evidence as
`PARTIAL_IDENTITY_CHECK` but then returned `RESUME`. Required checkpoint fields
marked `MISSING` also did not necessarily force a stop.

**Repair:** the A1 scan is a second, fail-closed gate. Any missing/unreadable
required checkpoint evidence, missing acquired-parent receipt, missing/invalid
dose receipt, missing executable receipt, orphan partial without an exact
checkpoint, multiple active checkpoints, or partial identity yields:

`STOP — CHECKPOINT IDENTITY FAILURE`

Uncertainty is not promoted to PASS.

### A1-03 — executable identity was not campaign-bound

The notebook pin reduced risk, but the checkpoint/campaign evidence did not itself
prove which exact engineering executable was allowed to continue the run.

**Repair:** before campaign execution A1 creates exactly one immutable
`EXECUTABLE_IDENTITY_A1.json` using exclusive creation. It binds:

- git commit,
- V4 core SHA-256,
- frozen scientific runner SHA-256,
- A1 durability overlay SHA-256,
- hardened operator runner SHA-256,
- preregistration SHA-256.

Every later resume must match it exactly. A different commit or file hash fails
closed before the scientific runner mutates state.

### A1-04 — exact-resume smoke under-exercised controller state

The frozen production smoke directly tested model/optimizer/scaler/RNG and stream
equivalence, but the controller and registry were mostly static.

**Repair:** A1 composes the frozen production smoke with a second checkpoint
round-trip that crosses:

CONTROL observation -> formal degradation -> Guardian REPLAY32 activation ->
real replay selection -> capability-registry change -> checkpoint -> reload ->
continued controller observation.

The same production `save_checkpoint` / `load_checkpoint` path is used. A1 passes
only if both the frozen production equivalence and the transition fixture pass.

## Evidence preservation

Historical ARK-020 V1, V2, V3, and frozen V4 files are not rewritten by this
amendment. The operator wrapper is prospective. Any existing V4 campaign state
that lacks the A1 executable receipt is not silently adopted.

## Claim ceiling

Unchanged:

> controlled real-text-proxy evidence for a multi-capability continual-learning
> controller candidate.

No 500M training authorization. No broad continual-learning claim. No AGI claim.
Demonstrated AGI remains 0%.

## Run policy

Until A1 regression tests, the full inherited V4 suite, hardened CLI path, and
CUDA preexecution gates pass on the pinned A1 executable, the correct recommendation
is:

`DO_NOT_RUN`

A later readiness artifact may upgrade this only with exact commit/hash evidence.
