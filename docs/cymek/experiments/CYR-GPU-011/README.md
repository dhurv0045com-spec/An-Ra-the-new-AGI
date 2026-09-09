# CYR-GPU-011

Long exposure-matched capability-emergence bridge for Google Colab GPU.

## Current state

Preexecution only. Do not run until this directory contains a hash-bound `PREREGISTRATION.json` and a `RUN_READINESS.json` with `ready_for_operator_colab_gpu_run=true`.

## Why V11 exists

CYR-GPU-009 and the superseded V10 design were semantically underexposed relative to the successful Arkenstone ARK-002B reference. V11 uses the exact ARK-002B frozen manifest and measures exposure in **semantic row presentations**, targeting the same 1,152,000-row box when wall time permits.

## Experiment order

1. `COMPACT_BRIDGE` — real Cymek V5 4L/128w, exact ARK data, 19-symbol arithmetic representation, **canonical Cymek causal objective**, max 25 minutes.
2. `PRODUCTION_PRIMARY` — same V5 geometry/data/objective with frozen 24,576-token tokenizer, receives the remaining science wall.
3. `PRODUCTION_REPLICATION` — only if primary reaches qualified G90 and at least 40 minutes remain.

The compact stage is a bridge, not an exact ARK reproduction: ARK-002B supervised an answer-prefix BOS while Cymek's canonical objective excludes BOS targets. V11 deliberately keeps Cymek's objective rather than changing production semantics.

Hardware calibration selects batch64/32/16 before outcomes. Target updates scale to preserve semantic dose: 18k/36k/72k respectively. A slower GPU does not trigger an all-or-nothing feasibility failure; the experiment records actual exposure and timeboxes honestly.

## Primary gate

Candidate-free exact answer with valid EOS stop, using a fixed eight-new-token cap. Controller G90 is three consecutive DEV_CONTROLLER evaluations >=.90. It is recorded but does not stop a bridge by itself: **qualified G90** additionally requires contemporaneous DEV_MEASUREMENT STANDARD >=.90. Final decision logic rechecks the larger measurement split.

Final controller/structural/sealed predictions are retained row-by-row with canonical SHA-256 prediction receipts. Milestone/final checkpoints retain their model/optimizer SHA-256 values and durable Drive paths.

## Diagnostics

STANDARD, COMMUTED, LOCALITY, CARRY, TRIPLE_ADD, THREE_DIGIT and VERBAL are reported separately. They characterize controlled transfer; they are not an aggregate reasoning/AGI score.

## Notebook

`notebooks/cymek_colab_gpu_research_v11.ipynb`

When eventually ready:

- use a fresh Colab GPU runtime;
- Cell 0 first requires `RUN_READINESS.json=true`, copies the preregistration outside the repo, checks out the exact frozen executable, verifies all bound blobs, reruns deterministic + one/two-update runtime contracts, then calibrates and resolves;
- Cell 0 must end with `CYR-GPU-011 PREEXECUTION GATE: PASS`;
- Cell 1 mounts Drive and performs the long run;
- Cell 2 verifies/downloads `CYMEK_GPU_RESEARCH_V11_RESULTS.zip`.

Expected Drive root: `/content/drive/MyDrive/CYMEK/CYR-GPU-011`.

## Evidence boundary

No result from this experiment alone can authorize broad reasoning claims, TPU equivalence, PRE500M, 500M training, production promotion, or 5B-corpus work.
