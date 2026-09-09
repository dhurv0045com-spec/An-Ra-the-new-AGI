# CYR-GPU-011

Long exposure-matched capability-emergence bridge for Google Colab GPU.

## Current state

Preexecution only. Do not run until this directory contains a hash-bound `PREREGISTRATION.json` and a `RUN_READINESS.json` with `ready_for_operator_colab_gpu_run=true`.

## Why V11 exists

CYR-GPU-009 and the superseded V10 design were semantically underexposed relative to the successful Arkenstone ARK-002B reference. V11 uses the exact ARK-002B frozen manifest and measures exposure in **semantic row presentations**, targeting the same 1,152,000-row box when wall time permits.

## Experiment order

1. `COMPACT_BRIDGE` — real Cymek V5 4L/128w, 19-symbol arithmetic representation, ARK-style supervised answer-prefix BOS, max 25 minutes.
2. `PRODUCTION_PRIMARY` — same V5 geometry with frozen 24,576-token tokenizer and normal Cymek rendering, receives the remaining science wall.
3. `PRODUCTION_REPLICATION` — only if primary reaches qualified G90 and at least 40 minutes remain.

Hardware calibration selects batch64/32/16 before outcomes. Target updates scale to preserve semantic dose: 18k/36k/72k respectively. A slower GPU does not trigger an all-or-nothing feasibility failure; the experiment records actual exposure and timeboxes honestly.

## Primary gate

Candidate-free exact answer with valid EOS stop. Sustained G90 requires three DEV_CONTROLLER evaluations >=.90, and a final claim additionally requires DEV_MEASUREMENT STANDARD >=.90.

## Diagnostics

STANDARD, COMMUTED, LOCALITY, CARRY, TRIPLE_ADD, THREE_DIGIT and VERBAL are reported separately. They characterize controlled transfer; they are not an aggregate reasoning/AGI score.

## Notebook

`notebooks/cymek_colab_gpu_research_v11.ipynb`

When eventually ready:

- use a fresh Colab GPU runtime;
- Cell 0 verifies the frozen executable, runs deterministic tests, calibrates and resolves;
- Cell 0 must end with `CYR-GPU-011 PREEXECUTION GATE: PASS`;
- Cell 1 mounts Drive and performs the long run;
- Cell 2 verifies/downloads `CYMEK_GPU_RESEARCH_V11_RESULTS.zip`.

Expected Drive root: `/content/drive/MyDrive/CYMEK/CYR-GPU-011`.

## Evidence boundary

No result from this experiment alone can authorize broad reasoning claims, TPU equivalence, PRE500M, 500M training, production promotion, or 5B-corpus work.
