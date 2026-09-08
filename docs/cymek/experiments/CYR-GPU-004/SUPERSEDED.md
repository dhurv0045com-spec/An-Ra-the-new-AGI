# CYR-GPU-004 — SUPERSEDED_BEFORE_EXECUTION

Date: 2026-09-08. Superseded by **CYR-GPU-005** before any GPU execution.

## Why

An independent pre-execution audit (`PREEXECUTION_AUDIT.md`, 13 defects)
was followed by a second hard audit of the corrected runner
(`experiments/ARK-007/run_v4.py`) against the CYR-GPU-005 fork contract.
That audit confirmed the runner's central defect was NOT repaired by the
v4 corrections:

- **No shared parent.** `run_arm(seed, lr, ...)` performs acquisition AND
  continuation inside one process for a single `(seed, lr)` pair. Running
  HIGH and LOW means two completely independent experiments
  (independent init, independent data stream): LOW is applied DURING
  acquisition. This is not the Arkenstone retention experiment and does
  not isolate post-acquisition LR.
- **Dead snapshot.** The G90 checkpoint (`snap_model` / `snap_opt`) is
  built at confirmation and never restored by any fork; the optimizer
  "copy" is a shallow dict aliasing live tensors.
- **No future-stream identity.** Forks would consume per-run
  `torch.randint` streams; no tail hashing, no cross-arm equality test.
- Additional unresolved: hardcoded `MANIFEST_SHA` (rows never verified
  against it), hardcoded EOS=3, silent CPU fallback, step-based (not
  actual-token) targeting, one conflated test set for control + claim,
  and a 6L/256w proxy labelled "P35-proxy".

## Disposition

- Per section 45/2 discipline this identity is NOT patched further.
- All files preserved as historical evidence, including the untracked
  `experiments/COLAB/build_004.py` builder.
- The corrected design lives in `../CYR-GPU-005/` with the
  shared-parent / shared-future-tail contract implemented and locally
  proven (plumbing E2E, parent-equivalence and tail-equality receipts).
