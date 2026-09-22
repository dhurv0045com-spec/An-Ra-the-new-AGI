# Signac 100M

Signac is the An-Ra research branch's **100M-class experiment package**. It reuses the existing Cymek V5.1 core and its already-listed M102 scale recipe. It does not introduce an unevidenced “AGI module,” new loss, router, recurrence, or memory mechanism. The research target is capability formation and transfer: whether the existing core acquires reusable operations, follows query changes, composes rules, generalizes under serialization changes, and preserves capabilities under later learning.

The executable contract is [`signac_100m/spec.py`](../../signac_100m/spec.py). Its exact V5 parameter receipt is **101,790,080**, not an even 100,000,000. The name “100M-class” denotes the planned scale rung; all run receipts must report the exact count. The architecture retains the 24,576-token byte-level BPE working choice and tied full-softmax output because the representation experiments are non-monotonic and have not established a production-optimal vocabulary.

This package is architecture-complete enough to instantiate the core and inspect its resource budget. It is **not authorized for a real training run**. The canonical Kaggle notebook, [`notebooks/SIGNAC_100M_KAGGLE_TPU_PREFLIGHT.ipynb`](../../notebooks/SIGNAC_100M_KAGGLE_TPU_PREFLIGHT.ipynb), discovers the runtime and emits a fail-closed report. It stops short of a TPU training claim until the TPU model, optimizer, checkpoint-resume, corpus, and evaluation gates pass.

## Contents

- [`ARCHITECTURE.md`](ARCHITECTURE.md): geometry, parameter receipt, and system boundaries.
- [`READINESS.md`](READINESS.md): gate-by-gate status and blockers before a production run.
- [`EVIDENCE_LEDGER.md`](EVIDENCE_LEDGER.md): cross-branch findings carried into the design, including negative and inconclusive results.
- [`TRAINING_PLAN.md`](TRAINING_PLAN.md): ordered qualification and launch gates.
- [`KAGGLE_TPU_RUNBOOK.md`](KAGGLE_TPU_RUNBOOK.md): notebook use and the evidence still required on target hardware.
- `python tools/signac_100m_preflight.py --target tpu`: local report; a blocked verdict is expected until external gates are attached.

No document here changes historical results or authorizes the 1.94B-token generic planning prior. Token count is a budget hypothesis, not a cognition target or readiness signal.
