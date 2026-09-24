# Signac 100M

Phase-One evidence and claim limits are defined in [PHASE1_TWIN_CONTRACT.md](PHASE1_TWIN_CONTRACT.md). The Kaggle free TPU qualification procedure is in [KAGGLE_TPU_RUNBOOK.md](KAGGLE_TPU_RUNBOOK.md). A successful all-core canary remains engineering evidence and cannot authorize research training.

The development trainer now combines generated causal-cognition exercises with 81 structured experiment-ledger records and the curated Signac evidence index. Source status and claim limits remain attached, and the bundle records hashes for its 2026-09-13 cross-branch snapshot and current Signac index. This helps condition the Phase-One research model on the repository's own findings; it does not qualify the production corpus or create capability evidence.

The updated E0 evaluator has a versioned [development certificate](../../artifacts/e0/development_certificate_e0_eval_0_7_0.json) for generator `e0-eval/0.7.0`: 440 canonical development cases, all nine training families represented, family-specific faithful-realization and interference tasks, and a 64-seed, per-cell interference shortcut screen with balanced target positions. Its scope is development infrastructure only; the certificate explicitly says that no model was evaluated and the sealed seed remains under independent custody.

Signac is the An-Ra research branch's **100M-class experiment package**. It reuses the existing Cymek V5.1 core and its already-listed M102 scale recipe. It does not introduce an unevidenced “AGI module,” new loss, router, recurrence, or memory mechanism. The research target is capability formation and transfer: whether the existing core acquires reusable operations, follows query changes, composes rules, generalizes under serialization changes, and preserves capabilities under later learning.

The executable contract is [`signac_100m/spec.py`](../../signac_100m/spec.py). Its exact V5 parameter receipt is **101,790,080**, not an even 100,000,000. The name “100M-class” denotes the planned scale rung; all run receipts must report the exact count. The architecture retains the 24,576-token byte-level BPE working choice and tied full-softmax output because the representation experiments are non-monotonic and have not established a production-optimal vocabulary.

This package can instantiate and update the exact core locally, but it is **not authorized for a real research training run**. Run `python tools/signac_100m_model_smoke.py --candidate m102_primary --device cpu --workdir <output-dir>` to exercise one synthetic-token forward/backward/update and confirm model/optimizer mutation. The optional `--verify-checkpoint` also runs a single-process save/restore/next-update comparison. The canonical Kaggle notebook, [`notebooks/SIGNAC_100M_KAGGLE_TPU_PREFLIGHT.ipynb`](../../notebooks/SIGNAC_100M_KAGGLE_TPU_PREFLIGHT.ipynb), runs bounded synthetic updates through all eight expected Kaggle TPU workers and emits per-rank receipts plus a fail-closed static report. These plumbing checks stop short of a TPU training claim until measured target fit, numerical parity, exact multiprocess resume, corpus, and evaluation gates pass.

## Contents

- [`ARCHITECTURE.md`](ARCHITECTURE.md): geometry, parameter receipt, and system boundaries.
- [`READINESS.md`](READINESS.md): gate-by-gate status and blockers before a production run.
- [`PHASE1_TWIN_CONTRACT.md`](PHASE1_TWIN_CONTRACT.md): measurable completion predicates and the boundary of the Phase-One claim.
- [`EVIDENCE_LEDGER.md`](EVIDENCE_LEDGER.md): cross-branch findings carried into the design, including negative and inconclusive results.
- [`TRAINING_PLAN.md`](TRAINING_PLAN.md): ordered qualification and launch gates.
- [`KAGGLE_TPU_RUNBOOK.md`](KAGGLE_TPU_RUNBOOK.md): notebook use and the evidence still required on target hardware.
- `python tools/signac_100m_preflight.py --target tpu`: local report; a blocked verdict is expected until external gates are attached. After a Kaggle canary, add `--runtime-receipt <m102_primary/aggregate.json>` and keep its eight rank receipts beside it; add `--qualification-receipt <qualification/m102_primary/aggregate.json>` when the 20-update profile ran. The preflight rebuilds each aggregate from its rank files and records hashes, but never treats synthetic evidence as production authorization.

No document here changes historical results or authorizes the 1.94B-token generic planning prior. Token count is a budget hypothesis, not a cognition target or readiness signal.
