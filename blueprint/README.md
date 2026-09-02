# An-Ra V5 execution blueprint

This directory is the single human-facing authority for the V5 Core. Code and
receipts remain in their executable packages; this index prevents duplicated
constants and competing documents.

## Read in this order

1. [`V5_TRAINING_SPEC_v1.0.md`](V5_TRAINING_SPEC_v1.0.md) — exact Core,
   tokenizer, data, cognition, optimization, topology, checkpoint, and
   evaluation constants.
2. [`V5_MASTER_BLUEPRINT.md`](V5_MASTER_BLUEPRINT.md) — scientific rationale,
   evidence classes, and unresolved hypotheses.
3. [`IMPLEMENTATION_BLUEPRINT.md`](IMPLEMENTATION_BLUEPRINT.md) — package and
   infrastructure boundaries.
4. [`BENCHMARK.md`](BENCHMARK.md) — measurement and anti-shortcut contract.
5. [`EXPERIMENTS.md`](EXPERIMENTS.md) — smallest high-information experiments.
6. [`EXECUTION.md`](EXECUTION.md) — exact operator sequence.
7. [`LAUNCH_GATES.json`](LAUNCH_GATES.json) — machine-readable evidence slots.

Governance is in [`DECISIONS.md`](DECISIONS.md),
[`DECISION_LOG.md`](DECISION_LOG.md), [`FREEZE_CHECKLIST.md`](FREEZE_CHECKLIST.md),
[`OPEN_QUESTIONS.md`](OPEN_QUESTIONS.md), and [`STATUS.md`](STATUS.md).

## Executable authorities

- Constants: [`../v5_contracts/training_spec.py`](../v5_contracts/training_spec.py)
- Generated constant receipt:
  [`../artifacts/v5/training_spec_v1.json`](../artifacts/v5/training_spec_v1.json)
- Launch gate evaluator:
  [`../v5_contracts/launch_readiness.py`](../v5_contracts/launch_readiness.py)
- Training state/checkpoints: [`../v5_training/`](../v5_training/)
- E0–E3 harnesses: [`../e0_cognition/`](../e0_cognition/),
  [`../e1_tokenizer/`](../e1_tokenizer/),
  [`../e2_architecture/`](../e2_architecture/), and
  [`../e3_data_objective/`](../e3_data_objective/)

The present package is **ready for local contract checks and prelaunch work**.
The learned E1–E5 runners and production trainer are not implemented yet.
The readiness command inventories evidence; it never authorizes the 250M/5B
main run or replaces independent scientific and custody review.
