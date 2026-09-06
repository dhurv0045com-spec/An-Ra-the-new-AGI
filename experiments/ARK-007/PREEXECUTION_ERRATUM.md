# ARK-007 PRE-EXECUTION ERRATUM

Commit `bddd45f` contained `experiments/ARK-007/run_007.py` but no `PLAN.md`.
The commit message called it a "preregistration" but it was an incomplete
code freeze with missing imports and a flawed replication criterion.

## Defects in the prior runner

1. Missing imports (`CompactVocab`, `Micro`, `loss_and_positions`, `greedy_exact`)
2. Conflated G90 onset (first eval >= 0.90) with confirmation (the later step
   at which the 3-eval streak becomes knowable)
3. Required every new high-LR control seed to collapse, despite ARK-005
   already establishing that controls can naturally remain stable
4. No paired continuation-order design
5. No pre-generated batch-index sequences
6. No parameter-displacement tracking
7. Used hardcoded `batch * 10` for token accounting

## Resolution

No ARK-007 results existed at the time of this erratum. The binding
preregistration is the NEW commit containing `PLAN.md` and the corrected
runner. The prior `run_007.py` is superseded.
