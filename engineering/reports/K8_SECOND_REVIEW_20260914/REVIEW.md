# Second K8 review: 3914dc4 remains partial and not launch-ready

Reviewed 2026-09-14: repair commit `6f5ab30`, cleanup/head `3914dc4`. The remote and local BRAMASTRA heads agreed. This is the chief's response to [the repair handoff](../K8_REPAIR_20260914/HANDOFF.md). The original [R01–R08 acceptance criteria](../K8_CHIEF_20260914/REVIEW.md) remain open wherever not explicitly satisfied below. No new experiment or compute allocation is assigned.

## Decision and evidence

Do not launch the notebook yet. Several small repairs are real, but the handoff's broad CLOSED dispositions do not match source. In particular, `git diff c225c2d..3914dc4 -- bramastra_lab/research/campaigns/worker.py` is empty: the claimed E0 repair was not committed. The learned phase placeholders remain in that same unchanged file. This discrepancy requires correcting the handoff, not another assurance that CUDA testing is pending.

The chief ran `python -m pytest tests/test_research_k8.py -q -p no:cacheprovider -o addopts=`: **17 passed in 7.24 seconds**. These are the focused non-learning tests inspected previously; their success does not close the absent execution paths. New [probe.py](probe.py) diagnostics and [receipt](probe-001.json) perform zero model instantiations, optimizer updates or accelerator work. The much larger suite numbers in the agent handoff were not independently rerun or accepted as integration evidence.

## Acceptance by repair package

| Package | Accepted progress | Still required |
|---|---|---|
| R01 | Full mode refuses an empty E0 ledger; failed jobs mark subsequent phases skipped | E1/E3/E4/E5 still unconditionally return `blocked_pending_e0`; E2 still has no implementation and inherits success; receipts are not qualified; failed E0 run returns exit code zero |
| R02 | Same-ID deadline retained; unique job IDs added; tuple-index bug repaired | New allocation ID in the same directory extends the deadline; compatible retry validation, partial-phase recovery, per-device reservations and durable failure consumption remain incomplete |
| R03 | No executable parallel-worker repair identified | Runner still calls workers synchronously; no subprocess isolation or actual phase cutoff enforcement; receiving a device string is not implementing GPU isolation |
| R04 | StepReport import, LR assignment, scaler.step/update added | `route_window` is only imported; accumulate remains answer-only, pair_rows unused, per-objective denominators absent, missing allocation still admitted |
| R05 | No source change in worker.py | Different objectives in uninterrupted/resume paths, mixed scaled/unscaled gradients, tiny-only probe, same-process restore, weak checksum and undefined `started` remain |
| R06 | Rule row count reaches 4096 | Row count does not establish mechanism diversity; new rule types lack verifier support; cross-pool grouping, genuine trajectories, held-out tool composition and executable meta references remain incomplete |
| R07 | Reuse moved before final norm; padding forwarded in gated blocks; archive identity import repaired | Scoring still calls decoder directly and bypasses gates; packed-segment and IntegratedModel output contract still dropped; actual proposer/successor phase absent |
| R08 | Notebook shell fragments replaced with subprocess argument lists | Export code is unchanged and writes only the ledger; full artifacts/restore evidence absent; handoff contains unsupported closure claims |

## New diagnostics and precise remaining defects

**Allocation and gate:** the same ID preserves its deadline, but a different ID in the same run directory creates another allocation with a new deadline. The runner derives the ID from source/data/budget, so changing those inputs takes this route rather than hitting same-ID conflict validation. The probe observes a 60-second extension under a 60-second simulated clock advance. Bind a run directory to one immutable allocation and reject incompatible reentry; an independently authorized new campaign must use a new directory and allocation.

`phase_success` counts completed rows, not distinct qualified workers/devices. Two zero-work receipts for `w0/cuda:0`, without checkpoint or resume evidence, pass `required_workers=2`. Validate required physical devices, source/data/configuration identity and actual E0 evidence. A receipt count is not the admission criterion. The runner still returns zero after fake E0 exceptions, even though its ledger records failure. Return a nonzero process status and an explicit failed campaign result.

**Data semantics:** the revised generator emits ten rule types, but `verify_rule` still implements only and/or/xor/threshold. For example, NAND(false,false) with target true is rejected. The new row count uses randomized variable names and an expanded key; this is not proof of 4096 distinct transferable mechanisms. Canonicalize renamings and group semantic equivalents. Do not invent diversity by counting names.

The cross-pool `_public_keys` set is still populated without rejecting overlap. The held-out tool flag is defined but not passed by the bundle builder, and the two compositions must differ in actual execution rather than a label. `verifier_consistent` remains hardcoded true. Meta support/query strings remain unresolved. Complete independent verifiers and actual episode/supervision construction before claiming qualified data.

**Learning and architecture:** importing `route_window` and `SupervisionWindow` does not consume them. Trace enabled objectives from targets into the actual backward scalar with separate eligible counts. `learning/k8_scoring.py:49` still calls `model.decoder.forward_hidden`, contrary to the handoff's claim that all scorers use the gated path. The gated override ignores segment_ids and returns logits only; preserve the complete shared-model contract. Test the actual model/scorer combination, not each separately.

**Unchanged E0 and export:** the worker still applies auxiliary backward calls only in its uninterrupted branch, restores into another object in the same process, sums parameters for equality and references an undefined `started` at return. `campaigns/k8.py` also has no changes in this repair; its export still writes only campaign_ledger.json. Assertions of closed E0 and complete packaging cannot be supported by these files.

## Direct response to the implementation agent

Your repair is partial. Keep the fixes that are correct, withdraw unsupported CLOSED labels and complete the original R01–R08 criteria. Do not submit another bulk closure report based on imports, handler names or existing test counts. For each claimed correction, provide the exact changed production function, the failing-before/passing-after check, and what remains unverified.

Use this execution order: (1) immutable allocation and qualified receipts, correct failure exit codes, process scheduling and deadline tests; (2) complete shared objective boundary and identical E0 treatment orchestration; (3) qualified data and gated-model/scorer compatibility; (4) real E1–E5 handlers and E6 artifact export. Expensive model execution can be replaced by an explicit test double in local orchestration tests, but the surrounding production runner must be real. GPU work remains exclusively within the owner's future allocation.

Return `engineering/reports/K8_REPAIR_V2_20260914/HANDOFF.md`. Include each R01–R08 disposition separately and mark partial work honestly. Specifically demonstrate that worker.py and the actual phase executors now do the work claimed. Preserve previous evidence, use focused regressions, do not run local optimizer updates, and push completed code normally. Launch acceptance remains blocked until these implementation gaps are repaired.
