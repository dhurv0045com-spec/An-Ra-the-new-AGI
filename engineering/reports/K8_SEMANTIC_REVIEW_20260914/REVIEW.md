# Delivery review of 5c0567f: distinguish real operations from declared outcomes

The D1–D5 delivery adds useful runtime work and real phase modules. The phase modules, however, do not implement the scientific treatments they describe. Chief acceptance remains blocked by source defects before any accelerator run. This review and [the execution contracts](EXECUTOR_CONTRACTS.md) supersede the delivery handoff's claim that remaining work is hardware-only.

## Direct evidence

The [zero-learning diagnostic](probe.py) at reviewed source `5c0567f` produced [this receipt](probe-001.json): E5 returns completed and one committed update while a test boundary explicitly forbids training and receives zero training calls. E6 returns completed with no .pt payloads anywhere in the run directory. The checkpoint module has no publish_checkpoint symbol, which ProductionOps.save_checkpoint imports. These are executable counterexamples, not hypothetical concerns.

A small implementation-readiness gate now prevents the owner-facing runner from starting E0/full while the known phase defects remain. It is an explicit chief disposition, not a source-string scanner or a runtime claim that the architecture is good. Preparation and data validation remain available. Clearing the gate requires the corrected implementation and acceptance evidence; CUDA availability or successful imports do not clear it. This prevents spending the allocation on a campaign already known to be invalid.

## Phase findings

**E1 and shared operations:** execute defaults to four updates and worker dispatch supplies no frozen calibrated target. The development profile uses context 256 while the campaign specifies 512. ProductionOps creates a trainer with require_allocation=False. The production-type-name branch constructs auxiliary terms even for answer-only A; B enables pair supervision without supplying its required loss. The action candidates, zero-value regression and `result: ok` transition are invented fixture targets rather than targets from the dataset. Model-type branching makes the RecordingDouble take a materially different path from production. The fixed generation prompt does not evaluate held-out mechanism success. `_load_stream` also admits training-controller names through substring matching and can synthesize fallback rows. Repair these before interpreting any loss as an experiment.

**E2:** the executor names an E1 parent but calls random initialization instead of loading its payload. It generates from `[259, case]`, invents action/call/node counts from case numbers and records no actual environment transitions or task-success results. It does not instantiate the declared cognitive comparison arms. Naming a checkpoint `frozen-...` cannot bind an untrained model to E1.

**E3:** parent and child are newly initialized models, not E1-B restoration. Both T0/T1 enable the same objectives; missing extra losses prevent actual training under the strict trainer. The tool stream mixes tool-training and held-out rows and overwrites provenance as training. There is no replay stream; the receipt claims 50/50 rather than the specified T1 75/25. Verification builds its observation from the saved correct answer instead of a tool execution receipt. This is answer checking, not learned tool control or retention measurement.

**E4:** migration is checked on a separate tiny model that is discarded. The actual optimized handle is a new base development model for both S0 and S1. Consequently `gates_enabled: true` in a receipt does not describe the trained model. The actual E1-B parent is not restored, and enabled auxiliary objectives have no supplied terms.

**E5:** `_build_archive` calls run_fixture_generation with confirmed=True. The main executor applies recipes to `_FakeTrainer`, never trains P0/P1/P_fixed, never uses a generated choice to update a successor, and returns task-count-derived committed updates. P_fixed is labeled with M2 rather than the specified M0. The anchor comparison compares a recipe with itself; future_outcomes_seen is a constant false, not an information-boundary check. These routines cannot produce RSI evidence even when ProductionOps is selected.

**E6:** five JSON filenames and nonempty checkpoint IDs are accepted without loading any parent payload. Hashing whatever files currently exist is not comparison against an independently bound expected artifact manifest. The top-level export implementation still exports metadata only. It must preserve actual recoverable state and explicitly report incomplete failed-run export separately from successful full-campaign export.

## Immediate implementation assignment

Keep the correct slot, process termination, physical/local device and lease improvements. Complete [EXECUTOR_CONTRACTS.md](EXECUTOR_CONTRACTS.md) in stages. First prove real checkpoint save/load and production E1 objective construction with optimizer steps replaced only at the final boundary. Then implement parent-based cognition/tools/architecture. Finally implement measured RSI and complete export. Do not replace missing work with a permissive double, declared success or fabricated counters.

Return `engineering/reports/K8_REAL_EXECUTION_20260914/HANDOFF.md`. Each phase needs its real called symbols, source/data/configuration identities, failing-before/passing-after integration evidence and precise pending GPU checks. No local optimizer updates, no paid compute and no reset of the old ledger. Only after all phase criteria are supported should the chief's readiness dispositions be updated. Do not add a skip-readiness flag to make the notebook run.
