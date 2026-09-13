# BRAMASTRA EXECUTION LESSONS (inspect, don't absorb)

**Sources inspected (branch BRAMASTRA @ `bab6ca5`):** `engineering/STATUS.md`, `engineering/master_program_20260913/`, `engineering/reports/B2_CHIEF_20260913/REVIEW.md`, `engineering/reports/B2/HANDOFF.md`. BRAMASTRA's own status: M00–M18 is *written, not implemented or learned*; the B2/B2.1 build is not accepted as fully integrated after the chief audit; build ledger 193/200 CPU updates, zero GPU; real corpus DATA_NOT_READY.

## General engineering lessons adopted (mechanism-free)

1. **Chief-review discipline:** BRAMASTRA's chief independently ran 63 non-training tests and reproduced defects *without a model* — integration defects surface without expensive compute. Task 3 adopts the same posture: the entire CPU qualification suite runs before any GPU minute (§42).
2. **State continuity as an explicit gate:** the chief audit flagged state-continuity and evaluator gates as unclosed. Task 3's runner treats state continuity as a first-class receipt: TrainingState sha256, sampler cursor, and RNG restoration are compared bitwise across processes, not asserted.
3. **Receipt semantics — "never forgeable success":** BRAMASTRA B09 recomputes exact+EOS scoring rather than trusting a success field. Task 3's evaluation recomputes exact-with-valid-EOS from decoded text and stop reason; no self-declared pass flags.
4. **Failure-recovery is a documented procedure:** the stash/staging lessons (staging generation must be resolved after a crash; last-known-good preserved) are load-bearing here too — Task 3's crash-injection test exercises exactly this path against the production CheckpointStore.
5. **Worktree hygiene:** BRAMASTRA's handoff documents worktree/branch containment in detail (which checkout, which baseline, no force-push). Task 3 keeps the same containment discipline: single branch, no cross-branch merges.
6. **DATA_NOT_READY honesty:** BRAMASTRA refuses to run past a data-not-ready state. Task 3's `scan` returns explicit safe actions (START/RESUME/COMPLETE/FAIL_CLOSED) for the same reason.

## Explicitly NOT imported

- planner and imagined-step machinery (B10);
- action/value heads and their training targets;
- candidate transactions, inquiry/collection loops, memory architecture;
- the qualification-gated controller and clustered-bootstrap mechanisms (B2.1) — these are BRAMASTRA research hypotheses, unqualified by its own admission.

V5.1 imports **no** BRAMASTRA learning mechanism. The canary's only overlap is contract-level: answer+EOS supervision, fail-closed checkpoints, recomputed scoring — each independently evidenced in the Task-1 ledger.
