# CYR-GPU-006

CYR-GPU-006 is **SUPERSEDED_AFTER_PREEXECUTION_CALIBRATION_BEFORE_SCIENTIFIC_EXECUTION**; do not launch it. Its historical operator notebook was removed from the active `notebooks/` list. The supersession record and immutable preregistration remain available here.

The executable is frozen at `125b25c19204cce1994deebbfc4957119f2ae31f` and bound by `PREREGISTRATION.json`. Cell 0 copies the preregistration outside the repository, checks out that exact executable, verifies the frozen hashes and dependencies, runs deterministic preexecution checks, calibrates real CUDA training plus candidate-free generation, and resolves the largest affordable scientific proxy without looking at scientific outcomes.

The design incorporates the live Arkenstone Discovery V6 evidence at `6acd9dcbdd28d00f387ffcd004253a813aca4b66`: ARK-011 supports same-task adaptive protection at Micro scale; ARK-012 leaves the exact switch threshold unresolved; ARK-013 shows LOW alone does not solve no-replay cross-task interference; ARK-014 shows order augmentation is needed for robust binding acquisition. Therefore the non-arithmetic stage prospectively compares equal-age/equal-exposure `HYSTERETIC_HIGH_LOW` and `LOW_CONTINUE` states from the same parent, then trains both with the same HIGH-LR order-augmented binding stream plus fixed old-T2 replay.

`RUN_READINESS.json` is now mechanically true with no blockers. Dedicated CYR preexecution CI passes 25/25 deterministic tests and compiles all 3 notebook cells. The broader production suite passed 522 substantive tests with one historical closure-receipt self-check failure; that meta-only failure is repaired by the final receipt-only closure commit, without changing executable or test code.

No CYR-GPU-006 scientific GPU result exists. Its calibration finding is
engineering feasibility evidence only; it authorizes no TPU, PRE500M, 500M,
production scheduler, or corpus claim.
