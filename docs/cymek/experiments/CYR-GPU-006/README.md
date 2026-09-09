# CYR-GPU-006

CYR-GPU-006 is the current **READY_FOR_OPERATOR_COLAB_GPU_RUN** Cymek research campaign. It supersedes CYR-GPU-005 before execution and uses the real Cymek V5 implementation, the frozen 24,576 tokenizer, three independent arithmetic acquisition parents, true shared-parent matched retention forks, actual-token budgets, candidate-free generation, a hardware-only runtime resolver, Drive-backed stage durability, and failure-preserving evidence packaging.

The executable is frozen at `125b25c19204cce1994deebbfc4957119f2ae31f` and bound by `PREREGISTRATION.json`. Cell 0 copies the preregistration outside the repository, checks out that exact executable, verifies the frozen hashes and dependencies, runs deterministic preexecution checks, calibrates real CUDA training plus candidate-free generation, and resolves the largest affordable scientific proxy without looking at scientific outcomes.

The design incorporates the live Arkenstone Discovery V6 evidence at `6acd9dcbdd28d00f387ffcd004253a813aca4b66`: ARK-011 supports same-task adaptive protection at Micro scale; ARK-012 leaves the exact switch threshold unresolved; ARK-013 shows LOW alone does not solve no-replay cross-task interference; ARK-014 shows order augmentation is needed for robust binding acquisition. Therefore the non-arithmetic stage prospectively compares equal-age/equal-exposure `HYSTERETIC_HIGH_LOW` and `LOW_CONTINUE` states from the same parent, then trains both with the same HIGH-LR order-augmented binding stream plus fixed old-T2 replay.

`RUN_READINESS.json` is now mechanically true with no blockers. Dedicated CYR preexecution CI passes 25/25 deterministic tests and compiles all 3 notebook cells. The broader production suite passed 522 substantive tests with one historical closure-receipt self-check failure; that meta-only failure is repaired by the final receipt-only closure commit, without changing executable or test code.

Operator flow:

1. Open `notebooks/cymek_colab_gpu_research_v6.ipynb` in Google Colab and select a GPU runtime.
2. Run Cell 0 only. Continue only if it prints exactly `CYR-GPU-006 PREEXECUTION GATE: PASS`.
3. Run Cell 1. Google Drive is the durable stage/evidence store; completed stages are reused and incomplete arms restart from their immutable source state.
4. Run Cell 2 and return `CYMEK_GPU_RESEARCH_V6_RESULTS.zip` for post-run audit.

No CYR-GPU-006 scientific GPU result exists yet. No TPU run, PRE500M run, 500M campaign, production scheduler promotion, or 5B-corpus build is authorized by this preparation.
