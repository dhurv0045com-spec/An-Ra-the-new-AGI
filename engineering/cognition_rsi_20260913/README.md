# BRAMASTRA cognition and recursive improvement extension

Chief design, 2026-09-13, after inspecting agent push `acfa249`. This extension makes cognition and learning-to-learn explicit parts of the [complete learner program](../master_program_20260913/README.md). The full assignment is now **M00–M24**. M19–M24 below extend the original M00–M18; they do not replace the shared decoder, data contracts, trainer or independent evaluator.

The intended system must do more than predict answers. It must maintain uncertain beliefs, bind goals to relevant evidence, plan and revise subgoals, select mental or external work, acquire reusable knowledge, and improve the procedure by which it learns. Those abilities have executable definitions in this packet. They are proposed capabilities to build and qualify, not properties already established by the branch name.

## Read this complete assignment

1. Root AGENTS and engineering STATUS.
2. The original [master program](../master_program_20260913/README.md) and all its linked design files.
3. [Cognition](COGNITION.md): working state, beliefs, reasoning, executive control and self-monitoring.
4. [Recursive improvement](RSI.md): meta-episodes, learning-rate-of-progress objectives, model-origin proposals and repeated generations.
5. [Learning and qualification](LEARNING_AND_EVALUATION.md): legitimate supervision, transfer and ablations.
6. [M19–M24 work orders](EXECUTION.md) and the [extension manifest](program.json).
7. The [new agent prompt](AGENT_PROMPT.md), which invokes the combined program.

## Branch review changes the immediate gate

The agent's B2.2 push contains substantial repair work. Chief review independently passed 60 non-training tests but reproduced prepared-identity, row-trainability and promotion defects, and confirmed resume/setup/API gaps in source. The [new chief review](../reports/B2_2_CHIEF_20260913/REVIEW.md) defines F1–F6. Close those under M00; the blanket “R1–R7 closed” claim is not accepted yet. Preserve the agent's repairs and existing evidence.

**Current resource override:** the live ledger at `acfa249` records 206/200 CPU updates and 177.234/300 learned-smoke seconds, zero GPU usage. The last phase used one failed update and two six-update comparisons. The old master packet's remaining-six-update instruction is exhausted and superseded. This extension authorizes **no optimizer updates or accelerator experiments**. Continue design, implementation, pure checks and bounded gradient wiring; any new learned execution needs a new owner allocation. Do not rerun the probe to recover its output.

## The central cognitive loop

`observe -> update beliefs -> bind the goal -> choose a cognitive operation -> predict/test/remember/plan -> check consequences -> revise or act -> store admissible experience`.

The decoder chooses the cognitive operation and its content once trained. Host code enforces schemas, evidence scope, resource accounting and tool execution. An if/else host policy solving the task is an engineered control, not model cognition. The initial random model will not become competent merely because these interfaces exist; the learning contracts specify how it could acquire the behavior.

## What RSI means in this repository

Ordinary practice changes weights. Continual learning adds skills while retaining earlier ones. Meta-learning improves adaptation to new tasks. Recursive improvement additionally lets a qualified learner propose and evaluate changes to its own learning procedure, then use the accepted successor to propose the next change. The evidence must show better **future learning at matched resources**, not just a more-trained checkpoint or an external coding agent's patch.

Start with a bounded typed learning-method language and isolated candidate lineages. Later source-patch proposals can use the same evidence gate, but are not silently enabled by the first implementation. The model may propose tests and design changes; it cannot certify itself by rewriting the examiner or redefining success. Larger changes remain possible through explicit reviewed contracts. This preserves an avenue to real method invention while keeping claims testable.

## Completion of this extension

M24 integrates cognitive sessions, meta-episode preparation, model-proposal provenance, candidate-method validation and repeated-generation transactions with the existing M18 operator flow. A build can demonstrate these connections with labeled fixtures and remain learned-unqualified. Its handoff must identify who actually chose each operation/proposal: the BRAMASTRA checkpoint, a fixed host rule, a symbolic teacher, a human, or an external implementation agent.
