# V.153 — Elendril completion workstream

**Label: V.153 (owner correction).** This is a continuation/completion workstream built on the completed V7.1 core, not a new proven neural version and not a replacement of V7.1's historical evidence.

- [Research completion plan](COMPLETION_PLAN.md)
- [Existing V7.1 core results](../../Experiment%20Result/V7.1/20c80f66e76e4b589aed2053afbddffc/RESULTS.md)
- [Independent V7.1 audit](../../Experiment%20Result/V7.1/RESEARCH_REVIEW_2026-11-09.md)
- [Existing V7.1 paired diagnostic design](../v7.1/NEXT_EXPERIMENT.md)
- [V.153 evidence status](../../Experiment%20Result/V.153/STATUS.json)

The initial V.153 source contribution is a CPU-auditable anti-shortcut route benchmark at anra_v5/elendril/v153_diagnostics.py and tests/test_elendril_v153_diagnostics.py. It covers 1–4 hops with reverse order, shifted timestamps, removal/replacement of distractors and removal of required evidence. It does not train or evaluate a neural checkpoint. GitHub CI passed for the benchmark and portable preparation CLI (356 pytest tests and 14 subtests, run 42577968675); this qualifies engineering tests, not neural capability.

For CPU-only, non-neural preparation, from the repository root:

    python -m tools.qualify_elendril_v153 --output /outside-repo/v153-prepared --train /path/to/real-v71-train.json

Omit --train only when the original training dataset is unavailable; the resulting report then explicitly labels training disjointness NOT_RUN. The tool refuses to overwrite an existing output and writes a portable evaluator-side suite and a SHA-bound audit receipt. The portable file contains scoring truth and must never be used directly as a model-visible prompt.

**Remaining execution:** frozen restored-checkpoint diagnostics, memory/action follow-ups, 12 larger-model S8/S20 training jobs, nine continual-learning jobs and optional old-V7 frozen evaluation. The owner operates any Kaggle GPU sessions after verified engineering qualification; completion is assessed from archived evidence, not from a document or a green CPU CI badge.

Codex is invited to improve or challenge the proposal. Preserve source lineage, no silent result or protocol rewrites, and do not rename prior V7.1 scores as fresh V.153 experiments.
