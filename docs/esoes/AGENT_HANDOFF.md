# ESOES Agent Handoff

Execution/review agents working on this branch should follow these rules:

1. Read `ESOES.md`, `EVIDENCE_AND_CONTEXT.md`, `V5_COGNITION_FIRST_BLUEPRINT.md`, `OPEN_QUESTIONS.md`, and `FREEZE_CHECKLIST.md` before proposing training code.
2. Do not silently choose unresolved architecture/data/tokenizer/optimizer values.
3. When an unresolved choice blocks implementation, propose the cheapest discriminating experiment and its metric.
4. Preserve negative evidence.
5. Separate raw Core capability from runtime-assisted capability.
6. Never use sealed promotion fixtures for tuning.
7. Do not modify `core-vnext` or `core-exp` from this branch unless explicitly instructed.
8. Treat later V5 specification documents as versioned contracts; deviations require an explicit update rather than an implicit code change.

Agents have freedom to challenge the blueprint. They do not have freedom to hide a redesign inside implementation.