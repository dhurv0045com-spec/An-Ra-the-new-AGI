# ESOES — An-Ra V5 Cognition-First Foundation Lab

ESOES is a clean-sheet research branch for designing the next An-Ra neural Core. Its Git ancestry passes through `core-vnext`, but V4, VNext, PGE, SFT, and EXP are evidence sources—not inherited implementation.

Start with [`blueprint/README.md`](blueprint/README.md).

Current state: **V5 contracts, local canaries, and experiment plans are executable; learned E1–E5 runners and the production trainer still require implementation. `python -m v5_contracts.launch_readiness --output artifacts/v5/launch_readiness.json` checks the evidence inventory. It never authorizes the main 250M/5B run.**

Previous-system evidence remains available at its original immutable branch/commit paths, especially:

- `core-vnext@054619f` — canonical PGE audit and token-provenance evidence;
- `core-vnext@4ee180a` — latest hardened VNext reference infrastructure;
- `core-exp@51124de` — latest EXP causal-policy evidence;
- `core-frozen-v4@f72f193` — frozen V4 reference.

Nothing in those branches is silently treated as V5 code.
