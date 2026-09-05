# BRAMASTRA — An-Ra research from first principles

Start with [BRAMASTRA.md](BRAMASTRA.md) and [the target research loop](docs/bramastra/RESEARCH_LOOP.md): AGI as the long-term objective, all core weights trained from scratch, and experiments planned around approximately 100 owner-reported Kaggle TPU-hours per week.

The first [executable experiment](bramastra_lab/README.md) now trains a small random-initialized core, compares terminal supervision, measures fresh-world failures, and verifies local checkpoint continuation. [Results and limitations](docs/bramastra/RESULTS.md) distinguish demonstrated answer learning from unresolved query-sensitive transfer. Historical executable contracts remain unchanged.

The ESOES overview below is preserved as historical context. Its architecture choices and phase ordering are not automatically adopted by BRAMASTRA.

## Historical ESOES overview

ESOES is a clean-sheet research branch for designing the next An-Ra neural Core. Its Git ancestry passes through `core-vnext`, but V4, VNext, PGE, SFT, and EXP are evidence sources—not inherited implementation.

> **Read [`AN_RA_PROGRAM.md`](AN_RA_PROGRAM.md)** — the complete research guide: what was proved, what was falsified, the causal decomposition results, the non-negotiable rules, and what comes next.

Start with [`blueprint/README.md`](blueprint/README.md).

Current state: **V5 contracts, local canaries, and experiment plans are executable; learned E1–E5 runners and the production trainer still require implementation. `python -m v5_contracts.launch_readiness --output artifacts/v5/launch_readiness.json` checks the evidence inventory. It never authorizes the main 250M/5B run.**

Previous-system evidence remains available at its original immutable branch/commit paths, especially:

- `core-vnext@054619f` — canonical PGE audit and token-provenance evidence;
- `core-vnext@4ee180a` — latest hardened VNext reference infrastructure;
- `core-exp@51124de` — latest EXP causal-policy evidence;
- `core-frozen-v4@f72f193` — frozen V4 reference.

Nothing in those branches is silently treated as V5 code.
