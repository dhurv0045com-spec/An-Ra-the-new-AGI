# BRAMASTRA — An-Ra research from first principles

**Agents start at [AGENTS.md](AGENTS.md), then [engineering headquarters](engineering/README.md) and [current status](engineering/STATUS.md).** The chief owns architecture, algorithms and experiments; execution agents receive substantial [work orders](engineering/work_orders/README.md) with interfaces, dependencies, file ownership and acceptance evidence. The active system is designed directly for BRAMASTRA.

The [full AGI blueprint](AGI_BLUEPRINT.md) states the broader objective: intelligence trained from scratch that investigates, plans, consolidates skills and improves under independent evaluation. The [engineering specifications](engineering/SYSTEM_ARCHITECTURE.md) define the current implementation program.

The new [discovery prototype review](engineering/reports/PROTOTYPE_REVIEW.md) records executable rule-learning, learned inquiry, acquisition/replay comparisons and two development seeds. It also records unresolved retention failures. This is an experimental system, not an achieved AGI.

[BRAMASTRA.md](BRAMASTRA.md) preserves the initial experiment plan; the [research-loop note](docs/bramastra/RESEARCH_LOOP.md) records its earlier outline.

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
