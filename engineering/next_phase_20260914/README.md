# Next major phase: an operational cognitive learner

Chief assignment after review of **44e3905**, 2026-09-14. This phase connects the implemented contracts into a learner that can act on evidence, acquire skills, retain earlier competence and choose improvements whose effects can be measured. AGI remains the research objective. This milestone does not claim to establish AGI; it creates the operational and experimental basis for deciding whether the proposed mechanisms help.

Read [the branch assessment](ASSESSMENT.md), [architecture and algorithms](ARCHITECTURE.md), [execution packages](EXECUTION.md) and [experiment readiness](READINESS.md). Root [experiment.md](../../experiment.md) remains authoritative for the seven K8 experiment phases, treatments, data requirements and statistical interpretation. These documents are the current implementation assignment; older packets provide background where consistent.

## What changes in this milestone

The branch has made real progress on exact lineage, collision-free checkpoint names, real supervision channels, strict evidence kinds and parent restoration. Preserve these pieces. The central remaining problem is that important contracts stop at interfaces: a reservation is not attached to the optimizer, cognitive modes are labels on the same generation path, and production RSI trial measurement still refuses rather than trains and evaluates.

Finish the system in dependency order: authorized learner session -> complete update/checkpoint lifecycle -> environment-driven cognitive loop -> parent-based acquisition and retention -> measured learning-method trials -> learned successor selection -> complete experiment export. A change downstream must consume the actual artifact produced upstream. There must be no equivalent-looking reconstruction from a name, seed or task count.

The primary deliverable is **one runnable K8 notebook backed by the repository**, with a prepared-data recipe and a verified launch protocol. Alongside it, deliver a traceable internal architecture that later work can extend toward longer-horizon memory, richer tools and broader transferable tasks. Avoid building separate demo pipelines for those ambitions.

## Ownership and autonomy

The chief owns design, integration decisions and experiment acceptance. Implementation agents own code and focused execution evidence. One integrator owns shared interfaces. Luna can handle independent packages with explicit path ownership; merge dependencies before changing their consumers. Package estimates of roughly 3–5 hours indicate substantive engineering scope, not a requirement to pad time. Finish acceptance criteria rather than chasing line or token counts.

There is no new local optimizer allowance or additional accelerator session. The owner will run the planned two-T4 campaign. Engineering, preparation, backward checks, non-updating checkpoint round trips and process-control tests can proceed without waiting for GPUs. The readiness gate stays closed until the code and local integration evidence are accepted; then E0 supplies actual GPU qualification before E1–E5.

## Output

Submit `engineering/reports/OPERATIONAL_LEARNER_20260914/HANDOFF.md` with O01–O10 acceptance rows, exact source/data/configuration identities, commands/results, remaining GPU-only checks and the notebook launch steps. Store failed as well as successful diagnostic outcomes. Commit compact evidence and source; keep weights and large datasets outside Git. Push BRAMASTRA normally.
