# Gandiva branch instructions

## Mission

Read [`GANDIVA.md`](GANDIVA.md) for the branch purpose, inheritance, current state, and work ahead. Gandiva is an An-Ra research branch for building and testing a small from-scratch cognition/learning system through the K8 campaign. The long-term AGI objective is a research aim, not a claim about this implementation.

The implementation and local build checks are complete for the current K8 contract. The next milestone is the owner's fresh Kaggle run using the normal [`notebooks/bramastra_k8.ipynb`](notebooks/bramastra_k8.ipynb). Do not treat historical dispatches that say “implement all of FINAL-K8” as a new open-ended assignment. See [`engineering/STATUS.md`](engineering/STATUS.md) and [`engineering/final_delivery/RUN_EXPERIMENT.md`](engineering/final_delivery/RUN_EXPERIMENT.md).

## Work rules

- Before editing, inspect `git status`, current branch, and local `AGENTS.md` files. Preserve unrelated changes and evidence. Stage only files needed for the scoped task.
- Keep the primary model randomly initialized. Do not import pretrained weights or claim symbolic teachers, tools, tests, fixtures, or reference solvers as learned model capability.
- Do not run local optimizer updates or launch paid/owner GPU work unless the owner explicitly authorizes that run. CPU diagnostics may execute forward/backward but must discard gradients and state zero optimizer commits.
- Treat `experiment.md` and `engineering/FINAL_EXPERIMENT_EXECUTION.md` as frozen K8 protocol/acceptance records. A proposed protocol change must identify the question, affected schema/data/checkpoint identities, controls, and evidence; do not silently alter the campaign after results are seen.
- Keep source, data, run, and checkpoint identities attached to every result. A report from an older code or data closure does not verify a newer run. Preserve failed-run artifacts and use unique run IDs.
- Keep the two T4 workers independent as registered. Do not introduce DDP, extra sessions, hidden retries, or a different notebook as a convenience change.
- Update the current branch guide/status only when the evidence changes. Preserve dated reports and historical plans; label stale material instead of erasing provenance.
- Run focused checks for any code change and the full build verifier when its source contract changes. State exactly what was implemented, tested, run on hardware, and still unverified.
- No force-push, destructive cleanup, or third-party messaging. Push only scoped, reviewed changes to `Gandiva` when publication is within the user's standing authorization.

## Documentation authority

For present purpose and scope, read [`GANDIVA.md`](GANDIVA.md). For current evidence and gates, read [`engineering/STATUS.md`](engineering/STATUS.md). For owner operations, read [`engineering/final_delivery/RUN_EXPERIMENT.md`](engineering/final_delivery/RUN_EXPERIMENT.md). The K8 protocol and F01–F24 details remain in their linked frozen documents. Older plans describe the state and instructions of their dates.
