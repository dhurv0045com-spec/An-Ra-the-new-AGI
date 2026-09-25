# Gandiva — a from-scratch cognition experiment

Gandiva is an An-Ra research branch with a narrow job: turn claims about learning, cognition, tool use, architecture adaptation, and recursive improvement into a real, inspectable experiment. Its current deliverable is the K8 campaign and its self-sustaining Kaggle notebook. It is a research instrument, not a claim that AGI has been built.

This branch grew from `BRAMASTRA` at `63d60b3`. It keeps the shared `bramastra_lab` package, model, and broad from-scratch objective, then adds the production paths, evidence contracts, tests, and notebook work needed for the Gandiva K8 campaign. The relationship and exact boundaries are in [GANDIVA.md](GANDIVA.md). The detailed code difference is available with `git diff BRAMASTRA...Gandiva`.

The latest local build verification passed F01–F24, seven test groups, and seven production-interface exercises with zero local optimizer updates. It permits a fresh owner-run experiment; Kaggle's two-T4 qualification and all learned outcomes remain untested. Read the [current status](engineering/STATUS.md) and [owner run guide](engineering/final_delivery/RUN_EXPERIMENT.md) before launching. Use the normal, self-sustaining [K8 notebook](notebooks/bramastra_k8.ipynb), not the separate GPU-only notebook.

Start here according to what you need:

- [Gandiva's purpose, inheritance, differences, and next work](GANDIVA.md)
- [Agent instructions for this branch](AGENTS.md)
- [Current engineering status and evidence](engineering/STATUS.md)
- [Frozen K8 design and registered comparisons](experiment.md)
- [Build acceptance contract F01–F24](engineering/FINAL_EXPERIMENT_EXECUTION.md)
- [Kaggle launch and recovery instructions](engineering/final_delivery/RUN_EXPERIMENT.md)
- [Previous owner-run results and their limits](engineering/reports/K8_EXPERIMENT_20260923/RESULTS.md)
- [Separate 100M TPU preflight status](engineering/TPU_100M_COGNITION_PROGRESS.md)

Older blueprints and reports remain in the repository because they record how the work arrived here. Their wording and claims belong to their recorded dates; use the current status and evidence links above for today's state.
