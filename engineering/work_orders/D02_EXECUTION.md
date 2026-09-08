# D02 bounded execution: matched inquiry teaching

Chief design, 2026-09-08. This is the independently executable first portion of W04, not completion of PPO or the integrated learner. Read `reports/D02_REVIEW.md`, W04 and the experiment registry first.

## Question and intervention

Does depth-two target-information teaching improve learned prediction after two real queries relative to one-step teaching? Use the existing discovery investigator and trainer as the controlled substrate. The intervention changes only demonstration gain labels. Joint predictor/policy representation changes caused by those labels are part of this intervention; do not attribute results exclusively to action selection.

## Frozen development design

- Generate `make_worlds(4)` with the saved family-stratified semantic split. Record full inventories and their hashes. Use the identical inventory for seeds 801 and 802. Confirmation worlds remain unscored.
- Per seed: 256 demonstration episodes, two-query horizon, uniform legal history collection, zero query cost. Both arms share every non-gain tensor and row order.
- Width 32; random initialization per seed, cloned exactly across arms. AdamW via existing `Trainer`: 500 updates, batch 32, learning rate 0.001, policy weight 0.25, existing weight decay and clipping. Identical sampler initial state; no tuning after development results.
- Evaluate all development worlds, four deterministic targets per world, evaluation seed 17. Report budgets 0, 1 and 2; only budget 2 is primary. Policies: learned, random, coverage and no_memory, each evaluated with its own arm's predictor. Pair world/target keys across arms.
- Primary contrast: depth-two learned minus one-step learned accuracy at budget 2, by seed. Secondary: Brier difference, per-family outcomes, and learned minus each fixed control within each arm. A fixed control using another arm's predictor is a different comparison and must be named explicitly.
- Use 1,000 deterministic world-cluster bootstrap resamples for descriptive intervals. Never pool target rows as independent worlds or treat the two training seeds as a confirmatory sample. Publish both seeds without choosing the favorable one.

## Execution and evidence

Own `bramastra_lab/research/learning/inquiry/**`, `tests/test_research_inquiry*.py`, `engineering/reports/W04/**`, and a new uniquely named `artifacts/bramastra/d02_*` directory. Other research modules and discovery source are read-only. No accelerator, paid compute, network, nested agents or Git operations.

Repair the inconsistent posterior fixture as directed in the chief review; do not distort the teacher to satisfy it. Add runner checks for paired initialization/data/sampler identity, disjoint semantic inventories, legal actions and horizon handling. A tiny isolated smoke run may use different settings, but is never one of the two primary replications.

Implement a callable runner with a new-directory requirement. Freeze its configuration and snapshots of the imported local source closure before training. Record Python/torch/numpy versions, CPU threads (2), parameter count, tensor identities, starting fingerprints, and timings for generation, training, evaluation and reporting. Verify source hashes again after execution. Save raw prediction/action rows, training histories, comparisons and a completion or failure manifest. Keep checkpoint weights out of Git.

The primary campaign has a five-minute CPU wall-time allowance, including both seeds and teacher generation. Check the deadline between bounded work units; preserve partial evidence on timeout. Do not silently shrink settings, discard a failed seed or rerun until favorable. If the frozen design cannot fit, return the measured bottleneck and a proposed revision before another primary campaign.

## Acceptance and next decision

Chief acceptance requires reproducible raw scoring, matching labels/keys, the exact delayed-information diagnostic, passing focused tests, source integrity and an honest two-seed or partial-run report. A positive result supports only this finite rule-learning development setting. A null result triggers diagnosis of teacher disagreement, imitation quality and predictor accuracy before more compute. PPO, distractor transfer, strategy-validation and broader self-improvement remain W04 follow-up work.
