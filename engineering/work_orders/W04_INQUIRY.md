# W04 — Multi-step inquiry and outcome-trained policy

**Status:** D02 exact diagnostic can start; learned integration depends on W01–W03. **Effort:** 3–5 hours. **Role:** learning-algorithm engineer. **Compute:** CPU exact diagnostics and bounded small-model pilots; accelerator campaign separately allocated.

Read LEARNING_ALGORITHMS.md B–D and experiment entries D02/D04. Own `research/learning/inquiry/`, inquiry tests and `engineering/reports/W04/`.

## Deliverable

Implement one-step and exact depth-two teaching controls, then on-policy outcome-trained inquiry with the documented actor-critic/PPO choices. Reuse public model interfaces. Store collection probabilities, returns, advantages and masks. The learned evaluation policy receives no teacher/posterior access.

## Central hypothesis

Preparatory actions that have zero immediate information can be learned when their delayed consequences improve final task success. The exact parity case proves the opportunity; the trained experiment must establish whether the policy learns and transfers the strategy.

## Required comparisons

1. Random and fixed coverage.
2. One-step teacher and depth-two teacher as privileged diagnostic controls.
3. Supervised policy imitating each teacher.
4. Outcome-trained policy with a frozen predictor.
5. Joint predictor/policy learning only as a separately named follow-up.

Equalize real interactions and final inference budget. Count teacher search and training cost. Include noisy distractors and held-out mechanism compositions. A policy rewarded for uncertainty alone is not an acceptable substitute for the outcome objective.

## Acceptance evidence

- Exact diagnostic verifies zero first-step information and positive two-step information.
- PPO ratio, terminal/truncation returns, legal masks and advantage calculations match independent tiny examples.
- Fixed predictor weights remain unchanged in the isolation arm.
- On-policy collection and old-policy likelihoods are consistent; no stale rollout is mislabeled on-policy.
- Evaluation policy inputs exclude labels, hidden mechanism state and teacher scores.
- A bounded development run produces paired success/cost outcomes across controls, including failures and a fresh strategy-validation pool.
- Report whether improvement survives the strongest affordable fixed baseline, not only random inquiry.

Success is resolving the hypothesis under a correct comparison. If one-step and multi-step methods both fail, use diagnostics to separate representation, exploration and teacher limitations. Do not add an oracle at inference to manufacture a learned-policy gain.
