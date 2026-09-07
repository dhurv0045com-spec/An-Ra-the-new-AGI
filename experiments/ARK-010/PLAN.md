# ARK-010 PLAN INDEX — POST-EXECUTION POINTER

**This file is not the preregistration commit.** It is a post-execution index added for standard Arkenstone experiment layout while preserving the actual pre-execution record.

Binding preregistration:
- `experiments/COLAB/MASTER_GPU_PLAN.md`
- commit `3d98103cacc38177390df78ee0eff402da687fcf`

Pre-execution V5 addendum:
- `experiments/COLAB/MASTER_GPU_PLAN_V5_ADDENDUM.md`
- commit `6809bfe8ca70661a4f8b2cd42679db594b944668`

Frozen ARK-010 design:
- source candidates are prospectively captured HIGH-LR collapse-confirmation states from ARK-007R;
- collapse confirmation = 3 consecutive post-treatment evals below 0.90;
- both recovery forks start from the exact same model/optimizer snapshot;
- both consume the identical unused continuation tail;
- `HIGH_CONTINUE = 1e-3`;
- `RECOVERY_LOW = 1e-5`;
- 4000 recovery steps;
- recovery90 = 3 consecutive evals >=0.90;
- if fewer than 2 collapse sources exist, classify `INCONCLUSIVE_LOW_EVENT_RATE`.

See `RESULT.json`, `ANALYSIS.md`, `REDTEAM.md`, and `NOVELTY.md` for the validated imported outcome.
