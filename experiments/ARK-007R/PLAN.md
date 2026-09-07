# ARK-007R PLAN INDEX — POST-EXECUTION POINTER

**This file is not the preregistration commit.** It is a post-execution index added so the experiment directory follows the standard Arkenstone layout without rewriting history.

Binding preregistration:
- `experiments/COLAB/MASTER_GPU_PLAN.md`
- commit `3d98103cacc38177390df78ee0eff402da687fcf`

Pre-execution V5 hardening addendum:
- `experiments/COLAB/MASTER_GPU_PLAN_V5_ADDENDUM.md`
- commit `6809bfe8ca70661a4f8b2cd42679db594b944668`

Frozen ARK-007R design from those pre-execution commits:
- acquisition seeds 909, 1010, 1111;
- canonical T2 manifest SHA256 `0dd9305697045b0fbf4e7f268b46a4d7276e4794af5d78b60e999df914ae4236`;
- continuation seeds 2701..2704;
- paired HIGH `1e-3` vs LOW `1e-5`;
- 6000 treated steps;
- sustained G90 = 3 consecutive evals >=0.90;
- treatment begins only after G90 confirmation;
- matched pairs consume identical frozen continuation indices.

See `RESULT.json`, `ANALYSIS.md`, `REDTEAM.md`, and `NOVELTY.md` for the validated imported outcome.
