# ARK-009 PLAN INDEX — POST-EXECUTION POINTER

**This file is not the preregistration commit.** It is a post-execution index added for standard Arkenstone experiment layout while preserving immutable history.

Binding preregistration:
- `experiments/COLAB/MASTER_GPU_PLAN.md`
- commit `3d98103cacc38177390df78ee0eff402da687fcf`

Pre-execution V5 addendum:
- `experiments/COLAB/MASTER_GPU_PLAN_V5_ADDENDUM.md`
- commit `6809bfe8ca70661a4f8b2cd42679db594b944668`

Frozen ARK-009 design:
- non-arithmetic symbolic variable binding/retrieval;
- train/test split by complete fact-set before query expansion;
- 400 train fact-sets × 3 queries = 1200 examples;
- 100 held-out fact-sets × 3 queries = 300 examples;
- acquisition seeds 1201, 1202;
- acquisition LR `1e-3`, batch 64, max 24000 steps;
- qualification requires sustained ordinary exact >=0.90 and query-swap exact >=0.85;
- retention forks run only after qualification;
- if qualified: paired HIGH `1e-3` / LOW `1e-5` on continuation seeds 3701..3706.

Post-execution red-team note: the implemented query-swap also reversed fact order, so the strict qualification failure is scientifically classified as a composite query/order robustness failure, not an isolated query-conditioning failure.
