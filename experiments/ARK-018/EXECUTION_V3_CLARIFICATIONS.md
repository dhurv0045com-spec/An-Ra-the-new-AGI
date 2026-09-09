# ARK-018 V3 — PRE-EXECUTION SCORE CLARIFICATIONS

**Status: FROZEN BEFORE GPU EXECUTION.**

This note removes implementation ambiguity without changing any arm, seed, data source, horizon, or primary causal comparison.

1. **BirthContentScore** is the accuracy on the `SEALED` partition of `BIRTH_BOOK_PROBES.json`, scored by normalized continuation log-likelihood over the listed answer choices. The CONTROL partition is reported separately as a diagnostic and never replaces the SEALED score in the primary C-vs-D comparison.
2. The fixed algorithmic OOD battery is secondary and is labelled **narrow transfer**. It cannot by itself establish broad reasoning.
3. The temporary-binding adaptation battery is secondary. Its main outputs are acquisition-to-qualification step, SEALED robustness at qualification, and matched HIGH-vs-LOW post-acquisition robustness curves. It is not allowed to select or modify the four pretraining arms.
4. `allenai/sciq` is a tertiary external diagnostic only. Network failure or inability to establish contamination exclusion cannot invalidate the primary ARK-018 experiment.
5. Full-model displacement from initialization is exact at evaluation milestones. Any per-step path/update or cross-source gradient comparison computed on a parameter subset must be prefixed/labeled `PROJECTED_` and must list the included parameter tensors.
6. A two-seed cognition/learning statement requires the same-direction effect on a predeclared downstream metric in both seeds; content-internalization alone remains a content result.
