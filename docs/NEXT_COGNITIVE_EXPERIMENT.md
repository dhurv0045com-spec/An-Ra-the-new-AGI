# Next decisive experiment

Question: can the existing decoder learn ANSWER / QUERY / ABSTAIN and use the returned evidence in the same bounded V4 episode, at lower request cost than always-query? The first result must establish formation, not transfer or self-improvement.

Freeze `public-action-worlds-v1`, legacy tokenizer identity, complete raw-token+EOS scoring, query cost 0.1 and the three recipe arms in [algorithm comparisons](ALGORITHM_COMPARISONS.md). Screen on Colab with two initialization seeds and confirm conditionally on Kaggle with five seeds, using separate physical GPUs for independent lineages. Exact source/data/runtime and every committed generation accompany all outcomes. Device availability and free quota are checked at run time, never assumed.

The local four-prompt QUERY/EOS canary passed eight updates, but does not qualify learned triage. The owner-run GPU overfit, full episode scoring, same-device resume and independently reattached fresh-session recovery are admission gates. Test failures stop the launch and preserve partial evidence. The expensive switches remain OFF by default.

| Decision | Current status | Evidence required |
| --- | --- | --- |
| Owner-run device/correctness canary | Conditional GO after reviewing pinned notebook | CUDA/T4 identity, memory/disk, exact source/runtime, positive overfit, deterministic same-device continuation and independent scoring. |
| Colab action campaign | BLOCKED_PENDING_GPU | Canary plus fresh-session durable restore and measured forecast admitting complete matched comparison rounds with export reserve. |
| Kaggle confirmation | BLOCKED_PENDING_GPU | Complete valid Colab screen, prospective signal gate and owner review, two distinct physical UUIDs, both device/recovery gates and full-round forecast. |
| V3 graph portfolio | PAUSED, preserved by SHA | Separate owner decision. Its old notebooks do not train cognitive actions. |
| Learned metacognition / retention / forward transfer / self-improvement | NO-GO for claims or promotions | Independent evidence beyond this narrow generator; retention and new-skill acquisition separately demonstrated at equal cumulative work. |

Adaptation timescales remain explicit. Public observations change working state immediately; versioned memory changes under frozen weights and admitted provenance; weight learning requires retention gates; architecture changes remain isolated proposals subject to independent verification. Stage migration and rollback are implemented seams, not an autonomous improvement agent. Arkenstone motivates replay/update-budget controls only after both old and new skills qualify.
