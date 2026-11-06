# ELD-DEV-004 architecture and compute comparison

| Method | Parameters | Reads per model call | Mechanism |
| --- | ---: | ---: | --- |
| Seeded dense control | 44,112 | 0 | Tokenwise residual capacity control over the shared dense core. |
| Contextual pointer | 44,097 | 1 | Contextual attention, vocabulary copy scatter, and a learned copy/generation gate. |
| Hybrid associative | 44,106 | 1 | Contextual pointer plus learned lexical equality across relative windows. |
| Two-read | 44,110 | 2 | First copy distribution supplies soft symbol matches over the second read's source windows. Four new scalar parameters learn offsets and strength. |

All arms use the same fixed batch stream through update 384. Dense continues to update 576 under the same frozen optimizer schedule. Wall-clock training time excludes evaluation and source import. Nearest dense points were update 480, 480 and 576 for seeds 7, 19 and 41. Their times were 6.748, 6.763 and 7.927 s against candidate times of 7.389, 7.365 and 7.582 s. All three are inside the preregistered 10% tolerance. Dense matched-time scores were 0/64 in each seed, with 64/64 for two-read. Timing depends on the host. This is a time-matched screen, with separate exposure and parameter comparisons.

A parameter-only estimate of 6 × parameters × non-BOS tokens describes rough dense training work. It excludes masked attention, sparse and dense vocabulary scatter, recurrent read scoring, optimizer traffic and generation. For one 384-update two-read arm it is roughly 42.3 GFLOPs under that convention. This is a scale indicator. Use the measured training time and forward-position counters for the actual comparison. Kernel-level FLOPs and energy were unmeasured.

The pointer-generator reference uses the mixture principle established in See, Liu and Manning, adapted here to a causal context. [ACL Anthology](https://aclanthology.org/P17-1099/).
