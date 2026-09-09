# CYR-GPU-010 DESIGN REASONING

## V9 evidence

The returned V9 bundle was audited directly. Tesla T4, TINY 1,647,104 params, two seeds, 2,000,000 actual real tokens / 15,632 optimizer updates each. Train-probe candidate-free exact reached 0.94–1.00 while DEV_CONTROLLER remained 0–1.04%. Neither seed reached G90. Therefore the immediate unknown is capability formation, not retention.

## Arkenstone evidence used

ARK-002B is the key acquisition prior: a 4L/128w subject showed replicated memorize-first -> generalize-later behavior, with fresh-seed sustained G90 around 12k for one seed and still climbing at 18k for another. ARK-003 is an important negative result: simple curriculum and teacher/decomposition suffixes did not demonstrate faster G90 and introduced compute asymmetry. We therefore keep primary acquisition flat.

ARK-015 adds a second lesson: after robust non-arithmetic order invariance was acquired, narrow HIGH continuation caused 8/8 failures, while LOW and fully augmented HIGH caused 0/8. CYR-010 imports only the experimental lesson that learned structure may be brittle and that data support can matter. ARK-016 is inconclusive due low event rate. ARK-017 is currently preregistered/unexecuted and contributes no result claim.

## Why RESEARCH_SMALL

Cymek RESEARCH_SMALL is 4L/128w/FFN512, the closest real V5 geometry to Arkenstone's historical 4L/128w Micro while retaining Cymek QK norm, production tokenizer and real training backend. V9's T4 calibration measured ~1.887 updates/sec on RESEARCH_SMALL, making an 18k update box approximately a 2h40m acquisition commitment and suitable for a single ~3h Colab session.

## Why dense candidate-free evaluation

V9 evaluated every ~400k tokens (~3.1k updates), which is too coarse for studying a grokking-like transition. Arkenstone used 200-update cadence. CYR-010 adopts 200-update candidate-free cadence so onset, G50 and sustained G90 can be located instead of inferred from sparse snapshots.

## Why the reasoning battery

Held-out exact arithmetic alone can hide brittle shortcuts. The battery asks whether generalization has structural signatures: digit-wise correctness, counterfactual locality, commutation behavior, format robustness and three-digit no-carry extrapolation. Only STANDARD controls G90; the rest cannot steer training.

## Decision after the run

- RESEARCH_SMALL G90 + strong locality -> replicate at this scale, then move to second-family reasoning.
- RESEARCH_SMALL G90 without locality -> investigate shortcut/generalization quality before any retention work.
- No RESEARCH_SMALL G90 by 18k -> compare tokenizer/representation/objective/data details against Arkenstone rather than blindly spending another session on LR retention.

This is intentionally one deep experiment, not another many-arm tournament.
