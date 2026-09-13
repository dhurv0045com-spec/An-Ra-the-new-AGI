# SHORTCUT AUDIT (canary instrument)

**Method:** deterministic trivial baselines scored as exact-match rate per family on the development split (`anra_v5/v51_canary_data.shortcut_baselines`), executed at `prepare` time and bound into the DATA receipt. A family any baseline solves at ≥ 0.35 is repaired or removed BEFORE training (§13).

| Baseline | What it models |
|---|---|
| constant / answer_frequency | global modal answer |
| latest_position | answer = last prompt word |
| first_color_in_prompt | answer = first mentioned color |
| answer_is_fixed_none | abstention-rate exploitation |

## Results (seed 20260913, worlds_per_family 2,000, development split)

Worst baseline across all families and baselines: **0.267** (< 0.35 gate). Per-construction guarantees:

- binding: query rank counterbalanced per world over 4–5 alphabetically-sorted facts with distinct colors → positional baseline pinned near 1/n ≈ 0.225.
- missing_info: ~20% abstention (independent hash domain) → fixed-"none" baseline ≈ 0.20.
- termination: lengths 3–8 counterbalanced → constant/answer-frequency ≈ 0.167–0.267.
- identity/state_order/composition: answer spaces too large for constant/frequency; composition mention order is rotated against weight order so positional heuristics ≤ 1/4.

Verdict: **no prohibited shortcut exists on any family** — exact scores above chance-level baselines therefore reflect learned behavior, not surface statistics.
