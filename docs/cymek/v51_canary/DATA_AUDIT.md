# DATA AUDIT (canary instrument)

**Receipt:** `experiments/V5_1_CANARY/receipts/DATA.json` (content-addressed; generator source hash bound). **Generator:** `anra_v5/v51_canary_data.py` v`v51-canary-data/v1`, seed 20260913 (amendment 1: worlds_per_family 2,000 — see PREREGISTRATION.json `amendments[1]`).

## Split structure (latent-world group level)

| Split | Examples | Worlds/family | Generated |
|---|---:|---:|---|
| sealed (test) | 2,400 | 400 | FIRST, hashed before training |
| development | 2,400 | 400 | second |
| training | 7,200 | 1,200 | packed via the real tokenizer/pack path |

Six families: identity (copy), query-conditioned binding, state/order, transitive composition, termination (counting + EOS), missing-information (abstention). Every family has a reference solver verified against every rendering (`test_reference_solver_matches_every_rendering`).

## Screens (fail closed)

- **Exact duplicates across splits:** 0
- **Normalized duplicates across splits:** 0 (whitespace/case-normalized prompt+answer)
- **Latent-group collisions across splits:** 0 (a causal world never spans two splits)
- **Shortcut screen:** all trivial baselines < 0.35 on every family (see SHORTCUT_AUDIT.md)
- **Counterbalancing (by construction):** binding/missing-info query rank drawn from an independent per-world hash domain; ~20% abstention rate; termination lengths 3–8 uniform.

## History of this audit (defects caught before/without any scientific outcome)

1. v0: binding solved 0.583 by first-color-in-prompt (2-fact worlds) → repaired to 4–5 facts + distinct colors + counterbalanced query rank.
2. v1: termination constant baseline 0.583 (skewed lengths) → lengths counterbalanced 3–8.
3. Pack-stream shortfall: 300 worlds/family produced ~18 update windows vs the 491,520-token budget → **amendment 1** (2,000 worlds/family); runner refused with `FAIL_CLOSED` as designed.
4. Amendment-1 pack measured 315,200 tokens (canary prompts average ~31 tokens) → **amendment 2** (3,305 worlds/family ≈ 521k tokens measured, > budget with margin).
5. Freshness-redraw storms at 3,305 worlds: identity (16³–16⁵ space) and state_order (16P3×2 space) too small → identity lengths 3–8, state_order 4 placed/1 removed.
6. Cross-FAMILY normalized collision (binding vs missing_info share the fact-prompt format) → freshness set made global (the screen is global).

**No training ran against any defective version** — every defect was caught by the runner's fail-closed gates or the pre-training screens.
