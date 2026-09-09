# Discovery V7 — validated imported campaign

## Provenance

User-supplied Colab bundle: `ARKENSTONE_DISCOVERY_V7_RESULTS.zip`

- bundle SHA256: `25c4ac0aa01478cb2067147a240a12b4e30516810dad465e619391ff8a6f7faf`
- bundle size: 231,942 bytes
- JSON receipts in bundle: 11
- canonical receipt hashes independently revalidated: **11/11 PASS**
- failure receipt present: **NO**
- GPU smoke receipt: **PASS**
- runner commit: `4a8ffe84223fa64a7409b7d9936f2ff80d3828df`
- master preregistration: `cd058b9067c3fe2e001c70969a1966928d74c7df`
- device: CUDA
- torch: `2.11.0+cu128`
- program-reported runtime: **166.24 minutes**

The ARK-015 task manifest matches the frozen ARK-014 binding manifest SHA256 `fbc8605dc1cfc19ac8692d2b2c6338fcb2af3e02b24907f3b5cca3571947fee3`. The V7 smoke test also revalidated the canonical T2 manifest, CONTROL/SEALED split, deterministic order augmentation, exact model+optimizer fork restoration, deterministic identical-next-update behavior, and the applied-delta cap primitive.

## Campaign verdicts

| experiment | validated outcome |
|---|---|
| ARK-015 | **SUPPORTED_NONARITHMETIC_INVARIANCE_PROTECTION** — 3/3 fresh acquisitions qualified; 8 SEALED-qualified matched pairs; NARROW_HIGH failed 8/8, NARROW_LOW failed 0/8, paired risk difference -1.00; AUGMENTED_HIGH_REFERENCE failed 0/8. |
| ARK-016 | **INCONCLUSIVE_LOW_EVENT_RATE** — all 3 fresh T2 acquisitions qualified, but only 1/12 continuation opportunities produced a collapse→recovery fork; all four post-recovery arms were stable in that single fork. |
| training-design handoff | **RETENTION_EFFECT_TRANSFERRED_MECHANISM_UNRESOLVED** — transfer is now demonstrated at Micro scale under a second task family/stress, but update-magnitude vs data-support mechanism credit remains unresolved. |

## ARK-015 mechanistic signature

Across the 8 executed matched triplets:

- mean final SEALED canonical exact: **1.000** for NARROW_HIGH, NARROW_LOW, and AUGMENTED_HIGH_REFERENCE;
- mean final SEALED ORDER_ONLY / QUERY_ORDER: **0.4683** NARROW_HIGH, **0.9775** NARROW_LOW, **1.0000** AUGMENTED_HIGH_REFERENCE;
- mean SEALED robust-qualification retention fraction: **0.0646** NARROW_HIGH, **1.0000** NARROW_LOW, **1.0000** AUGMENTED_HIGH_REFERENCE;
- mean cumulative parameter path length: **216.14** NARROW_HIGH, **2.02** NARROW_LOW, **413.01** AUGMENTED_HIGH_REFERENCE.

This matters because the high-LR narrowed model did not lose canonical task performance; it selectively lost presentation-order invariance. Also, large parameter movement alone is insufficient as an explanation: AUGMENTED_HIGH_REFERENCE moved farther than NARROW_HIGH while preserving the invariant capability.

The strongest current causal picture is therefore an interaction: **high plasticity plus narrowed training support can drive capability narrowing; either reducing plasticity or continuing to reinforce the broader invariant can protect it.** This remains a Micro-scale result on a controlled binding task, not a universal cognition law.

## Claim boundary

Discovery V7 does **not** establish that low LR is universally optimal, that all large updates are harmful, that replay is universally sufficient, that cross-task continual-learning is solved, or that a Cymek production scheduler should change. It does establish a second-domain retention phenomenon strong enough to justify mechanism dissection and real-data/scale transfer work.
