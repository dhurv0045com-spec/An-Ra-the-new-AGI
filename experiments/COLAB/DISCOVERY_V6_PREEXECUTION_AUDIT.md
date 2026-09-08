# DISCOVERY V6 — PRE-EXECUTION IMPLEMENTATION AUDIT

**No Discovery V6 GPU result existed when this audit was committed.**

## Bound preregistrations

- ARK-011 plan: `1d5fc08000614e740b7c93d87e8d233c903bcddf` (existing experiment; no result yet).
- ARK-012 plan: `324b762b3cd3e362e1a28a34bd609c30fbc1f171`.
- ARK-013 plan: `fefbe39350d39e25e998a24e81e1caf0090ea8e4`.
- ARK-013 feasibility addendum: `8d086a501ef579574947b506fec6e0c71bd80711`.
- ARK-013 metric addendum: `76bd40a5efd4799c11ba156bb082e1506be5a894`.
- ARK-014 plan: `59e01a33a11aea1222368553d0819764341e2577`.
- Master V6 plan: `f46bae74c783821452947f3927acddbc14cc6dbf`.

All scientific plans above entered Git history before their experiment runner was executed. The ARK-013 feasibility correction was also committed before the runner; the numeric materiality addendum was committed before execution and matches the already-implemented `0.10` thresholds in `run_ark013.py`.

## Implemented source

- `experiments/COLAB/discovery_v6_common.py`
- `experiments/COLAB/run_discovery_v6.py`
- `experiments/ARK-012/run_ark012.py`
- `experiments/ARK-013/run_ark013.py`
- `experiments/ARK-014/run_ark014.py`
- existing `experiments/ARK-011/run_ark011.py`
- `experiments/COLAB/arkenstone_discovery_v6.ipynb`

Implementation was syntax-checked during construction. This is **not** runtime certification. The Colab launcher is required to compile the exact pinned checkout again and pass the GPU smoke test before any full campaign is allowed to start.

## Static scientific checks

### ARK-012
- Source combinations are explicitly selected from prior high-instability evidence; the runner labels them selected-event screens and does not count them as fresh incidence evidence.
- All six schedules start from the exact same prospective collapse snapshot and index the same absolute continuation stream.
- OOD_SEALED is read only for measurement; switch thresholds use OOD_CONTROL only.

### ARK-013
- Deterministic forced-carry generator was feasibility-audited before execution: the revised domain can provide 500 train + 200 commutation-filtered OOD examples.
- Every generated row is asserted to satisfy ones-column carry.
- T3 CONTROL/SEALED are disjoint and exhaustive over the frozen OOD membership.
- FIXED_HIGH, FIXED_LOW, and ADAPTIVE_HIGH_LOW start from the same T2 snapshot and consume identical T3 semantic minibatches.
- T2 and T3 sealed sets never control switching.

### ARK-014
- Complete fact-set train/test split precedes query/order expansion.
- The 100 held-out fact-sets are split 50/50 into CONTROL and SEALED before diagnostics.
- ORDER_ONLY changes order while keeping query/answer fixed; QUERY_ORDER changes both; QUERY_ONLY is reported explicitly but acknowledged as redundant because all three same-order queries are already enumerated by CANONICAL.
- ORDER_AUGMENTED permutations are deterministic functions of seed/step/batch-position/semantic-id.
- Retention eligibility is CONTROL-only; sealed values cannot select which acquisition regime proceeds.

## Runtime budget semantics

The master target is 240 minutes and uses launch gates/reserves between experiments. Scientific integrity has priority over exact wall-clock cutoff: once a matched schedule/triplet/pair has started, the runner completes that matched unit rather than truncating one arm and comparing unequal budgets. Therefore actual runtime may exceed the nominal 240-minute target slightly if an in-flight matched unit crosses the boundary. No sleeping, filler loops, or artificial runtime padding is permitted.

## Mandatory smoke-test gates

Before full execution the pinned Colab checkout must demonstrate:
- canonical T2 manifest hash;
- T2 CONTROL/SEALED disjoint + exhaustive;
- deterministic T3CARRY manifest, forced-carry invariant, zero train/OOD commutation overlap;
- binding fact-set CONTROL/SEALED zero overlap;
- deterministic order augmentation;
- CUDA forward/backward + optimizer update;
- exact snapshot reload;
- identical next-parameter hash from same snapshot/minibatch/LR.

Any failure blocks the full run.

## Claim discipline

Even if every V6 screen is positive, this does not authorize a universal LR law, AGI claim, or Cymek production scheduler change. The campaign is designed to decide which mechanism deserves fresh multi-seed transfer/scale replication next.