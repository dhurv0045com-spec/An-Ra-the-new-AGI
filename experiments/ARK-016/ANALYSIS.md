# ARK-016 ANALYSIS — update-magnitude mechanism red team

## Status

**EXECUTED. Verdict: `INCONCLUSIVE_LOW_EVENT_RATE`.**

ARK-016 attempted to distinguish literal LOW LR from applied-update magnitude by prospectively producing T2 collapse→recovery states and then comparing LOW, HIGH, HIGH with a per-step LOW-matched applied-delta cap, and HIGH with a 10× LOW cap.

## Acquisition and event rate

All three fresh T2 parents acquired CONTROL G90:

- seed 1919: confirmation 10,400;
- seed 2020: 17,800;
- seed 2121: 15,800.

The event generator then failed: **11/12 continuation opportunities never produced a CONTROL collapse**, leaving only one sealed-qualified recovery fork. That is far below the preregistered event requirement.

## Single available fork

All four post-recovery arms remained stable with RET90 = 1.0:

- LOW_REFERENCE path: ~0.695
- HIGH_UNCAPPED path: ~66.51
- HIGH_CAP_1X path: ~0.695
- HIGH_CAP_10X path: ~6.95

This confirms that the cap implementation can produce the intended movement ratios in a real training fork, but it provides **no mechanism discrimination** because the uncapped HIGH reference did not fail.

## Interpretation

No conclusion should be drawn about whether update magnitude mediates the retention effect. `HIGH_CAP_1X` matching LOW path and remaining stable is not evidence of protection when HIGH_UNCAPPED is also stable.

The useful lesson is experimental: stochastic T2 recollapse is now a poor mechanism laboratory. ARK-015 produced a substantially better event generator, with 8/8 NARROW_HIGH failures under the non-arithmetic distribution-narrowing stress.

## Next step

Repeat the update-magnitude causal intervention inside the reliable ARK-015 event regime and independently manipulate invariant-supporting replay. That experiment can separate:

1. movement magnitude;
2. data support for the invariant;
3. their interaction.
