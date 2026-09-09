# ARK-015 ANALYSIS — non-arithmetic invariance retention under narrowing

## Status

**EXECUTED. Verdict: `SUPPORTED_NONARITHMETIC_INVARIANCE_PROTECTION`.**

ARK-015 is the first Arkenstone experiment in which the LR-retention phenomenon passes a preregistered positive criterion outside arithmetic. The subject is the robust order-augmented symbolic binding task repaired in ARK-014.

## Primary result

All three fresh acquisition seeds qualified the robust binding gate. Eight matched continuation triplets completed before the campaign budget blocked the ninth.

- NARROW_HIGH `1e-3`: **8/8 SEALED robust-retention failures**
- NARROW_LOW `1e-5`: **0/8 failures**
- AUGMENTED_HIGH_REFERENCE `1e-3`: **0/8 failures**
- paired risk difference LOW − HIGH: **-1.00**
- HIGH fail / LOW stable discordance: **8**
- reverse discordance: **0**

The result holds within every independent acquisition seed: 3/3 vs 0/3, 3/3 vs 0/3, and 2/2 vs 0/2.

This exceeds the preregistered event-count, replication, risk-difference, reverse-discordance and grouped-direction requirements.

## What was actually lost

The critical signature is not generic task forgetting. Mean final SEALED canonical exact remained **1.000 in all three arms**. The narrowed HIGH arm therefore remained perfect on the presentation distribution it continued to see.

What disappeared was the broader presentation-order invariant:

| arm | final canonical | final ORDER_ONLY / QUERY_ORDER | robust-retention fraction |
|---|---:|---:|---:|
| NARROW_HIGH | 1.000 | 0.4683 | 0.0646 |
| NARROW_LOW | 1.000 | 0.9775 | 1.0000 |
| AUGMENTED_HIGH_REFERENCE | 1.000 | 1.0000 | 1.0000 |

This is best described as **capability narrowing / invariance erosion under specialization pressure**, not ordinary forgetting.

A training loop watching only loss and canonical held-out accuracy would miss the degradation almost completely.

## Mechanistic implications

Mean cumulative parameter path length was approximately:

- NARROW_LOW: **2.02**
- NARROW_HIGH: **216.14**
- AUGMENTED_HIGH_REFERENCE: **413.01**

LOW therefore remains compatible with a near-freezing explanation. However, parameter movement alone cannot explain the full result: AUGMENTED_HIGH_REFERENCE moved roughly twice as far as NARROW_HIGH while keeping order robustness at 1.0.

The strongest current causal picture is an interaction:

`high plasticity + narrowed support for an invariant -> specialization / invariant erosion`

while either:

`reduced plasticity`

or

`continued training support for the invariant`

can preserve the broader capability.

This is stronger and more useful than the earlier one-dimensional hypothesis "large updates destroy cognition".

## Cognition interpretation

**DEMONSTRATED at Micro scale:** a model can keep perfect ordinary task performance while losing a more general transformation-invariant solution. Continued training can therefore make a learned capability narrower without looking worse on the narrow distribution.

**NOT DEMONSTRATED:** general human-like reasoning degradation, arbitrary continual-learning protection, language-model-scale transfer, or a universal optimizer law.

The practical implication is nonetheless important: capability evaluation for future training must include robustness/invariance probes that are not identical to the currently reinforced distribution.

## Next bottleneck

The reliable 8/8 NARROW_HIGH event generator should replace stochastic T2 collapse as the main mechanism laboratory. The next causal experiment should independently manipulate applied update magnitude and invariant-supporting replay under this same stress.
