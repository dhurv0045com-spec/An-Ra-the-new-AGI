# ARK-013 ANALYSIS — stability–plasticity frontier on a new carry skill

## Status

**EXECUTED. Preregistered verdict: `INCONCLUSIVE_NEW_SKILL_NOT_ACQUIRED`.**

Both fresh T2 subjects acquired sustained T2 OOD_CONTROL G90 (seed 1717 confirmation 11,800; seed 1818 confirmation 15,200). All four matched T3 triplets then executed.

The decisive prerequisite failed: FIXED_HIGH never reached sustained T3_CONTROL G90 on any triplet. T3_SEALED peak exact ranged only about 0.30–0.45 under FIXED_HIGH. Therefore the experiment cannot answer whether LOW imposes a plasticity cost relative to a HIGH arm that successfully acquires the new skill, and ADAPTIVE never switched because its T3 trigger was never reached.

Aggregate OOD_SEALED means across four triplets:

| arm | T3 area | T3 final | T2 area | T2 final |
|---|---:|---:|---:|---:|
| FIXED_HIGH | 0.147 | 0.071 | 0.0026 | 0.000 |
| FIXED_LOW | 0.209 | 0.265 | 0.0608 | 0.038 |
| ADAPTIVE_HIGH_LOW | 0.147 | 0.071 | 0.0026 | 0.000 |

Because ADAPTIVE never switched, it is behaviorally identical to FIXED_HIGH in this box.

## Important secondary result

There is a useful boundary result even though the primary plasticity experiment is inconclusive. Training only on T3CARRY for 12k updates, with **no T2 replay**, destroyed sustained old-skill T2 retention under every tested arm. FIXED_LOW slowed the loss somewhat but did not preserve T2: T2 RET90 was 0 in every matched triplet and final T2_SEALED exact was 0–0.082 under LOW, versus 0 under HIGH/ADAPTIVE.

That means ARK-007R/011's LOW-LR protection should currently be interpreted as protection under **same-task continuation**, not as a general antidote to cross-task interference. This is a critical limit on the mechanism.

## What is not demonstrated

- `H-PLASTICITY-COST` is not resolved because HIGH did not acquire T3.
- `H-ADAPTIVE-FRONTIER` is not tested because ADAPTIVE never reached its switch trigger.
- No claim should be made that LOW learns new skills better merely because its mean T3 metrics were higher here; no arm acquired the new skill.

## Next bottleneck

Before repeating the Pareto test, T3CARRY acquisition itself must be made reliable under a frozen, separately validated acquisition protocol. Only then is a stability–plasticity comparison scientifically interpretable.
