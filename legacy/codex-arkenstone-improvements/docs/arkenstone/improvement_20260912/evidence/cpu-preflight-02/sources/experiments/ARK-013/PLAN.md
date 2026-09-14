# ARK-013 — STABILITY–PLASTICITY FRONTIER ON A NEW CARRY SKILL

## Status

**PREREGISTERED BEFORE EXECUTION.**

## Question

Low LR protects an already-generalized T2 state, but that protection may simply freeze the network. A useful controller must preserve old capability without preventing acquisition of a new one.

> After T2 generalization, how do HIGH, LOW, and state-conditional HIGH→LOW training trade off acquisition of a genuinely harder carry-addition skill against retention of the old no-carry skill?

## Hypotheses

- `H-PLASTICITY-COST`: fixed LOW (`1e-5`) preserves T2 better but learns the new carry skill more slowly or incompletely than HIGH (`1e-3`).
- `H-ADAPTIVE-FRONTIER`: state-conditional HIGH→LOW acquires the new skill comparably to HIGH, then improves old-skill retention and/or new-skill stability after the switch.

A positive result would address a major alternative explanation for ARK-007R: "LOW works only because it nearly stops learning."

## Old skill

Canonical T2 no-carry task and ARK-011 OOD_CONTROL / OOD_SEALED firewall.

## New skill: T3CARRY

Two-digit addition with a forced ones-column carry:
- operand A tens band train: 1..5;
- operand A tens band OOD: 6..7;
- ones digits satisfy ua + ub >= 10;
- choose operand-B tens so the final result remains two digits;
- deterministic dataset seed 131313;
- 500 train examples;
- draw a larger OOD pool, remove any sorted-operand commutation overlap with old/new train sets, retain 200 OOD examples;
- OOD is deterministically split within tens band into `T3_CONTROL` and `T3_SEALED` using the same hash-rank method as ARK-011.

The runner must emit an exact task manifest with all membership hashes and overlap checks.

## Subject and frozen sources

Acquire fresh T2 generalized checkpoints using seeds:
- 1717
- 1818

For each acquired subject, use new-skill minibatch order seeds:
- 6801
- 6802

Maximum 4 matched triplets. These are fresh with respect to prior ARK experiments.

## Arms

At exact T2 OOD_CONTROL sustained G90 confirmation, fork from identical model/optimizer/RNG state. Each arm trains **only on T3CARRY training examples** for a fixed 12,000-step horizon and consumes byte-identical T3 minibatches.

1. `FIXED_HIGH`: lr=1e-3 for all 12k steps.
2. `FIXED_LOW`: lr=1e-5 for all 12k steps.
3. `ADAPTIVE_HIGH_LOW`: begin at 1e-3; switch once, irreversibly, to 1e-5 after 3 consecutive `T3_CONTROL >=0.90`; if that trigger never occurs, remain HIGH for the full box.

No replay of T2 is allowed in this experiment. This intentionally exposes interference/plasticity rather than hiding it with rehearsal.

## Evaluation firewall

Every 200 steps measure:
- T3_CONTROL (may trigger ADAPTIVE switch)
- T3_SEALED (measurement only)
- T2_CONTROL (measurement only during this phase)
- T2_SEALED (primary old-skill retention measurement)

Neither sealed set can affect optimization, stopping, or switching.

## Primary endpoints

At matched 12k new-skill updates:
- new-skill T3_SEALED area, final exact, sustained G90 onset/confirmation;
- old-skill T2_SEALED area, final exact, RET90;
- a preregistered two-objective Pareto table: `(T3_SEALED final, T2_SEALED final)` and `(T3_SEALED area, T2_SEALED area)`;
- switch step for ADAPTIVE;
- parameter displacement from the pre-T3 fork.

## Verdicts

`PLASTICITY_COST_DEMONSTRATED` if FIXED_LOW has materially lower T3_SEALED acquisition than FIXED_HIGH across a majority of matched triplets while preserving T2 better.

`ADAPTIVE_PARETO_IMPROVEMENT` if, on at least 2 matched triplets spanning both acquisition seeds, ADAPTIVE is not worse than FIXED_HIGH by >0.05 on T3_SEALED final and improves T2_SEALED final by >=0.10, or vice versa with area metrics showing the same direction.

`ADAPTIVE_NO_BENEFIT` if enough triplets execute but ADAPTIVE does not improve the stability–plasticity frontier.

`INCONCLUSIVE_NEW_SKILL_NOT_ACQUIRED` if FIXED_HIGH itself fails to reach sustained T3_CONTROL G90 on both acquisition seeds.

## Claim limits

T3CARRY is still arithmetic and Micro-scale. This experiment measures the stability–plasticity tradeoff; it does not establish non-arithmetic transfer or production readiness.