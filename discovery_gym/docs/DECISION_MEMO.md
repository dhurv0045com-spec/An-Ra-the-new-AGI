# DG0 Decision Memo

## Decisions made

- **Base:** Gandiva at `49a3717ae9dc81138f55dc2398c7615f8c80b4b1`, because it is the current owner-facing integration point and contains the existing discovery precursor.
- **Scope:** DG0 only. No model, optimizer, hardware, or scientific learning claim.
- **World family:** finite deterministic binary causal programs with explicit graph-shape and operator identities.
- **Action costs:** read `1`, probe `2`, intervention `3`, submit `1`, stop `0`.
- **Default smoke budget:** `8` action units, chosen to leave room for reads, multiple probes, and a submission in the compact suite.
- **Independent unit:** one latent world, regardless of query or rendering count.
- **Shortcut policy:** surface, position, identity, and no-experiment controls are retained even if they are weak, so future apparent gains can be compared against them.
- **Intervention safety:** interventions are anchored to one public probe context whose one-variable neighborhood is disjoint from every target-candidate assignment; no target-dependent legal-action filtering is used.

## Unresolved choices requiring a later protocol

- **Model family and size:** no architecture is selected by DG0. DG1 must justify any model from a measured learnability limitation.
- **Scientific effect margin:** no DG1–DG3 numerical success margin is silently chosen here. The protocol template requires a prospective margin, uncertainty method, and sample-size rationale before confirmation.
- **Belief representation:** a recurrent state, explicit hypothesis posterior, or neural state update is not selected. DG1 must compare the representation against full-history and no-memory controls.
- **Noise:** deterministic worlds come first. Any later noise contract must state distribution, seed, calibration, and information-gain interpretation.
- **External independence:** generated confirmation/transfer namespaces are not externally authored or sealed. A stronger independence claim requires a separately owned fixture source and custody record.
- **Inference-cost accounting:** transparent DG0 controls have zero model inference cost. A learned policy must report calls, candidate rows, wall time, and training cost separately.

## Why the default budget is prospective

Budget `8` is a smoke/instrument setting, not a scientific threshold. It is large enough to expose multiple evidence types and a final submission in the finite suite, while remaining cheap and deterministic. It must be recalibrated from DG1 development measurements and frozen before confirmation; confirmation outcomes cannot select a favorable budget.

## Stop condition

If a future learned policy cannot beat the strongest equal-cost transparent control, including `transparent-information-gain/v1` and `no-experiment/v1`, on fresh development worlds, do not add a larger model or claim discovery. First diagnose whether the finite task is too easy, whether the policy is memorizing, or whether the information boundary is wrong.
