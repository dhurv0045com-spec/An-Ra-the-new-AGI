# CYR-GPU-006 — DESIGN REASONING

## Evidence first

The live Arkenstone branch was re-audited at `6acd9dcbdd28d00f387ffcd004253a813aca4b66`, including raw/results for ARK-011 through ARK-014 and the independently validated Discovery V6 bundle (`1ec3224075de49b080e229111e4c4f897430c3eca888c487116aff07a65c8d15`). The design follows what those experiments support, not stale summary prose.

The important pattern is narrower than “LOW LR is better”:

- ARK-011: after HIGH-LR recovery, switching LOW reduced recurrent same-task instability (3/6 HIGH failures vs 0/6 switched LOW).
- ARK-012: exact switch threshold was not identified; timing/state remained aliased.
- ARK-013: LOW slowed but did not prevent loss of old T2 under long pure-new-skill/no-replay training, and the intended new-skill plasticity test was inconclusive because HIGH never acquired the new skill.
- ARK-014: deterministic order augmentation repaired the non-arithmetic binding acquisition gate, but the HIGH-vs-LOW retention screen had zero failures in either arm and therefore did not demonstrate LR transfer.

So the next useful question is not “does LOW win again?” It is whether a state-dependent policy survives on the real Cymek V5 implementation **and** whether the retained state remains useful for learning a genuinely different computation.

## Primary causal unit

An independent acquired parent is the subject. Each parent is trained exactly once at HIGH until candidate-free G90 or its actual-token ceiling. If qualified, four continuations restore the exact same model+optimizer checkpoint and consume the same frozen future index stream:

1. HIGH_CONTINUE
2. LOW_CONTINUE
3. FIXED_TIME_HIGH_TO_LOW
4. HYSTERETIC_HIGH_LOW

The comparison is paired within parent and replicated across parents. A single parent can never produce a scientific winner. FIXED_TIME is mandatory because ARK-012 does not establish that state, rather than ordinary decay timing, is the causal variable.

## Why the second-family stage is HYST vs LOW

An earlier CYR-GPU-006 candidate compared the pre-continuation G90 parent against the final adaptive state. That is a bad causal comparison: the adaptive state is older and has consumed additional same-task tokens. Any plasticity difference would mix policy with model age/exposure.

The revised transfer stage instead uses two **equal-age, equal-exposure** retention states from the same parent:

- prospective candidate: HYSTERETIC_HIGH_LOW;
- mechanistic comparator: LOW_CONTINUE.

Both must remain T2-qualified and have consumed the same continuation dose. Both then move to the same HIGH LR and learn the same deterministic non-arithmetic stream. This directly attacks the strongest alternative explanation from ARK-007R: LOW may protect mainly because it nearly freezes movement. If the adaptive state preserves arithmetic yet learns robust binding no worse than the equal-age LOW state, it is more useful than a pure freeze interpretation. If LOW learns just as well, the adaptive mechanism claim weakens even if retention is good.

## Why robust binding + replay

ARK-014 shows fixed-order canonical binding is a poor transfer subject because it can look perfect while failing under order perturbation. CYR therefore uses order-augmented registry binding and keeps CANONICAL, QUERY_ONLY, ORDER_ONLY and QUERY+ORDER separate.

ARK-013 already established a no-replay cross-task interference boundary on Micro arithmetic. Repeating only pure-new-skill training would risk spending expensive GPU time rediscovering catastrophic interference. CYR uses a fixed 18 binding + 2 old-T2 replay row batch and reports the **actual token fraction**. This is not claimed to be an optimal replay rate; it is a preregistered control intended to keep the plasticity question measurable.

## What is deliberately not changed

- No new optimizer family: there is not enough evidence to change LR/state and optimizer simultaneously.
- No architecture mutation: the experiment uses the real Cymek V5 implementation with only `ModelSpec` scale changes.
- No P35-by-default rule: proxy size is subordinate to meaningful actual-token dose and replication.
- No teacher-forced capability gate: candidate-free complete generation with valid EOS is authoritative.
- No post-hoc transfer candidate: HYSTERETIC_HIGH_LOW and LOW_CONTINUE are fixed from prior evidence before CYR outcomes.
- No production scheduler mutation from GPU evidence.

## Runtime design

Hardware resolution uses only measured real training throughput, measured batched-generation throughput, model fit and the frozen wall budget. It first proves the minimum replicated design is affordable, then expands actual-token dose toward target. The resolver also emits prospective acquisition/retention/transfer stage budgets. Accuracy, loss and treatment outcomes are forbidden resolver inputs.

## Mechanism interpretation

The continuation red team reports parameter displacement, gradient norms and Adam-moment norms. It does **not** pretend those are a direct decomposition of consolidation. The transfer stage preserves full optimizer state, so a plasticity difference could arise from parameters, moments, or both. That ambiguity is recorded rather than hidden. A later mechanistic experiment may reset/match optimizer state if CYR-GPU-006 produces a strong effect worth decomposing.

## Claim boundary

A positive result can nominate a **GPU V5-proxy research candidate** only. It cannot certify TPU/XLA behavior, cannot modify the frozen production WSD schedule, cannot authorize PRE500M, and cannot authorize 500M. Those require later exact-SHA review, independent validation and real TPU evidence.
