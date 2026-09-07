# ARK-007R RED TEAM

Positive claim under attack: lowering LR from `1e-3` to `1e-5` after confirmed G90 causally reduces post-generalization instability on canonical Micro T2.

- Fresh checkpoints: PASS — seeds 909/1010/1111.
- Paired continuation: PASS — HIGH/LOW pair shares continuation seed and order hash.
- Treatment timing: PASS — onset/confirmation distinguished; treatment begins after confirmation.
- Reverse failures: none; LOW collapse = 0/12.
- Acquisition-level direction: PASS on all 3 checkpoints.
- Independence caveat: 12 forks are nested within 3 acquisitions.
- Collapse semantics: several HIGH arms later recovered; `collapse90` is an instability event, not necessarily irreversible forgetting.
- Freeze alternative: mean relative displacement HIGH ≈0.379 vs LOW ≈0.008; protection may primarily be reduced parameter motion.
- Transfer: NOT DEMONSTRATED.
- Scale: Micro only.

**Verdict:** credible matched-pair causal effect inside the T2 micro regime; mechanism interpretation remains open.
