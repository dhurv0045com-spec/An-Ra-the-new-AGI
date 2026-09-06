# EXPERIMENT LOG (Arkenstone)

| ID | Question | Arms | Status | Result pointer |
|----|----------|------|--------|----------------|
| ARK-001 | Where is the first exact-match lift-off (dose) on the simplest symbolic tasks; does vocabulary change it? | T1-COMPACT / T1-BYTE / T2-COMPACT / T0-COMPACT / T1-COMPACT-LARGE | EXECUTED (this run) | experiments/ARK-001/RESULT.json + ANALYSIS.md |
| REDTEAM-BV2 | Does cymek's binding-v2 pair qualification survive independent replication with a trained escalation baseline? | 3 seeds x cymek's own suite + trained logistic n-gram | EXECUTED | experiments/BINDING-V2-REDTEAM/RECEIPT.json |
| ARK-002a | Does the T2 grokking transition saturate at 1.0 with extended budget; does lift-off replicate on seed 29? | T2-COMPACT (30-min box) + T1-COMPACT seed 29 | EXECUTED — see ERRATUM: OOD reached 1.0 on seed 13 (noisy 0.89-1.00 tail; sustained-G90 demonstrated); T1 lift-off replicated; T2 trajectory SINGLE-SEED. Preregistration: CLAIMED_PREEXECUTION_PLAN, NOT_GIT_TIMESTAMP_VERIFIED | experiments/ARK-002/ANALYSIS.md + ERRATUM_002a.json |
| ARK-002B | Independent T2 replication (seeds 29/47, frozen commutation-free manifest, separated init/order seeds) | EXECUTED — qualitative transition REPLICATED (both seeds); G90 dose seed-variable (12k vs >18k) | experiments/ARK-002B/ANALYSIS.md |
| ARK-003 | Can curriculum/aligned-teacher accelerate OOD emergence vs flat (4 matched arms, screening seed 29)? | EXECUTED — NULL for acceleration; B delayed memorization; C/D compute-handicapped; locality-OOD co-emergence tentative | experiments/ARK-003/ANALYSIS.md + REDTEAM.md |
