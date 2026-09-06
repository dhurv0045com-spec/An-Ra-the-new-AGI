# ARK-003 RED TEAM (to be filled after screening results)

Reviewer: independent adversarial pass (mission section 12).
Job: explain any apparent arm-C/B win WITHOUT invoking improved cognition.

## Attack checklist (each must be answered with evidence from the receipts)
- [ ] Data leakage (train/test overlap incl. commutation; manifest sha check)
- [ ] Split weakness (ones-pair overlap documented in TASK_MANIFEST)
- [ ] Lucky seed (screening seed 29 only — winners must replicate on 47/101)
- [ ] Checkpoint cherry-picking (sustained thresholds only; max forbidden)
- [ ] Extra tokens (supervised-token accounting per arm)
- [ ] Longer sequences / more optimizer steps (steps fixed; suffix tokens recorded)
- [ ] Changed initialization (init_seed identical across arms)
- [ ] Changed data order (order_seed identical across arms)
- [ ] Teacher-answer leakage (suffix contains gold digits — C sees its OWN row's
      decomposition; the EVAL never contains suffixes; but note: C's training
      stream includes the gold answer inside the suffix — same as the answer
      itself; no eval leakage)
- [ ] Target duplication / frequency shifts (suffix repeats answer digits;
      count per-position target frequencies per arm)
- [ ] Parameter mismatch (same 20-token vocab for all arms; count recorded)
- [ ] Output formatting effects (eval decodes digits only; suffix never emitted
      at eval; but training on suffixes could change stopping behavior — check
      stop histograms)
- [ ] Early stopping asymmetry (all arms share the wall box; steps recorded)
- [ ] Task-generator bugs (manifest audit; determinism test)
- [ ] Evaluation bugs (greedy decode unit-tested; locality scorer reviewed)
- [ ] Repeated evaluation influencing decisions (eval cadence fixed at 200)

## Verdict (post-screening)
The strongest attack on any 'teacher fails' reading is the WALL-TIME HANDICAP:
C/D executed ~5300 steps vs A's 9600 because their suffix tokens cost ~2.7x
supervised tokens per step. The screening therefore supports only the narrow
claim: 'under equal wall-time budget including suffix cost, neither curriculum
nor aligned decomposition accelerated OOD emergence.' A step-matched rerun is
required before any strong teacher-fails claim. A's mid-transition snapshot
prevents checkpoint cherry-picking claims (nothing was won). Seed 29 is a
screening seed only; nothing advanced to replication. Sustained-metric
discipline held: no G90 was claimed for any arm despite A's 0.421 snapshot.
