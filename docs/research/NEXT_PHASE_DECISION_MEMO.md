# NEXT PHASE DECISION MEMO

**Date:** 2026-09-13 · **Basis:** `EXPERIMENT_EVIDENCE_LEDGER.json` (81 entries at the last validated Task-1 snapshot) plus the completed external `CYR-GPU-014-R1C` result bundle recorded in [`R1C_FINAL_EVIDENCE_2026-09-13.md`](R1C_FINAL_EVIDENCE_2026-09-13.md).  
**Standard applied:** every recommendation carries ACTION / WHY / EVIDENCE / UNCERTAINTY / FALSIFIER / CHEAPEST_NEXT_TEST. New external evidence may update a decision without retroactively rewriting the historical ledger entry count until the ledger itself is revalidated.

---

## 1. What have we actually demonstrated?

1. **Loss is not cognition** (R3 family): PGE continuation improved held-out loss while probed cognition stayed at 0/chance; T1-series/T1D replicated the dissociation at TPU scale. [ESO-PGE, CIT-T1-series, CIT-T1D]
2. **Delayed generalization after memorization** replicates at micro scale. M99 does not imply G90. [ARK-002B, ARK-004A]
3. **Physical representation/class-space controls formation**: changing the actual tied class-space size produced large, non-monotonic formation differences, with V4096 strongly outperforming V24576 in the controlled developmental studies. [CYR-011, CYR-012-R1, CYR-013-R1B]
4. **R1C rules out inactive-softmax competition as a sufficient carrier of that physical class-space effect.** All 24/24 arms completed. `MASK_4096 - FULL_24576` formation-AUC gaps were `[-0.139792, -0.242215, -0.019377, -0.040138]`, mean `-0.110381`; functional and structural primary tests both returned unsupported / not sufficient. [CYR-GPU-014-R1C]
5. **Retention is phase-dependent and levers are separable**: lower plasticity and sparse treatment-exact replay can each protect acquired invariants under tested regimes; total parameter movement alone does not explain outcomes. [ARK-007R, ARK-010, ARK-011, ARK-015, ARK-017-V2]
6. **EOS supervision is mechanically required** across independent programs. [BRM-terminal-EOS, CIT-T1D]
7. **Measurement can be validated and can fail loudly**: several apparently positive systems were correctly rejected by bias, baseline, or readiness attacks. [CIT-scoring-policy-tournament, CIT-e0, TQ-readiness, TQ-X1]
8. **Real-text substrate behavior**: a ~20–25M model can assimilate a small repeated corpus while still failing a preregistered internalization threshold and paying measurable science/plasticity costs. [ARK-018-V4]
9. **New-skill formation gates continual learning**: no controller verdict is interpretable before the unprotected reference can acquire the new skill. [ARK-013, ARK-019-V3.1]

## 2. What have we falsified?

The previous falsification list still stands (curriculum/teacher acceleration in tested regimes; EMA/WD-removal stabilization; universal LOW-LR claims; movement-causes-forgetting; several timing/predictor/objective claims; monotonic vocabulary stories; query-blind binding claims; T1/X1 false positives; scorer/readiness false greens; replay-universal claims; native-BF16 optimizer-state assumptions).

**New falsification from R1C:**

> The earlier physical V4096 advantage is primarily explained by inactive classes competing in the 24,576-way softmax denominator during training.

R1C held the physical 24,576 tied matrix fixed and removed output competition in several ways; the preregistered MASK_4096 mechanism did not reproduce the physical V4096 advantage. This sharply redirects the causal search toward physical embedding/output geometry, parameterization/initialization, optimization geometry, tokenization/representation, or interactions among them.

## 3. What is merely implemented or still externally unresolved?

- **ARK-020 V4** — executed externally/partially according to operator state, but final scientific completion must be established from its final result receipt before promotion.
- **ARK-021** (retention-vs-reacquisition), **ARK-022** (dormant retention), **CIT-T1E** (EOS-corrected successor), **ESO-E3** (mixture screens; blocked upstream), and production corpus/entry-point work remain separate lines.
- **V5.1 canary** exists on `cymek-v51-canary`; its own result is a development-scale execution/formation result and must not be conflated with R1C's output-space mechanism question.

R1C is **no longer** in this category: it is executed and complete.

## 4. What remains speculative?

- Which **physical** class-space mechanism carries the V4096↔V24576 formation difference.
- Whether the effect transfers to larger models, natural-text / production-tokenizer tasks, and clean non-arithmetic surfaces.
- Whether 24,576 is adequate or optimal for the eventual production representation.
- Guardian as a production prevention mechanism until final audited multi-set evidence is complete.
- Dose-universal replay fractions and scale transfer of Micro retention laws.
- The 65/20/15 mixture and 5B WSD constants until production-path evidence exists.

Inactive-softmax competition is no longer the leading sufficient-mechanism hypothesis; R1C materially downgraded it.

## 5. The largest remaining unknowns

1. **What physical geometry/representation factor actually carries the class-space formation effect?** R1C eliminated the simplest denominator-competition explanation.
2. **Does the physical V4096↔V24576 effect transfer** to a larger development model and a clean production-tokenizer/natural-text-mapped task?
3. **Does the Guardian/continual-learning result survive final byte-level audit and matched-set completion?**
4. **What minimum viable new-skill dose** permits formation while preserving science competence?
5. **Which retention lever is necessary in which regime**, and how doses transfer beyond Micro.
6. **Can production evaluation become candidate-free, attack-screened, and clean enough for promotion decisions?**
7. **What does the intended production corpus actually contain**, and is it large/clean enough for the target training plan?
8. **Can the canonical WSD / production execution path run end-to-end with trustworthy receipts?**
9. **Can development-scale formation replicate multi-seed on the chosen production representation?**
10. **What useful evidence remains in unreachable history**, before it is lost to pruning?

## 6. Highest-information next experiments

| # | Experiment | Why it dominates | Depends on |
|---|---|---|---|
| 1 | **CS-TRANSFER-001 — actual physical V4096 vs V24576 transfer** | R1C says masking is not the carrier; the next decisive test must change the real tied geometry. Use a larger development rung plus clean attack-screened / production-tokenizer rendering and matched seeds. | clean generator/eval surface |
| 2 | **ARK-020 / Guardian final audit or completion** | Determines whether continual-controller work has a production-scale signal beyond static replay/caps. | final operator receipts / matched-set completion |
| 3 | **Physical-class mechanism dissection after transfer** | If transfer survives, separate matrix-size/parameterization/init/tokenization causes with one-factor matched interventions. If transfer fails, do not spend further mechanism GPU. | outcome of CS-TRANSFER-001 |

## 7. What should NOT be built yet

- **Do not promote MASK_4096 or any masked-output treatment into production.** R1C did not support it.
- Do not infer that FULL_24576 is optimal; R1C was a sufficiency test, not an optimality proof.
- Do not make a production tokenizer/vocabulary change before physical transfer evidence.
- Do not schedule a 500M campaign: corpus/evaluation/representation/scale gates remain open.
- Do not add exotic architecture (MoE/SSM/recurrence/latent-thought/neural memory) without bottleneck evidence.
- Do not promote Guardian into the production Core before its final evidence clears.
- Do not rerun falsified experiments without a materially different causal hypothesis.

## 8. What must be true before a 500M-scale run is rational

1. Production corpus materialized, manifest-bound, deduplicated and contamination-qualified.
2. Production entry point wired and canonical token-indexed WSD executed end-to-end with durable receipts.
3. **Physical tokenizer/output geometry resolved by transfer evidence.** R1C has closed the softmax-competition mechanism question but explicitly did not authorize a tokenizer change.
4. At least one capability demonstrably forms held-out on the chosen production representation at a larger development rung, replicated multi-seed.
5. Sealed evaluation fixtures are clean, hash-bound and consumed exactly once.
6. Citadel PRE500M green decision recorded with hashes.

R1C itself records `production_tokenizer_change_authorized: false`, `pre500m_authorized: false`, and `training_500m_authorized: false`.

## 9. Components that still deserve promotion into the next Core/tooling

- EOS-supervised answer+termination contract.
- Candidate-free primary evaluation + orthogonal invariance axes.
- Shared-parent matched-fork retention harness.
- Formation-first gating for continual-learning experiments.
- Subject/readiness qualification before mechanism studies.
- Fail-closed launch/hardware/checkpoint gates.
- Treatment-exact sparse replay as a reference experimental implementation, dose unresolved.
- Canonical full-softmax path for V5.1 canaries **only as the conservative default**, not because R1C proved it optimal.

## 10. Current representation decision

**ACTION:** keep V5.1 Candidate A / canonical full-softmax as the conservative canary path; reject masked-output Candidate B promotion.  
**WHY:** R1C's primary mask treatment underperformed FULL_24576 on formation AUC in every matched seed.  
**EVIDENCE:** 24/24 completed arms; mean paired gap `-0.110381`; bundle SHA-256 `2a9359e49f792f962774e42b4e5825b93fb7c91ddcad8f530c1c6f16c6f9bf9e`.  
**UNCERTAINTY:** R1C held the physical matrix fixed, so physical vocabulary geometry/tokenization remains unresolved.  
**FALSIFIER:** a clean, matched actual-geometry transfer study showing no meaningful V4096↔V24576 effect.  
**CHEAPEST_NEXT_TEST:** `CS-TRANSFER-001` at development scale with actual physical output matrices and clean attack-screened evaluation.
