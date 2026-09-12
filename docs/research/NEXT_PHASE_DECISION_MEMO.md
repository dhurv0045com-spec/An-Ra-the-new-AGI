# NEXT PHASE DECISION MEMO

**Date:** 2026-09-13 · **Basis:** `EXPERIMENT_EVIDENCE_LEDGER.json` (78 entries, validated by `tools/validate_evidence_ledger.py`, audited from primary artifacts across 14 live branches + history forensics).
**Standard applied:** every recommendation carries ACTION / WHY / EVIDENCE / UNCERTAINTY / FALSIFIER / CHEAPEST_NEXT_TEST. A recommendation with no realistic falsifier is not evidence-driven and has been excluded.

---

## 1. What have we actually demonstrated?

1. **Loss is not cognition** (R3 family): PGE continuation improved held-out loss 2.1884→1.9710 with all probed cognition at 0/chance; T1-series/T1D replicated the dissociation at TPU scale. [ESO-PGE, CIT-T1-series, CIT-T1D]
2. **Delayed generalization after memorization** replicates at micro scale (2 seeds); M99 does not predict G90 (ρ=0.00). [ARK-002B, ARK-004A]
3. **Representation/class-space controls formation**: with everything else fixed, declared tied class-space size moved held-out formation from 0% to 100% (V4096) vs 0% (V24576) and 12.94% (V19) at 512k rows; the intermediate 4096–16384 region is strongest across two fresh seeds; both monotonic stories are falsified. [CYR-011, CYR-012-R1, CYR-013-R1B]
4. **Retention is phase-dependent and levers are separable**: LOW LR protects an acquired Micro-T2 invariant (0/12 vs 9/12 failures; 0/6 vs 3/6 recurrence); HIGH recovers what immediate LOW cannot (8/9 vs 2/9); narrowed high-plasticity support erodes broader invariance while canonical accuracy stays 1.0 (8/8 vs 0/8 vs 0/8); and ARK-017 V2 showed lowered update magnitude (CAP1X) and sparse treatment-exact replay each protect independently — total parameter movement does not explain outcomes. [ARK-007R, ARK-010, ARK-011, ARK-015, ARK-017-V2]
5. **EOS supervision is mechanically required** (0/32→32/32 ×2 seeds; T1D's 15,000/15,000 MAX_TOKENS postmortem independently agrees). [BRM-terminal-EOS, CIT-T1D]
6. **Measurement can be validated and can fail loudly**: calibrated scorers fail bias screens (1.000 fewest-token in 15/15 CUDA cells), generator v0.3.0 false green was caught and repaired, readiness v1 false green was self-caught, X1 self-model PASS was invalidated by a trivial baseline (0.9545 vs 0.9733 always-negative at prevalence 0.0267). [CIT-scoring-policy-tournament, CIT-e0, TQ-readiness, TQ-X1]
7. **Real-text substrate behavior**: a ~20–25M model on peS2o+Birth assimilates a small repeated corpus but does not meet a preregistered content-internalization threshold, pays 3.7–3.9% sealed science NLL, and slows later binding acquisition ~4–5×. [ARK-018-V4]
8. **New-skill formation gates continual learning**: no controller verdict is interpretable before the unprotected reference acquires (T3 never acquired in ARK-013; SKILL_B never formed in any ARK-019 V3.1 arm — official verdict `CONTROLLER_NOT_SUPPORTED`). [ARK-013, ARK-019-V3.1]

## 2. What have we falsified?

Curriculum/teacher acceleration (two programs); EMA/WD-removal stabilization; "LOW LR is universal"; "large movement causes forgetting"; exact LR-switch thresholds; memorization-timing prediction; precursor predictors; whole-vocab first-order effects at micro lift-off; monotonic vocabulary stories in both directions; query-blind-explainable binding claims as "selection"; E5-template internalization; margin objectives; EXP v10/v11 composition claims; X1 self-model; readiness-gate v1; calibrated candidate scorers; data-variety ⇒ query control; learned discovery > random (at scale); depth-two inquiry > one-step (at budget 2); "this-regime replay rescues retention"; native-BF16 optimizer states.

## 3. What is merely implemented?

- **CYR-GPU-014-R1C** — frozen, twice engineering-repaired, unexecuted (and launch-blocking until its launcher is rebound to a pushed commit).
- **ARK-020 V1→V4** — 4-skill continual battery, 39/39 tests, no executed run in-repo; A1 durability amendments pending validation.
- **ARK-021** (retention-vs-reacquisition; core+tests), **ARK-022** (dormant retention; plan), **CIT-T1E** (EOS-corrected successor; plan), **ESO-E3** (mixture screens; blocked upstream), **Cymek production corpus/entry-point** (missing components), **P35A** (gated on external identities).

## 4. What remains speculative?

Capability Guardian as a *prevention* mechanism (V4 candidate is transcribed-only); inactive-softmax competition as THE mechanism (R1C's hypothesis); any dose-universal replay fraction; scale transfer of every Micro law; the 65/20/15 mixture; the 5B WSD schedule constants; ARK-022 dormant retention; portfolio ranks below the top three.

## 5. The 10 largest unknowns (ranked by information gain × impact ÷ cost)

1. **Is inactive-softmax competition the causal carrier of the class-space effect?** — R1C is designed to answer; blocks ARK-025 and any tokenizer decision.
2. **Does the Guardian result survive byte-level audit?** — the entire continual-learning direction currently rests on a transcription; also blocks ARK-020 interpretation.
3. **Does the class-space/formation effect transfer beyond the arithmetic micro-task** (larger models, natural text, real tokenizer)? — nothing yet tests it.
4. **What is the minimum viable new-skill dose** that lets SKILL_B form while science competence is preserved (V4/ARK-020 dose-selection machinery exists)?
5. **Which retention lever is necessary in which regime** (cap vs replay interaction), and do doses transfer beyond micro?
6. **Can a certified candidate-free scorer + fresh-tier replication be achieved** (unblocks every assisted-scoring comparison; `production_scoring_mode` still null)?
7. **Does the production 24,576 representation fail on natural text the way it fails on the arithmetic bridge** (or is the failure task-specific)?
8. **What does the corpus actually contain** (production 5B mixture unqualified; E3 mixture screens blocked)?
9. **Is the checkpoint subject pool qualifiable** (Triquetra readiness v2 says no local subject qualifies — unblocks all mechanism instrumentation when fixed)?
10. **What is in the unreachable history** (SENORA program, CYR-GPU-006 smoke, frozen CYR-005 executable) — cheap recovery, nonzero information, before `git gc` destroys it.

## 6. The three highest-information next experiments

| # | Experiment | Why it dominates | Depends on |
|---|---|---|---|
| 1 | **CYR-GPU-014-R1C** (after launcher rebind) | Converts the program's strongest discovery from correlation to mechanism; dual structural/functional endpoints pre-registered; its result redirects either the production-representation design or the initialization/geometry search; unblocks ARK-025 | launcher rebind (minutes of work) |
| 2 | **ARK-019 V4 bundle recovery + byte audit** (or controlled rerun) | Decides whether continual-learning controllers are a live direction; zero new GPU needed if the bundle exists externally; then ARK-020 runs on the same substrate | operator Drive access |
| 3 | **Transfer probe of the class-space effect**: replicate V4096-vs-V24576 at one larger scale / one natural-text-mapped task with matched everything (e.g. 8L/256w, or a number-formatting task over the production tokenizer) | Tests the only effect with formation-level effect size for scale generality; cheap relative to its gating value for 500M design | R1C outcome (to know which mechanism variant to carry) |

## 7. What should NOT be built yet

- Any production tokenizer/vocabulary change (mechanism unverified; natural-language transfer untested).
- Any Guardian/continual controller in production code (V3.1 negative; V4 unverified; CAP16X value undemonstrated).
- Any 500M campaign (corpus MISSING, entry point MISSING at last audit; no scientific authorization; R1C/transfer unknowns open).
- Exotic architecture (MoE/SSM/recurrence/latent-thought/neural memory) — no bottleneck evidence demands them.
- A learned experiment-selection controller (BRAMASTRA discovery null stands).
- Any query-conditioning training objective on unqualified substrates (three past claim deaths).
- Re-running anything in the falsified list (§2) without a materially different hypothesis.

## 8. What must be true before a 500M-scale run is rational

1. Production corpus materialized, manifest-bound, dedup/contamination-qualified (external gate).
2. Top-level production entry point wired (Citadel blocker #14) and canonical token-indexed WSD executed at least once end-to-end (blocker #9).
3. Tokenizer identity resolved — including the R1C mechanism verdict, because vocabulary size is no longer a free parameter: it has a demonstrated 0%↔100% formation effect at development scale.
4. At least one capability (arithmetic or transfer probe) demonstrably forms held-out on the production representation at development scale, replicated multi-seed.
5. Sealed evaluation fixtures committed and consumed exactly once; milestone evaluation wired.
6. Explicit PRE500M green decision recorded with hashes.

## 9. Components that deserve promotion into the next Core

- **EOS-supervised answer+termination contract** (R3 across three programs) — already in Cymek contracts; keep.
- **Candidate-free primary evaluation + orthogonal invariance axes** (canonical/query/order/query+order separate; COMMUTED lesson) — keep.
- **Shared-parent matched-fork retention harness** (ARK-007R/011/015/017 pattern) — promote into Cymek's retention-controller testing.
- **Formation-first gating** (V4/ARK-020 dose selection + reference-must-acquire rule) — promote into any continual-learning experiment template.
- **Readiness-v2-style subject qualification** before any mechanism study — promote Triquetra's gate into the common toolkit.
- **Fail-closed launch/hardware gates with progressive fixed-wall fallback** (CYR-008 → 009 lesson; ARK-020 A1 amendments).
- **Treatment-exact sparse replay construction** (ARK-017) as the reference replay implementation, dose TBD.

## 10. Assumptions that could invalidate the current direction

1. **"The class-space effect is about the softmax denominator"** — R1C could falsify; then the tied-geometry/optimization search restarts.
2. **"Micro-scale retention laws transfer to production scale"** — no test exists; the entire LOW/CAP/replay policy toolbox could be regime-specific.
3. **"The ARK-019 V4 Guardian candidate is real"** — if the bundle cannot be produced or fails re-audit, the continual-learning program loses its only positive signal and falls back to V3.1's formation-bottleneck interpretation.
4. **"The production tokenizer is adequate for language even though it fails the arithmetic bridge"** — plausible but untested; a natural-text formation failure would invalidate the 500M data plan wholesale.
5. **"Eval fixtures are clean"** — one more fixture leak like scorer-fixture v1 would invalidate any sealed claim built on them.
6. **"Unreachable history is worthless"** — if SENORA's dry-run receipts encode a validated cheaper campaign design, recovering them changes the cost model.
