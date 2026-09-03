# BOTTLENECK_RANKING.md

Candidate bottlenecks for the central question ("what prevents stronger transferable internal
cognition per parameter and per training token"), ranked primarily by
**expected information gain / experimental cost** — not novelty, not ambition.
Each candidate keeps the full field set; Priority is the rank order for Citadel's attention.

Fields: Evidence for / Evidence against / Uncertainty / Expected capability impact /
Experiment cost / Expected information gain / Dependency on stronger checkpoint /
Risk of confounding / Priority.

---

### B1. The candidate-selection measurement instrument is invalid (`production_scoring_mode = null`)
- Evidence for: preregistered, powered, replicated tournament failure (N3/E3); `production_scoring_mode: null` recorded in every downstream program gate; triquetra's training-program Stage 0 blocked on it; every future learned cognition comparison needs this metric.
- Evidence against: none — the failure is unchallenged; only the *scope* of the failure (policy family vs task construction) is unknown.
- Uncertainty: low that the blocker is real; high on which fix works.
- Expected capability impact: none directly (instrument, not capability) — but it gates *measurement* of every capability claim; unmeasured capabilities cannot be promoted.
- Experiment cost: very low. No training; untrained P35 weights; existing fixture/null/decoy machinery; ≤ ~2 GPU-hours.
- Expected information gain: very high — either a valid policy family is found (unblocks E1–E3 and all Citadel training experiments) or the task/candidate construction itself is indicted (reshapes benchmark design before any training claims are made).
- Dependency on stronger checkpoint: none (bias is a property of the scoring function + tokenizer; untrained weights suffice, as the null receipts already showed).
- Risk of confounding: low; the failed policies are built in as screen-validity negative controls and the oracle path as a positive control.
- **Priority: 1 — selected as C0.**

### B2. No trainable V5 path exists at any scale (trainer + tokenizer artifact + data glue missing)
- Evidence for: `production_launcher_implemented: false`, E1–E6 PENDING, six external identities null (E10); cymek explicitly trains nothing (Y1); every real-model result is borrowed V4 history.
- Evidence against: the contracts + canaries are complete and tested; the missing piece is glue (a caller-supplied backend), plus the provisional 24k tokenizer for development use.
- Uncertainty: low.
- Expected capability impact: high — nothing can be learned empirically until something can be trained.
- Experiment cost: medium — implementation effort, modest compute at P35/sub-P35 scale (RTX 4050-class local GPU proven adequate for canaries).
- Expected information gain: high but *derivative* — it enables Q-C/Q-D rather than answering anything alone.
- Dependency on stronger checkpoint: inverse — it is how stronger checkpoints get produced.
- Risk of confounding: medium (new code paths must pass Cymek's fail-closed invariants; implementation-validation rules §18 apply).
- **Priority: 2 — begin immediately after C0 in parallel with C0 analysis.**

### B3. Checkpoint capability floor — all real-model research is gated on a stronger checkpoint
- Evidence for: triquetra `WAITING_FOR_STRONGER_CHECKPOINT`; readiness v2 NOT_READY/INSUFFICIENT/NOT_IDENTIFIABLE (T6); P1 canary 8.3%, B0 raw 0.0; local inventory exhausted.
- Evidence against: it is a constraint on *research*, not yet a demonstrated cause of the cognition gap itself.
- Uncertainty: medium (what capability level restores identifiability is unmapped — Q-B).
- Expected capability impact: high (it is the substrate everything runs on).
- Experiment cost: high — producing a stronger checkpoint requires B2 plus significant training compute; not a cheap experiment.
- Expected information gain: high once B2 exists (the readiness-calibration sweep answers Q-B).
- Dependency on stronger checkpoint: it *is* the checkpoint problem.
- Risk of confounding: medium — readiness gate thresholds are reasonable but not independently validated.
- **Priority: 3.**

### B4. Training objective may not create query-conditioned pressure (CE-only launch)
- Evidence for: N1 (loss without cognition); T3 (copy/readout emerged, query control did not); ESOES froze λ=0 launch posture as a direct consequence (D-041) while implementing the query-swap challenger.
- Evidence against: no controlled test of λ>0 has ever run at any scale; absence of query control on a floor-limited substrate is weak evidence about the objective.
- Uncertainty: high.
- Expected capability impact: potentially decisive — this is the leading *scientific* explanation for the founding negative.
- Experiment cost: medium — micro-scale CE vs CE+λ (one variable) after B2; compute modest.
- Expected information gain: high — directly tests Q-D.
- Dependency on stronger checkpoint: none (trains its own); depends on B1 (valid scorer) and B2 (trainer).
- Risk of confounding: medium — λ adds parameters; FLOP-matching and loss-matched checks required.
- **Priority: 4.**

### B5. Curriculum/mixture unvalidated (0.15 cognition slice frozen by fiat)
- Evidence for: Y2 — fractions are implementation constants; E3 `BLOCKED_UPSTREAM_INPUTS`; ESOES's own data-responsibility doctrine rejects "more tokens = more intelligence".
- Evidence against: the E3 ladder (5/15/30%) is preregistered and contract-enforced — the *design* exists, only the execution is blocked.
- Uncertainty: medium-high.
- Expected capability impact: potentially large (capability per training token is exactly this question).
- Experiment cost: high at contract scale (200M-token screens × 3 arms); medium sub-scale.
- Expected information gain: high, but downstream of B1/B2.
- Dependency on stronger checkpoint: none (self-trained), but needs tokenizer artifact + manifests.
- Risk of confounding: medium — mixture changes interact with curriculum ordering; contract pins everything else.
- **Priority: 5.**

### B6. E0's discriminative power on trained models is unverified
- Evidence for: benchmark never scored a trained model (Q-H); sealed T2 custody absent.
- Evidence against: strong infrastructure certification and heuristic resistance (E4).
- Uncertainty: medium.
- Expected capability impact: indirect (measurement credibility).
- Experiment cost: low once B2's first model exists (piggybacks on Q-C).
- Expected information gain: medium-high; protects every later claim.
- Dependency on stronger checkpoint: none beyond any trained model.
- Risk of confounding: low.
- **Priority: 6.**

### B7. Tokenizer/data representation may destroy task-relevant structure
- Evidence for: E8 is TENTATIVE on a non-representative corpus; Q-G untested.
- Evidence against: roundtrip/unknown/determinism invariants all pass; perturbation sweep clean.
- Uncertainty: high but consequence currently unquantified.
- Expected capability impact: unknown.
- Experiment cost: low for static probes; higher for learning effects.
- Expected information gain: medium.
- Dependency on stronger checkpoint: none for static probes.
- Risk of confounding: medium (static structure ≠ learning-relevant structure).
- **Priority: 7.**

### B8. Optimization configuration (batch, LR, schedule)
- Evidence for: constants are hypothesis-grade (blueprint ~3e-4, 131k batch) and untested.
- Evidence against: standard WSD shape; no anomaly observed (nothing has trained).
- Uncertainty: high; information gain currently low.
- Expected capability impact: unknown.
- Experiment cost: medium-high (needs B2 + compute).
- Expected information gain: low until a cognition signal exists to optimize against.
- Dependency on stronger checkpoint: none; depends on B2.
- Risk of confounding: low.
- **Priority: 8.**

### B9. Architecture limits (memory modules, GQA, context, scale)
- Evidence for: none at the learning level; deferred list parked deliberately.
- Evidence against: ESOES rule — architecture comes after diagnosis (§22 of the operating instruction); canaries show local mechanism health.
- Uncertainty: high.
- Expected capability impact: unknown.
- Experiment cost: high.
- Expected information gain: low/undefined until B1–B4 produce diagnoses.
- Dependency on stronger checkpoint: partial.
- Risk of confounding: high (architecture changes multiply variables).
- **Priority: 9 — parked.**

### B10. Procedural gaps (sealed custody, remote durability, promotion signing)
- Evidence for: E10/Y1 — all absent by design until launch.
- Expected capability impact: none for diagnosis; blocks promotion only.
- Experiment cost: procedural.
- Expected information gain: low now.
- **Priority: 10 — respect, don't build.**

---

## C0 selection

**Primary target: B1.** It has the highest information-gain-to-cost ratio in the queue, no
upstream dependencies, built-in negative/positive controls, and its outcome changes what
Citadel does next under *every* branch of the result. Preregistration:
`experiments/C0/PLAN.md`.
