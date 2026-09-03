# OPEN_QUESTIONS.md

Unresolved questions extracted from the branch audit. Each is experimentally actionable;
none is assumed to be a confirmed problem. "Cheapest discriminator" names the smallest
experiment that would move belief. Ordering is by current relevance to Citadel, not importance.

---

### Q-A. Can any answer-blind selection policy pass the calibrated bias screen?
- Why open: `production_scoring_mode = null` after N3; every learned cognition comparison is blocked on a valid selection metric. Only likelihood-aggregation + two calibrated policies were ever screened.
- Existing evidence: `artifacts/e2/scoring_policy_development.json`; triquetra's behavioral generation scorer passed its own firewall but was never put through ESOES's screen.
- Cheapest discriminator: **C0** (preregistered in `experiments/C0/PLAN.md`).
- Status: OPEN — C0.

### Q-B. What minimum substrate capability makes binding mechanisms identifiable?
- Why open: triquetra is formally `WAITING_FOR_STRONGER_CHECKPOINT`; readiness v2 gives concrete thresholds (primitive canaries, Wilson-lo ≥ 0.5, legal headroom ≥ 0.05, diversity, power) but the capability→identifiability curve is unmapped.
- Existing evidence: `output/readiness_v2_calibrate_30400.json` (NOT_READY at P1 0.083); substrate-adequacy table in `AN_RA_PROGRAM.md`.
- Cheapest discriminator: a readiness-calibration sweep across checkpoints of different training levels — requires producing such checkpoints (blocked on Q-D's trainer work).
- Status: OPEN, blocked on producing stronger checkpoints.

### Q-C. Does "loss improves, cognition flat" reproduce from scratch at small scale?
- Why open: the founding negative (N1) is n=1 lineage on a V4 substrate with an older probe set. If it reproduces from scratch on the cognition curriculum with the current E0 benchmark, it becomes a measured baseline rather than a motivating anecdote; if it doesn't, V4-specific history was confounding it.
- Existing evidence: `docs/esoes/EVIDENCE_AND_CONTEXT.md` §10; E0 certificate (benchmark discriminates heuristics from nulls but has never scored a trained model).
- Cheapest discriminator: one short controlled P35-scale CE run on the E0 training curriculum, evaluated with E0 + heuristic nulls (needs C0's scorer first).
- Status: OPEN, blocked on Q-A + minimal trainer.

### Q-D. Does query-conditioned training pressure create query-conditioned selection?
- Why open: CE-only training built copy/readout but not query control (T3); the query-swap contrastive challenger is implemented and frozen (cymek `v5_objectives/query_swap.py`) but disabled pending E3 Phase B — never run at any scale.
- Existing evidence: N1, T2, T3; E3 Phase B preregistration (λ ∈ {0, 0.05, 0.15}, FLOP-matched).
- Cheapest discriminator: micro-scale CE vs CE+λ comparison, one variable, same tokens (blocked on Q-A, Q-C).
- Status: OPEN.

### Q-E. What cognition fraction maximizes cognition-per-token without substrate regression?
- Why open: the frozen 0.15 slice (and all nine family fractions) is an implementation decision (Y2), not a measurement; E3 Phase A (5/15/30% @ 200M-token screens) is `BLOCKED_UPSTREAM_INPUTS`.
- Existing evidence: `artifacts/e3/static_plan.json`; `v5_contracts/run_spec.py` fractions.
- Cheapest discriminator: E3 Phase A itself once a trainer and tokenizer artifact exist; sub-scale screens may be possible earlier at reduced token budgets.
- Status: OPEN, blocked upstream.

### Q-F. Is the value-recency effect (T1) a readout artifact or a representational fact?
- Why open: T1 is DEV-tier on one floor-limited checkpoint; bare-value insertion may exploit surface salience rather than reveal addressing. The unfrozen value-prior/position decomposition (position dominates query-match 19–35×) would discriminate, but no protocol is frozen.
- Cheapest discriminator: preregistered value-prior vs position decomposition on a qualified substrate (triquetra's own "next mechanism" candidate — Citadel must not run it until preregistered and the substrate qualifies).
- Status: OPEN, blocked on Q-B.

### Q-G. Does the current tokenizer/data representation preserve the structure the cognitive targets need?
- Why open: 24k BPE was chosen on a non-representative local corpus (E8 TENTATIVE); whether tokenization damages entity/value structure relevant to binding is untested.
- Cheapest discriminator: structural perturbation probes (entity nonce density, value ambiguity) scored through the E1 static harness — cheap, no model needed for first pass.
- Status: OPEN.

### Q-H. Does the E0 benchmark actually detect cognition in a trained model?
- Why open: E0 has passed infrastructure certification and resisted heuristics, but has never scored a trained model; its discriminative power on real trained-checkpoint variation is unknown; sealed T2 custody does not exist.
- Cheapest discriminator: Q-C's run doubles as this test (does E0 separate the trained model from untrained/heuristic baselines as designed?).
- Status: OPEN.

### Q-I. Is architecture limiting cognition, or is optimization preventing existing capacity from being used?
- Why open: no learning experiment exists at all; deferred architecture ideas (SSM/memory, MoE, byte-latent) are explicitly parked (blueprint deferred list).
- Cheapest discriminator: none yet — architecture questions stay parked until Q-A–Q-D produce a measurement layer and a baseline.
- Status: PARKED by discipline (ESOES rule: architecture comes after diagnosis).

### Q-J. Does task-family interference harm transfer?
- Why open: nine cognition families + natural/code slices train together by contract; interference has never been measured.
- Cheapest discriminator: family-holdout arms at micro scale (after Q-C establishes the baseline harness).
- Status: OPEN, downstream.
