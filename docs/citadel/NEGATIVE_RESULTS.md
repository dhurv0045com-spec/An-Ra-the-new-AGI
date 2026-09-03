# NEGATIVE_RESULTS.md

Failures are evidence. None of these may be silently rerun, reinterpreted, or dropped.
"Should we repeat it?" is answered per the rule: repeat only if the original was confounded,
underpowered, mis-implemented, or targeted a materially different hypothesis.

---

### N1. Generic continuation (PGE) training produced loss but no tested cognition — the founding negative
- Hypothesis: better language-model loss implies better internal cognition.
- Expected: improved held-out loss with improved binding/context/composition probes.
- Observed: loss 2.1884 → 1.9710 while copy RAW 0/6, nonce-context 0/8, multi-fact FREE 0/48, composition FREE 0/12; selection ~chance. (core-vnext@054619f audit; recorded in `docs/esoes/EVIDENCE_AND_CONTEXT.md`.)
- Was implementation validated?: yes — token count certified (329,908,224); weight/optimizer movement verified (branch was founded partly on the 203/203-identical-tensors XLA stale-optimizer failure, now guarded by real-update invariants).
- What was ruled out: on that lineage, loss improvement alone did not create the probed abilities.
- What remains possible: probe floor effects; abilities not covered by that battery; SFT/EXP interventions *did* create capabilities — so the gap is about pressure, not impossibility.
- Should we repeat it?: partially — as a from-scratch micro-replication (C1 candidate).
- Why: the original is n=1 lineage on a V4 substrate; a controlled small-scale reproduction with today's E0 benchmark would convert a motivating anecdote into a measured baseline. Not urgent until the scorer exists.

### N2. E0 generator v0.3.0 state family was a false green (positional shortcut)
- Hypothesis: E0 v0.3.0 measured state tracking.
- Expected: heuristics near chance.
- Observed: bag-of-words pooled state accuracy 81.77%, lexical overlap 71.09% vs 13.89% chance; legacy versions 100% for latest_fact/nearest_position (`artifacts/e0/shortcut_repair_receipt.json`; DECISION_LOG D-022, 2026-08-31).
- Was implementation validated?: yes — the receipt records before/after with fixed suite hashing.
- What was ruled out: v0.3.0 (and an insufficient intermediate repair) as shortcut-free; a whole class of serialization-coupled shortcuts now has an analytic permutation null.
- What remains possible: unknown shortcut families in v0.4.0; sealed tier never exercised.
- Should we repeat it?: no.
- Why: repaired with calibrated nulls and pooled seeds; the repair is itself receipted. New heuristics can be added to the existing gate cheaply instead.

### N3. Both calibrated candidate scorers failed the preregistered bias screen
- Hypothesis: domain-PMI / contextual-calibration neutralize length bias in candidate scoring.
- Expected: selection deviation within TOST margin ±0.05 of null on decoy axes.
- Observed: fewest-token role selected 1.000 in all 15 CUDA cells (5 seeds × 3 tokenizers); `policy_survives_bias_screen: {domain_pmi: false, contextual_calibration: false}`; `production_scoring_mode = null` (`artifacts/e2/scoring_policy_development.json`).
- Was implementation validated?: yes — rotation geometry validated (first_position_equivalence 0.3333 in 15/15); position-bias negative control caught; parity receipts exact (486 paired scores, max abs err 0.0028).
- What was ruled out: these two calibrated policies; and (via nulls) sum/byte/token aggregation as production policies.
- What remains possible: scoring-policy families never tried — notably answer-blind generation-based matching (used successfully in triquetra but never screened) and changing candidate construction itself.
- Should we repeat it?: no rerun; but C0 extends the screen to the untried policy family. Repeat of the failed arms is built in as screen-validity negative controls.
- Why: cheapest path to unblocking `production_scoring_mode = null`, the recorded blocker for all learned cognition comparisons.

### N4. Scoring-policy fixture v1 invalidated before powered execution
- Hypothesis: fixture v1 (group % 3) was adequate.
- Observed: hidden label perfectly predictable from surface family — structural leak; invalidated in design review (BENCHMARK.md §8.1; DECISION_LOG 2026-09-01); replaced by schema-2 crossing every family×role cell.
- Was implementation validated?: n/a (caught pre-execution by design audit).
- What was ruled out: fixture v1 as bias-screen basis.
- Should we repeat it?: no — v2 is in place and was used for N3.

### N5. Native BF16 parameters + AdamW rejected for production
- Hypothesis: full-BF16 training (params + moments) is safe at P35.
- Observed: BF16 moments + clip-norm overshoot 1.0 by ~0.3% (`artifacts/e2/local_*_real_update_native.json`, status FAIL); FP32-master parity clean (worst loss rel err 0.000118).
- What was ruled out: native-BF16 optimizer path (D-028).
- What remains possible: other mixed-precision layouts; long-horizon effects of the adopted layout.
- Should we repeat it?: no. Why: deterministic invariant violation, cheap to re-verify if precision layout ever changes.

### N6. RoPE calibration first receipt FAIL (tolerances too strict)
- Observed: float32 round-off exceeded 2e-6/3e-6 limits → verdict MIXED/FAIL; retained as negative evidence; analytically revised to 5e-5; fresh conformance receipts pass vs float64 oracle (D-027).
- What was ruled out: nothing scientific — an over-strict instrument.
- Should we repeat it?: no. Why: already superseded by passing conformance receipts; retained deliberately as instrument-tolerance evidence.

### N7. Native GQA unusable on the Windows/PyTorch local stack
- Observed: native scaled-dot-product GQA math-backend only ("No available kernel" for flash/efficient): 5.20× MHA latency, 13.86× memory; repeat-KV equivalence ≤ 0.0078 max abs err (`artifacts/e2/local_cuda_attention_aggregate.json`).
- What was ruled out: native GQA kernels on this stack for local experiments.
- Should we repeat it?: only if the torch/driver version changes materially.

### N8. SFT7 margin objective falsified (inherited)
- Hypothesis: margin-style ranking objective improves intended rank-1 selection.
- Observed: lift Δ 0.1049 nats (CI [0.0420, 0.1697]) but rank-1 66→64/119 — hypothesis falsified (core-exp).
- Should we repeat it?: no. Why: clean preregistered falsification; margin objective also banned in cymek (`margin must be exactly 0`).

### N9. EXP v10/v11 pair-action composition claims declared contaminated (inherited)
- Observed: stale `candidates` bug, missing fixed baselines, irreproducible trainer (EVIDENCE_AND_CONTEXT.md).
- What was ruled out: those composition claims; the 166-VIE bank as a causal bank.
- Should we repeat it?: no. Why: the claims' provenance is broken; rebuilding them means new preregistered experiments, not reruns.

### N10. X1-REAL self-model "PASS" invalidated (triquetra)
- Hypothesis: pre-intervention observations predict which intervention repairs failures (prospective accuracy 0.9545).
- Observed: an always-negative predictor scores 0.9733 (cell prevalence 0.0267; oracle coverage 0.0889) — matrix is imbalanced; claim INVALID (`output/ibq_legacy_basis_verdict.json`; ledger row "X1-REAL-0 | INVALID").
- Was implementation validated?: the *instrument* failed validation (IBQ v1: degenerate probes, coverage fail, entropy not above matched nulls).
- What was ruled out: self-model training on that basis; trusting raw cell accuracy under imbalance.
- What remains possible: X0/X1 with a qualified basis on a capable substrate (gated, not dead).
- Should we repeat it?: only after IBQ qualifies a basis on a research-subject checkpoint.

### N11. IBQ v2-DEV harvest basis NOT QUALIFIED
- Observed: oracle coverage 0.0877, degenerate probes [1,6], fails G10 vs null families (`output/ibq_dev_harvest.json`).
- Implementation note (audit flag): observations show `output_len: 0`, `distinct_ratio: 0.0` on every listed failure — generations appear empty; harvest numbers should not be trusted beyond the gate verdict until this is explained.
- Should we repeat it?: no on this substrate (checkpoint floor is the limiting factor, not the probe design alone).

### N12. E5 duplication is template-bound on the weak substrate; E5 line closed
- Observed: E5dup−sham = 0.0 (p=1.0) under compound structural OOD; oracle ≈ chance (0.2417) there (`output/structural_ood_e5.json`). Ledger: "floor-limited weak V4 substrate — NOT a proof that addressing cannot work in a stronger Core"; "Do NOT train internalization off E5."
- Should we repeat it?: not on this substrate; reopen only on a qualified checkpoint with the same protocol.

### N13. Competitive-binding "beyond-length" effect NOT supported
- Observed: competing−filler ≈ 0 at L2–L4 (p=1.0); floor accuracies; ledger: "CBL-as-beyond-length NOT earned" (`output/competitive_binding_dev.json`).
- Audit flag: a *facilitating* competing-vs-filler difference at L1 (+0.125, p=0.002) is recorded but interpreted away at floor — carried here as an unresolved L1 anomaly, not a finding.
- Should we repeat it?: no at floor; revisit on a qualified substrate.

### N14. Entity-addressing claims from the binding factorial INVALID; interference threshold withdrawn
- Observed: P3_entity_dup_rate / P3_fact_dup_rate arrays were never populated ("Those runs never occurred" — helper returned 0.0 for empty arrays); interference threshold claim confounded (`output/binding_factorial.json`; ledger §7).
- What was ruled out: entity-marking/entity-duplication claims; the interference-threshold claim.
- What remains possible: entity×value decomposition — properly measured later as T1.
- Should we repeat it?: the missing cells were superseded by the preregistered entity×value factorial (T1) — do not rerun the broken version.

### N15. Causal elicitation decomposition downgraded
- Observed: selection contrast +0.3115 confounded (intervention also removed distractors, shortened context, moved and explicitly selected the fact) — "Does NOT cleanly isolate addressing" (`output/causal_decomposition.json`; ledger §6 downgrades to STRONG DEVELOPMENT CLUE).
- Should we repeat it?: superseded by the preregistered QV matrix (T2).

### N16. QV GATE1 marginal/NO — training on this basis not justified
- Observed: raw rank 0.2500 = chance; QCS CI includes 0 (both seeds); 60/80 rank1-but-genfail ("conditional realization gap atop chance-level ranking — not evidence of latent knowledge") (`output/query_value_evidence_dev.json` + rep; `AN_RA_PROGRAM.md`).
- What was ruled out: routing/self-model training justification on step-30400.
- Should we repeat it?: only on a stronger checkpoint per the arrival contract.

### N17. Readiness gate v1 READY verdict was a false green
- Observed: v1 emitted `READY_FOR_BINDING_CAUSAL_RESEARCH` on step-30400; v2 calibration on the same checkpoint returns NOT_READY / INSUFFICIENT / NOT_IDENTIFIABLE; v1 downgraded to `CALIBRATION_ONLY / GATE_V0_NOT_QUALIFIED` (commit 44643c7; `output/readiness_pilot_30400.json`, `output/readiness_v2_calibrate_30400.json`).
- What was ruled out: v1 gate thresholds; chance-naive accuracy reading (k=1 free-generation null now separated from k-way).
- Should we repeat it?: v1 — never; v2 — is the gate to use.

### N18. Six-group scorer pilot consumed; proposed six-group calibration rejected as underpowered
- Observed: DECISION_LOG 2026-09-01 records the pilot consumed and the proposed run rejected pre-execution; a one-group CUDA smoke passed execution-only and was discarded as non-evidence.
- What was ruled out: small-N scorer calibration as a path.
- Should we repeat it?: no. Why: power analysis already rejects it; the powered tournament (N3) superseded it.

### N19. The "+0.669 nats, p=0.018" query-conditioning SFT claim is UNVERIFIED
- Observed: ledger states the claim was attributed to `x1_real_receipt.json`, which does not contain it; "Do not repeat it as fact."
- Should we repeat it?: the claim is unattributable — treat as folklore until a receipt exists.

### N20. Readiness v2 calibration is a deliberate negative control (recorded for completeness)
- Observed: running the qualified v2 gate on the weak checkpoint *should* and does fail (`PRIMITIVE_CANARY_FAILED`, P1 0.083, P4 0.083; frontier unstable; `candidacy suspended`) — this is the gate proving it no longer passes a floor substrate (`output/readiness_v2_calibrate_30400.json`).
- Should we repeat it?: it *is* the repeat; keep as the standing negative control for gate drift.
