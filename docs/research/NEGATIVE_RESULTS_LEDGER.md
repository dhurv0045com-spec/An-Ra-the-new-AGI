# NEGATIVE RESULTS LEDGER

**Synthesis date:** 2026-09-13. Consolidated from Arkenstone `docs/arkenstone/NEGATIVE_RESULTS.md`, Citadel `docs/citadel/NEGATIVE_RESULTS.md` (N1–N20), Cymek `docs/cymek/experiments/NEGATIVE_RESULTS.md`, the experiment RESULT files themselves, and this audit's cross-checks. Negatives are first-class results: none of these may be silently rerun, reinterpreted, or dropped. Repeat only if the original was confounded, underpowered, mis-implemented, or targeted a materially different hypothesis.

Legend: **[E]** executed scientific negative · **[F]** falsified/invalidated positive claim · **[I]** instrument/infrastructure failure with design consequence · **[G]** engineering-only failure (no scientific content).

---

## A. The founding dissociations

| # | Result | Class | Rules out / design consequence | Evidence |
|---|---|---|---|---|
| A1 | Generic continuation (PGE, 329.9M certified tokens) improved held-out loss 2.1884 → 1.9710 while copy/context/multi-fact/composition probes stayed 0 or chance | [E] | Loss is not a cognition criterion; behavior beats loss everywhere | esoes EVIDENCE_AND_CONTEXT §10; citadel E1 |
| A2 | T1C/T1D: substantial loss decrease with TEST exact 0–6.6% across six arms and all T1-series arms (raw 0/1,000) | [E] | At 2–8M tokens, CE training learns distribution/format without exact computation; curriculum, teacher, 2× scale, output masking, self-knowledge each insufficient for lift-off at tested budgets | citadel T1D RESULTS.md |
| A3 | Teacher primitives reached 0.515 held-out while composed T2+ arithmetic stayed ≈ 0 | [E] | Primitives do not automatically compose; measure composition explicitly | T1D POSTMORTEM 3 |
| A4 | M99 (memorization) timing does not predict G90 (generalization) timing, ρ = 0.00; precursor selectivity failed 3/4 | [E] | Do not stop training at train saturation; do not trust precursor predictors | ARK-004A + REANALYSIS |
| A5 | BRAMASTRA binding-diversity: 48.4% fresh accuracy fully explained by query-blind copying (deterministic 50% baseline; same answer in 62/64 despite changed query) | [E] | Data variety alone does not create query control; aggregate accuracy hides shortcuts | bramastra binding_diversity_dev_seed601 |

## B. Optimization / retention

| # | Result | Class | Rules out / design consequence | Evidence |
|---|---|---|---|---|
| B1 | "LOW LR is universally better" — false: immediate LOW recovered collapsed capability only 2/9 vs HIGH 8/9 | [E] | `collapse → lower LR immediately` is falsified at Micro T2; recovery needs plasticity | ARK-010 |
| B2 | LOW LR does not preserve old T2 under 12k no-replay T3-only training: every arm lost sustained T2 | [E] | LR alone is not a continual-learning solution; replay/support must be measured | ARK-013 |
| B3 | Exact LR-switch threshold not identified: `TIME_NOT_STATE_SCREEN`, thresholds alias | [E] | Do not promote 0.85/0.90 as constants; keep FIXED_TIME controls | ARK-012 |
| B4 | Update-cap mechanism credit unassignable at 1/12 event rate | [E] | Event-starved screens cannot assign credit; redesign event generation (done by ARK-017 V2) | ARK-016 |
| B5 | "Large parameter movement causes forgetting" — insufficient: AUGMENTED_HIGH moved farther and retained; ARK-017 sparse replay protected despite movement > failing HIGH | [E] | Total displacement is not the causal variable; support × plasticity interaction is the better model | ARK-015, ARK-017-V2 |
| B6 | EMA-0.999 and weight-decay removal do not prevent decay; no consolidation winner | [E] | Naive stabilizers rejected at micro scale | ARK-005 |
| B7 | Native BF16 AdamW moments overshoot clip-norm ~0.3% (invariant violation) | [I] | BF16 compute + FP32 master params adopted (D-026/D-028) | citadel N5 |
| B8 | XLA per-microstep all-reduce of the ACCUMULATED buffer scaled early-microstep gradients by replica-count powers | [G] | One collective at the accumulation boundary + CPU oracle negative regression; hardware status stays pending certification | cymek closure cycle |
| B9 | CLIP_BREACH: post-clip norm 1.0000042915 exceeded 1.0+1e-6 on R1C arm S1_MASK_8192 — float32 reduction-order noise ~4.3e-6 at 4.13M params | [G] | Tolerance 1e-6→1e-4 with documented derivation + e2e preflight test; zero scientific content | CYR-GPU-014-R1C EXPERIMENT_LOG |
| B10 | T4 CUDA SDPA backward nondeterminism broke exact-resume equality (CPU passed) for ARK-020 V4 | [G] | Deterministic kernels enforced inside the exact-resume smoke; launcher repinned `f4244e2` | Arkenstone f4244e2 |
| B11 | R1C optimizer constructor `TypeError: betas` at exact-resume smoke | [G] | Wrapper repair, Amendment 1; protocol unchanged | RUN_READINESS_V4 |
| B12 | CYR-GPU-004 fork contract: independent acquisitions per LR, parent snapshot never restored | [I] | Killed pre-execution; shared-parent design became mandatory | CYR-GPU-004/SUPERSEDED.md |
| B13 | CYR-GPU-008 all-or-nothing preflight overconstrained for free-Colab hardware (calibration passed, resolver rejected) | [I] | Progressive fixed-wall + partial-receipt technique adopted | CYR-GPU-008/SUPERSEDED_AFTER_CALIBRATION.md |

## C. Curriculum / data / supervision

| # | Result | Class | Rules out / design consequence | Evidence |
|---|---|---|---|---|
| C1 | Easy→hard curriculum delayed memorization and yielded zero OOD in box | [E] | Staging is not automatically helpful at micro scale | ARK-003; converged with T1D arm B |
| C2 | Aligned digit-decomposition teacher did not shorten the delay at equal wall budget; teacher pool diversity exhausted (10–13 unique rows/kind replayed ~2,500–3,260×) | [E] | Diversity of teacher data, not repetition, is the binding constraint | ARK-003; T1D POSTMORTEM 3 |
| C3 | 15,000/15,000 T1D generations ended MAX_TOKENS — EOS never a supervised target | [I] | Termination vs content failure inseparable there; contract corrected: supervise answer + EOS, +1 generation step (adopted cross-program; BRAMASTRA 0/32→32/32 ×2 seeds) | T1D POSTMORTEM 1; BRM-terminal-EOS |
| C4 | T1D self-knowledge arm invalid: 57/96 probe targets exceed MAX_ANSWER_TOKENS=8 | [I] | Self-knowledge negative is not clean; corrected contract required | T1D POSTMORTEM 4 |
| C5 | Vocabulary dead-space (H-REPR) not first-order at micro lift-off; capacity pathology (H-FLOOR) not the universal anomaly explanation | [E] | Do not attribute the citadel anomaly to either alone | ARK-001 |
| C6 | Production 24,576 BPE representation: train M99 but **0% held-out STANDARD and 0/48 SEALED at 100% of the ARK-002B exposure box** on real V5 | [E] | Semantic dose alone is not sufficient under that representation; representation/output-space is the primary formation bottleneck | CYR-GPU-011 |
| C7 | Simple monotonic stories falsified: "smaller vocab always better" (V19 12.94% < V4096 100%) and "more classes/params always better" (V24576 0%) | [E] | Non-monotonic class-space response; parameter displacement does not explain capability | CYR-GPU-012-R1 |
| C8 | "This-regime replay rescues retention" rejected (BRAMASTRA) | [E]* | Replay construction matters; do not conflate with ARK-017's treatment-exact sparse replay, which DID protect. *Original receipt not re-located in this audit — provenance gap | cymek NEGATIVE_RESULTS carry; EVIDENCE_GAPS |
| C9 | T1D budget confound: D/E arms change size/masking AND budget simultaneously | [I] | Do not conclude "scale does not help" or "masking does not help" from T1D | T1D POSTMORTEM 5 |

## E. Evaluation / measurement

| # | Result | Class | Rules out / design consequence | Evidence |
|---|---|---|---|---|
| E1 | Both calibrated candidate scorers (domain-PMI, contextual calibration) select the fewest-token role 1.000 in 15/15 CUDA cells; `production_scoring_mode: null` | [E] | No learned-cognition comparison via assisted ranking until a certified scorer exists; candidate-free generation is primary | citadel scoring_policy_development.json |
| E2 | Scoring fixture v1 (group % 3) had a structural leak (hidden label predictable from surface family) | [I] | Invalidated pre-execution; schema-2 crossing adopted | citadel N4 |
| E3 | E0 generator v0.3.0 false green: bag-of-words 81.77% / lexical 71.09% vs 13.89% chance | [I] | Calibrated + permutation nulls adopted; v0.4.0 receipted | citadel N2 |
| E4 | COMMUTED=100% in CYR-GPU-011 is NOT commutation invariance: reversing operands also changed the OOD tens-band role | [I] | A metric name is not evidence the metric isolates what it claims; post-run audit downgrade | CYR-GPU-011 RESULT correction |
| E5 | ARK-009 composite query-swap diagnostic changed query AND order together | [I] | Cannot attribute to query conditioning; orthogonal axes mandatory | ARK-009 |
| E6 | TQ readiness gate v1 emitted READY on a floor substrate (false green, self-caught; v2 returns NOT_READY) | [I] | A self-diagnosis basis on a floor-limited substrate can look accurate because the evaluation basis is degenerate | triquetra readiness artifacts; citadel N17/N20 |
| E7 | X1-REAL self-model "PASS" (0.9545) invalidated by always-negative baseline 0.9733 at prevalence 0.0267 | [F] | Never trust raw accuracy under class imbalance; constant baselines mandatory | TQ-X1-REAL |
| E8 | Readiness-gate false green + X1 false green are stale strings in immutable receipts (`x1_real_receipt.json`, `readiness_pilot_30400.json`) | [I] | A receipt's verdict field is never authoritative without its ledger | citadel cross-cutting observation 1 |
| E9 | TQ binding-factorial entity-duplication arrays never populated (helper returned 0.0 for empty arrays); entity-addressing + interference-threshold claims withdrawn | [F] | Silent zero-coercion is a provenance hazard; claims withdrawn | citadel N14 |
| E10 | SFT7 margin objective: lift +0.1049 nats but rank-1 66→64/119 | [F] | Margin objectives banned in cymek (margin must be exactly 0) | citadel N8 |
| E11 | EXP v10/v11 composition claims contaminated (stale candidates, missing baselines, irreproducible trainer) | [F] | Claims withdrawn; 166-VIE bank not a causal bank | citadel N9 |
| E12 | "+0.669 nats, p=0.018" query-conditioning SFT claim unattributable (cited receipt does not contain it) | [F] | Folklore until a receipt exists | citadel N19 |
| E13 | IBQ v2-DEV harvest basis NOT QUALIFIED (oracle coverage 0.0877) with suspected empty generations (`output_len: 0`) | [I] | Harvest numbers untrustworthy beyond the gate verdict | citadel N11 |
| E14 | E5 duplication assist is template-bound (E5dup−sham = 0.0, p=1.0 under structural shift) | [E] | E5-internalization line closed; do not train internalization off E5 | citadel N12 |
| E15 | Competitive-binding beyond-length effect not supported at floor; one unresolved L1 +0.125 anomaly | [E] | Revisit only on a qualified substrate | citadel N13 |
| E16 | Causal elicitation decomposition downgraded (intervention confounded: removed distractors, shortened context, moved + selected the fact) | [I] | Superseded by the preregistered QV matrix | citadel N15 |
| E17 | Six-group scorer calibration rejected pre-execution by power analysis; one-group smoke discarded as non-evidence | [I] | Small-N calibration is not a path | citadel N18 |
| E18 | Learned discovery policy does not beat random (n.s., 2 seeds); depth-two inquiry ≈ one-step teaching (Δ ≈ 0–0.02) | [E] | No controller/discovery advantage at these scales | BRM-discovery-dev, BRM-D02 |

## F. Mission-level mythologies corrected by this audit

| # | Mythology | Correction |
|---|---|---|
| F1 | "ARK-017 not executed" (master doc 2026-09-10) | ARK-017 **V2 EXECUTED**, verdict `BOTH_LEVERS_SUFFICIENT` |
| F2 | "ARK-018 implemented but not executed" | ARK-018 **V4 EXECUTED and audited**; primary threshold NOT met |
| F3 | "ARK-019 blocked / not implemented" | ARK-019 **V3.1 EXECUTED** (`CONTROLLER_NOT_SUPPORTED`); V4 has a transcribed external result awaiting byte re-audit |
| F4 | "ARK-020 V4 completed externally" (mission-brief belief) | Audited as **NOT in the repository**; excluded from design inputs; launcher work continued through 2026-09-13 |
| F5 | "CYR-GPU-013-R1B ready, not executed" (CURRENT_STATE.md) | R1B **EXECUTED/COMPLETE** with bundle `7ffebfd4…` |
| F6 | "T1C core exact 0/500" | Raw receipts say **0/1,000** (BRAMASTRA audit correction) |
| F7 | "Guessed V4 Guardian result `GUARDIAN_CONTINUAL_PROXY_CANDIDATE`" | Exists only as (a) a transcribed ARK-019-V4 external audit and (b) an unverified mission-brief reference; no ARK-020 V4 result exists anywhere |
| F8 | Deleted-branch folklore (~15 scratch/temp branches) | Their only recoverable content: the SENORA P35 program (unreachable `30a8fa7`) and the CYR-GPU-006 smoke results (stash-only) — both one `git gc` from loss |
