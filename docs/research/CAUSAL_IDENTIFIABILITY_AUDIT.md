# CAUSAL IDENTIFIABILITY AUDIT

**Phase 2 · 2026-09-13.** For every Phase-1 experiment labeled DEMONSTRATED or SUPPORTED (37 entries): what variable *actually changed*, and what the result therefore identifies.

Classifications: `CAUSALLY_ISOLATED` (one variable moved against matched controls) · `PARTIALLY_ISOLATED` (primary variable isolated but a mechanism-defining covariate moved with it, or single-seed) · `COMPOSITE_TREATMENT` (multiple factors moved together — cannot attribute to any one) · `CORRELATIONAL` (no manipulated variable; observational or evaluator-side) · `NON_SCIENTIFIC` (engineering/instrument evidence).

## 1. Cymek

| Experiment | What actually changed | Class | Notes / flags |
|---|---|---|---|
| CYR-GPU-011 | representation **bundle** (vocabulary + segmentation + number atomization + tied-output burden) at fixed geometry/objective/data | **COMPOSITE_TREATMENT** | condition-level divergence is real but no single cause attributable; its own RESULT says so; COMMUTED probe additionally confounded (post-run downgrade) |
| CYR-GPU-012-R1 | declared tied class-space size ONLY (active IDs, init, data, optimizer fixed) | **CAUSALLY_ISOLATED** (effect) / PARTIALLY_ISOLATED (mechanism) | single seed by documented calibration; tied matrix means embedding+output change jointly → mechanism not isolated |
| CYR-GPU-013-R1B | declared class-space size, 2 fresh seeds, 6 levels | **CAUSALLY_ISOLATED** (effect) | amplitude seed-sensitive; early-formation endpoint |
| CYMEK-closure-cycle | none (invariant verification) | **NON_SCIENTIFIC** | engineering canaries; correctly never claimed as science |
| CYMEK-e1-tokenizer-tournament | tokenizer size on local corpus | **PARTIALLY_ISOLATED** → CORRELATIONAL for capability | compression/params outcome only; corpus non-representative; no learning outcome measured |

## 2. Arkenstone

| Experiment | What actually changed | Class | Notes / flags |
|---|---|---|---|
| ARK-002/002B | seeds (replication) | CAUSALLY_ISOLATED (of the phenomenon's existence) | task manifest frozen |
| ARK-003 | curriculum structure / teacher signal (vs flat) | **CAUSALLY_ISOLATED** (per contrast) | negative |
| ARK-004A (+R) | none (probe battery + reanalysis) | **CORRELATIONAL** | marker not precursor; correctly downgraded |
| ARK-005 | EMA / WD removal / LR decay arms | **CAUSALLY_ISOLATED** (per arm) | negative |
| ARK-006 | LR dose | **CAUSALLY_ISOLATED** | provenance-limited (flagged) |
| ARK-007R | LR (HIGH vs LOW) after shared parent | **CAUSALLY_ISOLATED** | 3 parents × 4 orders; model benchmark |
| ARK-009 | acquisition regime + composite diagnostic | **COMPOSITE_TREATMENT** (diagnostic) | query AND order moved together in the swap probe — flagged in Phase 1 |
| ARK-010 | continuation policy after matched collapse | **CAUSALLY_ISOLATED** | 9 prospective states |
| ARK-011 | post-recovery switch policy | **CAUSALLY_ISOLATED** | 6 sealed forks |
| ARK-012 | screen threshold (observational event screen) | **CORRELATIONAL** | TIME_NOT_STATE_SCREEN; 2 sources |
| ARK-013 | LR policy × no-replay T3 stream | **CAUSALLY_ISOLATED** (for the no-replay regime) | T3 formation failure voids the primary question — result is the boundary, not the frontier |
| ARK-014 | acquisition presentation distribution (canonical vs order-augmented) | **CAUSALLY_ISOLATED** | one family; retention screen zero-event |
| ARK-014-codex-rerun | same, independent execution | **CAUSALLY_ISOLATED** (replication) | same lineage codebase — not a fully independent implementation |
| ARK-015 | continuation support × plasticity (3 single-factor contrasts) | **CAUSALLY_ISOLATED** (per contrast: NARROW_LOW vs NARROW_HIGH isolates LR; AUGMENTED_HIGH vs NARROW_HIGH isolates support) | displacement covariate explicitly used as evidence |
| ARK-017-V2 | update magnitude (CAP) / support (replay) / both — treatment-exact | **CAUSALLY_ISOLATED** (levers) / PARTIALLY_ISOLATED (mechanism) | secondary dose screens single-order |
| ARK-018-V4 | corpus identity at token-matched dose (Birth vs science) | **CAUSALLY_ISOLATED** (corpus-identity question) | token-matching controls dose; content+style change together inside "Birth" — internalization vs assimilation boundary is the residual confound, handled by preregistered threshold |
| ARK-019-V3.1 | controller arms × matched sets | **COMPOSITE_TREATMENT** for controller-vs-static attribution (escalation policy, dose, and observation interval co-varied); **CAUSALLY_ISOLATED** for PLASTIC vs protected contrasts | SKILL_B never formed → primary question voided by design gate |
| ARK-019-V4 | controller arms, dose-qualified parents | **PARTIALLY_ISOLATED** | TRANSCRIBED external audit; bundle absent — provenance caps this at SUPPORTED |

## 3. Triquetra

| Experiment | What actually changed | Class | Notes / flags |
|---|---|---|---|
| TQ-entity-value-factorial | inserted content class (evaluator-side) | **CAUSALLY_ISOLATED** (intervention) / CORRELATIONAL (for any mechanism claim) | answer-bearing salience confound is the point of the reattribution |
| TQ-query-value-matrix | none (measurement ladder) | **CORRELATIONAL** | preregistered, replicated absence |
| TQ-structural-OOD-E5 | structural shift type | **CAUSALLY_ISOLATED** (negative) | floor substrate |
| TQ-readiness-gates | none (gate calibration) | **NON_SCIENTIFIC** (instrument) | v2 standing negative control |

## 4. Citadel

| Experiment | What actually changed | Class | Notes / flags |
|---|---|---|---|
| CIT-T1-series (T0/T1/T1B/T1C) | objective masking / corpus size / pool narrowness / scale (2.3×) — **plus** budget varies across arms | **COMPOSITE_TREATMENT** | capacity accounting obscured by embedding-dominated params; EOS contract unsupervised; nulls still stand (nothing succeeded to misattribute) |
| CIT-T1D | curriculum / teacher / scale / masking / self-knowledge arms — budget confound B vs D/E | **COMPOSITE_TREATMENT** | POSTMORTEM 5 explicitly blocks scale/masking conclusions; evaluation surface later found SHORTCUT_COMPROMISED + LEAKAGE_COMPROMISED (CITADEL-EVAL-001) — cannot invalidate the nulls (all arms failed) but blocks future positives |
| CIT-scoring-policy-tournament | scoring policy (5 families) | **CAUSALLY_ISOLATED** | fixture v2; decoy axes; TOST+Holm |
| CIT-e0-generator-repairs | generator version | **CAUSALLY_ISOLATED** (instrument) | engineering-tier receipt |
| CIT-500M-production-path-audit | none (audit) | **NON_SCIENTIFIC** | stale pin noted |
| CITADEL-DATA-001 (NEW) | none (mechanical audit of real corpus) | **NON_SCIENTIFIC** (instrument verdict with scientific consequence) | verified: shortcut 1.000/tier, 530 verbatim cross-split docs, 13.5% duplication, 0.004× supply |
| CITADEL-EVAL-001 (NEW) | none (attack battery on evaluation surfaces) | **NON_SCIENTIFIC** (instrument) | t1d surface SHORTCUT+LEAKAGE compromised; e0 + triquetra surfaces PROVISIONALLY_VALID; PRE50M smoke VALIDATED |

## 5. ESOES / BRAMASTRA

| Experiment | What actually changed | Class | Notes / flags |
|---|---|---|---|
| ESO-PGE-continuation | continuation training (vs parent) | **CAUSALLY_ISOLATED** (lineage-level) | probe-floor alternative recorded |
| ESO-SFT6-replication | targeted SFT | **CAUSALLY_ISOLATED** | assisted-scoring instrument dependence flagged |
| ESO-e2-mechanism-canaries | init scaling / QK norm / precision layout (stress tests) | **CAUSALLY_ISOLATED** (mechanism priors) | zero learning evidence by design |
| BRM-terminal-EOS | EOS supervision (matched everything else) | **CAUSALLY_ISOLATED** | tiny lab |
| BRM-transfer-baseline | none (measurement) | **CORRELATIONAL** | rendering shift multi-factor |
| BRM-binding-diversity | training-pool size (exploratory, post-hoc arm) | **PARTIALLY_ISOLATED** → result is the query-blind *null* | post-hoc arm selection flagged; recomputed analysis binds receipts |

## 6. Headline identifiability warnings for the architecture agent

1. **"Compact representation is better" is NOT established** — CYR-011 was a composite treatment; the isolated variable is *declared class-space size* (R1/R1B), and even that is mechanism-unresolved until R1C.
2. **"Replay works" needs a qualifier** — treatment-exact sparse replay protects (isolated); "this-regime" replay failed (provenance-weak); dose universality untested.
3. **T1D-family nulls are robust but their surface is dead** — future arithmetic claims require a regenerated, attack-screened corpus.
4. **The Guardian question currently has NO in-repo decisive evidence** — V3.1 is voided by formation failure; V4 is a transcription.
5. **No experiment has ever isolated "architecture" from "parameter count"** — every architecture-family claim in the repo is a default, not a finding.
