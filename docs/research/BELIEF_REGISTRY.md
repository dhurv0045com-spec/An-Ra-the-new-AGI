# BELIEF REGISTRY

**Phase 2 · 2026-09-13 · evidence snapshot:** Phase-1 branch `research/evidence-consolidation-2026-09-13` @ `a350c2b` (pushed, validator green). Deltas since Phase 1, audited before this analysis: citadel `CITADEL-DATA-001` (executed corpus audit), new branch `eval-integrity-001` with `CITADEL-EVAL-001` (evaluation integrity audit), BRAMASTRA `B00–B12` integrated build (engineering, no science), codex refactor only.
**Machine source:** [`BELIEF_REGISTRY.json`](BELIEF_REGISTRY.json) · statuses: `STRONGLY_SUPPORTED / SUPPORTED / WEAKLY_SUPPORTED / OPEN / CONTESTED / CONTRADICTED / NOT_TESTED`.
**Rule:** implementation convention is not empirical belief. Estimates labeled here are decision-model judgments, not measured probabilities.

## Registry summary (32 beliefs)

| ID | Class | Belief (abridged) | Status | Confidence | Cheapest falsifier |
|---|---|---|---|---|---|
| B01 | ARCHITECTURE | Dense decoder-only Transformer is the right Core family | SUPPORTED (as baseline, not proven superiority) | MED-HIGH | none until an isolated bottleneck implicates architecture |
| B02 | GENERALIZATION | Delayed memorize→generalize transition replicates; don't stop at train saturation | STRONGLY_SUPPORTED | HIGH (micro) | CS-TRANSFER-001 timing records |
| B03 | GENERALIZATION | Memorization timing does not predict generalization timing | STRONGLY_SUPPORTED | HIGH | dense probes in next campaign |
| B04 | REPRESENTATION | Declared class-space size causally changes formation, non-monotonically | SUPPORTED | MED-HIGH effect / LOW mechanism | **R1C (frozen)** |
| B05 | OUTPUT_COMPETITION | Inactive-softmax competition is THE mechanism | NOT_TESTED | speculative by design | **R1C** |
| B06 | REPRESENTATION | Production 24,576 BPE cannot form held-out arithmetic at full exposure (4L/128w) | SUPPORTED | MED (single seed) | one V24576 rerun arm |
| B07 | TOKENIZATION | The class-space effect transfers to natural language / larger models | NOT_TESTED | NONE | **CS-TRANSFER-001** |
| B08 | LEARNING_RATE | LOW update magnitude protects an acquired invariant | STRONGLY_SUPPORTED | HIGH (micro) | larger-scale retention fork |
| B09 | PLASTICITY | HIGH plasticity required for recovery; immediate LOW harmful | STRONGLY_SUPPORTED | HIGH | none pending |
| B10 | INVARIANCE | Canonical accuracy can stay 1.0 while broader invariance collapses | STRONGLY_SUPPORTED | HIGH | embed robustness probes anywhere |
| B11 | RETENTION | Retention = interaction(update magnitude × capability support), not displacement | SUPPORTED | MED-HIGH | gradient-direction diagnostics in next fork |
| B12 | RETENTION | Parameter displacement causes capability loss | CONTRADICTED | HIGH (falsified) | n/a |
| B13 | LEARNING_RATE | A unique numeric LR-switch threshold exists | CONTRADICTED | HIGH (scoped) | none planned |
| B14 | REPLAY | A universal replay dose (e.g. 1/64) exists | NOT_TESTED | LOW | two-dose contrast in next campaign |
| B15 | INTERFERENCE | LR alone manages cross-task interference | CONTRADICTED | HIGH | n/a |
| B16 | CONTINUAL_LEARNING | New-skill formation gates all controller verdicts | SUPPORTED (methodological law) | HIGH | n/a |
| B17 | GUARDIAN_CONTROL | A dynamic Guardian preserves old skill at lower replay cost than permanent rehearsal | **CONTESTED** (V3.1 no / V4 transcribed yes) | LOW-MED | recover+audit V4 bundle (zero GPU if it exists) |
| B18 | BINDING | Weak V4 checkpoint contains usable latent query-conditioned signal | CONTRADICTED (substrate-scoped) | HIGH | re-run QV matrix on qualified checkpoint |
| B19 | BINDING | Value-recency repair demonstrates addressing | CONTRADICTED (salience reattribution) | HIGH | n/a |
| B20 | TERMINATION | Answer+EOS supervision required; termination ≠ content failure | STRONGLY_SUPPORTED | HIGH (R3) | n/a |
| B21 | OBJECTIVE | Loss improvement implies cognition improvement | CONTRADICTED | HIGH | n/a |
| B22 | COGNITION_DATA | Curriculum / aligned teacher accelerate OOD emergence | CONTRADICTED | HIGH (tested budgets) | n/a |
| B23 | COMPOSITION | Primitives compose automatically into full operations | CONTRADICTED | HIGH | n/a |
| B24 | CAUSAL_DIAGNOSIS | A learned self-model is trainable on the current V4 basis | CONTRADICTED | HIGH (scoped) | readiness calibration on next real-text parent |
| B25 | EVALUATION_VALIDITY | Calibrated candidate scorers neutralize length bias | CONTRADICTED | HIGH | C0 extended screen |
| B26 | EVALUATION_VALIDITY | Aggregate accuracy suffices for capability/robustness claims | CONTRADICTED | HIGH | n/a |
| B27 | EVALUATION_VALIDITY | The tiered arithmetic surface (T1D-era) is a valid instrument | **CONTRADICTED (NEW)** — shortcut 1.000/tier, 530 verbatim leaks, 13.5% duplication, 0.004× supply | HIGH (mechanically verified) | re-run `tools/audit_tiered_corpus.py` on regenerated corpus |
| B28 | QK_NORMALIZATION | QK norm + residual 1/√2L scaling are required scale-control | SUPPORTED (mechanism prior only) | HIGH prior / zero cognition evidence | no-QK ablation arm |
| B29 | OPTIMIZER | BF16 compute + FP32 master/moments safe; native BF16 states unsafe | STRONGLY_SUPPORTED (local) | HIGH locally | none planned |
| B30 | DATA_MIXTURE | 65/20/15 mixture + cognition fractions are good defaults | NOT_TESTED | NONE (planning prior) | E3 when unblocked |
| B31 | SCHEDULE | 5B WSD schedule constants validated | NOT_TESTED | NONE | execute schedule once at PRE500M smoke |
| B32 | SCALING_TO_500M | Scaling resolves the formation bottleneck | NOT_TESTED | NONE | the three Phase-2 experiments collectively |

## Observation → interpretation → mechanism → design consequence (high-impact beliefs)

### B04/B05/B06 — representation & class space
- **OBSERVATION:** production rep: M99 + 0% held-out + 0/48 sealed at 100% exposure; compact: 56.47% at 44.89%; declared class space alone: V4096 100% vs V19 12.94% vs V24576 0%; curve replicated directionally (R1B).
- **INTERPRETATION:** output-space geometry, not semantic dose, is the binding variable at development scale.
- **MECHANISM HYPOTHESES (competing, unresolved):** H1 inactive-softmax partition competition; H2 tied-output gradient burden; H3 embedding-row geometry/initialization; H4 seed lottery modulating amplitude.
- **DESIGN CONSEQUENCE:** tokenizer/vocabulary/tied-output decisions are BLOCKED on R1C + CS-TRANSFER-001; no production tokenizer change now.

### B08–B11 — retention
- **OBSERVATION:** 9/12 vs 0/12; 3/6 vs 0/6; 8/8 vs 0/8 vs 0/8; ARK-017: HIGH 4/6 fail vs LOW/CAP1X/replay/joint/augmented 0/6; replay arm moved farther than failing HIGH.
- **INTERPRETATION:** protection comes from reduced effective update magnitude and/or continued capability-relevant support.
- **MECHANISM HYPOTHESES:** H1 replay repairs destructive gradient directions; H2 replay continuously reacquires a lost capability; H3 support regularizes shared representations; H4 cap acts as implicit LR decay (partially conflated with H1 by design).
- **DESIGN CONSEQUENCE:** replay and cap remain justified EXTERNAL protection mechanisms; internalization and dose universality are NOT justified; measure old+new skill separately always.

### B17 — Guardian
- **OBSERVATION:** V3.1: PLASTIC_HIGH destroyed A 4/4 (≈0.005), STATIC 1/64 ≈0.882, Guardian ≈0.963 (recovery after formal failure, not prevention), SKILL_B never formed; V4 (transcribed): Guardian arms old 4/4 + new 4/4 at lower replay cost.
- **INTERPRETATION:** escalation-based recovery works at proxy scale; prevention unproven; V4's positive claim is externally audited only.
- **MECHANISM HYPOTHESES:** reactive replay-dose effect; anticipatory prediction; task-ID leakage into controller features (must be excluded); evaluation-timing artifacts.
- **DESIGN CONSEQUENCE:** do not internalize any controller; ARK-020 interpretation is blocked until the V4 bundle is audited or rerun.

### B27 — evaluation surface (NEW from snapshot delta)
- **OBSERVATION:** mechanically verified: latest_position shortcut 1.000 on all 5 tiers; 173 dev + 357 test docs verbatim in train; 13.5% train duplication; 6.42M rows ≈ 2.1M BPE tokens = 0.004× of 500M demand.
- **INTERPRETATION:** any future *positive* lift-off claim on this surface is uninterpretable; the historical T1-series/T1D nulls still stand (leakage/shortcut can only have inflated, and everything failed anyway; EOS confound additionally recorded).
- **DESIGN CONSEQUENCE:** no new arithmetic campaign may use this surface; regeneration requires passing `tools/audit_tiered_corpus.py` + eval-attack screens; data gate blockers now include supply shortfall.
