# CAUSAL KNOWLEDGE GRAPH

**Synthesis date:** 2026-09-13. Companion to [`EXPERIMENT_EVIDENCE_LEDGER.json`](EXPERIMENT_EVIDENCE_LEDGER.json) (experiment IDs cite its entries).

Edge classes:
- **CAUSALLY ISOLATED** — treatment randomized/matched against controls within the cited experiment; alternative explanations bounded.
- **CORRELATIONAL** — observed association without isolation.
- **MECHANISTIC HYPOTHESIS** — proposed mechanism not yet causally tested (a designed test exists or is named).

Confidence: HIGH (replicated + matched design) / MEDIUM (single-run but well-controlled) / LOW (single confounded or DEV-tier run).

## 1. Capability formation (the current scientific frontier)

```
REPRESENTATION_CLASS_SPACE (declared tied embedding/output class count)
   --[non-monotonically modulates]--> HELDOUT_CAPABILITY_FORMATION
   class: CAUSALLY ISOLATED (active token IDs, segmentation, data, geometry,
          optimizer, seeds, non-embedding init all fixed; only declared class count varies)
   evidence: CYR-GPU-012-R1 (V19 12.94% / V4096 100% / V24576 0%), CYR-GPU-013-R1B
             (intermediate 4096–16384 region strong in both fresh seeds; extremes ≈ 0)
   confidence: MEDIUM-HIGH (R1; seed-sensitive amplitude, stable ordering)
   counterevidence: ARK-001 refuted whole-vocab as first-order at micro lift-off —
             resolved as a DIFFERENT manipulation (see ledger §10)
   open: which correlated factor (vocabulary size, segmentation, tied-output burden,
         inactive-row participation) carries the effect
```

```
TRAINING_TIME_INACTIVE_SOFTMAX_COMPETITION
   --[hypothesized cause of]--> CLASS_SPACE_EFFECT
   class: MECHANISTIC HYPOTHESIS
   designed test: CYR-GPU-014-R1C (fixed 24,576 matrix; MASK_19/4096/8192/16384,
             OFFSET_EQ4096 vs FULL_24576; dual structural/functional endpoints)
   status: NOT_EXECUTED (engineering-ready; launcher currently binds a commit absent
           from origin — see EVIDENCE_GAPS)
```

```
REPRESENTATION_BUNDLE (compact 19-symbol vs production 24,576 BPE)
   --[gates whether semantic dose produces held-out generalization]--> HELDOUT_CAPABILITY_FORMATION
   class: CAUSALLY ISOLATED at condition level (geometry/objective/data fixed; bundle varied)
   evidence: CYR-GPU-011 (compact 56.47% @ 44.89% exposure vs production 0% @ 100%)
   confidence: HIGH for the divergence, LOW for any single causal factor (bundle is
               correlated changes; superseded as a question by CYR-012/013/014)
```

```
SEMANTIC_DOSE
   --[insufficient under production representation]--> HELDOUT_CAPABILITY_FORMATION
   class: CAUSALLY ISOLATED (dose varied within fixed representation)
   evidence: CYR-GPU-011 production arm (1,152,000 rows = 100% reference box → 0%);
             CYR-GPU-009/010 establish <25% dose is also uninformative (negative controls)
   confidence: HIGH
```

```
MEMORIZATION (train saturation)
   --[does NOT predict timing of]--> GENERALIZATION (M99 vs G90, rho 0.00)
   class: CAUSALLY ISOLATED (prospective probe battery, 4 seeds)
   evidence: ARK-004A, ARK-004A-R
   confidence: HIGH (within micro T2)
```

```
DELAYED_GENERALIZATION_TRANSITION
   --[replicates]--> (memorize-first, generalize-later, large seed variance)
   class: CORRELATIONAL (phenomenon replication)
   evidence: ARK-002B (seeds 29/47), reused as the exposure benchmark by CYR-011/012/013
   confidence: HIGH
```

## 2. Retention / plasticity (the best-mapped region)

```
LOW_APPLIED_UPDATE_MAGNITUDE (LR 1e-5 or CAP1X)
   --[protects]--> ACQUIRED_INVARIANT (matched continuation)
   class: CAUSALLY ISOLATED
   evidence: ARK-007R (HIGH 9/12 vs LOW 0/12, RD −0.75), ARK-011 (recurrence 3/6 vs 0/6),
             ARK-015 (NARROW_LOW 0/8), ARK-017-V2 (LOW 0/6)
   confidence: HIGH (R1, four experiments)
   boundary: NOT a recovery tool (ARK-010: LOW 2/9) and NOT a cross-task
             interference solution (ARK-013: all arms lost T2)
```

```
HIGH_PLASTICITY + NARROWED_SUPPORT
   --[erodes]--> BROADER_INVARIANCE (canonical exact stays 1.0)
   class: CAUSALLY ISOLATED
   evidence: ARK-015 (NARROW_HIGH 8/8 failures; canonical 1.0; robust mean ~0.468)
   confidence: HIGH
   lesson: canonical accuracy cannot detect retained capability
```

```
CONTINUED_CAPABILITY_SUPPORT (augmented/continued diverse presentation)
   --[protects]--> ACQUIRED_INVARIANT AND BROADER_INVARIANCE
   class: CAUSALLY ISOLATED
   evidence: ARK-015 (AUGMENTED_HIGH 0/8 despite larger movement), ARK-017-V2
             (exact noncanonical 1/16 replay 0/6)
   confidence: HIGH
```

```
TOTAL_PARAMETER_MOVEMENT
   --[does NOT explain]--> RETENTION/EROSION
   class: CAUSALLY ISOLATED (movement varied independently of outcome)
   evidence: ARK-015 (augmented moved farther, retained), ARK-017-V2 (replay arm's
             cumulative movement exceeded failing HIGH while protecting)
   confidence: HIGH
```

```
RECOVERY (capability absent)
   <--[requires]-- HIGH_PLASTICITY
   class: CAUSALLY ISOLATED
   evidence: ARK-010 (HIGH 8/9 vs immediate LOW 2/9)
   confidence: HIGH (Micro T2)
```

```
STATE_THRESHOLD_FOR_LR_SWITCH (exact numeric)
   --[not identifiable]--> (time/state aliasing; TIME_NOT_STATE_SCREEN)
   class: CORRELATIONAL (negative result)
   evidence: ARK-012
   confidence: HIGH (within its 2-source screen)
```

```
PLASTICITY_VS_SUPPORT_INTERACTION
   --[is the working model of]--> RETENTION
   class: MECHANISTIC HYPOTHESIS (consistent with all edges above; the two levers
          were shown independently sufficient, not dissected into one mechanism)
   evidence: ARK-017-V2 verdict BOTH_LEVERS_SUFFICIENT
```

## 3. Continual learning / controllers

```
NEW_SKILL_FORMATION
   --[gates]--> ANY_CONTINUAL_LEARNING_VERDICT
   class: CAUSALLY ISOLATED (formation-first gate design)
   evidence: ARK-013 (T3 never acquired → primary inconclusive), ARK-019-V3.1
             (SKILL_B never formed in ANY arm incl. unprotected reference →
             CONTROLLER_NOT_SUPPORTED reinterpreted as NEW-SKILL-FORMATION BOTTLENECK)
   confidence: HIGH
   consequence: ARK-019-V4/ARK-020 add prospective dose selection and
                formation-first verdicts; never call a controller failure
                before the reference can acquire
```

```
STATIC_SPARSE_REPLAY (1/64)
   --[strongly protects]--> OLD_SKILL (under continued real-text learning)
   class: CAUSALLY ISOLATED (arm contrast, 4 matched sets)
   evidence: ARK-019-V3.1 (STATIC 1/64 ≈ 0.882 vs PLASTIC_HIGH ≈ 0.005)
   confidence: HIGH at proxy scale
   note: NOT evidence that 1/64 is a universal dose (explicitly disclaimed)
```

```
REACTIVE_GUARDIAN (escalating replay)
   --[recovers after collapse; did NOT prevent]--> OLD_SKILL health
   class: CAUSALLY ISOLATED
   evidence: ARK-019-V3.1 (≈0.963; every Guardian failed formally by the first
             100-update observation, then replay reconstructed it; CAP16X never triggered)
   confidence: HIGH for recovery; PREVENTION UNTESTED (interval too coarse)
```

```
DYNAMIC_GUARDIAN_WITH_DOSE_QUALIFICATION (ARK-019-V4)
   --[claimed to preserve old AND acquire new]--> at proxy scale
   class: CORRELATIONAL / TRANSCRIBED EVIDENCE
   evidence: ARK-019-V4 FINAL_RESULT_AUDIT_V4.md (GUARDIAN_CONTINUAL_PROXY_CANDIDATE;
             PLASTIC_HIGH 0/4 old; Guardian arms 4/4 old + 4/4 new; lower replay cost
             than permanent rehearsal; CAP16X benefit unclear)
   confidence: MEDIUM-LOW until the raw bundle is committed and byte re-audited
   conflict: V3.1 (CONTROLLER_NOT_SUPPORTED) vs V4 (candidate) is explained by V4's
             design repairs (dose selection, science-preserving parents) — the repair
             logic is coherent but currently unverifiable in-repo
```

```
QUERY_CONDITIONED_ADDRESSING
   --[≈ absent on weak V4 substrate]--> (no latent query control measurable)
   class: CAUSALLY ISOLATED (preregistered matrices, replicated seeds)
   evidence: TQ-query-value-matrix (rank-1 0.2500 = chance; QCS CI ∋ 0; position 19–35×),
             BRM-binding-diversity (48.4% ≈ 50% query-blind; 62/64 same answer)
   confidence: HIGH (two independent programs)
   boundary: floor-limited substrate — absence of measurable signal is not
             proof of representational absence
```

```
VALUE_RECENCY (answer-bearing insertion)
   --[elicits repairs]--> failures (46.7%/46.2%)
   class: CORRELATIONAL (evaluator-side intervention; salience vs addressing confound)
   evidence: TQ-entity-value-factorial (replicated)
```

```
EOS_SUPERVISION
   --[required for]--> COMPLETE_ANSWERS
   class: CAUSALLY ISOLATED (matched arms, 2 seeds) + cross-program convergence
   evidence: BRM-terminal-EOS (0/32 → 32/32 ×2), CIT-T1D postmortem (15,000/15,000
             MAX_TOKENS), adopted as Cymek contract
   confidence: HIGH (R3)
```

```
LOSS_IMPROVEMENT
   --[does NOT produce]--> TESTED_COGNITION
   class: CAUSALLY ISOLATED across substrates (R3 family)
   evidence: ESO-PGE (loss −0.2174, probes 0), CIT-T1-series/T1D, ARK-001 era
   confidence: HIGH for tested probe batteries; probe-floor alternative recorded
```

```
CURRICULUM / ALIGNED_TEACHER
   --[does NOT accelerate]--> OOD_EMERGENCE
   class: CAUSALLY ISOLATED (two programs)
   evidence: ARK-003, CIT-T1D arms B/C
   confidence: HIGH at tested budgets
```

```
SCALE (2–2.3× params) and CORPUS SIZE (rich 6.5M-row)
   --[insufficient for lift-off]--> at 1.6–4M-token budgets
   class: CAUSALLY ISOLATED within T1-series arm contrasts
   evidence: CIT-T1-series (0/1,000 everywhere; copy heuristic 2.7%)
   confidence: MEDIUM-HIGH (embedding-dominated capacity accounting noted)
```

## 4. Node status summary

| Node | Best-supported current statement |
|---|---|
| REPRESENTATION / CLASS_SPACE | Non-monotonic formation effect, isolated from everything except the softmax-partition question (R1C pending) |
| OUTPUT_COMPETITION | Mechanistic hypothesis; discriminating test frozen but unexecuted |
| OPTIMIZATION / LR | Phase-dependent: HIGH for acquisition/recovery, LOW/CAP1X for retention; no universal law, no exact threshold |
| REPLAY | Treatment-exact sparse support protects (0/6); dose universality unknown; "this-regime" replay variant failed |
| INTERFERENCE | LR alone fails; support needed; formation of the new skill is the binding gate |
| BINDING / ADDRESSING | Query control not demonstrated on any substrate so far; three claim attempts died (floor substrate, confounded diagnostics, query-blind baselines) |
| COMPOSITION | Primitives ≠ composition (teacher 0.515 vs composed ≈ 0) |
| SELF_MODEL | All positive claims invalidated or unqualified; readiness-v2 gate is the standing negative control |
| EVALUATION | Scorer certification still blocks assisted comparisons; candidate-free primary is the settled policy |
| TERMINATION | EOS contract settled cross-program |
| SCALE | 500M blocked (corpus, entry point; audit age noted); no scientific authorization from any result |
| AGI | 0% — no current result supports a general claim |
