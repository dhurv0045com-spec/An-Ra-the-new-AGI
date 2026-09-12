# EXPERIMENT EVIDENCE LEDGER

**Synthesis date:** 2026-09-13
**Authoritative machine copy:** [`EXPERIMENT_EVIDENCE_LEDGER.json`](EXPERIMENT_EVIDENCE_LEDGER.json) (schema `anra.evidence-ledger/v1`)
**Audit basis:** all 14 live origin branches at the heads recorded in the JSON `branch_heads_at_audit` block, plus git-forensics over unreachable history.
**Purpose:** one canonical row per recoverable experiment, reconstructed from primary evidence, so that future architecture work starts from truth rather than accumulated prose.

Evidence rules used (binding):
- Raw artifacts (Tier 0) beat RESULT.md (Tier 1) beat code (Tier 2) beat plans (Tier 3) beat prose/README/older syntheses (Tier 4).
- Code that exists without execution is `IMPLEMENTED_NOT_EXECUTED`, never a result.
- Engineering failures are never converted into scientific results.
- Replication levels: `R0` one run/seed · `R1` multi-seed same implementation · `R2` independent execution/branch · `R3` cross-task/substrate · `R4` prospectively isolated mechanism.

---

## 1. Status counts (78 ledger entries; validated by `tools/validate_evidence_ledger.py`)

| Status | Count |
|---|---:|
| DEMONSTRATED | 22 |
| SUPPORTED | 15 |
| INCONCLUSIVE | 13 |
| SUPERSEDED | 8 |
| CONTRADICTED | 9 |
| NOT_TESTED | 5 |
| IMPLEMENTED_NOT_EXECUTED | 3 |
| INVALIDATED | 2 |
| SPECULATIVE | 1 |

(`IN_PROGRESS`: 0. Several entries carry dual character — e.g. `CYR-GPU-014-R1C` is engineering-ready and scientifically `NOT_EXECUTED`; the JSON records both. Status counts include engineering-canary entries classified DEMONSTRATED-as-engineering, e.g. the closure cycle, Discovery bundles, and the TPU runtime line.)

## 2. Live branch roles (derived from evidence, 2026-09-13)

| Branch | Head | Role (evidence-derived) |
|---|---|---|
| `cymek-500m-readiness` | `f2c27a6` | Cymek V5 production core + controlled GPU representation/mechanism campaigns (CYR-GPU-001…014-R1C) |
| `Arkenstone` | `4ae9e3b` | Discovery program: ARK-001…022 + Guardian/continual-learning line; newest executed science (ARK-017 V2, ARK-018 V4, ARK-019 V3.1) |
| `arkenstone-ark020-v4` | `65d1ef3` | Dedicated ARK-020 V4 execution branch + durability amendments A1.1–A1.3; no results |
| `codex/arkenstone-improvements` | `1511321` | Runtime integrity + independent ARK-014 replication (RTX 4050, raw receipts) |
| `arkenstone-astra` | `ccb84fb` | BRAMASTRA + one research-environments commit (switch/qualification env code); no results |
| `BRAMASTRA` | `02b94d3` | Discovery/binding lab: EOS contract, transfer baselines, discovery controller, D02; own cross-branch audit |
| `triquetra` | `f23f0af` | Cognition laboratory (V4-substrate diagnostics); WAITING_FOR_STRONGER_CHECKPOINT; no new science since 2026-09-05 |
| `citadel` | `1d27f9b` | Independent auditor: T1-series, scoring-policy tournament, 500M production-path audit, negative-results registry |
| `esoes` | `85f44b7` | TPU-era V5 blueprint + founding negative (PGE) + mechanism canaries (D-024…D-028) |
| `core-exp` | `51124de` | Historical V4-era self-model/policy promotion line (MC-v9/10/11) + milestone 0001 |
| `core-frozen-v4` | `f72f193` | Frozen inference-only V4 core (32,768-token tokenizer) |
| `main` | `b620f1c` | Frozen V4 research system (2026-08-15, PR #41) |
| `iterate500` / `iterate900` | `b438420` / `6fbd2c0` | Historical TPU/SFT engineering lineages |
| *(deleted)* `senora` | *(unreachable `30a8fa7`)* | Entire P35-CMS-1 + CAD program survives only in unreachable commits |

**History note:** the repository has two disconnected shards. The new shard (cymek-500m-readiness, Arkenstone, ark020-v4, codex) is rooted at a parentless squash commit `28bf57a` (2026-09-05); the only bridge to pre-2026-09-05 history is the local branch `cymek` (`4abeaeb`, 463 commits). `6653b4ce` (the R1C audited executable chain head) is an ancestor of NO live branch and is **not served by GitHub**.

## 3. CYMEK — Cymek V5 GPU campaign series

| ID | Question (abridged) | Status | Replication | Key numbers |
|---|---|---|---|---|
| CYR-GPU-001 | First tournament design | SUPERSEDED | – | never executed |
| CYR-GPU-002 | Rebuilt tournament stack | SUPERSEDED | – | test receipt only |
| CYR-GPU-003 | Successor | SUPERSEDED | – | 13 defects, 3 blockers, pre-execution |
| CYR-GPU-004 | Corrected retention campaign | SUPERSEDED | – | fork-contract defect (no parent restore) |
| CYR-GPU-005 | Shared-parent fork retention | SUPERSEDED | – | frozen prereg; never launched; frozen executable only in unreachable `fcd9178` |
| CYR-GPU-006-smoke | Engineering smoke of campaign driver | INCONCLUSIVE (engineering) | R0 | receipt + 36 checkpoints exist ONLY in stash untracked parent `87ea5d6`; scientific campaign stopped by hardware gate after calibration (~180 tok/s RS / 749 tok/s TINY) |
| CYR-GPU-007 | Successor | SUPERSEDED | – | decision-wrapper self-recursion defect |
| CYR-GPU-008 | V8 all-or-nothing forks | SUPERSEDED | – | rejected at calibration by 170-min wall; not a scientific negative |
| CYR-GPU-009 | TINY acquisition at 2M tokens | INCONCLUSIVE | R1 | train probes 0.94–1.0, DEV_CONTROLLER ≈ 0; **<22% of ARK-002B exposure** → underdosed; verdict `INCONCLUSIVE_NO_COMPLETE_MATCHED_PAIR`; bundle `dc15f14d…` |
| CYR-GPU-010 | 18k-update retry | SUPERSEDED | – | batch-16 would give 25% of reference exposure |
| **CYR-GPU-011** | Compact (19-symbol) vs production (24,576 BPE) representation on real V5 | **DEMONSTRATED** | R0 | compact: 56.47% held-out STANDARD @ **44.89%** exposure (517,184 rows, G50@2200); production: **0%** STANDARD, 0/48 SEALED @ **100%** exposure (1,152,000 rows, 18,000 updates); COMMUTED=100% post-run-audited as confounded; bundle `fbec390f…` |
| **CYR-GPU-012-R1** | Declared tied class-space size alone (V19/V4096/V24576, active IDs fixed) | **DEMONSTRATED** | R0 | V19 12.94% / **V4096 100%** / V24576 **0%** at 512k rows; non-monotonic; displacement ≠ capability; verdict `MIXED_OR_INTERMEDIATE_REPRESENTATION_EFFECT`; bundle `a22b5383…` |
| **CYR-GPU-013-R1B** | Replicated response curve, 2 fresh seeds × 6 levels | **SUPPORTED** | R1 | V4096 0.506/0.494; V8192 0.718/0.0; V16384 0.647/0.129; V19/V1024/V24576 ≈ 0; verdict `MIXED_OR_SEED_SENSITIVE_RESPONSE_CURVE`; bundle `7ffebfd4…` |
| **CYR-GPU-014-R1C** | Softmax-competition mechanism at fixed 24,576 matrix (6 arms × 4 seeds) | IMPLEMENTED_NOT_EXECUTED | – | 72,000 updates / 4.608M rows planned; two engineering failures repaired (optimizer API TypeError; CLIP_BREACH 1.0000042915 > 1.0 float32 reduction-order, tolerance 1e-6→1e-4 with e2e regression); **AUDIT FINDING: launcher v4 checks out `6653b4ce` which is unreachable locally AND absent from GitHub — fresh-clone launch will fail** |
| CYMEK-P35A | Matched-compute cognition experiment | NOT_TESTED | – | gated behind external identities |
| CYMEK-closure-cycle | E1–E11 production-path engineering | DEMONSTRATED (engineering) | R1 | exact-resume ≡ uninterrupted; XLA accumulation defect fixed (per-microstep all-reduce scaled early microstep gradients by replica-count powers); V5-A 250,216,960-param CUDA canary; receipt meta-checks STALE-BY-DESIGN after R1C constant consolidation |
| CYMEK-e1-tokenizer-tournament | 16k/24k/32k byte-BPE Pareto | INCONCLUSIVE (planning prior) | R0 | 0.23826/0.23217/0.23022 tokens/byte; non-representative corpus; superseded in relevance by class-space results |

**Cymek bottom line:** the representation/class-space discovery chain (011 → 012 → 013) is the program's strongest current science; the mechanism test (014-R1C) is frozen, twice engineering-repaired, and — as of this audit — **unlaunchable from GitHub because its pinned commit does not exist on origin**.

## 4. ARKENSTONE — discovery program

| ID | Question (abridged) | Status | Replication | Key numbers |
|---|---|---|---|---|
| ARK-001 | Lift-off location; H-FLOOR/H-REPR | CONTRADICTED (both hypotheses refuted) | R0 | vocab dead-space & capacity pathology not first-order at micro; ByteVocab bug caught by impossible-loss signature |
| ARK-002 / 002a | T2 transition saturation | DEMONSTRATED | R0 | single-seed saturation |
| **ARK-002B** | Delayed memorize→generalize replicates | **DEMONSTRATED** | R1 | seeds 29/47; large seed variance; delayed region ~9k–18k updates |
| ARK-003 | Curriculum / aligned teacher accelerate? | CONTRADICTED | R0 | curriculum delayed memorization, zero OOD in box |
| ARK-004A (+R) | Precursors of transition | SUPPORTED (marker, not precursor) | R1 | M99-vs-G90 timing rho 0.00; precursor fails 3/4; post-G90 instability exists |
| ARK-005 | EMA / WD-removal stabilization | CONTRADICTED | R0 | no consolidation winner |
| ARK-006 | LR dose response | SUPPORTED | R0 | threshold candidate < 1e-4; provenance-limited |
| ARK-007 | LOW protection (first) | SUPPORTED | R1 | superseded by 007R |
| **ARK-007R** | LOW protection replicated | **DEMONSTRATED** | R1 | HIGH fails **9/12**, LOW **0/12**, risk difference **−0.75** |
| ARK-009 | Non-arithmetic transfer gate | INCONCLUSIVE | R0 | heldout 1.0, composite robustness failed; diagnostic confounded (query+order together) |
| **ARK-010** | Recovery after collapse | **DEMONSTRATED** | R1 | HIGH recovers **8/9**, immediate LOW **2/9** |
| **ARK-011** | HIGH→LOW switch reduces recurrence | **DEMONSTRATED** | R1 | HIGH recollapse 3/6, SWITCH_LOW 0/6, RD −0.5, LOW sealed RET90 1.0 |
| ARK-012 | Exact switch threshold | CONTRADICTED | R0 | `TIME_NOT_STATE_SCREEN`; thresholds alias to switch times |
| ARK-013 | LR policy vs cross-task interference | INCONCLUSIVE | R0 | T3 never acquired; every arm lost T2 without replay; LOW only slowed interference |
| **ARK-014** | Order-augmented acquisition repairs binding robustness | **DEMONSTRATED** | R2 | canonical ≈0.33 order-only → order-augmented qualified @1800, sealed ≈0.987; retention screen zero-event |
| ARK-014-codex-rerun | Independent re-run (codex branch) | DEMONSTRATED | R2 | RTX 4050, 41.3 min; sealed ORDER_ONLY/QUERY_ORDER **0.993** @step 2000; canonical never qualified in 24,000 steps (plateau 0.38–0.46); raw RESULT/PARTIAL/MANIFEST receipts committed |
| **ARK-015** | Invariance narrowing under distribution shift | **DEMONSTRATED** | R1 | NARROW_HIGH **8/8** failures @ canonical 1.0; NARROW_LOW 0/8; AUGMENTED_HIGH 0/8 despite larger movement; RD −1.0 |
| ARK-016 | Update-cap mechanism credit | INCONCLUSIVE | R0 | only 1/12 qualifying events |
| **ARK-017-V2** | Mechanism credit: update magnitude vs invariant-support replay | **DEMONSTRATED** | R1 | `BOTH_LEVERS_SUFFICIENT`: HIGH 4/6 fail; LOW / CAP1X / replay-1/16 / joint / augmented-HIGH **0/6** each; secondary: CAP4X, CAP16X, replay-1/32, replay-1/64 all 0/3; "retention = small movement" falsified |
| **ARK-018-V4** | Birth-Book internalization vs matched science replay (~20–25M on peS2o) | **DEMONSTRATED** (primary negative) | R1 | internalization +0.000/+0.067 vs required +0.10 (**NOT MET**, 2 seeds); SEALED science NLL +3.68%/+3.91%; binding-acquisition slowdown 1200/>1500 vs 300/300 steps; bundle `cea50622…`, 40/40 hashes |
| **ARK-019-V3.1** | Guardian (real-text proxy) | **DEMONSTRATED** (controller NOT supported) | R1 | verdict **`CONTROLLER_NOT_SUPPORTED`**; SKILL_B never formed in ANY arm → new-skill-formation bottleneck; PLASTIC_HIGH destroyed A 4/4 (robust-min ≈0.005); STATIC 1/64 ≈0.882; Guardian ≈0.963; CAP16X never triggered; bundle `fcc14c53…`, 45/45+46/46 |
| ARK-019-V4 | Science-preserving, dose-qualified Guardian | SUPPORTED *(transcribed external audit)* | R0 | `GUARDIAN_CONTINUAL_PROXY_CANDIDATE`: PLASTIC_HIGH old 0/4; GUARDIAN_REPLAY & HYBRID old 4/4 + new 4/4; lower replay cost than permanent; **raw bundle NOT in repo; RUN_READINESS_V4 still says NOT_EXECUTED — contradiction recorded, byte re-audit required** |
| ARK-020-V1→V4 | Multi-skill continual battery (4 skills × 7 arms × 4 sets) | IMPLEMENTED_NOT_EXECUTED | – | 39/39 tests; executable pinned `b0d345d`; V1 5 defects → V2 4 blockers → V3 7 blockers repaired; A1.1–A1.3 durability amendments on dedicated branch (A1 audit: repo readiness claim was stronger than executable evidence); **no result artifacts on any branch; Drive outputs inaccessible from this audit** |
| ARK-021 | Retention-vs-reacquisition | IMPLEMENTED_NOT_EXECUTED | – | core+tests committed, portfolio top-1 |
| ARK-022 | Dormant retention | NOT_TESTED | – | PLAN only |
| ARK-023…030 portfolio | Information-gain ranking | SPECULATIVE | – | ARK-025 blocked on R1C evidence; ARK-028 blocked on V4 data |
| DISCOVERY-V6/V7 | Campaign bundle integrity | DEMONSTRATED (engineering) | R0 | 179.09 min 14/14; 166.24 min 11/11; V7 decision `RETENTION_EFFECT_TRANSFERRED_MECHANISM_UNRESOLVED` |

## 5. TRIQUETRA — cognition laboratory (all DEV-tier on weak V4 checkpoints)

| ID | Question (abridged) | Status | Key numbers |
|---|---|---|---|
| TQ-entity-value-factorial | Value vs entity vs pair repair | SUPPORTED | C2 value **46.73%** DEV, **46.15%** rep; C1 entity 0%; pair 26.17%; CI [32.7, 52.3]pp |
| TQ-query-value-matrix | Latent query-conditioned selection | SUPPORTED | raw rank-1 **0.2500 = chance exactly**; QCS CI includes 0 (both seeds); position ≈ 19–35× query |
| TQ-checkpoint-comparison | 22517→30400 what emerged | INCONCLUSIVE | duplication-elicitability 0.6→23.1%, generation 0.9→12.2%, QCS ~0 both |
| TQ-structural-OOD-E5 | E5 assist under structural shift | SUPPORTED (line closed) | E5dup−sham 0.0 (p=1.0); oracle 0.2417; "format hack; do not train" |
| TQ-X1-REAL-self-model | Learned self-model | **CONTRADICTED** | claimed 0.9545; always-negative baseline **0.9733** @ prevalence 0.0267 |
| TQ-readiness-gates | Subject qualification v1/v2 | SUPPORTED | v1 false green downgraded; v2: P1 0.083, P4 0.083 → **NOT_READY / INSUFFICIENT / NOT_IDENTIFIABLE**; no qualified local subject |
| TQ-binding-factorial | Entity-duplication claims | **INVALIDATED** | arrays never populated (empty→0.0 helper bug) |
| TQ-causal-decomposition | Addressing isolation | INCONCLUSIVE | selection contrast +0.3115 confounded (multi-factor) |
| TQ-competitive-binding | Beyond-length effect | INCONCLUSIVE | ~0 at L2–L4; unresolved L1 +0.125 anomaly |
| TQ-IBQ-v2-harvest | Basis qualification | CONTRADICTED | oracle coverage 0.0877; suspected empty generations |

**Triquetra bottom line:** its value is instrumentation discipline (false greens caught, gates that fail floors), not positive mechanism results. No new Triquetra execution since 2026-09-05.

## 6. CITADEL — independent auditor

| ID | Question (abridged) | Status | Key numbers |
|---|---|---|---|
| CIT-T1-series (T0/T1/T1B/T1C) | Any arm lifts off arithmetic on TPU? | SUPPORTED (negative) | core exact **0/1,000** in every arm (raw receipts; prose 0/500 is wrong); loss→1.90; copy heuristic 2.7%; EOS never supervised (MAX_TOKENS 1000/1000) |
| CIT-T1D | Six arms at 2–8M tokens | INCONCLUSIVE (official) | TEST exact 0–6.6%; **15,000/15,000 MAX_TOKENS endings**; content ≈5% post-hoc; teacher primitives **0.515** without composition; self-knowledge contract invalid (57/96 targets > 8 tokens); budget confound B/D/E |
| CIT-scoring-policy-tournament | Calibrated scorers neutralize length bias? | **DEMONSTRATED** (negative) | calibrated policies select fewest-token role **1.000 in 15/15 CUDA cells**; `production_scoring_mode: null` |
| CIT-e0-generator-repairs | Generator shortcut-free? | DEMONSTRATED (engineering) | v0.3.0 false green (81.77% bag-of-words) → v0.4.0 passes calibrated gate |
| CIT-500M-production-path-audit | 500M path connected? | SUPPORTED | at pin `28bf57a` (2026-09-06): corpus + entry point MISSING; tokenizer/schedule/eval AMBIGUOUS; **500M BLOCKED** (audit predates ~236 commits of cymek work) |
| CIT-T1E | EOS-corrected successor | NOT_TESTED | plan only |

## 7. ESOES — TPU-era blueprint + founding evidence

| ID | Question (abridged) | Status | Key numbers |
|---|---|---|---|
| ESO-PGE-continuation | Loss ⇒ cognition? (founding negative) | **DEMONSTRATED** | loss 2.1884→**1.9710**; probes 0/6, 0/8, 0/48, 0/12; selection ~chance; 329,908,224 certified tokens |
| ESO-SFT6-replication | Selection vs realization separable | SUPPORTED | lift 2.5052 vs 0.0192 nats (Δ 2.486, CI [1.748, 3.258], 35/40 positive); rank-1 63/119 vs greedy 24/119 |
| ESO-SFT7-margin | Margin objective helps rank-1? | CONTRADICTED | lift +0.1049 nats but rank-1 66→**64**/119 |
| ESO-EXP-v10/v11 | Pair-action composition | **INVALIDATED** | contaminated (stale candidates, missing baselines, irreproducible trainer) |
| ESO-e2-mechanism-canaries | Architecture mechanism priors | SUPPORTED (engineering) | residual init ratio 0.122–0.230; QK-norm invariance ≤1.0001 vs 256× unnormalized; native-BF16 optimizer REJECTED (clip overshoot ~0.3%); RoPE tolerance corrected; GQA math-backend-only on Windows |
| ESO-E3-data-objective | Cognition-fraction mixture | NOT_TESTED | `BLOCKED_UPSTREAM_INPUTS`; 65/20/15 is an implementation decision, not a measured optimum |

## 8. BRAMASTRA — discovery/binding lab (+ its audits)

| ID | Question (abridged) | Status | Key numbers |
|---|---|---|---|
| BRM-terminal-EOS | EOS supervision repairs complete answers | **DEMONSTRATED** | 0/32 → **32/32**, both seeds (601/602); adopted cross-program |
| BRM-transfer-baseline | Tiny-set learning transfers? | SUPPORTED (negative) | fresh 17.2%/25.8%; both-variants 1–2/64; rendering 0/128 |
| BRM-binding-diversity | Does variety create query control? | **CONTRADICTED** | 48.4% fresh ≈ query-blind baseline 50%; same answer in 62/64 despite changed query; 0/64 both-correct |
| BRM-discovery-dev | Learned discovery beats random? | INCONCLUSIVE | n.s. (seeds 701/702) |
| BRM-replay-retention | This-regime replay rescues retention? | CONTRADICTED *(provenance weak)* | carried at Tier-3; original receipt not re-located — see EVIDENCE_GAPS; do not conflate with ARK-017's treatment-exact replay |
| BRM-D02-depth-two | Depth-two inquiry beats one-step? | INCONCLUSIVE | Δ accuracy 0.0 / +0.022 (CIs include 0), 23 worlds, budget 2 |
| BRM cross-audit | Independent audit of all branches | SUPPORTED | found T1C raw 0/1,000 vs prose 0/500; independent EOS contract mismatch discovery |

## 9. Historical / infrastructure

| ID | Branch(es) | Status | Note |
|---|---|---|---|
| CORE-MC-selfmodel-line (MC-v9/10/11, milestone 0001) | core-exp | INCONCLUSIVE (historical) | V4-era promotions not re-audited under current standards; related X1 claims contradicted; V4 substrate later failed readiness v2 |
| SENORA-P35-CMS1-CAD | deleted `senora` (unreachable `30a8fa7`) | NOT_TESTED | entire program (dry-run receipts, sbatch launchers, decision receipts) survives only in unreachable objects |
| ITER-TPU-runtime-line | iterate500/900, main, core-frozen-v4 | DEMONSTRATED (engineering) | TPU hardening, SFT resume fix, frozen V4 core; no cognitive claims |

## 10. Duplicate / novelty map (cross-experiment)

| Relation | Experiments | Verdict |
|---|---|---|
| REPLICATION | ARK-002 → ARK-002B | transition replicated (R1) |
| REPLICATION | ARK-007 → ARK-007R | LOW protection replicated (R1) |
| REPLICATION | ARK-014 ↔ ARK-014-codex-rerun | order-robustness repair replicated on an independent run (R2); retention still zero-event |
| REPLICATION | CYR-GPU-011 production ↔ CYR-GPU-012 V24576 arm | production-representation 0% reproduced |
| REPLICATION (R3) | ESO-PGE ↔ CIT-T1-series/T1D ↔ ARK-001 | loss-without-cognition / no-lift-off family across substrates |
| REPLICATION (R3) | TQ-query-value-matrix ↔ BRM-binding-diversity | query control absent — two methods, two programs, same conclusion |
| REPLICATION (R3) | BRM-terminal-EOS ↔ CIT-T1D postmortem ↔ Cymek adopted contract | EOS supervision mechanically required |
| REPLICATION | ARK-003 ↔ CIT-T1D arms B/C | curriculum/teacher nulls converge |
| EXTENSION | ARK-013 → ARK-019 → ARK-020 | continual-learning question scaled from micro T2→T3 to real-text proxy to four-skill battery |
| EXTENSION | ARK-015 → ARK-016 → ARK-017-V2 | invariance narrowing → mechanism credit resolved (`BOTH_LEVERS_SUFFICIENT`) |
| CONFLICTING (apparent) | ARK-001 vs CYR-GPU-012/013 | vocab "doesn't matter" vs class-space "matters enormously": resolved by noting different manipulations (whole-vocab swap vs declared class space with active IDs fixed); R1C is the discriminating test |
| CONFLICTING (to resolve) | ARK-019-V3.1 `CONTROLLER_NOT_SUPPORTED` vs ARK-019-V4 `GUARDIAN_CONTINUAL_PROXY_CANDIDATE` (transcribed) | V4 fixed V3.1's two design defects; the positive claim currently rests on an out-of-repo external audit |
| CONFLICTING (to resolve) | BRM-replay-retention (rejected) vs ARK-017-V2 (replay protects) | different replay constructions ("this-regime" vs treatment-exact sparse); receipt for the BRAMASTRA arm missing |
| NEAR_DUPLICATE | CYR-GPU-009 ↔ CYR-GPU-010 | same underdosed-dose design family |
| ORTHOGONAL | Everything vs ARK-018 | only real-text-substrate experiment |

**Anti-novelty warnings for future work:**
1. Do not re-run "does LOW LR protect an acquired capability" (answered twice, R1) or "does curriculum/teacher accelerate" (answered twice, R3).
2. Do not design another query-conditioning claim without first passing readiness-v2-style subject qualification (three past claims died on this).
3. Do not run another vocabulary sweep: R1B already spent that compute; the open question is the R1C mechanism partition.
4. Do not re-test "more data variety ⇒ query control" without a query-swap-paired objective arm (BRAMASTRA's null stands).
