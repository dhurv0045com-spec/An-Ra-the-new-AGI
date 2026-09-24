# EXPERIMENT EVIDENCE LEDGER

**Synthesis date:** 2026-09-24
**Re-freeze phase:** 3
**Historical phase-2 snapshot:** 2026-09-13 (preserved in the JSON `basis_head` and `branch_heads_at_audit` blocks)
**Authoritative machine copy:** [`EXPERIMENT_EVIDENCE_LEDGER.json`](EXPERIMENT_EVIDENCE_LEDGER.json) (schema `anra.evidence-ledger/v1`)
**Authority:** [`EVIDENCE_SOURCE_MANIFEST_2026-09-24.json`](EVIDENCE_SOURCE_MANIFEST_2026-09-24.json), SHA-256 `c164c735ec4e628a6311fe4c52117c8b58370321df61b00d4976c11769f0215f`, plus byte-identical imported evidence.
**Pre-consolidation tip:** `research/evidence-consolidation-2026-09-25` @ `90f77b7fa6ffd99f5a982263f03b2908298805ec`; parent of the consolidated head.

Evidence rules remain binding: raw artifacts beat result notes beat code beat plans; implementation without execution is never a scientific result; engineering failures are not scientific results; imported raw bundles remain external provenance references.

## Status counts

**90 ledger entries.** Counts are calculated from the JSON on 2026-09-24.

| Status | Count |
|---|---:|
| DEMONSTRATED | 27 |
| SUPPORTED | 16 |
| INCONCLUSIVE | 16 |
| SUPERSEDED | 8 |
| CONTRADICTED | 10 |
| NOT_TESTED | 7 |
| IMPLEMENTED_NOT_EXECUTED | 2 |
| INVALIDATED | 2 |
| SPECULATIVE | 1 |
| IN_PROGRESS | 1 |

`IN_PROGRESS` is used only for the partial Formation-Mux frontier. K8 has ended as a completed engineering-partial `INCONCLUSIVE` record. `IMPLEMENTED_NOT_EXECUTED` rows have empty metrics. R1C is no longer an unexecuted row.

## Branch and pre-consolidation-tip tables

The historical 2026-09-13 snapshot contains **15** tracked heads; the former “14” statement omitted `eval-integrity-001`. The historical values remain unchanged in the JSON. The 2026-09-24 import adds the compact evidence sources below without rewriting the historical table.

| Historical branch | 2026-09-13 head | Role/status in the historical snapshot |
|---|---|---|
| `cymek-500m-readiness` | `f2c27a6` | Cymek controlled GPU representation/mechanism line |
| `Arkenstone` | `4ae9e3b` | ARK discovery and continual-learning line |
| `arkenstone-ark020-v4` | `65d1ef3` | ARK-020 execution/readiness line; historical snapshot had no result |
| `codex/arkenstone-improvements` | `516c280` | ARK-014 independent rerun and compact receipts |
| `arkenstone-astra` | `ccb84fb` | BRAMASTRA implementation branch |
| `BRAMASTRA` | `f25fc96` | Engineering build and binding lab |
| `triquetra` | `f23f0af` | Weak-substrate cognition diagnostics |
| `citadel` | `e96c9a9` | Data/evaluation audits and T1-series evidence |
| `esoes` | `85f44b7` | TPU-era mechanism canaries and negatives |
| `core-exp` | `51124de` | Historical self-model line |
| `core-frozen-v4` | `f72f193` | Frozen inference-only V4 core |
| `main` | `b620f1c` | Frozen V4 research system |
| `iterate500` | `b438420` | Historical TPU/SFT engineering |
| `iterate900` | `6fbd2c0` | Historical TPU/SFT engineering |
| `eval-integrity-001` | `915e23f` | Independent evaluation-integrity audit |

| 2026-09-24 imported source | Source commit | Current evidence boundary |
|---|---|---|
| `origin/cymek-500m-readiness` | `dc915d4424103a7d528e519d50bd7fb460a83a9a` | Completed R1C controlled development mechanism evidence |
| `archive/deleted/cymek-v51-canary-v2-22d1bd4f05f1` | `22d1bd4f05f1deb62679e6f06a8dbe268edee9c0` | Narrow canary formation failure |
| `origin/cymek-next-core-architecture` | `28e3cd025e67c4922da24e20791d807474759674` | Completed S5 v8, floor-limited |
| `origin/cymek-beta` | `194b98c8bddbe85d362fd5bf391ad68e3303ec79` | Updated recovery-CI/custody records, blocked ROLE-TRANSFER-001 design, and HORM miniature results |
| `origin/cymek-cs-transfer-001` | `b52fe453bdc68a558b8db1752758ec89832d53f0` | CS protocol/amendments; final result remains the existing canonical note |
| `archive/deleted/eval-integrity-001-915e23ff22db` | `915e23ff22dbd6ab9125bbca3b8c8081795f073b` | Citadel data/evaluation not ready; T1D archived |
| `archive/deleted/codex__arkenstone-improvements-516c28060ab4` | `516c28060ab4e06cd990ab2a8bb9fef1fbf30628` | Narrow ARK-014 binding evidence |
| `origin/codex/cyhex-integrity-audit` | `67220e2cbf012b4fa6f5a37990bb15e19cbc0d64` | Provenance/custody diagnostics only |
| `origin/Gandiva` | `49a3717ae9dc81138f55dc2398c7615f8c80b4b1` | Completed K8 engineering-partial result; TPU handoff not run |
| `origin/arkenstone-ark020-v4` | `308a9865af519ccedd1fe13759e3b8ccdca85907` | ARK-020 DO_NOT_RUN readiness |
| consolidation target | `90f77b7fa6ffd99f5a982263f03b2908298805ec` | Pre-consolidation parent metadata; the consolidated state is committed on the target branch |

## Current-state summary

| ID | Status | Replication | Boundary and key result |
|---|---|---|---|
| **CYR-GPU-014-R1C** | **CONTRADICTED** | R2 | COMPLETE 24/24; `SOFTMAX_COMPETITION_NOT_SUFFICIENT`; mean `MASK_4096-FULL_24576` gap `-0.11038062283737024`; fixed physical V24576 controlled development mechanism evidence only. |
| **CS-TRANSFER-001** | **INCONCLUSIVE** | R1 | COMPLETE; `PARTIAL_OR_INTERACTION`; development identity AUC gap `-0.034765625`; sealed endpoint gap `-0.1395833333333333`; 1/4 positive development pairs; 3/4 sealed pairs favor V24576; one reversal. Raw Drive result remains external with SHA-256 `37e4bf741f9e55fc942fe4ade62d2d1c9619e4992e4a42ab507730809c12b150`. |
| **CYMEK-V5.1-CANARY-V2** | **DEMONSTRATED** | R0 | 360 updates, 1,474,560 tokens, all mechanical gates pass; `CANARY_V2_FAIL_FORMATION`; dev/sealed overall `0.519415/0.516641`; identity `0.177/0.189`; narrow canary only. |
| **FORMATION-MUX-001-S5-V8** | **INCONCLUSIVE** | R1 | COMPLETE 24/24 and sealed; formal `CS-MECH-002` and `REP-FORM-003A` NULL, but primary identity is near the zero baseline. M3-M2 `+0.277083` is exploratory only; no mechanism exoneration. |
| **FORMATION-MUX-001-V12-FRONTIER-PARTIAL** | **IN_PROGRESS** | R0 | 24/24 S5 development arms plus 2/24 TIE-role frontier arms; no sealed/final/checkpoint payload. Recovery-preflight engineering passed remotely, but exact recovery remains blocked on the absent original checkpoint-bearing Output; not a scientific verdict. |
| **ROLE-TRANSFER-001** | **NOT_TESTED** | R0 | Preregistered execution-blocked design; no trainer, official arm, sealed evaluation, or result. It supersedes the old tied-row placeholder as a design of record but is not authorized to run. |
| **HORM-003** | **DEMONSTRATED** | R1 | Separate five-seed miniature prospective `NOT_SUPPORTED` negative. |
| **HORM-004** | **DEMONSTRATED** | R1 | Separate five-seed miniature prospective `NOT_SUPPORTED` negative; success fraction `0.0` on every seed; rich dynamic appraisal was not disproved. |
| **BRAMASTRA-K8-20260922** | **INCONCLUSIVE** | R0 | Completed engineering-partial result: E0/E1 executed, E2 not a positive cognition result, E3 blocked by insufficient data, E4-E5 not run; no tool-learning/AGI/RSI inference. |
| **GANDIVA-TPU-100M-PREFLIGHT** | **NOT_TESTED** | R0 | Engineering handoff only: no TPU run, no optimizer update, no qualification. |
| **ARK-020-V4** | **IMPLEMENTED_NOT_EXECUTED** | R0 | Imported readiness is `DO_NOT_RUN/NOT_EXECUTED`; phase-boundary resume and identity defects confirmed; all authorization flags false. |
| **ARK-014 / ARK-014-codex-rerun** | **DEMONSTRATED** | R2 | Local imported compact receipts resolve the narrow order-robustness binding result; retention screen had zero events. |
| **CITADEL-EVAL-001** | **SUPPORTED** | R0 | Local imported evaluation-integrity audit resolves to `NOT_READY`; T1D is shortcut/leakage compromised; no cognition or production claim. |

## Complete experiment evidence index

| ID | Status | Replication | Result / boundary |
|---|---|---|---|
| CYR-GPU-001 | SUPERSEDED | R0 | Preregistered tournament design never executed; superseded by CYR-GPU-002. |
| CYR-GPU-002 | SUPERSEDED | R0 | Local test receipt only; no operator or scientific execution; superseded by CYR-GPU-003. |
| CYR-GPU-003 | SUPERSEDED | R0 | Pre-execution audit found 13 defects including three blockers; never run. |
| CYR-GPU-004 | SUPERSEDED | R0 | Runner violated the parent-restore contract; never executed. |
| CYR-GPU-005 | SUPERSEDED | R0 | Frozen retention design never launched; superseded before operator execution. |
| CYR-GPU-006-smoke | INCONCLUSIVE | R0 | Smoke receipt is stash-only; the scientific campaign stopped at hardware feasibility, so no retention result exists. |
| CYR-GPU-007 | SUPERSEDED | R0 | A self-recursion compatibility defect was found before execution; never run. |
| CYR-GPU-008 | SUPERSEDED | R0 | Calibration-only run failed the 170-minute wall gate; engineering evidence only. |
| CYR-GPU-009 | INCONCLUSIVE | R1 | Underdosed V5 TINY reached no G90 and no continuation arm; narrow result below full reference exposure. |
| CYR-GPU-010 | SUPERSEDED | R0 | Pre-execution arithmetic showed the planned retry remained under target exposure; never run. |
| CYR-GPU-011 | DEMONSTRATED | R0 | Production V24576 reached 0% held-out and 0/48 sealed, while compact representation reached 56.47%; correlated representation factors prevent unique causal attribution. |
| CYR-GPU-012-R1 | DEMONSTRATED | R0 | Fixed active IDs produced a non-monotonic V19/V4096/V24576 response; no universal optimum or unique mechanism. |
| CYR-GPU-013-R1B | SUPPORTED | R1 | Two fresh seeds replicated the intermediate-class-space direction, with a mixed seed-sensitive verdict. |
| CYR-GPU-014-R1C | CONTRADICTED | R2 | Completed 24/24 arms; masked softmax was not sufficient, with no V24576-optimality or production-remedy claim. |
| CS-TRANSFER-001 | INCONCLUSIVE | R1 | Completed paired physical-V4096/V24576 test was partial or interaction; V4096 was not a robust remedy and transfer remained unproven. |
| CYMEK-V5.1-CANARY-V2 | DEMONSTRATED | R0 | Completed 360-update canary passed mechanical gates but failed the frozen identity/copy formation threshold. |
| CYMEK-P35A | NOT_TESTED | R0 | Preregistered matched-compute experiment remained blocked on external identities and was never run. |
| CYMEK-closure-cycle-engineering | DEMONSTRATED | R1 | Resumable, fail-closed engineering canaries passed; production corpus and TPU certification remained open. |
| CYMEK-e1-tokenizer-tournament | INCONCLUSIVE | R0 | 24k was a local compression/parameter planning center only; the corpus was non-representative and capability validation remained open. |
| ARK-001 | CONTRADICTED | R0 | Micro-scale universal capacity-pathology and whole-vocabulary hypotheses failed; later class-space manipulations remained a distinct question. |
| REDTEAM-BV2 | DEMONSTRATED | R1 | Binding-v2 survived the finite red-team set; unconditional robustness was not established. |
| ARK-002 | DEMONSTRATED | R0 | A single-seed extended-budget run showed T2 transition saturation and motivated ARK-002B. |
| ARK-002B | DEMONSTRATED | R1 | Two fresh seeds replicated delayed generalization with substantial timing variance and no universal timing law. |
| ARK-003 | CONTRADICTED | R0 | Curriculum and aligned-teacher arms did not accelerate OOD emergence at the tested micro budget. |
| ARK-004A | SUPPORTED | R1 | Memorization timing did not predict generalization timing; precursor status was demoted to a marker and post-G90 instability appeared. |
| ARK-004A-R | SUPPORTED | R0 | Deterministic raw-receipt reanalysis retained precursor as a marker, not a prospective predictor. |
| ARK-005 | CONTRADICTED | R0 | EMA and weight-decay removal did not stabilize the generalized state; the LR-decay positive was single-seed only. |
| ARK-006 | SUPPORTED | R0 | A sub-1e-4 LR threshold candidate was provisional and provenance-limited, not a universal threshold. |
| ARK-007 | SUPPORTED | R1 | LOW continuation protected acquisition in the initial screen; ARK-007R superseded it as authority. |
| ARK-007R | DEMONSTRATED | R1 | HIGH continuation failed 9/12 matched forks and LOW failed 0/12 on micro T2. |
| ARK-009 | INCONCLUSIVE | R0 | Ordinary held-out accuracy reached 1.0 but composite robustness failed and no retention forks ran. |
| ARK-010 | DEMONSTRATED | R1 | Continued HIGH recovered 8/9 collapses versus 2/9 for immediate LOW. |
| ARK-011 | DEMONSTRATED | R1 | Post-recovery LOW reduced recollapse from 3/6 to 0/6 on micro T2. |
| ARK-012 | CONTRADICTED | R0 | Behavioral thresholds aliased to switch times; no unique capability-state threshold was identified. |
| ARK-013 | INCONCLUSIVE | R0 | New T3 skill never formed and all no-replay arms lost sustained T2; controller and plasticity-frontier claims remained blocked. |
| ARK-014 | DEMONSTRATED | R2 | Order augmentation repaired narrow binding robustness; the retention screen had zero events. |
| ARK-014-codex-rerun | DEMONSTRATED | R2 | Independent run replicated order-robustness qualification; retention again had zero events. |
| ARK-015 | DEMONSTRATED | R1 | Canonical accuracy stayed perfect while broader invariance collapsed; LOW and continued support protected robustness. |
| ARK-016 | INCONCLUSIVE | R0 | Only 1/12 continuation opportunities qualified, so update-cap mechanism credit was unresolved. |
| ARK-017-V2 | DEMONSTRATED | R1 | Lower applied update magnitude and exact sparse replay were independently sufficient; displacement was not causal. |
| ARK-018-V4 | DEMONSTRATED | R1 | Birth-content internalization missed its threshold while assimilation, science-NLL cost, and slower binding acquisition replicated; no identity claim. |
| ARK-019-V3.1 | DEMONSTRATED | R1 | New-skill formation failed, voiding a clean controller verdict; unprotected continuation lost the old skill while replay and reactive recovery protected it. |
| ARK-019-V4 | SUPPORTED | R0 | Conditional Guardian proxy candidate was externally audited and transcribed, but the raw bundle was absent and no byte-level re-audit existed. |
| ARK-020-V4 | IMPLEMENTED_NOT_EXECUTED | R0 | Readiness is DO_NOT_RUN/NOT_EXECUTED with resume and identity defects; all authorization flags are false. |
| ARK-021 | IMPLEMENTED_NOT_EXECUTED | R0 | Plan, core, and tests exist, but the retention-versus-reacquisition experiment was not executed. |
| ARK-022 | NOT_TESTED | R0 | Design-ready plan only; no dormant-retention result exists. |
| ARKENSTONE-portfolio-ARK-023-030 | SPECULATIVE | R0 | Portfolio-ranking artifact only; no experimental outcome. |
| DISCOVERY-V6 | DEMONSTRATED | R0 | Campaign bundle integrity passed 14/14 receipt hashes; this was engineering evidence. |
| DISCOVERY-V7 | DEMONSTRATED | R0 | Campaign bundle integrity passed 11/11 hashes and carried the ARK-015 transfer plus ARK-016 unresolved mechanism. |
| TQ-entity-value-factorial | SUPPORTED | R1 | Value recency repaired roughly 44–47% of failures; this was salience/recency, not learned addressing. |
| TQ-query-value-matrix | SUPPORTED | R1 | Raw rank stayed at chance and position effects dominated query effects on the weak V4 substrate. |
| TQ-checkpoint-comparison | INCONCLUSIVE | R0 | Readout machinery changed across two checkpoints while query control stayed absent; attribution was observational. |
| TQ-structural-OOD-E5 | SUPPORTED | R0 | E5 duplication gained no structural-OOD advantage and remained template-bound. |
| TQ-X1-REAL-self-model | CONTRADICTED | R0 | The reported pass was invalidated by a 0.9733 always-negative baseline versus 0.0267 positive prevalence. |
| TQ-readiness-gates | SUPPORTED | R0 | The v1 false green was downgraded and v2 found no qualified local research subject. |
| TQ-binding-factorial | INVALIDATED | R0 | Unpopulated arrays were silently coerced to zero, invalidating entity-duplication and interference claims. |
| TQ-causal-decomposition | INCONCLUSIVE | R0 | The multi-factor intervention did not isolate addressing and was downgraded to a development clue. |
| TQ-competitive-binding | INCONCLUSIVE | R0 | No beyond-length effect appeared at the floor; one L1 anomaly remained unresolved. |
| TQ-IBQ-v2-harvest | CONTRADICTED | R0 | Empty-generation and degeneracy flags prevented basis qualification. |
| CIT-T1-series | SUPPORTED | R1 | No tested objective, corpus, or 2.3x-scale arm produced lift-off; EOS and budget confounds bounded interpretation. |
| CIT-T1D | INCONCLUSIVE | R0 | All six arms scientifically failed and cross-arm comparison was inconclusive amid EOS, budget, teacher-diversity, and probe-contract confounds. |
| CIT-scoring-policy-tournament | DEMONSTRATED | R1 | Both calibrated policies selected the fewest-token role in all cells and failed the bias screen. |
| CIT-e0-generator-repairs | DEMONSTRATED | R1 | A false-green generator was repaired and v0.4.0 passed the calibrated shortcut gate. |
| CIT-500M-production-path-audit | SUPPORTED | R0 | The audit found production corpus and entry-point blockers plus ambiguous tokenizer, schedule, and evaluation wiring; it was stale at later pins. |
| CIT-T1E | NOT_TESTED | R0 | EOS-corrected successor plan existed but was never executed. |
| ESO-PGE-continuation | DEMONSTRATED | R0 | Held-out loss improved while all probed cognitive abilities remained at zero or chance on one lineage. |
| ESO-SFT6-replication | SUPPORTED | R1 | Assisted ranking improved while free generation failed, separating selection from realization. |
| ESO-SFT7-margin | CONTRADICTED | R0 | Margin loss lift did not improve rank-1 selection, so the margin objective was rejected. |
| ESO-EXP-v10-v11 | INVALIDATED | R0 | Candidate contamination, missing baselines, and irreproducible training invalidated composition claims. |
| ESO-e2-mechanism-canaries | SUPPORTED | R1 | Local engineering priors supported residual scaling, QK normalization, precision layout, and durability checks; no cognition benefit was tested. |
| ESO-E3-data-objective | NOT_TESTED | R0 | Mixture screens remained blocked on upstream inputs and were never run. |
| BRM-terminal-EOS | DEMONSTRATED | R1 | Complete answers rose from 0/32 without EOS to 32/32 with EOS across two seeds. |
| BRM-transfer-baseline | SUPPORTED | R1 | Tiny-set training showed substantial fresh-world failure and no complete changed-rendering transfer; the shift was multi-factor. |
| BRM-binding-diversity | CONTRADICTED | R0 | Fresh accuracy was fully explained by query-blind value copying, so query control was not earned. |
| BRM-discovery-dev | INCONCLUSIVE | R1 | Learned discovery selection did not significantly beat random selection at the tiny scale. |
| BRM-replay-retention | CONTRADICTED | R0 | A carried this-regime replay negative lacked a relocated receipt and remained Tier-3 provisional. |
| BRM-D02-depth-two | INCONCLUSIVE | R1 | Depth-two teaching showed no measurable accuracy advantage over one-step teaching at the tested budget. |
| CORE-MC-selfmodel-line | INCONCLUSIVE | R0 | Historical promotions were not re-audited and lacked a qualified substrate or current scientific authority. |
| SENORA-P35-CMS1-CAD | NOT_TESTED | R0 | Dry-run artifacts survive only in unreachable commits; no executed scientific campaign is evidenced. |
| ITER-TPU-runtime-line | DEMONSTRATED | R0 | Historical TPU, resume, and frozen-core work supplied engineering lineage only, with no learning claim. |
| CITADEL-DATA-001 | DEMONSTRATED | R0 | Mechanical audit found a last-number shortcut, 530 cross-split leaks, 13.5% duplication, and inadequate supply. |
| CITADEL-EVAL-001 | SUPPORTED | R0 | Local audit resolved to NOT_READY; T1D was shortcut/leakage compromised and PRE500M remained unexecuted. |
| BRM-B00-B12-build | DEMONSTRATED | R0 | Integrated B-series engineering built and was testable; no learning or capability result followed. |
| FORMATION-MUX-001-S5-V8 | INCONCLUSIVE | R1 | Completed 24/24 with sealed formal NULLs at a near-zero identity floor; no mechanism exoneration. |
| FORMATION-MUX-001-V12-FRONTIER-PARTIAL | IN_PROGRESS | R0 | Only 2/24 TIE-role frontier arms are present, with no sealed evaluation, final result, or checkpoint payload. |
| HORM-003 | DEMONSTRATED | R1 | Five-seed miniature hypothesis was NOT_SUPPORTED with a near-zero median loss difference. |
| HORM-004 | DEMONSTRATED | R1 | Five-seed miniature hypothesis was NOT_SUPPORTED under a zero-success, zero-entropy signal that could not test rich appraisal. |
| BRAMASTRA-K8-20260922 | INCONCLUSIVE | R0 | Completed engineering-partial result: E0/E1 executed, E2 not positive cognition, E3 blocked, and E4-E5 not run. |
| GANDIVA-TPU-100M-PREFLIGHT | NOT_TESTED | R0 | No TPU run, optimizer update, checkpoint result, or qualification receipt exists. |
| ROLE-TRANSFER-001 | NOT_TESTED | R0 | Prospective preregistered four-arm role-transfer design with 12 fresh blocks and fixed exposure; execution blocked, no official arm, sealed evaluation, or scientific result exists. |

## Interpretation boundaries

- R1C rejects inactive-softmax competition **as a sufficient mechanism within the tested scope**; it does not globally disprove every softmax interaction and does not establish V24576 optimality.
- R1C and CS together leave tied-row geometry, parameterization/initialization, optimizer/weight-decay/denominator interactions, and task/scale transfer open.
- Formation-Mux S5 formal NULLs remain authoritative but are floor-limited. Recovery-preflight CI qualifies engineering only; the absent checkpoint-bearing Output still blocks exact continuation. The partial TIE-role snapshot cannot lock a mechanism.
- ROLE-TRANSFER-001 is a hash-bound preregistered design, not a result. It supersedes the older tied-row placeholder as the design of record but remains blocked on recovery/frontier completion, implementation, evaluator freeze, no-overlap proof, and remote qualification.
- HORM-003 and HORM-004 are miniature prospective negatives. HORM-004’s zero success fraction means the signal lacked variance; rich dynamic appraisal was not disproved.
- K8 and TPU are engineering-only. No row authorizes production vocabulary, PRE500M, 250M, 500M, cognition, AGI, tool learning, TPU qualification, or RSI.

## Relation map

| Relation | Experiments | Current reading |
|---|---|---|
| EXTENSION | CYR-GPU-011 → CYR-GPU-012-R1 → CYR-GPU-013-R1B → CYR-GPU-014-R1C | Development-scale class-space discovery followed by completed mechanism test |
| EXTENSION | CYR-GPU-014-R1C → CS-TRANSFER-001 → CYMEK-V5.1-CANARY-V2 context | R1C → K01 → CS path; CS is partial/interaction, not transfer proof |
| EXTENSION | FORMATION-MUX-001-S5-V8 → FORMATION-MUX-001-V12-FRONTIER-PARTIAL | Completed S5 and later partial frontier are distinct records; recovery CI is engineering-only |
| SUPERSEDES PLACEHOLDER | ROLE-TRANSFER-001 replaces TIED-ROW-GEOMETRY-WD-001 as design of record | More rigorous preregistration, but still NOT_TESTED and execution-blocked |
| ORTHOGONAL | CS-TRANSFER-001 ↔ FORMATION-MUX-001-S5-V8 | Separate controlled surfaces and endpoints |
| REPLICATION | ARK-014 ↔ ARK-014-codex-rerun | Independent narrow binding acquisition rerun; retention remains zero-event |
| BLOCKED | ARK-019-V4 raw Guardian gap → ARK-020-V4 | ARK-020 is DO_NOT_RUN and cannot be interpreted as a result |
| ENGINEERING-ONLY | BRAMASTRA-K8-20260922 → GANDIVA-TPU-100M-PREFLIGHT | Completed K8 engineering-partial execution does not unlock TPU qualification or science |

## Next discriminating action

Start with `FMUX-CONTROL-METRIC-PREFLIGHT`: a cheap multi-seed Formation-Mux control-capability and checkpoint metric-resolution audit. Recovery-preflight engineering is remotely qualified, but exact recovery remains blocked until the original checkpoint-bearing Output is supplied. Continue the frozen TIE-role frontier only if both custody and capability/metric gates pass. The prior `TIED-ROW-GEOMETRY-WD-001` placeholder is superseded by the preregistered but unexecuted `ROLE-TRANSFER-001`, which still requires upstream completion, implementation, evaluator freeze, no-overlap proof, and independent qualification. Corpus regeneration remains required before natural-language or scale transfer; no broad expensive mechanism campaign starts from the S5 floor.
