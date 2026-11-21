# V8-X3 — Beyond Prediction: Towards General Learning and Cognitive Intelligence (Execution Specification)

## 0. Execution Directive (Read First)
- **Execution Agent Manual:** Read [`execution_agent.md`](../../execution_agent.md) completely before coding.
- **Orchestrator Manual:** Designed strictly per [`orchestrator_agent.md`](../../orchestrator_agent.md) (`V8-X3` research program).
- **Target Git Branch:** `Elendril` (checked out cleanly on `origin/Elendril` at `a6c29ffd9e489c5005a91378d6cb93f87041ae75`; push solely via `git push origin HEAD:Elendril` without creating any other branch).
- **Quality-First Override:** Planthir / Palantiri (`anra_v5.palantiri`) is in an early starting condition. Reuse `anra_v5.palantiri` for custody, Launch Gates, paired stats, checkpoints, and verified archives by default. Whenever a helper (such as flat `run_registered_plan`, single-step `StaticPromptKernel`, Run-2 `BII-FROZEN` ordering, or hardcoded `build_pinned_kaggle_notebook`) blocks multi-stage DAG rigor, Run-2 missing-parent resilience, adaptive 11h one-go GPU execution, `LATEST`-only disk bounds, or batched evaluation speed, override or wrap it cleanly inside `experiments/x3/` or `experiments/x2/campaign.py` while keeping all Hard Scientific Laws intact, and document the reason in `PALANTIRI_FEEDBACK`.
- **Two-Layer Design:** Preserve existing `experiments/x3/{factorial,interventions,runner}.py` (the X2-gated pointer relation test) 100% intact, and add the full **V8-X3 Cognitive Intelligence Program** in `experiments/x3/{cognitive_battery,models,d3_objectives,module}.py` plus `tools/build_elendril_x3_notebook.py`.

---

## 1. Scientific Consolidation, Empirical Diagnosis, & Core Hypotheses

### 1.1 Consolidated 42/42 X1 Core Evidence (`A0`, `OWNER_RUN_01.md`, `OWNER_RUN_02.md`)
All 42 original X1 S20 jobs (`19.87M` params, `2,368` updates, `2,304` linked development evaluations per seed across seeds `7, 19, 41`) are verified complete:
1. **`P` vs `F-cap` (Typed Pointer Readout Advantage):** `P (A0)` `62.02%` vs `F-cap (A1)` `28.10%` (`+33.93` pts overall; `80.94%` vs `11.35%` on eligible evidence-change pairs, nearly identical `19.95M` parameter counts). Yet **narrative rendering (`16.15%`)** stays low and **P fails whenever answers cannot be copied verbatim from visible candidate tokens** (abstract rule induction, arithmetic synthesis, compositional string generation).
2. **`B0` vs `J` vs `P` (Typed Action Head vs Pointer Readout):** `B0` `29.99%` vs `J` `33.16%` (`+3.17` pts) vs `P` `62.02%` (`A0`). Typed action heads alone do not match pointer selection, **but `B0` (`20.31%`) and `J` (`26.22%`) beat `A0 P` (`16.15%`) on unfamiliar narrative renderings**, showing `P` over-relies on structured ledger surface tokens (`s=... -> ...`).
3. **`R` vs `R-noedge` vs `D0/D1/D2` (Architecture × Objective Interaction & A0 Deficit):**
   - `R (D2)` `47.64%` (`+2.53` pts over `R (D1)` `45.11%`, `+6.96` pts over `R (D0)` `40.68%`), while `P (D2)` `39.99%` regressed against `P (D1)` `41.04%` (`-1.06` pts) and `P (D0)` `42.13%`.
   - Yet **both `D2 R` (`47.64%`) and `D2 P` (`39.99%`) fell ~14–22 points behind `A0` (`61.65%` / `62.02%`), and `D2 R` narrative collapsed to `9.81%` vs `A0 R` `25.43%`**.
   - **Direct Code/Corpus Diagnosis (`anra_v5/elendril/v8x/data.py`, verified):** Across `quota_per_family=24` (`288` unique base worlds, `488` `D1-H` states vs `488` groups × `4` linked views = `1,952` `D2` samples), `D2` draws 4 linked presentations (`canonical`, `perm`, `irrel_row`, `cf`) from the same base world inside one micro-batch (`sampling="uniform"`, 1 group/step -> 4x fewer unique base worlds per update than `D1-H` `interleaved` batches) while equalizing `steps` without increasing world diversity or protecting CE gradients from auxiliary gradient conflicts. Furthermore, `A2` had `R-noedge` on `standard` only (`61.91%` overall, narrative `16.75%` vs `R` `25.43%`), leaving `R` vs `R-noedge` under `D1`/`D2` completely untested.
4. **Run-2 Operational Defect Found & Resolved Here:** `select_session_rows(FULL_X2_SESSIONS)` sorted registry rows before session order, putting `X2-BII-FROZEN-A0-P-s7` at index 36 without hosted A0 weight files and risking `FileNotFoundError` inside `build_lineage_runner_command`. V8-X3 preserves `FULL_X2_SESSIONS` topological order AND upgrades `run_admitted_waves` to record unbacked parents honestly as `BLOCKED_MISSING_PARENT` rather than crashing independent jobs.

### 1.2 Core X3 Scientific Hypotheses & Falsifiers
- **Hypothesis `H1` (Causal Test of `R`'s Cross-Slot Communication Under `D1`/`D2`):** Under matched `D1` and `D2` exposure, `R`'s advantage on edit pairs and narrative robustness (`+8.68` pts in `A0`) depends causally on its cross-slot edge messages (`R - R-noedge > +3.0` pts on narrative + edit-pair difference-in-differences `(R_D2 - R_D1) - (R-noedge_D2 - R-noedge_D1)`). *Falsifier:* If `R-noedge` matches or exceeds `R` under `D2` and narrative, edge messaging is redundant and D2 helps via generic slot-aggregator regularization.
- **Hypothesis `H2` (`D3-Hybrid`: Sampler Diversity + Conflict-Projected Gradients Solve the `A0` vs `D2` Trade-Off):** Injecting 75% diverse independent `interleaved` experiences + 25% paired counterfactual/invariance batches, plus **PCGrad-style gradient conflict projection** (`aux_grad <- aux_grad - proj(aux_grad, ce_grad)` whenever `dot(aux_grad, ce_grad) < 0` on shared backbone params), preserves `A0` base/narrative capability (`>= A0` floor) while gaining `D2`'s counterfactual precision on **BOTH `P` and `R`**. *Falsifiers:* Active ablations `D3-NoSurgery` (unprojected joint gradients) and `D3-NoAux` (mixed sampler without `eq/cf` losses) test whether world diversity alone vs gradient surgery accounts for the gain.
- **Hypothesis `H3` (`P-R-Iter`: Iterative Pointer-Relational Working Memory + Delta Revision + Metacognitive Halting + Dual Copy/Generate Output):** A bounded architecture where query-conditioned multi-read pointer attention feeds iterative cross-slot relational message passing over persistent delta-gated working memory `W^(t) = W^(t-1) + g_t * (v_t - W^(t-1))`, monitored by learned halting (`ACT` ponder gate) and dual uncertainty heads (epistemic reasoning uncertainty vs missing-information uncertainty), paired with a **Dual Copy/Generate Head** (pointer distribution + byte-level generative synthesis), solves BOTH pointer-copy tasks AND **answers strictly absent from context** (`C01`, `C02`, `C03`, `C04`, `C07`, `C10`).
  - *Matched Active Controls (`<= 5%` head overhead over shared `core.`, `<= 2%` parameter parity vs `P-R-Iter`):*
    1. `P-R-Iter-Fixed` (fixed `T=4` steps, no adaptive halting) -> tests adaptive computation vs fixed compute.
    2. `P-R-Iter-NoRel` (per-slot MLP replacing cross-slot edges) -> tests relational slot communication.
    3. `P-R-Iter-AddMem` (additive memory `W^(t) = LayerNorm(W^(t-1) + g_t * v_t)`) -> tests selective delta replacement vs additive accumulation.
    4. `P-R-Iter-NoCopy` (generative byte/slot head without pointer copy) -> tests whether pointer selection remains necessary once relational working memory exists.
- **Hypothesis `H4` (Evidence-Gated `S50` `50.55M` Scaling Law):** Only if `P-R-Iter + D3` beats `A0 P`, `D2 R`, and matched active controls on S20 (`20M`) does Stage 4 unlock `S50` (`width=768, layers=8, query_heads=12, kv_heads=4, ffn_width=2048 -> 50,552,576 params`), comparing `Base-S50`, `P-S50`, and `P-R-Iter-S50` (`CE` vs `D3`). This directly tests whether scaling conventional transformers erases or widens the cognitive gap.

---

## 2. Exact 10-Family Cognitive Testing Battery, 5 Learning Modes, & Split Custody

### 2.1 10 Substantially Independent Benchmark Families (`experiments/x3/cognitive_battery.py`)
Every family provides an independent generator, **strictly tool-free neural prompt** (`<= 600` bytes for zero-truncation S20/S50 `context_length=640`), exact reference oracle (`verify_cognitive_case`), canonical family signature (`canon_sig`), shortcut comparators (`prior_majority`, `copy_last_token`, `copy_first_token`, `nearest_demo_retrieval`, `always_abstain`), and destructive masks:
1. **`C01_NOVEL_OP_INDUCTION`**: Unfamiliar binary/ternary operators `a * b = y` across disjoint held-out rule families (`affine` in `train`/`dev`, `modular`, `piecewise`, `quadratic_xor` withheld strictly for `confirm`/`ood_transfer`), with version-space uniqueness verified across `k in {0, 1, 4, 16, 64}` demos; **gold answer absent from input**.
2. **`C02_FST_RULE_COMPOSITION`**: 4-state × 4-event finite-state transducers (`f` and composed `f o g`) using vectorized NumPy transition tensors (`(4, 4)`), exact candidate version-space solver across `0/1/4/16/64` shots; held-out composed laws withheld from training.
3. **`C03_GRID_TRANSFORM_ARC`** *(ARC-AGI-1 matched internal analog + external JSONL loader)*: Symbolic `3x3`/`4x4` grid transformations over disjoint geometric/color rule families (`color_remap`, `flip_h`, `flip_v`, `rotate90` vs `gravity_down`, `border_fill`, `tile_xor`), 2–3 demo input-output pairs -> synthesize novel test grid without tools.
4. **`C04_COMPOSITIONAL_SCAN`** *(SCAN matched internal analog + external loader)*: Compositional instruction sequences (`jump/walk/run/look` + `twice/thrice/opposite/around/and/after`) with **held-out primitive-modifier pairings & deeper nesting depths** (`train depth <= 2`, `eval/ood depth 3..4`).
5. **`C05_RELATIONAL_GRAPH_CF`** *(CLUTRR & X2-X3 crossed factorial)*: Multi-hop relational graphs crossing structured ledger vs unfamiliar prose/kinship narrative with interior-edge swaps, untouched sibling queries, irrelevant `ep=0` edits, and wrong-last-row distractors (`depth 4..5`).
6. **`C06_OPEN_WORLD_DEDUCTION`** *(ProofWriter matched internal analog)*: Multi-step Horn-clause / negation deduction with **3-valued open-world labels (`TRUE`, `FALSE`, `UNKNOWN`)**, depth `1..5`, and premise-deletion counterfactuals distinguishing false from unknown.
7. **`C07_HIERARCHICAL_LISTOPS`** *(ListOps matched internal analog)*: Nested prefix expressions (`MAX`, `MIN`, `MED`, `SUM10`, `FIRST`, `LAST`) evaluated at **held-out recursion depth** (`train d=2..3`, `eval/ood d=4..6`) + single-leaf counterfactual flips.
8. **`C08_ADAPTIVE_MEMORY_REVISION`**: Long multi-fact state-change streams featuring explicit mid-episode corrections (`Update/Correction`), untouched sibling facts, and stale late distractors -> 3-way selective-revision conjunction.
9. **`C09_METACOGNITION_SELF_CHECK`** *(bAbI/T14 belief-calibrated uncertainty & self-correction)*: Paired `solvable_easy`, `solvable_hard` (epistemic difficulty where iterative deliberation helps), and `missing_premise` (aleatoric/information uncertainty where querying/abstaining is optimal per exact Bellman DP `V*`), plus hint-based self-correction rounds.
10. **`C10_LANGUAGE_MATH_TRANSFER`** *(GSM8K-style multi-step prose + symbolic-to-prose transfer)*: Natural-language multi-step arithmetic reasoning and verbal rule explanations transferring rules first seen symbolically (`C01`/`C02`) into ordinary English word problems, answer strictly absent from prompt.

**External Benchmark Provenance Law (§10):** External datasets (ARC-AGI-1, SCAN, CLUTRR, ProofWriter, bAbI, ListOps, BIG-Bench, GSM8K) load via `load_external_benchmark_jsonl(...)` with `benchmark_provenance = "official_external"` when unmodified or `"interface_adapted_external"` when formatted, **reported strictly separately** from `"internal_matched_analog"` families.

### 2.2 Five Distinct Learning Mechanisms Tested Separately
`evaluate_five_learning_modes(...)` reports 5 orthogonal curves on `C01`–`C10`:
1. **Frozen-Weight In-Context Learning:** `0, 1, 4, 16, 64` demonstrations (`model.eval()`, zero parameter edits, reset working memory per episode).
2. **Persistent Internal-State / Working-Memory Adaptation:** Multi-round episode state carrying `W^(t)` across sequential examples/corrections with weights frozen.
3. **Bounded Gradient Parameter-Update Learning:** Fast adaptation inside `with torch.enable_grad():` over a few labeled demos with floating-param update norm cap (iterating strictly `model.named_parameters()`, never `bool` buffers in `state_dict()`).
4. **Tool-Free vs. Tool-Assisted Separation:** Comparing pure neural accuracy vs external solver/DP oracle (`oracle_gap`).
5. **Cross-Domain Rule Transfer:** Symbolic rule exposure (`C01`/`C02`) evaluated zero-shot on prose/grid formulations (`C10`/`C03`).

### 2.3 Split Custody & Complete-Cohort Sealed Confirmation Lock
- Fixed deterministic seeds: `TRAIN_SEED=20261`, `DEV_SEED=20262`, `CONF_SEED=20263`, `OOD_SEED=20264`, training replicates `(7, 19, 41)`.
- Fail-closed `verify_x3_split_disjointness(...)` rejects ANY shared canonical skeleton or withheld rule family across splits (`SPLIT_LEAK_DETECTED`).
- **`evaluate_confirmation_lock(...)`:** Computes SHA-256 over the immutable sealed `confirm` split (`CONFIRMATION_LOCK.json`) and **raises/returns `SEALED_CONFIRMATION_LOCKED` unless 100% of preregistered confirmation-eligible jobs across Stage 0–4 are complete (`pending_confirm_jobs == []`)**. Never opens `confirm` during `PARTIAL` or deferred sessions.

---

## 3. Multi-Stage Job Ledger, Hardware, Disk, Wall-Clock, & Smoke Math

### 3.1 Preregistered Job Ledger (`57` GPU Jobs Across `5` Evidence-Gated Stages + Stage 5 Sealed Confirmation)
| Stage | Study | Models × Objectives | Seeds | Updates / Job | Jobs | Parent / Gate |
|---|---|---|---:|---:|---:|---|
| **Stage 0** | `S0-SAMPLER` + `S0-REDGE` | Empirical corpus audit + (`R`, `R-noedge`) × `D2` *(crossed w/ existing D1)* | `7,19,41` | `1,024` | `6` | Unconditional (CPU audit `0` GPU s) |
| **Stage 1** | `D3` Algorithm Study | (`P`, `R`) × (`D3-Hybrid`, `D3-NoSurgery`, `D3-NoAux`) | `7,19,41` | `1,024` | `18` | Unconditional |
| **Stage 2** | `X3` Cognitive Architecture Screen (`S20`) | `P-R-Iter` + 4 active controls (`-Fixed`, `-NoRel`, `-AddMem`, `-NoCopy`) on `D3-Hybrid` | `7,19,41` | `1,024` | `15` | Starts parallel or right after D3 canary |
| **Stage 3** | Integration & Dual-Interface Transfer | `P-R-Iter` × (`D3-Hybrid`, `D2`) on 400-case 10-family curriculum + generative LM loss | `7,19,41` | `1,024` | `6` | Unconditional after Stage 2 base |
| **Stage 4** | Evidence-Gated `S50` (`50.55M`) Scaling | (`Base-S50`, `P-S50`, `P-R-Iter-S50 [CE]`, `P-R-Iter-S50 [D3]`) | `7,19,41` | `1,024` | `12` | **GATED on S20 Gate (`PASS`)**; `0.0` GPU s if `FAIL` |
| **Stage 5** | Sealed `confirm`, OOD & 5 Learning Modes | All qualified checkpoints + representative 5-mode panel + mistakes | `7,19,41` | `0` eval | Auto | **STRICTLY GATED (`pending_confirm_jobs == []`)** |

### 3.2 Real Kaggle 2×T4 Hardware, Disk, & Wall-Clock Math (Single-Session One-Go Budget)
- **Parameter & Checkpoint Memory:**
  - During an active wave (`2` concurrent workers): `FP32 model + AdamW ≈ 12 bytes/param` (`<= 625 MB` each -> `<= 1.25 GB` active mid-run checkpoints).
  - Upon job completion, final `LATEST` checkpoints strip redundant finished optimizer states (`optimizer.bin = {}` ~`60` B, preserving valid `CheckpointStore` SHA-256 manifests and mid-run resume compatibility): `S20` finished checkpoint `~80.3 MB` (`45` jobs `~3.61 GB`); `S50` `~204.1 MB` (`12` jobs `~2.45 GB`) -> **all `57` finished jobs total `<= 6.1 GB`**, well inside Kaggle's `19.5 GiB` disk ceiling.
  - `ELENDRIL_V8_X3_ANALYSIS.zip` (`~15–40 MB`, containing 100% of JSON metrics, Stage 5 confirm/OOD/5-modes, and mistake records) exports unconditionally; `ELENDRIL_V8_X3_FULL.zip` writes whenever `free_bytes > full_weights + 2 GiB`. Mirroring to `/kaggle/working/` uses zero-copy `os.link` (fallback `shutil.copy2`).
- **One-Go Wall-Clock Budget (Kaggle 12h max = `720m`):**
  - Default one-go session: `OWNER_AUTHORIZED = True`, `SESSION_MINUTES = 660`, `RESERVE_MINUTES = 25`, `ONE_GO = True`, `PER_JOB_DISK_MB = 95`, `DEVICES = "0,1"`.
  - CUDA FP16 AMP (`torch.amp.autocast("cuda", dtype=float16)` + `GradScaler("cuda")` with FP32 loss/PCGrad) + pre-packed device batches (`prepack_d3_batches`, `400` train cases = `25` batches/epoch -> `~41` epochs in `1,024` updates) accelerates step throughput `4–6×` (`~0.18–0.32 s/update` on T4 for S20 -> `~4.8 min/job` -> **`45` S20 jobs `~1.8–2.9h`** across `2×T4`; **`12` S50 jobs `~1.1–1.5h`** -> **entire 57-job program + Stage 5 completes in ONE single `~3.0–4.5h` Kaggle session**).
  - **Batched GPU Evaluation (`batch_size=32–64`, FP16 AMP `no_grad`):** Right-padded batched inference across `dev`/`confirm`/`ood_transfer` (`50–100×` faster than serial decode), verified to `atol=1e-4` on CPU/FP32.
- **Every-Architecture `u0000` CPU Smoke Mandate:** Before ANY multi-hour GPU wave, execute CPU `u0000` init + 1 train step + 1 eval step + save/restore across **EVERY** registered architecture (`P`, `R`, `R-noedge`, `Base`, `P-R-Iter`, `P-R-Iter-Fixed`, `P-R-Iter-NoRel`, `P-R-Iter-AddMem`, `P-R-Iter-NoCopy`) at both tiny/S20 and S50 specs!

---

## 4. `PALANTIRI_GAPS_AND_OVERRIDES` (Integration Map)

| Area | `anra_v5.palantiri` Primitives Reused | Gaps Found / Pre-Authorized Clean Overrides |
|---|---|---|
| Plugin, Launch Gates, CLI | `ExperimentModule(...).register()`, `CartridgeRunner`, `init/validate/run/verify-bundle/replay-bundle`, `doctor`, `calibrate` | Use directly (`backend_id="elendril_x3"`). Ensure `UniversalObservation.metadata` has zero truth keys and `decompose_subgoals()` returns `Sequence[SubgoalProbeSpec]`. |
| Multi-Stage DAG & Adaptive 11h Dispatch | `CampaignOrchestrator`, `reconcile_verified_resume(..., fail_on_rejection=True)`, `admit_slice`, `run_waves`, `IndependentWorkerScheduler`, `SessionJob`, `WorkerJobSpec` | Flat `run_registered_plan` lacks conditional stage gating (`Stage 4` S50 gate) & live elapsed-time stopping between waves. Override with `run_x3_campaign` composing `reconcile_verified_resume` + `admit_slice` + per-wave `run_waves` + `rem_min > reserve` check + Run-2 `BII-FROZEN` / `BLOCKED_MISSING_PARENT` graceful ledger deferral in `experiments/x2/campaign.py`. |
| Dual-GPU `CUDA_VISIBLE_DEVICES` Isolation | `IndependentWorkerScheduler` sets `CUDA_VISIBLE_DEVICES="i"` per lane & remaps `--device cuda:1` -> `--device cuda:0` | Child workers in `experiments/x3/module.py` always train on the `--device` string passed to them (`cuda:0`), avoiding double-indexing `cuda:1` inside single-GPU child process. |
| Paired Statistics, Controls & Reports | `verify_shared_backbone_and_head_overhead`, `bootstrap_paired_families`, `build_paired_seed_observations`, `compare_paired_episode_records`, `build_analyst_first_page_report`, `render_campaign_results_markdown` | Use directly for every contrast (`R vs R-noedge`, `D3 vs D2/D0`, `P-R-Iter vs 4 controls`, `S50` scaling). Never drop unpaired seeds; report `missing_pairs`. |
| Archives & Pinned Notebooks | `export_session_archives`, `verify_session_archive`, `rescore_compact_analysis_archive` | Hardcoded `build_pinned_kaggle_notebook` lacks multi-architecture `u0000` preflight, dual S20/S50 `cuda.synchronize` profiling, `os.link` mirroring, `LATEST`-only checkpoint pruning, and auto-unpacked `/kaggle/input/**` scanning. Author dedicated `tools/build_elendril_x3_notebook.py` passing `--check` + `ast.parse`. |

---

## 5. Preregistered Pass/Fail Qualification Gates

1. **`G0` Engineering & Anti-Shortcut Gate (CPU + Canary):**
   - `doctor == GREEN_SIGNAL`, `calibrate == 6/6 + SHORTCUTS_REJECTED_AND_ORACLE_QUALIFIED`.
   - All 10 families (`C01`–`C10`) oracle accuracy == `1.0`; max shortcut baseline `<= 0.35`; destructive mask pairs exist per family; `verify_x3_split_disjointness == True` (planted leak raises `SPLIT_LEAK_DETECTED`); batched-vs-serial eval diff `<= 1e-4`; `u0000` every-architecture smoke passes 100%; CLI `init -> validate -> run -> verify-bundle -> replay-bundle` reports `verified=True, MATCHED`.
2. **`G1` `H1` Relational Edge Gate (Stage 0 GPU):**
   - `(R - R-noedge) | D2` paired seed mean > `+2.0` pts on edit pairs + narrative OR `D2 - D1` interaction positive on all 3 seeds -> `VALIDATED`; otherwise `FALSIFIED` (reported honestly without hiding negative result).
3. **`G2` `H2` `D3-Hybrid` Learning Algorithm Gate (Stage 1 GPU):**
   - `D3-Hybrid` exceeds matched `D2` on `mean_all` AND matches/exceeds `A0` baseline `-3.0` pts on narrative while outperforming `D3-NoSurgery` on shared-gradient cosine stability (`>= +1.5` pts).
4. **`G3` `H3` `P-R-Iter` Cognitive Architecture Gate (Stage 2–3 GPU):**
   - `P-R-Iter + D3` beats EVERY active control (`-Fixed`, `-NoRel`, `-AddMem`, `-NoCopy`) and `A0 P` on: (a) 10-family macro mean (`>= +8.0` pts over `A0 P`), (b) `absent_from_input` families (`C01/C02/C03/C07/C10 >= +15.0` pts over `P`), (c) positive `0 -> 64`-shot slope (`auc_improvement >= +0.10`), and (d) metacognitive `epistemic_vs_missing_info` AUROC `>= 0.70`.
5. **`G4` `H4` Evidence-Gated `S50` Scaling Gate (Stage 4 GPU):**
   - **Executes ONLY if `G3` passes.** Evaluates whether `P-R-Iter-S50 (D3)` beats both `P-S50` and `Base-S50` on worst-dimension floor and shot-learning efficiency (`+5.0` pts).
6. **`G5` Sealed Confirmation Gate (Stage 5 GPU):**
   - Opens `confirm` **ONLY when `pending_confirm_jobs == []`**. Requires Holm-corrected `p < 0.05` across paired seeds & families on the primary cognitive contrast before declaring `QUALIFIED_GENERAL_LEARNER`.

---

## 6. Acceptance Gates & Final Copy-Paste Execution Contract

Required before pushing to `Elendril`:
1. Comprehensive `tests/test_elendril_x3_cognitive_program.py` covering all 10 cognitive families, split custody + planted leak, shortcut panel, destructive mask, `verify_shared_backbone_and_head_overhead` (`<=5%`/`<=2%`), `u0000` smoke across ALL 9 architectures + `S50` construction check, batched eval `atol=1e-4`, 5-mode evaluator, PCGrad conflict surgery fixture, mocked 2-GPU `FULL` dispatch (`in_process_runner`), tight-clock deferral/resume, complete-cohort Confirmation Lock guard, evidence-gated S50 gate, 3 fault-recovery tests (`in_process_runner`), X2 Run-2 `BII-FROZEN` ordering + `BLOCKED_MISSING_PARENT` regression test, and fresh-subprocess `PLANTHIR_BACKEND_MODULES=experiments.x3.module` CLI replay chain (`MATCHED`).
2. `python -m anra_v5.palantiri doctor` (`GREEN_SIGNAL`) + `calibrate` (`6/6`).
3. Fast tiered pytest run (`tests/test_elendril_x3_cognitive_program.py`, `tests/test_x2_handoff_repairs_and_x3.py`, `tests/test_elendril_x2_campaign_and_recovery.py`, `tests/test_elendril_tree.py`, `tests/test_planthir_two_agent_manuals.py`, `tests/test_planthir_v2_audit_upgrades.py`, `tests/test_planthir_declarative_cli_and_usability.py`).
4. Two-commit source pinning (`CODE_SHA` commit -> generate & verify `notebooks/ELENDRIL_V8_X3_KAGGLE_T4X2.ipynb` and updated `notebooks/ELENDRIL_V8_X2_KAGGLE_T4X2.ipynb` + `ast.parse` + `--check` + `git archive <CODE_SHA>` clean import -> curated `Experiment Result/V8-X3/{README.md,TEMPLATE_METRICS.json}` + `<=5`-line `progress.md` -> `FINAL_HEAD` commit -> push solely to `origin/Elendril`) + `PALANTIRI_FEEDBACK` section.
