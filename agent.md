# Agent brief: CYMEK research + readiness handoff

Branch: `cymek-500m-readiness`.

## CURRENT EXPERIMENT

`CYR-GPU-010` is the current operator Colab GPU experiment.

History:
- CYR-GPU-001 through CYR-GPU-005: superseded before scientific execution.
- CYR-GPU-006/008: operator Cell-0 hardware-feasibility failures before scientific training; these are NOT negative ML evidence.
- CYR-GPU-007: superseded before execution after a final audit found a compatibility recursion risk.
- CYR-GPU-009: EXECUTED on Tesla T4. Returned bundle SHA256 `dc15f14d3bc81551b1f0b00285faa4b23c9e68f1341405377959a7aba108f216`.
- CYR-GPU-010: READY_FOR_OPERATOR_COLAB_GPU_RUN.

Executable freeze (Commit A):
`4f6e04e870e7fad0af7311e2f57979203ca4aa85`

Preregistration commit (Commit B):
`ea32a94e8877f1bb820c2925fc9e944ad3181dac`

Preregistration: `docs/cymek/experiments/CYR-GPU-010/PREREGISTRATION.json`.
Readiness: `docs/cymek/experiments/CYR-GPU-010/RUN_READINESS.json`.
Notebook: `notebooks/cymek_colab_gpu_research_v10.ipynb`.
Expected bundle: `CYMEK_GPU_RESEARCH_V10_RESULTS.zip`.

## WHY 010 EXISTS

CYR-GPU-009 showed the real Cymek TINY proxy memorizing its training probe without candidate-free held-out generalization: both seeds consumed 2,000,000 actual real tokens / 15,632 optimizer updates; neither reached G90. No HIGH/LOW retention fork executed. The immediate question is therefore capability formation/generalization, not post-G90 retention.

Live Arkenstone was audited at `59e1b805b7d93b7f2e1e9d3ea66b34c4fabca9c8`:
- ARK-002B replicated memorize-first -> delayed generalize-later behavior on a 4L/128w Micro subject; transition timing spans roughly 9k-18k updates across seeds.
- ARK-003 did not demonstrate acceleration from simple curriculum or teacher/decomposition suffixes; CYR-010 therefore keeps flat acquisition.
- ARK-015 demonstrated non-arithmetic invariance brittleness under narrow HIGH continuation: 8/8 NARROW_HIGH failures vs 0/8 LOW and 0/8 fully augmented HIGH.
- ARK-016 was inconclusive due low event rate.
- ARK-017 is preregistered/implemented but had no RESULT artifact at the CYR-010 freeze; do not treat it as evidence.

## CYR-GPU-010 SCIENTIFIC CONTRACT

Primary subject: real Cymek V5 `RESEARCH_SMALL`, the closest Cymek scale match to Arkenstone's historical 4L/128w Micro: 4 layers, width 128, Q4/KV2, head32, FFN512, context512, QK norm, production 24,576 tokenizer, real `v5_model.core.initialize()` and production backend.

Fresh fixed seed: 3101. HIGH LR 1e-3. Batch rows 16. RESEARCH_SMALL maximum 18,000 optimizer updates. Candidate-free evaluation every 200 updates. Hard Colab wall 175 minutes with 5 minutes reserved for packaging. Hardware-only fallback to TINY 36k updates is allowed only if RESEARCH_SMALL cannot prospectively fit; TINY carries a lower claim ceiling.

Milestones: M99 train-probe >=.99; G50 DEV_CONTROLLER >=.50; G90 onset >=.90; sustained G90 = three consecutive candidate-free DEV_CONTROLLER evaluations >=.90. M99/G50/G90/final checkpoints are persisted.

Structural reasoning battery is measurement-only except STANDARD:
- STANDARD: frozen held-out T2 capability metric.
- COMMUTED: operand-order robustness diagnostic only.
- LOCALITY: paired counterfactual interventions; score exact output-delta relation and both-exact.
- RENDERING: unseen natural-language surface-form robustness.
- THREE_DIGIT: deterministic no-carry 3-digit length/compositional extrapolation diagnostic.
- Per-digit accuracy is recorded.

If G90 is confirmed early enough, remaining wall may run an exploratory matched post-G90 stress from the exact G90 model+optimizer checkpoint: `NARROW_HIGH` vs `SUPPORT_HIGH_1OF16`, same HIGH LR and same semantic-example stream, with only 1/16 presentation support differing. This imports the ARK-015 experimental lesson but does not claim ARK-017 mechanism credit.

## EVIDENCE / TEST STATUS

Executable Commit A dedicated CI: PASS, run `34376821098`.
Preregistration Commit B dedicated CI: PASS, run `34377200356`; all 3 notebook cells compile and focused CYR-010 contracts pass.

`RUN_READINESS.json` is `READY_FOR_OPERATOR_COLAB_GPU_RUN` with zero known blockers.

## CLAIM BOUNDARY

Even a positive CYR-GPU-010 result is single-seed controlled-task GPU development evidence. It does not establish AGI, broad natural-language reasoning, TPU equivalence, or permission to change production training.

TPU: no CYR-GPU-010 evidence; PRE500M not run/authorized.
Production corpus: DATA_NOT_READY; future 5B corpus untouched.
500M campaign: NOT AUTHORIZED.

## NEXT OPERATOR ACTION

Use a fresh Colab GPU runtime. Open `notebooks/cymek_colab_gpu_research_v10.ipynb`. Run Cell 0 and require `CYR-GPU-010 PREEXECUTION GATE: PASS`. Confirm the selected proxy (expected RESEARCH_SMALL on hardware similar to the prior T4). Then run Cell 1, authorize Google Drive, leave the long run alone, then run Cell 2 and return `CYMEK_GPU_RESEARCH_V10_RESULTS.zip` for raw evidence audit.
