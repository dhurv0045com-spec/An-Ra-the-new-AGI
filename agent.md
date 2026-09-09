# Agent brief: CYMEK research + readiness handoff

Branch: `cymek-500m-readiness`. No CYR-GPU-006 scientific training has run.

## CURRENT EXPERIMENT

`CYR-GPU-006` is the current Colab GPU experiment. CYR-GPU-001 through CYR-GPU-005 are `SUPERSEDED_BEFORE_EXECUTION`; none is GPU scientific evidence.

Executable freeze (Commit A):
`125b25c19204cce1994deebbfc4957119f2ae31f`

Preregistration commit (Commit B):
`ede8aec71f903e13061064742fcd0229ee093ce5`

Preregistration: `docs/cymek/experiments/CYR-GPU-006/PREREGISTRATION.json`.
Readiness: `docs/cymek/experiments/CYR-GPU-006/RUN_READINESS.json`.
Notebook: `notebooks/cymek_colab_gpu_research_v6.ipynb`.
Expected returned bundle: `CYMEK_GPU_RESEARCH_V6_RESULTS.zip`.

## SCIENTIFIC DESIGN

- Real Cymek V5 only (`ModelSpec` + `v5_model.core.initialize`).
- Frozen 24,576 tokenizer.
- Three independent acquisition seeds: 707/808/909; every seed acquires once at HIGH LR.
- Candidate-free G90: DEV_CONTROLLER complete exact with valid EOS stop >=0.90 for 3 consecutive evaluations.
- Each qualified parent forks `HIGH_CONTINUE`, `LOW_CONTINUE`, `FIXED_TIME_HIGH_TO_LOW`, and `HYSTERETIC_HIGH_LOW` from identical model+optimizer bytes and identical future batch hashes.
- Exposure is actual real tokens; the hardware-only resolver measures real training throughput and batched candidate-free generation throughput before scientific outcomes.
- Retention requires >=2 independent contract-valid parents; one parent can never produce a winner.
- Transfer is prospectively fixed, not outcome-selected: equal-age/equal-exposure `HYSTERETIC_HIGH_LOW` versus `LOW_CONTINUE`, both moved to HIGH LR on the same order-augmented registry-binding stream with fixed old-T2 replay.
- Transfer keeps canonical/query-only/order-only/query+order metrics separate and requires >=2 independent parent pairs.
- LOW protection is treated as same-task Micro evidence, not universal consolidation.

## LIVE CROSS-BRANCH AUTHORITY

- Arkenstone: `6acd9dcbdd28d00f387ffcd004253a813aca4b66`.
- Discovery V6 bundle SHA256: `1ec3224075de49b080e229111e4c4f897430c3eca888c487116aff07a65c8d15`.
- BRAMASTRA observed head: `415250f44179f3310e5dc55addb21722290604fa`; no new execution claim is inferred merely from its later documentation commit.
- Evidence matrix: `docs/cymek/research/CROSS_BRANCH_EVIDENCE_AUDIT.md`.

Arkenstone interpretation frozen into CYR-GPU-006: ARK-011 supports same-task adaptive protection; ARK-012 leaves exact timing/state threshold unresolved; ARK-013 shows LOW alone does not solve no-replay cross-task interference; ARK-014 shows order augmentation repairs robust binding acquisition while its LR retention screen is zero-event/inconclusive.

## SOFTWARE / SAFETY

- Production XLA accumulation ordering is repaired locally: accumulate local microsteps first, then one gradient SUM at the logical update boundary, one clip, one optimizer step. CPU oracle evidence is local-only; TPU remains unmeasured.
- CYR full mode is CUDA-only and has no CPU fallback.
- Cell 0 copies preregistration outside the repo, checks out Commit A, verifies hashes/dependencies, calibrates, resolves hardware-only, and must print `CYR-GPU-006 PREEXECUTION GATE: PASS` before Cell 1 is allowed.
- Cell 1 consumes Cell-0 `RESOLVED` + `CALIBRATIONS` verbatim; the scientific runner does not silently re-resolve from outcomes.
- Candidate-free generation is batched and included in prospective runtime cost.
- Google Drive is the durable stage/evidence store. Complete stages are reused; ambiguous partial arms restart from the same frozen source state/tail.
- Exceptions package `FAILURE.json` and partial evidence before re-raising.
- SEALED arithmetic is measurement-only and used after optimization/transfer decisions.

## TEST / READINESS STATUS

**READY_FOR_OPERATOR_COLAB_GPU_RUN.** `RUN_READINESS.json` has `ready_for_operator_colab_gpu_run=true` and no blockers.

Evidence:
- Dedicated CYR preexecution CI run `34333048587`: PASS, 25/25 deterministic CYR contract/transfer tests, 3/3 notebook code cells compile.
- Broader Cymek production run `34333048641`: 522 substantive tests passed, 1 skipped; the only raw failure was `test_exact_head_test_receipt` because the historical closure receipt still named the superseded CYR-GPU-005 tree. No model, training, tokenizer, XLA-oracle, CYR, or notebook scientific test failed.
- The closure receipt is refreshed in the final receipt-only commit against the finished readiness tree. Do not weaken or bypass the exact-head verifier.

## TPU / DATA / 500M

- TPU: `IMPLEMENTED_PENDING_PRE500M_TPU`; zero CYR-GPU-006 TPU evidence.
- PRE500M: NOT RUN / NOT AUTHORIZED in this cycle.
- Production data: `DATA_NOT_READY`; future 5B corpus remains untouched.
- 500M campaign: NOT AUTHORIZED.

## NEXT OPERATOR ACTION

Open the Colab notebook, select GPU, run Cell 0 only, require the exact PASS message, then Cell 1 -> Cell 2. Return `CYMEK_GPU_RESEARCH_V6_RESULTS.zip`. The next research step after that is evidence audit, not immediate promotion.
