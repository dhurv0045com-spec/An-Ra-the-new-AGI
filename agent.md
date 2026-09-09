# Agent brief: CYMEK research + readiness handoff

Branch: `cymek-500m-readiness`. No CYR-GPU-006 scientific training has run.

## CURRENT EXPERIMENT

`CYR-GPU-006` is the current Colab GPU experiment. CYR-GPU-001 through CYR-GPU-005 are `SUPERSEDED_BEFORE_EXECUTION`; none is GPU scientific evidence.

Executable freeze (Commit A):
`125b25c19204cce1994deebbfc4957119f2ae31f`

Preregistration commit (Commit B):
`ede8aec71f903e13061064742fcd0229ee093ce5`

The preregistration is `docs/cymek/experiments/CYR-GPU-006/PREREGISTRATION.json`. The notebook always reads that file from the branch, copies it outside the repo, checks out Commit A, verifies SHA256/file blob identities, and only then allows calibration.

## SCIENTIFIC DESIGN

- Real Cymek V5 only (`ModelSpec` + `v5_model.core.initialize`).
- Frozen 24,576 tokenizer.
- Three independent acquisition seeds: 707/808/909; every seed acquires once at HIGH LR.
- Candidate-free G90: DEV_CONTROLLER complete exact with valid EOS stop >=0.90 for 3 consecutive evaluations.
- Each qualified parent forks `HIGH_CONTINUE`, `LOW_CONTINUE`, `FIXED_TIME_HIGH_TO_LOW`, and `HYSTERETIC_HIGH_LOW` from identical model+optimizer bytes and identical future batch hashes.
- Exposure is actual real tokens; hardware-only resolver measures training throughput and batched-generation throughput before scientific outcomes.
- Arithmetic retention winner requires >=2 independent contract-valid parent experiments; one parent can never win.
- Arkenstone-informed transfer is prospectively fixed: equal-age/equal-exposure `HYSTERETIC_HIGH_LOW` versus `LOW_CONTINUE`, both moved to HIGH LR on the same order-augmented registry-binding stream with fixed old-T2 replay.
- Transfer keeps canonical/query-only/order-only/query+order metrics separate and requires >=2 independent parent pairs.
- LOW protection is treated as same-task Micro evidence, not universal consolidation. ARK-012 leaves switch thresholds unresolved; ARK-013 rejects LR-only no-replay protection; ARK-014 establishes order augmentation as necessary for the chosen binding subject.

## LIVE CROSS-BRANCH AUTHORITY

- Arkenstone: `6acd9dcbdd28d00f387ffcd004253a813aca4b66`.
- Discovery V6 bundle SHA256: `1ec3224075de49b080e229111e4c4f897430c3eca888c487116aff07a65c8d15`.
- BRAMASTRA observed head: `415250f44179f3310e5dc55addb21722290604fa`; no new execution claim is inferred merely from its later documentation commit.
- Current matrix: `docs/cymek/research/CROSS_BRANCH_EVIDENCE_AUDIT.md`.

## SOFTWARE

- Production XLA accumulation math is repaired: all accumulation microsteps remain local, followed by one gradient SUM collective at the logical update boundary, one clip and one optimizer step. CPU oracle evidence is local-only; real TPU remains unmeasured.
- CYR full mode is CUDA-only and has no CPU fallback.
- Cell 0 calibration/resolution is passed verbatim into Cell 1; the scientific runner does not silently re-resolve from outcomes.
- Candidate-free generation is batched and included in prospective runtime cost.
- Google Drive is the durable stage/evidence store. Complete stages are reused; ambiguous partial arms restart from the same frozen parent/tail.
- Exceptions write `FAILURE.json` and package partial evidence before re-raising.

## TEST / READINESS STATUS

Dedicated CYR preexecution CI at executable Commit A: PASS — 3/3 notebook code cells compiled and 25/25 deterministic CYR contract/transfer tests passed. Executable SHA256 values are frozen in the preregistration.

Broader production-contract CI on the same executable tree passed every substantive current-code test but failed the historical exact-head closure-receipt meta check because that receipt still points at the superseded CYR-GPU-005 tree. That receipt must be refreshed from the current tree-equivalent CI evidence before `RUN_READINESS=true` is written.

Current status: `PREREGISTERED_PENDING_FINAL_READINESS`. Do not run Cell 1 until `docs/cymek/experiments/CYR-GPU-006/RUN_READINESS.json` exists with `ready_for_operator_colab_gpu_run=true`.

## TPU / DATA / 500M

- TPU: `IMPLEMENTED_PENDING_PRE500M_TPU`; zero TPU evidence for CYR-GPU-006.
- PRE500M: NOT RUN / NOT AUTHORIZED in this cycle.
- Production data: `DATA_NOT_READY`; future 5B corpus remains untouched.
- 500M campaign: NOT AUTHORIZED.

## NEXT ACTION

Finish final branch CI/closure-receipt refresh, write truthful `RUN_READINESS.json`, then operator opens `notebooks/cymek_colab_gpu_research_v6.ipynb`, selects a GPU runtime, runs Cell 0 only, and proceeds to Cells 1→2 only if Cell 0 prints `CYR-GPU-006 PREEXECUTION GATE: PASS`.
