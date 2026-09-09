# Agent brief: CYMEK research + readiness handoff

Branch: `cymek-500m-readiness`.

## CURRENT EXPERIMENT

`CYR-GPU-008` is the current operator Colab GPU experiment.

History:
- CYR-GPU-001 through CYR-GPU-005: `SUPERSEDED_BEFORE_EXECUTION`.
- CYR-GPU-006: preregistered, then rejected by the operator's real Colab hardware-feasibility gate before any scientific training. This is NOT negative ML evidence.
- CYR-GPU-007: `SUPERSEDED_BEFORE_EXECUTION` after final audit found a compatibility final-decision recursion hazard. No scientific execution occurred.
- CYR-GPU-008: frozen/preregistered, pending final receipt-only closure and operator execution.

Executable freeze (Commit A):
`504fe13e735e71baa3a01898f42930486067e5be`

Preregistration commit (Commit B):
`703a94114f34a6f9115d59f7b847722642378e61`

Preregistration: `docs/cymek/experiments/CYR-GPU-008/PREREGISTRATION.json`.
Notebook: `notebooks/cymek_colab_gpu_research_v8.ipynb`.
Expected returned bundle: `CYMEK_GPU_RESEARCH_V8_RESULTS.zip`.

## WHY 008 EXISTS

The operator ran CYR-GPU-006 Cell 0 on a real Colab GPU. Calibration completed, but the fail-closed resolver proved that the frozen worst-case 3-parent + 4-fork + 3-transfer-pair campaign could not fit the 170-minute hard wall. No scientific optimization began. 008 therefore adapts scope using hardware throughput only, before outcomes exist.

## SCIENTIFIC CONTRACT

- Real Cymek V5 only (`ModelSpec` + `v5_model.core.initialize`).
- Frozen 24,576 tokenizer.
- Candidate-free sustained G90: complete exact with valid EOS stop >= 0.90 for 3 consecutive DEV_CONTROLLER evaluations.
- Every qualified parent forks `HIGH_CONTINUE`, `LOW_CONTINUE`, `FIXED_TIME_HIGH_TO_LOW`, and `HYSTERETIC_HIGH_LOW` from identical model+optimizer bytes and the same future example stream.
- Exposure and switches are in actual real tokens, not nominal steps.
- Replicated claims require >=2 independent parent subjects; one parent cannot produce a winner.
- If hardware affords transfer, it is prospectively fixed: equal-age/equal-exposure `HYSTERETIC_HIGH_LOW` vs `LOW_CONTINUE`, both moved to HIGH LR on the same order-augmented registry-binding + fixed replay stream.
- Transfer keeps canonical/query-only/order-only/query+order measurements separate and requires >=2 independent parent pairs.
- LOW is treated as same-task retention evidence, not universal consolidation.

## HARDWARE-ONLY TIER RESOLUTION

The resolver uses only calibration status, measured training real-tokens/sec, measured candidate-free generation examples/sec, and the fixed wall budget. Scientific loss/accuracy/treatment outcomes are forbidden inputs.

Priority:
1. Search non-TINY proxies (`MIDI`, `MICRO`, `RESEARCH_SMALL`) across the preregistered tiers.
2. Only if no non-TINY tier fits, consider `TINY`.

Tiers:
- `FULL_3P_TRANSFER3`: 3 parents, retention + transfer.
- `CORE_2P_TRANSFER2`: 2 parents, retention + transfer.
- `RETENTION_2P_ONLY`: 2 parents, replicated same-task retention only.

All tiers preserve the preregistered floors of 2,000,000 actual acquisition tokens per parent and 500,000 actual continuation tokens per arm. Transfer, when enabled, has a 500,000-token/state floor. `TINY` can never become a production-facing research candidate.

## LIVE CROSS-BRANCH AUTHORITY

- Arkenstone audited SHA: `6acd9dcbdd28d00f387ffcd004253a813aca4b66`.
- Discovery V6 bundle SHA256: `1ec3224075de49b080e229111e4c4f897430c3eca888c487116aff07a65c8d15`.
- Evidence matrix: `docs/cymek/research/CROSS_BRANCH_EVIDENCE_AUDIT.md`.

Interpretation frozen into 008: ARK-011 supports same-task adaptive protection; ARK-012 leaves exact switch timing/state threshold unresolved; ARK-013 shows LOW alone does not solve no-replay cross-task interference; ARK-014 shows order augmentation repairs robust binding acquisition while its LR retention transfer screen remains zero-event/inconclusive.

## SOFTWARE / SAFETY

- Production XLA accumulation ordering is locally repaired: local microsteps accumulate first, then one gradient SUM at the logical update boundary, then one clip and one optimizer step. CPU oracle evidence is local-only; TPU remains unmeasured.
- Full CYR scientific mode is CUDA-only.
- Cell 0 copies preregistration outside the repo, checks out Commit A, verifies executable hashes/dependency blobs, runs deterministic tests, requires CUDA, verifies tokenizer/data identities, calibrates all allowed proxies, writes calibration JSON before resolving, then resolves hardware-only.
- Cell 1 consumes Cell-0 `RESOLVED` + `CALIBRATIONS` verbatim. The runner does not re-resolve from scientific outcomes.
- Candidate-free generation is batched and included in prospective runtime cost.
- Google Drive is the durable evidence/checkpoint store. Completed stages are reusable; incomplete stages restart from immutable source checkpoints/streams.
- Exceptions package partial evidence and `FAILURE.json` before re-raising.
- SEALED arithmetic remains post-decision measurement-only.

## TEST / CLOSURE STATUS

Dedicated CYR-GPU-008 CI is green: notebook 3/3 cells compile and 22/22 deterministic 008 + inherited scientific contract tests pass.

Full Cymek CI on preregistration head `703a9411...` produced 533 passed, 1 skipped, 40 subtests passed. The only failure was `test_exact_head_test_receipt`, correctly detecting that the historical closure receipt still described the older CYR-GPU-006 readiness tree. No model, optimizer, tokenizer, XLA-oracle, CYR-008, notebook, or scientific contract test failed.

Final release procedure is intentionally strict: finish operator/readiness metadata, run that exact tree, then refresh `artifacts/v5/cymek_500m_closure_test_receipt.json` in a receipt-only final commit. Do not weaken the verifier.

## TPU / DATA / 500M

- TPU: `IMPLEMENTED_PENDING_PRE500M_TPU`; zero CYR-GPU-008 TPU evidence.
- PRE500M: NOT RUN / NOT AUTHORIZED.
- Production data: `DATA_NOT_READY`; the future 5B corpus remains untouched.
- 500M campaign: NOT AUTHORIZED.
- A positive GPU result cannot directly authorize a production scheduler change.

## NEXT OPERATOR ACTION

Do not use V6 or V7. Once the final receipt-only closure is green, open `notebooks/cymek_colab_gpu_research_v8.ipynb`, select GPU, run Cell 0 only, and require `CYR-GPU-008 PREEXECUTION GATE: PASS`. Then run Cell 1 -> Cell 2 and return `CYMEK_GPU_RESEARCH_V8_RESULTS.zip`. The next step is raw evidence audit, not immediate promotion.
