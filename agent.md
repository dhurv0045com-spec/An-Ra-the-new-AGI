# Agent brief: CYMEK research + readiness handoff

Branch: `cymek-500m-readiness`.

## CURRENT EXPERIMENT

`CYR-GPU-011` is the current operator Colab GPU experiment. It is an exposure-matched capability-emergence bridge designed from the executed CYR-GPU-009 result and demonstrated Arkenstone ARK-002B behavior.

History:
- CYR-GPU-001 through CYR-GPU-005: superseded before scientific execution.
- CYR-GPU-006/008: operator Cell-0 hardware-feasibility failures before scientific training; not negative ML evidence.
- CYR-GPU-007: superseded before execution after a compatibility recursion risk was found.
- CYR-GPU-009: EXECUTED on Tesla T4; both TINY parents memorized but failed candidate-free held-out G90 at 2M real tokens each. Bundle SHA256 `dc15f14d3bc81551b1f0b00285faa4b23c9e68f1341405377959a7aba108f216`.
- CYR-GPU-010: superseded before execution after semantic-dose audit showed batch16 x 18k was only 25% of ARK-002B's 1,152,000-row exposure box.
- CYR-GPU-011: frozen and preregistered; pending final exact-head readiness closure and operator execution.

Executable freeze (Commit A):
`0a97257e2b38db6dfa85cc6e58da0697591dde6b`

Preregistration commit (Commit B):
`486309fda09b1d89dd14a3655014f89e41c28883`

Preregistration: `docs/cymek/experiments/CYR-GPU-011/PREREGISTRATION.json`.
Readiness: `docs/cymek/experiments/CYR-GPU-011/RUN_READINESS.json`.
Notebook: `notebooks/cymek_colab_gpu_research_v11.ipynb`.
Expected bundle: `CYMEK_GPU_RESEARCH_V11_RESULTS.zip`.

## WHY CYR-GPU-011 EXISTS

CYR-GPU-009 established a clean bottleneck: TINY reached near-perfect train-probe behavior but failed candidate-free held-out arithmetic generalization. Audit against ARK-002B then exposed a dose confound: Arkenstone used batch64 for up to 18,000 updates = 1,152,000 semantic row presentations, whereas V9 and the first V10 design delivered only about 22% and 25% of that semantic exposure.

CYR-GPU-011 therefore measures semantic row presentations directly and, wall permitting, targets the same 1,152,000-row box at batch64/32/16 using 18k/36k/72k updates respectively.

Live Arkenstone was re-audited through `c16718a7841c3cc3eba2b4b2c0388a0e36c0b530`. ARK-017 and ARK-018 have plans/implementation but no raw scientific result at this freeze and are not treated as evidence.

## SCIENTIFIC CONTRACT

Two bridges use the real Cymek V5 4L/128w geometry: Q4/KV2, head32, FFN512, context512, QK norm, tied embeddings and `v5_model.core.initialize()`.

1. `COMPACT_BRIDGE`: exact frozen ARK-002B data, 19-symbol task vocabulary, 987,392 parameters. It keeps Cymek's canonical causal objective; unlike Arkenstone's helper it does not supervise an answer-prefix BOS. This residual difference is explicit.
2. `PRODUCTION_BRIDGE`: same geometry with the frozen 24,576-token production tokenizer, 4,130,688 parameters.

Hard wall is 175 minutes with 5 minutes reserved for packaging. Compact is capped at 25 minutes and can stop early on qualified G90; production receives the remaining science wall. A second independent production seed starts only if primary G90 is qualified and at least 40 minutes remain.

Candidate-free milestones use three consecutive evaluations. A final G90 claim additionally requires DEV_MEASUREMENT/STANDARD complete-exact-with-valid-EOS >=0.90. Training sees only the 500 frozen train rows; DEV_CONTROLLER may control milestone timing, DEV_MEASUREMENT cannot alter training, and SEALED_RESERVED is post-decision only.

Structural diagnostics are reported independently: STANDARD, COMMUTED, LOCALITY, CARRY, TRIPLE_ADD, THREE_DIGIT and production-only VERBAL. No aggregate 'reasoning score' and no broad reasoning/AGI claim is authorized.

## EVIDENCE / TEST STATUS

Executable Commit A focused CYR-GPU-011 CI: PASS, run `34393235657`.
On the exact executable tree, the full Cymek suite produced 568 passed, 1 skipped and 40 subtests passed. Its only failure was the expected stale exact-head closure-receipt meta-test; no model, optimizer, tokenizer, XLA, notebook or CYR-GPU-011 contract failed.

The final branch closure must still refresh the strict exact-head receipt after all non-receipt readiness metadata is committed. Do not weaken the verifier.

## PRODUCTION / TPU BOUNDARY

CYR-GPU-011 is GPU controlled-task development evidence only.

- GPU scientific execution: NOT RUN yet.
- TPU scientific execution: NONE.
- XLA accumulation-boundary mathematics: locally repaired/tested, not TPU-certified.
- PRE500M: NOT RUN / NOT AUTHORIZED.
- Production corpus: DATA_NOT_READY; future 5B corpus untouched.
- 500M campaign: NOT AUTHORIZED.
- Production promotion from CYR-GPU-011 alone: FORBIDDEN.

## NEXT OPERATOR ACTION

Only after `RUN_READINESS.json` is true and the final exact-head Cymek workflow is green: use a fresh Colab GPU runtime, open `notebooks/cymek_colab_gpu_research_v11.ipynb`, run Cell 0 and require `CYR-GPU-011 PREEXECUTION GATE: PASS`, then run Cell 1, authorize Google Drive, leave the long run alone, then run Cell 2 and return `CYMEK_GPU_RESEARCH_V11_RESULTS.zip` for raw evidence audit.
