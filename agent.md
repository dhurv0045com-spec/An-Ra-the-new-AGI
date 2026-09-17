# Agent brief: CYMEK research + readiness handoff

Branch: `codex/cyhex-integrity-audit` (audit baseline `d2abbdf`).

Current interpretation corrections: `docs/cymek/experiments/REPLAY_SOURCE_AUDIT.md`.
Historical initial-tensor identity and kernel selection both remain UNKNOWN.
The CPU norm replay is not a historical replication. FORMATION-MUX remains
`INCONCLUSIVE_AT_ZERO_BASELINE`; zero-baseline contrasts exonerate no mechanism.
No training, pilot/frontier launch, PRE500M, or production authorization follows.

## CURRENT SCIENTIFIC STATE

CYR-GPU-012 has **EXECUTED** locally (RTX 4050 Laptop, torch 2.11.0+cu128, 54.14 min) on 2026-09-17.
Registered plan: `experiments/CYR-GPU-012/PLAN.md` (preregistered before execution).
Result: `docs/cymek/experiments/CYR-GPU-012/RESULT.md`; receipt:
`artifacts/v5/cyr_gpu_012_result_receipt.json`. Frozen executable:
`1a1624ed4c62e1a176dc4bed97e628f0b648a31b`, isolated descendant of V11.
**Official verdict: `NO_G90_AT_FULL_EXPOSURE` (Branch B).** Compact at the full
1,152,000-presentation ARK-002B dose: train M99 confirmed (1,400), but G50/G90
never reached; final DEV_CONTROLLER 0% (maximum 12.5%); final STANDARD 0%; sealed 0/48.
Corrected same-band commutation: 52/52 pairs exact in both orders (narrow in-band
transfer only). V11's compact G50-at-2,200 trajectory was NOT reproduced.
No concrete training-path defect was found in the independent source audit.
The cause is unresolved; hardware/numerics have not been demonstrated causal.
Follow-up hash-verification of the original V11 ZIP confirms matching recorded
sampling-index digests and initial weight norms; scores already differ at update
200. Initial tensor identity remains unverified. No repeat training launched.

`CYR-GPU-011` has **EXECUTED** on a Google Colab Tesla T4. Do not describe it as pending.

Frozen executable: `0a97257e2b38db6dfa85cc6e58da0697591dde6b`.

Source bundle SHA-256: `fbec390f66223a19a998db6519f42c046ad8cb0a345c205b168f5bd1a86668e5`.

The raw V11 ZIP is intentionally **not stored in git**. Preserve only the distilled scientific record:

- reasoning + interpretation: `docs/cymek/experiments/CYR-GPU-011/RESULT.md`
- compact structured result: `artifacts/v5/cyr_gpu_011_result_receipt.json`

Official preregistered verdict: `NO_G90_WITH_INCOMPLETE_EXPOSURE`.

## WHAT V11 ACTUALLY FOUND

The run completed in 8383.78 s (~139.73 min) on a Tesla T4.

### COMPACT_BRIDGE

Real Cymek V5 4L/128w, 19-symbol arithmetic vocabulary, 987,392 parameters.

- 8,081 updates, batch 64
- 517,184 semantic row presentations = 44.89% of ARK-002B reference exposure
- 7,240,576 actual real tokens
- train M99 confirmed at update 1,400
- G50 confirmed at update 2,200
- no G90
- final DEV_CONTROLLER exact-with-EOS 54.69%
- final DEV_MEASUREMENT STANDARD exact-with-EOS 56.47%
- max observed controller exact 59.38%
- final STANDARD digit accuracy: ones 80.0%, tens 56.47%
- locality relation consistency 81.25%, both-exact 60.42%
- carry 0%, triple-add 37.5%, three-digit 0%

Interpretation: partial controlled-task generalization emerged, but the 25-minute compact cap stopped the subject at only 44.89% of the ARK reference exposure and before the main 9k–18k delayed-generalization region. This is not a clean compact null and not G90.

### PRODUCTION_PRIMARY

Same real Cymek V5 geometry/data/objective, frozen 24,576-token production tokenizer, 4,130,688 parameters.

- 18,000 updates, batch 64
- 1,152,000 semantic row presentations = 100% ARK reference exposure
- 9,216,000 actual real tokens
- train M99 confirmed at update 2,200
- G50 never reached
- G90 never reached
- DEV_CONTROLLER exact-with-EOS remained 0% throughout recorded trajectory
- final DEV_MEASUREMENT STANDARD 0%
- SEALED_RESERVED 0/48
- STANDARD tens-digit 0%, ones-digit 12.94%
- COMMUTED 2.35%, locality 0%, carry 0%, triple-add 4.17%, three-digit 0%, verbal 6.25%

Interpretation: strong single-seed evidence that semantic dose alone is insufficient under the current production representation. The model memorized the training probe but never formed the held-out arithmetic capability across the complete ARK semantic exposure box.

## POST-RUN DIAGNOSTIC ERRATUM

Do NOT treat the compact battery's `COMMUTATION_INVARIANCE=true` flag as demonstrated invariance.

ARK-002B's OOD axis places the first operand in unseen tens bands 6/7; reversing operands changes that axis and moves the first-operand role toward the train-like band. Therefore compact STANDARD 56.47% versus COMMUTED 100% is evidence of operand-role/order asymmetry, not a clean symmetric commutation test.

The correction is preserved in `RESULT.md`.

## STRONGEST CURRENT HYPOTHESIS

Capability formation remains the immediate bottleneck, not retention.

The historical V11 gap motivates representation/tokenization/output-space hypotheses inside Cymek, but CYR-GPU-012 did not reproduce that compact trajectory. Neither result isolates vocabulary size, BPE segmentation, number-token atomization, tied-embedding/output competition, or another correlated representation factor.

Do not claim that the tokenizer is proven causal yet.

## NEXT HIGHEST-INFORMATION EXPERIMENT

CYR-GPU-012 executed and returned **Branch B**: compact at full ARK-002B exposure reached NO_G90_AT_FULL_EXPOSURE on local hardware, with V11's early compact trajectory unreproduced. This fixed dose was insufficient for this seed/configuration/run. Per the CYR-GPU-012 preregistration, shift discovery toward Cymek-vs-Arkenstone objective/BOS supervision, initialization, optimizer grouping/precision, and architecture differences rather than vocabulary alone.

Before any factorial consumes compact-vs-production as ground truth, the CYR-GPU-011-compact vs CYR-GPU-012 reproducibility gap (same nominal seeds, same core training source, different environment) must be closed or explained. No causal explanation has been established.

The corrected paired commutation result (52/52 in-band exact vs 0/50 unseen-band, with 0% STANDARD) is a concrete partial-transfer phenomenon worth a dedicated preregistered probe experiment; it is not a G90 and does not authorize anything.

Do **not** rerun the whole V11 campaign and do not return immediately to HIGH-vs-LOW retention.

## HISTORY

- CYR-GPU-001..005: superseded before scientific execution.
- CYR-GPU-006/008: Cell-0 hardware-feasibility failures before scientific training; not ML evidence.
- CYR-GPU-007: superseded before execution.
- CYR-GPU-009: executed on T4; TINY memorized but no candidate-free held-out G90 at 2M real tokens per parent. Source bundle SHA `dc15f14d3bc81551b1f0b00285faa4b23c9e68f1341405377959a7aba108f216`.
- CYR-GPU-010: superseded before execution after semantic-dose audit.
- CYR-GPU-011: executed; distilled result described above. Raw ZIP not tracked in git.
- CYR-GPU-012: executed locally (RTX 4050, 54.14 min); **Branch B — NO_G90_AT_FULL_EXPOSURE**. Compact at full 1,152,000-presentation dose: M99 yes (1,400), G50/G90 never; final controller/STANDARD 0%; corrected in-band commutation 52/52 exact both orders; V11 compact trajectory unreproduced (cause unresolved). Frozen executable `1a1624ed4c62e1a176dc4bed97e628f0b648a31b`. Raw ZIP not tracked in git. Receipt `artifacts/v5/cyr_gpu_012_result_receipt.json`; RESULT `docs/cymek/experiments/CYR-GPU-012/RESULT.md`.

## PRODUCTION / TPU BOUNDARY

- broad reasoning / AGI claim: NO
- production promotion from V11: FORBIDDEN
- TPU scientific evidence from V11: NONE
- XLA accumulation-boundary mathematics: locally repaired/tested, not TPU-certified
- PRE500M: NOT RUN / NOT AUTHORIZED
- production corpus: DATA_NOT_READY
- 500M campaign: NOT AUTHORIZED
- future 5B corpus: untouched
