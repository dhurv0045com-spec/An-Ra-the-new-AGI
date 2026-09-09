# CYR-GPU-011 — RESULT

Status: **EXECUTED / COMPLETE** on Google Colab Tesla T4.

Raw bundle: `artifacts/v5/CYMEK_GPU_RESEARCH_V11_RESULTS.zip`

Bundle SHA-256: `fbec390f66223a19a998db6519f42c046ad8cb0a345c205b168f5bd1a86668e5`

Frozen executable: `0a97257e2b38db6dfa85cc6e58da0697591dde6b`

Wall time: **8383.78 s = 139.73 min**.

The bundle contains 12 JSON evidence files; all parsed successfully. Candidate-free prediction receipts for final controller and SEALED outputs were independently re-hashed from their row payloads and matched their recorded SHA-256 values.

## Official preregistered verdict

`NO_G90_WITH_INCOMPLETE_EXPOSURE`

This wording is correct for the combined bridge decision because the compact bridge timeboxed at only 44.89% of the ARK-002B reference semantic exposure and never qualified G90. It does **not** erase the stronger condition-specific observation that the production-tokenizer bridge completed the entire ARK semantic exposure box and still had no held-out lift-off.

No broad reasoning claim, production promotion, PRE500M authorization, TPU claim, or 500M authorization follows from this result.

## Compact bridge — partial structural emergence, no G90

Model: real Cymek V5, 4L/128w, 19-symbol arithmetic vocabulary, 987,392 parameters.

- status: `TIMEBOX_NO_G90`
- batch: 64
- updates: 8,081
- semantic row presentations: 517,184 / 1,152,000 = **44.89% ARK exposure**
- actual real tokens: 7,240,576
- M99 confirmed: update 1,400
- G50 confirmed: update 2,200
- G90: **not reached**
- final DEV_CONTROLLER exact-with-EOS: **54.69%**
- final DEV_MEASUREMENT STANDARD exact-with-EOS: **56.47%**
- maximum observed controller exact: **59.38%** at update 5,200
- final two-digit ones accuracy: **80.0%**
- final two-digit tens accuracy: **56.47%**
- locality counterfactual relation consistency: **81.25%**; both members exactly correct: **60.42%**
- carry: 0%
- triple-add: 37.5%
- three-digit extrapolation: 0%

The compact subject therefore did more than memorize: train-probe capability became saturated while held-out performance rose into a stable ~0.55 regime. But this is **partial task generalization**, not sustained G90 and not broad reasoning.

### Important post-run correction: COMMUTED is not a clean invariance test

The raw battery reports `COMMUTED = 100%` and sets `COMMUTATION_INVARIANCE=true`. **Do not accept that label as evidence of commutation invariance.**

ARK-002B's OOD axis places the first operand in unseen tens bands 6/7 while training uses first-operand tens bands 1..5. The COMMUTED probe reverses the operands, which changes that OOD role and often moves the first operand back toward the train-like band. Therefore STANDARD 56.47% versus COMMUTED 100% is better interpreted as **strong operand-role/order asymmetry** than as proven invariance.

The raw receipt is preserved unchanged; this interpretation is an explicit post-run audit correction.

## Production bridge — memorization without structural generalization at full semantic dose

Model: real Cymek V5, same 4L/128w geometry, frozen 24,576-token production tokenizer, 4,130,688 parameters.

- status: `MAX_UPDATES_NO_G90`
- batch: 64
- updates: **18,000**
- semantic row presentations: **1,152,000 / 1,152,000 = 100% ARK reference exposure**
- actual real tokens: 9,216,000
- M99 confirmed: update 2,200
- G50: never reached
- G90: never reached
- DEV_CONTROLLER exact-with-EOS: **0% at every recorded evaluation**
- final DEV_MEASUREMENT STANDARD: **0%**
- SEALED_RESERVED: **0/48 = 0%**
- standard tens-digit exact: **0%**
- standard ones-digit exact: **12.94%**
- COMMUTED: 2.35%
- locality relation consistency: 0%
- carry: 0%
- triple-add: 4.17%
- three-digit: 0%
- verbal rendering: 6.25%

This is the clearest V11 result: the production bridge reached essentially perfect train-probe behavior but showed **no candidate-free held-out arithmetic generalization across the complete ARK-002B semantic exposure box**.

Within V11, architecture geometry, data and Cymek's objective were held fixed while the compact vs production bridge changed representation/tokenization/vocabulary size and consequently embedding/output-space parameter burden. The result therefore strongly motivates investigating that representation burden. It does **not** isolate whether the causal factor is vocabulary size, BPE segmentation, number-token atomization, tied-embedding/output competition, or another correlated representation effect.

## Strongest scientific interpretation

1. **Capability formation, not retention, remains the immediate Cymek bottleneck.** There was no production G90 state to protect.
2. **Semantic dose alone is not sufficient under the current production representation.** Production received the full ARK-002B row-exposure reference and remained at 0% held-out exact.
3. **Compact representation changes the learning regime dramatically.** At less than half the reference exposure, compact reached sustained G50 and ~56% held-out exact while production remained at 0% even after full exposure.
4. **The compact bridge itself is underresolved.** It was capped at 25 minutes and stopped at update 8,081—just before the ~9k–18k delayed-generalization region seen in ARK-002B. The complete V11 session used only 139.73 of the 175-minute hard wall, so the experiment left enough wall budget that could have been used to extend compact exposure. This is a V11 design weakness and should not be repeated.
5. **Do not call the structural flags a reasoning score.** Locality is interesting but partial; COMMUTED is confounded as described above; carry and length extrapolation were absent.

## Highest-information next experiment

Do **not** rerun the full V11 campaign. Production already completed the intended semantic box.

The cheapest decisive closure is to resume/recreate only the compact bridge and drive it to the full **1,152,000 row presentations / 18,000 batch-64 updates**, with corrected structural probes. This should answer whether real Cymek V5 can reproduce the delayed ARK-like generalization regime under compact representation.

If compact reaches G90 at full exposure while production remains at its already-observed 0%, the representation/tokenization burden becomes the dominant next causal target. If compact also fails at full exposure, attention should shift toward Cymek-vs-Arkenstone objective/optimization/initialization/architecture differences rather than merely vocabulary size.

A subsequent representation-factorial should separately manipulate compact-vs-production tokenization/vocabulary/output-space burden rather than changing several factors at once.

## Authorization state

- broad reasoning / AGI claim: **NO**
- production recipe promotion: **NO**
- PRE500M: **NOT AUTHORIZED**
- TPU evidence: **NONE from V11**
- 500M campaign: **NOT AUTHORIZED**
- future 5B corpus: **untouched**
