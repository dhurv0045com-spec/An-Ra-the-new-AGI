# CYR-GPU-011 — DESIGN REASONING

## Decision

Use one long Colab GPU session to isolate **capability formation** rather than retention. Run two ordered bridges on the real Cymek V5 4L/128w geometry:

1. `COMPACT_BRIDGE`: exact ARK-002B data with the same 19-symbol arithmetic alphabet, but **Cymek's canonical causal objective**.
2. `PRODUCTION_BRIDGE`: the same data, V5 geometry and objective with Cymek's frozen 24,576-token tokenizer.

The comparison is controlled development evidence, not a broad reasoning or production claim.

## Why V9/V10 were insufficient

CYR-GPU-009 established a useful fact: TINY could memorize the train probe but did not reach candidate-free held-out G90 by its 2M-token parent budget. However, the relevant Arkenstone positive reference is ARK-002B, whose 4L/128w compact-vocabulary subject trained at batch 64 for as many as 18,000 updates: **1,152,000 semantic row presentations**.

V9 exposed roughly 15,632 × 16 = 250,112 rows (<22% of that box). The original V10 exposed at most 18,000 × 16 = 288,000 rows (25%). Therefore another V10 null would still leave semantic underdose as a live explanation.

## Live cross-branch evidence audit

The final prefreeze evidence review used these exact read-only heads:

- Arkenstone: `c16718a7841c3cc3eba2b4b2c0388a0e36c0b530`
- BRAMASTRA: `415250f44179f3310e5dc55addb21722290604fa`
- canonical Cymek: `28bf57a0d299a2c13a99fe0046616c00a1b8530c`
- Citadel: `1d27f9b0d770e30577de0a8671c909cb783b4ff1`
- Triquetra: `f23f0af42d90847cf1d2c244160c8203d1995b33`

Arkenstone had advanced eleven commits beyond the earlier V11 audit. The delta adds ARK-018 implementation/preexecution material; its experiment log still marks ARK-017 and ARK-018 as **not executed**, so it supplies no new outcome that changes V11. BRAMASTRA added a small two-seed matched-teaching D02 development comparison; its own chief review reports no reliable multi-step advantage and no accelerator/model-promotion implication. That result is orthogonal to V11's controlled arithmetic acquisition question, so it is recorded but does not change the plan.

Evidence that does control V11 remains:

- ARK-002B: exact commutation-free manifest, 500 train / 197 test. Seed 47 demonstrated sustained G90 around update 12k; seed 29 reached final OOD 0.924 but did not satisfy sustained G90 by 18k. Qualitative memorize-first → delayed-generalize behavior replicated, with large seed variance.
- ARK-003: simple curriculum/teacher suffixes did not reliably accelerate the transition. This argues against spending the session on clever curriculum before reproducing acquisition.
- ARK-011/012: adaptive LR protection is interesting after capability exists, but switch threshold remains unresolved. It is downstream of the present bottleneck.
- ARK-013: LOW LR did not solve prolonged cross-task interference.
- ARK-014/015: learned non-arithmetic invariance can depend strongly on data support; ARK-015 showed 8/8 narrow-HIGH failures versus 0/8 LOW and 0/8 augmented-HIGH. This motivates structural diagnostics but does not justify replacing the acquisition experiment.
- ARK-016: mechanism attribution remained inconclusive due low event count.
- ARK-017 and ARK-018: implemented/preregistered work exists, but neither has an executed scientific result at the audited Arkenstone head; both remain **UNEXECUTED** evidence-wise.

## What the compact bridge controls—and what it does not

The compact bridge intentionally matches more of ARK-002B than V9/V10 did:

- exact frozen ARK-002B examples;
- exact 19-symbol vocabulary and special-token identities;
- 4 layers / width 128;
- LR 1e-3;
- AdamW scalar values 0.9/0.95, eps 1e-8, wd0.1;
- candidate-free generation;
- semantic exposure measured in row presentations.

It is **not an exact ARK model/objective reproduction**. In particular, Arkenstone's helper encoded the answer with a BOS prefix and supervised that BOS, while Cymek's canonical causal objective always excludes BOS targets. V11 deliberately keeps the canonical Cymek objective rather than modifying production semantics to imitate Arkenstone. Cymek also retains its V5 GQA/QK-normalization architecture, initialization, semantic optimizer grouping, production backend and CUDA precision behavior.

Therefore:

- compact G90 shows real Cymek V5 can enter the delayed-generalization regime under the same task/data and compact symbol representation;
- compact no-G90 at near-full exposure shifts attention beyond dose toward objective/architecture/optimization/precision/init differences, but does not identify which one;
- no outcome may be described as an exact replication of ARK-002B.

## Why a production bridge follows

The production bridge changes **representation/tokenization** while keeping Cymek model geometry, causal objective, task and optimizer semantics fixed. It restores the frozen 24,576-token tokenizer and expands the embedding/output parameter burden from 987,392 total parameters to 4,130,688.

Batch size is selected before outcomes from calibration. The target semantic box is always 1,152,000 rows: batch64→18k updates, batch32→36k, batch16→72k. The wall clock may prevent reaching that target; actual exposure is recorded.

A production null is called representation divergence only if it receives **at least the semantic exposure at which the compact bridge reached G90**. Otherwise the verdict is explicitly underexposed.

## Capability and reasoning measurements

Primary capability gate: candidate-free exact answer **with valid EOS stop**. The capability and structural generators use the same fixed eight-token generation cap, avoiding a hidden final-measurement advantage from a longer decode budget.

A G90 claim requires:

- three consecutive `DEV_CONTROLLER` evaluations >=0.90; and
- final larger `DEV_MEASUREMENT/STANDARD` exact-with-EOS >=0.90.

The final controller, final structural battery and post-decision sealed measurement preserve row-level candidate-free predictions plus canonical SHA-256 digests in the returned evidence bundle. Milestone/final checkpoint receipts preserve the model and optimizer byte hashes in addition to the durable Drive path. This makes a returned ZIP auditable without pretending the model tensors themselves are embedded in that ZIP.

Structural diagnostics are measurement-only and never control training:

- STANDARD: frozen held-out no-carry task;
- COMMUTED: operand-order invariance;
- LOCALITY: controlled ±1 counterfactual response relation;
- CARRY: transfer to unseen carry mechanics;
- TRIPLE_ADD: operation composition;
- THREE_DIGIT: length extrapolation;
- VERBAL: unseen natural-language rendering, production tokenizer only.

These remain separate flags. There is no aggregate "reasoning score" and no AGI claim.

## Compute allocation

Hard campaign wall: 175 minutes, including a 5-minute packaging reserve. Cell 0 performs only short environment/calibration/preflight work before the scientific campaign; it is not scientific training evidence.

Compact bridge receives at most 25 minutes and stops early on controller G90 confirmation. The production bridge receives nearly all remaining compute. A second independent production seed launches only when the first subject has both sustained controller G90 and >=0.90 final DEV_MEASUREMENT/STANDARD, with >=40 minutes remaining. This is progressive use of the wall rather than an all-or-nothing feasibility gate.

## What changes our mind

High-value outcomes:

- **Compact G90 + production G90:** Cymek V5 can acquire the task under both representations; structural battery characterizes what transferred beyond the primary holdout.
- **Compact G90 + exposure-matched production null:** the production representation materially increases acquisition burden, though vocabulary size/embedding burden/token segmentation are not individually isolated.
- **Compact G90 + underexposed production null:** no representation conclusion; next action is more efficient production-representation exposure or a targeted representation study.
- **No compact G90 near full ARK exposure:** semantic dose and vocabulary alone are insufficient to reproduce ARK's transition; isolate objective/architecture/optimizer/precision/init differences before retention work.
- **Production G90 without compact G90:** unexpected and valuable; repeat before mechanism claims.

No CYR-GPU-011 result by itself authorizes PRE500M, 500M training, TPU claims, or production recipe changes.
