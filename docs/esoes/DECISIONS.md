# ESOES Decision Register

Ground Blueprint: v0.3
Iteration: evidence-building after 4/4 design attacks
Date: 2026-08-29
Commit: recorded by the commit containing this file

A later agent may not silently change a decision. It must add a reopening entry stating new evidence and the affected blueprint version.

## D-001 — Clean-sheet branch boundary

**DECISION:** ESOES contains its own research and future specifications. V4/VNext/EXP implementations remain evidence in their original branches and are not dependencies.
**STATUS:** [FROZEN]
**WHY:** Git ancestry is not architectural justification; inherited code would bias every future choice.
**EVIDENCE:** `core-vnext@054619f`, `core-exp@51124de`, and the branch inventory performed in Iteration 4.
**ALTERNATIVES CONSIDERED:** retain and gradually refactor VNext; create a new repository.
**WHAT WOULD CHANGE OUR MIND:** only a repository-level constraint that makes a clean branch operationally impossible.
**ITERATION:** 4.

## D-002 — Operational cognition contract

**DECISION:** Core cognition is measured as represent → address → transform → choose → realize under controlled OOD conditions.
**STATUS:** [FROZEN]
**WHY:** these stages distinguish stored signal, query use, computation, selection, and emission.
**EVIDENCE:** V4/PGE, SFT6/SFT7, normalization, and variable-binding research.
**ALTERNATIVES CONSIDERED:** single exact-match “reasoning” score; task-category checklist.
**WHAT WOULD CHANGE OUR MIND:** a more causally identifiable decomposition that predicts intervention outcomes better.
**ITERATION:** 1.

## D-003 — System ownership boundary

**DECISION:** Core owns local reusable neural operations; Connector owns retrieval, durable memory, long search/planning, routing, and intervention; runtime owns tools; evaluator owns success; Outer owns permissions and risk.
**STATUS:** [FROZEN]
**WHY:** environment-changing and authority-bearing behavior must remain auditable; repeated local cognition should not require permanent patches.
**EVIDENCE:** EXP intervention results and invalid label-leaking prototypes.
**ALTERNATIVES CONSIDERED:** neural-everything Core; Connector as only intelligence; Core/Connector only without Outer/evaluator separation.
**WHAT WOULD CHANGE OUR MIND:** replicated evidence that a currently external operation becomes broadly transferable, cheaper, and safer when trained natively.
**ITERATION:** 1/4.

## D-004 — Baseline architecture family

**DECISION:** the first V5 baseline is a dense causal Transformer without MoE, recurrence, neural memory, SSM blocks, latent-thought modules, or dormant cognition heads.
**STATUS:** [FROZEN] for V5-A baseline
**WHY:** a conventional baseline can learn variable binding, fits the 4k target, and makes data/objective causality interpretable.
**EVIDENCE:** public dense-model results, variable-binding mechanism evidence, and insufficient An-Ra evidence for alternatives.
**ALTERNATIVES CONSIDERED:** Mamba, Griffin, Titans, explicit memory, MoE, BLT.
**WHAT WOULD CHANGE OUR MIND:** E2 shows a measured dense bottleneck that a bounded alternative fixes at equal compute, or future context requirements exceed the dense cost envelope.
**ITERATION:** 2/4.

## D-005 — Main scale center

**DECISION:** center the first serious candidate at 250.22M parameters and 5B tokens; do not launch a billion-scale run.
**STATUS:** [PROVISIONAL]
**WHY:** the user selected a 250M implementation envelope; 5B preserves roughly 20 tokens/parameter. This remains provisional because V4 did not establish a capacity requirement.
**EVIDENCE:** PGE provenance plus Chinchilla/DataComp-LM.
**ALTERNATIVES CONSIDERED:** 195M, 300M, 600M, 1B, 3B.
**WHAT WOULD CHANGE OUR MIND:** M102 capacity curves, corpus audit, or a compute/data budget change.
**ITERATION:** 2/4.

## D-006 — V5-A shape

**DECISION:** use 26 layers × width 896, 14 Q heads, 7 KV heads, head dimension 64, and SwiGLU FFN 2368 as the center candidate.
**STATUS:** [EXPERIMENT REQUIRED: E2]
**WHY:** yields an executable 250,216,960-parameter receipt including affine QK-norm scales, adds eight layers over V4, and reduces GQA compression to two Q heads per KV head.
**EVIDENCE:** directional depth evidence only.
**ALTERNATIVES CONSIDERED:** deep/narrow and wide/shallow iso-parameter controls, MHA, 2-KV GQA.
**WHAT WOULD CHANGE OUR MIND:** E2 worst-family OOD and throughput results.
**ITERATION:** 2.

## D-007 — Attention and position

**DECISION:** center on full causal attention at native 4,096 context with RoPE; no extrapolation claim. QK norm and GQA are challengers, not frozen.
**STATUS:** [EXPERIMENT REQUIRED: E2]
**WHY:** full attention removes locality as a binding confound at manageable length.
**EVIDENCE:** V4 hybrid-attention failures are suggestive, while long-context literature warns nominal windows are not usable windows.
**ALTERNATIVES CONSIDERED:** 2k full, 4k hybrid/sliding, recurrence.
**WHAT WOULD CHANGE OUR MIND:** measured 4k cost overwhelms cognitive benefit or hybrid wins at matched FLOPs.
**ITERATION:** 2/4.

## D-008 — Tokenizer family

**DECISION:** identity-preserving byte-fallback subword tokenizer; 24,576 is the center vocabulary. No task/difficulty/answer-index tokens.
**STATUS:** [EXPERIMENT REQUIRED: E1]
**WHY:** exact bytes and no unknowns are mandatory; vocabulary size affects embeddings, sequence cost, numbers, identifiers, and copying.
**EVIDENCE:** tokenizer consistency, numeracy, and beyond-compression research.
**ALTERNATIVES CONSIDERED:** 16k, 32k, byte-level/BLT, number-specific tokens.
**WHAT WOULD CHANGE OUR MIND:** E1 byte-normalized loss, fragmentation, copy, number, code, and cognition results.
**ITERATION:** 2/4.

## D-009 — Data provenance and mixture

**DECISION:** require a source/license/filter/dedup/token manifest; center mixture at 65% high-quality natural, 20% code/math/formal, 15% verified cognition.
**STATUS:** provenance [FROZEN]; ratios [EXPERIMENT REQUIRED: E3]
**WHY:** data quality and causal balance are more likely current bottlenecks than parameter count.
**EVIDENCE:** DataComp-LM, DoReMi, V4/SFT evidence.
**ALTERNATIVES CONSIDERED:** generic-only; 5%, 15%, or 30% cognition.
**WHAT WOULD CHANGE OUR MIND:** E3 transfer and substrate-retention results.
**ITERATION:** 3/4.

## D-010 — Synthetic-data contract

**DECISION:** prefer executable generators with latent causal records; cap unverified LLM paraphrase at 5% total; enforce generator/template/vocabulary/topology/domain-disjoint evaluation.
**STATUS:** [FROZEN]
**WHY:** synthetic control is useful only when semantics are verified and shortcuts excluded.
**EVIDENCE:** Phi-style quality results, model-collapse risk, and An-Ra fixture failures.
**ALTERNATIVES CONSIDERED:** unrestricted model-generated reasoning; template-only train/test splits.
**WHAT WOULD CHANGE OUR MIND:** a stronger verifier/provenance mechanism, never surface quality alone.
**ITERATION:** 3/4.

## D-011 — Training objective

**DECISION:** causal next-token CE is universal; true query-swap contrast is the only candidate auxiliary objective. Same-query candidate margin is not the primary selection objective.
**STATUS:** CE [FROZEN]; auxiliary [EXPERIMENT REQUIRED: E3]
**WHY:** minimizes complexity while testing the strongest causal clue from SFT6/normalization.
**EVIDENCE:** SFT6 large query lift; SFT7's intended selection hypothesis failed despite smaller lift/realization effects.
**ALTERNATIVES CONSIDERED:** CE only, same-query margin, representation losses, separate heads, multiple objectives.
**WHAT WOULD CHANGE OUR MIND:** E3 shows no fresh transfer, calibration damage, or substrate regression.
**ITERATION:** 3/4.

## D-012 — Curriculum and sequence lengths

**DECISION:** compare uniform interleaving with one competence-staged schedule using ≥30% replay; center length mix at 50% 512–1k, 30% 2k, 20% 4k tokens.
**STATUS:** [EXPERIMENT REQUIRED: E4]
**WHY:** curriculum evidence is mixed; length mixing avoids paying full-attention cost on every token.
**EVIDENCE:** Skill-It, curriculum null results, and V4 forgetting.
**ALTERNATIVES CONSIDERED:** uniform only, all-4k packing, online adaptive curriculum.
**WHAT WOULD CHANGE OUR MIND:** E4 worst-family retention and compute-normalized results.
**ITERATION:** 3/4.

## D-013 — Optimization center

**DECISION:** AdamW 0.9/0.95, weight decay 0.1 on matrices, BF16/FP32 state, clip 1.0, WSD; center at 131,072 tokens/update and LR 3e-4.
**STATUS:** family [PROVISIONAL]; exact batch/LR/schedule [EXPERIMENT REQUIRED: E4]
**WHY:** stable baseline; 131k gives ~38.1k updates over 5B tokens versus ~19.1k at 262k.
**EVIDENCE:** open training reports and optimization arithmetic, not An-Ra-specific proof.
**ALTERNATIVES CONSIDERED:** 262k/524k batch, cosine, 2e-4/4e-4.
**WHAT WOULD CHANGE OUR MIND:** E4 stability, throughput, gradient-noise, and cognition curves.
**ITERATION:** 3/4.

## D-014 — Checkpoint and promotion contract

**DECISION:** full-resume rotating recovery every 10M tokens, immutable milestones every 100M (50M in final 500M), asynchronous behavior evaluation, and no automatic final promotion.
**STATUS:** integrity/promotion [FROZEN]; cadence [PROVISIONAL]
**WHY:** V4's best behavior preceded final and cloud outputs were fragile.
**EVIDENCE:** PGE audit and checkpoint incidents.
**ALTERNATIVES CONSIDERED:** final-only, weights-only, loss-only promotion.
**WHAT WOULD CHANGE OUR MIND:** measured checkpoint cost may change cadence, not integrity or promotion rules.
**ITERATION:** 3/4.

## D-015 — Evaluation and promotion

**DECISION:** report representation/addressing/transform/selection/realization separately; use worst-family raw-OOD gates, natural transfer, confidence intervals, fresh replication, and matched retrieval controls.
**STATUS:** [FROZEN] contract; numerical thresholds [EXPERIMENT REQUIRED: E0]
**WHY:** averages and assisted decoding hide catastrophic primitive failures.
**EVIDENCE:** PGE, SFT, normalization, evaluator bugs, and public counterfactual/OOD research.
**ALTERNATIVES CONSIDERED:** aggregate benchmark score, LM-loss gate, assisted best-of-policy score.
**WHAT WOULD CHANGE OUR MIND:** better preregistered causal metrics, not a higher benchmark score alone.
**ITERATION:** 1/4.

## D-016 — Experiment order and implementation gate

**DECISION:** E0 → E1/E2 → E3 → E4 → E5 → freeze review. Production implementation and main training remain prohibited.
**STATUS:** [FROZEN]
**WHY:** benchmark validity must precede model comparison; scale-transfer must precede the expensive run.
**EVIDENCE:** repeated evaluator/infrastructure failures and proxy-transfer research.
**ALTERNATIVES CONSIDERED:** build trainer first; broad factorial sweep; train 250M and inspect afterward.
**WHAT WOULD CHANGE OUR MIND:** a blocker that makes an experiment uninformative; record and reorder explicitly.
**ITERATION:** 4.

## Freeze gate for a future `V5_TRAINING_SPEC_v1.0.md`

All must be true:

- E0 suite/generator certification passes and sealed hashes exist.
- E1 tokenizer winner and artifact hash exist.
- E2 shape/attention winner replicates.
- E3 data/objective winner transfers to fresh synthetic and natural analogues.
- E4 optimizer/curriculum decisions are bounded and stable.
- E5 ~102M recipe beats its matched CE/general-data control across seeds.
- exact executable parameter count, memory, throughput, data, and compute receipts exist.
- exact-resume, real-update, remote-upload, and remote-restore canaries pass.
- source/license/contamination review passes.
- abort and checkpoint promotion thresholds are preregistered.

Until then: **READY TO FREEZE = NO**.

## D-017 — EXP evidence boundary correction

**DECISION:** use v7/v8/v9 only as evidence for three-action repair-success routing; exclude v10/v11 pair/composition promotion claims and the 166-VIE bank from V5 justification.
**STATUS:** [FROZEN]
**WHY:** v10/v11 pair applicability reads stale `candidates`; baselines and trainer reproduction are incomplete; old VIE qualification is not one-variable causal evidence.
**EVIDENCE:** direct source and receipt audit of `core-exp@51124de`.
**ALTERNATIVES CONSIDERED:** accept headline receipts; repair history before moving forward.
**WHAT WOULD CHANGE OUR MIND:** a prospective, fixed, preregistered rerun with complete fixed-action controls and reproducible training source.
**ITERATION:** Ground Blueprint v0.2.

## D-018 — E0 development contract

**DECISION:** make causal cases, evaluator-only truth, model views, counterfactual-pair assertions, split namespaces, and measurement separation executable before model comparisons.
**STATUS:** [FROZEN] contract; development implementation passed; sealed promotion [OPEN]
**WHY:** every architecture/data conclusion is invalid if the benchmark leaks or conflates representation, selection, and realization.
**EVIDENCE:** `artifacts/e0/development_certificate.json` plus 15 E0 regression/property tests, an independent surface solver, and explicit chance/position/power calibration.
**ALTERNATIVES CONSIDERED:** static JSON fixtures; reuse training templates; create sealed fixtures in Git.
**WHAT WOULD CHANGE OUR MIND:** a stronger causal representation may extend the schema, but may not weaken hidden-truth isolation or one-variable assertions.
**ITERATION:** Ground Blueprint v0.2.

## D-019 — Pair-action composition is not a native objective candidate

**DECISION:** retain query-swap contrast as the sole auxiliary-objective hypothesis. Teach composition through executable structured examples under CE first; test sparse traces only if E3 exposes a transfer gap.
**STATUS:** [PROVISIONAL / EXPERIMENT REQUIRED: E3]
**WHY:** the only An-Ra pair-action composition evidence is contaminated, while query-conditioned preference and realization separation have cleaner receipts.
**EVIDENCE:** corrected EXP audit, SFT6 replication, and public controlled composition evidence.
**ALTERNATIVES CONSIDERED:** pair-slot loss, multiple routing heads, direct imitation of Connector actions.
**WHAT WOULD CHANGE OUR MIND:** a clean prospective pair-action replication or E3 showing CE/query-swap cannot learn matched multi-hop transformations.
**ITERATION:** Ground Blueprint v0.2.

## D-020 — Implementation center reopened to 250M

**DECISION:** replace the 195.08M/4B center with an exact 250.22M/5B executable contract while retaining E2/E5 authority over whether that scale should be trained.
**STATUS:** [PROVISIONAL / USER-DIRECTED / EXPERIMENT REQUIRED]
**WHY:** the requested 250M target must propagate coherently through dimensions, tokens, FLOPs, memory, and infrastructure rather than remain a rounded label.
**EVIDENCE:** pure parameter receipt in `v5_contracts/model_spec.py`; 5B restores 19.99 tokens/parameter. There is no evidence yet that 250M beats 195M on cognition per compute.
**ALTERNATIVES CONSIDERED:** merely round the old model to 250M; widen to 1024 and reduce depth; retain 195M.
**WHAT WOULD CHANGE OUR MIND:** E2/E5 shows a different ~250M shape or the 195M-scale family dominates at matched compute/data.
**ITERATION:** Ground Blueprint v0.3.

## D-021 — Code-first infrastructure contract

**DECISION:** make model/run/lineage schemas, E0 statistics/custody, E1 artifact audits, CLI contracts, dependency rules, and milestone acceptance executable before production modules.
**STATUS:** [FROZEN] boundary; implementation milestones remain gate-controlled
**WHY:** a directory sketch does not prevent fake updates, identity drift, evaluator leakage, or non-resumable cloud runs.
**EVIDENCE:** `blueprints/IMPLEMENTATION_BLUEPRINT.md`, `v5_contracts/`, `e0_cognition/`, `e1_tokenizer/`, and their tests/certificates.
**ALTERNATIVES CONSIDERED:** implement a trainer immediately; create empty packages; reuse VNext modules.
**WHAT WOULD CHANGE OUR MIND:** stronger contracts may extend the schema but cannot remove fail-closed identity, causal isolation, or durable exact-resume requirements.
**ITERATION:** Ground Blueprint v0.3.
