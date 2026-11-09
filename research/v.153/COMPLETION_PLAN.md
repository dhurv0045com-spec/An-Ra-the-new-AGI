# V.153 — Completion and expansion plan for the existing Elendril research lineage

**Status: RESEARCH / ENGINEERING PREPARATION. Not a completed V.153 training campaign.**
Prepared: 2026-11-09. Label: **V.153**, per owner correction. This is the completion workstream following the V7.1 core; it does not rename or overwrite V7.1, invent a new lineage of trained models, or retroactively modify any sealed result. It should be revisited by Codex as a collaborator, not treated as immutable instructions.

## 1. What we already know

Original V7.1 run: 20c80f66e76e4b589aed2053afbddffc, executable source 6104ffd7134cb4cf4b3f340bd060d24a1e2dd200. Nine four-epoch S3 core jobs ran on the owner's Kaggle T4×2. Original findings are preserved at Experiment Result/V7.1/. This workstream imports their evidence but must never count it as newly run V.153 jobs.

Core confirmation across three seeds: uniform CE 207/336 (61.6%); blocked CE 270/336 (80.4%); blocked intermediate supervision 269/336 (80.1%). The three seeds reuse the same 112 worlds. Blocked CE is the working baseline. Intermediate supervision is not an upgrade: 29.6% more updates, 44.2% more supervised tokens, major memory-correction regression, no mean confirmation benefit.

Recorded limitations that V.153 must address: changed row order 270/336 to 139/336 for blocked CE; aliases 78/336; distractors 65/336; changed template 42/336; original memory 45/48 versus stale-corrected 28/48 (blocked), and 48/48 versus 2/48 (intermediate). Query-unavailable cases exhibit invalid query attempts. One-hop independent chains already fail (3/48 blocked), so simple multi-hop deficiency is not isolated. Generated hop chains can be solved by copying the last row or max timestamp; a high hop score on those data is not proof of composition. Balanced payload pairs show real but narrow evidence-responsive behavior: blocked 45/48 both-correct pairs, intermediate 47/48.

V7.1 reports 55 independently replayed suite reports and four positive controls from archived analysis data, not independent neural inference. The original FULL ZIP was reported verified by the Kaggle receipt but has NOT been independently supplied/restored. The most serious training and cognitive causes remain unresolved.

## 2. Research direction and decision discipline

V.153 should be a substantial **completion campaign**, not a cosmetically new version or a rush to parameter scaling. Cover positive capabilities and multiple failure families, use quantitative ablations, and retain a direct route to the proposed S8/S20 and continual phases. All phases must be explicitly listed as COMPLETED, NOT_RUN, BLOCKED, NOT_QUALIFIED or ENGINEERING_ERROR rather than implied finished.

Codex is encouraged to challenge these hypotheses, add stronger controls, reduce wasted compute and propose more discriminating variants. Any change to existing preregistered V7.1 source/protocol must create a separate, pre-execution V.153 protocol hash; do not revise the meaning of the completed V7.1 run. Use new sealed holdouts for unbiased final claims and keep development diagnostics separate.

## 3. Immediate P0: evidence custody and release reliability

1. Secure the original V7.1 full checkpoint archive (~1.96 GB; recorded SHA256 1826c85b5a858d2a600661f7e5916f0fa812c55a8d53c37013c695df28a15d19). Independently validate byte SHA, package inventory, checkpoint-component manifests and source/protocol identities. Restore weights without optimizer continuation when doing frozen evaluation. If unavailable, report CUSTODY_PENDING, never fabricate checkpoints.
2. Keep original V7.1 curated Markdown/JSON in Git and raw ZIP/checkpoint custody outside Git per the repository's policy. Record immutable hashes and retrieval paths; do not re-add multi-GB tensors or alter V6/V7/V7.1 measurements.
3. Ensure every notebook is pinned to a concrete source SHA; Run All behavior and selected PHASE are visible. Fail clearly when an archive is missing, source mismatches, GPU admission fails, or final export is unverified. Retain partial receipts and checkpoints.
4. Validate source-pin notebook regeneration, same-device recovery, interrupted-update replay, raw action replay and full ZIP integrity with CPU/CI fixtures before asking the owner for Kaggle execution. Fresh-VM GPU numerical recovery needs separate owner evidence.

## 4. P1: frozen-model diagnosis BEFORE major retraining

**Existing controlled dataset:** anra_v5/elendril/v71_followup.py creates 512 records, 32 cohorts, 48 predeclared one-factor contrasts over 16 route worlds. Vary timestamps, irrelevant rows and serialization, under original two-hop, first-hop, clamped-one-hop and direct lookup questions. The different questions are not the same original prompt and cannot establish an internal intervention. The dataset is currently CPU qualified; its frozen checkpoint inference/execution wrapper is still not built.

**New V.153 anti-shortcut developer suite:** anra_v5/elendril/v153_diagnostics.py creates 24 cohorts × 16 worlds = 384 case records. For each same-world one/two/three/four-hop question, compare original, reversed serialization, shifted timestamps, removed distractors, alternative distractor values at the same length/positions, and removal of a required edge (publicly correct behavior becomes abstain). It deliberately prevents last-visible-row and maximum-timestamp value copying from solving the original cases. Paired variants are not independent world samples. This is a development benchmark: its CPU unit and portable-readback qualification passed GitHub CI (356 pytest tests + 14 subtests, run 42577968675). Use tools.qualify_elendril_v153 with --train pointed to the actual V7.1 training JSON to generate a SHA-bound, replayable portable suite. A model has NOT yet been evaluated on it.

For both suites, propose a frozen GPU harness that verifies/restores all nine V7.1 S3 final checkpoints (if available), evaluates raw neural and separately labeled grammar decoding on all preregistered conditions and writes per-case prompts/actions/pair hashes, no-update proof, GPU cost and an independently replayable small analysis bundle. Use V7.1 core FULL weights for these nine checkpoints; the older original-V7 frozen phase uses the **different** V7 FULL archive/source and should never be silently substituted.

Control the evidence:
- Compare same worlds, targets, underlying facts and equal decision budgets whenever the contrast claims only a surface effect.
- Reverse presentation without changing semantics.
- Shift timestamps without changing relative order.
- Remove irrelevant rows and separately replace irrelevant values at constant length to distinguish clutter length from semantic content.
- Remove one required edge and test safe abstention, not hallucinated final answers.
- Compare correct intermediate cues, equally long sham cues and changed-question clamping; never pass the gold final answer to the model.
- Include public engineered teachers and simple last-row/max-timestamp/first-row/recency/no-query controls, tagged ENGINEERED, not learned.

Only after demonstrating ordinary one-hop and direct lookup under matched graph surfaces should one attribute a two-hop deficit specifically to composition.

## 5. P2: memory, query, surface and optimization diagnosis

A. **Equal-length memory correction factorial.** Same target, same key, same observation count and token positions where practical: correct original; two neutral history rows; stale same-key row followed by a correct later row; swapped history order; memory missing; irrelevant retrieved memory. Explicitly confirm that retrieval delivered both competing rows and that neutral rows survive goal filtering. Separate failures of retrieval, conflict resolution, recency binding, output selection and length/position shifts. Use matched counterfactual pairs and per-seed outcomes.

B. **Action availability and economics.** Cross query-needed/not-needed with channel advertised/not-advertised; memory relevant/irrelevant/unavailable; varied budgets and genuine cost exposure. Distinguish learned actions from legality-constrained grammar; report invalid QUERY/RETRIEVE, needless queries, successful answer updates and cost-adjusted utility. Fixed-policy repricing is not learned cost conditioning. Removed evidence may make a target unknowable; score abstention against publicly available evidence rather than penalizing principled uncertainty.

C. **Surface generalization.** Independent templates, aliases, reordered evidence, large but context-qualified distractor sets and independently authored holdout. Balanced counterfactual target labels across all 16 symbols. Test whether acquired evidence use survives perturbed surfaces. Keep semantic outcomes, symbol balance, token length and requested compute visible.

D. **Optimization diagnosis.** Compare observed loss and full training-set generalization rather than assuming train-perfect/test-brittle. Evaluate intermediate epochs and answer entropy; distinguish collapse, data insufficiency, capacity and decoding. Keep exact source, seed, exposure, optimizer updates, masked tokens and padded attention work.

E. **Focused training repairs (optional and controlled).** At S3, compare unchanged blocked CE against one factor at a time: row-order/template augmentation; correction-focused supervision; balanced distractors; action-availability training; rehearsal; and only if warranted a conventional pointer/copy read head. Include matched-work controls. Avoid declaring improvement merely from a cherry-picked seed or improved forced-decoding score.

## 6. P3: complete the previously planned BIG experiments

**Scale phase is not completed.** Existing research/v7.1 protocol and code define 12 S8/S20 runs: profiles S8, S20 × uniform/blocked CE × seeds 7/19/41, four epochs. Together with the six already run S3 CE baselines they form the proposed 18-run size/curriculum comparison, but S3 data are reference evidence, not new independent seeds. Qualify benchmark integrity and GPU capacity first; a failed cognitive pilot should not automatically suppress scaling **when the preregistered scientific question is whether capacity helps**. Compare equal training examples, supervised tokens, actual padded cost, wall time and seed variance. Report per-task benefits *and* regressions. Do not silently reduce seeds, worlds or epochs to fit Kaggle. If needed split the whole registered phase into source-pinned sessions with identity-linked receipts; register the altered orchestration rather than silently pretending the original phase ran.

**Continual phase is not completed.** Existing nine S3 jobs: B-only, B+replay and fresh-B × seeds 7/19/41, with two B-stage epochs and declared A acquisition gate. Report A-before/A-after and B-before/B-after per seed and confidence interval. An A skill not acquired before transition cannot be forgotten; similarly no B acquisition means retention/transfer is NOT_QUALIFIED. Replay adds A examples and changes work/order/schedule, so add a compute-matched comparator if credible interpretation requires it. A genuinely novel C learning curve is a separate deferred question and must not be silently claimed.

**Frozen historical V7 phase remains optional and unrun.** It needs the V7 FULL archive (not V7.1 weights); if absent report unavailable.

**Additional data scale:** a prospective 1× versus 4× corpus comparator at S3 and selectively S8 if affordable; isolate data scale from epochs and parameter count. This is an amendment, not part of V7.1's previous frozen protocol.

## 7. Phase order and scientific gates

1. Validate custody and CPU benchmark audits; create source-pinned owner-run frozen checkpoint notebook; verify it with fixtures and code-generation identity checks.
2. Run the frozen surface/route + removal ablation campaign on restored S3 weights. Analyse the earlier 512 development records and new 384 anti-shortcut records as **related conditions, not 896 independent worlds**.
3. Implement/run controlled memory and action-availability follow-ups, once their paired construction and public-answer tests pass. If a condition changes several variables, report it as confounded, not causal.
4. Build corrected, shortcut-resistant sealed evaluation with independent authorship where possible. Retain separate development tuning data.
5. Execute S8/S20 scaling and/or S3 mechanism arms in predeclared batches justified by GPU canaries; do not postpone scale indefinitely solely because S3 failed.
6. Run continual A→B retention only when checkpoint custody and acquisition qualifications are satisfied; keep negative/NOT_QUALIFIED evidence even if gate stops transfer claims.
7. Integrate only replicated nonregressing improvements with source and checkpoint lineage, and evaluate all previously successful skills after each graft.

**Safety/validity gates** may prevent dangerous or invalid computation (hardware mismatch, leakage, invalid export, bad checkpoints, unavailable data), but a bad cognition score is a research result and should redirect diagnostic work rather than erase the study.

## 8. Removal / ablation experiments are first-class citizens

The owner's request to include removal is important. The following must be *comparisons with legitimate counterfactual controls*, not destructive source-code deletions: remove distractors; remove a necessary fact; remove or replace memory; remove a correction; remove query access; remove intermediate supervision; remove replay; remove constrained decoding; remove extra source hints. Every ablation should declare what information becomes publicly unavailable, whether its expected answer changes to abstention, and what it can and cannot identify. Preserve all historical code and raw results: DO NOT delete negative experiments.

## 9. Measurements and archival standards

For every seed/phase/condition report non-null correctness, valid abstention, query precision/recall, invalid actions, both-correct evidence swaps, composition by hop and matched world, memory correction survival, surface robustness, training and held-out learning curves, retention and generalization, cost, and work-normalized improvement. Keep action traces with first divergence, expected versus actual public behavior, competing hypotheses, and specific next falsifier. Keep public teachers separate from neural policies.

Every V.153 completed run should get a unique folder inside Experiment Result/V.153 with curated Markdown/JSON evidence, a provenance and status file, links/hashes to the raw analysis/FULL ZIP held outside Git, and methods for replay. Never re-upload multi-GB weights to Git. Report COMPLETED only after mandatory records and both delivered archives verify. Use CAUSE UNRESOLVED whenever mechanism is not established by a controlled replicated intervention.

## 10. Exact definition of 'V.153 complete'

V.153 can be marked scientifically executed only when registered and actually admitted phases have outcome receipts, including explicit NOT_QUALIFIED/DEFERRED status for infeasible or unexecuted work; all mandatory experiments and exclusions have documented reason codes. A complete engineering launch package alone does NOT complete the research. We should separate:
- ENGINEERING_READY (unit/CPU tests, source pins, qualified launcher, export/resume tests);
- GPU_QUALIFIED (real dual-T4 canaries, capacity, weights, checkpoint recovery);
- PHASE_EXECUTED (registered jobs/report coverage and sealed tests, or recorded owner-approved deferral);
- EVIDENCE_VERIFIED (archive integrity, independent behavioral replay, provenance);
- SCIENTIFIC_FINDINGS (including negative results and unresolved causes).

Codex should independently review this plan, correct errors, propose stronger tests, and contribute ideas. The collaboration should preserve scientific creativity; don't blindly implement a giant matrix without reviewing compute and interpretability.

**Current truthful checkpoint:** V7.1 core is done; V.153 research training is NOT_RUN. A first controlled development benchmark and CPU tests have been committed, but live frozen inference and the big phases remain future work.
