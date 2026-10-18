# Overlap Audit

Audit date: 2026-09-25. The audit used only locally available Git refs, branch-observatory reports, and read-only `git show`/`git grep` operations. No other checkout was edited, switched, reset, cleaned, stashed, merged, pushed, or used as a worktree.

## Canonical continuation decision

The canonical owner is `refs/remotes/origin/Gandiva` at `49a3717ae9dc81138f55dc2398c7615f8c80b4b1`. It is the current owner-facing BRAMASTRA/Gandiva integration point, contains the existing `bramastra_lab/discovery` and `bramastra_lab/research` contracts, and is a descendant of `refs/remotes/origin/BRAMASTRA` at `63d60b3523937703183f1e578fe066df08a53acd`. The isolated continuation branch is `codex/anra-discovery-gym`, based on that exact SHA.

The literal names `Discovery Gym`, `DG0`, `DG1`, `DG2`, `DG3`, `DG4`, and `DG5` do not identify an existing project in the inspected refs. The overlap is therefore partial but substantial. Discovery Gym extends the stable contracts and fills the missing causal-program, evidence-firewall, structural-split, and stage-gate boundaries; it does not create a competing `bramastra_lab/discovery`.

## Component table

| Existing component or proposal | Branch and exact ref/commit | What it already implements or specifies | Evidence status | What Discovery Gym adds | Decision |
|---|---|---|---|---|---|
| Boolean discovery instrument | `refs/remotes/origin/Gandiva` `49a3717ae9dc81138f55dc2398c7615f8c80b4b1`; ancestor `refs/remotes/origin/BRAMASTRA` `63d60b3523937703183f1e578fe066df08a53acd`; `bramastra_lab/discovery/*` | Deterministic Boolean worlds, hidden target, query budget, recurrent prediction/query model, random/coverage/uncertainty/no-memory controls, semantic splits, replay/retention pilot | Implemented and CPU-tested in bounded development; D02 result is explicitly inconclusive and not a causal-program transfer result | Replace the single truth-table family with composable causal programs, explicit structural transfer axes, and a hard public/evidence firewall | **Extend and reuse; do not duplicate** |
| Public interactive environments | Same Gandiva/BRAMASTRA refs; `bramastra_lab/research/environments/{base,worlds,oracles}.py` | Public action descriptors, budget/cost accounting, switch/inventory/program fixtures, independent oracle checks, rollout boundary | Implemented and locally verified for bounded fixtures; not a structural causal benchmark | Add a single formal contract for observations, interventions, predictions, rejected actions, target exclusion, and evaluator-owned ground truth | **Reuse the boundary concept; keep a separate DG0 contract** |
| Inquiry teaching | Same Gandiva/BRAMASTRA refs; `bramastra_lab/research/learning/inquiry/{teaching,run_d02}.py` | Training-only one-step and depth-two information-gain teachers, matched data, query-cost accounting, paired evaluation | Executed D02 pilot; review says no reliable multi-step advantage and no autonomous policy claim | Use information gain as a transparent baseline and provenance reference; require prospective commitment and structural transfer before any learned-policy claim | **Reuse as baseline/control, not as evidence of learned discovery** |
| X-factor and causal self-modeling | `refs/remotes/origin/triquetra` `f23f0af42d90847cf1d2c244160c8203d1995b33`; same X-factor tree is present in Gandiva `49a3717ae9dc81138f55dc2398c7615f8c80b4b1`; `x_factor/contracts.py`, `geometry.py`, `evaluation.py` | Observed-only feature firewall, intervention registry, latent intervention geometry, fixed/negative/oracle controls, answer-blind policy boundary | Implemented and locally audited; evidence remains bounded to its substrate and intervention families | Apply the firewall idea to hidden causal-program worlds; do not import its model, labels, or intervention semantics as Discovery Gym truth | **Reuse the leakage-law pattern; new boundary** |
| Arkenstone continual cognition | `refs/remotes/origin/Arkenstone` `4ae9e3b6dc806d6ada7b7b2f05d32271d072d543`; `refs/remotes/origin/arkenstone-ark020-v4` `308a9865af519ccedd1fe13759e3b8ccdca85907`; `refs/remotes/origin/arkenstone-astra` `249e3eb0ac1ebbb461c829fe1541ab307eea5e79` | Continual acquisition/retention portfolio, invariance and intervention experiments, ARK-020 readiness and custody records | Many narrow executed/replicated records; current ARK-020 is explicitly not executed | Keep DG4/DG5 out of DG0; reuse only stage vocabulary, equal-cost references, and retention/evaluator separation | **Reuse protocol concepts only; no merge** |
| Citadel evaluation and data quality | `refs/remotes/origin/citadel` `e96c9a9a07af1d9126930190cbf3d1c764523e2d` | Contamination, shortcut, provenance, negative-result, readiness, and claim-ceiling audits | Strong independent audit authority; its own documents reject unsupported generalization | Implement machine checks for structural duplicate programs, public-key leakage, direct target queries, invalid-cost accounting, and confirmation tuning | **Reuse audit principles; implement local checks** |
| Core world-model/evaluation utilities | `refs/remotes/origin/main` `b620f1c2d084b0db5fa0d24196949f88c92edc1e`; `robotics/world_model.py`, `data/causal_corpus.py`, `engine/eval_harness.py` | General world-model, corpus, and evaluation interfaces | Engineering reference only; not a hidden causal discovery result or task contract | Keep Discovery Gym standard-library and architecture-neutral; do not couple its scientific meaning to the core runtime | **Do not merge; optional future adapter** |
| Evidence consolidation and decision records | `refs/heads/research/evidence-consolidation-2026-09-25` `89b1c5b984128e61d8ad11f5059d11dc99e71817` | Cross-branch provenance, corrections, ledgers, and bounded architecture decision records | Canonical for consolidated evidence decisions, not for this executable environment | Use source/ref/hash discipline and preserve unresolved disagreements in the handoff | **Reuse provenance conventions** |
| Branch observatory | `refs/heads/branch-observatory` `d0c9ecd42a3602d5b76fc46ac0ec1bac66f2a3c6` | Read-only ref/worktree inventory, branch dossiers, document hashes, and local quality checks | PASS for its own inventory; no scientific authority | Use its recommendation that Gandiva is the owner-facing integration point; do not treat generated reports as capability evidence | **Use for audit navigation only** |

## Duplicate decision

There is no exact duplicate. The existing Boolean discovery package is a direct precursor and must remain the canonical historical component. Discovery Gym is a scoped continuation that changes the task boundary from a fixed Boolean truth-table learner to an auditable causal-program environment. Any future implementation that merely copies `bramastra_lab/discovery` under a new name should stop as duplicate.

## What remains owned elsewhere

- Gandiva/BRAMASTRA continues to own the broader from-scratch research architecture, active FINAL-K8 work, and existing Boolean discovery code.
- Citadel continues to own independent data/evidence audit authority.
- Arkenstone continues to own continual-cognition mechanism and retention experiments.
- Triquetra continues to own X-factor evidence and intervention-family interpretation.
- The branch observatory continues to own repository measurement, not Discovery Gym science.

## Boundary rule for future work

A later DG1 implementation may import a stable Gandiva contract only after an explicit compatibility review. It must not import hidden truth, evaluator labels, confirmation worlds, oracle actions, or model checkpoints into a learner input path. Any change to this boundary requires a new overlap audit and protocol revision.
