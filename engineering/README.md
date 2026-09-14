# BRAMASTRA engineering headquarters

**Current consolidated dispatch:** [Integrated readiness U01–U10](integrated_readiness_20260914/README.md) supersedes older dispatch notices below. Review baseline: 5020533 plus the hashed in-progress foundation changes; no newer remote implementation commit was present. Seven cross-component checks fail despite 26 focused test passes. Complete solvable task/label contracts, trained decision consumers, bounded episodes, real A/B controls, E0 accounting and measured proposer/successor integration. Experiment launch remains blocked; all existing compute limits remain unchanged.

**Current dispatch after ecc5953:** [Cognition foundation F1–F6](cognition_foundation_20260914/README.md) supersedes older dispatch notices below. Six CPU semantic counterexamples were reproduced despite five existing cognition tests passing. Repair evidence encoding, temporal conflict handling, observation-conditioned prediction, real two-step search and selected-action metrics. Cognition is not yet code-ready; preserve existing K8 limits and unresolved O01–O10 acceptance gates.

**Current assignment supersedes older dispatch notices:** [Operational cognitive learner O01–O10](next_phase_20260914/README.md), based on [44e3905 assessment](next_phase_20260914/ASSESSMENT.md). Implement complete session authority, calibration, live cognition and measured RSI; use the packet's explicit code-ready/GPU-qualified criteria. The K8 experiment budget and original treatments remain unchanged.

**Latest assignment:** [real executor contracts](reports/K8_SEMANTIC_REVIEW_20260914/EXECUTOR_CONTRACTS.md), following [semantic review of 5c0567f](reports/K8_SEMANTIC_REVIEW_20260914/REVIEW.md). The runner now refuses known incomplete implementations before E0 allocation. Build real checkpoint/target/evaluation/RSI paths and deliver K8_REAL_EXECUTION_20260914 before chief acceptance.

**Current dispatch:** [D1–D5 concrete implementation order](reports/K8_V2_CHIEF_20260914/EXECUTION.md), following [review of 30d0fcd](reports/K8_V2_CHIEF_20260914/REVIEW.md). Complete actual E1–E5 handlers and correct integration before owner GPU launch. This supersedes earlier review dispatch notices.

**Current dispatch supersedes older review notices:** [second K8 review of 3914dc4](reports/K8_SECOND_REVIEW_20260914/REVIEW.md). Complete remaining R01–R08 and submit K8_REPAIR_V2_20260914. The owner notebook is still not launch-ready.

**Latest dispatch (2026-09-14):** repair [R01–R08 from the chief's review of 2a1e35a](reports/K8_CHIEF_20260914/REVIEW.md). The current notebook is not launch-ready. The K8 design remains the target; fix CPU-detectable integration and accounting defects before owner GPU execution.

**Required execution quality extension:** the [chief's direct feedback](experiments/K8_20260913/CHIEF_TO_AGENT.md), [ten system themes](experiments/K8_20260913/SYSTEM_QUALITY.md) and [review protocol](experiments/K8_20260913/REVIEW_PROTOCOL.md) are part of K8 acceptance. Start with a complete vertical slice, then generalize; include production-path evidence and deliberately broken cases in the handoff.

**Current dispatch supersedes the earlier queue below:** implement [K8 readiness I01–I06](experiments/K8_20260913/READINESS.md) for [experiment.md](../experiment.md), then hand the notebook to the owner. The new authorization covers one owner-launched two-T4 session of at most480 minutes; no local training or extra sessions. Use the [K8 prompt](experiments/K8_20260913/AGENT_PROMPT.md). M00–M24 remains architectural context.

**Active assignment:** [Complete learner program](master_program_20260913/README.md) plus [cognition and recursive improvement](cognition_rsi_20260913/README.md), **M00–M24**. Read both packets' architectures, algorithms, data/training contracts, packages and evaluation rules. M00 includes [F1–F6 from the latest chief review](reports/B2_2_CHIEF_20260913/REVIEW.md) of `acfa249`. Use the [current agent prompt](cognition_rsi_20260913/AGENT_PROMPT.md). No new learned execution is authorized with the over-cap ledger.

The complete [BRAMASTRA research paper](../BRAMASTRA_PAPER.md) presents the architecture, formal learning objectives, preliminary evidence and deferred experimental program in manuscript form.

This directory makes the owner's AGI research program executable by independent agents. The chief designs the system and judges evidence; agents implement and test bounded packages. The [AGI blueprint](../AGI_BLUEPRINT.md) describes the broader research hypothesis. These engineering specifications define the active BRAMASTRA implementation program and supersede its older branch-integration recommendations.

## Read in this order

1. [Current state and next dispatch](STATUS.md): what exists, what is provisional, what is ready to assign.
2. [Complete learner program](master_program_20260913/README.md) and [cognition/RSI extension](cognition_rsi_20260913/README.md): combined execution authority; read every linked design file before shared-interface edits.
3. [Charter](CHARTER.md): objective, roles, claim standards and resource constraints. The following original specifications are background where the master supersedes them.
4. [System architecture](SYSTEM_ARCHITECTURE.md): historical modules, information flow and dependency boundaries.
5. [Data contracts](DATA_CONTRACTS.md): original identities, tensors, state and storage semantics.
6. [Learning algorithms](LEARNING_ALGORITHMS.md): original choices, alternatives and falsifying experiments.
   The [public evidence encoding specification](OBSERVATION_ENCODING.md) defines the shared model interface across environment families.
7. [Experiment registry](EXPERIMENT_REGISTRY.md): historical hypotheses, comparisons, budgets and decision rules.
8. [Original execution plan](EXECUTION_PLAN.md) and [work orders](work_orders/README.md): historical context; dispatch from the master program.

## Authority and evidence

| Material | Meaning |
|---|---|
| `engineering/` specifications | Proposed/current engineering decisions; status is explicit |
| `engineering/work_orders/` | Bounded implementation and experimental assignments |
| `engineering/DECISIONS.md` | Why a decision was adopted, revised or rejected |
| `bramastra_lab/discovery/` | Initial executable active-rule-learning prototype; not a production AGI |
| `bramastra_lab/research/` | Existing canonical implementation; repairs and missing learning connections assigned in M00–M18 |
| `artifacts/bramastra/` | Immutable run evidence, including negative outcomes and source snapshots |
| `engineering/reports/` | Agent handoffs and chief integration reviews |

Do not infer implementation from a proposed path. A module exists only when its implementation and verification evidence are linked in the status record.

## Immediate direction

Build and measure a learner that chooses informative actions in unfamiliar worlds, uses recurrent state and predictive planning, and consolidates what it learns. Demonstrate transfer and retention before interpreting the loop as useful self-improvement. Then expand toward language, code, broader reasoning and long-horizon tasks.

The active program contains twenty-five packages, M00–M24, across the master and cognition/RSI extension manifests, with dependencies, file ownership and acceptance criteria. The earlier eleven W packages are historical design assignments. Dispatch from the combined program and current status, not from the historical queue.
