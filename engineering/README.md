# BRAMASTRA engineering headquarters

**Active assignment:** [Complete learner execution program, M00–M18](master_program_20260913/README.md). Read its architecture, algorithms, data/training contracts, execution packages and evaluation rules. M00 closes [B2.2 integration correctness](phase_b22_20260913/README.md) and the [chief review](reports/B2_CHIEF_20260913/REVIEW.md); the remaining packages build the connected learner. Use the [master agent prompt](master_program_20260913/AGENT_PROMPT.md).

The complete [BRAMASTRA research paper](../BRAMASTRA_PAPER.md) presents the architecture, formal learning objectives, preliminary evidence and deferred experimental program in manuscript form.

This directory makes the owner's AGI research program executable by independent agents. The chief designs the system and judges evidence; agents implement and test bounded packages. The [AGI blueprint](../AGI_BLUEPRINT.md) describes the broader research hypothesis. These engineering specifications define the active BRAMASTRA implementation program and supersede its older branch-integration recommendations.

## Read in this order

1. [Current state and next dispatch](STATUS.md): what exists, what is provisional, what is ready to assign.
2. [Complete learner program](master_program_20260913/README.md): current execution authority; read every linked design file before shared-interface edits.
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

The active program contains nineteen packages, M00–M18, with a dependency manifest, file ownership and acceptance criteria. The earlier eleven W packages are historical design assignments. Dispatch from the master program and current status, not from the historical queue.
