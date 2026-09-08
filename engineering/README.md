# BRAMASTRA engineering headquarters

The complete [BRAMASTRA research paper](../BRAMASTRA_PAPER.md) presents the architecture, formal learning objectives, preliminary evidence and deferred experimental program in manuscript form.

This directory makes the owner's AGI research program executable by independent agents. The chief designs the system and judges evidence; agents implement and test bounded packages. The [AGI blueprint](../AGI_BLUEPRINT.md) describes the broader research hypothesis. These engineering specifications define the active BRAMASTRA implementation program and supersede its older branch-integration recommendations.

## Read in this order

1. [Current state and next dispatch](STATUS.md): what exists, what is provisional, what is ready to assign.
2. [Charter](CHARTER.md): objective, roles, claim standards and resource constraints.
3. [System architecture](SYSTEM_ARCHITECTURE.md): modules, information flow and dependency boundaries.
4. [Data contracts](DATA_CONTRACTS.md): identities, tensors, state and storage semantics.
5. [Learning algorithms](LEARNING_ALGORITHMS.md): exact initial choices, alternatives and falsifying experiments.
   The [public evidence encoding specification](OBSERVATION_ENCODING.md) defines the shared model interface across environment families.
6. [Experiment registry](EXPERIMENT_REGISTRY.md): hypotheses, comparisons, budgets and decision rules.
7. [Execution plan](EXECUTION_PLAN.md): package dependencies, ownership and dispatch instructions.
8. Your assigned [work order](work_orders/README.md).

## Authority and evidence

| Material | Meaning |
|---|---|
| `engineering/` specifications | Proposed/current engineering decisions; status is explicit |
| `engineering/work_orders/` | Bounded implementation and experimental assignments |
| `engineering/DECISIONS.md` | Why a decision was adopted, revised or rejected |
| `bramastra_lab/discovery/` | Initial executable active-rule-learning prototype; not a production AGI |
| `bramastra_lab/research/` | Proposed canonical integrated system namespace, created by assigned agents |
| `artifacts/bramastra/` | Immutable run evidence, including negative outcomes and source snapshots |
| `engineering/reports/` | Agent handoffs and chief integration reviews |

Do not infer implementation from a proposed path. A module exists only when its implementation and verification evidence are linked in the status record.

## Immediate direction

Build and measure a learner that chooses informative actions in unfamiliar worlds, uses recurrent state and predictive planning, and consolidates what it learns. Demonstrate transfer and retention before interpreting the loop as useful self-improvement. Then expand toward language, code, broader reasoning and long-horizon tasks.

The program contains eleven substantial work packages. They form one shared architecture with controlled interfaces and an independent examiner. W11 separates storage/replay from W01's contract work so both assignments remain bounded.
