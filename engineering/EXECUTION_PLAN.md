# Agent execution plan

## Dispatch protocol

Give an agent one work-order ID and a checked-out repository. The agent reads root `AGENTS.md`, this directory's entry point, the work order and its required specifications. No chat history is necessary.

Suggested dispatch message:

> Execute `engineering/work_orders/W02_ENVIRONMENTS.md` on BRAMASTRA. Read root AGENTS.md and the packet's required specifications. Own only its declared paths. Use the packet's CPU budget; do not start TPU/paid runs. Implement, test and submit a handoff under engineering/reports/W02/. Report blockers and negative findings. Do not merge or push other agents' changes.

The chief fills in actual branch/worktree, resource allocation and prerequisite artifact IDs before dispatch. A filename is a pointer to the assignment, not a substitute for checking readiness.

## Dependency graph

```text
W01 contracts -----------------> W03 state/model ------> W04 inquiry learning
          +--> W11 experience/replay ----------------------------+
          |                           |                         |
          +--> W02 environments -------+-----> W05 planning ----+
          |                           |                         |
          +--> W08 examiner ----------+-----> W06 consolidation |
          |                                                     |
          +--> W07 runtime -------------------------------------+
          |                                                     |
          +--> W09 language/data -------------------------------+
                                                                |
                                           W10 integrated research loop
```

W02/W07/W08/W09 may design and implement against the frozen documented contracts while W01 works, but cannot claim integration until W01's actual contract implementation passes compatibility tests. W03 can use a fixture adapter initially. No agent changes a shared contract unilaterally.

## Waves and concurrency

Under constrained agent capacity, use one primary implementation agent and the chief's review. Add a second execution/review agent only for an independent, useful scope. After a shared usage-limit failure, do not repeatedly dispatch the same batch: inspect saved work, check available capacity once, and try one bounded continuation when there is evidence it can run. Mark stopped packages queued or interrupted rather than active. Reuse completed agents for subsequent packages when practical.

**Wave 0:** close the current prototype's correctness review and bounded development runs; publish honest findings. This is underway, not one of the full 3–5-hour packets.

**Wave 1:** W01, W02, W07 and W08; W11 follows the W01 interface freeze. With fewer execution slots, prioritize W01/W02/W08 and leave W07 design-only until the model interface stabilizes. W09 is independently useful when data/provenance access is available.

**Wave 2:** W03 followed by W04; W05 can proceed against a frozen dynamics API. Run D02 early because it can reject a flawed inquiry objective cheaply.

**Wave 3:** W06 with W11 replay plus W09 integration, then W10. W10 requires real component results, not merely empty module implementations.

## Shared file ownership

| Package | Main owned namespace |
|---|---|
| W01 | `research/contracts`, contract tests |
| W11 | `research/experience`, experience tests |
| W02 | `research/environments`, environment tests/generator evidence |
| W03 | `research/models`, model tests |
| W04 | `research/learning/inquiry`, inquiry tests |
| W05 | `research/planning`, planning tests |
| W06 | `research/learning/consolidation`, retention tests |
| W07 | `research/runtime`, runtime tests |
| W08 | `research/evaluation`, evaluation tests |
| W09 | `research/data`, language adapters, data tests |
| W10 | `research/orchestration`, integration tests and campaign CLI |

`research/__init__.py`, package configuration, top-level README/AGENTS and shared CLI registration belong to the integrator. Agents request changes in their handoff rather than racing to edit them. They may create their owned subpackage initializers. Prefer narrow interfaces over importing another agent's unreviewed internals.

## Definition of done for every packet

- Working implementation within the assigned scope; no placeholder pretending to perform learning.
- Tests cover meaningful failure modes and independent controls.
- A bounded executable example and exact reproduction command.
- Source/data identities and raw outcomes for any experiment.
- Written comparison against acceptance criteria, with unresolved items marked explicitly.
- Resource consumption and run status, including interrupted/failed work.
- Handoff describing how the next packet consumes the result.

The chief may accept implementation while rejecting a scientific claim. A failed hypothesis can complete a packet if the experiment is valid and the assigned uncertainty is resolved.

## Estimated effort

Eleven packets at roughly 3–5 engineering hours each imply an initial **33–55 agent-hours**, before integration and any substantive accelerator campaigns. This is a planning estimate, not elapsed work or a promise of completion. Do not instruct agents to stay busy until the clock expires. A packet may require revision if prerequisites, findings or repository conditions differ from its assumptions. Namespace entries above are relative to `bramastra_lab/`.
