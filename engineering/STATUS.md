# Current engineering state

Updated: 2026-09-13. This is the dispatch entry point. Update from actual evidence, not intention.

Owner direction: build and execute the integrated system, with only bounded local smoke work; larger experiments remain for later. The active [B2 execution packet](build_20260912/README.md) incorporates refreshed Arkenstone/Cymek evidence and supersedes the paper-only pause. **B00–B12 implementation is now complete on branch `BRAMASTRA`**: the integrated learner, public codec, data path, trainer, atomic checkpoints with fresh-process resume, plasticity controller, ledger/replay, environments, evaluation, planning and the full CLI are implemented and tested (all new `tests/test_research_*.py` suites pass; the integrated command path prepare→train→resume→infer→evaluate→package was executed end to end on tiny fixtures, including a verified fresh-process resume). Bounded CPU learned smoke consumed 65/200 updates and 66/300 seconds (see `reports/B2/SESSION_LEDGER.json`); no GPU smoke ran (the installed torch is CPU-only; the optional GPU session was not used). No qualified production corpus exists, so real-data readiness is `DATA_NOT_READY` and every optional mechanism remains scientifically unqualified. See [the B2 handoff](reports/B2/HANDOFF.md). [The manuscript](../BRAMASTRA_PAPER.md) remains a research design/preliminary report. No experimental agent is running.

The build was prepared in a separate `BRAMASTRA` worktree because the original checkout is now on an Arkenstone branch. Resolve Git worktrees before editing. The W01–W11 table below records historical packages; B00–B12 is the current integrated execution order.

## Current state

| Area | State | Evidence / next action |
|---|---|---|
| Overall AGI | Not achieved | Research objective; no sufficient recipe or broad validation |
| Chief-engineer organization | Written | Root AGENTS, specifications, eleven work orders and handoff templates |
| Active discovery prototype | Implemented; bounded CPU development runs completed | [Chief review](reports/PROTOTYPE_REVIEW.md) |
| Learned inquiry | Exploratory improvement in two seeds' point estimates; uncertainty/strong-baseline limits remain | D01 results; execute W04 for delayed-information mechanisms |
| Consolidation | Unstable; not accepted as durable improvement | Family regressions; W06 required |
| Canonical integrated architecture | Specified; W01 contracts accepted | W01–W11 dependency plan; environment/model integration remains unfinished |
| **B2 integrated build (B00–B12)** | **Implemented, unit-tested, tiny-fixture smoke passed incl. fresh-process resume; DATA_NOT_READY; not scientifically qualified** | [Handoff](reports/B2/HANDOFF.md); [smoke ledger](reports/B2/SESSION_LEDGER.json) |
| Real TPU runtime | Not validated for this prototype | W07 and live hardware/quota record |
| Autonomous method/code improvement | Not implemented | Later isolated experiment after inquiry/retention evidence |

## Ready assignments

1. **W01:** contracts and identities; W11 follows with storage/replay.
2. **W02:** environment qualification and shared public interface, initially using W01 fixture contracts.
3. **W08:** independent examiner and honest promotion decisions.
4. **W07:** CPU update/restore and runtime design; accelerator execution waits for an actual allocation.

W04's exact delayed-information diagnostic can start independently; its learned integration waits for the model/environment interfaces. W09 can inventory qualified local data in parallel. See [execution plan](EXECUTION_PLAN.md) for the full dependency graph and file ownership.

## Work-order ledger

| Packet | Status | Owner / acceptance |
|---|---|---|
| W01 | Accepted for bounded contract scope; 22 tests independently passed | [Chief acceptance](reports/W01_REVIEW.md) and [handoff](reports/W01/HANDOFF.md) |
| W02 | Stopped after usage interruption; queued, not running | Saved qualification and task reward require [chief corrections](reports/W02_REVIEW.md) |
| W03 | Waiting on W01/W02 fixtures | Unassigned |
| W04 | D02 two-seed CPU comparison completed; no reliable advantage established; 8 inquiry tests passed | [Chief result](reports/D02_REVIEW.md); [next diagnostic](work_orders/D02_DIAGNOSIS.md). Luna stopped on usage limit; PPO remains unrun |
| W05 | Waiting on W02/W03 | Unassigned |
| W06 | Waiting on canonical data/model; prototype evidence available | Unassigned |
| W07 | CPU/design ready; accelerator validation unallocated | Unassigned |
| W08 | Design/fixtures ready; integration waits on W01/W02 | Unassigned |
| W09 | Local inventory/design ready | Unassigned |
| W10 | Waiting on accepted components | Unassigned |
| W11 | Waiting on W01 interfaces | Unassigned |

The bounded agents that closed the prototype are not counted as having completed these new 3–5-hour packages. No 33–55-hour campaign has been executed merely because its work orders exist. Use one Luna executor, with the chief owning experiment design and acceptance. No execution agent is currently running. W01 has passed review; D02's initial comparison is complete, and diagnosis precedes further training. W02 remains queued for its recorded corrections.

## Chief's next acceptance decisions

- Accept strict data/public-information contracts and qualified environments before broadening learning claims.
- Determine whether multi-step inquiry overcomes the exact greedy-teacher limitation on new mechanisms.
- Reject promotion if acquisition gains hide protected-family regression.
- Allocate TPU only after the intended learner's actual update/restore path and an informative experiment are ready.
