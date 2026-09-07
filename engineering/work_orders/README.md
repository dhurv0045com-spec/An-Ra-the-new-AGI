# Substantial agent work orders

Each packet targets roughly 3–5 hours of useful engineering once prerequisites exist. Estimates exclude waiting on other agents and unallocated accelerator campaigns. Finish criteria, not time. The chief assigns actual execution resources and accepts results.

| ID | Assignment | Required predecessors |
|---|---|---|
| [W01](W01_CONTRACTS.md) | Canonical contracts and content identities | Documented specifications |
| [W11](W11_EXPERIENCE.md) | Immutable experience and reproducible replay | W01 interfaces |
| [W02](W02_ENVIRONMENTS.md) | Three discovery/planning environment families and qualification | W01 interface; fixtures may precede integration |
| [W03](W03_STATE_MODELS.md) | Shared learner, state and full-history controls | W01; W02 fixtures |
| [W04](W04_INQUIRY.md) | Multi-step and outcome-trained inquiry | W01–W03; D02 exact diagnostic first |
| [W05](W05_WORLD_MODEL_PLANNING.md) | Learned dynamics and bounded planning | W02–W03 |
| [W06](W06_CONSOLIDATION.md) | Replay, acquisition and retention | W01–W03, W11; W04 for learned-selection arm |
| [W07](W07_RUNTIME.md) | CPU/TPU execution, profiling and full restore | W01, frozen model adapter |
| [W08](W08_EXAMINER.md) | Independent evaluation and promotion gates | W01, environment public API |
| [W09](W09_LANGUAGE_DATA.md) | From-scratch language/code data and interfaces | W01; W03 for integrated model run |
| [W10](W10_RESEARCH_LOOP.md) | Integrated experiment/learning loop | Accepted W01–W08 and W11; W09 for language extension |

Every agent reads [execution rules](../EXECUTION_PLAN.md), [data contracts](../DATA_CONTRACTS.md), the relevant [algorithms](../LEARNING_ALGORITHMS.md), and its packet. Report using [HANDOFF.md](../templates/HANDOFF.md). No packet is accepted because it contains many lines or tests.

Owned namespace paths abbreviated as `research/...` in packets are relative to `bramastra_lab/`, as defined in the execution plan. They do not authorize creating a separate repository-root `research/` tree.
