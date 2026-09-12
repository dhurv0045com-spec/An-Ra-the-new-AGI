# BRAMASTRA integrated build — execution authority

Owner-directed build revision, 12 September 2026. This packet supersedes the September 9 paper-only pause for implementation. It does not authorize a long training campaign. The intended product is one from-scratch learner that can be trained, resumed, evaluated and used through a single interface.

## Start here

Read, in order: root `AGENTS.md`, `engineering/STATUS.md`, this file, [evidence decisions](EVIDENCE_DECISIONS.md), [system design](SYSTEM_DESIGN.md), [execution backlog](EXECUTION_BACKLOG.md), and [completion contract](COMPLETION.md). The [handoff prompt](EXTERNAL_AGENT_PROMPT.md) can start an agent without this conversation. Older specifications remain useful where this packet does not explicitly revise them.

[build_contract.json](build_contract.json) is the machine-readable specification for profiles, feature defaults, resource limits and command names. It is not yet a runnable trainer configuration. B01 must implement its validation and translate profiles into the existing decoder's `ModelConfig` without changing the declared semantics.

**Branch:** `BRAMASTRA`. During preparation, the original shared checkout was on `codex/improve-arkenstone-branch`; BRAMASTRA was checked out separately in `bramastra-build-worktree`. Do not assume the application's initial working directory is the target branch. Resolve it with Git before editing. Never change another active checkout or nested worktree to obtain this branch.

## What the owner is asking you to finish

Deliver operational implementation, not another paper or a disconnected tournament. The model, public event interface, data pipeline, training objective, persistent state, inference, evaluator and command-line entry point must work together. Reuse the accepted BRAMASTRA decoder/contracts; use external branches as audited implementation references, not wholesale dependencies.

The owner requests an unusually deep execution effort, described as at least **100 million implementation-agent tokens**. This is not a training-token target. The originating assistant cannot reserve, enforce or verify another application's model usage. Record actual usage if the execution system exposes it; otherwise write `unavailable`. Do not invent counts, pad reasoning, repeat successful tests or run purposeless experiments to reach a token floor. Work until the completion contract is satisfied or a precise external blocker remains. Provider context limits require durable checkpointed handoffs, not simulated continuity.

## Resource allowance

- Implementation and focused CPU correctness checks are authorized. Use inexpensive Luna agents for bounded non-overlapping execution if your app supports them; the integrator owns design and acceptance.
- At most **one local GPU smoke session totaling 10 minutes and 200 optimizer updates**, using a tiny configuration, is permitted. Include setup/compilation/evaluation in the wall budget; preserve failure evidence. If the device cannot fit, use CPU. Do not launch a second GPU session without a changed allowance.
- CPU learned smoke work: at most 200 optimizer updates and five minutes total. Unit tests of arithmetic and state transitions are separate; do not disguise training sweeps as unit tests.
- No paid compute, TPU campaign, 500M-model training, corpus download or background long run. The approximately 100 weekly TPU-hours remain an unverified future envelope.
- A build may prepare a launch manifest and commands for later owner execution. Readiness must distinguish implementation from live accelerator certification and actual data availability.

## Current committed starting point

`bramastra_lab/model.py` supplies a dense causal decoder and exact parameter-count function. `bramastra_lab/research/contracts/` contains accepted strict records. The discovery and inquiry packages contain small diagnostic experiments. There is no committed integrated research model/runtime/experience/environment stack. Earlier W02 source saved in another checkout was unaccepted and is not present in this branch; do not depend on it or copy its fabricated qualification scores.

This revision supplies the evidence-bound design, configuration contract and implementation backlog. It does **not** falsely label the future integrated system implemented. The execution agent closes that gap and updates the status ledger from actual work.
