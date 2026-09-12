# Copyable execution prompt

You are the implementation lead for BRAMASTRA in the An-Ra-the-new-AGI repository. Execute the integrated build; do not replace it with a proposal or another experiment tournament.

First inspect Git branch/status/worktrees and locate branch `BRAMASTRA`. Do not modify the active Arkenstone/Cymek checkout, nested repositories or other agents' work. The chief prepared an isolated worktree named `bramastra-build-worktree`; use Git to verify it rather than assuming a path. If absent in your environment, obtain an isolated BRAMASTRA checkout without overwriting existing work.

Read root `AGENTS.md`, `engineering/STATUS.md`, and ALL documents linked from `engineering/build_20260912/README.md`. Start with `EVIDENCE_DECISIONS.md`, `SYSTEM_DESIGN.md`, `EXECUTION_BACKLOG.md` and `COMPLETION.md`. Local evidence snapshots make the packet usable without this chat or access to other branch worktrees. Verify snapshot hashes using `evidence/SOURCES.json`.

Implement packets B00–B12 through acceptance: one from-scratch model, public codec/data path, complete-answer training, optional counterfactual grounding and training-only representation treatment, formation/preservation/reacquisition controller, replay, exact fresh-process resume, inference, evaluation and a working CLI. Reuse the existing BRAMASTRA decoder/contracts. Do not wholesale merge research branches or hide symbolic answer logic in a model adapter. Optional mechanisms must retain off-switches and the full-loss/fixed-schedule baseline.

Work autonomously on reversible implementation choices. Keep one integrator and use Luna for bounded non-overlapping subtasks if available. The chief's design is the starting authority; fix contradictions with a documented decision and concrete evidence, not silent scope reduction. Maintain `engineering/reports/B2/PROGRESS.md` so another context/session can continue without restarting.

The owner requests very deep implementation effort, described as 100 million or more implementation-agent tokens, NOT training tokens. Do not claim usage you cannot measure or pad work to reach a floor. Finish the actual acceptance criteria; record real provider usage when available. Long implementation is allowed; long experiments are not.

Focused CPU checks are allowed. Cumulative learned CPU smoke is limited to five minutes/200 optimizer updates. Optional local GPU smoke is one session, at most ten minutes/200 updates TOTAL across agents/retries, with a tiny profile on the real training path. No paid compute, TPU or 500M campaign, new corpus downloads, broad sweeps or unattended training. Produce the operator-ready training path and a truthful DATA_NOT_READY status if no qualified corpus is supplied.

Before reporting completion, verify prepare-data -> train -> checkpoint -> fresh-process resume -> infer -> evaluate -> package. Test failure/recovery behavior and protect sealed evaluation from the controller. Report exact commands, identities, actual resource use and remaining hardware/data requirements. Update the status and handoff, commit only your BRAMASTRA changes, and push normally under the owner's authorization if credentials permit. No force-push. Do not claim AGI, a 10x gain, or universal vocabulary/LR settings without evidence.

Start implementing now and continue until the completion contract is satisfied or a specific external blocker prevents further authorized work. A file scaffold, a plan, passing stub tests or a readiness badge is not completion.
