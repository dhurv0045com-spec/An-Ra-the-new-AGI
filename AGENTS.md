# BRAMASTRA agent entry point

## Mission and authority

Latest chief dispatch: [K8 review of 2a1e35a, repairs R01–R08](engineering/reports/K8_CHIEF_20260914/REVIEW.md). Current notebook launch is not accepted; repair implementation and return the specified handoff before owner GPU execution.

The owner's objective is general intelligence trained from scratch, including learning to investigate, acquire transferable skills and improve while retaining earlier capabilities. This repository is a research program; neither an architecture proposal nor a passing test suite establishes AGI.

The chief engineer owns architecture, algorithms, data contracts, experiment design, integration decisions and claim acceptance. Implementation agents own their assigned code, focused tests and experimental execution. Follow the owner's current request if it changes a role or scope. Do not substitute a long implementation by the chief for delegation when the owner has requested engineering leadership.

Start at [engineering/README.md](engineering/README.md), then [engineering/STATUS.md](engineering/STATUS.md). Read the assigned work order, its required specifications, and relevant local instructions before editing. The work order is the assignment; this conversation is not required context.

The active milestone is **K8: build the owner's Kaggle experiment notebook**. Read [experiment.md](experiment.md), the [readiness work order](engineering/experiments/K8_20260913/READINESS.md), [campaign manifest](engineering/experiments/K8_20260913/campaign.json), and [current agent prompt](engineering/experiments/K8_20260913/AGENT_PROMPT.md). The owner newly authorizes one future owner-launched two-T4 campaign, at most480 elapsed minutes including all phases/retries. This overrides older no-accelerator instructions only for that named campaign; local optimizer updates remain unallocated and the old206-update CPU ledger stays unchanged. The [master](engineering/master_program_20260913/README.md) and [cognition/RSI](engineering/cognition_rsi_20260913/README.md) packets remain the architecture reference. Latest inspected implementation is edee727; real CUDA/multi-objective/executive/proposer integration must precede the campaign.

## Working boundaries

K8 agents must also read the [chief's direct feedback](engineering/experiments/K8_20260913/CHIEF_TO_AGENT.md), [system quality explanations](engineering/experiments/K8_20260913/SYSTEM_QUALITY.md) and [review protocol](engineering/experiments/K8_20260913/REVIEW_PROTOCOL.md). Their evidence requirements extend I01–I06 without enlarging the compute allocation.

- Work on BRAMASTRA. Historical branches are not prerequisites or authorities for new design. Do not change `.codex-worktrees`, nested repositories, unrelated source trees or user changes.
- Use isolated worktrees for concurrently assigned work where available. If sharing a directory, enforce work-order file ownership and stop conflicting edits until the chief resolves ownership.
- Never overwrite an existing run directory or evidence snapshot. Use a new run ID for new code, data or settings.
- Keep model weights, optimizer files and large datasets out of Git; commit manifests, compact outcomes and source identities. Preserve failed-run evidence.
- All learned core weights start randomly initialized unless the owner explicitly changes that constraint. Declare tokenizer, symbolic teacher, retrieval and external-tool priors separately.
- Prefer bounded CPU diagnostics before accelerator runs. The owner's approximately 100 TPU-hours/week is a planning envelope, not a verified quota. Confirm live hardware and remaining allowance in the run manifest.
- Do not launch paid compute or an unattended long run merely because a packet describes it. Observe the active session's actual authorization and resource budget.
- Do not message third parties. Publishing code follows the owner's explicit Git authorization; do not force-push or silently include others' work.

## Delegation and completion

Work packages are intended to sustain roughly 3–5 hours of useful engineering when their prerequisites are available. Time is an estimate, not a requirement to pad work or keep compute busy. Complete the acceptance criteria, or deliver a precise partial result and blocker. Never report elapsed time that was not measured.

When delegating, specify packet ID, prerequisite artifacts, owned paths, excluded paths, resource allowance and evidence location. Use the owner's preferred inexpensive models, such as Sol or Luna, for bounded execution/review. Do not create agents with vague instructions to “build AGI.”

Before reporting completion:

1. Check the implemented behavior against every acceptance criterion.
2. Run focused correctness tests and the assigned bounded experiment, if authorized and runnable.
3. Distinguish implemented, locally tested, accelerator-tested, experimentally supported and untested claims.
4. Produce the [handoff report](engineering/templates/HANDOFF.md) with exact commands, identities, results, limitations and next step.
5. Update only the assigned evidence and work-order status; the chief accepts integration and broader claims.

Code volume, test count, lower training loss, an interesting example and a favorable selected seed are not intelligence evidence. Negative results are valid deliverables when the experiment is correct and informative.
