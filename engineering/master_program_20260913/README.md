# BRAMASTRA: complete learner execution program

Chief-engineer specification, 2026-09-13. Baseline inspected: `baa655df7989ab86c447c853fc23703c8b894fe7`, containing agent implementation `c3662b3`. **Status: designed and assigned; the extensions below are not implemented or experimentally established by this document.**

The owner's objective is general intelligence learned from scratch: a system that understands unfamiliar tasks, investigates uncertainty, uses knowledge across domains, acquires capabilities, and improves without destroying earlier capabilities. This packet turns that objective into a connected implementation program. It specifies the complete next build, beyond the immediate B2.2 repairs. No known result in this repository establishes a sufficient recipe for AGI. The architecture is a set of testable engineering hypotheses with explicit failure paths.

## The central design decision

Build one learner whose language understanding, public-world prediction, action selection, answer production and value estimation share a randomly initialized decoder. Connect it to bounded memory, an experiment selector, a qualified training-task factory, and an independent evaluator. Every real interaction can become several legitimate training examples; predictions remain predictions. Improvement means a child model earns promotion on new capabilities and protected old capabilities under a declared resource comparison.

The main proposed advantage is **counterfactual inquiry followed by consolidation**: predict how different observations would change the answer, choose an affordable observation, observe the real outcome, learn from the discrepancy, then preserve useful behavior. This is more specific than adding a controller or labeling a text continuation a plan. Whether it beats simpler methods is an empirical question.

## Read and execute

| File | What it decides |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System boundaries, shared computation, model interfaces, memory and deployment |
| [ALGORITHMS.md](ALGORITHMS.md) | Exact losses, gradient normalization, inquiry, branching planning, retention and candidate promotion |
| [DATA_AND_TRAINING.md](DATA_AND_TRAINING.md) | Typed records, split identities, task generation, training stages, corpus and hardware readiness |
| [EXECUTION.md](EXECUTION.md) | M00–M18 dependency graph, owned paths, deliverables and acceptance checks |
| [EVALUATION.md](EVALUATION.md) | Claim ladder, falsifying comparisons, historical-branch comparisons and decision rules |
| [AGENT_PROMPT.md](AGENT_PROMPT.md) | Complete operator prompt for the implementation agent in another app |
| [program.json](program.json) | Machine-readable package dependencies and status; a dispatch manifest, not a model config |

Read root `AGENTS.md`, `engineering/STATUS.md`, this index, then every file above before changing shared interfaces. The [B2 source-evidence decisions](../build_20260912/EVIDENCE_DECISIONS.md) and [chief review](../reports/B2_CHIEF_20260913/REVIEW.md) explain the current constraints. Old W01–W11 and B00–B12 packets are historical context, not competing active queues.

## What the inspected implementation actually lacks

The code already contains useful data, checkpoint, evaluation, collection and controller infrastructure. Preserve it. The additional audit found these missing learning connections:

1. `IntegratedModel` exposes action and value heads, but `Trainer` optimizes answer/EOS loss and optional pair loss. No action targets or returns train those heads.
2. `BoundedRolloutPlanner.plan` ranks roots and emits generated continuation text. Continuation content does not determine a recursive expected-return computation over typed outcomes.
3. `LearnedCollectionPolicy` is explicitly an untrained hook. Its existence is not a learned inquiry result.
4. Collection stores receipts, but no canonical transition-to-objective path trains prediction, decisions and values together.
5. Replay is a training storage mechanism. It is not working memory or retrieval available during inference.
6. Teacher and curriculum ideas have not been joined to the canonical trainer with auditable training-only supervision.

The next build must close these connections. A second set of similarly named modules without end-to-end callers is not acceptable.

## Execution authority and gates

**M00 first:** execute [B2.2 C1–C6](../phase_b22_20260913/README.md). The reproduced integrity and state defects invalidate downstream evidence. Preserve the original report and append corrections.

This expanded assignment replaces the earlier instruction to stop after the B2.2 return handoff. Once M00's acceptance checks pass and its immutable report exists, the implementation agent may proceed through M01–M18 as **provisional integration**, using the explicit dependencies. This does not let the agent claim chief acceptance, approve scientific claims, reset budgets, or promote a learned candidate. If M00 remains broken, work on specifications and isolated pure contracts can continue, but do not present an integrated runnable learner as accepted.

There are two separate gates: an implementation dependency gate (interfaces and focused correctness evidence) and a scientific promotion gate (authorized runs, sealed evidence, chief decision). Most of this program is build work and can finish without a training campaign. Missing licensed corpus or live TPU hardware blocks its dependent execution, not unrelated implementation.

## Resource and effort contract

Preserve `engineering/reports/B2/SESSION_LEDGER.json`. At this specification's baseline it records **193 CPU optimizer updates and 111.234 CPU learned-smoke seconds; zero GPU usage**. The existing cumulative CPU cap is 200 updates/300 seconds. M00 may use at most six remaining real updates for its specified uninterrupted-versus-resumed comparison. The optional existing local GPU allowance is one session, at most 200 updates/600 seconds, subject to available hardware; it is shared across packages and retries. No TPU training, paid compute, long run or new allocation is authorized by these documents.

Pure-function, schema, deterministic fake-model, bounded forward/backward-without-update and crash-injection checks should cover most implementation. Count their wall time separately; do not conceal repeated expensive gradient sweeps as unit tests. Do not run learned checks merely because a test collection includes them. New modules can be implemented and gradient-wired without claiming they have learned.

The owner requested at least 100 million implementation-agent tokens. Record actual app-reported usage when available and `unknown` otherwise. This repository cannot allocate or verify another app's token budget. Complete the work at the best useful quality; do not generate filler, duplicate code, unnecessary agents or artificial tests to consume tokens. Each M package is roughly 3–5 hours of useful engineering when prerequisites exist; M00 inherits its larger repair scope. These are planning estimates, not measured effort or compulsory duration.

## Definition of the next finished build

A finished implementation has a single auditable path:

`manifest -> public episodes/documents -> typed batches -> shared model and losses -> checkpoint -> fresh-process restore -> bounded interaction -> observed ledger -> candidate update -> independent comparison -> accepted/rejected candidate record`.

It also has an answer-only baseline using the same paths, an inference service with explicit memory and resource scope, a local operator runbook, and an honest readiness report. M18 must distinguish build-complete, locally verified, accelerator-unverified, DATA_NOT_READY and scientifically unqualified states. The source program may be finished while learning capability remains unproven. AGI remains the research objective, not the name of the completion flag.
