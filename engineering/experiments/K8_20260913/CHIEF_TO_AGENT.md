# Chief engineer to implementation agent: build the complete causal path

This is a direct work instruction for the BRAMASTRA implementation lead. Read it with [READINESS.md](READINESS.md), [SYSTEM_QUALITY.md](SYSTEM_QUALITY.md), [REVIEW_PROTOCOL.md](REVIEW_PROTOCOL.md) and [experiment.md](../../../experiment.md). It extends I01–I06 acceptance evidence. It does not enlarge the eight-hour allocation, add experimental arms, or authorize local optimizer updates. Where a proposed diagnostic needs learning, place it inside the already allocated E0 budget or report it unverified.

The reviewed source is edee727. Its modular work is useful, but isolated components do not yet constitute a learned system. Do not interpret this criticism as a request to rewrite everything. Preserve correct contracts and finish their consumers. First inspect the current source; findings may already have been repaired by a later commit. For each finding, identify the actual caller, evidence and remaining gap before changing code.

Your deliverable is an operational chain: generated public evidence becomes a prepared batch; that batch produces differentiable objectives; those objectives update the intended shared model; its checkpoint drives real actions; actions produce observations and costs; evaluation measures transfer and retention; method selection produces a controlled successor. A broken link invalidates the capability downstream of it. Naming a class `Executive` or `SelfImprover` does not repair that link.

## First deliver a vertical slice

Before broad parallel work, connect one tiny deterministic episode through the real preparation, objective routing, trainer forward/backward, checkpoint serialization, scorer adapter, executive and tool receipt path. Local verification stops before optimizer.step. Use an explicitly labeled test model where necessary; do not emit learned-capability evidence from it. The owner-launched E0 supplies the actual update and restore check.

Record the paths and symbols of this slice in the handoff. Then generalize it to the declared families and phases. This ordering exposes incompatible masks, identifiers, devices and state schemas early. Building six separate demos and joining them in the notebook is unacceptable: the notebook must exercise the same repository interfaces that were checked locally.

## Six concrete corrections to prioritize

1. Trace device selection from the campaign worker through train and restore. A CUDA availability printout is not proof that parameters, optimizer state, targets and losses use the assigned GPU.
2. Trace every configured objective into the scalar used for backward. A router receipt cannot substitute for an actual differentiable term. Missing eligible supervision must fail visibly.
3. Separate inference helpers from training APIs. Returning a Python float is reasonable at an inference boundary; using it as a learned objective silently severs the gradient.
4. Connect the actual checkpoint scorer to the executive. Capture which checkpoint produced each chosen action. Do not retain an unnoticed scripted fallback in an experimental arm.
5. Connect generated method bytes to a validated, applied method and its successor checkpoint. Fixture generation remains a test control. It must never enter the learned E5 evidence namespace.
6. Make interruption, export and restoration part of the implementation. Evidence that cannot be recovered after a Kaggle kernel stops is an incomplete deliverable.

## Work as an engineer who checks their own assumptions

For each package, write a short prediction before implementation: what observable behavior should change, and what deliberately broken implementation must be rejected? Implement the smallest coherent change, run its focused check, inspect a real trace, then compare the result with that prediction. If it disagrees, fix the cause or document a blocker. Do not change the expectation merely to accept the output.

Review your own work in three passes: mathematical correctness of targets and reductions; integration correctness of callers and state; experimental correctness of comparisons and information boundaries. A second agent can inspect one bounded seam, but you remain responsible for integration. Delegate to Luna only with explicit owned paths, prerequisites, exclusions and expected evidence. Do not spend tokens producing duplicate architecture essays or repeated full-suite runs.

When a test passes suspiciously easily, inspect what it actually exercises. A test that mocks both model scoring and the trainer says nothing about their connection. A tensor with a nonzero gradient somewhere says nothing about the intended head. A generated AST with a model-origin label says nothing about who generated it. The review protocol supplies stronger checks.

## What you return to the owner

Deliver the source, thin notebook, reproducible generated-data preparation and offline runbook. The operator should select input/output locations, inspect the printed allocation and launch the documented cells. They should not need to repair imports, infer missing configuration or copy code from conversation history.

In `engineering/reports/K8_BUILD/HANDOFF.md`, include a disposition for every I01–I06 criterion, the causal-path map, focused commands and results, the negative-control inventory, source/data identities, remaining GPU-only checks and exact owner actions. Link compact evidence rather than listing test counts as the main argument. Commit source and compact reports; keep weights and large generated data outside Git.

Write candidly: implemented, locally verified, CUDA-unverified, experimentally measured, or blocked. If a criterion is not met, your package is partial even when the notebook opens. Do not promise AGI, tenfold gains, or a successful experiment. Build a system whose failures are informative enough to guide the next improvement.
