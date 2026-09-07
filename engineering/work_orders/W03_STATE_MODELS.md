# W03 — Shared learner and state controls

**Status:** depends on W01 and usable W02 fixtures. **Effort:** 3–5 hours. **Role:** model engineer. **Compute:** CPU model tests and short learnability pilots; no large pretraining.

Read architecture, contracts and algorithms section A. Own `research/models/`, `tests/test_research_models*`, model fixtures and `engineering/reports/W03/`.

## Deliverable

Implement the width-128 integrated shared learner with typed observation encoder, recurrent state, goal-conditioned outcome head, legal-action scorer and value head. Implement a full-history control with the same public information. Provide explicit inference/reset APIs and a parameter/compute specification derived from real modules.

Do not add a large memory subsystem, mixture-of-experts or a second unrelated language trainer. First establish whether the shared learner can use observed evidence and goals under the canonical interface.

## Experiments and controls

Run small learnability checks on known rules and state transitions. Evaluate paired changed-goal cases, delayed observations, irrelevant events and overwrite sequences. Compare memory-enabled/disabled behavior and full-history/recurrent controls. Use a simple task that genuinely requires history, not one recoverable from the final observation alone.

Match training data and report parameters, inference operations, history truncation and wall time. A bigger control may be useful but is not an architecture-only contrast.

## Acceptance evidence

- Padding and unrelated episodes cannot affect a sample's prediction; state resets are explicit and tested.
- Changing the target affects outputs on independently specified counterfactual cases.
- Legal-action masks are respected for sampling and greedy inference; no-legal-action states terminate cleanly.
- All enabled heads receive intended gradients on an informative fixture; finite updates mutate expected parameters.
- A tiny learnable fixture is acquired from random initialization with a fixed, reported budget.
- Memory and full-history ablations run with the same observation access, and raw outcomes are retained whether positive or negative.
- W04 receives policy/value APIs; W05 receives next-observation prediction access; W07 receives state serialization requirements.

Tests passing establishes implementation correctness, not transfer. Report failed learning curves rather than selecting a favorable seed or quietly changing the task. Any new architecture proposal goes into a separate experiment revision.
