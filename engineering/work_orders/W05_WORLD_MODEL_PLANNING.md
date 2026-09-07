# W05 — Predictive dynamics and bounded planning

**Status:** depends on W02/W03 interfaces. **Effort:** 3–5 hours. **Role:** planning/model engineer. **Compute:** CPU small-world training and planning checks.

Read architecture and LEARNING_ALGORITHMS.md E. Own `research/planning/`, planning tests and `engineering/reports/W05/`. Coordinate any requested model-head change through W03; do not edit another agent's model implementation concurrently.

## Deliverable

Train next-observation/reward/termination predictions from real transition records. Implement bounded action-sequence planning with at most four candidates and depth two initially. Support no-planning, learned-dynamics, oracle-dynamics and exact-small-planner diagnostic adapters through explicit interfaces.

## Required design

Separate observed transitions from imagined ones in storage. Use true termination and truncation correctly. Record rollout horizon, candidate count, model calls and wall time. Action legality must be enforced along imagined trajectories, including cases where the predicted state is uncertain.

Define a conservative response to badly calibrated dynamics: shorten the horizon or use a real observation. Do not treat an uncertain prediction as a verified environment fact.

## Acceptance evidence

- Prediction targets match actual environment transitions, including delayed effects and termination.
- Planner respects a hard model-call/inference budget and always returns a legal action or explicit failure.
- Oracle diagnostic recovers a known short plan in an independently specified tiny world.
- Learned dynamics error is reported by horizon and held-out mechanism family.
- Four-way decomposition separates dynamics error from planner error and from lack of benefit over reactive action.
- Counterexamples where longer planning worsens outcomes are retained and reported.
- A small development comparison measures real task success per total inference cost, not only imagined return.

Do not implement unbounded tree search, claim simulator rollouts are learned imagination, or reuse evaluation answers as dynamics labels. A negative planning result can complete the packet if its cause is exposed and the interfaces remain useful.
