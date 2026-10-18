# An-Ra Discovery Gym

Discovery Gym is a bounded research instrument for testing whether a from-scratch learner can investigate unfamiliar causal programs, predict withheld outcomes, and transfer discoveries under equal cost. It is an extension of the existing BRAMASTRA research family, not a second BRAMASTRA discovery implementation.

## Research question

> Can a small, randomly initialized model learn to select useful legal experiments, update its beliefs from observations and interventions, predict a withheld causal outcome, and transfer that competence to structurally new worlds better than strong transparent baselines at equal experiment and inference cost?

The first executable milestone is DG0 only: establish that the environment, evaluator, provenance, split boundary, evidence firewall, legality checks, and baseline interface are valid. DG0 does not train a model and cannot establish learning.

## Bounded DG0 hypothesis

A deterministic finite causal-program environment can expose a public action schema and budget, hide its rule and target answer, commit each action and prediction before exposing the consequence, and allow an independent evaluator to reproduce every outcome without trusting policy-provided labels.

The DG0 hypothesis is falsified if any of the following occurs:

- a legal action, cost, outcome, or state transition disagrees with the contract;
- a public policy view contains a hidden rule, target assignment, seed, evaluator label, or future consequence;
- a direct target query is legal or an experiment can silently reveal the withheld answer;
- a renamed or isomorphic latent program crosses a split boundary;
- the independent evaluator cannot reproduce an outcome;
- a shortcut control reaches the oracle ceiling without using causal evidence.

## Component boundaries

| Component | Owner in DG0 | Capability credit |
|---|---|---|
| Causal program generator | Evaluator-owned generator | World validity and provenance only |
| Environment state machine | Environment | Legal transitions and costs only |
| Public evidence firewall | Evaluator/runtime | Boundary enforcement, not discovery |
| Independent evaluator | Evaluator | Ground-truth scoring, not policy choice |
| Transparent baselines | Baseline code | Control performance, not learned capability |
| Oracle policy | Evaluator diagnostic | Ceiling only; never learner-visible |
| Learned policy | Not run in DG0 | No capability claim |

A future learned policy may use the same public interface, but its decisions, predictions, belief updates, inference cost, and checkpoint identity must be recorded separately from the environment and evaluator.

## Stage boundary

- **DG0:** instrument validity. Current implementation target.
- **DG1:** small from-scratch learner improves over no-learning controls on fresh development worlds.
- **DG2:** prospective experiment choice on untouched confirmation worlds at equal cost.
- **DG3:** structural transfer by declared axis.
- **DG4:** sequential acquisition and retention.
- **DG5:** internalized discovery; future preregistration only, not implemented here.

## Claim ceiling

The strongest current claim is: *a deterministic causal discovery instrument can be generated, audited, and scored under an explicit information firewall.* No claim is made that a model discovered a rule, selected experiments intelligently, transferred, retained, or achieved AGI.
