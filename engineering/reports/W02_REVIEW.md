# W02 chief review — incomplete environment qualification

Date: 2026-09-08. Status: not accepted; corrections delegated. These findings concern saved, uncommitted implementation from an agent interrupted by a usage limit.

## Blocking findings

1. `qualification.py` assigned oracle success from the number of fixtures, coverage success as 1.0, random success from coin draws, and ambiguity/diagnostic flags as constants. These are not policy evaluations. Replace them with executed public-API rollouts, independent oracle/identifiability checks and raw outcomes. No such report may be called measured qualification.
2. The inventory environment awarded success for predicting that the goal was not achieved. This allows an initially idle agent to win without doing the task. Goal-attainment reward must require actually reaching the goal; diagnostic state prediction is a different task and metric.
3. Inventory ticking occurred before argument validation, so rejected actions could change state. Rejected invalid actions must not advance time, consume budget or alter hidden state unless an explicit environment contract says they are charged actions.
4. Switch canonicalization, declared gate order, output-node selection and surface permutations could disagree. Use validated topology and an explicit output. Define mechanism identity independently of rendering and query choice, with exact truth-function checks in the finite domain.
5. Program identity included the scored input and could distinguish semantically identical functions. Different queries to the same hidden mechanism must remain in one semantic split cluster.
6. Strict bit/budget/type validation and actual public-schema enforcement need independent negative cases. A dictionary without an `answer` key is not proof that hidden information is absent.

## Required acceptance evidence

Produce nontrivial deterministic task populations across all three families, public policy traces, actual costs/rewards/termination reasons, independent evaluator outcomes, semantic overlap reports and ambiguity rates. Include a deliberately failing baseline and assert that qualification records its failure. Demonstrate that changing policy execution changes the measured report.

Name the budget unit accurately. If the shared environment charges every action, including final submission, its limit is an action budget, not an inquiry-only budget. Report inquiries, submissions and total actions separately so comparisons with D02's query budget are not silently mismatched. Specify whether invalid actions are rejected without change or charged as failed actions before training a policy against that behavior.

The resumed W02 execution agent owns corrections, tests and a new immutable qualification run. The chief owns acceptance. No earlier placeholder score is accepted evidence of solvability or intelligence.
