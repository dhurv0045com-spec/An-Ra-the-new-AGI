# Metric Contract

DG0 reports measurements without turning them into a learning claim.

| Metric | Definition | DG0 status |
|---|---|---|
| `final_task_success` | submitted answer equals the evaluator-owned hidden answer | reported per world and policy |
| `outcome_prediction_accuracy` | committed binary predictions match observed read/probe/intervention or final target outcomes | reported; not rule identification |
| `calibration_brier` | mean squared error of committed binary predictions | reported when predictions exist |
| `information_gain_per_cost` | reduction in a declared belief entropy per action cost | `not computed` until a learner belief trace is declared |
| `regret_vs_oracle` | oracle success rate minus policy success rate at the same budget | reported for overall policy groups |
| `environment_cost` | all attempted action costs, including rejected attempts | reported exactly |
| `inference_cost` | model or controller inference work | zero for transparent controls; future learner must report it |
| `invalid_action_rate` | rejected action attempts per episode | reported |
| `budget_overrun_rate` | episodes whose attempted cost exceeds the declared budget | reported |
| `abstention_rate` | episodes ending with `stop` and no answer | reported |
| `causal_rule_recovery` | recovery of the evaluator-owned program or graph | not measured in DG0 |
| `transfer_result` | performance on a declared held-out axis | not evaluated in DG0 |

The independent world is the statistical unit. Multiple queries, targets, renderings, or events from one world do not increase the denominator. Future confirmation intervals must cluster by world and must be preregistered before outcomes are opened.

A reported `None` means the metric was not defined for this instrument, not zero. A failed or missing episode remains in the denominator for the primary task-success contract.
