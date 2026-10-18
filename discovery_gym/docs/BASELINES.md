# Baseline Interface and Definitions

## Interface

Every policy implements:

```python
class Policy:
    identity: str
    uses_hidden_truth: bool

    def reset(self, episode_id: str) -> None: ...
    def act(self, view: PublicView) -> Action: ...
```

The policy receives a `PublicView` only. It cannot request a world object, hidden answer, evaluator label, confirmation split, or future transcript. The runner resets the policy for each independent world and records the policy identity, action cost, prediction rows, and attribution fields.

## Definitions

| Baseline ID | Behavior | Hidden truth | Role |
|---|---|---:|---|
| `random-legal/v1` | Seeded uniform choice among current legal descriptors | no | random legal control |
| `fixed-schedule/v1` | Read variables, then use a fixed non-target probe order, then submit a public-prior answer | no | schedule control |
| `uncertainty-seeking/v1` | Select the first legal probe/intervention with a binary outcome, reserving submission cost | no | simple uncertainty control |
| `transparent-information-gain/v1` | Maintain a posterior over the declared finite rule catalog, read current inputs, and choose the legal action with the best declared entropy reduction per cost while reserving submission cost | no | strong transparent decision-tree/information policy |
| `transparent-lookahead/v1` | Enumerate bounded public belief-state action branches and choose the action with the best finite-horizon terminal probability; optional extended control | no | planning/control reference, not learner evidence |
| `no-experiment/v1` | Submit the declared-prior majority immediately | no | zero-information control |
| `surface-position-shortcut/v1` | Read the first public variable, then submit zero | no | surface/position control |
| `task-identity-shortcut/v1` | Hash the opaque episode ID and submit the resulting bit | no | identity/order control |
| `evaluator-oracle/v1` | Submit the evaluator-owned hidden answer | yes | diagnostic ceiling only |

The optional `extended_policy_factories()` adds `transparent-lookahead/v1` for deeper finite-horizon planning; it is intentionally excluded from the default smoke runtime until its cost is measured and frozen. The information-gain policy is deliberately transparent. Its finite catalog is a declared public prior; it uses the public probe complement as its target-candidate prior, filters rule hypotheses from observed consequences, scores expected answer-entropy reduction per unit cost, and predicts each action's own observed outcome. The selected program and target assignment are not in its input. It is not evidence that a neural model learned exploration. The runner rejects any policy with `uses_hidden_truth=true` unless the caller explicitly sets `allow_hidden_truth=True`; the DG0 CLI sets that flag only for its separately labeled diagnostic suite.

## Cost matching

All policies receive the same maximum action budget. The result records experiment cost, rejected-action cost, total cost, invalid actions, budget overruns, and abstentions. Inference cost is zero for these hand-written controls and is recorded as such. A future learned policy must report its own model invocations, candidate rows, and training/inference cost separately.

## Development smoke result

The deterministic DG0 run used seed `1701`, six worlds per split, 24 worlds, and budget `8`. It produced 192 episodes across eight policy identities. The observed controls were:

| Policy | Final success | Mean total cost | Mean experiment cost | Invalid actions |
|---|---:|---:|---:|---:|
| evaluator oracle | 1.000 | 1.00 | 0.00 | 0.0 |
| transparent information gain | 0.667 | 7.42 | 6.42 | 0.0 |
| fixed schedule | 0.542 | 8.00 | 7.00 | 0.0 |
| uncertainty seeking | 0.708 | 8.00 | 7.00 | 0.0 |
| no experiment | 0.708 | 1.00 | 0.00 | 0.0 |
| surface/position shortcut | 0.458 | 2.00 | 1.00 | 0.0 |
| task-identity shortcut | 0.333 | 1.00 | 0.00 | 0.0 |
| random legal | 0.208 | 5.29 | 4.63 | 0.0 |

These are small deterministic development diagnostics, not independent confirmation statistics. The transparent policies are controls, not a performance claim; a learned policy must beat the strongest observed control at equal cost. If no control is beaten, the benchmark or learned mechanism is not demonstrated.
