# Formal World and Action Contract

## World

A world is a deterministic evaluator-owned record containing:

- `world_id`: opaque `dgw-...` identity;
- a hidden `RuleSpec` from the finite causal catalog;
- a hidden current-state assignment in `{0,1}^3`;
- four public non-target probe contexts;
- a public intervention anchor when interventions are available;
- a public variable rendering;
- a target context identity, but not its assignment;
- a generation seed and complete input hash, retained evaluator-side;
- a split and structural identity, retained evaluator-side.

The hidden rule maps an assignment `x = (x0,x1,x2)` to a binary `y`. The finite grammar currently includes `xor2`, `and2`, `or2`, `threshold2`, `xor3`, `and3`, `or3`, `majority3`, a nested `xor(a,and(b,c))`, a `mux`, `xor_and`, and a negated threshold. These are deterministic program shapes, not natural-language claims.

The target assignment is withheld. A public intervention anchor is sampled first; its three one-variable neighbors and the anchor form the four public probe contexts, and the target is sampled from the disjoint four-context complement. Every allowed intervention starts at that anchor, so no one-variable intervention can reach any target-candidate assignment. The partition is independent of the target value, and the world is one statistical unit even when it produces many transcript events.

## Public state

The policy receives only `PublicView`:

- opaque episode ID;
- goal text;
- public variable names and roles;
- target context ID;
- public probe contexts;
- the public intervention anchor through action descriptors when that affordance is available;
- legal action descriptors;
- prior public action commitments and feedback;
- remaining action budget and accumulated cost;
- a public belief-space identifier.

It does not receive the rule, rule hash, structure ID, variant, target assignment, generation seed, split, evaluator record, future outcome, or oracle verdict. `belief_space_id` identifies the declared finite prior only; it is not the selected rule.

## Actions

| Kind | Mode | Preconditions | Cost | Outcome domain | State change |
|---|---|---|---:|---|---|
| `read` | observational | named public variable is unread | 1 | `{0,1}` | reveals one current-state input |
| `probe` | observational | public context is unused and is not the withheld target | 2 | `{0,1}` | reveals one non-target rule output |
| `intervene` | interventional | public intervention anchor, variable, and value are unused; the resulting assignment is outside the target-candidate set | 3 | `{0,1}` | temporarily overrides one variable for one rule evaluation and reveals the result |
| `submit` | evaluative | binary answer is supplied | 1 | none | terminates; the evaluator records correctness separately |
| `stop` | control | episode is active | 0 | none | terminates as an abstention |

The action descriptor is generated from public state only. Hidden state never removes a candidate action. A repeated read, repeated probe, repeated intervention, target-context query, target-effective intervention, malformed payload, or over-budget legal-looking attempt is rejected. Target safety is guaranteed by the public anchor construction rather than by a target-dependent action list.

## Prediction commitment

Before applying an action, the environment appends an `ActionCommitment` containing the action, a complete pre-action public-view snapshot, its hash, and the optional binary prediction. Only after that append can the environment compute feedback. The snapshot may contain earlier observed consequences, but it cannot contain the current action's consequence. The event is then appended with acceptance, cost, and feedback. `assert_commitment_precedes_outcome` checks the ordering and snapshot hash.

For `submit`, the answer is the learner's committed prediction and the prediction field must equal it. For experiment actions, a prediction is optional in DG0 but all supplied predictions are logged and scored. For `stop`, no outcome prediction is allowed.

## Cost and termination

The budget unit is one action. The declared cost is charged for every attempted action. A rejected action is charged, receives explicit invalid feedback, and does not mutate hidden state. An attempt whose declared cost exceeds remaining budget is rejected, charged, reported as a possible budget overrun, and truncates when the remaining budget reaches zero. This makes repeated, failed, and uninformative attempts visible rather than free.

An accepted `submit` or `stop` terminates the episode. A nonterminal action that exhausts the budget truncates the episode. A successful final answer is computed by the independent evaluator from the hidden rule and target assignment; the policy cannot submit a label to the evaluator as proof.

## Independent evaluation

The evaluator recomputes each accepted read, probe, intervention, and submission from the evaluator-owned world. It records expected values and correctness separately from public transcript events; the public submit feedback contains the submitted answer and termination, not a success label. The oracle policy receives the hidden world only in an evaluator diagnostic path and is marked `uses_hidden_truth=true`; it is never a learner baseline.

## Extension boundary

Later noise, richer graph shapes, additional variable roles, and new renderings require a new generator version, split manifest, protocol hash, and audit. They cannot be silently added to DG0 results.
