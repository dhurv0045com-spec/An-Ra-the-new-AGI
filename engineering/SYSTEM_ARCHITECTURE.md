# Integrated system architecture

Status: proposed canonical system; `bramastra_lab/discovery` is the initial prototype. This design has no runtime dependency on Cymek, Citadel or another branch. Reuse reviewed BRAMASTRA components through explicit adapters, not silent inheritance.

## 1. A single learning loop

```text
training environment -> public observation/action history -> shared learner
                                                           |       |
                                                      prediction  policy
                                                           \       /
                                                        bounded planner
                                                             |
                                                         real action
                                                             |
                                               observed transition + cost
                                                             |
experience ledger -> qualified replay -> training update -> candidate checkpoint
                                                             |
                              independent transfer/retention examiner
                                                             |
                                                accept / reject / diagnose
```

The important unit is the entire loop. Training an encoder, planner and memory separately does not establish that they cooperate. Every experiment must identify whether it evaluates core prediction, policy, memory, planning, or the full system.

## 2. Canonical modules and dependency direction

Proposed namespace: `bramastra_lab/research/`.

| Module | Responsibility | May depend on | Must not depend on |
|---|---|---|---|
| `contracts` | Versioned records, validation, identities and capability schemas | Standard library | Models, hidden environment implementations |
| `environments` | Training worlds, observations, actions, rewards | Contracts | Learner internals |
| `experience` | Episode storage, split/dedup, replay and data packing | Contracts | Examiner answer storage |
| `models` | Encoders, recurrent state, prediction/policy/value heads | Contracts, tensor backend | Hidden simulator state, evaluator labels |
| `learning` | Supervised bootstrap, actor-critic, consolidation | Models, experience, contracts | Confirmation outcomes during optimization |
| `planning` | Bounded imagined action sequences and real-feedback checks | Model public inference API, contracts | Simulator oracle except named diagnostic adapters |
| `runtime` | Device execution, checkpoint, sampler/RNG, resource limits | Contracts, backend adapters | Task-specific answer logic |
| `evaluation` | Independent tasks, paired metrics, retention and claim gates | Public model interface, contracts, examiner-owned worlds | Training mutation during frozen evaluation |
| `orchestration` | Experiment execution and evidence ledger | Public module APIs | Undeclared access to privileged state |

These dependency restrictions need import and behavioral tests. A class named `PublicObservation` does not prevent leakage if it carries a world object in an extra field.

## 3. Learner capacity and state

Use three stages of capacity, measured rather than assumed sufficient:

- **Diagnostic learner:** the existing small recurrent discovery model. Cheap enough to expose target, memory and objective failures locally.
- **Integrated control:** shared width-128 observation encoder and recurrent state, with small task-conditioned prediction, policy and value heads. Parameter count must be derived from the actual implementation and recorded. R0 full-history control and R1 recurrent state use the same observation/action interface.
- **Language-capable candidate:** the existing BRAMASTRA B1 configuration is a candidate at 42,742,272 parameters, not a required first run or AGI size claim. Add language only through a frozen vocabulary/data interface, and scale after smaller learning curves support the need.

The initial working state is a fixed-size tensor. A learned slot memory is an ablation, not a default extra module. Compare R0/R1 with the same visible history and report compute, memory, parameters and context truncation. Do not claim recurrence is superior solely because it receives more processing steps.

## 4. Information boundaries

The learner sees goal, public schema, observation, previous legal action, feedback, remaining budget and permitted episodic retrieval. It does not see hidden rule identifiers, simulator adjacency, future outcomes, evaluation labels or the examiner's selection criteria.

Training-only privileged teachers may produce labels. Their domain, code identity and privileges must be recorded. Test-time teacher access changes the system being evaluated and must be a separately named diagnostic.

The initial prototype passes world objects to the evaluator process, which then calls the model with tensors. That supports a local API boundary, not an adversarial process-isolation guarantee. Stronger boundaries are required before evaluating model-generated code.

## 5. Three kinds of learning

**Within-task adaptation:** weights fixed, working state changes from observed experience. Reset state by task identity and measure learning curves versus real inquiries.

**Across-task consolidation:** parameters change from training experiences and replay. Parent and child are compared on fresh tasks with matched inference settings.

**Method improvement:** candidate changes to experience selection or the learning procedure compete against the parent procedure. Include the cost of searching for the change. Initially a fixed orchestrator runs these comparisons; learned method selection is a later hypothesis.

Keep these claims separate in every report.

## 6. Broader capability path

Begin with hidden rules, causal interventions and stateful planning because their truth can be checked cheaply. Add program execution and repair, then grounded document/code tasks and natural-language instructions. Later expand to visual observations, longer horizons and independently authored practical tasks.

Avoid a disconnected collection of specialist solvers. Shared representations and transfer tests must connect new domains to the same learner. Compare with domain-specific oracles only as upper-bound diagnostics. An oracle's competence is not the model's competence.

## 7. System failure taxonomy

Every failed episode receives one primary observable failure category plus optional diagnostics: invalid action, missing termination, exhausted budget, wrong world prediction, state loss, bad inquiry selection, planning error, unlearnable ambiguity, optimization failure, data leakage, runtime failure, or unknown.

Do not attribute a failure to “insufficient intelligence” before ruling out broken targets and inaccessible information. Conversely, do not explain every scientific failure away as a need for more engineering.
