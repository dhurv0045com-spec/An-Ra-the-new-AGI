# Discovery Gym Charter and DG0 Plan

## Mission

Build a durable, architecture-neutral environment for testing whether a model trained from random initialization can learn to investigate unfamiliar worlds rather than memorize answers. The environment must separate prediction, experiment choice, belief update, transfer, retention, and policy learning so that a result cannot silently credit a hand-written controller, symbolic solver, evaluator, or external teacher to the model.

## Formal first-milestone question

Can a small from-scratch model use a limited and costly sequence of legal experiments to discover hidden causal rules, predict withheld outcomes, and transfer those rules to structurally new worlds better than strong transparent baselines at equal experiment and inference cost?

## DG0 hypothesis

A compact deterministic causal-program environment can:

1. generate a hidden rule and withheld target from an evaluator-owned program;
2. expose only a public state, legal actions, declared costs, and prior observations;
3. commit an action and any prediction before applying the action;
4. charge and record every attempted action, including rejected actions;
5. return only declared consequences;
6. terminate through a submitted answer or an explicit abstention;
7. reproduce every outcome with an independent evaluator; and
8. keep one latent world, its queries, renderings, and provenance in one split.

This is an instrument hypothesis. It makes no claim that a model learns anything.

## Falsifier

DG0 fails if a hidden rule, target assignment, seed, future outcome, evaluator label, or direct target query reaches a policy view; if a legal action or cost is inconsistent; if a renamed/isomorphic program crosses a split; if the evaluator cannot recompute a result; or if a surface shortcut reaches the oracle ceiling without causal evidence.

A wide uncertainty interval is not a DG0 pass. A missing result is not a negative result. A passing unit test is implementation evidence, not a scientific result.

## Initial hypothesis boundary

The first executable environment has three binary variables and a finite, composable rule catalog containing two-parent operators, three-parent operators, and mixed graph shapes. A world contains a hidden program, an unknown current-state assignment, and a fixed set of non-target probe contexts. The policy may read current-state variables, probe other contexts, perform a temporary intervention from a public anchor whose one-variable neighborhood excludes every target candidate, and submit one binary answer.

The public view does not identify the selected structure. A declared finite rule catalog is available to transparent diagnostic policies as a prior; the hidden selected program and current-state assignment remain evaluator-only.

## Scientific stages

| Stage | Question | Status in this package |
|---|---|---|
| DG0 | Is the instrument valid and auditable? | Implemented and run locally |
| DG1 | Can a small random-initialized learner fit a learnable signal? | Not run |
| DG2 | Does learned experiment choice beat fixed/random controls prospectively? | Not run |
| DG3 | Does the learner transfer across declared structural axes? | Not run |
| DG4 | Can it acquire later skills while retaining earlier ones? | Not run |
| DG5 | Can external discovery guidance be removed? | Future preregistration only |

## Attribution rule

Every result identifies the component that made each decision. The generator chooses worlds, the environment executes actions, the evaluator supplies truth, a baseline chooses actions, and a future model chooses actions and updates parameters or state. An oracle is labeled `evaluator-oracle/v1` and is never counted as learner behavior. No local optimizer update is part of DG0.

## Resource and claim boundary

DG0 uses only Python's standard library, deterministic CPU execution, small worlds, and no downloads. It does not use pretrained weights, Hugging Face inference, external datasets, GPUs, TPUs, sealed data, or long-running jobs. The current claim ceiling is instrument validity only. The project does not claim causal discovery by a model, transfer, retention, recursive improvement, or AGI.
