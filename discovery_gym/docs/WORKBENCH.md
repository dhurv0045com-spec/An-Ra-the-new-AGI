# Discovery Workbench

The workbench is the practical layer above the DG0 instrument. DG0 answers whether a controlled discovery environment is valid. The workbench turns ordinary discrete tables into an auditable investigation package:

```text
CSV/JSON -> immutable dataset identity -> profile -> graph hypotheses
         -> bootstrap stability -> ranked experiments -> replayable session
```

## Quick start

```text
python -m discovery_gym workbench-inspect --input data.csv --output profile.json
python -m discovery_gym workbench-plan --input data.csv --output plan.json --budget 8
python -m discovery_gym workbench-fit --input data.csv --output mechanism.json
python -m discovery_gym workbench-benchmark --input data.csv --output predictive-benchmark.json
python -m discovery_gym workbench-session --input data.csv --output session.json
python -m discovery_gym workbench-execute --input session.json --output session-ran.json --budget 4
python -m discovery_gym workbench-result --input session.json --output session-next.json --recommendation rec-id --executor lab-1 --assignment 1 --outcome '"observed"' --cost 3
python -m discovery_gym workbench-replay --input session-next.json
```

The plan command accepts `--max-parents`, `--stability-replicates`, and `--seed`. It never launches an intervention, claims causality, or silently turns an observational association into a fact.

For multi-receipt data inventories:

```text
python -m discovery_gym workbench-inventory --input receipts.json --output readiness.json
```

Duplicate dataset identities are counted once, conflicting lineage/tokenizer identities are blockers, and a missing qualification cannot produce a runnable state.

## Python API

```python
from pathlib import Path
from discovery_gym.workbench import (
    DiscoverySession,
    build_workbench_report,
    discover_graph,
    load_dataset,
    recommend_experiments,
    stability_report,
)

dataset = load_dataset(Path("data.csv"))
graph = discover_graph(dataset, max_parents=2)
stability = stability_report(dataset, graph, replicates=32, seed=17)
plan = recommend_experiments(dataset, graph, stability=stability)
session = DiscoverySession.create(dataset, graph, plan)
session.record("dataset_verified", {"source_sha256": dataset.source_sha256})
session.save(Path("session.json"))
```

`TableDataset` accepts scalar categorical values from CSV, JSON, or JSONL. It normalizes cell values, validates row shape, computes a content-addressed dataset identity including provenance metadata, reports entropy and duplicate rows, and produces deterministic train/holdout partitions. Duplicate rows are split as groups to prevent exact-row leakage. Empty, non-finite, nested, and malformed cells fail closed. `readiness()` reports `DECLARED -> MATERIALIZED -> VERIFIED -> QUALIFIED -> RUNNABLE` as separate dimensions and never calls an unqualified dataset runnable. Numeric columns can be explicitly converted with `discretize()` and retain parent identity in metadata.

## Graph hypotheses

For up to eight columns, `discover_graph` uses exact dynamic programming over acyclic parent sets with a smoothed BIC-style local score. Larger tables use a deterministic acyclic greedy fallback. The output is explicitly an `observational DAG hypothesis`; its edges are not causal claims.

`stability_report` bootstraps rows, rediscovers the graph, and reports edge support, topology stability, and alternative edges. A low-support edge is a request for better evidence, not a reason to promote it to truth.

`fit_mechanism()` provides a small from-scratch discrete conditional model using only the selected parent sets. `cross_validate_mechanism()` evaluates it with leave-one-duplicate-group-out folds, so repeated rows cannot masquerade as independent evidence. `predictive_benchmark()` compares it with majority and full-conditional predictors on the same folds. These predictions are diagnostics, not causal proof.

## Experiment plans

Recommendations are ranked by edge uncertainty, entropy, and downstream relevance. They distinguish:

- randomized interventions that can test an edge;
- replicate measurements that can reduce sampling uncertainty.

Each recommendation carries a stable ID, cost, target, values, rationale, and evidence. The plan carries dataset and graph hashes, an optional experiment budget, the selected planned cost, and its own content hash. `record_external_result()` validates that a result refers to a planned recommendation, uses an allowed intervention value, names its executor, and records the actual cost.

## Executing a bounded plan

`execute_experiment_plan` closes the loop between a frozen plan and an executor. It walks the plan in priority order, stops at the budget or `max_recommendations`, and records every outcome as a ledger event:

```python
from discovery_gym.workbench import LocalMechanismExecutor, execute_experiment_plan, fit_mechanism

executor = LocalMechanismExecutor(fit_mechanism(dataset, graph))
report = execute_experiment_plan(session, executor, budget=4, max_recommendations=2)
```

The report accounts for `attempted`, `completed`, `failed`, and `skipped` recommendations, the `actual_cost` and `remaining_budget`, the count of `useful_experiments`, and `useful_per_cost`, which is the unit the 100x roadmap measures. Failures are not silent: a raising executor costs its planned budget and appends an `execution_failure` event with the exception text, so a crashed run is still auditable evidence. Executor-reported costs below the planned cost are rejected.

An executor is any object with a nonempty `identity` and an `execute(recommendation, assignment)` method. `CallbackExecutor` wraps a function for external owners; `LocalMechanismExecutor` simulates interventions against the fitted mechanism so the loop can be rehearsed offline. Only the callback path with a real owner produces external evidence; the local simulator carries an explicit claim boundary in its report.

```text
python -m discovery_gym workbench-execute --input session.json --output session-next.json --budget 4 --max-recommendations 2
```

## Durable evidence

`EvidenceLedger` is an append-only hash chain. `DiscoverySession` binds the dataset, graph, plan, and ledger into one replayable record. `verify()` detects deletion, insertion, reordering, and payload mutation. A session can be reloaded without rerunning discovery. The serialized session and dataset identities also have JSON schemas under `discovery_gym/schemas/`.

## What is useful now

This layer provides immediate value for dataset triage and experiment design:

- finds a compact set of candidate dependencies;
- identifies which edges are fragile under resampling;
- converts uncertainty into a prioritized experimental queue;
- preserves every plan and observation identity;
- makes a negative or incomplete investigation easy to resume.

## Boundaries

The workbench does not yet infer latent variables, nonlinear mechanisms, continuous-time dynamics, or unmeasured confounding. `workbench-execute` only runs the bundled local simulator; it never touches real hardware. A recommendation becomes evidence only after an external executor records the intervention identity, assignment policy, outcome, cost, and provenance. Those records should be appended to a session and independently reviewed.

The next practical upgrades are explicit measurement error modeling, continuous-variable modeling beyond explicit discretization, DAG equivalence-class reporting, and adapters that execute approved plans against real experiment owners. The current workbench is deliberately useful before it is autonomous: it tells an investigator what is known, what is fragile, what to measure next, and how to preserve the result.
