# An-Ra Discovery Gym

This directory is the scoped DG0 continuation package for the Gandiva/BRAMASTRA research family. It adds a small deterministic causal-program environment with explicit public/private information, target-safe intervention anchoring, legal and costly experiments, prospective action commitments, independent evaluation, structural split provenance, shortcut audits, and stage gates. The refined contract, generator, and protocol identities are versioned `v2`.

## Run

```text
python -m pytest tests/test_discovery_gym.py -q --tb=short
python -m discovery_gym dg0 --output <new-empty-directory> --seed 1701 --worlds-per-split 6 --budget 8
```

DG0 uses only the Python standard library and does not train a model. Read `docs/RUNBOOK.md` before running it and `docs/HANDOFF.md` for scope, evidence, blockers, and the DG1 next action.

## Documents

- `CHARTER.md`: mission, question, falsifier, component boundaries, and claim ceiling.
- `docs/OVERLAP_AUDIT.md`: exact branch/ref/commit overlap table and reuse decision.
- `docs/CONTRACT.md`: formal world and action contract.
- `docs/SPLITS_AND_PROVENANCE.md`: structural split and identity rules.
- `docs/BASELINES.md`: policy interface, definitions, attribution, and smoke results.
- `docs/METRICS.md`: metric denominators, costs, unavailable fields, and independent-world unit.
- `docs/AUDITS.md`: leakage, shortcut, legality, and confirmation-firewall checks.
- `docs/STAGE_GATES.md`: DG0–DG5 conditional gates.
- `docs/PROTOCOL_TEMPLATE.json`: prospective DG1 template.
- `docs/RESULT_SCHEMA.md` and `schemas/`: result, protocol, checkpoint, dataset, and workbench identity contracts.
- `docs/RUNBOOK.md`: owner reproduction and hardware boundary.
- `docs/WORKBENCH.md`: practical discrete-data discovery, experiment planning, and replayable evidence sessions.
- `docs/ROADMAP_100X.md`: measurable useful-experiments-per-interaction target and acceptance protocol.
- `docs/RESOURCE_STATUS.md`: captured host capacity and unclaimed hardware status.
- `docs/HANDOFF.md`: final scoped continuation package.

## Current status

The deterministic DG0 smoke passes its instrument audit, and the practical workbench now provides data inspection, graph-hypothesis generation, stability analysis, experiment ranking, and replayable evidence sessions. No scientific learning, causal proof, transfer, retention, recursive improvement, or AGI result is claimed. The next action is to validate the workbench on owner-supplied tabular fixtures before connecting an external experiment executor.
