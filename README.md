# An-Ra Research

An-Ra is a research codebase for studying how neural systems form, transfer, and retain capabilities, and how proposed learning and self-improvement mechanisms can be tested with controlled evidence. The repository is an experimental program, not a claim that it already contains AGI.

## Current focus

- **Signac 100M** packages the M102 model recipe at exactly **101,790,080 parameters**, with model, optimizer, checkpoint, preflight, and Kaggle TPU qualification code. The Kaggle notebook runs synthetic engineering canaries; a research training run is still blocked on qualified data, evaluation, TPU runtime and memory evidence, production resume, and durable-output checks.
- **Cymek** investigates capability formation, representation, transfer, retention, and continued learning. Read the current [research state](docs/cymek/research/CURRENT_STATE.md) and [next-core readiness](docs/cymek/next_core/NEXT_CORE_READINESS.md) before interpreting experiments.
- **Arkenstone and Citadel** contain related experiments and evaluation work. Their results have explicit evidence and claim limits in the research records; a result from one system does not automatically qualify another.

## Start here

- [Signac 100M overview](docs/signac_100m/README.md)
- [Signac readiness and blockers](docs/signac_100m/READINESS.md)
- [Kaggle TPU runbook](docs/signac_100m/KAGGLE_TPU_RUNBOOK.md)
- [Phase-one research contract](docs/signac_100m/PHASE1_TWIN_CONTRACT.md)
- [Cross-branch evidence ledger](docs/signac_100m/EVIDENCE_LEDGER.md)
- [Research evidence index](docs/research/EXPERIMENT_EVIDENCE_LEDGER.md)

## Check the 100M package

Run the static target preflight locally:

```powershell
python tools/signac_100m_preflight.py --target tpu
```

Until the required external evidence is supplied, the expected verdict is `BLOCKED`. The preflight deliberately cannot authorize training from architecture estimates or synthetic canary results. For Kaggle setup, outputs, and qualification steps, follow the [TPU runbook](docs/signac_100m/KAGGLE_TPU_RUNBOOK.md).
