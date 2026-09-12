# Arkenstone — An-Ra cognition research

Arkenstone studies how small models acquire, retain, recover and transfer exact symbolic skills. It is a research branch, with historical V5/ESOES infrastructure retained below the current experiment layer.

**Next implementation assignment:** [Visible capability work order for the next agent](NEXT_AGENT_CAPABILITY_WORK_ORDER.md).

**Start here:** [clarity and improvement report](ARKENSTONE_CLARITY.md), [research evidence index](docs/arkenstone/README.md), and [V6 campaign plan](experiments/COLAB/MASTER_DISCOVERY_V6_PLAN.md).

The strongest recorded findings concern Micro-scale arithmetic retention and recovery. They do not establish AGI, non-arithmetic transfer, or production scheduling readiness. New runtime correctness checks are separate from those scientific results.

## Run the bounded integrity preflight

Use Python 3.10+ with PyTorch installed, from this checkout:

```powershell
python experiments/COLAB/run_discovery_v6.py
python -m unittest discover -s tests -p 'test_discovery_v6*.py' -v
```

The default preflight uses CPU, randomly initialized Micro weights and a few optimizer updates. It checks task membership, CONTROL/SEALED separation, exact optimizer/RNG fork replay, source-snapshot immutability and identical sampler streams. It writes a receipt and ZIP to a fresh directory under `artifacts/arkenstone/discovery_v6/`. `--output-dir` must name a directory that does not already exist. Set `ARKENSTONE_RESULTS_ROOT` to change the default parent.

`--campaign ARK-012` or `--campaign ARK-013` selects a single long experiment and additionally requires `--device cuda` and an explicit adequate `--budget-minutes`. The same preflight runs on the requested device before training. No GPU campaign has been validated by this improvement pass. ARK-014 and the full V6 orchestrator remain unfinished; older pinned notebooks do not automatically include these changes.

## Where the code lives

| Path | Purpose |
|---|---|
| `experiments/ARK-011/` | State-conditional recovery/retention controller |
| `experiments/ARK-012/` | Recovery-switch threshold screen |
| `experiments/ARK-013/` | Carry-skill stability/plasticity experiment |
| `experiments/COLAB/discovery_v6_common.py` | Shared budget, receipt, metric and sampling runtime |
| `experiments/COLAB/run_discovery_v6.py` | CPU/GPU preflight and explicit single-campaign entry point |
| `docs/arkenstone/improvement_20260912/` | Work order, engineering handoffs and local evidence |
| `blueprint/` and `v5_*` | Inherited ESOES design/contracts; not evidence that Arkenstone achieved their objectives |

Historical plans and experiment receipts remain unchanged. New code and runs must carry their own source identities; a passing test suite is engineering evidence, not a scientific promotion.
