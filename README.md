# Arkenstone — An-Ra cognition research

Arkenstone studies how small models acquire, retain, recover and transfer exact symbolic skills. It is a research branch, with historical V5/ESOES infrastructure retained below the current experiment layer.

**Next implementation assignment:** [Visible capability work order for the next agent](NEXT_AGENT_CAPABILITY_WORK_ORDER.md).

**Start here:** [clarity and improvement report](ARKENSTONE_CLARITY.md), [research evidence index](docs/arkenstone/README.md), and [V6 campaign plan](experiments/COLAB/MASTER_DISCOVERY_V6_PLAN.md).

The strongest recorded findings concern Micro-scale arithmetic retention and recovery. They do not establish AGI, non-arithmetic transfer, or production scheduling readiness. New runtime correctness checks are separate from those scientific results.

## ARK-014 status (this branch)

ARK-014 (order-robust non-arithmetic binding + LR-retention screen) is **implemented and runnable locally**: frozen task contract, matched baseline/candidate acquisition arms, CONTROL-only qualification controller, paired HIGH/LOW retention forks, and a local before/after demo. Protocol interpretations frozen before execution are in [the prospective clarification note](docs/arkenstone/next_capability/PROTOCOL_CLARIFICATIONS.md); the run outcome and honest status are in [the ARK-014 handoff](docs/arkenstone/next_capability/HANDOFF.md).

## Run the bounded integrity preflight

Use Python 3.10+ with PyTorch installed, from this checkout:

```powershell
python experiments/COLAB/run_discovery_v6.py
python -m unittest discover -s tests -p 'test_discovery_v6*.py' -v
```

The default preflight uses CPU, randomly initialized Micro weights and a few optimizer updates. It checks task membership, CONTROL/SEALED separation, exact optimizer/RNG fork replay, source-snapshot immutability and identical sampler streams. It writes a receipt and ZIP to a fresh directory under `artifacts/arkenstone/discovery_v6/`. `--output-dir` must name a directory that does not already exist. Set `ARKENSTONE_RESULTS_ROOT` to change the default parent.

`--campaign ARK-012`, `--campaign ARK-013` or `--campaign ARK-014` selects a single long experiment and additionally requires `--device cuda` and an explicit adequate `--budget-minutes`. The same preflight runs on the requested device before training.

## Run the ARK-014 matched experiment and demo

Bounded preflight only (any device):

```powershell
python experiments/ARK-014/run_ark014.py --device cpu --budget-minutes 5 --output-dir <new-dir> --preflight-only
```

Full frozen matched experiment (CUDA; 2 matched acquisition arms at seed 2201, up to 24,000 steps each, then paired 1e-3/1e-5 retention forks for orders 7701–7703 if a regime qualifies on BIND_CONTROL):

```powershell
python experiments/ARK-014/run_ark014.py --device cuda --budget-minutes 120 --output-dir artifacts/arkenstone/ark014/<new-run-id>
```

Visible before/after demonstration from the completed run's real checkpoints (CPU, no hosted service):

```powershell
python experiments/ARK-014/demo.py --run-dir artifacts/arkenstone/ark014/<completed-run-id> --output <new-demo-directory>
```

The demo verifies checkpoint hashes and the task-manifest identity against the run receipt, recomputes all four held-out diagnostics for both arms, displays real per-example predictions (baseline vs candidate vs ground truth) on a frozen, outcome-independent selection of SEALED fact-sets, and writes `report.html` plus machine-readable `examples.json`. It refuses to fabricate a comparison when checkpoints are missing, modified or unable to support the receipt's claims. Focused checks: `python -m unittest tests.test_ark014 tests.test_ark014_demo -v`.

## Where the code lives

| Path | Purpose |
|---|---|
| `experiments/ARK-011/` | State-conditional recovery/retention controller |
| `experiments/ARK-012/` | Recovery-switch threshold screen |
| `experiments/ARK-013/` | Carry-skill stability/plasticity experiment |
| `experiments/ARK-014/` | Order-robust non-arithmetic binding runner + demo |
| `experiments/COLAB/discovery_v6_common.py` | Shared budget, receipt, metric and sampling runtime |
| `experiments/COLAB/run_discovery_v6.py` | CPU/GPU preflight and explicit single-campaign entry point |
| `docs/arkenstone/improvement_20260912/` | Work order, engineering handoffs and local evidence |
| `docs/arkenstone/next_capability/` | ARK-014 protocol clarifications and handoff |
| `blueprint/` and `v5_*` | Inherited ESOES design/contracts; not evidence that Arkenstone achieved their objectives |

Historical plans and experiment receipts remain unchanged. New code and runs must carry their own source identities; a passing test suite is engineering evidence, not a scientific promotion.
