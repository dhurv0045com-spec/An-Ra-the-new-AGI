# K8 chief design handoff

2026-09-13. Reviewed agent source: `edee72726706e0f8f46ec98bb3c979ebc2930233`. The owner requested a substantial7–8-hour two-T4 Kaggle experiment including cognition, tools, architecture and RSI, then emphasized efficiency and independent engineering judgment.

Delivered [root experiment.md](../../../experiment.md), [I01–I06 readiness](READINESS.md), [agent prompt](AGENT_PROMPT.md), and [machine-readable campaign allocation](campaign.json). The package contains6,310 whitespace-delimited Markdown words at validation. It defines one480-minute campaign, not eight hours per arm or automatic repeat campaigns.

## Experiment decisions

- Two independent workers supply two from-scratch acquisition seeds and paired controls; no DDP or unverified pooled-memory assumption.
- E0 spends at most30 minutes establishing actual CUDA training, gradient routing and resume, then freezes throughput-derived targets and case inventories.
- E1 compares answer-only versus unified supervision; E2 measures cognitive behavior; E3 tests tool acquisition and retention; E4 tests a function-preserving gated block-reuse candidate.
- E5 is an actual model-origin method-selection experiment. P0 selects its own successor's learning method. P1 and P_fixed receive matched archive/time under selected versus fixed methods; P1/P_fixed/P0 choices are scored on fresh meta-confirmation tasks. A gain over P0 alone cannot establish a self-selected-method benefit.
- Qualified generated worlds/tool tasks are the primary dataset. A huge owner-provided corpus is not a prerequisite. Synthetic-domain limits and narrow RSI scope are explicit.
- New learning stops at minute450, reserving30 minutes for evidence/restore/export. All setup, failed attempts, search, proposer training and comparisons fit one session allowance. The old206-update CPU ledger is preserved.

One existing Luna agent performed bounded source and design audits. The chief verified the source's CPU defaults, disconnected objective router and inference-only adapters directly. The readiness order identifies concrete implementation owners and a proposed campaign supervisor module; it does not pretend the notebook/runner already exists.

Review corrections include a hard real-CUDA E0 gate; immutable per-decision archive cutoffs; a single authoritative campaign supervisor with transactional reservations; and a same-data/same-time fixed-method proposer-successor control. These avoid confusing more training, fixture behavior or retrospective selection with RSI.

## Verification

Chief focused local command:

```text
python -m pytest tests/test_research_accounting.py tests/test_research_evaluation.py tests/test_research_master_m05_m06_m12.py -q -p no:cacheprovider -o addopts=
```

Result:62 passed in8.06 seconds, one non-failing warning from a test converting a gradient-bearing tensor to a scalar. No optimizer steps or accelerator experiments were run. This is not a full-suite, CUDA or learned-capability result.

The [design validator](validate_design.py) checked E0–E6 continuity,480 total wall minutes,960 provisioned GPU-minutes, E5 nested trial/training/overhead arithmetic, E1 arm/seed coverage,75 local links and the zero-local-training authorization. [Recorded result](design-validation-001.json): no errors, with canonical UTF-8/LF hashes of the primary packet files. Git whitespace validation is separate. No empirical campaign result exists.

Technical runtime references were checked against official Kaggle notebook/GPU documentation and PyTorch AMP examples; links are located beside the relevant instructions in experiment.md. Actual hardware/quota and installed runtime remain launch-time facts, not assumptions imported from those pages.

## Next dispatch

The external agent implements I01–I06, prepares source/data artifacts, delivers `notebooks/bramastra_k8.ipynb` and writes `engineering/reports/K8_BUILD/HANDOFF.md`. The owner launches the notebook. After the run, the chief reviews the actual result bundle and chooses which mechanisms to keep, reject or replicate. No AGI, improvement multiplier or recursive-learning success is asserted by this design.
