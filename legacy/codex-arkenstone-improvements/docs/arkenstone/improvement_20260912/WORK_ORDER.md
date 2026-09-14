# ARK-HARDEN-20260912 work order

Owner request: inspect Arkenstone, substantially improve it, and produce a clear Markdown report. Baseline: origin/Arkenstone at 933d4f3. This is a bounded engineering pass, not a measured 10x cognition claim.

## Execution packet A: scientific runner correctness
Prerequisites: ARK-012 and ARK-013 PLAN.md, ARK-013 PREEXECUTION_ADDENDUM.md, COLAB MASTER_DISCOVERY_V6_PLAN.md, current runners.
Owned paths: experiments/ARK-012/run_ark012.py, experiments/ARK-013/run_ark013.py, tests/test_discovery_v6_campaigns.py, docs/arkenstone/improvement_20260912/RUNNER_HANDOFF.md.
Excluded: shared runtime, other tests/docs, historical plans/results, all other branches and nested repositories.
Acceptance: audit and correct false-positive or premature summary verdicts; preserve completed arm evidence before later failures; add focused synthetic/CPU tests for corrections and deterministic carry/firewall invariants. Preserve frozen scientific thresholds; document any ambiguity without fabricating preregistration. No training, paid compute, network or Git publishing. Use local Python with CPU torch; tests bounded to two minutes each. Evidence: handoff with commands/results, defects, limitations, source baseline. Shared worktree uses exclusive file ownership.

## Integration packet B: shared runtime and usability
Owned by primary: experiments/COLAB/discovery_v6_common.py, new runtime tests and diagnostic entrypoint, README.md, new clarity report and evidence under docs/arkenstone/improvement_20260912/.
Acceptance: isolated run outputs, atomic verifiable receipts, source provenance, no import-time output writes, failure-safe packaging, bounded CPU integrity preflight including exact fork replay; focused verification and report distinguishing code correctness from scientific findings. Do not launch long GPU experiments. Preserve historical evidence and plans.

Root ownership extension: ARK-011/run_ark011.py runtime helpers (lazy output directory, CPU optimizer snapshot isolation, continuation sampler). No frozen task/model or plan changes. Entry point: experiments/COLAB/run_discovery_v6.py. Budget checks apply at training-loss calls; a single in-flight update/evaluation or packaging may finish beyond the deadline. No claim of process-level preemption.

Execution preference update: at the owner's request, the remaining packet-A fixes and tests are assigned to Luna. Sol's initial implementation was stopped; Luna inherits only packet-A ownership. Root remains integration reviewer.

Final status: packets A/B complete for bounded local engineering acceptance. Primary reviewed Luna output, added focused integration guards/tests, and recorded final 50-test/CPU evidence in HANDOFF.md. Scientific/accelerator work remains unrun.
