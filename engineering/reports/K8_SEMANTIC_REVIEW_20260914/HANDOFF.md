# Chief handoff: semantic review, launch protection and executor contracts

Reviewed source: 5c0567f. Date: 2026-09-14. The chief audited all phase handlers and delegated only the bounded readiness module/tests to Luna; the chief reviewed and integrated those changes. The owner’s role split remains: chief owns contracts and acceptance, implementation agent completes the real executors.

## Changes

- Added campaigns/readiness.py with explicit reviewed E1–E6 blockers and required evidence. It is a transparent acceptance disposition, not a capability detector. Current readiness is false.
- Wired runner._preflight to reject the known incomplete implementation for E0 and full mode before creating a ledger or acquiring a lease. No training, handler or dataset preparation logic was otherwise changed.
- Added tests for readiness and both owner-facing launch paths. Preparation/validation remain accessible; no bypass option was introduced.
- Added a reproducible zero-learning semantic probe, compact receipt, source review and detailed real executor contracts. Updated engineering entry points and the external-agent prompt.

## Verification

From the BRAMASTRA worktree:

```powershell
$env:PYTHONPATH='.'
python engineering/reports/K8_SEMANTIC_REVIEW_20260914/probe.py
python -m pytest tests/test_research_k8.py tests/test_research_k8_readiness.py tests/test_research_k8_launch_gate.py -q -p no:cacheprovider -o addopts=
python -m pytest tests/test_research_k8_readiness.py tests/test_research_k8_launch_gate.py -q -p no:cacheprovider -o addopts=
python engineering/experiments/K8_20260913/validate_design.py
git diff --check
```

Semantic probe: E5 reports one committed update with no training calls; E6 completes without payloads; publish_checkpoint API absent. These characterize reviewed source, not learned experimental outcomes.

Combined tests: 20 passed, 2 test-setup errors due to access denied on the shared pytest temporary directory. Changed the new launch tests to use their own uniquely allocated TemporaryDirectory; reran readiness/launch tests: **5 passed in 0.26 seconds**. The unchanged 17 existing K8 tests passed in the combined run. All 22 selected tests are therefore covered by passing results across these runs; no claim is made of a single clean 22-test invocation. No optimizer updates or accelerator work occurred.

Design validation: no errors, 119 local links, unchanged 480 wall minutes/960 provisioned GPU-minutes. Diff whitespace check passed. Large historical suites were not rerun.

## Current limit and next action

This delivery protects launch and specifies missing implementation; it does not implement the complete learner, run Kaggle or establish AGI. Do not claim phase readiness from imports, fixture receipts or generated counts. The implementation agent must follow [EXECUTOR_CONTRACTS.md](EXECUTOR_CONTRACTS.md), finish real parent/target/evaluation/RSI/payload paths and return K8_REAL_EXECUTION_20260914/HANDOFF.md. Chief acceptance can then revise readiness based on code and local integration evidence before owner GPU execution.
