# Agent handoff — Gandiva cognition follow-up

## Assignment and status

- Agent / role: Chief engineering integration.
- Work order and revision: Cognition privacy and diagnostic-quality follow-up; `719bcbcf5b84e80900e87f21a0cea2f49d550ff1` on `Gandiva`.
- Starting commit / worktree: `9856eae7`; `bramastra-build-worktree`.
- Status: **Partial.** This closes focused privacy-test and probe-validity gaps. It does not accept the complete F1–F6 cognition foundation or the full K8 owner campaign.
- Compute: no GPU runs, no optimizer updates, and no checkpoint or dataset writes. Wall-clock effort was not tracked.

## What changed

`tests/test_research_cognition_planning.py` now checks both the `ModelWorldModel`
input filter and the actual episode path. In the twin-world test, the public goal,
received history, action set, and seed are identical; the hidden correct answer
changes from `north` to `south`. The environment outcomes differ while the
pre-observation model prompts stay byte-identical.

`engineering/cognition_foundation_20260914/probe.py` now exercises an actual
depth-two prediction. The previous probe used eight legal roots and eight nodes,
which spent the entire budget on depth-one calls, then compared two root states
and reported a false C04 defect. The revised probe makes a two-root, three-node
search, checks that the second-depth state contains the first imagined
transition, and separately confirms eight-of-eight root coverage. The original
`engineering/cognition_foundation_20260914/BASELINE.json` was left unchanged.

## Acceptance criteria

| Criterion | Status | Evidence / limitation |
|---|---|---|
| Unreceived top-level private and evaluator fields do not alter a world-model prompt | PASS | Direct regression in `tests/test_research_cognition_planning.py` |
| Actual run-episode prompts are invariant across paired hidden-world twins | PASS | Same test; opposing hidden answers produce opposing outcomes |
| C01–C06 current-source smoke diagnostic observes no listed defect | PASS | `probe_v2.json`; diagnostic only, not a substitute for the acceptance suite |
| Cognition planning tests | PASS | 11 tests passed |
| Cognition-focused unittest discovery | PASS | 50 tests passed, including the planning module |
| RSI integration tests | PASS | 7 tests passed |
| Full cognition-foundation F1–F6 production acceptance | PARTIAL / NOT COMPLETE | Remaining required negative cases, family/mode traces, and whole-packet chief review are not established by this follow-up |
| Owner two-T4 runtime gates G01–G04 | NOT RUN | Requires the owner's Kaggle session |
| Learned cognition, recursive self-improvement, or AGI | NOT ESTABLISHED | No optimizer update or experiment was run |

## Verification and reproduction

Runtime: local CPU, Python unittest and the read-only diagnostic. From the repo
root in PowerShell:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -B tests/test_research_cognition_planning.py
python -B -m unittest discover -s tests -p 'test_research_cognition*.py'
python -B -m unittest discover -s tests -p 'test_research_gandiva_rsi_cognition.py'
python -B engineering/cognition_foundation_20260914/probe.py
```

The first command reported 11 passed; the second, 50 passed; the third, 7
passed. The diagnostic receipt records `optimizer_updates: 0`, one actual
depth-two call, all eight roots, and six false defect flags. There is no model
checkpoint or learned artifact associated with this work.

## Risks and next action

The paired-world check covers the current `run_episode` → `BoundedPlannerAdapter`
→ `ModelWorldModel` input path, but only a representative tiny environment. It
does not by itself establish every environment-family information boundary or
all F1–F6 requirements. Continue by auditing those requirements against the
current production consumers and add targeted tests for any uncovered public
field, type, context-budget, training/inference parity, and calibration cases.
Keep E0 G01–G04 and the registered from-scratch GPU campaign separate from
CPU fixture evidence.

## Chief review

This follow-up is accepted as a focused test/diagnostic correction only. Broader
cognition-foundation readiness and scientific claims remain unaccepted.
