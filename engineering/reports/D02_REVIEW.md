# D02 chief review — matched delayed-information teaching

Date: 2026-09-08. Status: experiment not yet accepted or completed; saved implementation under revision.

## Required corrections before execution

- The saved runner assigned three-bit worlds to seed 801 and four-bit worlds to seed 802. Those are different experimental conditions, not two seed replications. The revised primary study uses the same four-bit task population/configuration for both seeds.
- Teacher planning horizon must respect remaining real inquiries. At zero remaining queries, policy supervision is disabled. At one remaining query, both arms use one-step net information gain. Only states with at least two remaining queries may receive depth-two labels.
- Apply costs and legality consistently in both arms. Never include the scored target or an already used inquiry in teacher search.
- Hold histories, row order, observed labels, initialization, optimizer sampling and update budget fixed across teaching arms. The teacher-label tensor is the intervention.
- Freeze settings and source identities before development evaluation. Archive the actual imported implementation, not merely the new runner; verify that source did not change during execution.
- Report policy comparisons against coverage as well as random inquiry, per-family outcomes, world-clustered intervals and full teacher-generation/training/evaluation cost.

## Scope

This study tests supervised one-step versus depth-two teaching. Joint predictor/policy training remains a bundled mechanism whose effects need interpretation. It does not implement outcome-trained PPO, autonomous learning-method selection or broad self-improvement.

The exact parity construction establishes that delayed useful information exists. Only executed, correctly controlled training/evaluation can establish whether the learner acquires that strategy. Negative or inconclusive neural results remain valid outcomes.

## Saved-work audit after interruption

A chief verification of the current `teaching.py` on 12 deterministic episodes found zero terminal policy targets in both arms, and identical teacher scores in every state with one remaining query. Source identity and measurements are preserved in [the audit receipt](../../artifacts/bramastra/chief_d02_horizon_audit_20260908/audit.json), alongside the inspected source bytes. This is progress on horizon correctness, not a trained-model result. The saved `run_d02.py` was absent at this audit.

Focused tests produced four passes and one failure. The failing fixture marks query `(1,0)` as already used but leaves all four coefficient hypotheses in its posterior, then expects a positive gain from the remaining `(0,1)` query. With that unconditioned posterior, the remaining observations cannot determine the XOR target and zero gain is correct. The execution agent should preserve an unconditioned-zero check and add a separately conditioned-posterior case with one-bit gain. Do not change the teacher to satisfy the inconsistent fixture.

Next execution package: repair that fixture, restore the matched same-configuration two-seed runner, pass focused checks, then execute the bounded development comparison with immutable evidence. No current report should imply that comparison has run.

## Executed matched development comparison

The subsequent campaign `d02_matched_teaching_20260908_luna` completed both frozen seeds in 26.783 seconds of recorded CPU campaign time. The chief independently verified source snapshot hashes, unique raw evaluation keys, cross-arm labels/families and thresholded probability scoring. Each arm evaluated 23 development mechanisms with four targets each; confirmation remained unscored.

| Seed | One-step learned | Depth-two learned | Coverage, either predictor | Depth-two minus one-step |
|---|---:|---:|---:|---:|
| 801 | 61/92 | 61/92 | 61/92 | 0 percentage points |
| 802 | 63/92 | 65/92 | 62/92 | +2.17 percentage points |

Descriptive world-cluster intervals for the primary accuracy contrast were [0, 0] and [0, 5.43] percentage points respectively. The first degenerate interval describes zero observed paired correctness differences; it does not establish equivalence of the learning methods. Brier differences (depth-two minus one-step, lower is better) were +0.004166 and -0.004145, respectively. Both teacher arms matched their seed's random and coverage control accuracies, but their learned inquiry results differed in seed 802.

Chief decision: retain this as a valid small development result, with no claim of reliable multi-step advantage. Only three development mechanisms belong to the threshold family. The joint predictor/policy intervention and uniformly collected histories limit attribution. No model promotion, accelerator allocation or AGI claim follows from these results.

Before expanding the experiment, measure teacher action disagreement on initial states, imitation accuracy on those states, and final predictor accuracy conditional on informative versus uninformative query sequences. These distinguish a weak intervention, failure to imitate, and failure to use acquired information. That diagnostic design must precede any larger training campaign.

The saved run's immutable source remains the authority for its results. Subsequent runner failure/deadline regression tests are infrastructure hardening and must not be represented as checks executed before this historical run.

On 2026-09-09, the chief ran `.venv/Scripts/python.exe -m pytest tests/test_research_inquiry_teaching.py -q --basetemp=.codex-test-tmp-d02-chief-final -p no:cacheprovider`: **8 passed in 9.14 seconds**. This covers duplicate/mismatched pair rejection, zero-deadline timeout, injected training failure and mocked source-identity mutation, in addition to the teaching checks. The current runner also checks deadlines around evaluation/reporting and returns a nonzero CLI exit for failed or timed-out runs. These are cooperative checks, not an operating-system hard wall-time limit. W04 remains partial; the next assignment is `work_orders/D02_DIAGNOSIS.md`.
