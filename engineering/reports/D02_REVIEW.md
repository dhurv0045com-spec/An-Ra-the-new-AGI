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
