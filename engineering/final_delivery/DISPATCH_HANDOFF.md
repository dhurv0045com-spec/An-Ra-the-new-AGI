# Chief handoff — complete experiment execution mandate

## Assignment and status

Owner request: write the complete remaining assignment so the implementation agent finishes the experiment build, including model, cognition, learning algorithms and architecture, rather than stopping after H01. Checked baseline: 29df755, synchronized with origin/BRAMASTRA; no newer implementation push at review start.

Dispatch/design integration: complete. Production implementation, GPU qualification and experimental results: not completed by this document update. Local optimizer updates and accelerator runs: zero. Total chief elapsed effort: not measured.

## What changed

Created FINAL_EXPERIMENT_EXECUTION.md, one self-contained implementation mandate covering F01–F24, and REQUIREMENTS.json, a machine-readable requirement inventory explicitly labeled as a contract rather than evidence. Updated all active entry points to FINAL-K8 and marked H01's one-hour stopping condition superseded. The full remaining scope is one continuous delivery; old packets are optional references.

The mandate specifies actual architecture and trained-head consumers, solvable task/label contracts, complete data and canonical splits, loss/window algorithms, evidence/planning state, real controls, tool retention, gated architecture, trial/proposer/successor order, allocation/calibration, resume/export, statistics and owner notebook. It also assigns the missing verify-build CLI and authorizes evidence-backed build-readiness maintenance without another manual chief-edit round. Runtime hardware and allocation qualification remain mandatory.

## Source audit and verification limits

The chief inspected readiness.py, the campaign manifest, model configuration, supervision/window contracts, calibration and experiment schedule. Readiness currently returns false unconditionally; this requires an implemented conditional replacement, not a documentation-only claim. The current prepare CLI defaults are smoke sizes and must not leak into the final full bundle.

A bounded read-only Luna runbook audit identified the actual CLI flags, hardcoded notebook source path, in-notebook data preparation, unpinned dependency mechanism and the E0/full dependency. It reported 36 passed and one failed test in its launch-gate/readiness/operational selection: the E2 matched double evaluation did not complete. This is attributed audit evidence, not an independently rerun full suite or GPU result. The final mandate explicitly requires its resolution.

This update verifies document links, unique requirement coverage, required budget constants and Git whitespace. It does not change production model/runtime code or clear current readiness. REQUIREMENTS.json cannot itself make the build ready; implementing verify-build and satisfying the real local assertions are assigned work.

## Acceptance for this chief task

- Full remaining scope written with no H01 stopping point: complete.
- Every delivery requirement has a stable ID and acceptance description: complete, 24 requirements.
- Hardware-only qualification distinguished from missing code: complete, four runtime gates.
- Active repo instructions agree on the final assignment: complete.
- Owner has a copyable execution prompt: final section of the execution document and the K8 agent prompt.
- Experiment implementation ready now: no; the assigned agent must execute and verify the mandate.

## Next action

Give the implementation agent the FINAL-K8 prompt. It should implement through a complete source-bound build report, published compatible input bundle and runnable owner notebook, then push the scoped release. Its final ready-to-launch claim must have complete F01–F24 evidence; actual E0 hardware qualification and scientific outcomes occur only when the owner runs the campaign.
