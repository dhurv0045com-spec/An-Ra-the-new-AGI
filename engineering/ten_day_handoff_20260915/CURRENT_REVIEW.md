# Current findings and durable history

## What was checked on 15 September

The worktree was clean at inspection. The latest local commits are `5d335c1` and `58e517b`; origin was fetched. The first adds a final readiness report and small cognition/environment edits. The second edits the environment and repair handoff. The agent has done code work, but these commits do not complete FINAL-K8.

`engineering/reports/FINAL_K8/HANDOFF.md` and BUILD_READINESS.json both say readiness is false. The report estimates additional work; that estimate is not an external blocker or a reason to stop implementing. The following are directly visible in current source:

- `campaigns/phases/compiler.py:155` takes only the first six input tokens; candidate encodings at lines 185, 200 and 210 take only eight bytes. These discard task/action content.
- `campaigns/phases/e5.py:666–670` assigns P1 and P0 the same captured choice and assigns P_fixed M0 instead of independently decoding trained successors.
- `campaigns/readiness.py:157` still returns `ready=False`. There is no implemented verify-build command in the inspected CLI.

Line numbers refer to 58e517b. Recheck named functions after edits. These are source findings, not a new model experiment. No test suite, checkpoint run, optimizer step or GPU experiment was performed in this review.

## Correct the acceptance accounting

Do not overwrite the old report. In the next unique report distinguish component existence from requirement completion. F03 says PASS but its gap says the full data bundle has not been generated and held-out tools were not exercised. F05 says PASS while its gap says the router is not fully consumed by the trainer. F06 lacks verified training/inference parity. F07 says the workspace is not consumed by E2. F08 says the learned prediction connection is absent. F18 lacks actual E2/E5 statistical consumers. These notes contradict complete requirement acceptance.

Other gaps need finer classification. Unrun CUDA checks for F10/F12/F16 do not by themselves invalidate otherwise complete CPU-verifiable code, but they also do not establish it. Missing actual-model tool/replay checks, full payload checks, concurrent-ledger tests and hardcoded notebook paths are locally addressable work where the environment permits. An estimated module count or passing helper test is insufficient evidence.

The statement that generating the full synthetic data requires a Kaggle session is not an established hardware limitation. Build/validate the offline bundle without optimizer updates. If a concrete local memory, disk or dependency limit prevents full preparation, measure and report it; provide a reproducible CPU preparation route and do not spend the GPU allocation on avoidable data generation.

F6/O05 failures are not dismissed merely by calling them test-setup gaps. Correct the setup to create valid real parent records, then run the intended consumer assertions. A correctly refusing parent gate is useful coverage; it does not prove that valid parents can execute the full phase. Previously deselected liveness tests must run in bounded subprocesses after repair. Do not carry permanent exclusions forward.

## What the preceding work established

This is a summary of evidenced milestones, not a claim that ten full days of engineering were measured:

| Commit | Date | Durable contribution |
| --- | --- | --- |
| acede7c / 373a9fe | 14 Sep | Operational learner contracts and follow-up review |
| 84554fa / ecc5953 | 14 Sep | Agent episode/session/trial implementations and provenance changes |
| 5020533 | 14 Sep | Chief cognition counterexamples and foundation repair order |
| 99b1dee | 14 Sep | Cross-component review: task identifiability, consumer mismatch, budget and RSI issues |
| 79dcb18 / ba038c6 | 14 Sep | Agent foundation code/tests and episode/measurement repairs |
| 29df755 | 14 Sep | Scorer device/context/mask fix plus focused H01 assignment |
| 0db7495 | 14 Sep | Complete FINAL-K8 specification, 24 requirements and conditional build-readiness authority |
| 5d335c1 / 58e517b | 15 Sep | Agent false-readiness handoff and small subsequent repairs |

Keep the useful accumulated decisions: random core initialization; real public observations; teacher/oracle provenance; separate actual and imagined states; complete optimizer windows; actual parent/architecture identities; independent proposer/successor decisions; immutable allocation accounting; and export with real payload restoration. The recurring failure to eliminate is disconnected consumers hidden behind individually passing tests.

## Decisions the next agent does not need to ask again

FINAL-K8 sections 2–21 define model geometry, losses, data counts, cognition, tools, gated reuse, RSI, schedule and statistics. Section 22 conditionally authorizes implementing evidence-backed build readiness without another chief code-edit round. Local backward-only checks are permitted; optimizer steps are not. Scoped Git publication is authorized. Historical one-hour/3–5-hour estimates do not limit the complete delivery.

A complete build may be ready for owner launch with explicit E0 hardware checks pending. It may not be called ready with missing production code, missing full input bundle, unresolved relevant failures or an always-false startup gate. Training results and AGI claims require evidence beyond implementing the experiment.
