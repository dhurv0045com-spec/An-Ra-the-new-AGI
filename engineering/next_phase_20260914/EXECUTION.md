# Execution packages O01–O10

The [architecture](ARCHITECTURE.md) and [readiness criteria](READINESS.md) define acceptance. Packages are substantive engineering units, typically 3–5 hours where prerequisites exist; time is an estimate, not a quota. No package authorizes local learning. Use one integrator and bounded Luna delegation with exclusive file ownership. Shared interfaces are merged before downstream edits.

| ID | Outcome | Dependencies | Owned paths / integration boundary |
|---|---|---|---|
| O01 | Authorized learner-session lifecycle | none | phases/types.py, new session adapter; integrator owns supervisor/trainer interface changes |
| O02 | Complete optimizer windows and checkpoint lifecycle | O01 | learning/k8_trainer.py, ProductionOps update/checkpoint methods, focused integration tests |
| O03 | Frozen protocol from E0 calibration | O01–O02 | calibration adapter, runner/worker protocol loading, launch manifest |
| O04 | Live cognitive episode kernel | O01 | new cognitive session runner and environment adapters; reuse canonical renderers |
| O05 | Matched workspace/planner/control evaluation | O04 | E2 executor, planner/workspace adapters, evaluation receipts |
| O06 | Parent-based tool acquisition and retention | O02,O04 | E3 streams/tool execution/retention; independent data compiler ownership |
| O07 | Actual migrated architecture treatment | O02,O05 | E4 handle migration and execution-cost integration; shared model edits via integrator |
| O08 | Real budgeted method-trial service | O02,O04 | new trial executor, archive transactions, E5 trial adapter |
| O09 | Learned proposer and successor comparison | O08 | E5 three-block scheduler, proposer training/capture, confirmation comparisons |
| O10 | Complete notebook qualification and evidence export | all | readiness dispositions, notebook/runbook, E6 export and final handoff |

## O01 — Bind authority to the learner

Pass a reservation record from the actual supervisor into the worker and every newly created/restored/forked training session. Attach it to the trainer through the supported admission API. Reject missing/mismatched job, device, phase, source or deadline. Restoring a model does not grant a new update allowance. Use exact parent checkpoint records and writer fences. Local acceptance must drive the real phase entry point to a no-op optimizer boundary and prove missing authority fails while a compatible simulated reservation reaches the boundary. No successful real optimizer step is allowed locally.

## O02 — Make a complete window executable

Connect the entire training path including pair rows, real targets and microbatch partitioning. Add named-gradient and disabled-forward checks using unequal lengths/counts. Exercise actual checkpoint save/load with a random-initialized, non-updated model and complete session state. Test wrong config, lost cursor, changed source, stale writer and sibling mutation. Do not replace state validation with a string hash field that is never recomputed.

## O03 — Calibrate and freeze once

Build the E0 profiling code for the actual K8 configuration and heaviest active arms on both workers. Apply the existing update-target and confirmation-inventory selection rule before learning outcomes can influence it. Persist exact selected settings and source/data/device identities. Resume consumes that frozen artifact; runner defaults must not masquerade as calibrated targets. Test with injected timing samples locally, including insufficient capacity, one slower GPU, changed hardware and interrupted E0. The actual timing and six-update resume evidence happen only in the owner allocation.

## O04 — Execute observations and actions

Use real finite environments with reset/step/verify interfaces. Route model decisions through legal action parsing and execution. Implement explicit terminal/truncation/invalid-action results and cost accounting. Model responses may be deterministic doubles locally, but the environment and trace reducer must be real. Check that imagined results cannot become real observations and that omitted tool calls cannot produce success receipts. Provide a minimal complete episode trace for each family.

## O05 — Compare actual cognitive modes

Construct matched case groups and adapter selection from the frozen E2 protocol. Restore B and A/control checkpoints explicitly. Run the same cases under each assigned treatment, with separate state and unchanged real-action budgets. Change goal before rendering; deliver real contradictions and complementary queries. Verify which adapters/model checkpoints were called. Local negative controls replace a scorer with a constant, remove a required observation and reorder independent case execution. Report whether each change is correctly reflected, without asserting every perturbation must reduce success on every fixture.

## O06 — Train on tool receipts and protect earlier skills

Compile generated tool trajectories into targets only from allowed execution histories. Run the exact T0/T1 sampler/objective combinations. Require held-out/protected exclusion by canonical ID before batching. Check actual output artifacts with independent verifiers. Return realized new/replay counts and per-family retention outcomes. Local tests use real disposable tool artifacts and deterministic learner outputs; no stored answer may stand in for a performed tool action.

## O07 — Train the architecture being reported

Ensure the migrated handle owns the gate parameters and actually routes token/action/value/world prediction through its gated computation. Check parent equality at zero gates, appropriate nonzero-path gradients, shared storage, optimizer uniqueness and packed-segment isolation. Persist changed architecture identity and compare real execution cost. Keep E4 independent of E3. Reuse existing tests, expanding only to missing production consumers.

## O08 — Implement measured trials

Build a real TrialRequest -> restore/fork -> recipe application -> update loop -> query/protected evaluation -> checkpoint -> TrialResult path. Inputs identify actual support/query examples, not synthetic labels. Trial time includes the prescribed work and cannot leak into the next slot. Use real counters and retain failed trials. Local tests replace learning at its boundary but exercise exact phase scheduling, parent equality, recipe dispatch and archive commits. A production handle with a valid simulated reservation must reach the real trial path, not unconditional refusal.

## O09 — Make RSI selection learned

Implement the existing 12/12/6 task blocks and 45/45/90-second trial caps within their 135-minute allocation. Train P0's method-token objective from the measured training archive. Capture the decoder's method choice; apply it to P1's actual learning session and M0 to P_fixed. Use the same successor data/time. Freeze all confirmation choices before trial outcomes. A host majority/argmax is allowed only as an explicitly named teacher/control. Counterexamples must reject substituted methods, mutated anchors and future outcomes. All local doubles remain fixture evidence.

## O10 — Deliver an experiment-ready build

Execute the complete production orchestration locally with non-learning expensive-operation substitutes and real data/checkpoint/environment boundaries. Verify cumulative slot timing with simulated clocks, process cleanup and resume, not by sleeping for eight hours. Ensure E0/full cannot start with missing implementation or incompatible inputs. Export actual non-updated test payloads through the full artifact pipeline and reload them in a fresh process. Correct all handoff claims; classify code-ready versus GPU-qualified versus experimentally supported.

Produce the final operator runbook and `engineering/reports/OPERATIONAL_LEARNER_20260914/HANDOFF.md`. Chief readiness disposition changes only after the actual acceptance evidence passes. No owner approval question is needed for routine implementation; accelerator launch remains the owner's action within the existing allocation.
