# Cognition foundation: chief response to ecc5953

This is the current dispatch. It specializes O04/O05 of the [operational learner packet](../next_phase_20260914/README.md); the other O01–O10 obligations remain open until their evidence is accepted. Read this file, [DESIGN.md](DESIGN.md), and [ACCEPTANCE.md](ACCEPTANCE.md), then implement. Do not substitute another proposal for the requested production repairs.

## Decision

**The cognition foundation is not code-ready for the owner experiment.** The source has made useful progress: a real episode interface, model-versus-double origins, real observation/imagined-node separation, explicit context refusal, and production phase connections now exist. These are infrastructure gains, not evidence of learned cognition. The operational handoff's overall code-ready claim is not accepted by this review.

Reviewed source: `ecc5953`, including `84554fa`. This is a focused cognition review, not a fresh acceptance of all ten operational packages. Existing launch blocking remains in force. The older readiness manifest is a historical chief disposition; its old wording is not evidence that recently added code is absent.

## Reproduced findings

Run `python engineering/cognition_foundation_20260914/probe.py` with the repository on PYTHONPATH. It imports the actual cognition implementation, uses capture doubles only at the model boundary, and performs no training. On ecc5953 all six diagnostic flags are true:

| ID | Source behavior | Consequence |
| --- | --- | --- |
| C01 | `_compact_history_entry` maps both variable and value to `v`; observations x=7 and y=7 render identically | Evidence loses its subject before the learner sees it |
| C02 | `admit_observation_evidence` uses event kind as entity; `mark_conflicts` compares unequal payloads without temporal scope | Different variables conflict, and ordinary changes over time conflict |
| C03 | `ModelWorldModel` serializes goal/action/depth, omits history, then slices the prompt to 64 byte tokens | Observations cannot condition its predictions; action identity may also be truncated |
| C04 | The second planning depth receives the original state | This is not a conditional two-step rollout |
| C05 | First root expansion spends the remaining node budget on its children | With eight legal actions and eight nodes, only the first root is scored |
| C06 | `_planner_prediction_gap` takes the first depth-one node without matching the executed action | The reported gap is not reliably chosen-action calibration; established by source inspection |

The sixth executable flag checks root starvation separately; C02 has two executable counterexamples. Further source issue: `_compact_evidence_record` stringifies and slices values at 120 characters, and admission also slices serialized observations. Calling this lossless is incorrect. Fix the encoding contract, not just its description.

Five existing operational cognition tests passed locally: trace integrity, imagined-state separation, invalid-action accounting, explicit truncation, and negative controls. These checks did not catch the semantic failures above. Keep their useful coverage and add adversarial behavioral tests. The pytest cache permission warning did not affect their execution. No optimizer updates or GPU runs occurred.

## Direct response to the implementation agent

Your boundary work is useful. Your next deliverable must demonstrate that information changes predictions and that predictions change decisions for the right reason. A correctly typed trace can still represent a broken learner. Do not classify a production model call as learned competence, a class named world model as a working world model, or a second loop as depth-two planning.

Own `bramastra_lab/research/cognition/episode.py`, relevant live-environment observation contracts, E2 adapters/metrics, shared rendering consumers, their focused tests, and a new evidence report. Coordinate shared codec/data changes before editing concurrent agents' files. Exclude unrelated branches, old evidence, allocation changes, and readiness bypasses. Work in the BRAMASTRA worktree. Use Luna or Sol only for bounded independent reviews where helpful; the lead remains responsible for integration.

Deliver C01–C06 together with train/inference parity and budget accounting from the design. Return `engineering/reports/COGNITION_FOUNDATION_<unique_run_id>/HANDOFF.md` using the repository handoff template. Include source and parent identities, commands, precise outcomes, model/double origins, remaining blockers, and links to immutable small receipts. Never overwrite a prior report. Push only your own reviewed files and report the commit.

The intended workload is a substantial connected engineering package, approximately 3–5 useful hours when prerequisites are available. Neither time nor token consumption is an acceptance criterion. Do not consume compute to satisfy a duration target.

## Scope and experiment decision

Keep the existing K8 phases, treatments and owner-launched maximum 480 minutes on two T4 GPUs, with training stopping at minute 450 and 30 minutes for export. No local optimizer updates; the earlier CPU ledger is unchanged. CPU semantic tests and backward-only checks are sufficient for this repair stage. Random core initialization remains required.

The creative extension in DESIGN is a research hypothesis, not a new mandatory campaign arm or a claim of novelty. First repair and qualify the baseline. The chief will accept code readiness only after production-consumer evidence, then the owner can perform the separately budgeted GPU qualification. Passing this packet alone does not clear unresolved E1/E3/E4/E5/E6 obligations or establish AGI.
