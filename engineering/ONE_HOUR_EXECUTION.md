# H01 — Repair the data-to-action path in one focused execution session

**Historical timebox; included repair recipe only.** The owner has superseded the one-hour stopping condition with [FINAL-K8](FINAL_EXPERIMENT_EXECUTION.md). Complete these repairs as applicable, then continue through every final requirement; the final architecture/consumer contract governs conflicts.

**Assignment: change production code now.** Start a 60-minute engineering session. Read this file and AGENTS.md, then begin edits. You do not need to read the historical blueprint packets or ask the owner to prioritize. Time allocations below guide execution; finish early if the acceptance criteria pass. At 60 minutes deliver tested work and exact remaining failures; never claim unfinished work complete or wait idle to fill time.

## Required outcome

The learner must receive distinguishable public inputs and action candidates, and its compact terminal action must retain its typed payload through the real policy adapter and environment. Complete all three fixes below and their regression tests. This is a code implementation assignment, not a documentation/review assignment. Do not return only a diagnosis, another plan, import repairs, or the old 30–50-hour estimate.

Starting review: ba038c6. That push fixed several episode/test issues, but compiler.py still slices inputs to six tokens and candidates to eight bytes, and the compact action codec cannot encode a real submission with a payload. The prior diagnostic currently aborts on that submission; an aborted diagnostic is not a passing result.

## Ownership and starting commands — minutes 0–5

Own `bramastra_lab/research/campaigns/phases/compiler.py`, `bramastra_lab/research/cognition/episode.py`, a new `tests/test_research_h01_execution.py`, and your new handoff directory. Read relevant consumers in ops.py, k8_scoring.py, sequences.py and k8_live.py. Do not edit scorer/device fixes already integrated by the chief unless your regression identifies a concrete fault. Do not edit other agents' dirty work or old reports. If another live agent owns an overlapping file, resolve that exact ownership conflict; continue independent assigned files meanwhile.

One existing test-file repair is also assigned: `tests/test_research_k8_foundation.py` has a missing `torch` import in its F6 integration test. Add the needed import if still missing; preserve its assertions. This is required to execute the test, not evidence that F6 behavior is correct.

Run `git status --short` and `git log -3 --oneline`. Record start time and commit. Run the executable acceptance checks in `engineering/h01_acceptance.py` with PYTHONPATH set to the repository root. They are deliberately failing regression contracts at the reviewed baseline, not a training experiment. Run only those checks first; do not spend the opening session running the entire repository suite.

## Fix 1 — Preserve decision inputs and action content — minutes 5–25

In `compiler.compile_channels_for_row`, replace `batch.input_ids[0][:6]` with an explicit complete public state for the decision whose target is being compiled. For the existing first-history-action channel, the prefix contains the public goal and no received future feedback. Build it from the shared renderer/codec, not from supervised answer tokens in the batch. Keep returned channel keys/API compatible. The action/value prefix must change when a relevant public goal changes and must remain identical when only answer labels or future observations change.

Replace every action `encode_text(... )[:8]` with complete canonical candidate serialization. Preserve every declared candidate and legal mask; reject context overflow explicitly before model execution. Do not substitute one candidate merely to make distinct candidates fit. Program's genuinely single-candidate path may stay a declared degenerate control for this patch; do not call it discriminative learning. Keep objective weights and numerator/denominator conventions unchanged. Verify world-channel conditioning stays compatible with `world_transition_token_loss`; avoid introducing duplicated state content that overflows context. A full train/inference architecture migration is queued separately and is not a prerequisite for removing these information-destroying slices.

Acceptance: inspect-x and inspect-y have different candidate tokens; full identifiers survive; public-goal changes alter the prefix; changing target answer/next-feedback does not change that decision prefix; oversized state/candidate rejects rather than truncates. Add a backward-only test that distinguishable candidates can produce different live scores. Do not perform optimizer steps.

## Fix 2 — Preserve typed terminal action payloads — minutes 25–40

Keep the existing `encode_action_code(action, legal_actions)` and `decode_action_code(code, legal_actions)` APIs. Preserve `{ "a": index }` for exact legal actions. For the public generic submit template, support exactly one typed payload with compact keys: `b` maps to Boolean `answer`, `i` maps to string `item`, `n` maps to integer `value`. Thus rule submission can be `{"a":3,"b":true}`, inventory `{"a":3,"i":"item_561"}`, and program `{"a":3,"n":45}`. Serialize without whitespace.

Validate index type/range, payload type, exactly one payload key, and compatibility with submit. Reject bool where integer is required, unknown keys, multiple payloads, payloads attached to inquiry actions, and extra input action fields that would otherwise be dropped. Existing complete candidate actions remain losslessly supported. Restore the registered 24-byte-token action response envelope rather than the in-progress eight-token constant; refuse oversized encodings explicitly without slicing. Do not enlarge campaign budgets or prediction envelopes.

Acceptance: construct each real live environment, reset it, obtain an oracle action only inside the test, encode/decode it, execute the decoded action, and verify correctness. The oracle must never enter production policy input. Test all three families and malformed payloads. Enumerating legal templates without real payloads is insufficient.

## Fix 3 — Connect the codec to actual policy execution — minutes 40–50

LearnedPolicyAdapter and WorkspacePolicyAdapter currently call the full-action parser. Route compact outputs through the validated codec using the current legal-action set, while retaining the existing full-action JSON compatibility. Retain model origin and generation IDs through both success and parse failure. The input presented to a code-generating model must contain the public canonical action-index mapping and payload schema; changing that mapping must change the rendered model input. Reserve context and response space explicitly; do not silently truncate when adding the map. No teacher fallback.

Acceptance: a model double returning a compact inventory/program submission must succeed through `run_episode`, the real adapter, codec and environment. Assert action type, payload, call count and origin. A malformed compact code must become a bounded invalid-action outcome, not a default guess. A full JSON action must still work. These are mechanics tests; random or scripted model outputs do not establish trained cognition. Candidate-head migration and broader learning-interface alignment remain H02 work, not a reason to leave this concrete consumer disconnected.

## Verification and delivery — minutes 50–60

Run `python engineering/h01_acceptance.py`, then `python -m pytest tests/test_research_h01_execution.py tests/test_research_k8_foundation.py -q -p no:cacheprovider` in a subprocess with a 90-second timeout. Set OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1. If a foundation test still hangs, terminate only your test child, record its name and fix the affected assigned path; do not blindly rerun it. Do not skip a required H01 check, weaken assertions, or patch fixtures to avoid the production consumer.

Create `engineering/reports/H01_<unique_id>/HANDOFF.md`: starting/final source IDs, changed functions, exact commands, pass/fail/skip counts, the three fix statuses, actual measured elapsed time, zero optimizer updates, and any remaining blocker. Keep this report concise. Commit and push only your scoped files. Do not stage user changes, weights, data or previous reports. The existing chief-owned readiness gate stays blocked until broader integration acceptance.

If H01 finishes early, first complete missing edge-case coverage. Then proceed to H02's first code task only if time remains: correct E5 `_train_method_selection` to bind an actual JobInput or the correct validated mapping API, with a real bindable-trainer no-step test. Do not begin the complete RSI redesign in leftover minutes.

## Resource and completion rules

No local optimizer updates or GPU training. Random initialization remains mandatory. Preserve the owner's future single two-T4 campaign (480 minutes, training stop 450, export 30). CPU source checks, model forward/backward diagnostics and tests are allowed. Use Luna/Sol for an independent bounded lane if useful, with exclusive file ownership; no mandatory delegation overhead. A failed test is a repair task, not permission to end after writing another plan. Finish H01's scope or identify the exact criterion still failing. Do not claim the repository or AGI is complete when only this execution slice is complete.
