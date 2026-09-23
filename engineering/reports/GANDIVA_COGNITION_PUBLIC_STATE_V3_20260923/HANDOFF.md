# Gandiva cognition public-state v3 handoff

**Date:** 2026-09-23
**Branch:** `Gandiva`
**Parent revision:** `7e64c6d42743b7d317ffc5f851f8f21cddddf2e1`
**Scope:** shared training/inference cognition inputs and bounded memory retrieval.

## Result

The current implementation closes a concrete data-path defect: the live
renderer and the K8 trajectory compiler now produce the same versioned public
state, including ordered received history and explicit remaining budgets.
History is encoded without losing unknown fields, and exact K8 inspection
observations have a compact reversible form. The public-state protocol
identity is part of tokenizer/checkpoint identity and compiled-row provenance.

Memory retrieval now spends the actual serialized-event budget. It can pass
over an oversized high-ranked candidate and retain a smaller eligible one;
when the renderer must remove optional context, it retains higher-ranked
memories. Receipts count records actually rendered, not only records selected
by retrieval. Retrieved memory remains an auditable lexical reference
mechanism, not a learned memory system.

## Contract changes

- `bramastra-public-state/v3` is the shared training and inference protocol.
  The prompt carries a short `instruction` version tag; the full schema and
  key-code tables are bound by `public_state_identity()`.
- A history entry of the exact form `inspect(variable)` followed by
  `observation(variable, value)` serializes as a versioned typed tuple. The
  decoder restores both operation labels and both variable fields. Variable
  comparison is canonical JSON comparison, so `false` and `0` do not alias.
- All other history entries use a versioned presence mask or a whole-payload
  envelope. Known nested keys use aliases; unknown keys use pair lists so
  user keys cannot collide with codec fields. JSON field/value types survive;
  Python tuples are explicitly normalized to JSON arrays at the wire boundary.
- Workspace evidence uses a versioned outer envelope. Unsupported Python
  objects, non-string object keys, and nonfinite numbers fail closed.
- The common prefix includes instruction, goal, received history, and
  remaining action/call/node budgets. The compiler derives direct-policy
  teacher budgets from earlier actions. Its convention is versioned as part
  of the public-state identity.
- Memory content has a versioned prompt event. Structured JSON examples are
  supplied as JSON values, while prose stays text. Retrieval cost includes
  the role tag and envelope. The retrieval-rule identity records the new
  exact-event-budget behavior.
- Temporal evidence with an unknown valid time no longer acquires an invented
  same-time conflict or supersession relation. Explicit `false` remains
  distinct from numeric zero, and missing observation values remain distinct
  from explicit `null` through typed admission.

## Verification evidence

The validated K8 training bundle contains 49,152 rows. A CPU-only complete
decision-size sweep over every row reported:

| Check | Result |
|---|---:|
| Rows visited | 49,152 |
| Prompt compilation errors | 0 |
| Complete decision examples over 512 tokens | 0 |
| Maximum complete length, including answer and EOS | 464 |
| Median complete length | 336 |
| 95th percentile | 464 |
| Over-limit rows in rule-inquiry, inventory, program | 0, 0, 0 |

The size calculation uses the exact prompt-token result plus the UTF-8 answer
bytes and one EOS token. It does not instantiate the model or perform an
optimizer update.

The focused suite passed in one process:

```powershell
python -m pytest -p no:cacheprovider `
  tests/test_research_k8_foundation.py `
  tests/test_research_cognition_planning.py `
  tests/test_research_cognition.py `
  tests/test_research_cognition_runtime.py `
  tests/test_research_gandiva_rsi_cognition.py `
  tests/test_research_k8_operational.py `
  tests/test_research_tpu_100m.py -q --maxfail=1
```

Result: **141 passed, 24 subtests passed**. Python compilation and
`git diff --check` also passed. The suite was kept in one process after
parallel invocations showed test interference in a shared fixture.

## Recovery and worktree boundaries

The source work described here was built on top of parent `7e64c6d4`. The
user's existing changes in `notebooks/bramastra_k8.ipynb` and
`tests/test_research_k8_real.py` remain untouched and are excluded from this
implementation. A separate `stash@{0}` remains preserved on the older
`BRAMASTRA` base. It contains E2 and episode-kernel work (742 additions and
160 deletions); several of its behaviors are already present in Gandiva.
The stash was inspected but not applied or dropped because its episode file
overlaps the current renderer. Treat it as a recoverable reference, not as a
clean patch for this branch.

## Limits and next gates

This proves host-side serialization, context-fit, and memory-selection
mechanics for the checked bundle. It does **not** prove that a trained model
uses evidence correctly, generalizes to new tasks, performs recursive
self-improvement, or reaches AGI. The full F1-F6 cognition acceptance program
is not complete, and the reference memory index is still lexical.

No Kaggle session, TPU forward/backward, optimizer update, checkpoint/resume,
optimizer-state memory measurement, or sustained training run occurred here.
The separate Kaggle notebook remains a zero-update TPU preflight. Before a
100M training campaign can be called ready, the TPU path still needs a
cognitively meaningful sharded stream, update-level loss/gradient checks,
optimizer-window partitioning, peak-memory evidence including optimizer
state, atomic checkpoint/resume, and a bounded sustained run on Kaggle.
