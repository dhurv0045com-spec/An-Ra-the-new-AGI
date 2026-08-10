# V4 Intelligence Runtime Audit

Scope: the local V4-SFT inference path, cognition, verification, memory,
retrieval, tools, and agent control as implemented in the repository. This is
an implementation audit, not an AGI claim.

## Findings before this change

- `cognition/self_correction.py` contained a useful model-agnostic correction
  loop, but the local checkpoint chat route never called it. It therefore did
  not improve actual conversations.
- `memory/memory_router.py` already exposed hybrid, episodic, journal, graph,
  and short-term retrieval with experience-ledger events. Automatically
  activating durable memory in the prototype would have mixed unreviewed
  context into answers, so it remains outside the canonical chat path.
- `verification/registry.py` is the shared verifier dispatch and evidence
  chokepoint. Exact symbolic and structural checks can establish narrow facts;
  the open-ended heuristic cannot establish factual correctness.
- `agents/plan_act_verify.py` provides fail-closed action verification, while
  the older broad agent loop and tool surfaces remain operator-gated. The local
  checkpoint app correctly did not silently grant tool or filesystem powers.
- Existing cognition modules such as epistemic tracking, CEC/CDSE, reasoning
  budgets, and proof memory are separate pilots. File existence is not evidence
  that the checkpoint benefits from them.

## Implemented improvement

The local app now has an opt-in **Verified deliberation** mode backed by
`cognition/deliberation.py`. It uses the existing SFT generation path, local
session memory, symbolic generation evidence, and experience ledger. It does
not introduce a second model, memory database, verifier registry, or telemetry
stream.

The controller adds explicit understanding, provenance-labelled retrieval of
user-provided session facts (never prior model output as evidence),
a deterministic plan, bounded candidate generation, scope-labelled
verification, one or more bounded revisions, abstention, and evidence
persistence. Candidate count, revision count, retrieval count, verifier calls,
generated tokens, and stage deadline are all capped. Direct mode and the
environment hard-off switch provide immediate rollback without touching the
checkpoint.

## Evidence and limits

Focused controller and local-runtime tests prove orchestration, budget stops,
abstention, symbolic-evidence use, session provenance, hard-off behavior, and
reuse of the experience ledger. They do not prove that the current 181M
checkpoint becomes more accurate. Promotion remains blocked until a matched
checkpoint-running evaluation shows higher task success after charging the
extra tokens and latency.

Durable memory writes, external retrieval, tool execution, autonomous agents,
self-modification, and long-term learning remain outside this mode. Each needs
its own permission, verifier, contamination, rollback, and evaluation gate.
