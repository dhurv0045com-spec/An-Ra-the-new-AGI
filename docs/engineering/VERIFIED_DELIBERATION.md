# Verified Deliberation Runtime

Status: implemented as an opt-in local V4-SFT runtime mode. It changes the
inference process; it does not change checkpoint weights or prove AGI.

The controller in `cognition/deliberation.py` follows one bounded sequence:

`understand -> retrieve -> plan -> candidate -> verify -> revise/abstain -> persist`

The local app exposes it as **Reasoning: Verified deliberation**. Direct mode
remains the default and is the rollback path. `ANRA_VERIFIED_DELIBERATION=0`
hard-disables the controller without changing code or checkpoint state.

Budgets are typed and capped: candidate count, revision count, retrieval
results, verifier calls, total generated tokens, and a soft wall-clock deadline
checked between generation stages. A single in-flight model call is not forcibly
cancelled because terminating a CUDA call is not a safe serving primitive. In
deterministic mode every generation uses greedy decoding and a stable
seed-derived sequence. No adaptive model state is persisted by deliberation.

Verification reports its scope. Symbolic answers may receive exact checking;
JSON receives structural checking; Python receives syntax checking; factual
answers can only receive local-session retrieval-overlap evidence; general
answers receive generation-integrity checking. The latter checks do **not**
establish factual truth. When the requested proof scope cannot pass inside the
budget, the controller abstains instead of returning the rejected draft.

The final gate, budget use, verifier scope, checkpoint hash, and answer are
written to the existing `runtime.experience_ledger` stream under
`verified_deliberation`. This deliberately avoids a second observability or
memory truth system. The public stage trace says `persist` only after the ledger
accepts the event and says `persistence_failed` when its fail-open writer cannot
store it. Rejected draft text is omitted from the public API trace.

Current boundary: retrieval is limited to provenance-labelled user turns in
the active local session; prior model outputs cannot become factual evidence.
Durable retrieval, tool execution, and agent action are
not silently enabled by this mode. They require separate permission and
verification gates before promotion.
