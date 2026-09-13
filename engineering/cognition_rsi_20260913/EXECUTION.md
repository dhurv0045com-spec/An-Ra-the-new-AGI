# M19–M24 cognition and recursive-improvement orders

These extend M00–M18 in the master packet. The combined dependency graph is in `program.json`; its external dependencies refer to the master's existing package IDs. The integrator retains ownership of shared config, trainer, wrapper, CLI, checkpoint and status files. Use Luna for bounded independent implementation/review with explicit ownership. No optimizer updates are authorized by this extension.

Each package owns corresponding focused tests and `engineering/reports/COGNITION_RSI_20260913/Mxx/` evidence. Target roughly 3–5 hours of useful implementation per package when prerequisites exist, without padding. A package may finish software against fixtures while remaining learned-unqualified. Integrating the final learned runtime still depends on closing the [current M00 findings F1–F6](../reports/B2_2_CHIEF_20260913/REVIEW.md).

## M19 — Grounded cognitive workspace

**Depends on:** M01, M04. **Owned paths:** new `bramastra_lab/research/cognition/workspace.py`, `beliefs.py`, workspace codec and schema adapters; shared codec changes by integrator.

Implement typed goals, public evidence aliases, belief status/support links, subgoal DAG, pending commitments and capability estimates. Implement finite-support reference belief updating as a declared diagnostic/teacher. Preserve uncertainty, contradiction and source ancestry. Provide deterministic context budgeting and episode reset through the existing memory/input adapter.

**Acceptance:** a hypothesis cannot become observed evidence by editing a confidence field; correlated copies retain a common source; contradictory evidence triggers a valid revision or model-mismatch result; zero-normalizer cases do not produce NaN certainty. Reject cyclic subgoals, missing evidence references and hidden metadata in rendered tokens. Serialize/restore workspace with identical next public input. Use no learned model for these checks.

## M20 — Learned executive and metacognitive decision interfaces

**Depends on:** M19, M02, M05, M07. **Owned paths:** new `cognition/executive.py`, `deliberation.py`, operation registry and objective adapters.

Connect typed operations to the shared action scorer and argument decoder. Route cognitive actions to retrieval, prediction, planning, queries, tools, verification and submission through the actual public runtime. Implement the value-of-computation target interface, abstention semantics and explicit decision-origin/fallback reporting. Keep fixed and direct-answer controls.

**Acceptance:** candidate isolation and legal masks apply to cognitive operations; operation costs aggregate once into ResourceVector; a frozen fake-model decision changes the actual operation executed; malformed/unbounded/no-progress decisions terminate correctly. A capability summary derived from query/confirmation outcomes is rejected before rendering, including if retrieved from memory. Bounded backward checks establish supervision routing without optimizer steps. Pair a useful-check and useless-check fixture to verify target construction, but do not call a fixture-trained selector competent.

## M21 — Verified reasoning, abstraction and cognitive experience

**Depends on:** M19, M06, M09, M14. **Owned paths:** new `cognition/derivations.py`, `abstractions.py`, and cognitive-task preparation adapters.

Build independently checked subgoal/derivation traces for qualified structured tasks, belief-update training records and memory abstractions with source/counterexample links. A procedure includes preconditions and primitive-action cost. A candidate rule proposed from support cases is verified on distinct training cases before admission. Route accepted supervision into the same prepared-data and objective infrastructure.

**Acceptance:** invalid derivations and false completed subgoals do not receive gold labels; renaming preserves bindings while relationship changes alter the checker result; counterexamples restrict or retract a rule; source examples are distinct from rule-validation examples. Failed episodes retain valid prediction targets. No examiner labels enter the workspace or abstraction archive.

## M22 — Meta-episodes and learning-to-learn measurement

**Depends on:** M20, M21, M11, M12. **Owned paths:** new `metalearning/episodes.py`, `curves.py`, `comparison.py`, and objective/receipt adapters.

Implement support/query/retention meta-episode records, shared-start paired adaptation jobs, budget-grid learning curves, proposal-outcome experience and R0–R4 attribution. Build the dispatcher to the canonical trainer; do not add a private optimizer loop. In this phase jobs are validated and exercised with deterministic callbacks only.

**Acceptance:** support/query mechanism leakage rejects; parent/candidate starting-state mismatch rejects a method-only comparison; missing curve points follow the declared conservative rule; equal-budget and cost-frontier claims are distinguished. Changing proposal-origin or evidence-class fields cannot create authentic provenance. Failed candidate costs remain in the total. No meta-training or query evaluation is launched without an allocation.

## M23 — Model-origin learning-method proposals

**Depends on:** M22, M13, M15. **Owned paths:** new `metalearning/method_language.py`, `proposer.py`, `compiler.py` and archive schemas.

Implement the typed method AST, bounded interpreter/compiler and model-output adapter in RSI.md. Support a small declared set of objective/sampling/schedule/gradient-transformation expressions with explicit state. Preserve the original proposal text/AST, proposer checkpoint and compilation identity. Prepare syntax, measured-ranking and gain-calibration supervision from eligible method experience.

**Acceptance:** malformed types, nonfinite/domain-invalid math, prohibited inputs, arbitrary code/imports and unsupported state migrations reject before execution. Compiling the same method/config gives the same identity and bounded behavior. An external agent's invented patch is labeled external/assisted. A model proposal with a forged checkpoint reference or an AST differing from the controlled runner's parsed output fails origin validation. Transcript hashing alone cannot qualify arbitrary external output; test that rejection. The unchanged parent/no-change proposal is always representable. General source-patch execution remains a separate future work order.

## M24 — Recursive generation integration and final owner handoff

**Depends on:** M18, M23. **Owned paths:** new `metalearning/generations.py`, generation registry, operator adapters and combined handoff through integrator.

Connect workspace-driven failure summaries, a proposer checkpoint, compiled method, matched meta-episode jobs, independent comparison and chief publication into a generation chain. Reuse M12's transaction/fencing rules. The accepted successor must actually become the next proposer; a host substituting another model changes attribution. Preserve all attempts and keep fixture registries separate from learned parents.

**Acceptance:** three fixture generations prove chain linkage and successor dispatch, including reject, crash, retry and no-change paths; they are clearly not three learned improvements. A fake examiner, rewritten threshold, stale approval, omitted rejected-candidate cost, changed archive cutoff or wrong proposer identity cannot publish a learned parent. The operator can inspect cognitive state, decision origin, method diff and resource vector from one public workflow. The generation runner refuses learned execution without an allocation.

Write `engineering/reports/COGNITION_RSI_20260913/HANDOFF.md` covering M19–M24 and referencing the M00–M18 handoff. Include exact source/config/schema/test identities, capability status, remaining blockers and the next proposed learning-to-learn experiment. Update root/status/prompt links and push BRAMASTRA normally under owner authorization. No force push or unrelated files. A connected cognitive/RSI runtime is the build deliverable; actual cognitive competence and recursive learning benefit require separate measured evidence.

## Six decisive integration stories

1. New contradictory evidence revises a belief and changes the next selected action.
2. A goal switch preserves public facts but changes relevant subgoals and the final answer.
3. A useful verification operation is chosen in one task and skipped in an equal-cost distractor task.
4. A proposed abstraction admits a new supporting case and then handles a counterexample without hiding it.
5. A method candidate runs through identical-start meta-comparison with every attempted cost included.
6. An accepted fixture successor becomes the next proposer, while fixture evidence remains unable to update any learned parent.

Use actual public orchestration seams for these stories and label injected components precisely. As learned checkpoints become available under future allocations, replace test doubles one at a time and retain the same invariants. This is how the program advances from designed cognition to accountable learned cognition.
