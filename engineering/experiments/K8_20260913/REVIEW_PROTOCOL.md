# Review protocol: make weak implementations fail visibly

This is required evidence for [I01–I06](READINESS.md). It supplements the design with integration and counterexample checks, not a new benchmark. Use small deterministic fixtures and simulated clocks locally. No additional local learning is authorized. GPU-dependent assertions belong to E0 within its existing cap.

## Evidence map

In the K8_BUILD handoff, provide one row per seam below: production caller and callee, input identity, output identity, focused check, observed result, and unresolved execution environment. Link exact code symbols and compact receipts. If evidence is a fixture, label it. Do not replace the row with “all tests pass.”

| Seam | Required observation | Deliberate defect that must be detected |
|---|---|---|
| Preparation -> renderer | Public tokens and target masks match inference conventions | Inject a hidden generator field or shift a target span |
| Router -> trainer | Every eligible configured term reaches backward | Detach or omit one active term |
| Trainer -> parameter groups | Intended named groups receive appropriate gradients | Remove an action head from the optimizer inventory |
| Restore -> next batch | State and next input identities continue correctly | Alter cursor, scaler state or source identity |
| Checkpoint -> executive | Real scorer is called with the declared checkpoint and public history | Replace scorer with a silent constant fallback |
| Executive -> tool | Action and receipt IDs bind actual verified outputs | Return a claimed success without the required output |
| Architecture -> optimizer | Shared blocks registered once; gates retain state | Clone shared blocks or omit a gate from restoration |
| Proposer -> successor | Generated method controls the actual update recipe | Change dispatch to fixed M0 while keeping the proposal label |
| Confirmation -> archive boundary | Choices precede outcome availability | Insert current-task outcomes into proposal context |
| Supervisor -> recovery | Time and reservations survive crashes | Restart with a fresh clock or duplicate a completed job |

These counterexamples may be small test doubles or deliberate perturbations of inputs. They must exercise the production validation seam. Do not create a second mock validator merely to reject your own fixture. Avoid a broad mutation-testing framework unless the repository already provides one; a few decisive counterexamples are sufficient.

## Integrate in dependency order

1. Inventory current implementations and map unresolved I01–I06 criteria to owned files. Preserve already-correct work.
2. Build the tiny vertical slice described in [CHIEF_TO_AGENT.md](CHIEF_TO_AGENT.md). Resolve shared schemas before parallel edits.
3. Complete device routing and objective normalization. Then connect data, executive and tools through those interfaces.
4. Integrate architecture migration and proposer dispatch on the same trainer. Do not fork a separate optimizer implementation for E5.
5. Exercise the supervisor through simulated phase completion, timeout and restart. Verify notebook cells invoke these modules without redefining logic.
6. Prepare source/data artifacts and dry-validate CLI arguments and generated manifests. Hand off actual commands with GPU-only gates explicitly pending.

## Review failures without hiding them

Classify each failure as an implementation defect, invalid experiment, valid negative result, insufficient budget or missing environment. Keep its run/test identity. An implementation defect is repaired and rechecked with the smallest relevant test. A valid negative result is reported; it is not repaired by changing labels or selecting a favorable seed.

Before marking a package done, inspect its diff for unused configuration, unused outputs, duplicated training logic, broad exception handlers, silent fallbacks, implicit CPU allocations, and mutable global state shared by treatment arms. These are prompts for source review, not proof that every occurrence is wrong. Explain any intentional exception at the actual boundary.

For each local check record whether it performed forward, backward, optimizer updates or accelerator work. This keeps a backward-only wiring check from being misreported as a trained-model result. If a check fails because GPU hardware is absent, do not relabel a CPU fixture as GPU verification.

## Readiness disposition

Use separate dispositions for code readiness and scientific claims. Code readiness may be accepted locally with explicitly pending E0 hardware checks, provided the full campaign is gated on those checks. Scientific claims stay unmeasured until raw campaign evidence exists. A notebook file, successful import and design validator are necessary packaging evidence, not training readiness by themselves.

The final handoff must list exact remaining owner actions and expected output paths. Include dataset generation/reconstruction instructions and hashes, source identity, launch command, allocation identity policy, resume behavior, export destination and how to recover a failed run. Never leave the owner to guess whether pressing the full-run cell will repeat E0 or reset the allowance.
