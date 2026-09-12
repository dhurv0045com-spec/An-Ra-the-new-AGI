# B2 completion contract

The build is complete when the following behavior is demonstrated and documented. Statistical improvement is a later question; operational readiness is a device/data-specific claim.

| Deliverable | Required proof | Does not establish |
|---|---|---|
| One integrated learner | All profiles resolve to the same model/trainer interfaces; tiny real forward/backward | General intelligence or large-model fit |
| Public-only information | Codec invariance, reset/isolation and schema negative tests | Adversarial process isolation |
| Usable data path | Local manifest -> deterministic packed batches; real denominator and cursor identities | Sufficient production corpus supply |
| Configurable objectives | Full-loss parity; valid masked/offset training; pair-loss tiny examples; ordinary full-vocab generation | Better transfer from optional treatment |
| Controller | Deterministic transitions, frozen criteria, pool firewall and stateful restore | Learned autonomous method improvement |
| Persistence | Fresh-process next-update equivalence; interrupted-publication recovery | Live TPU/remote certification |
| Inference and scoring | Actual responses/actions; exact+EOS and goal-sensitivity results recomputed | High accuracy from a tiny smoke |
| Integrated command path | Prepare/train/resume/infer/evaluate/package with one config/data identity chain | Permission to start long training |

## Readiness reporting

Report separate booleans/statuses for implementation, CPU smoke, local GPU smoke, real-data supply, target-device validation and scientific qualification. Do not collapse them into one “ready” flag. A passed CPU run with no production corpus should say `LOCAL_SMOKE_PASSED / DATA_NOT_READY / TARGET_DEVICE_UNVERIFIED`.

Fixture-based end-to-end integration is the build acceptance target. Without an operator corpus, the real-corpus path is accepted only for truthful validation and DATA_NOT_READY reporting; production readiness stays blocked. This external data requirement does not prevent completing the implementation.

For each optional mechanism record four separate fields: implemented, unit_tested, enabled_in_smoke, scientifically_qualified. Set scientifically_qualified=false unless an actual qualifying experiment supports the claim.

No existing receipt transfers automatically to new source. Record the source commit or dirty-source snapshot used for each check. Exact binary hashes may differ across precision/backends; use a declared numerical criterion where appropriate, never invent bit equality.

## Resource ledger

Reserve at least 40 optimizer updates and 120 seconds of the CPU smoke allowance for fresh-process resume comparison before any other learned smoke. If that comparison uses the optional GPU session, reserve the same updates/time there. If insufficient budget remains, deliver the precise missing check instead of claiming complete.

One authoritative session ledger tracks cumulative CPU learned-smoke seconds/updates and GPU smoke seconds/updates across all helper agents and retries. Limits from README apply to the whole assignment, not independently to each subagent. Deadline handlers write partial receipts. A budget limit is a stopping boundary, not an excuse to silently reduce the model and report the original profile tested.

The owner's 100-million-plus implementation-token request is recorded. Provider usage is reported only when available; token volume is not a completion criterion or a reason to repeat work. No new training-token budget is inferred from it.

## Final report template

1. Branch, worktree, commits and push result.
2. Implemented modules and integration behavior, with exact CLI commands.
3. Focused tests and one bounded smoke: source/data/config identities, runtime, counts and failures.
4. Fresh-process restore result and comparison scope/tolerance.
5. Optional mechanisms: implemented, enabled, tested or still unqualified.
6. Data/hardware blockers and exact operator-supplied inputs required for later training.
7. Scientific claims explicitly supported by existing evidence; no invented benchmark results.
8. Actual measured time/tokens if available; otherwise unavailable.

Use `engineering/reports/B2/HANDOFF.md`. Preserve failed-run evidence. A precise partial handoff is preferable to a fictional complete build.
