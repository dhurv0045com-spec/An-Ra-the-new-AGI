# W01 chief review log

Date: 2026-09-07. Status: revisions requested on the first in-progress implementation; final acceptance pending.

The initial implementation was useful scaffolding but did not yet satisfy strict contracts. Passing serialization examples would not be enough to accept it. The chief requested these concrete corrections, which are part of W01 acceptance:

1. Map actual prototype outcome fields (`target`, `policy`, `probability`) to canonical target/arm/prediction fields. Reject missing required evidence rather than silently producing `unknown`/null predictions.
2. Reject nonstring JSON keys instead of coercing keys that can collide. Do not encode bytes through a sentinel representation that can collide with an ordinary JSON object.
3. Prevent caller-owned dictionaries/lists from mutating an allegedly immutable record after construction.
4. Validate direct constructors as well as deserialization: types, finite values, integer-versus-boolean distinctions, nonnegative counts/costs, nonempty identities and supported split names.
5. Validate nested transition/action/observation structure and task/episode/step consistency. Empty mappings and synthetic fallback step numbers must not create valid episodes. Reject events after termination/truncation.
6. Validate metadata-envelope subclasses' own required semantics; inherited nonempty generic identities do not validate weights, arms or decisions.
7. Include explicit, correct tensor byte order and reject unsupported representations clearly.
8. Hash the normalized semantic split inventory so input order does not change its identity; reject declared/assigned split mismatches.
9. Provide explicit schema/allowlist validation for public payloads. A blacklist of suspicious key names cannot establish that an observation is free of hidden state. Environment/schema provenance still requires independent review.

The execution agent owns implementation and regression tests. The chief will review results before marking W01 complete. Any remaining scope limitation must be visible in its handoff and the main status ledger.

## 2026-09-08 revalidation

The previous execution agent is no longer live. Its handoff claims completion with five focused tests, but inspection of the current source contradicts full acceptance. The adapter still expects `case_id` instead of actual `world_id`; public/action/transition and other nested payloads remain mutable; explicit public-schema validation is absent; checkpoint state identities remain incomplete; and several strict type, episode consistency, split normalization and promotion checks are unfinished.

A new Sol execution agent owns revision of W01 and its handoff. W02 is coordinating against the preserved TaskSpec/PublicObservation/Action field interfaces while validation is strengthened. No downstream package may treat the previous five-test handoff as certification of the full contract.

## Latest saved revision: chief verification

The primary Sol agent stopped on a usage-limit error. Its saved revision substantially addresses the nine findings above. Chief execution of `.venv/Scripts/python.exe -m pytest tests/test_research_contracts.py -q --basetemp=.codex-test-tmp-w01-chief-current -p no:cacheprovider` produced **19 passed in 8.14 seconds**. This includes adapting the actual discovery development rows and validating a downstream fixture dataset. The old handoff remains stale; its five-test completion statement is superseded by this review.

Two independently reproduced defects still prevent acceptance:

1. `Outcome(..., label=0.0, ...)` and `label=1.0` are accepted despite the integer-label contract. Equality membership in `(0, 1)` does not enforce integer type. Require the strict integer validator and add both regression cases.
2. A valid array enum fails after public-record freezing. For example, observable `{"x": [1, 2]}` with an array schema whose enum is `[[1, 2]]` becomes a tuple inside `PublicObservation`; `validate_public` then rejects it because enum comparison requires identical Python container types. Compare canonical JSON representations so supported list/tuple and dict/frozen-mapping representations retain JSON semantics, while boolean and integer values remain distinct. Exercise both array and object enums before and after record round-trip.

The next executor owns only W01 paths and should repair these defects, run focused verification, refresh `HANDOFF.md` and `validator_report.json`, and return for chief acceptance. Do not restart the entire package or rewrite the now-working contracts. W02 and W04 remain queued; no implementation agent is currently running.

## Chief acceptance of repaired implementation

Luna repaired both defects and added regressions. Independent chief verification with `.venv/Scripts/python.exe -m pytest tests/test_research_contracts.py -q --basetemp=.codex-test-tmp-w01-chief-accept -p no:cacheprovider` produced **22 passed in 4.87 seconds**. Source inspection confirms explicit integer labels and canonical JSON enum comparison. W01 is accepted for its bounded record/identity/schema scope. This does not certify environment privacy, storage transactions, checkpoint restoration or learning capability. Public-information safety requires the environment to call the reviewed schema validator; arbitrary JSON key names are not blacklisted.
