# W01 — Canonical contracts and identities

**Status:** ready to assign. **Effort estimate:** 3–5 engineering hours. **Owner role:** data/runtime engineer. **Compute:** CPU tests and small fixture generation only.

## Required reading and ownership

Read root AGENTS.md, engineering DATA_CONTRACTS.md, SYSTEM_ARCHITECTURE.md and EXECUTION_PLAN.md. Own `bramastra_lab/research/contracts/`, `tests/test_research_contracts*`, and `engineering/reports/W01/`. Do not edit other agents' modules or global package registration. Storage and replay belong to W11.

## Deliverable

Implement strict versioned records for task/public observation/action/transition/episode/batch and their content identities. Specify and validate the outer checkpoint/experiment/outcome/promotion metadata envelopes; backend payload behavior remains with W07/W08/W10. Build canonical identity calculation and semantic split validation. Add an adapter for the reviewed discovery prototype without asserting that its old artifacts satisfy stronger new contracts. Discovery outcome rows are a reduced exploratory format: missing canonical cost/failure/timing fields must be marked unavailable or supplied from actual run evidence, never fabricated.

## Engineering sequence

1. Translate documented fields/invariants into explicit validators; reject unknown or privileged public-observation fields.
2. Implement canonical JSON/tensor identities and a split inventory keyed by semantic content.
3. Implement strict record serialization/deserialization and explicit version rejection.
4. Supply fixtures for episode shards, replay descriptors and checkpoints to W11/W07; do not implement their storage backends here.
5. Demonstrate identity/validation compatibility and hand off fixtures to model/environment agents.

## Acceptance evidence

- Equivalent canonical records hash equally; changed bytes, dtype or shape change identity.
- Duplicate semantic tasks cannot cross splits even when their surface text differs.
- Missing/extra fields, nonfinite numbers, inconsistent episode order and private-state objects are rejected.
- Outer manifests reject missing payload identities and unsupported versions; storage interruption behavior is tested by W11/W07.
- Adding or altering a dataset changes its identity and is not silently accepted.
- Tests use independently constructed cases, including overlapping manifests and changed tensor bytes with unchanged descriptive metadata.

Produce a compact fixture dataset and validator report. Do not build a production database, sampler, distributed service or large corpus in this packet. Report any schema contradiction to the chief before editing shared specifications. Strict serialization, identities and meaningful validation are the outcome; a directory of placeholder dataclasses is insufficient.
