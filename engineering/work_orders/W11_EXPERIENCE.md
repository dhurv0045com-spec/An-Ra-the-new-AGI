# W11 — Immutable experience storage and reproducible replay

**Status:** depends on W01 record interfaces; design can proceed from documented contracts. **Effort:** 3–5 hours. **Role:** data-systems engineer. **Compute:** CPU fault injection and small fixture datasets.

Read DATA_CONTRACTS.md and W01's accepted handoff. Own `bramastra_lab/research/experience/`, `tests/test_research_experience*` and `engineering/reports/W11/`. Do not change contract definitions without a chief-reviewed amendment.

## Deliverable

Implement append-only episode shards, complete-manifest publication, verified reads, global semantic deduplication and deterministic uniform/family-balanced replay. Make replay state independently serializable for W07 checkpoints and W06 consolidation.

## Engineering sequence

1. Define local shard/index layout against W01 records; publish only complete hash-verified shards.
2. Preserve prior committed manifests through interruptions and corrupt writes.
3. Build semantic split and source-union indexes that cannot double-count overlapping receipts.
4. Implement replay sampling with explicit family weights, RNG state, cursor and dataset identity.
5. Connect a small prototype-to-canonical episode conversion and replay continuation example.

## Acceptance evidence

- Repeated publication is idempotent; conflicting content under the same identity is rejected.
- Interrupted/partial writes are not readable as committed data, and older committed data remains usable.
- Equivalent semantic tasks cannot enter incompatible splits through different surfaces/shards.
- Repeated or overlapping source manifests do not inflate unique supply.
- Exact next-sample continuation survives process restart with the same dataset and sampler identity.
- Changed dataset contents, weights or replay policy require a new identity and cannot silently reuse old cursors.
- Uniform/family-balanced sampling agrees with independent small fixtures; no evaluation data enters replay.
- A bounded fault-injection/continuation report is usable by W06/W07 without a production database.

Do not implement a distributed object store or priority scheme in this packet. Prioritized replay is a later measured change with explicit sampling probabilities. Hand off real storage/restore behavior, not metadata that merely declares successful publication.
