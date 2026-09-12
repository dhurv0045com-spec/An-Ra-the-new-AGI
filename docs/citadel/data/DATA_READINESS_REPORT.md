# DATA READINESS REPORT

OVERALL VERDICT: FAIL

## Failed gates
1. DATA_NOT_READY — no materialized production corpus exists
2. No natural/code corpus loader — 65% natural / 20% code slice cannot be filled
3. Verified-cognition tokens consumed by production path: 0
4. Mixture allocation not enforced at consumption time
5. No production tokenizer artifact frozen
6. No production evaluation harness at token milestones
7. Near-duplicate detection not exercised on real corpora
8. Contamination scan not exercised against real benchmark suites

## What exists
- Citadel tiered arithmetic corpus: 6.4M unique rows, ~97 MB (T1D/T1E experiment data)
- Foundry pipeline: normalize → quality → exact dedup → near-dedup → tokenize → account
- Source registry: immutable data-source identity with lifecycle states
- Acquisition manifests: reproducible download resolution
- Near-duplicate detection: MinHash LSH + Jaccard
- Contamination scan: n-gram overlap
- Exact dedup: content-hash clustering
- Packing: multi-segment stream-fill with exact ledger
- Checkpoint transactions: staging/atomic/manifest/fence

## What doesn't exist
- Real natural text corpus (the 65% slice)
- Real code corpus (the 20% slice)
- Production tokenizer artifact (frozen 24,576 BPE)
- Milestone-crossing logic wired into the production trainer
- Multi-session checkpoint persistence
- Evaluation harness over frozen eval sets at token milestones
- Production training entry point wired to real data

## Data readiness ladder
Current state per source family: **DECLARED** (not MATERIALIZED)
Target state for 500M launch: **RUNNABLE**

## One data experiment before 500M
**Cognition mixture ablation (5% vs 15% vs 30%)** on the tiered arithmetic corpus.
This is cheap (2-8M tokens), uses existing data, and directly tests whether
the cognition fraction hypothesis is correct. It does not require external data.
