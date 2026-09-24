# CONTAMINATION REPORT

## Status: WAITING_FOR_DATA

No materialized production corpus exists. Contamination scanning cannot
be executed against real benchmark suites until:
1. A production corpus is materialized with manifests
2. Evaluation benchmark suites are frozen and committed
3. The contamination scan is wired into the production data pipeline

## Existing capability
- `v5_data.contamination.scan_contamination(documents, benchmarks, ngram_order=8)`
- Fail-closed: any 8-gram collision flags the pack
- Previously used in T1D with `MINIATURE_EVAL_TASKS` as benchmarks
- Does NOT currently scan against real external benchmark suites

## Required before 500M launch
- Frozen eval benchmark suite committed to the repo
- Contamination scan executed against the full training corpus
- Zero-collision receipt
