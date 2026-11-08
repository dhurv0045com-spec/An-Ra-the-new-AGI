# V7.1 owner results

**Independent 2026-11-09 review:** [Review of V7.1 evidence, failure mechanisms, scaling gaps and suggested next steps](RESEARCH_REVIEW_2026-11-09.md). The review reads the committed metrics and source but does not replace or modify the owner-run artifacts.

Core run [20c80f66e76e4b589aed2053afbddffc](20c80f66e76e4b589aed2053afbddffc/RESULTS.md) contains the original analysis ZIP, owner final-status JSON, independently replayed compact metrics, trajectories, focused failures and a reproducible figure. Nine training jobs completed; 55 suite reports and four learning-control records replay successfully. Full checkpoint custody remains owner-required. Historical V7 results remain separate.

Blocked CE remains the baseline; intermediate supervision does not improve aggregate confirmation and regresses on stale-memory correction. Underlying neural causes remain unresolved. The next paired diagnostic dataset and execution limitations are documented in `research/v7.1/NEXT_EXPERIMENT.md`.
