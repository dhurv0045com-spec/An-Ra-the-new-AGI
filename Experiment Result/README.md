# Experiment Result

Permanent research-results archive for **AN-RA / Elendril** on the `Elendril` branch.

This is a **results folder only**. It is not an instruction to Codex and does not define or commission any new experiments.

## Saved results

- **[V6 — Results of all 19 registered panels](V6/RESULTS.md)**: 15 primary evaluations, four exploratory probes, 14 training-arm results, measured failure patterns, limitations, and provenance.

- **[V7.15 — Six-job S8 scale results](V7.15/3981e34a2ffe4b4b9528b4b4d588883b/RESULTS.md)**. Per-seed comparisons with S3, replay verification, failures and the next diagnostic phase.

- **[V16 — First owner T4×2 mechanism screening slice (H2B-NORM / seed 7)](V16/RESULTS.md)**. Real Kaggle run: custody PASS, both pipeline lanes PASS, job COMPLETE (2,368 updates, loss 5.684→0.757). Final free decoding base **60/112** (learning curve 42→52→60), row-reversal exactly **0.5000**, all hop cohorts **0/16**; trained policy far below the hand-written public-algorithm ceiling (112/112). Includes the analysis-summary report-writer defect found in the run (epoch-1 mislabeled as final) and its source fix; curated numbers in [METRICS.json](V16/METRICS.json).

## Filing rules for future research

Create a new subfolder per campaign or run (for example `V7/` or `YYYY-MM-DD-run-name/`). Save actual observations and evidence before conclusions. State whether each panel was **passed, failed, not qualified, skipped, interrupted, or exploratory**. For every failure distinguish (a) observed metric, (b) tested falsifier, (c) plausible explanation, (d) unresolved alternatives, and (e) raw artifact/source locations. Include dataset, notebook and source hashes when available. Never call an unexecuted panel a failed experiment or claim a causal root cause without a discriminating test.

**Raw-data retention:** The V6 report summarizes the uploaded 11-file analysis archive; its entire 70 MB raw arm-results report and multi-GB checkpoints remain outside this GitHub folder. Preserve the original Kaggle export for full reproduction.
