# Experiment Result

Permanent research-results archive for **AN-RA / Elendril** on the `Elendril` branch.

This is a **results folder only**. It is not an instruction to Codex and does not define or commission any new experiments.

## Saved results

- **[V7.17 — S20 optimization and seed investigation](V7.17/f443d3f4141b4163bbc130d46ebd7395/RESULTS.md)**. Eight verified runs; lower LR improves all three matched seeds, while novel chain tasks and alias robustness remain failures.

- **[V6 — Results of all 19 registered panels](V6/RESULTS.md)**: 15 primary evaluations, four exploratory probes, 14 training-arm results, measured failure patterns, limitations, and provenance.

- **[V7.15 — Six-job S8 scale results](V7.15/3981e34a2ffe4b4b9528b4b4d588883b/RESULTS.md)**. Per-seed comparisons with S3, replay verification, failures and the next diagnostic phase.

- **[V7.16 — S20 scale results](V7.16/51f3aeb7cb604c6d8c80f6ae4cd9f3ff/RESULTS.md)**. Six completed jobs, three-size comparison, verified replay, failures and next diagnostic steps.

## Filing rules for future research

Create a new subfolder per campaign or run (for example `V7/` or `YYYY-MM-DD-run-name/`). Save actual observations and evidence before conclusions. State whether each panel was **passed, failed, not qualified, skipped, interrupted, or exploratory**. For every failure distinguish (a) observed metric, (b) tested falsifier, (c) plausible explanation, (d) unresolved alternatives, and (e) raw artifact/source locations. Include dataset, notebook and source hashes when available. Never call an unexecuted panel a failed experiment or claim a causal root cause without a discriminating test.

**Raw-data retention:** The V6 report summarizes the uploaded 11-file analysis archive; its entire 70 MB raw arm-results report and multi-GB checkpoints remain outside this GitHub folder. Preserve the original Kaggle export for full reproduction.
