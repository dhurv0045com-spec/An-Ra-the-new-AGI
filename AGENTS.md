# Repository instructions

## Progress register

After each meaningful change, upgrade or execution in this repository, append an entry to root `progress.md` before the final handoff or commit.

- Include the local date (`YYYY-MM-DD`), AI name (for example, `Codex`) and a short activity title.
- Limit each entry to five nonempty lines, including its heading. Multiple entries on the same date are allowed.
- Describe actual work and verification concisely. Distinguish implemented code, engineering controls and observed research results; state pending execution when relevant.
- Preserve previous entries. Record later corrections as new entries rather than silently rewriting history.

## Curated owner results

In `Experiment Result/V7/` and `Experiment Result/V7.1/`, commit only focused Markdown reports and essential curated JSON metrics/provenance. Keep owner-supplied ZIPs, full raw receipts, plots, duplicate CSV exports and raw intake directories outside Git. Maintain reusable analysis code under `tools/`.
