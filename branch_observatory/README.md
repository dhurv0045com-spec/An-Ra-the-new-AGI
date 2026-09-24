# An-Ra Branch Observatory

The Observatory is a read-only, refreshable cartography of the locally available An-Ra Git repository. It inventories refs, linked worktrees, branch purposes, specifications, evidence records, committed-tree size, historical change flow, ancestry, and goal-specific comparisons. It does not merge branches, fetch refs, run experiments, or turn activity into a scientific claim.

## Open the latest report

From this worktree:

```text
branch_observatory/reports/latest.md
```

The companion reports are `branch_index.md`, `branches/`, `specs_index.md`, `project_map.md`, `comparisons.md`, and `data_quality.md` under `branch_observatory/reports/`.

## Refresh

Run from the repository root or pass `--repo`:

```text
python branch_observatory/scripts/branch_atlas.py refresh
python branch_observatory/scripts/branch_atlas.py --repo "C:\path\to\An-Ra" refresh
```

`refresh` reads only local Git objects and refs, writes an append-only timestamped snapshot, updates `data/latest_snapshot.json`, regenerates reports, and rebuilds the daily CSV. It never fetches, pulls, switches, resets, cleans, stashes, merges, rebases, or edits another worktree.

`scan` performs the same read-only inventory and report generation without appending a historical snapshot. `verify` compares a new read-only capture with the last snapshot and writes `reports/verification.json`; it reports moved/added/removed refs, worktree changes, document blob changes, schema/tool mismatches, and stale generated files. `python branch_observatory/scripts/branch_atlas.py --help` documents the command line.

The default output is the directory containing this README. Use `--output` only for an Observatory output directory; generated files are written atomically inside that directory. Historical snapshot filenames are never overwritten.

## What is automatic

The tool automatically discovers all local heads, remote-tracking refs, tags, symbolic refs, linked worktrees, readable worktree status, configured remotes with scrubbed URLs, base-tree submodules, commit metadata, tree blobs, document hashes, source classifications, line stock, commit numstat flow, merge first-parent deltas, detected renames, daily UTC buckets, branch ancestry, duplicate commit membership, and comparisons to the configured base.

A configured base is recorded in `config.json`; the first capture deliberately uses `refs/remotes/origin/main` as a clean, portable base. The active checkout is not assumed to be authoritative. Each branch records its own upstream merge-base when available. A branch may therefore be behind, ahead, divergent, stale-looking, dirty, or incomparable without being assigned a universal quality score.

## What needs human judgment

`config.json` contains reviewed branch-family profiles, mission interpretations, claim-ceiling notes, comparison rubrics, recommendations, and glossary definitions. A future maintainer must review these when branches change. The scanner does not infer scientific success from a filename, document confidence, commit count, line count, model size, benchmark value, or prose. A status remains UNKNOWN or NOT_VERIFIED when evidence is missing.

Document indexing is deliberately conservative. A document is indexed when its path/name/content indicates a specification, plan, handoff, result, receipt, ledger, audit, runbook, notebook, readiness record, decision, or negative-results record. Authority labels are document-level classifications, not truth guarantees. A receipt reference is recorded as a reference; it is not automatically treated as validated scientific support.

## Size and growth

`source_lines` means production source lines in the committed tree. Tests, notebooks, documentation, generated code/artifacts, data, configuration, vendor/dependency trees, excluded caches, and binaries are separate categories. A line is a newline count plus one for a final non-newline-terminated line. Notebooks are counted as notebooks with raw lines and parsed cell counts; JSON notebook lines are not production source.

Daily history uses UTC commit metadata. The required fields include additions, removals, net, churn, start/end stock, growth percentage, commits, files, docs, tests, notebook changes, and generated-artifact changes. Zero denominators produce `N/A`. Daily stock is an estimate when commit metadata order and merge/side-branch flow overlap. Exact base-to-tip stock change is reported separately. Project-wide totals deduplicate commit SHAs while branch-specific views retain their own histories.

## History and snapshots

`data/history/snapshots/` is append-only. Each refresh records ref tips, worktree state, LOC stock, activity metrics, document identities, tool/schema identity, and differences from the previous snapshot. The first snapshot cannot reconstruct earlier dirty states, deleted commits, lost worktrees, external receipts, or branch names that were never present in local Git. Future refreshes can show observed additions/removals and ref movement; rewritten or force-updated history is shown as movement and may change retrospective attribution.

## Safety

The Observatory was developed in a separate worktree on `branch-observatory`. The active checkout and pre-existing linked worktrees are not edited. The tool uses subprocess argument arrays, skips large blobs above the configured limit, does not recursively crawl caches or the disk, does not load model weights, and does not execute repository prose. Remote URLs are scrubbed before display; document summaries also redact local paths, email addresses, and credential-like assignments. The output directory is the only normal write target.

## Reading the reports

Start with `reports/latest.md` for the executive summary, then use `branch_index.md` to choose a ref. Open a branch dossier for its mission, status labels, document index, LOC stock, activity, and claim ceiling. Use `comparisons.md` only within the stated goal and family; its recommendations are not a league table. Use `data_quality.md` and `verification.json` before relying on a refresh. See `methodology.md` for formulas, exclusions, merge handling, and limitations.
