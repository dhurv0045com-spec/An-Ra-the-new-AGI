# Observatory methodology

## Scope and safety

The scanner operates on one explicitly supplied Git repository. It uses `git for-each-ref`, `git worktree list --porcelain`, `git status --porcelain=v2 --branch`, local object queries, tree queries, commit metadata, and diffs. It does not fetch, pull, switch, checkout, reset, clean, stash, merge, rebase, prune, or run project documents. Submodules are read from the selected base tree; untracked nested repositories are not recursively crawled. Remote URLs are scrubbed before snapshot/report output.

The output directory is the only normal write target. JSON and Markdown are written with same-directory temporary files and atomic replacement. Refresh snapshot filenames include a capture identifier and are never overwritten. `scan` updates latest views without appending history; `refresh` appends history; `verify` writes only a verification report.

## Ref and worktree discovery

All refs under `refs/` are inventoried and sorted. A ref is a dossier candidate when it is a local head or non-symbolic remote-tracking ref. `origin/HEAD` remains an alias record because it is not an independent branch. Every ref records its object name, resolved commit when available, type, subject, dates, upstream fields, and missing-object state. Duplicate commit tips are grouped while each ref name remains present.

For each linked worktree, the tool records path, HEAD, branch/detached state, lock/prunable flags, and readable `porcelain=v2` status. User changes and Observatory-generated overlay entries are separated. A dirty worktree is an overlay; it is never included in committed-tip LOC. Unreadable worktrees remain in the inventory with an explicit error.

## Comparison bases

`config.json` names a clean project base. The initial base is `refs/remotes/origin/main`, chosen because it is a stable local remote-tracking tip and avoids the dirty active checkout. The choice is recorded with its SHA, rationale, and ambiguity. For each branch, the tool prefers the merge-base of its configured upstream and otherwise uses the project base. If no common commit exists, the branch is `UNCOMPARABLE`; no unrelated ahead/behind number is presented as meaningful.

Ahead and behind are counts relative to the recorded comparison base. Pairwise comparisons use a local merge-base, unique commit counts, stock differences, and changed document paths. Pairwise validity is limited when histories or purposes are incomparable. Family labels in `config.json` are interpretive and are not inferred from Git.

## Document indexing and authority

A committed file is indexed when its extension or path/name/content clearly identifies a specification, blueprint, architecture, contract, plan, roadmap, protocol, preregistration, amendment, readiness record, result, postmortem, audit, receipt, evidence ledger, runbook, notebook, operator guide, decision/open-question record, negative-results record, or historical/supersession record. For each record the tool stores ref, commit, path, blob SHA, byte size, title, date and date source when present, type, authority status, dependencies, bounded summary, headings, conflict/correction signals, and receipt/test references.

`canonical`, `current`, `historical`, `superseded`, `draft`, and `unclear` describe what a document claims or how it is positioned; they do not certify scientific truth. The scanner never executes prose. A document's claim ceiling is bounded by the evidence it cites. A missing external Drive result, checkpoint, raw receipt, or target-hardware run remains an explicit dependency.

## Source classification and line counting

Classification is path/extension based and deterministic. Categories include production source, tests, notebooks, documentation/specifications, build/configuration, generated code, generated artifacts/receipts, data, binary, vendor/dependency, and excluded. The default exclusions are Git internals, deleted-remnant object trees, caches, dependency trees, virtual environments, and the Observatory's own output directory. Large blobs, model weights, datasets, and binary artifacts are not recursively loaded.

A committed-tree text line is `data.count(b"\\n")` plus one when the non-empty blob does not end in a newline. This explicitly counts a final non-newline-terminated line. Files above `max_text_blob_bytes` are counted by metadata but their line content is not loaded. Binary files are counted by file/byte totals and never assigned synthetic source lines. Notebooks are separate: raw JSON lines and parsed cell counts are reported, while notebook cells are not production source.

The exact tip stock is a tree measurement. It is distinct from historical flow, working-tree edits, generated/non-source changes, and removals. A line removed is not interpreted as reduced capability or bad work.

## Commit flow, merges, renames, and daily arithmetic

Branch-unique commits are `base..tip` objects reachable from the tip but not the comparison base. Commit metadata is grouped by UTC calendar day. Ordinary commit deltas come from `git diff-tree --numstat --find-renames`; merge deltas are measured against the first parent and retained separately. Renames are detected where Git reports them, so a pure rename is not described as wholesale source creation/deletion. Binary numstat rows are counted as binary file changes without invented line counts. The tracked deleted-remnant object tree, Observatory output, and model/binary payload paths are excluded from flow queries for safety and are identified in stock/exclusion metadata.

For each UTC day the tool reports:

```text
source_lines_net = source_lines_added - source_lines_removed
source_lines_churn = source_lines_added + source_lines_removed
source_growth_percent = source_lines_net / source_size_start_of_day * 100
cumulative_net = cumulative_source_additions - cumulative_source_removals
cumulative_size_percent = cumulative_net / base_source_lines * 100
```

If a denominator is zero or unavailable, the percentage is `N/A`. Daily start stock begins at the base tree and is replayed in UTC commit-metadata order. Because side-branch commits and first-parent merge deltas can overlap, daily stock is labeled an estimate; exact base-to-tip stock change is separately computed from committed trees. `notebook_changes` means changed notebook paths, not a claim about cell count or scientific execution. Daily rows include commit/file counts, merge/ordinary counts, docs, tests, notebooks, generated artifacts, category maps, language maps, and representative changed paths. Recent 1/7/30/90-day windows are summed from these rows. First and last branch-unique dates, days since the last commit, and largest churn changes are retained.

Project-wide activity deduplicates commit SHAs across branch views. It does not pretend that divergent branch tips share one meaningful source-size denominator. Branch-specific activity remains visible even when a commit belongs to several branches.

## Evidence states and comparison rubric

The dossier emits all required labels: Proposed, Specified, Implemented, Locally verified, CPU-tested, GPU-qualified, TPU-qualified, Executed scientifically, Replicated, Supported within a bounded regime, Negative result, Inconclusive, Superseded, Blocked, and Unknown. Labels are non-scalar and can coexist. `NOT_VERIFIED` means no matching evidence was found; it is not a negative result. A status evidence record must resolve to an indexed document at the relevant ref or it is downgraded to low confidence.

The comparison rubric uses 5 for strong directly evidenced scope, 4 for good bounded evidence, 3 for mixed/partial, 2 for substantial blockers or stale authority, 1 for weak/contradicted, and `unknown` for missing evidence. Hardware, implementation, reproducibility, custody, scientific maturity, negative-evidence handling, operational readiness, blockers, stale-instruction risk, and unique contribution are separate dimensions. The tool does not rank by LOC, commits, last-modified date, model size, benchmark score, or document length. Recommendations are goal-specific and cite source refs/paths.

## Snapshots and verification

A refresh stores capture time in UTC and local time, tool/schema identity, ref tips, worktree status, committed-tree stock, activity, documents, comparisons, warnings, and the delta from the previous snapshot. A first snapshot cannot contain observed state from before capture. A refresh can detect ref movement, additions/removals, worktree status changes, and document blob changes. A force-update or deletion is shown as movement; retrospective commit attribution may consequently change.

`verify` builds a fresh read-only capture, compares it with `data/latest_snapshot.json`, checks schema/tool identity, and reports stale generated reports, refs, worktrees, and document blobs. Internal validation checks required fields, local SHA resolution, daily percentage arithmetic, project commit deduplication, dirty-overlay separation, category totals, and report-citation resolution. No experiment, training, inference, benchmark, sealed evaluation, or hardware job is launched.

## Known limitations

The scanner cannot reconstruct deleted commits, deleted branch names, previous dirty overlays, unavailable worktrees, external receipts, remote state, private data, or the owner's intent at a historical time. Git dates can be non-monotonic with topology. Document authority and branch purpose require human review. Some source and artifact boundaries are inherently interpretive, and a file's presence never proves that its claim is true. These limitations are reported rather than hidden.
