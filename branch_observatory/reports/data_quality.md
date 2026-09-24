# Data quality and validation

Status: **PASS** at `2026-09-24T21:08:59Z`.

| Check | Status | Details |
|---|---|---|
| `required_fields` | **PASS** | `[]` |
| `reported_shas_resolve` | **PASS** | `[]` |
| `daily_growth_percent_recomputes` | **PASS** | `[]` |
| `project_commits_deduplicated` | **PASS** | `{"count": 1208, "unique_count": 1208}` |
| `dirty_overlay_separate_from_tip_loc` | **PASS** | `[]` |
| `file_categories_sum_to_tracked_files` | **PASS** | `[]` |
| `inaccessible_worktrees_reported` | **PASS** | `[]` |
| `report_citations_resolve` | **PASS** | `[]` |
| `deterministic_unchanged_scan` | **PASS** | Two in-memory scans against unchanged refs/worktree state matched after removing capture timestamps and tool capture metadata. |

## Validation method

- JSON required fields, local object resolution, daily percentage recomputation, commit deduplication, dirty-overlay separation, category totals, and report-citation resolution are checked without experiments.
- Branch LOC is read from committed trees; worktree changes are separate. Observatory-generated files in the dedicated worktree are labeled generated overlay.
- Two scans against unchanged refs/worktree state are compared after removing capture timestamps and tool capture metadata; sorted refs, paths, classifications, and atomic writes make the remaining content deterministic.
- `make lint` and `make typecheck` could not run because `make` is unavailable; direct `ruff` and `mypy` modules are also unavailable in the runtime. No lint/typecheck success is inferred.
- No training, inference, GPU/TPU work, benchmark, sealed evaluation, fetch, merge, reset, clean, stash, or branch-switch command is part of the tool.

## Limitations

- The first snapshot cannot recover old dirty overlays, deleted commits, external receipts, or prior working-tree states.
- Commit dates are metadata; daily source stock rows are estimates when merge/side-branch flow overlaps.
- Diff flow excludes the tracked deleted-remnant object tree and model/binary payload paths listed in the tool implementation; their stock metadata remains visible where committed.
- Large artifacts, weights, datasets, binary files, and ignored trees are not recursively loaded.
- Document authority and mission profiles are human-reviewed interpretation, not automatic truth.

## Warnings

- No submodule entries were found in the selected base tree; untracked nested repositories were not recursively crawled.
- Remote refs were not fetched or refreshed; this snapshot uses only locally present Git objects and refs.
