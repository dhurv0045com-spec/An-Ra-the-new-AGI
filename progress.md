# Progress register

Append an entry after each meaningful change, upgrade or execution. Use the local date and AI name, with at most five nonempty lines per entry including its heading. Multiple entries per day are welcome. Record actual outcomes and distinguish implementation, validation and experiments awaiting execution.

### 2026-11-07 | Codex | V6 campaign foundation

- Implemented raw-evidence data generation, controlled training and the original 15 diagnostic panels on the existing V5 foundation.
- Added two-T4 worker scheduling, multiple seeds, conditional model scaling and independent action replay.
- Added checkpoint recovery, incremental saving, failure preservation and verified results-ZIP export.
- Preserved historical evidence; committed `6273b0dd`, `1cb3f377` and initial notebook publication `9dec5c0e`.

### 2026-11-07 | Codex | Four additional experiments

- Added S16 effort budgets, S17 symbol remapping, S18 stale-memory interference and S19 query-cost robustness to the same notebook.
- Added per-condition recovery, raw replay, summaries and charts without additional training arms.
- Repaired P14 tensor/checkpoint identity checks and strengthened diagnostic receipt verification.
- Pushed implementation `6ad49799` and notebook/evidence publication `ca7d46f6`; notebook pins `6ad497997f728d8161a42f7525e1dfb91ffe6d22`.

### 2026-11-07 | Codex | Engineering qualification

- Linux CI passed 213 tests plus seven subtests; closing Windows suite passed 212 tests plus seven subtests, with one symlink-privilege skip.
- Reverified 78 historical evidence hashes and 1,659 checkpoint components; checked notebook schema, disabled defaults and pinned-source reproduction.
- Saved engineered-control traces for effort, permutation and stale memory; the separate tiny QUERY/EOS correctness check scored 4/4 on its training prompts.
- No GPU or research training campaign ran; full launch still requires live T4 canaries and fresh-VM recovery. These checks establish mechanics, not learned cognition.

### 2026-11-07 | Codex | Progress register

- Created this dated register and added the future logging convention to root `AGENTS.md`.
- Documentation-only change; checked formatting and entry lengths without rerunning model tests.

### 2026-11-14 | Codex | Hitchhiker mechanism audit and experiment design
- Reviewed the supplied proposal against V7.17 source 0c9cb284 and its committed owner report; documented the older hitchhiker source base.
- Added research/hitchhiker/CONCLUSION_AND_EXPERIMENT_DESIGN.md with A-H critiques, matched diagnostics, training contrasts, falsifiers and an implementation prompt.
- Checked source references and document formatting; no code implementation, checkpoint inference or training ran. Proposed mechanisms and capability conclusions remain pending execution.

### 2026-11-14 | Codex | Hitchhiker documentation publication
- Preserved the user's proposal verbatim as Markdown and linked it with the experiment design from README.md and context.md.
- Recorded branch/source ancestry, audit findings, evidence limits, and the implementation sequence; all new experiments remain pending.
- Verified attachment-copy hashes and documentation formatting/links before committing for the requested branch push.
