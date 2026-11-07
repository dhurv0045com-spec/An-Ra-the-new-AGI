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

### 2026-11-07 | ChatGPT | V6 export-retry safety fix

- Audited V6 Kaggle notebook against its generated source and observed CI evidence.
- Removed the unnecessary T4 requirement from read-only, source-bound ZIP export retry; GPU training and recovery gates remain unchanged.
- Replaced stale draft-only notebook footer with the accurate owner-run qualification state and added a regression contract.
- No GPU training or Kaggle environment was executed; source remains pending live hardware qualification.
