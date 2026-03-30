# Sovereignty Package — Self-Improvement Daemon

A transparent, user-level Python daemon that audits and benchmarks itself
every night, producing measurable proof of improvement (or regression) by
morning. Runs as your own user — no admin rights, no hidden installs,
no auto-boot hooks.

---

## What It Is

Sovereignty runs a 4-pass nightly pipeline on any Python codebase:

| Pass | Name | What It Measures |
|------|------|-----------------|
| 1 | Code Audit | Cyclomatic/cognitive complexity, docstring coverage, comment ratios |
| 2 | Dead Code Sweep | Unused imports, unreachable code, magic numbers, long/deeply-nested functions |
| 3 | Performance Benchmarks | 10 benchmarks (B01–B10) compared to last night's baseline |
| 4 | Nightly Report | Plain-English report with delta arrows, sparklines, and concrete action items |

A local REST API lets you query status, fetch logs, and queue tasks —
all token-authenticated, all on `localhost` only.

A watchdog monitors all threads and restarts stalled ones automatically.

---

## System Requirements

- **OS**: Windows 10/11, macOS, or Linux
- **Python**: 3.10 or newer
- **Dependencies**: `psutil` (see `requirements.txt`)
- **Admin rights**: NOT required
- **Network**: No external connections — API binds to `127.0.0.1` only

---

## Install in 3 Commands

```bash
# 1. Install the only dependency
pip install psutil

# 2. Run the installer (no admin needed)
python -m sovereignty.install

# 3. Start the daemon
python -m sovereignty.service start
```

---

## Try It First (No Install Needed)

```bash
python -m sovereignty.demo
```

Runs all 7 demo scenarios in a temporary directory. Nothing is written
to your system. Zero admin rights required.

---

## API Endpoints

All examples use `$TOKEN` — find your token in `C:\sovereignty_data\token.key`
(or wherever you set `SOVEREIGNTY_DATA`).

### GET /ping — Liveness check (no auth)
```bash
curl http://localhost:45000/ping
# {"status": "alive", "uptime_seconds": 3600}
```

### GET /status — Full daemon status
```bash
curl -H "Authorization: Bearer $TOKEN" http://localhost:45000/status
# {
#   "service": "running",
#   "uptime_seconds": 3600,
#   "last_improvement_run": "2024-01-15 02:04:31",
#   "last_improvement_result": "success",
#   "next_scheduled_run": "2024-01-16 02:00:00",
#   "threads": {"MainLoop": "alive", "Watchdog": "alive", "APIServer": "alive"},
#   "resource": {"cpu_pct": 0.2, "ram_mb": 38.1}
# }
```

### GET /log — Recent log lines
```bash
# Last 50 lines, INFO level and above
curl -H "Authorization: Bearer $TOKEN" \
     "http://localhost:45000/log?lines=50&level=INFO"

# All levels (DEBUG)
curl -H "Authorization: Bearer $TOKEN" \
     "http://localhost:45000/log?lines=100&level=DEBUG"
```

### GET /report — Nightly report for a date
```bash
# Today's report
curl -H "Authorization: Bearer $TOKEN" \
     "http://localhost:45000/report"

# Specific date
curl -H "Authorization: Bearer $TOKEN" \
     "http://localhost:45000/report?date=2024-01-15"
```

### POST /task — Queue a manual task
```bash
# Run garbage collection
curl -X POST \
     -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"task": "gc"}' \
     http://localhost:45000/task

# Run the full improvement pipeline right now
curl -X POST \
     -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"task": "run_full_pipeline"}' \
     http://localhost:45000/task

# Run only Pass 1 (code audit)
curl -X POST \
     -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"task": "run_pass_1"}' \
     http://localhost:45000/task

# Available tasks: run_pass_1, run_full_pipeline, gc, rotate_logs
```

### GET /task/{task_id} — Check task status
```bash
curl -H "Authorization: Bearer $TOKEN" \
     http://localhost:45000/task/YOUR-TASK-UUID
# {"task_id": "...", "task": "gc", "status": "done", "created": "..."}
```

### POST /shutdown — Graceful shutdown
```bash
curl -X POST \
     -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"confirm": "yes"}' \
     http://localhost:45000/shutdown
```

---

## Reading the Nightly Report

Reports are saved to your data directory as `nightly_report_YYYYMMDD.txt`.

```
Section 1 — Executive Summary
  Three-sentence summary. Health score: EXCELLENT / GOOD / FAIR / POOR.

Section 2 — Code Quality (Pass 1)
  Table of complexity metrics with delta arrows:
    ↑ = improved since last night
    ↓ = regressed since last night
    → = unchanged

Section 3 — Dead Code Findings (Pass 2)
  Numbered list of issues. Plain English. Never auto-modified.

Section 4 — Performance (Pass 3)
  Benchmark table. ⚠ REGRESSION = >5% slower. ✓ IMPROVEMENT = >5% faster.
  Baseline is only updated when ALL benchmarks pass or improve.

Section 5 — Recommended Actions
  Concrete numbered actions: what to do, why, which file, estimated effort.

Section 6 — Historical Trend
  7-night table + ASCII sparklines for complexity, issues, and regressions.
  ▁▂▃▄▅▆▇█  (low → high)
```

---

## Querying Historical Benchmarks

Benchmark results are stored as JSON:
```bash
# Windows
type C:\sovereignty_data\benchmark_20240115.json

# macOS / Linux
cat $SOVEREIGNTY_DATA/benchmark_20240115.json | python -m json.tool
```

Each file contains:
```json
{
  "results": [
    {
      "id": "B01",
      "name": "Heartbeat write latency",
      "unit": "ms",
      "median": 0.2341,
      "baseline": 0.2280,
      "delta_pct": 2.68,
      "flag": "unchanged"
    }
  ],
  "regressions": 0,
  "improvements": 1,
  "baseline_updated": true
}
```

---

## File Layout

```
C:\sovereignty_data\           (or $SOVEREIGNTY_DATA)
├── config.json                Config overrides
├── token.key                  API bearer token
├── heartbeat.json             Updated every 60 seconds by main loop
├── run_log.json               Record of every nightly run
├── audit_baseline.json        Pass 1 baseline
├── benchmark_baseline.json    Pass 3 baseline
├── audit_YYYYMMDD.json        Pass 1 detailed results
├── audit_summary_YYYYMMDD.txt Pass 1 human-readable summary
├── dead_code_YYYYMMDD.json    Pass 2 findings
├── suggested_removals_YYYYMMDD.txt  Pass 2 plain-English suggestions
├── benchmark_YYYYMMDD.json    Pass 3 results
├── nightly_report_YYYYMMDD.txt      Pass 4 nightly report
└── logs\
    ├── service_20240115.log   Today's log
    └── service_20240114.log.gz  Yesterday's (compressed)
```

---

## Configuration

Edit `C:\sovereignty_data\config.json` to override defaults:

```json
{
  "API_PORT": 45000,
  "IMPROVEMENT_START": "02:00",
  "IMPROVEMENT_END": "05:00",
  "LOG_RETENTION_DAYS": 30,
  "RESOURCE_RAM_WARN_MB": 500,
  "BENCHMARK_REGRESSION_PCT": 5.0
}
```

Or set the data directory via environment variable:
```bash
set SOVEREIGNTY_DATA=D:\mydata    # Windows
export SOVEREIGNTY_DATA=/opt/sov  # macOS/Linux
```

---

## Uninstall

```bash
python -m sovereignty.uninstall
```

You will be asked:
- Confirm uninstall (y/N)
- Keep log files? (y/N)

The script stops any running daemon, removes the Task Scheduler entry
(if created), and deletes the data directory (or just the non-log files
if you chose to keep logs).

---

## Troubleshooting

### 1. `pip install psutil` fails
```bash
pip install --upgrade pip
pip install psutil
# On Linux you may need: sudo apt install python3-dev
```

### 2. Port 45000 already in use
Edit `config.json` and change `API_PORT` to any unused port (e.g. 45001),
then restart the daemon.

### 3. Heartbeat file not appearing
The main loop writes heartbeat every 60 seconds. Wait at least 65 seconds
after starting. Check the log file in `sovereignty_data/logs/`.

### 4. Pipeline never runs
The pipeline runs between `IMPROVEMENT_START` (default 02:00) and
`IMPROVEMENT_END` (default 05:00) local time. To run it immediately:
```bash
curl -X POST -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"task": "run_full_pipeline"}' \
     http://localhost:45000/task
```

### 5. 401 Unauthorized from API
Your token is in `sovereignty_data/token.key`. Read it:
```bash
type C:\sovereignty_data\token.key    # Windows
cat $SOVEREIGNTY_DATA/token.key       # macOS/Linux
```
Then use it as: `Authorization: Bearer <that-value>`

---

## Architecture

```
service.py (CLI entry point — human starts this)
    └── daemon.py (Daemon class — thread coordinator)
            ├── Thread 1: MainLoop (tick every 60s, heartbeat, scheduler)
            ├── Thread 2: Watchdog (heartbeat check, API ping, pipeline timeout)
            ├── Thread 3: APIServer (http.server on localhost:45000)
            ├── Thread 4: ResourceMonitor (CPU/RAM via psutil)
            └── Thread 5: Pipeline (spawned nightly — improver.py)
                    ├── Pass 1: auditor.py   (AST complexity)
                    ├── Pass 2: dead_code.py (quality sweep)
                    ├── Pass 3: benchmarks.py (10 benchmarks)
                    └── Pass 4: reporter.py  (nightly report)
```

All inter-thread communication uses `threading.Event` and `queue.Queue`.
No shared mutable state without a `threading.Lock`.

---

## Licence

MIT. Do what you want with it.
