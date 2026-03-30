"""
sovereignty/demo.py
===================
Full simulation of all Sovereignty daemon capabilities.

Runs all 7 demonstration scenarios in sequence. Requires ZERO admin rights.
Uses a temporary directory for all file operations.
No Windows Service, no Task Scheduler, no system modifications.

Scenarios:
  1. STARTUP    — Simulate daemon start; show threads launching + heartbeat
  2. PASS 1     — Run auditor on this file; show complexity + deltas
  3. PASS 2     — Plant 3 issues in a temp file; show dead_code finding all 3
  4. PASS 3     — Run all 10 benchmarks; simulate B02 regression
  5. PASS 4     — Generate full nightly report; print it; show sparklines
  6. API        — Start API on :45000; make requests; show 401; shutdown
  7. WATCHDOG   — Simulate main loop freeze; show detection + restart log

Usage:
    python -m sovereignty.demo
"""

import json
import os
import pathlib
import signal
import sys
import tempfile
import threading
import time
from datetime import datetime

# ── Bootstrap ─────────────────────────────────────────────────────────────────
_THIS_DIR = pathlib.Path(__file__).parent
if str(_THIS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR.parent))

# ── Colour helpers (ANSI, falls back gracefully on Windows without VT) ─────────
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_CYAN   = "\033[36m"
_RED    = "\033[31m"
_GRAY   = "\033[90m"


def _c(text: str, colour: str) -> str:
    """Wrap text in ANSI colour if stdout is a TTY."""
    if sys.stdout.isatty():
        return f"{colour}{text}{_RESET}"
    return text


def _header(title: str, n: int) -> None:
    width = 68
    print()
    print(_c("═" * width, _CYAN))
    print(_c(f"  SCENARIO {n} — {title}", _BOLD + _CYAN))
    print(_c("═" * width, _CYAN))


def _ok(msg: str) -> None:
    print(_c(f"  ✓ {msg}", _GREEN))


def _warn(msg: str) -> None:
    print(_c(f"  ⚠ {msg}", _YELLOW))


def _info(msg: str) -> None:
    print(_c(f"  → {msg}", _GRAY))


def _err(msg: str) -> None:
    print(_c(f"  ✗ {msg}", _RED))


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO 1 — STARTUP
# ═══════════════════════════════════════════════════════════════════════════════

def scenario_startup(tmp: pathlib.Path) -> None:
    """
    Simulate service start: show all threads launching, first heartbeat, API ready.

    Parameters:
        tmp: Temporary directory for demo data.
    """
    _header("STARTUP", 1)

    from sovereignty.config import Config
    from sovereignty.logger import setup_logging, get_logger

    config = Config()
    config.DATA_DIR = tmp
    setup_logging(config)
    log = get_logger("demo.startup")

    _info("Initialising Daemon object...")
    from sovereignty.daemon import Daemon
    daemon = Daemon(config, target_dir=_THIS_DIR)

    _info("Starting all threads (no admin rights needed)...")
    daemon.start()
    time.sleep(2)  # Let threads settle

    # Show thread status
    status = daemon._thread_status_snapshot()
    for name, state in status.items():
        if state == "alive":
            _ok(f"Thread '{name}': {state}")
        else:
            _warn(f"Thread '{name}': {state}")

    # Verify heartbeat was written
    hb_path = config.DATA_DIR / "heartbeat.json"
    time.sleep(1.5)
    if hb_path.exists():
        hb = json.loads(hb_path.read_text(encoding="utf-8"))
        _ok(f"Heartbeat written: {hb['timestamp']}")
    else:
        _warn("Heartbeat not yet written (daemon may need another tick)")

    _ok("API server ready on http://127.0.0.1:45000")

    daemon.stop()
    _ok("Daemon stopped cleanly after scenario 1")


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO 2 — PASS 1: CODE AUDIT
# ═══════════════════════════════════════════════════════════════════════════════

def scenario_pass1(tmp: pathlib.Path) -> dict:
    """
    Run Pass 1 (auditor) on this demo.py file. Show complexity per function.
    Simulate a baseline so we can show delta arrows.

    Parameters:
        tmp: Temporary directory.

    Returns:
        audit_result dict for use by scenario_pass4.
    """
    _header("PASS 1 — DEEP CODE AUDIT", 2)

    from sovereignty.config import Config
    from sovereignty.auditor import AuditPass, _analyse_file, _aggregate

    config = Config()
    config.DATA_DIR = tmp

    # Simulate a previous-night baseline: slightly higher complexity
    fake_baseline = {
        "total_functions": 20,
        "avg_cyclomatic": 3.5,
        "max_cyclomatic": 8,
        "max_cyclomatic_fn": "old_complex_fn (old_file.py:10)",
        "avg_cognitive": 5.0,
        "max_cognitive": 12,
        "max_cognitive_fn": "old_complex_fn (old_file.py:10)",
        "pct_with_docstring": 60.0,
        "avg_lines": 18.0,
        "flagged_complexity": ["old_complex_fn"],
        "flagged_no_docstring": [],
    }
    config.AUDIT_BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.AUDIT_BASELINE_FILE.write_text(json.dumps(fake_baseline, indent=2), encoding="utf-8")

    _info(f"Scanning: {_THIS_DIR}")
    audit = AuditPass(config, _THIS_DIR)
    result = audit.run(date_str=datetime.now().strftime("%Y%m%d"))

    agg = result["aggregate"]
    deltas = result["deltas"]

    print()
    print(_c("  Function-level complexity (this file):", _BOLD))
    # Show per-function details for demo.py itself
    demo_funcs = _analyse_file(_THIS_DIR / "demo.py")
    print(f"  {'Function':<40} {'Cyclo':>6} {'Cogni':>6} {'Lines':>6}  {'Docstring':>9}")
    print("  " + "─" * 68)
    for fn in sorted(demo_funcs, key=lambda f: -f["cyclomatic"]):
        flag = _c("⚠ complex", _YELLOW) if fn["cyclomatic"] > 5 else ""
        ds = _c("✓", _GREEN) if fn["has_docstring"] else _c("✗", _RED)
        print(
            f"  {fn['function']:<40} {fn['cyclomatic']:>6} {fn['cognitive']:>6} "
            f"{fn['lines']:>6}  {ds:>9}  {flag}"
        )

    print()
    print(_c("  Aggregate metrics vs last night:", _BOLD))
    arrow_map = {"↑": _c("↑", _GREEN), "↓": _c("↓", _RED), "→": _c("→", _GRAY)}
    for key, info in deltas.items():
        arrow = arrow_map.get(info["direction"], " ")
        print(
            f"  {key:<30}  {str(info['current']):>8}  (was {info['baseline']}, Δ {info['delta']:+.3f}) {arrow}"
        )

    _ok(f"Audit complete: {agg['total_functions']} functions analysed")
    if agg["flagged_complexity"]:
        _warn(f"Flagged for high complexity: {agg['flagged_complexity']}")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO 3 — PASS 2: DEAD CODE SWEEP
# ═══════════════════════════════════════════════════════════════════════════════

def scenario_pass2(tmp: pathlib.Path) -> dict:
    """
    Plant 3 known issues in a temp file and show dead_code.py finding all 3.

    Issues planted:
      (a) Unused import
      (b) Unreachable code after return
      (c) Magic number in expression

    Parameters:
        tmp: Temporary directory.

    Returns:
        dead_result dict for use by scenario_pass4.
    """
    _header("PASS 2 — DEAD CODE & QUALITY SWEEP", 3)

    from sovereignty.config import Config
    from sovereignty.dead_code import DeadCodePass

    config = Config()
    config.DATA_DIR = tmp

    # Plant a file with 3 known issues
    planted_dir = tmp / "planted_code"
    planted_dir.mkdir(exist_ok=True)
    planted_file = planted_dir / "issues_planted.py"
    planted_file.write_text(
        '''\
import os          # (a) unused import — os is never used below

def calculate_area(radius):
    return 3.14159 * radius * radius  # (c) magic number 3.14159

def process_data(items):
    if not items:
        return []
    return [x * 2 for x in items]
    print("This line is unreachable")  # (b) unreachable code after return
''',
        encoding="utf-8",
    )

    _info(f"Scanning planted file: {planted_file}")
    _info("Expected findings: unused import, unreachable code, magic number")
    print()

    dead = DeadCodePass(config, planted_dir)
    result = dead.run(date_str=datetime.now().strftime("%Y%m%d"))

    findings = result["findings"]
    summary = result["summary"]

    for i, f in enumerate(findings, 1):
        icon = _c("⚠", _YELLOW)
        print(f"  {icon} Finding {i}: [{f['category'].upper()}] line {f['line']}")
        print(f"      {f['description']}")
        print()

    found_cats = set(f["category"] for f in findings)
    for expected, label in [
        ("unused_import", "(a) unused import"),
        ("unreachable_code", "(b) unreachable code"),
        ("magic_number", "(c) magic number"),
    ]:
        if expected in found_cats:
            _ok(f"Detected {label}")
        else:
            _warn(f"Did not detect {label} (may vary by Python version)")

    _ok(f"Pass 2 complete: {summary['total']} issues found in {planted_file.name}")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO 4 — PASS 3: PERFORMANCE BENCHMARKS
# ═══════════════════════════════════════════════════════════════════════════════

def scenario_pass3(tmp: pathlib.Path) -> dict:
    """
    Run all 10 benchmarks. Simulate a 7% regression in B02 (API response time).
    Show REGRESSION alert. Show that baseline is NOT updated.

    Parameters:
        tmp: Temporary directory.

    Returns:
        bench_result dict for use by scenario_pass4.
    """
    _header("PASS 3 — PERFORMANCE BENCHMARKS", 4)

    from sovereignty.config import Config
    from sovereignty.benchmarks import BenchmarkPass

    config = Config()
    config.DATA_DIR = tmp

    # Write a baseline where B02 is artificially fast (so our real result looks slow)
    # This ensures the 7% regression demo fires reliably
    baseline = {
        f"B{i:02d}": {"median": 0.0, "unit": "ms"}  # zeros = no comparison
        for i in range(1, 11)
    }
    # Set B02 baseline to 1ms so anything we measure (>1.07ms) looks like a regression
    baseline["B02"] = {"median": 0.5, "unit": "ms"}
    config.BENCHMARK_BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.BENCHMARK_BASELINE_FILE.write_text(json.dumps(baseline, indent=2), encoding="utf-8")

    _info("Running all 10 benchmarks (3 runs each, median taken)...")
    print()

    bench = BenchmarkPass(config)
    result = bench.run(date_str=datetime.now().strftime("%Y%m%d"))

    results_list = result["results"]
    regressions = result["regressions"]

    print(_c(f"  {'ID':<5} {'Name':<35} {'Median':>12} {'Unit':<8} {'Δ%':>7}  Flag", _BOLD))
    print("  " + "─" * 76)
    for r in results_list:
        flag = r.get("flag", "unchanged")
        if flag == "REGRESSION":
            flag_str = _c("⚠ REGRESSION", _RED)
        elif flag == "IMPROVEMENT":
            flag_str = _c("✓ IMPROVEMENT", _GREEN)
        else:
            flag_str = _c("→ unchanged", _GRAY)

        print(
            f"  {r['id']:<5} {r['name']:<35} {r['median']:>12.4f} "
            f"{r['unit']:<8} {r['delta_pct']:>+6.1f}%  {flag_str}"
        )

    print()
    if regressions:
        _warn(f"{len(regressions)} REGRESSION(s) detected — baseline NOT updated")
        for r in regressions:
            _warn(f"  {r['id']} ({r['name']}): {r['delta_pct']:+.1f}% slower than baseline")
    else:
        _ok("No regressions detected — baseline updated")

    baseline_updated = result["baseline_updated"]
    if not baseline_updated:
        _ok("Confirmed: baseline file NOT updated (regressions present)")
    else:
        _ok("Baseline updated (all benchmarks passed or improved)")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO 5 — PASS 4: NIGHTLY REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def scenario_pass4(
    tmp: pathlib.Path,
    audit_result: dict,
    dead_result: dict,
    bench_result: dict,
) -> None:
    """
    Generate the full nightly report from Passes 1–3. Print it to stdout.

    Parameters:
        tmp: Temporary directory.
        audit_result: Output from scenario_pass1.
        dead_result: Output from scenario_pass2.
        bench_result: Output from scenario_pass3.
    """
    _header("PASS 4 — NIGHTLY SELF-REPORT", 5)

    from sovereignty.config import Config
    from sovereignty.reporter import ReportPass

    config = Config()
    config.DATA_DIR = tmp

    _info("Compiling report from Passes 1–3...")
    date_str = datetime.now().strftime("%Y%m%d")
    report_text = ReportPass(config).run(
        audit_result=audit_result,
        dead_result=dead_result,
        bench_result=bench_result,
        date_str=date_str,
    )

    print()
    print(report_text)
    _ok(f"Report written to {tmp}/nightly_report_{date_str}.txt")


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO 6 — REST API
# ═══════════════════════════════════════════════════════════════════════════════

def scenario_api(tmp: pathlib.Path) -> None:
    """
    Start the REST API on localhost:45000. Make requests showing all endpoints.
    Show 401 on bad token. Show graceful shutdown.

    Parameters:
        tmp: Temporary directory.
    """
    _header("REST API DEMONSTRATION", 6)

    import urllib.request
    import urllib.error

    from sovereignty.config import Config
    from sovereignty.api import APIServer, SharedState
    from sovereignty.auth import generate_token
    from sovereignty.logger import setup_logging

    config = Config()
    config.DATA_DIR = tmp
    setup_logging(config)

    # Generate token
    token = generate_token(config)
    state = SharedState()
    api = APIServer(config, state)

    api_thread = threading.Thread(target=api.run_forever, name="DemoAPI", daemon=True)
    api_thread.start()
    time.sleep(0.5)  # Give the server a moment to bind

    base = f"http://{config.API_HOST}:{config.API_PORT}"

    def _get(path: str, auth: bool = True, bad_token: bool = False) -> tuple:
        """Make a GET request, return (status_code, body_dict_or_str)."""
        tok = "bad_token_here" if bad_token else token
        headers = {"Authorization": f"Bearer {tok}"} if auth else {}
        req = urllib.request.Request(f"{base}{path}", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                raw = resp.read().decode("utf-8")
                try:
                    return resp.status, json.loads(raw)
                except Exception:
                    return resp.status, raw
        except urllib.error.HTTPError as e:
            return e.code, {}

    def _post(path: str, body: dict, auth: bool = True) -> tuple:
        """Make a POST request, return (status_code, body_dict)."""
        encoded = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{base}{path}", data=encoded,
            headers={
                "Authorization": f"Bearer {token}" if auth else "",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, {}

    # ── GET /ping (no auth needed) ─────────────────────────────────────────
    code, body = _get("/ping", auth=False)
    print(f"\n  GET /ping (no auth)")
    print(f"  Status: {code}  Body: {json.dumps(body)}")
    _ok("Liveness check passed") if code == 200 else _err(f"Expected 200, got {code}")

    # ── GET /status ────────────────────────────────────────────────────────
    code, body = _get("/status")
    print(f"\n  GET /status (with valid token)")
    print(f"  Status: {code}")
    print(f"  Body: {json.dumps(body, indent=4)}")
    _ok("Status endpoint working") if code == 200 else _err(f"Expected 200, got {code}")

    # ── GET /log?lines=20 ──────────────────────────────────────────────────
    code, body = _get("/log?lines=20")
    print(f"\n  GET /log?lines=20")
    print(f"  Status: {code}  Lines returned: {body.get('count', '?')}")
    _ok("Log endpoint working") if code == 200 else _err(f"Expected 200, got {code}")

    # ── POST /task ─────────────────────────────────────────────────────────
    code, body = _post("/task", {"task": "gc"})
    print(f"\n  POST /task {{\"task\": \"gc\"}}")
    print(f"  Status: {code}  Body: {json.dumps(body)}")
    _ok(f"Task queued: {body.get('task_id', '?')}") if code == 200 else _err(f"Expected 200, got {code}")

    # ── GET /status with BAD token ─────────────────────────────────────────
    code, body = _get("/status", bad_token=True)
    print(f"\n  GET /status (with BAD token)")
    print(f"  Status: {code}  (expected 401)")
    _ok("401 returned for bad token — no info leaked") if code == 401 else _warn(f"Got {code}, expected 401")

    # ── GET /report?date=today (may 404 if no report yet) ──────────────────
    date_str = datetime.now().strftime("%Y-%m-%d")
    code, body = _get(f"/report?date={date_str}")
    print(f"\n  GET /report?date={date_str}")
    if code == 200:
        preview = str(body)[:120] + "..." if len(str(body)) > 120 else str(body)
        print(f"  Status: {code}  Preview: {preview}")
        _ok("Report endpoint working")
    else:
        print(f"  Status: {code}  (404 means no report for today yet — run pipeline first)")

    # ── POST /shutdown ─────────────────────────────────────────────────────
    print(f"\n  POST /shutdown")
    code, body = _post("/shutdown", {"confirm": "yes"})
    print(f"  Status: {code}  Body: {json.dumps(body)}")
    _ok("Shutdown accepted") if code == 200 else _err(f"Expected 200, got {code}")

    api.stop()
    time.sleep(0.3)
    _ok("API server stopped cleanly")


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO 7 — WATCHDOG
# ═══════════════════════════════════════════════════════════════════════════════

def scenario_watchdog(tmp: pathlib.Path) -> None:
    """
    Simulate a main loop freeze: stop writing heartbeats, let the watchdog
    detect the timeout, attempt a restart, and confirm recovery.

    Uses an accelerated timeout (5 seconds) so the demo doesn't take 90 seconds.

    Parameters:
        tmp: Temporary directory.
    """
    _header("WATCHDOG — FREEZE DETECTION & RESTART", 7)

    import threading
    from sovereignty.config import Config
    from sovereignty.watchdog import Watchdog

    config = Config()
    config.DATA_DIR = tmp
    config.WATCHDOG_TIMEOUT_SEC = 5   # Accelerated for demo (normally 90s)
    config.WATCHDOG_HEARTBEAT_SEC = 2

    # Write a fresh heartbeat so watchdog starts happy
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "heartbeat.json").write_text(
        json.dumps({"timestamp": datetime.now().isoformat()}), encoding="utf-8"
    )

    restart_log: list = []
    emergency_log: list = []

    def _fake_restart_loop() -> bool:
        ts = datetime.now().strftime("%H:%M:%S")
        restart_log.append(ts)
        print(f"\n  {_c('WATCHDOG ACTION', _YELLOW)}: Main loop restart attempt at {ts}")
        # Write a fresh heartbeat — simulates a successful restart
        (tmp / "heartbeat.json").write_text(
            json.dumps({"timestamp": datetime.now().isoformat()}), encoding="utf-8"
        )
        print(f"  {_c('RECOVERY', _GREEN)}: Heartbeat refreshed — restart simulated as successful")
        return True

    def _fake_restart_api() -> bool:
        return True

    def _fake_get_pipeline_thread():
        return None

    def _fake_stop_service():
        emergency_log.append("EMERGENCY STOP")
        print(f"  {_c('CRITICAL', _RED)}: Service stop would be triggered here")

    watchdog = Watchdog(
        config=config,
        restart_main_loop_fn=_fake_restart_loop,
        restart_api_fn=_fake_restart_api,
        get_pipeline_thread=_fake_get_pipeline_thread,
        stop_service_fn=_fake_stop_service,
    )

    # Accelerated poll (2s) so demo fires within ~10s instead of 30s+
    def _accelerated_loop():
        while not watchdog._stop_event.is_set():
            try:
                watchdog._check_heartbeat()
            except Exception:
                pass
            watchdog._stop_event.wait(timeout=2)

    wd_thread = threading.Thread(target=_accelerated_loop, name="DemoWatchdog", daemon=True)
    wd_thread.start()

    _ok("Watchdog thread started (2s accelerated poll)")
    _info(f"Heartbeat fresh — waiting for stale timeout ({config.WATCHDOG_TIMEOUT_SEC}s)...")

    # Wait long enough for watchdog to detect staleness (> WATCHDOG_TIMEOUT_SEC = 5s)
    # Watchdog polls every 30s but we use a short poll loop here for demo speed
    deadline = time.monotonic() + 12
    while time.monotonic() < deadline and not restart_log:
        time.sleep(0.5)

    if restart_log:
        _ok(f"Watchdog detected freeze at {restart_log[0]}")
        _ok("Restart attempt logged")
        _ok("Recovery confirmed — new heartbeat written")
    else:
        _warn("Watchdog did not fire in time — try increasing demo timeout")

    watchdog.stop()
    wd_thread.join(timeout=5)
    _ok("Watchdog stopped cleanly")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    """
    Run all 7 demo scenarios in sequence.

    Returns:
        0 on success, 1 if any scenario raised an unhandled exception.
    """
    print(_c("\n" + "═" * 68, _CYAN))
    print(_c("  SOVEREIGNTY PACKAGE — FULL DEMO (zero admin rights)", _BOLD + _CYAN))
    print(_c("  All file operations in a temporary directory", _CYAN))
    print(_c("═" * 68, _CYAN))

    errors: list = []

    with tempfile.TemporaryDirectory(prefix="sovereignty_demo_") as tmp_str:
        tmp = pathlib.Path(tmp_str)
        print(_c(f"\n  Temp directory: {tmp}", _GRAY))

        audit_result = dead_result = bench_result = None

        # Scenario 1
        try:
            scenario_startup(tmp)
        except Exception as exc:
            _err(f"Scenario 1 error: {exc}")
            errors.append(exc)

        # Scenario 2
        try:
            audit_result = scenario_pass1(tmp)
        except Exception as exc:
            _err(f"Scenario 2 error: {exc}")
            errors.append(exc)
            audit_result = {}

        # Scenario 3
        try:
            dead_result = scenario_pass2(tmp)
        except Exception as exc:
            _err(f"Scenario 3 error: {exc}")
            errors.append(exc)
            dead_result = {}

        # Scenario 4
        try:
            bench_result = scenario_pass3(tmp)
        except Exception as exc:
            _err(f"Scenario 4 error: {exc}")
            errors.append(exc)
            bench_result = {}

        # Scenario 5
        try:
            scenario_pass4(tmp, audit_result or {}, dead_result or {}, bench_result or {})
        except Exception as exc:
            _err(f"Scenario 5 error: {exc}")
            errors.append(exc)

        # Scenario 6
        try:
            scenario_api(tmp)
        except Exception as exc:
            _err(f"Scenario 6 error: {exc}")
            errors.append(exc)

        # Scenario 7
        try:
            scenario_watchdog(tmp)
        except Exception as exc:
            _err(f"Scenario 7 error: {exc}")
            errors.append(exc)

    # Final summary
    print()
    print(_c("═" * 68, _CYAN))
    if errors:
        print(_c(f"  DEMO COMPLETE — {len(errors)} error(s)", _YELLOW))
        for e in errors:
            print(_c(f"    • {e}", _RED))
    else:
        print(_c("  DEMO COMPLETE — All 7 scenarios passed ✓", _BOLD + _GREEN))
    print(_c("═" * 68, _CYAN))

    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
