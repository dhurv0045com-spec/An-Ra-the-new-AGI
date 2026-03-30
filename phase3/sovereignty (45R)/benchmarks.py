"""
sovereignty/benchmarks.py
==========================
Pass 3 of the nightly improvement pipeline: 10-benchmark performance suite.

Benchmarks B01–B10 measure key operations of the daemon itself.
Each runs 3 times; the median is taken. Results are compared to the stored
baseline. Regressions (>5% slower) and improvements (>5% faster) are flagged.
The baseline is updated ONLY if all benchmarks pass or improve.

Output:
  benchmark_YYYYMMDD.json — all results, deltas, REGRESSION/IMPROVEMENT flags

Relationship to other modules:
    improver.py calls BenchmarkPass.run() during Pass 3.
    reporter.py reads benchmark_YYYYMMDD.json for the nightly report.
    config.py provides BENCHMARK_REGRESSION_PCT, BENCHMARK_IMPROVEMENT_PCT.
"""

import json
import pathlib
import queue
import statistics
import threading
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

from sovereignty.config import Config
from sovereignty.logger import get_logger

log = get_logger(__name__)


# ── Individual benchmark functions ────────────────────────────────────────────

def _bench_b01_heartbeat_write(data_dir: pathlib.Path) -> float:
    """
    B01: Heartbeat write latency — median of 1000 JSON writes.

    Measures how long it takes to write a heartbeat.json file,
    which the main loop does every 60 seconds.

    Returns:
        Median write time in milliseconds.
    """
    hb_path = data_dir / "bench_heartbeat.json"
    times = []
    for _ in range(1000):
        t0 = time.perf_counter()
        hb_path.write_text(
            json.dumps({"timestamp": datetime.now().isoformat()}),
            encoding="utf-8",
        )
        times.append((time.perf_counter() - t0) * 1000)
    try:
        hb_path.unlink()
    except Exception:
        pass
    return statistics.median(times)


def _bench_b02_api_status_response(port: int) -> float:
    """
    B02: API response time for GET /status — median of 100 calls.

    Returns:
        Median response time in milliseconds, or 9999 if API not running.
    """
    import urllib.request
    import urllib.error

    times = []
    url = f"http://127.0.0.1:{port}/ping"
    for _ in range(100):
        try:
            t0 = time.perf_counter()
            with urllib.request.urlopen(url, timeout=2):
                pass
            times.append((time.perf_counter() - t0) * 1000)
        except Exception:
            times.append(9999.0)
    return statistics.median(times)


def _bench_b03_log_write_throughput(data_dir: pathlib.Path) -> float:
    """
    B03: Log write throughput — lines per second.

    Returns:
        Lines written per second (higher is better).
    """
    log_path = data_dir / "bench_log.txt"
    line = "[2024-01-01 02:00:00.000] [INFO    ] [MainThread          ] [bench.test] Benchmark log line\n"
    count = 5000
    t0 = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as f:
        for _ in range(count):
            f.write(line)
    elapsed = time.perf_counter() - t0
    try:
        log_path.unlink()
    except Exception:
        pass
    return count / max(elapsed, 0.0001)


def _bench_b04_main_loop_tick(config: Config) -> float:
    """
    B04: Main loop tick duration — median of 100 simulated ticks.

    Simulates what the main loop does on each tick (heartbeat write,
    queue poll, scheduler time check).

    Returns:
        Median tick duration in milliseconds.
    """
    hb_path = config.DATA_DIR / "bench_tick.json"
    task_queue: queue.Queue = queue.Queue()
    times = []
    now = datetime.now()

    for _ in range(100):
        t0 = time.perf_counter()
        # Simulate heartbeat write
        hb_path.write_text(
            json.dumps({"timestamp": now.isoformat()}), encoding="utf-8"
        )
        # Simulate queue poll
        try:
            task_queue.get_nowait()
        except queue.Empty:
            pass
        # Simulate time check (cheap)
        _ = now.strftime("%H:%M")
        times.append((time.perf_counter() - t0) * 1000)

    try:
        hb_path.unlink()
    except Exception:
        pass
    return statistics.median(times)


def _bench_b05_scheduler_decision(config: Config) -> float:
    """
    B05: Scheduler decision time — median of 1000 should_run() calls.

    Returns:
        Median decision time in microseconds.
    """
    from sovereignty.scheduler import Scheduler

    sched = Scheduler(config)
    times = []
    for _ in range(1000):
        t0 = time.perf_counter()
        sched.should_run(datetime.now())
        times.append((time.perf_counter() - t0) * 1_000_000)
    return statistics.median(times)


def _bench_b06_json_serialisation() -> float:
    """
    B06: JSON serialisation speed — 1000 status objects per second.

    Returns:
        Objects serialised per second.
    """
    status = {
        "service": "running",
        "uptime_seconds": 3600,
        "last_improvement_run": "2024-01-01 02:00:00",
        "last_improvement_result": "success",
        "next_scheduled_run": "2024-01-02 02:00:00",
        "threads": {"MainLoop": "alive", "Watchdog": "alive", "API": "alive"},
        "resource": {"cpu_pct": 1.2, "ram_mb": 45.6},
    }
    count = 1000
    t0 = time.perf_counter()
    for _ in range(count):
        json.dumps(status)
    elapsed = time.perf_counter() - t0
    return count / max(elapsed, 0.0001)


def _bench_b07_token_validation(config: Config) -> float:
    """
    B07: Token validation speed — median of 10000 checks.

    Returns:
        Median validation time in microseconds.
    """
    from sovereignty.auth import validate_token

    times = []
    test_token = "a" * 64  # Fake token — will always fail, but measures the check
    for _ in range(10000):
        t0 = time.perf_counter()
        validate_token(test_token, config)
        times.append((time.perf_counter() - t0) * 1_000_000)
    return statistics.median(times)


def _bench_b08_thread_spawn_join() -> float:
    """
    B08: Thread spawn and join time — median of 100 spawns.

    Returns:
        Median spawn+join time in milliseconds.
    """
    def _noop():
        pass

    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        t = threading.Thread(target=_noop)
        t.start()
        t.join()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def _bench_b09_json_deserialisation() -> float:
    """
    B09: JSON deserialisation speed — 1000 status objects per second.

    Returns:
        Objects deserialised per second.
    """
    payload = json.dumps({
        "service": "running",
        "uptime_seconds": 3600,
        "threads": {"MainLoop": "alive"},
        "resource": {"cpu_pct": 1.2, "ram_mb": 45.6},
    })
    count = 1000
    t0 = time.perf_counter()
    for _ in range(count):
        json.loads(payload)
    elapsed = time.perf_counter() - t0
    return count / max(elapsed, 0.0001)


def _bench_b10_queue_throughput() -> float:
    """
    B10: Task queue throughput — items per second (put + get cycle).

    Returns:
        Queue operations per second.
    """
    q: queue.Queue = queue.Queue(maxsize=0)
    count = 10000
    item = {"task": "run_pass_1", "task_id": "abc123"}
    t0 = time.perf_counter()
    for _ in range(count):
        q.put_nowait(item)
    for _ in range(count):
        q.get_nowait()
    elapsed = time.perf_counter() - t0
    return count / max(elapsed, 0.0001)


# ── Benchmark registry ────────────────────────────────────────────────────────

def _build_suite(config: Config) -> List[Tuple[str, str, Callable[[], float], str]]:
    """
    Return the list of benchmarks as (id, name, fn, unit).

    All lambdas capture config and data_dir via closure.

    Parameters:
        config: Active Config instance.

    Returns:
        List of (benchmark_id, human_name, callable, unit_label).
    """
    data_dir = config.DATA_DIR
    port = config.API_PORT

    return [
        ("B01", "Heartbeat write latency",   lambda: _bench_b01_heartbeat_write(data_dir), "ms"),
        ("B02", "API /ping response time",   lambda: _bench_b02_api_status_response(port),  "ms"),
        ("B03", "Log write throughput",      lambda: _bench_b03_log_write_throughput(data_dir), "lines/s"),
        ("B04", "Main loop tick duration",   lambda: _bench_b04_main_loop_tick(config),     "ms"),
        ("B05", "Scheduler decision time",   lambda: _bench_b05_scheduler_decision(config), "µs"),
        ("B06", "JSON serialisation speed",  lambda: _bench_b06_json_serialisation(),       "obj/s"),
        ("B07", "Token validation speed",    lambda: _bench_b07_token_validation(config),   "µs"),
        ("B08", "Thread spawn+join time",    lambda: _bench_b08_thread_spawn_join(),        "ms"),
        ("B09", "JSON deserialisation speed",lambda: _bench_b09_json_deserialisation(),     "obj/s"),
        ("B10", "Queue throughput",          lambda: _bench_b10_queue_throughput(),         "ops/s"),
    ]


def _run_once(fn: Callable[[], float]) -> float:
    """Run a benchmark function once, returning its result."""
    try:
        return fn()
    except Exception as exc:
        log.warning("Benchmark raised exception: %s", exc)
        return 0.0


class BenchmarkPass:
    """
    Orchestrates Pass 3: runs all 10 benchmarks and compares to baseline.
    """

    def __init__(self, config: Config) -> None:
        """
        Parameters:
            config: Active Config instance.
        """
        self._config = config

    def run(self, date_str: Optional[str] = None) -> Dict:
        """
        Execute Pass 3 and write output files.

        Parameters:
            date_str: Date label for output files (YYYYMMDD).

        Returns:
            Dict with 'results' list, 'regressions', 'improvements'.
        """
        date_str = date_str or datetime.now().strftime("%Y%m%d")
        log.info("Pass 3: Running %d benchmarks", self._config.BENCHMARK_COUNT)

        suite = _build_suite(self._config)
        baseline = self._load_baseline()
        results = []
        has_regression = False

        for bench_id, name, fn, unit in suite:
            # Run BENCHMARK_RUNS times (default 3), take median
            runs = [_run_once(fn) for _ in range(self._config.BENCHMARK_RUNS)]
            median_val = statistics.median(runs)

            base_val = baseline.get(bench_id, {}).get("median", 0.0)
            delta_pct = 0.0
            flag = "unchanged"

            if base_val > 0:
                # For throughput metrics (obj/s, ops/s, lines/s): higher = better
                # For latency metrics (ms, µs): lower = better
                if unit in ("ms", "µs"):
                    delta_pct = ((median_val - base_val) / base_val) * 100
                    if delta_pct > self._config.BENCHMARK_REGRESSION_PCT:
                        flag = "REGRESSION"
                        has_regression = True
                    elif delta_pct < -self._config.BENCHMARK_IMPROVEMENT_PCT:
                        flag = "IMPROVEMENT"
                else:
                    delta_pct = ((median_val - base_val) / base_val) * 100
                    if delta_pct < -self._config.BENCHMARK_REGRESSION_PCT:
                        flag = "REGRESSION"
                        has_regression = True
                    elif delta_pct > self._config.BENCHMARK_IMPROVEMENT_PCT:
                        flag = "IMPROVEMENT"

            result = {
                "id": bench_id,
                "name": name,
                "unit": unit,
                "median": round(median_val, 4),
                "runs": [round(r, 4) for r in runs],
                "baseline": round(base_val, 4),
                "delta_pct": round(delta_pct, 2),
                "flag": flag,
            }
            results.append(result)

            log.info(
                "  %s %-30s %10.4f %-8s  %+.1f%%  %s",
                bench_id, name, median_val, unit, delta_pct, flag,
            )

        # Only update baseline if no regressions
        if not has_regression:
            self._save_baseline(results)
            log.info("Pass 3: Baseline updated (no regressions)")
        else:
            log.warning("Pass 3: Baseline NOT updated — regression(s) detected")

        regressions = [r for r in results if r["flag"] == "REGRESSION"]
        improvements = [r for r in results if r["flag"] == "IMPROVEMENT"]

        # Write output
        out_json = self._config.DATA_DIR / f"benchmark_{date_str}.json"
        out_json.write_text(
            json.dumps({
                "results": results,
                "regressions": len(regressions),
                "improvements": len(improvements),
                "baseline_updated": not has_regression,
            }, indent=2),
            encoding="utf-8",
        )

        log.info(
            "Pass 3 complete: %d regressions, %d improvements",
            len(regressions), len(improvements),
        )
        return {
            "results": results,
            "regressions": regressions,
            "improvements": improvements,
            "baseline_updated": not has_regression,
        }

    def _load_baseline(self) -> Dict:
        """Load benchmark_baseline.json, returning empty dict if missing."""
        path = self._config.BENCHMARK_BASELINE_FILE
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_baseline(self, results: List[Dict]) -> None:
        """Write current results as the new benchmark baseline."""
        baseline = {r["id"]: {"median": r["median"], "unit": r["unit"]} for r in results}
        self._config.BENCHMARK_BASELINE_FILE.write_text(
            json.dumps(baseline, indent=2), encoding="utf-8"
        )
