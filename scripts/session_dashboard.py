from __future__ import annotations

import platform
import time
from pathlib import Path

<<<<<<< HEAD
=======
try:
    import torch
except ModuleNotFoundError:
    torch = None

from anra_paths import OUTPUT_V2_DIR, STATE_DIR, V3_TOKENIZER_FILE, get_dataset_file, get_v2_checkpoint


def _file_info(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "modified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(path.stat().st_mtime)) if path.exists() else None,
    }


def build_session_report() -> dict[str, object]:
    metrics_path = OUTPUT_V2_DIR / "reports" / "metrics.json"
    metrics = {}
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception as exc:
            metrics = {"load_error": f"[session_dashboard] metrics load failed: {exc}"}
    return {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cwd": str(Path.cwd()),
        "torch": torch.__version__ if torch is not None else "unavailable",
        "cuda_available": torch.cuda.is_available() if torch is not None else False,
        "dataset": _file_info(get_dataset_file()),
        "tokenizer": _file_info(V3_TOKENIZER_FILE),
        "checkpoint": _file_info(get_v2_checkpoint("brain")),
        "state_dir": _file_info(STATE_DIR),
        "latest_metrics": metrics,
    }

>>>>>>> cf05483 (sing the moment)

def print_session_dashboard() -> None:
    print("=" * 64)
    print("AN-RA SESSION DASHBOARD")
    print("=" * 64)
    print(f"UTC Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}")
    print(f"Platform: {platform.platform()}")
    print(f"CWD: {Path.cwd()}")
    print("=" * 64)


if __name__ == "__main__":
    print_session_dashboard()


def print_anra_dashboard(
    *,
    session_n: int = 0,
    offline_minutes: int = 0,
    goal_queue=None,
    benchmark_result=None,
    drive_restore: dict | None = None,
    extra_checks: dict | None = None,
) -> None:
    """
    Structured, human-readable session dashboard.

    Printed at the start of every Colab session.
    """
    width = 64
    bar = "=" * width

    print(bar)
    ts = time.strftime("%Y-%m-%d  %H:%M UTC", time.gmtime())
    print(f"  AN-RA  |  SESSION {session_n}  |  {ts}")
    print(bar)

    print("\n  OFFLINE TIME:")
    if offline_minutes > 0:
        hours, minutes = divmod(int(offline_minutes), 60)
        label = f"{hours}h {minutes}m" if hours else f"{minutes}m"
        print(f"    {label}")
    else:
        print("    Online now")

    print("\n  DRIVE RESTORE:")
    if drive_restore:
        for name, (ok, detail) in drive_restore.items():
            sym = "OK" if ok else "X"
            print(f"    {sym:<2} {name:<22} {detail}")
    else:
        print("    No restore report provided")

    print("\n  ACTIVE GOAL:")
    if goal_queue is not None:
        try:
            report = goal_queue.status_report()
            if isinstance(report, dict):
                counts = report.get("counts", {})
                print(
                    f"    {counts.get('in_progress', 0)} active | "
                    f"{counts.get('queued', 0)} queued | "
                    f"{counts.get('done', 0)} done | "
                    f"{counts.get('failed', 0)} failed"
                )
                in_progress = [
                    item for item in goal_queue._items.values()
                    if item.status == "in_progress"
                ]
                if in_progress:
                    goal = in_progress[0]
                    print(f"    Current: [{goal.goal_id}] {goal.text[:72]}")
                else:
                    print("    Current: none")
            else:
                print(f"    {report}")
        except Exception as exc:
            print(f"    goal_queue error: {exc}")
    else:
        print("    No GoalQueue connected")

    print("\n  SYSTEM CHECKS:")
    if torch is None:
        print("    X  torch                  unavailable")
        print("    X  Flash Attention        INACTIVE - torch unavailable")
    else:
        cuda_ok = torch.cuda.is_available()
        print(f"    {'OK' if cuda_ok else 'X':<2} cuda                   {cuda_ok}")
        if cuda_ok:
            props = torch.cuda.get_device_properties(0)
            print(f"    OK gpu                    {props.name}")
        try:
            flash_ok = bool(torch.backends.cuda.flash_sdp_enabled())
            detail = "active" if flash_ok else "INACTIVE - sequences >1024 may OOM"
            print(f"    {'OK' if flash_ok else 'X':<2} Flash Attention        {detail}")
        except Exception as exc:
            print(f"    ?  Flash Attention        check failed: {exc}")
    if extra_checks:
        for name, (ok, detail) in extra_checks.items():
            sym = "OK" if ok else "X"
            print(f"    {sym:<2} {name:<22} {detail}")

    print("\n  LAST BENCHMARKS:")
    if benchmark_result is None:
        print("    No benchmark result provided")
    else:
        fields = [
            ("Validation perplexity", "val_perplexity"),
            ("RLVR pass@1", "rlvr_pass_at_1"),
            ("CIV score", "civ_score"),
            ("Coherence", "coherence"),
            ("Success", "success"),
        ]
        for label, attr in fields:
            value = getattr(benchmark_result, attr, None)
            print(f"    {label:<24} {value}")

    print(bar)
