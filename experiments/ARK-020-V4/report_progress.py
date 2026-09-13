"""Read-only progress reporter for the frozen ARK-020 V4 campaign.

This utility intentionally does NOT import or modify the scientific runner.
It exists because SESSION_STATE.json records only the wall time of the most
recent Colab session and is overwritten at each timebox. That field must never
be interpreted as cumulative campaign runtime or campaign progress.

Progress is reconstructed from durable RESULT.json receipts plus the newest
unfinished PARTIAL.json receipt under matched_sets/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path("/content/drive/MyDrive/genisis-arkenstone/ARK020_V4_CONTINUAL")

PARENT_SEEDS = (31801, 31902)
B_ORDER_SEEDS = (429001, 429002)
ARMS = (
    "PLASTIC_HIGH",
    "STATIC_REPLAY_1OF64",
    "STATIC_REPLAY_1OF32",
    "STATIC_CAP16X",
    "GUARDIAN_REACTIVE",
    "GUARDIAN_PREDICTIVE",
    "GUARDIAN_HYBRID",
)
UPDATES_PER_ARM = 2000 + 1500 + 1500
TOTAL_ARMS = len(PARENT_SEEDS) * len(B_ORDER_SEEDS) * len(ARMS)
TOTAL_MAIN_UPDATES = TOTAL_ARMS * UPDATES_PER_ARM


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def reconstruct(root: Path) -> dict[str, Any]:
    completed: list[str] = []
    unfinished: list[dict[str, Any]] = []

    for parent_seed in PARENT_SEEDS:
        for b_seed in B_ORDER_SEEDS:
            set_dir = root / "matched_sets" / f"p{parent_seed}_b{b_seed}"
            for arm in ARMS:
                arm_dir = set_dir / arm
                result_path = arm_dir / "RESULT.json"
                partial_path = arm_dir / "PARTIAL.json"

                result = _read_json(result_path)
                if result is not None and result.get("status") == "COMPLETE":
                    completed.append(str(result_path.relative_to(root)))
                    continue

                partial = _read_json(partial_path)
                if partial is not None and partial.get("status") == "PARTIAL_SESSION":
                    try:
                        step = int(partial.get("global_step", 0))
                    except (TypeError, ValueError):
                        step = 0
                    step = max(0, min(UPDATES_PER_ARM, step))
                    unfinished.append(
                        {
                            "parent_seed": parent_seed,
                            "b_order_seed": b_seed,
                            "arm": arm,
                            "phase": partial.get("phase"),
                            "phase_step": partial.get("phase_step"),
                            "global_step": step,
                            "mtime_ns": partial_path.stat().st_mtime_ns,
                        }
                    )

    # Sequential campaign execution means at most one unfinished arm is live.
    # If stale incomplete receipts ever coexist, use the newest one and report
    # the anomaly rather than silently summing mutually exclusive progress.
    unfinished.sort(key=lambda x: x["mtime_ns"], reverse=True)
    active = unfinished[0] if unfinished else None
    active_steps = int(active["global_step"]) if active is not None else 0

    completed_updates = len(completed) * UPDATES_PER_ARM + active_steps
    completed_updates = min(completed_updates, TOTAL_MAIN_UPDATES)
    progress = completed_updates / TOTAL_MAIN_UPDATES if TOTAL_MAIN_UPDATES else 0.0

    session = _read_json(root / "SESSION_STATE.json") or {}
    last_session_wall = session.get("wall_seconds")

    return {
        "schema": "arkenstone-ark020-v4-readonly-progress/v1",
        "completed_arms": len(completed),
        "total_arms": TOTAL_ARMS,
        "active": active,
        "completed_main_updates": completed_updates,
        "total_main_updates": TOTAL_MAIN_UPDATES,
        "remaining_main_updates": TOTAL_MAIN_UPDATES - completed_updates,
        "progress_fraction": progress,
        "progress_percent": 100.0 * progress,
        "last_session_wall_seconds": last_session_wall,
        "last_session_wall_is_cumulative": False,
        "session_state_warning": (
            "SESSION_STATE.json wall_seconds is the most recent session only; "
            "it is overwritten at each timebox and is not cumulative runtime."
        ),
        "unfinished_receipt_count": len(unfinished),
        "completed_result_paths": completed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    status = reconstruct(args.root)
    if args.json:
        print(json.dumps(status, indent=2, sort_keys=True))
        return 0

    print("=== ARK-020 V4 DURABLE PROGRESS ===")
    print(f"completed arms : {status['completed_arms']} / {status['total_arms']}")
    active = status["active"]
    if active:
        print(
            "active arm     : "
            f"p{active['parent_seed']}_b{active['b_order_seed']}/{active['arm']} "
            f"phase={active['phase']} phase_step={active['phase_step']} "
            f"global_step={active['global_step']}/{UPDATES_PER_ARM}"
        )
    else:
        print("active arm     : none")
    print(
        f"main updates   : {status['completed_main_updates']:,} / "
        f"{status['total_main_updates']:,}"
    )
    print(f"progress       : {status['progress_percent']:.2f}%")
    print(f"remaining      : {status['remaining_main_updates']:,} updates")
    if status["last_session_wall_seconds"] is not None:
        hours = float(status["last_session_wall_seconds"]) / 3600.0
        print(f"last session   : {hours:.2f} h (NOT cumulative)")
    print("WARNING        :", status["session_state_warning"])
    if status["unfinished_receipt_count"] > 1:
        print(
            "ANOMALY        : multiple unfinished receipts exist; newest selected. "
            "Inspect before drawing progress conclusions."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
