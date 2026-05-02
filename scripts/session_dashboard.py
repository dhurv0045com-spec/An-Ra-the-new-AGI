from __future__ import annotations

import platform
import time
from pathlib import Path


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
