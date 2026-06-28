"""AN-RA command-line entry point.

Usage after `pip install -e .`:
    anra                  # interactive chat
    anra --report         # system status report
    anra --help           # show all options
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    """Delegate to the existing project-level command-line interface."""
    root = Path(__file__).resolve().parents[1]
    cli = root / "anra.py"
    if not cli.exists():
        print("ERROR: anra.py not found at project root.", file=sys.stderr)
        sys.exit(1)
    sys.argv[0] = str(cli)
    runpy.run_path(str(cli), run_name="__main__")


if __name__ == "__main__":
    main()
