"""Command-line entry point alias `python -m anra_v5.palantiri` delegating to Planthir."""

from __future__ import annotations

from anra_v5.planthir.__main__ import build_parser, main

if __name__ == "__main__":
    raise SystemExit(main())
