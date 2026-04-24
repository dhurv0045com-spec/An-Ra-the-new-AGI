from __future__ import annotations

import json

from scripts.run_sovereignty_audit_v2 import run_sovereignty_audit_v2


def main() -> None:
    print(json.dumps(run_sovereignty_audit_v2(), indent=2))


if __name__ == "__main__":
    main()
