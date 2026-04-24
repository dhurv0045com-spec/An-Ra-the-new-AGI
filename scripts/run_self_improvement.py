from __future__ import annotations

import json

from scripts.run_self_improvement_v2 import run_self_improvement_v2


def main() -> None:
    print(json.dumps(run_self_improvement_v2(), indent=2))


if __name__ == "__main__":
    main()
