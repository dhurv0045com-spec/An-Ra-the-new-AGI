from __future__ import annotations

import json
from pathlib import Path


from runtime.training_readiness import assess_training_readiness


def main() -> int:
    readiness = assess_training_readiness()
    print(json.dumps(readiness.to_dict(), indent=2))
    return 0 if readiness.ready_for_session else 2


if __name__ == "__main__":
    raise SystemExit(main())
