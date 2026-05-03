from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anra_paths import DATASET, ROOT, inject_all_paths
from runtime.system_registry import component_registry, component_status, missing_required_components

inject_all_paths()


def _ensure_v2_tokenizer() -> None:
    if not DATASET.exists():
        return
    try:
        from training.v2_runtime import load_or_build_v2_tokenizer

        load_or_build_v2_tokenizer(dataset_path=DATASET)
    except ModuleNotFoundError:
        return


def main() -> int:
    _ensure_v2_tokenizer()
    rows = [component_status(component) for component in component_registry()]
    missing = missing_required_components(rows)
    if not DATASET.exists():
        missing.append(str(DATASET.relative_to(ROOT)))
    if missing:
        print("Missing required files:")
        for item in missing:
            print(f"- {item}")
        return 1
    print(f"Structure OK - {len(rows)} components verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
