from __future__ import annotations

from anra_paths import DATASET, ROOT
from runtime.system_registry import component_registry, component_status, missing_required_components



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
    optional_missing = []
    if not DATASET.exists():
        optional_missing.append(str(DATASET.relative_to(ROOT)))
    if missing:
        print("ERROR: Missing required structural components:")
        for item in missing:
            print(f"  - {item}")
        return 1
    if optional_missing:
        print("INFO: Optional artifacts not present (expected in CI):")
        for item in optional_missing:
            print(f"  - {item}")
    print(f"Structure OK — {len(rows)} components verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
