from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anra_paths import OUTPUT_V2_DIR, ROOT, V2_TOKENIZER_FILE, get_v2_checkpoint, inject_all_paths
from runtime.system_registry import build_system_manifest

inject_all_paths()


def _v2_report_path(kind: str) -> Path:
    names = {
        "eval_summary": "v2_eval_summary.json",
        "curriculum": "v2_next_session_curriculum.json",
    }
    return OUTPUT_V2_DIR / names[kind]


def _size(path: Path) -> str:
    return f"{path.stat().st_size / 1_048_576:.2f} MB" if path.exists() else "missing"


def main() -> None:
    manifest = build_system_manifest(ROOT)
    caps = manifest.get("capabilities", {})
    active = sum(1 for value in caps.values() if value)
    print(f"Capabilities: {active}/{len(caps)} active")
    for key, value in caps.items():
        print(f"  {'OK' if value else 'NO'} {key}")
    metrics = manifest.get("metrics", {})
    print(
        "\nSource Metrics:"
        f"\n  source_files   {metrics.get('source_files', 0)}"
        f"\n  python_files   {metrics.get('python_files', 0)}"
        f"\n  markdown_files {metrics.get('markdown_files', 0)}"
        f"\n  python_lines   {metrics.get('python_lines', 0)}"
    )

    print("\nV2 Artifacts:")
    for label, path in [
        ("brain", get_v2_checkpoint("brain")),
        ("identity", get_v2_checkpoint("identity")),
        ("ouroboros", get_v2_checkpoint("ouroboros")),
        ("tokenizer", V2_TOKENIZER_FILE),
        ("eval_summary", _v2_report_path("eval_summary")),
        ("curriculum", _v2_report_path("curriculum")),
    ]:
        print(f"  {label:<12} {path.name:<32} {_size(path)}")

    for db in sorted((ROOT / "state").glob("*.db")):
        print(f"DB: {db.name} | {_size(db)}")


if __name__ == "__main__":
    main()
