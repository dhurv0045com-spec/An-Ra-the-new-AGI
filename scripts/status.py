from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anra_paths import ROOT, V2_TOKENIZER_FILE, inject_all_paths
from training.v2_runtime import canonical_v2_checkpoint, v2_report_path

inject_all_paths()


def _size(path: Path) -> str:
    return f"{path.stat().st_size / 1_048_576:.2f} MB" if path.exists() else "missing"


def main() -> None:
    graph = ROOT / "system_graph.json"
    if graph.exists():
        caps = json.loads(graph.read_text(encoding="utf-8")).get("capabilities", {})
        active = sum(1 for value in caps.values() if value)
        print(f"Capabilities: {active}/{len(caps)} active")
        for key, value in caps.items():
            print(f"  {'OK' if value else 'NO'} {key}")

    print("\nV2 Artifacts:")
    for label, path in [
        ("brain", canonical_v2_checkpoint("brain")),
        ("identity", canonical_v2_checkpoint("identity")),
        ("ouroboros", canonical_v2_checkpoint("ouroboros")),
        ("tokenizer", V2_TOKENIZER_FILE),
        ("eval_summary", v2_report_path("eval_summary")),
        ("curriculum", v2_report_path("curriculum")),
    ]:
        print(f"  {label:<12} {path.name:<32} {_size(path)}")

    for db in sorted((ROOT / "state").glob("*.db")):
        print(f"DB: {db.name} | {_size(db)}")


if __name__ == "__main__":
    main()
