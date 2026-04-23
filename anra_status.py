from __future__ import annotations

import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def db_size(path: Path) -> str:
    return f"{path.stat().st_size / 1_048_576:.2f} MB" if path.exists() else "missing"


def main():
    graph = ROOT / "system_graph.json"
    if graph.exists():
        caps = json.loads(graph.read_text()).get("capabilities", {})
        active = sum(1 for v in caps.values() if v)
        print(f"Capabilities: {active}/{len(caps)} active")
        for k, v in caps.items():
            print(f"  {'✅' if v else '❌'} {k}")

    ckpts = [ROOT / "anra_ouroboros.pt", ROOT / "anra_brain_identity.pt", ROOT / "anra_brain.pt"]
    for ck in ckpts:
        if ck.exists():
            print(f"Checkpoint: {ck.name} | {ck.stat().st_size / 1_048_576:.1f} MB")
            break

    for db in sorted((ROOT / "state").glob("*.db")):
        print(f"DB: {db.name} | {db_size(db)}")


if __name__ == "__main__":
    main()
