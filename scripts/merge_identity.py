from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anra_paths import IDENTITY_DIR

IDENTITY_FILES = [
    IDENTITY_DIR / "anra_identity_v2.txt",
    IDENTITY_DIR / "anra_identity_v3_coding.txt",
    IDENTITY_DIR / "anra_identity_v4_fluent.txt",
]
OUTPUT = IDENTITY_DIR / "anra_identity_combined.txt"


def main() -> None:
    merged = []
    for f in IDENTITY_FILES:
        if not f.exists():
            print(f"WARNING: {f} not found — skipping")
            continue
        merged.append(f.read_text(encoding="utf-8", errors="ignore"))
        print(f"Added {f.name} ({f.stat().st_size / 1e3:.1f}KB)")

    OUTPUT.write_text("\n\n".join(merged), encoding="utf-8")
    print(f"Written: {OUTPUT} ({OUTPUT.stat().st_size / 1e3:.1f}KB total)")
    print("Run training/finetune_anra.py after this.")


if __name__ == "__main__":
    main()
