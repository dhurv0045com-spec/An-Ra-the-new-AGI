"""
Ensure anra_training.txt exists. Run once per fresh clone.
Called automatically from AnRa_Master.ipynb Cell 2.
"""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent.parent
canonical = ROOT / "training_data" / "anra_training.txt"
legacy = ROOT / "training_data" / "anra_dataset_v6_1.txt"

if canonical.exists():
    print(f"[setup] {canonical.name} already exists ({canonical.stat().st_size:,} bytes).")
elif legacy.exists():
    shutil.copy2(legacy, canonical)
    print(f"[setup] Copied {legacy.name} -> {canonical.name} ({canonical.stat().st_size:,} bytes).")
else:
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_text(
        "H: Who are you?\nANRA: I am An-Ra. I was built from scratch.\n\n"
        "H: What can you do?\nANRA: I can converse, write code, and learn.\n\n",
        encoding="utf-8",
    )
    print(f"[setup] Created bootstrap {canonical.name}. Add real training data to Drive.")

if __name__ == "__main__":
    pass
