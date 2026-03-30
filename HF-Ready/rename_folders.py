"""
rename_folders.py — Renames all 45X folders to descriptive names with (45X) suffix.
Also updates ALL references in Python files, Markdown files, and YAML files.

Run: python rename_folders.py
"""

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).parent

# ─── Folder rename map ───────────────────────────────────────────────────────
RENAMES = {
    # Phase 2
    "fine_tuning (45I)":  "phase2/fine_tuning (45I)",
    "memory (45J)":  "phase2/memory (45J)",
    "agent_loop (45k)":  "phase2/agent_loop (45k)",
    "self_improvement (45l)":  "phase2/self_improvement (45l)",
    "master_system (45M)":  "phase2/master_system (45M)",
    # Phase 3
    "identity (45N)":  "phase3/identity (45N)",
    "ouroboros (45O)":  "phase3/ouroboros (45O)",
    "ghost_memory (45P)":  "phase3/ghost_memory (45P)",
    "symbolic_bridge (45Q)":  "phase3/symbolic_bridge (45Q)",
    "sovereignty (45R)":  "phase3/sovereignty (45R)",
}

# Build string replacements (order: longest first to avoid partial matches)
# We need to handle both forward-slash and backslash paths
STRING_REPLACEMENTS = []
for old, new in RENAMES.items():
    old_name = old.split("/")[-1]  # e.g. "45I"
    new_name = new.split("/")[-1]  # e.g. "fine_tuning (45I)"

    # Path-style replacements (in sys.path, Path() etc.)
    STRING_REPLACEMENTS.append((f'"{old}"',  f'"{new_name}"'))   # "45I" -> "fine_tuning (45I)"
    STRING_REPLACEMENTS.append((f"'{old}'",  f"'{new_name}'"))
    STRING_REPLACEMENTS.append((f'/ "{old_name}"', f'/ "{new_name}"'))
    STRING_REPLACEMENTS.append((f"/ '{old_name}'", f"/ '{new_name}'"))

    # Forward slash paths
    STRING_REPLACEMENTS.append((f"/{old_name}/", f"/{new_name}/"))
    STRING_REPLACEMENTS.append((f"/{old_name}\"", f"/{new_name}\""))
    STRING_REPLACEMENTS.append((f"/{old_name}'", f"/{new_name}'"))
    STRING_REPLACEMENTS.append((f"/{old_name})", f"/{new_name})"))

    # Backslash paths (rare but possible)
    STRING_REPLACEMENTS.append((f"\\{old_name}\\", f"\\{new_name}\\"))
    STRING_REPLACEMENTS.append((f"\\{old_name}\"", f"\\{new_name}\""))

# Sort by length (longest first) to avoid partial replacements
STRING_REPLACEMENTS.sort(key=lambda x: -len(x[0]))

# File extensions to update
EXTENSIONS = {".py", ".md", ".txt", ".yaml", ".yml", ".json"}


def update_file_contents(filepath):
    """Replace all old folder references in a file."""
    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False

    original = content
    for old, new in STRING_REPLACEMENTS:
        content = content.replace(old, new)

    if content != original:
        filepath.write_text(content, encoding="utf-8")
        return True
    return False


def main():
    print("=" * 60)
    print("  AN-RA FOLDER RENAME")
    print("  45X -> descriptive_name (45X)")
    print("=" * 60)

    # Step 1: Update file contents BEFORE renaming folders
    print("\n[Step 1] Updating file references...")
    updated = 0
    for f in ROOT.rglob("*"):
        if f.is_file() and f.suffix in EXTENSIONS and ".git" not in str(f) and "__pycache__" not in str(f):
            if update_file_contents(f):
                print(f"  Updated: {f.relative_to(ROOT)}")
                updated += 1
    print(f"  {updated} files updated.\n")

    # Step 2: Rename folders (do it in reverse depth order to avoid conflicts)
    print("[Step 2] Renaming folders...")
    for old_rel, new_rel in RENAMES.items():
        old_path = ROOT / old_rel
        new_path = ROOT / new_rel
        if old_path.exists():
            if new_path.exists():
                print(f"  SKIP (already exists): {new_rel}")
            else:
                shutil.move(str(old_path), str(new_path))
                print(f"  Renamed: {old_rel} -> {new_rel}")
        else:
            print(f"  NOT FOUND: {old_rel}")

    # Step 3: Update anra.py specifically (it uses the phase3 folder names in a loop)
    anra_py = ROOT / "anra.py"
    if anra_py.exists():
        content = anra_py.read_text(encoding="utf-8")
        # Replace the for loop that iterates over phase3 folders
        old_loop = '["45N", "45O", "45P", "45Q", "45R"]'
        new_loop = '["identity (45N)", "ouroboros (45O)", "ghost_memory (45P)", "symbolic_bridge (45Q)", "sovereignty (45R)"]'
        if old_loop in content:
            content = content.replace(old_loop, new_loop)
            anra_py.write_text(content, encoding="utf-8")
            print(f"\n  Fixed anra.py phase3 loop")

        # Also fix the PHASE2_45M path
        content = anra_py.read_text(encoding="utf-8")
        old_45m = 'PHASE2_45M   = PROJECT_ROOT / "phase2" / "master_system (45M)"'
        new_45m = 'PHASE2_45M   = PROJECT_ROOT / "phase2" / "master_system (45M)"'
        if old_45m in content:
            content = content.replace(old_45m, new_45m)
            anra_py.write_text(content, encoding="utf-8")
            print(f"  Fixed anra.py PHASE2_45M path")

    print("\n" + "=" * 60)
    print("  RENAME COMPLETE!")
    print("=" * 60)
    print("\nNew folder structure:")
    for old_rel, new_rel in sorted(RENAMES.items()):
        new_name = new_rel.split("/")[-1]
        print(f"  {old_rel.split('/')[-1]:5s} -> {new_name}")


if __name__ == "__main__":
    main()
