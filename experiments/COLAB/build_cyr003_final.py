import json, re, sys
from pathlib import Path

MANIFEST_SHA = "0dd9305697045b0fbf4e7f268b46a4d7276e4794af5d78b60e999df914ae4236"
PLAN_SHA = "c34cc072c9bc6fe0c666a0f5c9beca13cfd7b838"

worktree = Path(r"C:\Users\ankit\.zcode\workspace\default\An-Ra-cymek-500m")
builder = worktree / "experiments" / "COLAB" / "build_cyr003.py"
src = builder.read_text(encoding="utf-8")

# Extract raw-string blocks
m_harness = re.search(r"HARNESS = r'''(.*?)'''", src, re.DOTALL)
m_acq = re.search(r"CELL_ACQ = r'''(.*?)'''", src, re.DOTALL)
m_fork = re.search(r"CELL_ARK007 = r'''(.*?)'''", src, re.DOTALL)

if not all([m_harness, m_acq, m_fork]):
    print("ERROR: could not extract code blocks")
    sys.exit(1)

harness = m_harness.group(1).replace("MANIFEST_SHA_HERE", MANIFEST_SHA)
acq = m_acq.group(1)
fork = m_fork.group(1)

summary = "\n".join([
    "# SUMMARY + DOWNLOAD",
    "import json as _json",
    "summary = {'device': str(DEVICE), 'torch': torch.__version__, 'manifest': MANIFEST_SHA}",
    "print(_json.dumps(summary, indent=1, default=str))",
    "try:",
    "    from google.colab import files",
    "    import shutil",
    "    shutil.make_archive('/content/CYR-GPU-003_RESULTS', 'zip', RESULTS_DIR)",
    "    files.download('/content/CYR-GPU-003_RESULTS.zip')",
    "except Exception as exc:",
    "    print('manual download from', RESULTS_DIR, ':', exc)",
])

cells = []
cells.append({"cell_type": "markdown", "metadata": {},
              "source": ["# CYR-GPU-003: P35-scale LR-retention replication\n",
                         "\n",
                         "**Runtime: T4 GPU -> Run all -> ~60-90 min -> auto-download**\n",
                         "\n",
                         f"Manifest: {MANIFEST_SHA[:12]} | Plan: {PLAN_SHA[:12]}\n"]})
for code_src in [harness, acq, fork, summary]:
    cells.append({"cell_type": "code", "metadata": {},
                  "source": code_src.splitlines(keepends=True),
                  "outputs": [], "execution_count": None})

notebook = {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"colab": {"provenance": [], "name": "CYR-GPU-003.ipynb"},
                         "kernelspec": {"name": "python3", "display_name": "Python 3"},
                         "language_info": {"name": "python"},
                         "accelerator": "GPU"},
            "cells": cells}

out = worktree / "experiments" / "COLAB" / "CYR-GPU-003.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
parsed = json.loads(out.read_text(encoding="utf-8"))
for cell in parsed["cells"]:
    if cell["cell_type"] == "code":
        compile("".join(cell["source"]), "cell", "exec")
print(f"notebook written: {out} | cells: {len(parsed['cells'])} | all compile")
