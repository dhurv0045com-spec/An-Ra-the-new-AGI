"""Validate and normalize the canonical Arkenstone master Colab notebook.

The notebook itself is the bound executable artifact for the current GPU
campaign (MASTER_GPU_PLAN.md).  Earlier versions of this builder embedded a
second copy of the notebook source and could silently regenerate the obsolete
TPU/XLA path.  That duplication caused the `xla_sync` NameError seen in Colab.

This tool now fails closed: it validates the canonical notebook, compiles every
code cell, checks the binding plan and required campaigns, and only then writes
normalized JSON back to the same path.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "arkenstone_master.ipynb"
PLAN = HERE / "MASTER_GPU_PLAN.md"
PLAN_SHA = "3d98103cacc38177390df78ee0eff402da687fcf"


def main() -> int:
    if not NOTEBOOK.exists():
        raise FileNotFoundError(NOTEBOOK)
    if not PLAN.exists():
        raise FileNotFoundError(PLAN)

    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    if nb.get("nbformat") != 4:
        raise RuntimeError(f"unexpected nbformat: {nb.get('nbformat')}")

    code_cells = [
        "".join(cell.get("source", []))
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code"
    ]
    if not code_cells:
        raise RuntimeError("master notebook has no code cells")

    combined = "\n\n".join(code_cells)
    required = {
        "plan binding": PLAN_SHA,
        "GPU guard": "torch.cuda.is_available()",
        "xla_sync compatibility fix": "def xla_sync():",
        "fresh retention replication": "ARK-007R",
        "non-arithmetic transfer": "ARK-009",
        "recovery experiment": "ARK-010",
        "canonical manifest": "0dd9305697045b0fbf4e7f268b46a4d7276e4794af5d78b60e999df914ae4236",
    }
    missing = [name for name, token in required.items() if token not in combined]
    if missing:
        raise RuntimeError(f"master notebook missing required content: {missing}")

    for i, source in enumerate(code_cells):
        try:
            ast.parse(source, filename=f"cell_{i}.py")
        except SyntaxError as exc:
            raise RuntimeError(f"syntax error in code cell {i}: {exc}") from exc

    # Preserve the notebook as the executable source of truth; normalize only.
    NOTEBOOK.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print(f"PASS: {NOTEBOOK}")
    print(f"code_cells={len(code_cells)} plan_sha={PLAN_SHA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
