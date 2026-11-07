"""Verify the shipped artifact, not only a hypothetical notebook build."""
import json
from pathlib import Path
import subprocess

import nbformat

from tools.build_elendril_v6_notebook import build


def test_published_notebook_matches_immutable_source_and_generator():
    root = Path(__file__).resolve().parents[1]
    path = root / "notebooks/ELENDRIL_V6_KAGGLE_T4X2.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    nbformat.validate(nbformat.from_dict(notebook))
    source = notebook["metadata"]["source_reference"]
    assert source == "1cb3f37772d25c7569314031014d5372d490ad2e"
    assert notebook == build(source)
    for name in ("tools/elendril_v6_campaign.py", "tools/elendril_v6_worker.py",
                 "tools/elendril_v6_export.py", "tools/verify_elendril_v6.py",
                 "tools/build_elendril_v6_notebook.py",
                 "anra_v5/elendril/v6_registry.py", "anra_v5/elendril/v6_data.py",
                 "anra_v5/elendril/v6_training.py", "anra_v5/elendril/v6_evaluation.py"):
        committed = subprocess.check_output(["git", "show", f"{source}:{name}"], cwd=root)
        assert (root / name).read_bytes().replace(b"\r\n", b"\n") == committed.replace(b"\r\n", b"\n"), name
