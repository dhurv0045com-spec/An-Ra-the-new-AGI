"""Verify the shipped artifact, not only a hypothetical notebook build."""
import json
from pathlib import Path
import subprocess
import io
import os
import sys
import tarfile

import nbformat

def test_published_notebook_matches_immutable_source_and_generator(tmp_path):
    root = Path(__file__).resolve().parents[1]
    path = root / "notebooks/ELENDRIL_V6_KAGGLE_T4X2.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    nbformat.validate(nbformat.from_dict(notebook))
    source = notebook["metadata"]["source_reference"]
    assert len(source) == 40 and all(c in "0123456789abcdef" for c in source)
    subprocess.run(["git", "merge-base", "--is-ancestor", source, "HEAD"], cwd=root, check=True)
    # A later runtime may legitimately coexist with an older published artifact.
    # Rebuild from that artifact's own exact source, not the development tree.
    archive = subprocess.check_output(["git", "archive", "--format=tar", source], cwd=root)
    checkout = tmp_path / "source"
    checkout.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive)) as handle:
        handle.extractall(checkout, filter="data")
    rebuilt = tmp_path / "rebuilt.ipynb"
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    subprocess.run([sys.executable, "-m", "tools.build_elendril_v6_notebook",
                    "--source-sha", source, "--output", str(rebuilt)],
                   cwd=checkout, env=environment, check=True, capture_output=True, timeout=60)
    assert notebook == json.loads(rebuilt.read_text(encoding="utf-8"))
