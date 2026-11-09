"""V7.15 source-pinned owner notebook validation."""
import ast
import json
import re

import pytest
from tools.build_elendril_v715_notebook import build


def test_default_is_actual_scale_launch_without_attachment():
    sha = "a"*40
    notebook = build(sha)
    assert notebook["metadata"]["source_reference"] == sha
    assert notebook["nbformat"] == 4 and len(notebook["cells"]) == 3
    cfg = "".join(notebook["cells"][1]["source"])
    launch = "".join(notebook["cells"][2]["source"])
    ast.parse(cfg)
    ast.parse(launch)
    assert re.search(r"RUN_STUDY\s*=\s*True", cfg)
    assert 'PHASE = "scale"' in cfg
    assert "PARENT_ARCHIVE = None" in cfg
    assert "tools.elendril_v715_diagnostic_campaign" in launch
    assert "tools.elendril_v71_campaign" in launch
    assert "git" in launch and "SOURCE_SHA" in launch
    assert "pip install" not in json.dumps(notebook)


def test_rejects_nonimmutable_source():
    for bad in ("", "main", "Elendril", "x"*40, "0"*39):
        with pytest.raises(ValueError):
            build(bad)
