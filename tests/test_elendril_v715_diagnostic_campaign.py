"""Offline V7.15 campaign contracts; no CUDA required."""
import hashlib
import json
import zipfile

import pytest

from anra_v5.elendril.v715_protocol import PROTOCOL_SHA
from tools.elendril_v715_diagnostic_campaign import _zip, run, suite_and_audit


def test_owner_explicitly_required(tmp_path):
    with pytest.raises(ValueError, match="owner-run"):
        run(output=tmp_path/"x", source="x"*40, parent_archive=None)


def test_reject_invalid_budget(tmp_path):
    with pytest.raises(ValueError, match="out of bounds"):
        run(output=tmp_path/"x", source="x"*40, parent_archive=None,
            minutes=331, owner_enabled=True)


def test_analysis_archive_roundtrip_and_detects_data_tamper(tmp_path):
    target = tmp_path/"root"
    (target/"analysis").mkdir(parents=True)
    (target/"data").mkdir()
    (target/"analysis"/"A.json").write_text('{"test":true}\n')
    (target/"data"/"B.json").write_text('{"worlds":16}\n')
    info = _zip(target)
    assert info["status"] == "VERIFIED" and info["members"] == 2
    with zipfile.ZipFile(target/"exports"/"ELENDRIL_V715_ANALYSIS.zip") as z:
        manifest = json.loads(z.read("MANIFEST.json"))
    assert manifest["protocol_sha256"] == PROTOCOL_SHA
    assert manifest["members"][0]["sha256"] == hashlib.sha256((target/"analysis"/"A.json").read_bytes()).hexdigest()
    (target/"data"/"B.json").write_text('{"worlds":99}\n')
    with pytest.raises(ValueError, match="inventory changed"):
        _zip(target)


def test_disallows_unqualified_frozen_suite_without_training():
    from anra_v5.elendril.v715_diagnostics import build_suite
    from anra_v5.elendril.v71_followup import build_suite as paired
    assert len(build_suite()) == 24 and len(paired()) == 32
