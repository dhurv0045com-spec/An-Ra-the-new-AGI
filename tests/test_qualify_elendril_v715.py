"""V7.15 CPU-only preparation and immutable output custody checks."""
import hashlib
import json
from pathlib import Path

import pytest

from tools.qualify_elendril_v715 import prepare
from anra_v5.elendril.v71_data import suite_from_record
from anra_v5.elendril.v715_diagnostics import audit


def test_prepare_writes_replayable_dataset_and_receipt(tmp_path):
    path = tmp_path / "prepared"
    report = prepare(path, worlds=2)
    assert report["status"] == "PASS"
    assert report["case_records"] == 48
    assert report["cohorts"] == 24
    assert report["training_disjoint"]["status"] == "NOT_RUN"
    data = (path / "V715_ROUTE_SUITE.json").read_bytes()
    receipt = json.loads((path / "V715_ROUTE_AUDIT.json").read_text())
    assert receipt["portable_file_sha256"] == hashlib.sha256(data).hexdigest()
    suite = suite_from_record(json.loads(data))
    assert audit(suite, seed=153071)["status"] == "PASS"
    assert report["scientific_status"].startswith("NOT_RUN")
    with pytest.raises(FileExistsError):
        prepare(path, worlds=2)


def test_invalid_request_leaves_no_published_output(tmp_path):
    path = tmp_path / "invalid"
    with pytest.raises(ValueError):
        prepare(path, worlds=0)
    assert not path.exists()
