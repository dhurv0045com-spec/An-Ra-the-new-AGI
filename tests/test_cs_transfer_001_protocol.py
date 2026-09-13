"""Protocol/amendment qualification for CS-TRANSFER-001."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from anra_v5 import cs_transfer_001_data as data
from anra_v5 import cs_transfer_001_run as base
from anra_v5 import cs_transfer_001_run_v3 as v3

ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = ROOT / "experiments" / "CS_TRANSFER_001" / "AMENDMENT_1.json"
ENCODING_AUDIT = ROOT / "artifacts" / "e1" / "local_tournament" / "encoding-24576.json"


def test_amendment_is_prospective_and_no_outcomes_were_observed():
    a = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    assert a["status"] == "PROSPECTIVE_PREEXECUTION"
    assert a["outcomes_observed_before_amendment"] is False
    assert a["change"]["prompt_suffix_before"] == "\nAnswer:"
    assert a["change"]["prompt_suffix_after"] == "\n"
    assert a["change"]["answer_prefix_before"] == " "
    assert a["change"]["answer_prefix_after"] == ""


def test_committed_tokenizer_audit_proves_old_answer_delimiter_is_not_common_vocab():
    audit = json.loads(ENCODING_AUDIT.read_text(encoding="utf-8"))
    probe = next(x for x in audit["encodings"] if x["probe_id"] == "answer-01")
    assert 18224 in probe["token_ids"]
    assert 18224 >= 4096
    # Newline is directly observed as low-ID 202 in the same probe.
    assert 202 in probe["token_ids"]
    assert 202 < 4096


def test_effective_data_boundary_matches_amendment():
    assert data.PROMPT_SUFFIX == "\n"
    assert data.ANSWER_PREFIX == ""
    effective = v3.assert_effective_protocol()
    assert effective["prompt_suffix"] == "\n"
    assert effective["answer_prefix"] == ""


def test_protocol_hash_binds_both_preregistration_and_amendment():
    digest = hashlib.sha256()
    digest.update(base.PREREG_PATH.read_bytes())
    digest.update(b"\x00CS_TRANSFER_001_AMENDMENT_1\x00")
    digest.update(AMENDMENT.read_bytes())
    assert v3.protocol_sha256() == digest.hexdigest()


def test_v3_rebinds_base_persistent_identity_to_effective_protocol():
    assert base._prereg_sha() == v3.protocol_sha256()
    source = (ROOT / "anra_v5" / "cs_transfer_001_run_v3.py").read_text(encoding="utf-8")
    assert "r._prereg_sha = protocol_sha256" in source
    assert 'r._write_receipt("PROTOCOL", protocol)' in source
