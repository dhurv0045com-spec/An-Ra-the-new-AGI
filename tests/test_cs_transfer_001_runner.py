"""Source-contract tests for the audited CS-TRANSFER-001 runner."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "anra_v5" / "cs_transfer_001_run_v2.py"
BASE = ROOT / "anra_v5" / "cs_transfer_001_run.py"


def _src() -> str:
    return RUNNER.read_text(encoding="utf-8")


def test_audited_runner_retains_real_before_state_for_certification():
    src = _src()
    assert "before = state" in src
    assert "report = backend.step(before, batch)" in src
    assert "after = before.advance(" in src
    assert "before=before" in src
    assert "after=after" in src
    assert "state.__class__.from_dict" not in src


def test_resume_replays_missing_dev_eval_before_next_update():
    src = _src()
    resume_eval = src.index("if current_update in eval_updates")
    loop = src.index("for update_index in range(current_update, target):")
    assert resume_eval < loop
    block = src[resume_eval:loop]
    assert 'r._evaluate(backend, prepared["rows"]["development"])' in block
    assert "r._persist_dev" in block


def test_all_fixed_dev_checkpoints_must_exist_at_endpoint():
    src = _src()
    assert "expected_dev = sorted(eval_updates)" in src
    assert "observed_dev != expected_dev" in src
    assert "FAIL_CLOSED EVALUATION" in src


def test_checkpoint_receipt_precedes_publication_and_eval_follows_durable_checkpoint():
    src = _src()
    start = src.index("if due_checkpoint:")
    persist = src.index("r._persist_training", start)
    publish = src.index("published = store.publish", start)
    eval_pos = src.index("if due_eval and not any", publish)
    assert persist < publish < eval_pos


def test_no_mask_or_alternate_scientific_endpoint_in_audited_runner():
    src = _src()
    assert "MASK_4096" not in src
    assert "target = int(p[\"training\"][\"target_updates\"])" in src
    assert "checkpoint_every = int(p[\"training\"][\"checkpoint_every_updates\"])" in src


def test_base_runner_sealed_firewall_and_pair_aggregation_are_present():
    src = BASE.read_text(encoding="utf-8")
    assert "SEALED_CONSUMPTION.json" in src
    assert "development_aggregate()" in src
    assert "if SEALED_LOCK.exists():" in src
    assert '"status": "STARTED"' in src
    assert 'marker["status"] = "CONSUMED_AND_FINALIZED"' in src
    assert "mean_paired_identity_auc_gap" in src
    assert "SUPPORTED_PHYSICAL_CLASS_SPACE" in src
