"""Science-S5 regression tests for long-run data/checkpoint amendment."""
import json
from pathlib import Path
from anra_v5 import formation_mux_train_v5 as train
from v5_experiments import formation_mux_protocol_v5 as proto
from v5_experiments import formation_mux_surface_v5 as surface


def test_s5_surface_budget_is_large_and_eval_sizes_unchanged():
    assert surface.PER_FAMILY_COUNTS == {"training": 10000, "development": 80, "sealed": 120}
    assert surface.TOTAL_COUNTS == {"training": 60000, "development": 480, "sealed": 720}
    assert proto.A_UPDATES * proto.BATCH_ROWS == 32000
    assert surface.TOTAL_COUNTS["training"] > proto.A_UPDATES * proto.BATCH_ROWS


def test_s5_checkpoint_policy_is_durable_without_changing_exposure():
    assert proto.A_UPDATES == 2000
    assert proto.B_PROCESSED_TOKEN_BUDGET == 500000
    assert proto.A_CHECKPOINT_EVERY_UPDATES == 200
    assert proto.B_CHECKPOINT_EVERY_TOKENS == 10000
    for exp in proto.EXPERIMENTS:
        payload = proto.protocol_payload(exp)
        assert payload["checkpoint_policy"]["exact_resume"] is True
        assert payload["checkpoint_policy"]["science_wall_time_stop"] is False
        assert payload["surface_counts_per_family"]["training"] == 10000


def test_progress_snapshot_written_at_checkpoint(tmp_path, monkeypatch):
    def fake_save(path, *, model, optimizers, torch, payload):
        Path(path).write_bytes(b"checkpoint")
        return "d" * 64
    monkeypatch.setattr(train, "_ORIGINAL_SAVE", fake_save)
    ckpt = tmp_path / "resume.pt"
    payload = {
        "experiment": proto.EXPERIMENT_A,
        "arm": "M0_STANDARD",
        "seed_bundle": proto.SEED_BUNDLES[0],
        "protocol_sha256": proto.protocol_sha(proto.EXPERIMENT_A),
        "data_manifest_sha256": "a" * 64,
        "updates": 200,
        "processed_tokens": 12345,
        "supervised_tokens": 2345,
        "trace": [{"axis": 100, "identity_exact_valid_eos": 0.1}, {"axis": 200, "identity_exact_valid_eos": 0.2}],
        "clip_events": 2,
        "timing": {"train_seconds": 1.0},
    }
    digest = train._save_checkpoint_with_progress(
        ckpt, model=None, optimizers={}, torch=None, payload=payload
    )
    assert digest == "d" * 64
    latest = json.loads((tmp_path / "LATEST_PROGRESS.json").read_text())
    assert latest["updates"] == 200
    assert latest["checkpoint_sha256"] == digest
    assert latest["latest_development"]["axis"] == 200
    immutable = tmp_path / "progress" / "UPDATE_00000200.json"
    assert immutable.exists()
    assert json.loads(immutable.read_text())["diagnostic_only"] is True
