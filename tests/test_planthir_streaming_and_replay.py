"""Stage 5 verification: streaming chunked raw evidence, capsule provenance links, custody, and independent replay."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from anra_v5.elendril.v8.data import D1_TRAIN_SEED, build_d1_training_dataset
from anra_v5.planthir import CartridgeArmSpec, CartridgeRunner, ExperimentCartridge
from anra_v5.planthir.adapters.x1_rd2_adapter import (
    X1RD2EnvironmentKernel,
    X1RD2Recipe,
    X1RD2Spec,
    prepare_x1_rd2_training_data,
)
from anra_v5.planthir.causal import _probe_split_keys
from anra_v5.planthir.evidence import (
    stream_raw_evidence_chunks,
    verify_raw_evidence_chunks,
    verify_tier2_capsule,
    write_tier2_diagnostic_capsule,
)
from anra_v5.planthir.replay import build_custody_receipt, independent_replay_bundle
from anra_v5.planthir.resources import ResourceQuotaExceededError

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMIT = "00b2128b34755d332414ffae8eb78dd3f430e04f"


def _tiny_x1_cartridge() -> tuple[ExperimentCartridge, list]:
    data, train_worlds = prepare_x1_rd2_training_data(num_groups=4)
    _, forbidden = _probe_split_keys(train_worlds)
    pool = build_d1_training_dataset(seed=D1_TRAIN_SEED + 41, quota_per_family=16).worlds
    disjoint = [w for w in pool if int(w["renderer_template"]) == 0 and not (_probe_split_keys([w])[1] & forbidden)]
    eval_w = disjoint[:4]
    spec = X1RD2Spec(architecture="X1-R", hidden_dim=16, num_layers=1, num_heads=2, pointer_dim=8, num_slots=2, slot_dim=8, slot_edge_hidden=8, max_sequence_length=512)
    recipe = X1RD2Recipe(seed=7, total_updates=2, batch_size=8, micro_batch_size=4, lr=1e-3, objective="pair_eq")
    arm = CartridgeArmSpec(
        arm_id="X1R",
        architecture_family="X1-R",
        is_control=True,
        v8_spec=None,
        recipe=recipe,
        dataset_identity=data.data_identity,
        milestone_updates=(1, 2),
        backend_id="x1_rd2",
        model_spec=spec,
    )
    cart = ExperimentCartridge(
        campaign_id="STREAM-REPLAY",
        program="X1-RD2",
        source_commit=COMMIT,
        kernel=X1RD2EnvironmentKernel(),
        control_arm=arm,
        treatment_arms=(),
        train_data=data,
        eval_worlds=tuple(eval_w),
        multihop_worlds=tuple(disjoint[4:6]),
        probe_train_worlds=tuple(disjoint[6:10]),
        train_worlds=train_worlds,
        backend_id="x1_rd2",
    )
    return cart, eval_w


def test_streaming_raw_evidence_chunks_and_write_time_quota(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw_stream"

    def _gen(n: int):
        rng = np.random.default_rng(7)
        for i in range(n):
            yield {
                "row_index": i,
                "world_id": f"w-{i}",
                "correct": bool(i % 2 == 0),
                "arrays": {"hidden": rng.standard_normal(8).astype(np.float32)},
            }

    manifest = stream_raw_evidence_chunks(
        raw_dir,
        _gen(45),
        max_chunk_records=12,
        max_chunk_bytes=64 * 1024,
        max_total_bytes=1024 * 1024,
    )
    assert manifest["chunk_count"] == 4
    assert manifest["total_records"] == 45
    assert verify_raw_evidence_chunks(raw_dir)["verified"] is True

    # Tampering chunk fails verification fail-closed
    c1 = raw_dir / "chunk-0001.jsonl"
    c1.write_text(c1.read_text(encoding="utf-8") + '{"tampered": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="checksum|mismatch"):
        verify_raw_evidence_chunks(raw_dir)

    # Tight write-time byte quota aborts early without draining an entire multi-thousand generator
    yielded = 0

    def _huge_stream():
        nonlocal yielded
        for i in range(1000):
            yielded += 1
            yield {
                "row_index": i,
                "payload": "X" * 200,
                "arrays": {"vec": np.ones(32, dtype=np.float32)},
            }

    quota_dir = tmp_path / "quota_stream"
    with pytest.raises(ResourceQuotaExceededError) as exc:
        stream_raw_evidence_chunks(
            quota_dir,
            _huge_stream(),
            max_chunk_records=4,
            max_chunk_bytes=1024,
            max_total_bytes=1200,
        )
    assert exc.value.quota_name == "max_raw_evidence_bytes"
    assert yielded < 50


def test_capsule_records_raw_manifest_link_and_lossy_quantization_metadata(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_manifest = stream_raw_evidence_chunks(
        raw_dir,
        ({"row_index": i, "arrays": {"decision_hidden": np.ones(8, dtype=np.float32) * i}} for i in range(10)),
        max_chunk_records=5,
        max_total_bytes=1024 * 1024,
    )
    cap_dir = tmp_path / "exp1" / "capsule"
    write_tier2_diagnostic_capsule(
        cap_dir,
        experiment_id="EXP1",
        probe_arrays={
            "decision_hidden": np.ones((10, 8), dtype=np.float32),
            "unaligned_scalar": np.array(1.0, dtype=np.float32),
        },
        diagnostics={"metric": 0.75},
        raw_evidence_link={
            "raw_manifest_sha256": raw_manifest["manifest_sha256"],
            "omitted_fields": ["raw_step_logits"],
        },
    )
    cap = verify_tier2_capsule(cap_dir)
    link = cap["raw_evidence_link"]
    assert link["raw_manifest_sha256"] == raw_manifest["manifest_sha256"]
    assert link["selected_row_indices"] == list(range(10))
    assert "unaligned_scalar" in link["omitted_fields"]
    assert "raw_step_logits" in link["omitted_fields"]
    assert link["lossy_quantization"]["float_cast"] == "float16"
    assert "decision_hidden" in link["lossy_quantization"]["quantized_keys"]


def test_custody_receipt_and_independent_subprocess_replay(tmp_path: Path) -> None:
    cart, eval_w = _tiny_x1_cartridge()
    work = tmp_path / "run"
    out = CartridgeRunner().run(cart, work_root=work, checkpoint_interval=1)
    bdir = work / "result_bundle"

    custody = build_custody_receipt(
        bdir,
        expected_source_commit=COMMIT,
        output_path=work / "CUSTODY_RECEIPT.json",
    )
    assert custody["verified"] is True
    assert (work / "CUSTODY_RECEIPT.json").is_file()

    # In-process isolated temp replay
    replay = independent_replay_bundle(
        bdir,
        expected_source_commit=COMMIT,
        worlds=eval_w,
        output_path=work / "REPLAY_RECEIPT.json",
    )
    assert replay["verified"] is True
    assert replay["status"] == "REPLAY_VERIFIED"
    assert replay["restored_lineages"][0]["checkpoint_sha256"] == out["arms"]["X1R"]["checkpoint_sha256"]
    assert replay["restored_lineages"][0]["finite_hidden"] is True

    # Separate clean subprocess replay verifying independent_environment=True
    code = (
        "import json; from anra_v5.planthir.replay import independent_replay_bundle; "
        f"r = independent_replay_bundle({str(bdir)!r}, expected_source_commit={COMMIT!r}, independent_environment=True); "
        "print(json.dumps({'verified': r['verified'], 'status': r['status'], 'indep': r['independent_environment']}))"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    parsed = json.loads(proc.stdout.strip().splitlines()[-1])
    assert parsed == {"verified": True, "status": "REPLAY_VERIFIED", "indep": True}

    # Tampering sidecar digest fails custody closed
    resolved_gen = Path(custody["resolved_generation_dir"])
    (resolved_gen / "result.zip.sha256").write_text("0" * 64 + "  result.zip\n", encoding="ascii")
    with pytest.raises(ValueError, match="SIDECAR_SHA256_MISMATCH"):
        build_custody_receipt(bdir, expected_source_commit=COMMIT)
