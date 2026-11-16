"""Verified historical S3 restoration, D0/D1 260-way binary logits, and D2 probes/patching tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from anra_v5.elendril.representation import UTF8
from anra_v5.elendril.training import configure_cpu
from anra_v5.elendril.y1_data import build_d2_probe_dataset, build_development_suite
from anra_v5.elendril.y1_diagnostics import (
    extract_and_restore_historical_s3,
    reproduce_historical_hh_counts,
    run_d0_d1_model_audit,
    run_d2_probe_and_patching,
    verify_and_inspect_hh_full_archive,
)
from anra_v5.elendril.y1_training import CPU_CANARY_SPEC, create_y1_trainer
from tools.elendril_y1_worker import _canary_eval_slice
from v5_model.core import initialize


HH_ZIP = Path("C:/Users/ankit/Downloads/ELENDRIL_HITCHHIKER_FULL.zip")


def test_y1_d0_d1_and_d2_probe_on_tiny_model(tmp_path: Path) -> None:
    configure_cpu()
    dev = build_development_suite()
    canary = _canary_eval_slice(dev)
    from anra_v5.elendril.y1_data import _case_to_decision_samples
    tiny_samples = [s for c in canary["cases"][:8] for s in _case_to_decision_samples(c)[0]]
    while len(tiny_samples) < 16:
        tiny_samples.extend(tiny_samples)
    from dataclasses import replace
    tiny_samples = tuple(replace(s, task_id=f"diag/{i}/{s.task_id}") for i, s in enumerate(tiny_samples[:16]))

    trainer, _ = create_y1_trainer(
        job={"job_id": "Y1-diag-tiny", "profile": "S3", "seed": 7, "updates": 2, "batch_size": 8, "micro_batch_size": 2, "lr": 1e-4, "group_ce_weight": 0.0},
        samples=tiny_samples,
        data_provenance={"schema": "test", "split": "train", "license": "repository LICENSE"},
        source_commit="64ed23a433ebb1ce2f620e0db8dcc24f4d8c23be",
        root=tmp_path / "cp",
        device="cpu",
        cpu_canary=True,
        spec_override=CPU_CANARY_SPEC,
        updates_override=2,
        batch_override=8,
    )
    trainer.step()
    sha = trainer.save()

    npz_file = tmp_path / "logits.npz"
    audit = run_d0_d1_model_audit(
        trainer.backend,
        canary,
        job_id="Y1-diag-tiny",
        checkpoint_sha256=sha,
        logits_npz_path=npz_file,
        run_supplemental_base=True,
    )
    assert npz_file.is_file()
    loaded = np.load(npz_file)
    assert loaded["base_logits"].dtype == np.float32
    assert loaded["base_logits"].shape[1] == UTF8.vocabulary_size == 260
    assert "descriptive_rescue_constrained_minus_raw" in audit["d1_summary"]

    # D2 deferred by default, executes cleanly when enabled
    d2_def = run_d2_probe_and_patching(trainer.backend, {}, enabled=False)
    assert d2_def["status"] == "DEFERRED_NOT_ADMITTED"

    d2 = build_d2_probe_dataset()
    small_d2 = {
        "seed": d2["seed"],
        "train_cases": tuple(c for h in (1, 2, 3, 4) for c in [x for x in d2["train_cases"] if x.hops == h][:20]),
        "test_cases": tuple(c for h in (1, 2, 3, 4) for c in [x for x in d2["test_cases"] if x.hops == h][:20]),
    }
    d2_run = run_d2_probe_and_patching(trainer.backend, small_d2, enabled=True)
    assert d2_run["status"] == "EXECUTED"
    assert set(d2_run["hops"]) == {"hop-1", "hop-2", "hop-3", "hop-4"}


@pytest.mark.skipif(not HH_ZIP.is_file(), reason="external ELENDRIL_HITCHHIKER_FULL.zip absent")
def test_y1_real_hitchhiker_archive_verification_and_checkpoint_restore(tmp_path: Path) -> None:
    info = verify_and_inspect_hh_full_archive(HH_ZIP)
    assert info["status"] == "VERIFIED"
    assert info["historical_s20_status"] == "NOT_AVAILABLE"
    assert len(info["verified_lineages"]) == 15

    restored = extract_and_restore_historical_s3(HH_ZIP, "HH-stage1_s3_screen-S3-C-s7", tmp_path / "ext", device="cpu")
    assert restored["status"] == "RESTORED_VERIFIED"
    assert restored["global_update"] == 2368
    assert restored["parameters"] == 3084800
    assert restored["checkpoint_sha256"] == "4465f6b4c420a2d5bd90a55b893cd0110c94a6e663800ed62262d2b08dd01cd4"

    hrep = reproduce_historical_hh_counts(HH_ZIP, "HH-stage1_s3_screen-S3-C-s7", restored["backend"], max_cases_per_cohort=8)
    assert hrep["status"] == "EXACT_MATCH"
    assert hrep["mismatch_blocker"] is False
