"""Acceptance tests for Avendi V8-Y2 P1 checkpoint restoration, Y1 legacy byte adapter, and case-by-case agreement."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import numpy as np
import pytest

from anra_v5.elendril.avendi.frozen import (
    build_deterministic_retraining_recipe,
    run_p1_and_p2_frozen_session,
    verify_regenerated_y1_case_agreement,
)
from anra_v5.elendril.avendi.module import AvendiY1LegacyByteAdapter
from anra_v5.elendril.avendi.objectives import CPU_CANARY_S3_SPEC
from anra_v5.elendril.avendi.protocol import Y1_BASELINE_LINEAGES
from anra_v5.elendril.avendi.worlds import build_avendi_suite
from anra_v5.elendril.backend import BackendIdentity, NeuralBackend
from anra_v5.elendril.representation import UTF8
from anra_v5.elendril.y1_data import build_development_suite
from v5_model.core import initialize


def test_y1_legacy_byte_adapter_token_for_token_equivalence() -> None:
    model = initialize(CPU_CANARY_S3_SPEC, seed=7)
    backend = NeuralBackend(
        model,
        BackendIdentity(
            CPU_CANARY_S3_SPEC.sha256(),
            UTF8.identity,
            CPU_CANARY_S3_SPEC.vocabulary_size,
            CPU_CANARY_S3_SPEC.context_length,
            checkpoint_sha256="0" * 64,
        ),
        pad_id=UTF8.pad_id,
        bos_id=UTF8.bos_id,
        eos_id=UTF8.eos_id,
    )
    adapter = AvendiY1LegacyByteAdapter(backend, model_identity="canary-s3")

    y1_dev = build_development_suite()
    ref_cases = [c for c in y1_dev["cases"] if c.panel == "main-base"][:4]
    for yc in ref_cases:
        eq = adapter.verify_token_equivalence_with_y1_case(yc)
        assert eq["matched"] is True
        assert eq["first_logits_shape"] == [260]


def test_regenerated_case_agreement_rejects_exchanged_predictions_and_logit_drift() -> None:
    ckpt_sha = "57ab8b7292a84689ee32d797257eb4a2c9c8cd3c95421df2ca254801c6cd1706"
    p1 = "Output A-P\nFollow 1 links from A. Query=0 Memory=0\nA=G@1000/s0\nAction:"
    p2 = "Output A-P\nFollow 1 links from B. Query=0 Memory=0\nB=H@1000/s0\nAction:"
    r1_sha = sha256(p1.encode("utf-8")).hexdigest()
    r2_sha = sha256(p2.encode("utf-8")).hexdigest()

    logits_1 = np.zeros(260, dtype=np.float32)
    logits_1[71] = 3.5
    logits_2 = np.zeros(260, dtype=np.float32)
    logits_2[72] = 4.0

    ref_rows = [
        {"case_id": "c1", "checkpoint_sha256": ckpt_sha, "decoder": "utf8-raw", "rendered_input_sha256": r1_sha, "raw_tokens": [71, 2], "first_logits": logits_1},
        {"case_id": "c2", "checkpoint_sha256": ckpt_sha, "decoder": "utf8-raw", "rendered_input_sha256": r2_sha, "raw_tokens": [72, 2], "first_logits": logits_2},
    ]
    prompt_map = {"c1": p1, "c2": p2}

    # 1. Exact agreement passes
    ok_res = verify_regenerated_y1_case_agreement(
        job_id="Y1-t1_factorial-S3-F00-s7",
        checkpoint_sha256=ckpt_sha,
        regenerated_rows=ref_rows,
        reference_rows=ref_rows,
        reference_Prompt_by_case_id=prompt_map,
    )
    assert ok_res["status"] == "PASS"
    assert ok_res["original_predictions_match"] is True
    assert ok_res["max_abs_logit_diff"] <= 1e-5

    # 2. Exchanging predictions between c1 and c2 (same aggregate count!) MUST fail agreement
    exchanged_rows = [
        {**ref_rows[0], "raw_tokens": [72, 2]},
        {**ref_rows[1], "raw_tokens": [71, 2]},
    ]
    ex_res = verify_regenerated_y1_case_agreement(
        job_id="Y1-t1_factorial-S3-F00-s7",
        checkpoint_sha256=ckpt_sha,
        regenerated_rows=exchanged_rows,
        reference_rows=ref_rows,
        reference_Prompt_by_case_id=prompt_map,
    )
    assert ex_res["status"] == "FAIL"
    assert ex_res["original_predictions_match"] is False
    assert ex_res["mismatch_count"] == 2

    # 3. Perturbing a logit by 1e-3 (> 1e-5 tolerance) MUST fail agreement while reporting tolerance sensitivity
    drifted_logits = logits_1.copy()
    drifted_logits[71] += 1e-3
    drift_rows = [
        {**ref_rows[0], "first_logits": drifted_logits},
        ref_rows[1],
    ]
    drift_res = verify_regenerated_y1_case_agreement(
        job_id="Y1-t1_factorial-S3-F00-s7",
        checkpoint_sha256=ckpt_sha,
        regenerated_rows=drift_rows,
        reference_rows=ref_rows,
        reference_Prompt_by_case_id=prompt_map,
    )
    assert drift_res["status"] == "FAIL"
    assert drift_res["tolerance_sensitivity_counts"]["1e-05"] == 1
    assert drift_res["tolerance_sensitivity_counts"]["0.001"] == 2


def test_missing_y1_full_archive_produces_blocked_receipt_and_retraining_recipe(tmp_path: Path) -> None:
    suite = build_avendi_suite(scale_factor=64)
    receipt = run_p1_and_p2_frozen_session(
        suite=suite,
        output_dir=tmp_path / "frozen_missing",
        checkpoint_root=None,
        evidence_path=None,
    )
    assert receipt["p1_stage_status"] == "BLOCKED"
    assert receipt["verification_rows"]["checkpoint_restore"] == "UNAVAILABLE"
    assert receipt["restored_lineages_count"] == 0
    assert receipt["expected_lineages_count"] == 6
    assert set(receipt["missing_lineages"]) == set(Y1_BASELINE_LINEAGES)
    assert receipt["retraining_recipe_if_missing"] is not None
    assert receipt["retraining_recipe_if_missing"]["status"] == "REQUIRES_FRESH_IDENTITY_AND_OWNER_COMPUTE_APPROVAL"
