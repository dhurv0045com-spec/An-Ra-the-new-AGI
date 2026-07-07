from __future__ import annotations

import json

from training.gepa import build_gepa_report, write_gepa_report
from training.v2_runtime import v2_report_path


def test_gepa_builds_trace_backed_candidates_from_eval_failures() -> None:
    report = build_gepa_report(
        eval_summary={
            "results": [
                {
                    "id": "symbolic_math",
                    "category": "symbolic",
                    "prompt": "Differentiate x^2.",
                    "response": "maybe 2",
                    "score": 0.2,
                    "reason": "verified against math reference",
                }
            ]
        },
        hard_examples=[],
    )

    assert report["training_enabled"] is False
    assert report["auto_apply_enabled"] is False
    assert report["traces"][0]["failure"] == "verifier_grounding_weak"
    assert report["candidates"]
    assert report["candidates"][0]["owner_approval_required"] is True
    assert report["scores"][0]["decision"] in {"owner_review", "collect_more_evidence"}
    assert report["scores"][0]["verifier"]["name"] == "gepa_candidate"
    assert report["scores"][0]["verifier"]["score"] == 1.0


def test_gepa_report_writer_uses_registered_path(tmp_path) -> None:
    output_path = tmp_path / "gepa.json"
    report = write_gepa_report(build_gepa_report(hard_examples=[]), output_path=output_path)

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved == report
    assert report["report_path"] == str(output_path)


def test_gepa_report_path_is_registered() -> None:
    assert v2_report_path("gepa_report").name == "v2_gepa_report.json"
