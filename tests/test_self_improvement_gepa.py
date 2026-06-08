from __future__ import annotations

import json
from pathlib import Path


def test_self_improvement_writes_gepa_report(tmp_path, monkeypatch) -> None:
    import scripts.run_self_improvement as runner

    paths = {
        "eval_summary": tmp_path / "eval.json",
        "hard_examples": tmp_path / "hard.json",
        "mix_report": tmp_path / "mix.json",
        "rlvr_report": tmp_path / "rlvr.json",
        "gepa_report": tmp_path / "gepa.json",
        "improvement_report": tmp_path / "improvement.json",
    }

    paths["eval_summary"].write_text(
        json.dumps(
            {
                "category_scores": {"symbolic": 0.2},
                "results": [
                    {
                        "id": "symbolic_math",
                        "category": "symbolic",
                        "prompt": "Differentiate x^2.",
                        "response": "I guess x.",
                        "score": 0.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    paths["hard_examples"].write_text(json.dumps({"examples": []}), encoding="utf-8")
    paths["mix_report"].write_text(json.dumps({"teacher_external_used": 1}), encoding="utf-8")
    paths["rlvr_report"].write_text(json.dumps({"verifier_pass_rate": 0.4, "task_id": "r1"}), encoding="utf-8")

    monkeypatch.setattr(runner, "v2_report_path", lambda key: paths[key])

    report = runner.run_self_improvement()

    assert paths["gepa_report"].exists()
    assert paths["improvement_report"].exists()
    assert report["gepa"]["trace_count"] >= 2
    assert report["gepa"]["candidate_count"] >= 1
    assert report["gepa"]["auto_apply_enabled"] is False
