from __future__ import annotations

import json

from training.data_ingestion import prepare_training_corpus


def test_prepare_training_corpus_merges_text_code_and_teacher(tmp_path):
    base = tmp_path / "base.txt"
    base.write_text("H: Who are you?\nANRA: I am An-Ra.\n", encoding="utf-8")
    code = tmp_path / "module.py"
    code.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    teacher = tmp_path / "teacher_reasoning.jsonl"
    teacher.write_text(
        json.dumps(
            {
                "prompt": "Solve 2 + 2.",
                "answer": "2 + 2 = 4.",
                "task_type": "math",
                "verified": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output = tmp_path / "merged.txt"
    teacher_output = tmp_path / "teacher_out.jsonl"
    report_path = tmp_path / "report.json"

    report = prepare_training_corpus(
        explicit_sources=[base, code, teacher],
        include_drive=False,
        output_path=output,
        teacher_output=teacher_output,
        report_path=report_path,
        mount_drive=False,
        mirror_merged=False,
        mirror_teacher=False,
    )

    merged = output.read_text(encoding="utf-8")
    teacher_lines = teacher_output.read_text(encoding="utf-8").splitlines()
    assert "H: Who are you?" in merged
    assert "def add(a, b):" in merged
    assert report.total_examples >= 3
    assert report.teacher_records == 1
    assert json.loads(teacher_lines[0])["verified"] is True
    assert report_path.exists()
