"""Cognition must run on the real full-system chat path, not only side routes.

The cognition registry (CRE causal classification, epistemic tracker) was
previously exercised only by dedicated /cognition and /goal endpoints while
the full-system contract claimed it as part of chat. These tests cover the
chat-path helpers: classification of the live message and recording of the
real generation outcome into the epistemic calibration history.
"""

from __future__ import annotations

import json
from pathlib import Path

import app
from cognition.services import CognitionServices


def test_classify_chat_cognition_returns_causal_judgment() -> None:
    result = app._classify_chat_cognition(
        "If we deploy the new cache, will response latency drop?"
    )
    assert result is not None
    assert result["release"] == "cognition-v1"
    causal = result["causal"]
    assert causal["causal_type"] in {"counterfactual", "interventional", "observational"}
    assert 0.0 <= float(causal["confidence"]) <= 1.0


def test_classify_chat_cognition_respects_feature_flag(monkeypatch) -> None:
    monkeypatch.setattr(app, "is_enabled", lambda _name: False)
    assert app._classify_chat_cognition("any message") is None


def test_classify_chat_cognition_degrades_on_error(monkeypatch) -> None:
    class Broken:
        def classify_goal(self, _message: str):
            raise RuntimeError("boom")

    monkeypatch.setattr(app, "COGNITION", Broken())
    assert app._classify_chat_cognition("any message") is None


def test_record_generation_epistemics_appends_real_history(
    tmp_path: Path, monkeypatch
) -> None:
    services = CognitionServices(
        state_dir=tmp_path / "state", output_dir=tmp_path / "output"
    )
    monkeypatch.setattr(app, "COGNITION", services)

    assert app._record_generation_epistemics("session-x", "req-1", accepted=True) is True
    assert app._record_generation_epistemics("session-x", "req-2", accepted=False) is True

    history = (tmp_path / "output" / "epistemic_history.jsonl").read_text(encoding="utf-8")
    records = [json.loads(line) for line in history.splitlines() if line.strip()]
    assert len(records) == 2
    assert records[0]["was_correct"] is True
    assert records[1]["was_correct"] is False
    assert all(record["verifier"] == "generation_quality" for record in records)
    assert all(record["domain"] == "conversation" for record in records)


def test_record_generation_epistemics_degrades_on_error(monkeypatch) -> None:
    class BrokenTracker:
        @staticmethod
        def record_outcome(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("boom")

    class BrokenServices:
        et = BrokenTracker()

    monkeypatch.setattr(app, "COGNITION", BrokenServices())
    assert app._record_generation_epistemics("sess", "req", accepted=True) is False
