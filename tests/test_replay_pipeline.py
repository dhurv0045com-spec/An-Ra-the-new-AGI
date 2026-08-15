from training.replay_pipeline import ReplayPipeline
from identity.falsification_ledger import FalsificationLedger


def test_replay_pipeline_add_sample_and_roundtrip(tmp_path):
    path = tmp_path / "replay.jsonl"
    pipe = ReplayPipeline(max_size=4, path=path)
    pipe.add("prompt", "target", source="unit", score=0.8, weight=0.5)

    assert len(pipe) == 1
    assert pipe.sample(1, seed=0).texts() == ["prompt\ntarget"]

    pipe.save()
    loaded = ReplayPipeline.load(path)
    assert len(loaded) == 1
    assert loaded.records[0].source == "unit"


def test_falsification_ledger_imports_only_rejected_claims(tmp_path):
    ledger = FalsificationLedger(tmp_path / "ledger.json")
    ledger.append("false claim", status="FALSIFIED", would_be_false_if="test failed")
    ledger.append("good claim", status="VERIFIED")

    pipe = ReplayPipeline()
    assert pipe.add_falsification_ledger(ledger) == 1
    assert pipe.records[0].prompt == "false claim"
    assert pipe.records[0].source == "falsification_ledger"
    assert "Do not repeat" in pipe.records[0].target
