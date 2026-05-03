from training.replay_pipeline import ReplayPipeline


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
