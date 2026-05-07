from engine import feature_flags
from engine.eval_harness import EvalHarness, EvalResult


def _tasks():
    return [{"id": 1}, {"id": 2}]


def test_run_baseline_disables_component(tmp_path, monkeypatch):
    monkeypatch.setattr(feature_flags, "FLAGS_FILE", tmp_path / "feature_flags.json")
    harness = EvalHarness(output_dir=tmp_path / "eval")
    seen = []

    def runner(task):
        seen.append(feature_flags.is_enabled("ouroboros"))
        return {"success": True}

    result = harness.run_baseline("ouroboros", _tasks(), runner)

    assert result.mode == "baseline"
    assert seen == [False, False]
    assert feature_flags.is_enabled("ouroboros") is True


def test_run_system_on_enables_component(tmp_path, monkeypatch):
    monkeypatch.setattr(feature_flags, "FLAGS_FILE", tmp_path / "feature_flags.json")
    feature_flags.set_flag("ouroboros", False)
    harness = EvalHarness(output_dir=tmp_path / "eval")
    seen = []

    def runner(task):
        seen.append(feature_flags.is_enabled("ouroboros"))
        return {"success": True}

    result = harness.run_system_on("ouroboros", _tasks(), runner)

    assert result.mode == "system_on"
    assert seen == [True, True]
    assert feature_flags.is_enabled("ouroboros") is False


def test_ablation_isolates_one_component(tmp_path, monkeypatch):
    monkeypatch.setattr(feature_flags, "FLAGS_FILE", tmp_path / "feature_flags.json")
    harness = EvalHarness(output_dir=tmp_path / "eval")

    def runner(task):
        return {"success": not feature_flags.is_enabled("ghost_memory")}

    result = harness.run_ablation("ghost_memory", _tasks(), runner)

    assert result.mode == "ablation"
    assert result.task_success_rate == 1.0
    assert feature_flags.is_enabled("ghost_memory") is True


def test_compare_detects_regression():
    harness = EvalHarness()
    baseline = EvalResult("memory", "baseline", 1.0, 10.0, 0.0)
    current = EvalResult("memory", "system_on", 0.8, 12.0, 0.0)

    report = harness.compare(baseline, current)

    assert report.regressed is True
    assert report.verdict == "regressed"


def test_compare_detects_improvement():
    harness = EvalHarness()
    baseline = EvalResult("memory", "baseline", 0.5, 10.0, 0.0)
    current = EvalResult("memory", "system_on", 0.8, 12.0, 0.0)

    report = harness.compare(baseline, current)

    assert report.regressed is False
    assert report.verdict == "improved"


def test_save_and_load_report(tmp_path):
    harness = EvalHarness(output_dir=tmp_path)
    baseline = EvalResult("memory", "baseline", 1.0, 10.0, 0.0)
    current = EvalResult("memory", "system_on", 1.0, 11.0, 0.0)
    report = harness.compare(baseline, current)

    path = harness.save_report(report)
    loaded = harness.load_last_report("memory")

    assert path.exists()
    assert loaded["component"] == "memory"
    assert loaded["verdict"] == "neutral"
