from engine import feature_flags
from engine.eval_harness import EvalHarness, EvalResult
from evaluation.elo_harness import EloHarness, calculate_expected_score


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
    assert feature_flags.is_enabled("ouroboros") is False


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
        return {"success": not feature_flags.is_enabled("ouroboros")}

    result = harness.run_ablation("ouroboros", _tasks(), runner)

    assert result.mode == "ablation"
    assert result.task_success_rate == 1.0
    assert feature_flags.is_enabled("ouroboros") is False


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


def test_elo_harness_blinded_match():
    harness = EloHarness(initial_rating=1200)
    
    # Mock generators
    def gen_a(prompt): return "response_a"
    def gen_b(prompt): return "response_b"
    
    # Judge always prefers "response_a" (A is better)
    def judge(prompt, r1, r2):
        if r1 == "response_a":
            return 1.0
        return 0.0

    score = harness.run_paired_comparison("ckpt_a", gen_a, "ckpt_b", gen_b, "prompt", judge)
    
    assert score == 1.0
    
    rating_a = harness.get_rating("ckpt_a")
    rating_b = harness.get_rating("ckpt_b")
    
    assert rating_a.rating > 1200.0
    assert rating_b.rating < 1200.0
    assert rating_a.matches == 1
    assert rating_b.matches == 1


def test_elo_regression():
    harness = EloHarness(initial_rating=1200)
    
    # A beats B twice
    def gen_a(prompt): return "response_a"
    def gen_b(prompt): return "response_b"
    def judge(prompt, r1, r2): return 1.0 if r1 == "response_a" else 0.0
    
    harness.run_paired_comparison("baseline", gen_a, "candidate", gen_b, "prompt1", judge)
    harness.run_paired_comparison("baseline", gen_a, "candidate", gen_b, "prompt2", judge)
    
    # Candidate should be considered regressed compared to baseline
    assert harness.check_regression("candidate", "baseline") is True
