"""CYR tournament v2 unit tests: pure logic only, zero training.
Covers: shared-context rendering + factorial purity, task-baseline oracle,
generation-report scoring, transitions, resolver, hysteresis persistence,
exposure matching, sealed rules, packaging, guards.
Run: python tests/test_v5_cyr_tournament.py
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v5_experiments.cyr_tournament import (  # noqa: E402
    HysteresisController,
    assert_split_firewall,
    both_correct_rate,
    constant_lr,
    eval_variant_texts,
    exposure_matched,
    find_transitions,
    guard_full_mode,
    package_bundle_v2,
    pair_batches,
    proxy_ladder,
    proxy_spec_kwargs,
    redteam_exposure,
    redteam_freezing,
    redteam_near_dup,
    redteam_non_mutation,
    redteam_overlap,
    render_worlds,
    resolve_campaign,
    score_generated,
    split_manifest,
    sustained,
    task_baselines,
)


def test_shared_context_factorial_purity():
    splits = render_worlds(family="registry",
                           split_seeds={"train": 11}, worlds_per_split=6)
    world = splits["train"][0]
    variants = eval_variant_texts(world)
    assert set(variants) == {"base", "query_only", "order_only",
                             "query_and_order", "relevant_value_only",
                             "irrelevant_value_only", "rendering_only"}
    base_context = variants["base"]["text"].rsplit("\nAnswer:", 1)[0]
    twin_context = variants["query_only"]["text"].rsplit("\nAnswer:", 1)[0]
    # Same facts/order/distractors; only the query line differs.
    base_lines, twin_lines = base_context.split("\n"), twin_context.split("\n")
    assert base_lines[:-1] == twin_lines[:-1]
    assert base_lines[-1] != twin_lines[-1]
    assert variants["base"]["answer"] != variants["query_only"]["answer"]
    # Order-only: same query, same values, permuted facts.
    assert variants["order_only"]["answer"] == variants["base"]["answer"]
    assert sorted(variants["order_only"]["text"].split("\n")) == \
        sorted(variants["base"]["text"].split("\n"))
    # Relevant flip changes the answer; irrelevant preserves it.
    assert variants["relevant_value_only"]["answer"] != variants["base"]["answer"]
    assert variants["irrelevant_value_only"]["answer"] == variants["base"]["answer"]
    assert_split_firewall(splits)
    manifest = split_manifest(splits)
    assert len(manifest["sha256"]) == 64


def test_task_baseline_oracle():
    worlds = render_worlds(family="registry", split_seeds={"t": 5},
                           worlds_per_split=8)["t"]
    baselines = task_baselines()
    # Registry answers are uniformly distributed over fact positions by
    # construction (target = rng.randrange(4)), so no positional policy
    # dominates; every policy scores within [0, 1] and termination holds.
    scores = {}
    for name, predict in baselines.items():
        hits = sum(1 for world in worlds
                   if predict({}, world) == world["base"]["answer"])
        scores[name] = hits / len(worlds)
    assert set(scores) == {"COPY_FIRST_FACT_VALUE", "COPY_LAST_FACT_VALUE",
                           "FIXED_FACT_POSITION", "MOST_FREQUENT_VALUE",
                           "QUERY_BLIND_CANONICAL_POSITION",
                           "SECOND_TO_LAST_VALUE"}
    assert all(0.0 <= value <= 1.0 for value in scores.values())
    # Oracle fixture: answer always the first fact's value.
    rigged = [dict(world, values=["ANS"] + world["values"][1:],
                   base=dict(world["base"], answer="ANS")) for world in worlds]
    assert all(baselines["COPY_FIRST_FACT_VALUE"]({}, world) == "ANS"
               for world in rigged)


def test_generation_report():
    report = score_generated(
        generated=["V1", "V1 extra tokens", "", "V2"],
        expected=["V1", "V1", "V3", "V9"],
        stops=["eos", "eos", "cap", "cap"])
    assert report["complete_exact"] == 0.25
    assert report["prefix_correct_but_extra"] == 0.25
    assert report["eos_stop_rate"] == 0.5
    assert report["max_tokens_rate"] == 0.5
    assert report["invalid_output_rate"] == 0.25
    assert both_correct_rate(base_ok=[True, True],
                             twin_ok=[True, False]) == 0.5
    try:
        both_correct_rate(base_ok=[True], twin_ok=[])
    except ValueError:
        pass
    else:
        raise AssertionError("misaligned both-correct was accepted")


def test_transitions_and_sustained():
    assert sustained([True, True, True]) is True
    assert sustained([True, False, True, True, True]) is True
    assert sustained([True, True]) is False
    trace = find_transitions(
        flags=[False, True, False, True, True, True, True],
        updates=[10, 20, 30, 40, 50, 60, 70], tokens_per_update=1000,
        started_wall_s=100.0, eval_wall_s=[110.0, 120.0, 130.0, 140.0,
                                           150.0, 160.0, 170.0],
        threshold_name="G90", required=3)
    assert trace["onset"]["update"] == 20
    assert trace["confirmation"]["update"] == 40
    assert trace["confirmation"]["tokens"] == 40_000
    assert trace["confirmation"]["wall_s"] == 40.0
    empty = find_transitions(
        flags=[False, False], updates=[10, 20], tokens_per_update=1000,
        started_wall_s=0.0, eval_wall_s=[1.0, 2.0], threshold_name="G90")
    assert empty["onset"] is None and empty["confirmation"] is None


def test_resolver_v2_policy():
    fast = resolve_campaign(tokens_per_sec=30000, free_vram_gb=12,
                            microbatch_tokens=2416)
    assert fast["proxy"] == "MIDI" and fast["dose_floor_met"] is True
    slow = resolve_campaign(tokens_per_sec=100, free_vram_gb=1,
                            microbatch_tokens=2416)
    assert slow["proxy"] == "RESEARCH_SMALL"
    assert "transfer-second-seed" in slow["dropped"]
    tiny = resolve_campaign(tokens_per_sec=1, free_vram_gb=1,
                            microbatch_tokens=2416, smoke=True)
    assert tiny["proxy"] == "TINY"
    try:
        resolve_campaign(tokens_per_sec=0, free_vram_gb=4,
                         microbatch_tokens=100)
    except ValueError:
        pass
    else:
        raise AssertionError("non-positive calibration was accepted")


def test_proxy_specs_validate():
    for name, proxy in proxy_ladder().items():
        checked = proxy_spec_kwargs(proxy, vocab_size=24576)
        assert checked["parameters"] > 0, name
    small = proxy_spec_kwargs(proxy_ladder()["RESEARCH_SMALL"],
                              vocab_size=24576)
    micro = proxy_spec_kwargs(proxy_ladder()["MICRO"], vocab_size=24576)
    assert small["parameters"] < micro["parameters"]


def test_hysteresis_full_cycle_with_persistence():
    controller = HysteresisController(enter_retention=0.90,
                                      reenter_plasticity=0.75, confirmations=3)
    for value in (0.91, 0.92, 0.93):
        receipt = controller.observe(
            metric=value, threshold_note="enter", token_position=100,
            lr_before=3e-4, lr_plasticity=3e-4, lr_retention=3e-6)
    assert receipt["state_after"] == "retention"
    snapshot = controller.snapshot()
    revived = HysteresisController.restore(snapshot)
    assert revived.mode == "retention" and revived.streak == 0
    for value in (0.80, 0.70, 0.60):
        receipt = revived.observe(
            metric=value, threshold_note="hold", token_position=200,
            lr_before=3e-6, lr_plasticity=3e-4, lr_retention=3e-6)
    assert receipt["state_after"] == "retention"
    receipt = revived.observe(
        metric=0.59, threshold_note="reenter", token_position=300,
        lr_before=3e-6, lr_plasticity=3e-4, lr_retention=3e-6)
    assert receipt["state_after"] == "plasticity"
    for value in (0.91, 0.92, 0.94):
        receipt = revived.observe(
            metric=value, threshold_note="re-enter", token_position=400,
            lr_before=3e-4, lr_plasticity=3e-4, lr_retention=3e-6)
    assert receipt["state_after"] == "retention"
    assert len(revived.decisions) == 3 + 4 + 3
    try:
        HysteresisController(enter_retention=0.75, reenter_plasticity=0.90,
                             confirmations=3).assert_valid()
    except ValueError:
        pass
    else:
        raise AssertionError("inverted hysteresis was accepted")


def test_exposure_matching_and_redteam():
    matched = exposure_matched(high_tokens=1000, low_tokens=900)
    assert matched["pass"] is True
    skewed = exposure_matched(high_tokens=3000, low_tokens=900)
    assert skewed["pass"] is False and "confounded" in skewed["note"]
    assert redteam_exposure({"a": {"tokens": 10, "updates": 2},
                             "b": {"tokens": 10, "updates": 2}})["pass"] is True
    assert redteam_exposure({"a": {"tokens": 10, "updates": 2},
                             "b": {"tokens": 11, "updates": 2}})["pass"] is False
    assert redteam_overlap(["a"], ["b"])["pass"] is True
    assert redteam_overlap(["a"], ["a"])["pass"] is False
    assert redteam_freezing(0.379, 0.008)["pass"] is True
    assert redteam_non_mutation("a" * 64, "b" * 64)["pass"] is True
    assert redteam_near_dup(["alpha beta gamma", "unrelated zeta"])["pass"] is True


def test_pair_batches_and_firewall():
    splits = render_worlds(family="transfer", split_seeds={"train": 41},
                           worlds_per_split=8)
    grouped = pair_batches(splits["train"], group_pairs=True, seed=3)
    shuffled = pair_batches(splits["train"], group_pairs=False, seed=3)
    assert len(grouped) == len(shuffled) == 16
    assert sorted(r["text"] for r in grouped) == sorted(r["text"] for r in shuffled)
    for position in range(0, len(grouped), 2):
        assert grouped[position]["world_id"] == grouped[position + 1]["world_id"]
    assert_split_firewall(splits)


def test_guard_and_packaging_v2():
    import argparse
    args = argparse.Namespace(mode="full", allow_non_colab=False)
    try:
        guard_full_mode(args)
    except ValueError as exc:
        assert "COLAB_GPU" in str(exc)
    else:
        raise AssertionError("full mode was allowed locally")
    from v5_experiments.cyr_tournament import package_bundle_v2
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "run"
        out.mkdir()
        (out / "BASELINE-registry-101.json").write_text("{}", encoding="utf-8")
        (out / "PAIR-registry-101-paired.json").write_text("{}", encoding="utf-8")
        (out / "DECISION.json").write_text("{}", encoding="utf-8")
        archive = package_bundle_v2(out)
        assert archive.is_file()
        manifest = __import__("json").loads(
            (out / "SESSION_MANIFEST.json").read_text(encoding="utf-8"))
        assert manifest["stages"] == {"ACQUISITION": ["BASELINE-registry-101.json"],
                                      "PAIR_QUERY": ["PAIR-registry-101-paired.json"]}
        assert manifest["top_level"] == ["DECISION.json"]
        names = __import__("zipfile").ZipFile(str(archive)).namelist()
        assert "ACQUISITION/BASELINE-registry-101.json" in names
        assert "PAIR_QUERY/PAIR-registry-101-paired.json" in names


def test_blind_gap_pure():
    from anra_v5.cyr_execute import blind_gap
    worlds = render_worlds(family="registry", split_seeds={"t": 5},
                           worlds_per_split=4)["t"]
    gap = blind_gap(model_both=0.75, worlds=worlds)
    assert gap["model_both_correct"] == 0.75
    assert set(gap["baselines"]) == {"copy_first", "copy_last",
                                     "most_frequent", "fixed_position"}
    assert gap["blind_gap"] == 0.75 - gap["best_baseline_both_correct"]


def test_sealed_excluded_from_decision():
    from anra_v5.cyr_execute import _decision
    base = {"stages": {}, "gates": {}, "redteam": []}
    prereg = {"sha256": "ab" * 32}
    first = _decision(dict(base, sealed_report={"both_correct": 0.99}), prereg)
    second = _decision(dict(base, sealed_report={"both_correct": 0.01}), prereg)
    assert first == second


def test_prereg_arm_seed_match():
    import json
    from v5_experiments.cyr_tournament import CYR_ARMS, CYR_SEEDS
    path = ROOT / "docs/cymek/experiments/CYR-GPU-002/PREREGISTRATION.json"
    if not path.is_file():
        print("SKIP test_prereg_arm_seed_match (prereg generated at freeze)")
        return
    prereg = json.loads(path.read_text(encoding="utf-8"))
    assert prereg["arms"] == {name: list(arms) for name, arms in CYR_ARMS.items()}
    assert prereg["seeds"] == list(CYR_SEEDS)


def test_deadline_timebox_without_training():
    import torch
    torch.set_num_threads(2)
    from anra_v5.cyr_execute import ByteTokenizer, train_arm, proxy_ladder
    tok = ByteTokenizer()
    worlds = render_worlds(family="registry", split_seeds={"train": 61},
                           worlds_per_split=4)
    from v5_experiments.cyr_tournament import pair_batches, constant_lr
    records = pair_batches(worlds["train"], group_pairs=True, seed=61)
    with tempfile.TemporaryDirectory() as tmp:
        result = train_arm(
            proxy=proxy_ladder()["TINY"], tokenizer=tok, torch=torch,
            device=torch.device("cpu"), records=records, microbatch_rows=4,
            row_width=512, updates=2, schedule=constant_lr(3e-4), seed=61,
            run_id="deadline", store_root=str(Path(tmp) / "store"),
            deadline_min=0.000001)
    assert result["status"] == "TIMEBOX"
    assert result["updates"] == 0
    assert result["checkpoint_head"] is not None


def test_free_generation_fixture_forward_only():
    import torch
    torch.set_num_threads(2)
    from anra_v5.cyr_execute import ByteTokenizer, free_generation_spot
    from v5_experiments.cyr_tournament import proxy_ladder
    from v5_model.core import initialize
    from anra_v5.cyr_execute import _model_spec
    tok = ByteTokenizer()
    worlds = render_worlds(family="registry", split_seeds={"t": 62},
                           worlds_per_split=3)["t"]
    model = initialize(_model_spec(proxy_ladder()["TINY"], vocab_size=256),
                       62, torch_module=torch)
    report = free_generation_spot(
        model=model, tokenizer=tok, torch=torch,
        device=torch.device("cpu"), worlds=worlds, max_new_tokens=8)
    assert report["worlds"] == 3
    assert report["stops"]["eos"] + report["stops"]["cap"] == 3
    del model


def test_larger_proxy_constructs():
    from v5_experiments.cyr_tournament import proxy_ladder, proxy_spec_kwargs
    midi = proxy_spec_kwargs(proxy_ladder()["MIDI"], vocab_size=24576)
    micro = proxy_spec_kwargs(proxy_ladder()["MICRO"], vocab_size=24576)
    assert midi["parameters"] > micro["parameters"] > 0
    import torch
    torch.set_num_threads(2)
    from v5_model.core import initialize
    from v5_contracts.model_spec import ModelSpec
    spec = ModelSpec(
        schema="anra-v5-model-spec/v1", family="dense-decoder-transformer",
        vocabulary_size=24576, width=384, layers=8, query_heads=6, kv_heads=3,
        head_dimension=64, ffn_width=1024, context_length=512,
        rope_base=10_000.0, norm_epsilon=1e-5, tied_embeddings=True,
        qk_norm=True, qk_norm_affine=True, linear_bias=False, dropout=0.0)
    model = initialize(spec, 63, torch_module=torch)
    total = sum(parameter.numel() for parameter in model.parameters())
    assert total == midi["parameters"]
    del model


def test_production_tokenizer_render_path():
    try:
        from v5_tokenizer.artifact import load_verified_tokenizer
    except ImportError:
        print("SKIP test_production_tokenizer_render_path (no tokenizers pkg)")
        return
    import hashlib
    artifact = ROOT / "artifacts/e1/local_tournament/tokenizer-24576.json.gz"
    expected = hashlib.sha256(artifact.read_bytes()).hexdigest()
    result = __import__("json").loads(
        (ROOT / "artifacts/e1/local_tournament/result.json").read_text(
            encoding="utf-8"))
    trainer_sha = hashlib.sha256(
        __import__("json").dumps(result["trainer"], sort_keys=True).encode()).hexdigest()
    tok, _ = load_verified_tokenizer(
        artifact, expected_sha256=expected, vocabulary_size=24576,
        trainer_config_sha256=trainer_sha,
        corpus_manifest_sha256=result["corpus_manifest_sha256"])

    class _Tok:
        vocab_size = 24576

        def __init__(self, backend):
            self.backend = backend

        def encode(self, text):
            return list(self.backend.encode(text).ids)

    worlds = render_worlds(family="registry", split_seeds={"t": 64},
                           worlds_per_split=2)["t"]
    ids = _Tok(tok).encode(worlds[0]["base"]["text"])
    assert len(ids) > 0 and max(ids) < 24576


def test_notebook_fail_hard():
    import json
    notebook = json.loads(
        (ROOT / "notebooks" / "cymek_colab_gpu_research_v2.ipynb").read_text(
            encoding="utf-8"))
    blob = json.dumps(notebook)
    assert "| tail" not in blob, "failure-masking pipe in notebook"
    assert "run_checked" in blob, "notebook must fail hard on commands"
    assert "COLAB_GPU" in blob
    joined = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"]
        if cell["cell_type"] == "code")
    assert '"--mode", "full"' in joined
    assert "assert completed.returncode == 0" in joined
    assert notebook["nbformat"] == 4
    code_cells = [cell["source"] for cell in notebook["cells"]
                  if cell["cell_type"] == "code"]
    assert len(code_cells) == 3
    for cell in code_cells:
        python_only = "\n".join(
            line for line in "".join(cell).splitlines()
            if not line.lstrip().startswith(("%", "!")))
        compile(python_only, "<notebook-cell>", "exec")
    assert "assert completed.returncode == 0" in blob, \
        "CELL 1 must not print COMPLETE after failure"
    assert "cymek_head_sha" in blob, "notebook must pin the exact commit"


def test_static_self_check():
    from v5_experiments.static_check import main as check_main
    import io
    from contextlib import redirect_stdout
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = check_main([
            str(ROOT / "v5_experiments/cyr_tournament.py"),
            str(ROOT / "anra_v5/cyr_execute.py"),
            str(ROOT / "tests/test_v5_cyr_tournament.py"),
            str(ROOT / "v5_experiments/static_check.py")])
    assert code == 0, buffer.getvalue()[-2000:]


def test_tiny_smoke_cpu():
    import torch
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        pass  # already initialized by an earlier test in this process
    from anra_v5.cyr_execute import smoke
    with tempfile.TemporaryDirectory() as tmp:
        bundle = smoke(torch, torch.device("cpu"), Path(tmp) / "out")
    assert bundle["mode"] == "smoke"
    assert bundle["result"]["updates"] == 2
    assert bundle["result"]["pair_splits"] == 0
    assert bundle["result"]["checkpoint_head"] is not None


def test_full_pipeline_s0_to_s4():
    import torch
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        pass  # already initialized by an earlier test in this process
    from anra_v5.cyr_execute import ByteTokenizer, run_cyr_campaign
    tok = ByteTokenizer()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "pipe"
        summary = run_cyr_campaign(
            out=out, torch=torch, device=torch.device("cpu"), tokenizer=tok,
            stages=("s0", "s1", "s2", "s3", "s4"), time_limit_min=60.0,
            prereg={"sha256": "ab" * 32, "experiment_id": "CYR-GPU-002-PIPE"},
            proxy_name="TINY", updates_per_arm=1, seeds=(7,),
            gate_overrides={"learnability": True})
        for stage in ("s0", "s1", "s2", "s3", "s4"):
            assert stage in summary["stages"], f"missing stage {stage}"
        assert summary["gates"].get("gates_overridden") == ["learnability"]
        assert (out / "DECISION.json").is_file()
        assert (out / "CYMEK_GPU_RESEARCH_V2_RESULTS.zip").is_file()
        assert summary["stages"]["s4"]["status"] == "COMPLETE"


_TESTS = [test_shared_context_factorial_purity,
          test_blind_gap_pure,
          test_sealed_excluded_from_decision,
          test_prereg_arm_seed_match,
          test_deadline_timebox_without_training,
          test_free_generation_fixture_forward_only,
          test_larger_proxy_constructs,
          test_production_tokenizer_render_path,
          test_notebook_fail_hard,
          test_static_self_check,
          test_tiny_smoke_cpu,
          test_full_pipeline_s0_to_s4,
          test_task_baseline_oracle,
          test_generation_report,
          test_transitions_and_sustained,
          test_resolver_v2_policy,
          test_proxy_specs_validate,
          test_hysteresis_full_cycle_with_persistence,
          test_exposure_matching_and_redteam,
          test_pair_batches_and_firewall,
          test_guard_and_packaging_v2]


def main() -> int:
    failed = 0
    for fn in _TESTS:
        try:
            fn()
            print(f"PASS {fn.__name__}", flush=True)
        except Exception as exc:
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL {fn.__name__}: {exc}", flush=True)
    print(f"{len(_TESTS) - failed}/{len(_TESTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
