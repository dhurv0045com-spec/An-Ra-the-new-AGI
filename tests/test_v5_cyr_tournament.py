"""CYR tournament runner tests: pure logic fast, one tiny CPU smoke.
Run: python tests/test_v5_cyr_tournament.py
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anra_v5.cyr_execute import ByteTokenizer, smoke  # noqa: E402
from v5_experiments.cyr_tournament import (  # noqa: E402
    HysteresisController,
    assert_split_firewall,
    constant_lr,
    guard_full_mode,
    heuristic_baselines,
    package_bundle,
    pair_batches,
    proxy_ladder,
    redteam_exposure,
    redteam_freezing,
    redteam_near_dup,
    redteam_non_mutation,
    redteam_overlap,
    render_worlds,
    resolve_proxy,
    split_manifest,
    sustained,
)


def test_proxy_ladder_and_resolver():
    ladder = proxy_ladder()
    assert set(ladder) == {"TINY", "MICRO", "MIDI", "P35"}
    assert resolve_proxy(micro_tokens_per_sec=1, free_vram_gb=1,
                         smoke=True)["proxy"] == "TINY"
    assert resolve_proxy(micro_tokens_per_sec=30000,
                         free_vram_gb=12)["proxy"] == "MIDI"
    assert resolve_proxy(micro_tokens_per_sec=8000,
                         free_vram_gb=6)["proxy"] == "MICRO"
    slow = resolve_proxy(micro_tokens_per_sec=100, free_vram_gb=2)
    assert slow["proxy"] == "MICRO" and slow["tokens_per_arm"] == 60_000
    try:
        resolve_proxy(micro_tokens_per_sec=0, free_vram_gb=4)
    except ValueError:
        pass
    else:
        raise AssertionError("non-positive calibration was accepted")
    try:
        constant_lr(0)
    except ValueError:
        pass
    else:
        raise AssertionError("non-positive research LR was accepted")


def test_worlds_deterministic_and_firewalled():
    first = render_worlds(family="registry",
                          split_seeds={"train": 11, "dev": 22}, worlds_per_split=6)
    second = render_worlds(family="registry",
                           split_seeds={"train": 11, "dev": 22}, worlds_per_split=6)
    assert first == second
    assert_split_firewall(first)
    manifest = split_manifest(first)
    assert len(manifest["sha256"]) == 64
    assert len(manifest["splits"]["train"]) == 6
    try:
        render_worlds(family="registry", split_seeds={"a": 5, "b": 5},
                      worlds_per_split=2)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate split seeds were accepted")
    twin_worlds = [world for worlds in first.values() for world in worlds]
    for world in twin_worlds:
        assert world["base"]["answer"] != world["twin"]["answer"]
        assert world["base"]["text"] != world["twin"]["text"]
        assert world["base"]["text"].split("\n")[0] == \
            world["twin"]["text"].split("\n")[0]


def test_worlds_exact_sizing():
    tok = ByteTokenizer()
    splits = render_worlds(family="transfer",
                           split_seeds={"train": 31}, worlds_per_split=4,
                           encode=tok.encode, row_content_tokens=400)
    for world in splits["train"]:
        for side in ("base", "twin"):
            record = world[side]
            assert len(tok.encode(record["text"])) == 400
            assert record["text"].endswith(record["answer"])


def test_pair_batches_grouped_and_matched():
    splits = render_worlds(family="registry", split_seeds={"train": 41},
                           worlds_per_split=8)
    grouped = pair_batches(splits["train"], group_pairs=True, seed=3)
    shuffled = pair_batches(splits["train"], group_pairs=False, seed=3)
    assert len(grouped) == len(shuffled) == 16
    assert sorted(r["text"] for r in grouped) == sorted(r["text"] for r in shuffled)
    for position in range(0, len(grouped), 2):
        assert grouped[position]["world_id"] == grouped[position + 1]["world_id"]
    again = pair_batches(splits["train"], group_pairs=True, seed=3)
    assert again == grouped


def test_sustained_and_controller():
    assert sustained([True, True, True]) is True
    assert sustained([True, False, True, True, True]) is True
    assert sustained([True, True]) is False
    controller = HysteresisController(enter_retention=0.90,
                                      reenter_plasticity=0.75, confirmations=3)
    first_lr, second_lr = 3e-4, 3e-6
    for value in (0.91, 0.92, 0.93):
        receipt = controller.observe(
            metric=value, threshold_note="enter>=0.90x3", token_position=100,
            lr_before=first_lr, lr_plasticity=first_lr, lr_retention=second_lr)
    assert receipt["state_after"] == "retention"
    assert receipt["lr_after"] == second_lr
    assert receipt["controller_split"] == "dev-controller only"
    for value in (0.80, 0.70, 0.69):
        receipt = controller.observe(
            metric=value, threshold_note="reenter<0.75x3", token_position=200,
            lr_before=second_lr, lr_plasticity=first_lr, lr_retention=second_lr)
    assert receipt["state_after"] == "retention"
    receipt = controller.observe(
        metric=0.68, threshold_note="reenter<0.75x3", token_position=300,
        lr_before=second_lr, lr_plasticity=first_lr, lr_retention=second_lr)
    assert receipt["state_after"] == "plasticity"
    assert receipt["lr_after"] == first_lr
    try:
        HysteresisController(enter_retention=0.75, reenter_plasticity=0.90,
                             confirmations=3).assert_valid()
    except ValueError:
        pass
    else:
        raise AssertionError("inverted hysteresis was accepted")


def test_baselines_and_redteam():
    baselines = heuristic_baselines()
    context = "alpha carries one\nbeta carries two"
    assert baselines["copy_first"](context, "q") == "one"
    assert baselines["copy_last"](context, "q") == "two"
    assert redteam_overlap(["a", "b"], ["c"])["pass"] is True
    assert redteam_overlap(["a", "b"], ["b"])["pass"] is False
    assert redteam_exposure({"a": {"tokens": 10, "updates": 2},
                             "b": {"tokens": 10, "updates": 2}})["pass"] is True
    assert redteam_exposure({"a": {"tokens": 10, "updates": 2},
                             "b": {"tokens": 11, "updates": 2}})["pass"] is False
    freezing = redteam_freezing(0.379, 0.008)
    assert freezing["pass"] is True and freezing["displacement_ratio"] > 5
    assert redteam_non_mutation("a" * 64, "a" * 64)["pass"] is False
    assert redteam_non_mutation("a" * 64, "b" * 64)["pass"] is True
    assert redteam_near_dup(["alpha beta gamma", "unrelated zeta"])["pass"] is True


def test_guard_and_packaging():
    import argparse
    args = argparse.Namespace(mode="full", allow_non_colab=False)
    try:
        guard_full_mode(args)
    except ValueError as exc:
        assert "COLAB_GPU" in str(exc)
    else:
        raise AssertionError("full mode was allowed locally")
    args = argparse.Namespace(mode="smoke", allow_non_colab=False)
    guard_full_mode(args)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "run"
        out.mkdir()
        (out / "A.json").write_text("{}", encoding="utf-8")
        archive = package_bundle(out)
        assert archive.is_file()
        manifest = __import__("json").loads(
            (out / "SESSION_MANIFEST.json").read_text(encoding="utf-8"))
        assert manifest["files"] == ["A.json"]


def test_tiny_smoke_cpu():
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    from anra_v5.cyr_execute import smoke
    device = torch.device("cpu")
    with tempfile.TemporaryDirectory() as tmp:
        bundle = smoke(torch, device, Path(tmp) / "out")
    assert bundle["mode"] == "smoke"
    assert bundle["result"]["updates"] == 2
    assert bundle["result"]["pair_splits"] == 0
    assert bundle["result"]["checkpoint_head"] is not None


def test_notebook_valid_and_bound():
    import json
    notebook = json.loads(
        (ROOT / "notebooks" / "cymek_colab_gpu_research.ipynb").read_text(
            encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code_cells = [cell["source"] for cell in notebook["cells"]
                  if cell["cell_type"] == "code"]
    assert len(code_cells) == 3
    for cell in code_cells:
        python_only = "\n".join(
            line for line in "".join(cell).splitlines()
            if not line.lstrip().startswith(("%", "!")))
        compile(python_only, "<notebook-cell>", "exec")
    blob = json.dumps(notebook)
    for required in ("PREREGISTRATION.json", "--mode full", "--mode smoke",
                     "CYMEK_GPU_RESEARCH_RESULTS.zip", "COLAB_GPU"):
        assert required in blob, f"notebook lacks {required}"


_TESTS = [test_proxy_ladder_and_resolver,
          test_notebook_valid_and_bound,
          test_worlds_deterministic_and_firewalled,
          test_worlds_exact_sizing,
          test_pair_batches_grouped_and_matched,
          test_sustained_and_controller,
          test_baselines_and_redteam,
          test_guard_and_packaging,
          test_tiny_smoke_cpu]


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
