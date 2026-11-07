"""No cloud or campaign is executed by these operator-contract tests."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from anra_v5.elendril.action_learning import (
    DecodedAction, NeuralActionPolicy, action_fixture, evaluate_policy,
)
from tools import build_elendril_action_notebooks as builder
from tools import elendril_action_cloud as operator


SOURCE = "a" * 40


def _recorded_model(data):
    # A scripted decoder stub qualifies replay only, never counted as learning.
    targets = {row.prompt: row.target_tokens for row in data.examples}
    return evaluate_policy(data, NeuralActionPolicy(
        lambda prompt, _: DecodedAction(targets[prompt], 2 * len(prompt) + 3, 2, .8), "stub-only"))


def _environment(uuid="GPU-abcd"):
    return {"physical": {"uuid": uuid}, "runtime": {"torch": "2.14+cu"}, "session": "current"}


def test_registered_full_epochs_distinct_gpus_and_budget_forecast():
    assert len(operator.PLANS["colab"].jobs()) == 6
    assert len(operator.PLANS["kaggle"].jobs()) == 15
    for plan in operator.PLANS.values():
        assert plan.updates % 14 == 0
        assert len(operator.fixture(plan.platform)[0].examples) == 224
        canaries = [{"results": [{"arm": arm, "train_seconds": 1., "evaluation_seconds": 1.}
                                  for arm in operator.ARMS]} for _ in range(plan.device_count)]
        assert operator.forecast(plan, canaries)["predicted_seconds"] < plan.train_seconds
        with pytest.raises(ValueError, match="cannot fit"):
            operator.forecast(plan, canaries, elapsed=plan.train_seconds)
        for value in (float("nan"), float("inf"), -1, True):
            with pytest.raises(ValueError):
                operator.forecast(plan, canaries, elapsed=value)
        for value in (float("nan"), float("inf"), 0, -1, True):
            corrupted = deepcopy(canaries)
            corrupted[0]["results"][0]["train_seconds"] = value
            with pytest.raises(ValueError):
                operator.forecast(plan, corrupted)
    rows = operator.inventory.__globals__["gpu_rows"]("0,Tesla T4,GPU-abcd,15360,15000,560\n1,Tesla T4,GPU-ef01,15360,15000,560")
    operator.admit_devices(rows, 2)
    with pytest.raises(ValueError):
        operator.inventory.__globals__["gpu_rows"]("0,Tesla T4,GPU-abcd,15360,15000,560\n1,Tesla T4,GPU-abcd,15360,15000,560")


def test_default_disabled_cells_schema_and_no_side_effects():
    import nbformat
    for platform in builder.FILES:
        book = builder.notebook(platform, SOURCE)
        nbformat.validate(nbformat.from_dict(book))
        namespace = {}
        with patch("subprocess.run", side_effect=AssertionError("disabled notebook launched subprocess")), \
             patch.object(Path, "mkdir", side_effect=AssertionError("disabled notebook wrote storage")):
            for cell in book["cells"]:
                if cell["cell_type"] == "code":
                    exec(compile("".join(cell["source"]), "disabled-cell", "exec"), namespace)
        assert namespace["RUN_CANARIES"] is False
        assert namespace["RUN_RECOVERY"] is False
        assert namespace["RUN_CAMPAIGN"] is False
        assert namespace["FRESH_SESSION_ACK"] is False
        assert namespace["SOURCE"] == SOURCE
        assert namespace["CANARIES"] == []


def test_replay_independently_rejects_wrong_truth_actions_costs_and_denominators():
    data = action_fixture("validation", worlds=10)
    report = _recorded_model(data)
    assert operator.rescore_evaluation(data, report) == report["totals"]
    for field, value in (("requests", 7), ("utility", 123)):
        broken = deepcopy(report)
        broken["episodes"][0][field] = value
        with pytest.raises(ValueError, match="replay"):
            operator.rescore_evaluation(data, broken)
    broken = deepcopy(report)
    broken["episodes"][0]["trace"]["decisions"][0]["origin"] = "fixed_baseline"
    with pytest.raises(ValueError, match="scripted"):
        operator.rescore_evaluation(data, broken)
    broken = deepcopy(report)
    broken["episodes"] = broken["episodes"][:-1]
    with pytest.raises(ValueError, match="denominator"):
        operator.rescore_evaluation(data, broken)
    broken = deepcopy(report)
    broken["totals"]["all"]["correct"] -= 1
    with pytest.raises(ValueError, match="totals"):
        operator.rescore_evaluation(data, broken)


def test_recovery_proof_bound_to_challenge_runtime_session_and_current_uuid(tmp_path, monkeypatch):
    monkeypatch.setattr(operator, "session_identity", lambda: "current")
    environment = _environment()
    original = {**environment, "physical": {"uuid": "GPU-eeee"}, "session": "previous"}
    challenge = {"schema": operator.SCHEMA + "/challenge", "source": SOURCE,
                 "environment": original, "checkpoint": "b"*64, "expected_checkpoint": "c"*64,
                 "end_update": 14}
    proof = {"schema": operator.SCHEMA + "/recovery", "source": SOURCE, "status": "PASS",
        "fresh_session": True, "off_session_owner_ack": True, "environment": environment,
        "original_environment": original, "challenge_document": challenge,
        "challenge_sha256": operator.sha256(operator.blob(challenge)).hexdigest(),
        "checkpoint": "b"*64, "expected_checkpoint": "c"*64, "restored_update": 14,
        "classification": "numerical-equivalent", "rtol": 1e-5, "atol": 1e-6}
    path = tmp_path / "proof.json"
    operator.write_json(path, proof)
    operator.recovery_gate([path], source=SOURCE, environments=[environment])
    for key, value in (("source", "d"*40), ("off_session_owner_ack", False),
                       ("challenge_sha256", "e"*64), ("rtol", .1), ("fresh_session", False)):
        broken = {**proof, key: value}
        operator.write_json(path, broken, replace=True)
        with pytest.raises(ValueError):
            operator.recovery_gate([path], source=SOURCE, environments=[environment])


def test_canary_gate_rejects_source_device_runtime_and_session_drift(monkeypatch):
    monkeypatch.setattr(operator, "session_identity", lambda: "current")
    environment = _environment()
    formation = _recorded_model(action_fixture("train", worlds=5, seed=1729))
    validation = _recorded_model(operator.fixture("colab")[1])
    canary = {"schema": operator.SCHEMA + "/canary", "source": SOURCE, "status": "PASS",
        "platform": "colab", "environment": environment, "formation": formation,
        "results": [{"arm": arm, "updates": [{"update": u} for u in range(1,15)],
                     "same_device_exact": True, "fresh_process": {"status": "PASS"},
                     "evaluation": validation} for arm in operator.ARMS]}
    operator.canary_gate([canary], source=SOURCE, platform="colab", environments=[environment])
    for key, value in (("source", "d"*40), ("platform", "kaggle"), ("status", "FAIL")):
        with pytest.raises(ValueError, match="canary"):
            operator.canary_gate([{**canary, key: value}], source=SOURCE,
                                  platform="colab", environments=[environment])
    broken = deepcopy(canary)
    broken["environment"]["physical"]["uuid"] = "GPU-ffff"
    with pytest.raises(ValueError):
        operator.canary_gate([broken], source=SOURCE, platform="colab", environments=[environment])


def test_screen_gate_frozen_formation_and_owner_review(tmp_path, monkeypatch):
    # Payload-byte verification is covered independently; this test isolates
    # the raw-episode/formation gate without manufacturing a trained checkpoint.
    monkeypatch.setattr(operator, "verify_result_checkpoint", lambda *_: None)
    data = operator.fixture("colab")[1]
    evaluation = _recorded_model(data)
    records = []
    for arm, seed in operator.PLANS["colab"].jobs():
        path = tmp_path / f"{arm}-{seed}.json"
        operator.write_json(path, {"source": SOURCE, "status": "COMPLETE", "arm": arm,
            "seed": seed, "global_update": 448, "evaluation": evaluation})
        records.append({"file": path.name, "sha256": operator.fingerprint(path)})
    summary = {"schema": operator.SCHEMA + "/campaign", "source": SOURCE,
               "platform": "colab", "status": "COMPLETE", "results": records}
    path = tmp_path / "campaign.json"
    operator.write_json(path, summary)
    with pytest.raises(ValueError, match="owner review"):
        operator.screen_gate(path, SOURCE, owner_reviewed=False)
    result = operator.screen_gate(path, SOURCE, owner_reviewed=True)
    assert result["formation_qualified"] == list(operator.ARMS)
    assert result["alternative_signal"] == []  # equal result is not a gain
    bad = deepcopy(evaluation)
    bad["episodes"][0]["outcome"]["correct"] = False
    first = tmp_path / records[0]["file"]
    record = json.loads(first.read_bytes()); record["evaluation"] = bad
    operator.write_json(first, record, replace=True)
    summary["results"][0]["sha256"] = operator.fingerprint(first)
    operator.write_json(path, summary, replace=True)
    with pytest.raises(ValueError, match="replay"):
        operator.screen_gate(path, SOURCE, owner_reviewed=True)


def test_cloud_recipe_uses_reconstructible_category_without_device_work(monkeypatch, tmp_path):
    from anra_v5.elendril import foundation_training, training
    captured = {}
    def fake(samples, **kwargs):
        captured.update(samples=samples, **kwargs)
        return "capture"
    monkeypatch.setattr(foundation_training, "FoundationTrainer", fake)
    monkeypatch.setattr(training, "configure_device", lambda device: None)
    assert operator.engine("colab", "uncertainty-first", 7, SOURCE, tmp_path, "test") == "capture"
    data = operator.fixture("colab")[0]
    assert [row.category for row in captured["samples"]] == [row.kind for row in data.examples]
    assert captured["recipe"].sampling == "uncertainty-first"
    assert captured["recipe"].total_updates == 448
    assert captured["device"] == "cuda:0"
    assert captured["data_provenance"] == data.manifest


def test_screen_checkpoint_requires_hash_readback_and_exact_state_bindings(monkeypatch, tmp_path):
    from anra_v5.elendril.codec import CODEC_SHA, digest
    from anra_v5.elendril.foundation_training import Recipe, TokenSample
    from anra_v5.elendril.training import PROFILES
    train = operator.fixture("colab")[0]
    samples = [TokenSample(row.task_id, row.tokens, row.eligible, row.world_id, row.kind) for row in train.examples]
    data_sha = digest({"samples": [asdict(row) for row in samples], "provenance": train.manifest})
    recipe = asdict(Recipe(seed=7,total_updates=448,batch_size=16,micro_batch_size=4,lr=.003))
    result = {"source": SOURCE, "platform": "colab", "arm": "ce", "seed": 7,
        "recipe": recipe, "training_fixture": train.manifest, "data_sha256": data_sha,
        "model_sha256": PROFILES["expanded"].sha256(), "tokenizer_sha256": CODEC_SHA,
        "checkpoint": "b"*64, "lineage": "test"}
    target = tmp_path / "checkpoints" / "test" / ("b"*64)
    target.mkdir(parents=True)
    state = {"identities": {"source_commit": SOURCE, "tokenizer_sha256": CODEC_SHA,
        "model_spec_sha256": result["model_sha256"], "pack_manifest_sha256": data_sha},
        "lineage_id": "test", "global_update": 448, "optimizer_step_max": 448,
        "cumulative_tokens": 14336, "token_budget": 14336,
        "cursor": {"sequence_ordinal": 448*16}}
    operator.write_json(target / "training_state.json", state)
    operator.write_json(target / "scheduler.json", {"metadata": {"recipe": recipe, "provenance": train.manifest}})
    verified = []
    monkeypatch.setattr(operator, "read_checkpoint", lambda path, sha: verified.append((path,sha)))
    assert operator.verify_result_checkpoint(tmp_path, result) == "b"*64
    assert verified == [(target, "b"*64)]
    operator.write_json(target / "training_state.json", {**state, "optimizer_step_max": 447}, replace=True)
    with pytest.raises(ValueError, match="committed"):
        operator.verify_result_checkpoint(tmp_path, result)
    with pytest.raises(ValueError, match="identity"):
        operator.verify_result_checkpoint(tmp_path, {**result, "tokenizer_sha256": "c"*64})


def test_worker_refuses_implicit_launch_before_any_device_access(tmp_path, monkeypatch):
    monkeypatch.setattr(operator, "worker_environment", lambda *_: (_ for _ in ()).throw(AssertionError("device access")))
    with pytest.raises(SystemExit):
        operator.main(["train", "--source", SOURCE, "--platform", "colab", "--output", str(tmp_path),
                       "--receipt", str(tmp_path / "receipt.json"), "--durable", str(tmp_path)])
