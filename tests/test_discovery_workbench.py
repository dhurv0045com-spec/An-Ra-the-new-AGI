from __future__ import annotations

import json
from pathlib import Path

import pytest

from discovery_gym.baselines import TransparentLookaheadPolicy
from discovery_gym.cli import main
from discovery_gym.workbench import (
    DiscoverySession,
    EvidenceEvent,
    ExperimentPlan,
    GraphHypothesis,
    TableDataset,
    benchmark_graph_discovery,
    build_workbench_report,
    cross_validate_mechanism,
    discover_graph,
    fit_mechanism,
    graph_recovery_metrics,
    inventory_readiness,
    predictive_benchmark,
    qualification_receipt,
    recommend_experiments,
    stability_report,
)


def _and_dataset() -> TableDataset:
    rows = [[a, b, a and b] for a in (0, 1) for b in (0, 1) for _ in range(12)]
    return TableDataset.from_rows("and-demo", ["a", "b", "c"], rows)


def test_dataset_identity_profile_and_split_are_deterministic():
    dataset = _and_dataset()
    assert dataset == TableDataset.from_rows("and-demo", ["a", "b", "c"], dataset.rows)
    assert TableDataset.from_rows("and-demo", ["a", "b", "c"], dataset.rows, metadata={"source": "fixture"}).source_sha256 != dataset.source_sha256
    profile = dataset.profile()
    assert profile["row_count"] == 48
    assert profile["columns"]["c"]["counts"] == {"0": 36, "1": 12}
    first = dataset.split(seed=17)
    second = dataset.split(seed=17)
    assert first.keys() == second.keys()
    assert first["train"].rows == second["train"].rows
    assert len(first["train"].rows) + len(first["holdout"].rows) == len(dataset.rows)
    assert not set(first["train"].rows) & set(first["holdout"].rows)
    assert first["train"].metadata["parent"] == dataset.source_sha256
    assert first["train"].metadata["split_unit"] == "duplicate_row_group"
    readiness = dataset.readiness()
    assert readiness["state"] == "VERIFIED"
    assert not readiness["qualified"]
    assert not readiness["runnable"]
    with pytest.raises(ValueError):
        TableDataset.from_rows("missing", ["a"], [[""]])


def test_inventory_deduplicates_and_requires_exact_qualification():
    dataset = _and_dataset()
    receipt = dataset.receipt(
        namespace="fixture",
        lineage_sha256="a" * 64,
        tokenizer_sha256="b" * 64,
        qualification_status="QUALIFIED",
    )
    qualified = inventory_readiness([receipt])
    assert qualified["state"] == "QUALIFIED"
    duplicate = inventory_readiness([receipt, dict(receipt)])
    assert duplicate["unique_dataset_count"] == 1
    assert duplicate["state"] == "BLOCKED"
    unqualified = dict(receipt)
    unqualified["qualification_status"] = "UNQUALIFIED"
    assert inventory_readiness([unqualified])["state"] == "VERIFIED"
    assert inventory_readiness([unqualified])["qualified_unique_dataset_count"] == 0
    qualification = qualification_receipt(dataset, protocol_id="fixture/v1", checks={"exact_data": True, "shortcut": True})
    assert dataset.readiness(qualification=qualification)["qualified"]


def test_numeric_discretization_is_explicit_and_reversible_by_provenance():
    dataset = TableDataset.from_rows("numeric", ["x", "y"], [[index, "low" if index < 4 else "high"] for index in range(8)])
    discretized = dataset.discretize(["x"], bins=2)
    assert discretized.metadata["operation"] == "quantile_discretization"
    assert set(discretized.value_counts("x")) <= {"q0", "q1"}
    assert discretized.metadata["parent"] == dataset.source_sha256


def test_graph_discovery_recovers_directed_structure_and_round_trips():
    dataset = _and_dataset()
    graph = discover_graph(dataset, max_parents=2)
    assert set(graph.edges) == {("a", "c"), ("b", "c")}
    assert set(graph.order) == set(dataset.columns)
    assert graph.parents["c"] == ("a", "b")
    assert GraphHypothesis.from_dict(graph.to_dict()) == graph
    recovery = graph_recovery_metrics(graph, [("a", "c"), ("b", "c")])
    assert recovery["f1"] == 1.0
    assert recovery["false_positive"] == 0


def test_from_scratch_mechanism_learner_predicts_and_validates():
    dataset = _and_dataset()
    graph = discover_graph(dataset)
    mechanism = fit_mechanism(dataset, graph)
    assert mechanism.predict("c", {"a": 1, "b": 1}) == 1
    assert mechanism.predict("c", {"a": 0, "b": 1}) == 0
    assert mechanism.to_dict()["mechanism_sha256"]
    validation = cross_validate_mechanism(dataset, seed=9)
    assert validation["by_child"]["c"]["rows"] == 48
    assert validation["by_child"]["c"]["brier"] is not None
    comparison = predictive_benchmark(dataset, seed=9)
    assert set(comparison["models"]) == {"majority", "full-conditional", "graph-mechanism"}
    assert comparison["winner_by_mean_accuracy"]


def test_stability_and_experiment_plan_prioritize_testable_edges():
    dataset = _and_dataset()
    graph = discover_graph(dataset)
    stability = stability_report(dataset, graph, replicates=8, seed=4)
    assert stability["baseline_edge_count"] == 2
    assert set(stability["edge_support"]) >= {"a->c", "b->c"}
    plan = recommend_experiments(dataset, graph, stability=stability, limit=4, budget=7)
    assert plan.planned_cost <= 7
    assert plan.recommendations[0].kind == "randomized_intervention"
    assert {item.target for item in plan.recommendations} >= {"a", "b"}
    assert ExperimentPlan.from_dict(plan.to_dict()) == plan
    report = build_workbench_report(dataset, stability_replicates=2)
    assert report["utility_status"] == "ready_for_external_observation_or_intervention"
    assert report["graph"]["graph_sha256"]
    assert report["audit"]["status"] == "PASS"
    benchmark = benchmark_graph_discovery([(dataset, [("a", "c"), ("b", "c")])], stability_replicates=2)
    assert benchmark["mean_f1"] == 1.0


def test_evidence_ledger_session_replay_and_tamper_detection(tmp_path: Path):
    dataset = _and_dataset()
    graph = discover_graph(dataset)
    plan = recommend_experiments(dataset, graph, stability=stability_report(dataset, graph, replicates=2))
    session = DiscoverySession.create(dataset, graph, plan)
    session.record("dataset_verified", {"source_sha256": dataset.source_sha256})
    recommendation = plan.recommendations[0]
    session.record_external_result(
        recommendation.recommendation_id,
        executor_id="fixture-executor",
        assignment=recommendation.values[0] if recommendation.values else None,
        outcome="observed",
        cost=recommendation.cost,
    )
    assert session.ledger.verify()["status"] == "PASS"
    assert session.next_recommendation() is not None
    assert len(session.replay()) == 2
    path = tmp_path / "session.json"
    session.save(path)
    loaded = DiscoverySession.load(path)
    assert loaded.to_dict() == session.to_dict()
    events = session.ledger.to_dict()["events"]
    events[0]["payload"]["source_sha256"] = "tampered"
    with pytest.raises(ValueError):
        EvidenceEvent.from_dict(events[0])


def test_workbench_cli_writes_inspection_and_plan(tmp_path: Path, capsys):
    source = tmp_path / "input.csv"
    source.write_text("a,b,c\n0,0,0\n0,1,0\n1,0,0\n1,1,1\n", encoding="utf-8")
    inspection = tmp_path / "inspection.json"
    plan = tmp_path / "plan.json"
    assert main(["workbench-inspect", "--input", str(source), "--output", str(inspection)]) == 0
    assert main(["workbench-plan", "--input", str(source), "--output", str(plan), "--stability-replicates", "2", "--discretize", "a", "--bins", "2"]) == 0
    assert json.loads(inspection.read_text())["row_count"] == 4
    fit = tmp_path / "fit.json"
    assert main(["workbench-fit", "--input", str(source), "--output", str(fit)]) == 0
    assert json.loads(fit.read_text())["mechanism"]["mechanism_sha256"]
    benchmark = tmp_path / "benchmark.json"
    assert main(["workbench-benchmark", "--input", str(source), "--output", str(benchmark)]) == 0
    assert json.loads(benchmark.read_text())["winner_by_mean_accuracy"]
    jsonl = tmp_path / "input.jsonl"
    jsonl.write_text('{"a":0,"b":0,"c":0}\n{"a":1,"b":1,"c":1}\n', encoding="utf-8")
    assert main(["workbench-inspect", "--input", str(jsonl)]) == 0
    saved = json.loads(plan.read_text())
    assert saved["schema"] == "anra-discovery-gym/workbench/v1"
    assert saved["plan"]["recommendations"]
    receipts = tmp_path / "receipts.json"
    receipt = _and_dataset().receipt(namespace="fixture", lineage_sha256="a" * 64, tokenizer_sha256="b" * 64, qualification_status="QUALIFIED")
    receipts.write_text(json.dumps([receipt]), encoding="utf-8")
    inventory = tmp_path / "inventory.json"
    assert main(["workbench-inventory", "--input", str(receipts), "--output", str(inventory)]) == 0
    assert json.loads(inventory.read_text())["state"] == "QUALIFIED"
    session_path = tmp_path / "session.json"
    updated_path = tmp_path / "session-updated.json"
    replay_path = tmp_path / "replay.json"
    assert main(["workbench-session", "--input", str(source), "--output", str(session_path), "--stability-replicates", "2"]) == 0
    session_payload = json.loads(session_path.read_text())
    recommendation = session_payload["plan"]["recommendations"][0]
    assignment = recommendation["values"][0] if recommendation["values"] else None
    result_args = [
        "workbench-result",
        "--input",
        str(session_path),
        "--output",
        str(updated_path),
        "--recommendation",
        recommendation["recommendation_id"],
        "--executor",
        "fixture-executor",
        "--outcome",
        '"observed"',
        "--cost",
        str(recommendation["cost"]),
    ]
    if assignment is not None:
        result_args.extend(["--assignment", assignment])
    assert main(result_args) == 0
    assert main(["workbench-replay", "--input", str(updated_path), "--output", str(replay_path)]) == 0
    assert json.loads(replay_path.read_text())["observed_events"] == 1
    captured = capsys.readouterr()
    assert "workbench-inspect" not in captured.out
