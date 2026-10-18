from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from .audits import audit_suite
from .baselines import policy_factories
from .contracts import content_hash
from .evaluation import run_suite
from .firewall import assert_public_payload
from .generator import generate_suite, public_descriptor, suite_manifest
from .protocol import freeze_dg0_protocol, protocol_template
from .receipts import make_identity, no_checkpoint_identity, result_receipt, source_closure, write_json_exclusive
from .workbench import (
    DiscoverySession,
    build_workbench_report,
    discover_graph,
    LocalMechanismExecutor,
    execute_experiment_plan,
    inventory_readiness,
    load_dataset,
    predictive_benchmark,
    fit_mechanism,
    cross_validate_mechanism,
    recommend_experiments,
    stability_report,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _source_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _public_transcript(run: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "schema": "anra-discovery-gym/public-episode/v1",
            "episode_id": record["episode_id"],
            "policy_id": record["policy_id"],
            "total_cost": record["total_cost"],
            "events": record["events"],
            "commitments": record["commitments"],
        }
        for record in run["records"]
    ]


def run_dg0(output: Path, *, seed: int, worlds_per_split: int, budget: int) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)
    root = _repo_root()
    suite = generate_suite(seed=seed, worlds_per_split=worlds_per_split)
    manifest = suite_manifest(suite, seed=seed)
    protocol = freeze_dg0_protocol(manifest, budget=budget).to_dict()
    run = run_suite(
        suite,
        policy_factories(seed),
        budget=budget,
        allow_hidden_truth=True,
    )
    audit = audit_suite(suite, run, protocol)
    public_worlds = sorted(
        [public_descriptor(world) for worlds in suite.values() for world in worlds],
        key=lambda item: str(item["world_id"]),
    )
    public_records = sorted(
        _public_transcript(run),
        key=lambda item: (item["policy_id"], item["episode_id"]),
    )
    assert_public_payload({"public_worlds": public_worlds, "public_transcripts": public_records})
    source = source_closure(root)
    identity = make_identity(
        source_commit=_source_commit(root),
        source_closure_sha256=source,
        generator_sha256=_file_hash(root / "discovery_gym" / "generator.py"),
        evaluator_sha256=_file_hash(root / "discovery_gym" / "evaluation.py"),
        protocol_sha256=protocol["protocol_sha256"],
        policy_id="all-transparent-baselines-and-oracle",
        model_id="none",
        seed=seed,
    )
    receipt = result_receipt(
        identity=identity,
        protocol=protocol,
        suite_manifest=manifest.to_dict(),
        run=run,
        audit=audit,
        command="python -m discovery_gym dg0",
    )
    artifacts = {
        "suite_manifest.json": manifest.to_dict(),
        "public_worlds.json": public_worlds,
        "protocol.json": protocol,
        "public_transcripts.json": public_records,
        "evaluator_run.json": run,
        "audit.json": audit,
        "result_receipt.json": receipt,
        "checkpoint_identity.json": no_checkpoint_identity(
            experiment_id=protocol["protocol_id"],
            protocol_sha256=protocol["protocol_sha256"],
        ),
    }
    for name, value in artifacts.items():
        write_json_exclusive(output / name, value)
    result = {
        "schema": "anra-discovery-gym/dg0-result/v1",
        "status": audit["status"],
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol["protocol_sha256"],
        "suite_manifest_sha256": content_hash(manifest.to_dict()),
        "record_count": len(run["records"]),
        "audit": audit,
        "claim_ceiling": "DG0 instrument validity only; no model learning or AGI claim",
        "artifacts": sorted(artifacts),
    }
    return result


def run_workbench(
    input_path: Path,
    output: Path | None,
    *,
    max_parents: int,
    stability_replicates: int,
    seed: int,
    discretize_columns: tuple[str, ...] = (),
    bins: int = 4,
    budget: int = 8,
) -> dict[str, Any]:
    dataset = load_dataset(input_path)
    if discretize_columns:
        dataset = dataset.discretize(discretize_columns, bins=bins)
    report = build_workbench_report(
        dataset,
        max_parents=max_parents,
        stability_replicates=stability_replicates,
        seed=seed,
        budget=budget,
    )
    if output is not None:
        write_json_exclusive(output, report)
        report = {**report, "output": str(output)}
    return report


def create_workbench_session(
    input_path: Path,
    output: Path,
    *,
    max_parents: int,
    stability_replicates: int,
    seed: int,
    discretize_columns: tuple[str, ...] = (),
    bins: int = 4,
    budget: int = 8,
) -> dict[str, Any]:
    dataset = load_dataset(input_path)
    if discretize_columns:
        dataset = dataset.discretize(discretize_columns, bins=bins)
    graph = discover_graph(dataset, max_parents=max_parents)
    stability = stability_report(dataset, graph, replicates=stability_replicates, max_parents=max_parents, seed=seed)
    plan = recommend_experiments(dataset, graph, stability=stability, seed=seed, budget=budget)
    session = DiscoverySession.create(dataset, graph, plan)
    session.save(output)
    return session.report()


def replay_workbench_session(input_path: Path, output: Path | None) -> dict[str, Any]:
    session = DiscoverySession.load(input_path)
    report = session.report()
    if output is not None:
        write_json_exclusive(output, report)
        report = {**report, "output": str(output)}
    return report


def record_workbench_event(
    input_path: Path,
    output: Path,
    *,
    kind: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    session = DiscoverySession.load(input_path)
    session.record(kind, payload)
    session.save(output)
    return session.report()


def record_workbench_result(
    input_path: Path,
    output: Path,
    *,
    recommendation_id: str,
    executor_id: str,
    assignment: str | None,
    outcome: Any,
    cost: int,
) -> dict[str, Any]:
    session = DiscoverySession.load(input_path)
    session.record_external_result(
        recommendation_id,
        executor_id=executor_id,
        assignment=assignment,
        outcome=outcome,
        cost=cost,
    )
    session.save(output)
    return session.report()


def run_workbench_fit(
    input_path: Path,
    output: Path | None,
    *,
    max_parents: int,
    seed: int,
) -> dict[str, Any]:
    dataset = load_dataset(input_path)
    graph = discover_graph(dataset, max_parents=max_parents)
    mechanism = fit_mechanism(dataset, graph)
    result = {
        "schema": "anra-discovery-gym/mechanism-fit/v1",
        "dataset": dataset.profile(),
        "graph": graph.to_dict(),
        "mechanism": mechanism.to_dict(),
        "validation": cross_validate_mechanism(dataset, max_parents=max_parents, seed=seed),
        "claim_boundary": "from-scratch discrete prediction diagnostic, not causal proof",
    }
    if output is not None:
        write_json_exclusive(output, result)
        result = {**result, "output": str(output)}
    return result


def run_predictive_benchmark(
    input_path: Path,
    output: Path | None,
    *,
    max_parents: int,
    seed: int,
) -> dict[str, Any]:
    dataset = load_dataset(input_path)
    result = predictive_benchmark(dataset, max_parents=max_parents, seed=seed)
    if output is not None:
        write_json_exclusive(output, result)
        result = {**result, "output": str(output)}
    return result


def run_workbench_execute(
    input_path: Path,
    output: Path,
    *,
    executor_name: str,
    budget: int,
    max_recommendations: int | None,
) -> dict[str, Any]:
    session = DiscoverySession.load(input_path)
    if executor_name != "local-mechanism":
        raise ValueError("the CLI currently supports only the local-mechanism executor")
    graph = discover_graph(session.dataset, max_parents=session.graph.max_parents)
    if graph.graph_sha256 != session.graph.graph_sha256:
        raise ValueError("session graph no longer matches its dataset")
    executor = LocalMechanismExecutor(fit_mechanism(session.dataset, session.graph))
    report = execute_experiment_plan(
        session,
        executor,
        budget=budget,
        max_recommendations=max_recommendations,
    )
    session.save(output)
    return {**report, "output": str(output)}


def run_inventory(input_path: Path, output: Path | None) -> dict[str, Any]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    receipts = payload.get("receipts", []) if isinstance(payload, dict) else payload
    if not isinstance(receipts, list):
        raise ValueError("inventory input must be a receipt list or an object with receipts")
    result = inventory_readiness(receipts)
    if output is not None:
        write_json_exclusive(output, result)
        result = {**result, "output": str(output)}
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="An-Ra Discovery Gym instrument and practical workbench")
    commands = parser.add_subparsers(dest="command", required=True)
    dg0 = commands.add_parser("dg0", help="generate, audit, and run the bounded DG0 suite")
    dg0.add_argument("--output", type=Path, required=True)
    dg0.add_argument("--seed", type=int, default=1701)
    dg0.add_argument("--worlds-per-split", type=int, default=6)
    dg0.add_argument("--budget", type=int, default=8)
    template = commands.add_parser("template", help="write the prospective DG1 protocol template")
    template.add_argument("--output", type=Path, required=True)
    inspect = commands.add_parser("workbench-inspect", help="profile a CSV or JSON dataset")
    inspect.add_argument("--input", type=Path, required=True)
    inspect.add_argument("--output", type=Path)
    plan = commands.add_parser("workbench-plan", help="discover graph hypotheses and rank follow-up experiments")
    plan.add_argument("--input", type=Path, required=True)
    plan.add_argument("--output", type=Path)
    plan.add_argument("--max-parents", type=int, default=2)
    plan.add_argument("--stability-replicates", type=int, default=8)
    plan.add_argument("--seed", type=int, default=0)
    plan.add_argument("--discretize", nargs="*", default=[])
    plan.add_argument("--bins", type=int, default=4)
    plan.add_argument("--budget", type=int, default=8)
    session = commands.add_parser("workbench-session", help="create a durable workbench session")
    session.add_argument("--input", type=Path, required=True)
    session.add_argument("--output", type=Path, required=True)
    session.add_argument("--max-parents", type=int, default=2)
    session.add_argument("--stability-replicates", type=int, default=8)
    session.add_argument("--seed", type=int, default=0)
    session.add_argument("--discretize", nargs="*", default=[])
    session.add_argument("--bins", type=int, default=4)
    session.add_argument("--budget", type=int, default=8)
    replay = commands.add_parser("workbench-replay", help="verify and replay a workbench session")
    replay.add_argument("--input", type=Path, required=True)
    replay.add_argument("--output", type=Path)
    record = commands.add_parser("workbench-record", help="append an evidence event to a new session file")
    record.add_argument("--input", type=Path, required=True)
    record.add_argument("--output", type=Path, required=True)
    record.add_argument("--kind", required=True)
    record.add_argument("--payload", required=True)
    result = commands.add_parser("workbench-result", help="record a validated external experiment result")
    result.add_argument("--input", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--recommendation", required=True)
    result.add_argument("--executor", required=True)
    result.add_argument("--assignment")
    result.add_argument("--outcome", required=True)
    result.add_argument("--cost", type=int, required=True)
    execute = commands.add_parser("workbench-execute", help="execute a bounded plan with the local mechanism simulator")
    execute.add_argument("--input", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--executor", choices=["local-mechanism"], default="local-mechanism")
    execute.add_argument("--budget", type=int, default=8)
    execute.add_argument("--max-recommendations", type=int)
    fit = commands.add_parser("workbench-fit", help="fit and cross-validate a from-scratch discrete mechanism")
    fit.add_argument("--input", type=Path, required=True)
    fit.add_argument("--output", type=Path)
    fit.add_argument("--max-parents", type=int, default=2)
    fit.add_argument("--seed", type=int, default=0)
    benchmark = commands.add_parser("workbench-benchmark", help="compare graph, full-conditional, and majority predictors")
    benchmark.add_argument("--input", type=Path, required=True)
    benchmark.add_argument("--output", type=Path)
    benchmark.add_argument("--max-parents", type=int, default=2)
    benchmark.add_argument("--seed", type=int, default=0)
    inventory = commands.add_parser("workbench-inventory", help="reconcile dataset receipts and qualification status")
    inventory.add_argument("--input", type=Path, required=True)
    inventory.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "dg0":
            result = run_dg0(
                args.output,
                seed=args.seed,
                worlds_per_split=args.worlds_per_split,
                budget=args.budget,
            )
            print(json.dumps(result, sort_keys=True))
            return 0 if result["status"] == "PASS" else 1
        if args.command == "workbench-inspect":
            dataset = load_dataset(args.input)
            result = {**dataset.profile(), "readiness": dataset.readiness()}
            if args.output is not None:
                write_json_exclusive(args.output, result)
                result = {**result, "output": str(args.output)}
            print(json.dumps(result, sort_keys=True))
            return 0
        if args.command == "workbench-plan":
            result = run_workbench(
                args.input,
                args.output,
                max_parents=args.max_parents,
                stability_replicates=args.stability_replicates,
                seed=args.seed,
                discretize_columns=tuple(args.discretize),
                bins=args.bins,
                budget=args.budget,
            )
            print(json.dumps(result, sort_keys=True))
            return 0 if result.get("audit", {}).get("status") == "PASS" else 1
        if args.command == "workbench-benchmark":
            result = run_predictive_benchmark(args.input, args.output, max_parents=args.max_parents, seed=args.seed)
            print(json.dumps(result, sort_keys=True))
            return 0
        if args.command == "workbench-execute":
            result = run_workbench_execute(
                args.input,
                args.output,
                executor_name=args.executor,
                budget=args.budget,
                max_recommendations=args.max_recommendations,
            )
            print(json.dumps(result, sort_keys=True))
            return 0
        if args.command == "workbench-fit":
            result = run_workbench_fit(args.input, args.output, max_parents=args.max_parents, seed=args.seed)
            print(json.dumps(result, sort_keys=True))
            return 0
        if args.command == "workbench-inventory":
            result = run_inventory(args.input, args.output)
            print(json.dumps(result, sort_keys=True))
            return 0 if result["state"] != "BLOCKED" else 1
        if args.command == "workbench-session":
            result = create_workbench_session(
                args.input,
                args.output,
                max_parents=args.max_parents,
                stability_replicates=args.stability_replicates,
                seed=args.seed,
                discretize_columns=tuple(args.discretize),
                bins=args.bins,
                budget=args.budget,
            )
            print(json.dumps(result, sort_keys=True))
            return 0
        if args.command == "workbench-replay":
            result = replay_workbench_session(args.input, args.output)
            print(json.dumps(result, sort_keys=True))
            return 0
        if args.command == "workbench-result":
            outcome = json.loads(args.outcome)
            result = record_workbench_result(
                args.input,
                args.output,
                recommendation_id=args.recommendation,
                executor_id=args.executor,
                assignment=args.assignment,
                outcome=outcome,
                cost=args.cost,
            )
            print(json.dumps(result, sort_keys=True))
            return 0
        if args.command == "workbench-record":
            payload = json.loads(args.payload)
            if not isinstance(payload, dict):
                raise ValueError("record payload must be a JSON object")
            result = record_workbench_event(args.input, args.output, kind=args.kind, payload=payload)
            print(json.dumps(result, sort_keys=True))
            return 0
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json_exclusive(args.output, protocol_template())
        print(json.dumps({"status": "WROTE", "path": str(args.output)}, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"discovery-gym: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
