from __future__ import annotations

import csv
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
from typing import Any

from .contracts import content_hash
from .receipts import write_json_exclusive


WORKBENCH_SCHEMA = "anra-discovery-gym/workbench/v1"
GENESIS_HASH = "0" * 64
MAX_EXACT_COLUMNS = 8


def _cell_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite cell values are not supported")
        return format(value, ".12g")
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("missing or blank cell values are not supported")
        return value
    raise TypeError(f"workbench cells must be scalar values, got {type(value).__name__}")


def _normalize_columns(columns: Sequence[Any]) -> tuple[str, ...]:
    values = tuple(str(column).strip() for column in columns)
    if not values or any(not value for value in values):
        raise ValueError("columns must be nonempty names")
    if len(values) != len(set(values)):
        raise ValueError("column names must be unique")
    return values


def _normalize_rows(rows: Iterable[Sequence[Any]], width: int) -> tuple[tuple[str, ...], ...]:
    normalized = tuple(tuple(_cell_text(value) for value in row) for row in rows)
    if not normalized:
        raise ValueError("a dataset needs at least one row")
    if any(len(row) != width for row in normalized):
        raise ValueError("every row must match the declared column count")
    return normalized


def _dataset_hash(
    name: str,
    columns: tuple[str, ...],
    rows: tuple[tuple[str, ...], ...],
    metadata: Mapping[str, Any] | None = None,
) -> str:
    return content_hash({
        "name": name,
        "columns": list(columns),
        "rows": [list(row) for row in rows],
        "metadata": dict(metadata or {}),
    })


@dataclass(frozen=True, slots=True)
class TableDataset:
    name: str
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    source_sha256: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("dataset name must be nonempty")
        columns = _normalize_columns(self.columns)
        rows = _normalize_rows(self.rows, len(columns))
        if self.source_sha256 != _dataset_hash(self.name, columns, rows, self.metadata):
            raise ValueError("dataset source hash does not match its contents")
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_rows(
        cls,
        name: str,
        columns: Sequence[Any],
        rows: Iterable[Sequence[Any]],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> "TableDataset":
        normalized_columns = _normalize_columns(columns)
        normalized_rows = _normalize_rows(rows, len(normalized_columns))
        return cls(
            name=name,
            columns=normalized_columns,
            rows=normalized_rows,
            source_sha256=_dataset_hash(name, normalized_columns, normalized_rows, metadata),
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TableDataset":
        if not isinstance(value, Mapping):
            raise TypeError("dataset payload must be a mapping")
        required = {"name", "columns", "rows", "source_sha256"}
        missing = required - set(value)
        if missing:
            raise ValueError(f"dataset payload is missing {sorted(missing)}")
        return cls(
            name=str(value["name"]),
            columns=tuple(value["columns"]),
            rows=tuple(tuple(row) for row in value["rows"]),
            source_sha256=str(value["source_sha256"]),
            metadata=dict(value.get("metadata", {})),
        )

    @classmethod
    def from_csv(cls, path: Path, *, name: str | None = None) -> "TableDataset":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError("CSV must have a header")
            rows = list(reader)
        return cls.from_rows(name or path.stem, reader.fieldnames, ([row[field] for field in reader.fieldnames] for row in rows))

    @classmethod
    def from_json(cls, path: Path, *, name: str | None = None) -> "TableDataset":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            if not payload or not all(isinstance(row, Mapping) for row in payload):
                raise ValueError("JSON row data must be a nonempty list of objects")
            columns = list(payload[0])
            if any(set(row) != set(columns) for row in payload):
                raise ValueError("JSON rows must share one schema")
            rows = [[row[column] for column in columns] for row in payload]
            return cls.from_rows(name or path.stem, columns, rows)
        if not isinstance(payload, Mapping):
            raise ValueError("JSON dataset must be a row list or object")
        return cls.from_rows(
            name or str(payload.get("name", path.stem)),
            payload.get("columns", ()),
            payload.get("rows", ()),
            metadata=payload.get("metadata", {}),
        )

    @classmethod
    def from_jsonl(cls, path: Path, *, name: str | None = None) -> "TableDataset":
        payloads = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not payloads or not all(isinstance(row, Mapping) for row in payloads):
            raise ValueError("JSONL must contain nonempty object rows")
        columns = list(payloads[0])
        if any(set(row) != set(columns) for row in payloads):
            raise ValueError("JSONL rows must share one schema")
        return cls.from_rows(
            name or path.stem,
            columns,
            [[row[column] for column in columns] for row in payloads],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "columns": list(self.columns),
            "rows": [list(row) for row in self.rows],
            "source_sha256": self.source_sha256,
            "metadata": dict(self.metadata),
        }

    def value_counts(self, column: str) -> dict[str, int]:
        if column not in self.columns:
            raise KeyError(column)
        index = self.columns.index(column)
        return dict(sorted(Counter(row[index] for row in self.rows).items()))

    def profile(self) -> dict[str, Any]:
        profiles: dict[str, Any] = {}
        for column in self.columns:
            counts = self.value_counts(column)
            total = sum(counts.values())
            probabilities = [count / total for count in counts.values()]
            entropy = -sum(probability * math.log2(probability) for probability in probabilities if probability)
            profiles[column] = {
                "counts": counts,
                "unique_count": len(counts),
                "entropy_bits": entropy,
                "missing_count": 0,
            }
        duplicate_count = len(self.rows) - len(set(self.rows))
        return {
            "schema": WORKBENCH_SCHEMA,
            "dataset_name": self.name,
            "source_sha256": self.source_sha256,
            "row_count": len(self.rows),
            "column_count": len(self.columns),
            "duplicate_row_count": duplicate_count,
            "metadata": dict(self.metadata),
            "columns": profiles,
        }

    def receipt(
        self,
        *,
        namespace: str,
        lineage_sha256: str,
        tokenizer_sha256: str,
        qualification_status: str = "UNQUALIFIED",
        qualification_protocol: str | None = None,
    ) -> dict[str, Any]:
        if not namespace or not lineage_sha256 or not tokenizer_sha256:
            raise ValueError("dataset receipts require namespace, lineage, and tokenizer identities")
        return {
            "schema": "anra-discovery-gym/dataset-receipt/v1",
            "dataset_sha256": self.source_sha256,
            "namespace": namespace,
            "lineage_sha256": lineage_sha256,
            "tokenizer_sha256": tokenizer_sha256,
            "example_count": len(self.rows),
            "qualification_status": qualification_status,
            "qualification_protocol": qualification_protocol,
        }

    def readiness(self, *, qualification: Mapping[str, Any] | None = None) -> dict[str, Any]:
        profile = self.profile()
        blockers: list[str] = []
        warnings: list[str] = []
        constant_columns = [
            column
            for column, values in profile["columns"].items()
            if values["unique_count"] <= 1
        ]
        if constant_columns:
            blockers.append(f"constant columns cannot support graph discovery: {sorted(constant_columns)}")
        if profile["duplicate_row_count"]:
            warnings.append("duplicate rows are grouped during splitting")
        qualification_value = dict(qualification or {})
        exact_qualification = (
            qualification_value.get("status") == "QUALIFIED"
            and qualification_value.get("dataset_sha256") == self.source_sha256
            and bool(qualification_value.get("protocol_sha256"))
        )
        if not exact_qualification:
            blockers.append("exact dataset qualification receipt is missing or mismatched")
        blockers.append("external experiment executor is not connected")
        return {
            "schema": "anra-discovery-gym/data-readiness/v1",
            "dataset_sha256": self.source_sha256,
            "state": "VERIFIED",
            "declared": True,
            "materialized": True,
            "verified": True,
            "qualified": exact_qualification,
            "runnable": False,
            "blockers": blockers,
            "warnings": warnings,
            "next_state": "QUALIFIED" if exact_qualification else "VERIFIED",
            "qualification": qualification_value,
        }

    def discretize(self, columns: Sequence[str], *, bins: int = 4) -> "TableDataset":
        if not 2 <= bins <= 32:
            raise ValueError("bins must be between 2 and 32")
        selected = tuple(columns)
        if len(selected) != len(set(selected)) or any(column not in self.columns for column in selected):
            raise ValueError("discretization columns must be unique and declared")
        if not selected:
            return self
        maps: dict[str, dict[str, str]] = {}
        for column in selected:
            try:
                numeric = sorted(float(value) for value in self.value_counts(column))
            except ValueError as exc:
                raise ValueError(f"column {column} is not numeric-like") from exc
            if len(numeric) < 2:
                raise ValueError(f"column {column} has too few numeric values to discretize")
            width = len(numeric) / bins
            maps[column] = {
                format(value, ".12g"): f"q{min(bins - 1, int(index * bins / len(numeric)))}"
                for index, value in enumerate(numeric)
            }
        rows = []
        for row in self.rows:
            output = list(row)
            for column in selected:
                output[self.columns.index(column)] = maps[column][format(float(row[self.columns.index(column)]), ".12g")]
            rows.append(output)
        return TableDataset.from_rows(
            f"{self.name}:discretized",
            self.columns,
            rows,
            metadata={
                "parent": self.source_sha256,
                "operation": "quantile_discretization",
                "columns": list(selected),
                "bins": bins,
            },
        )

    def split(self, *, seed: int = 0, holdout_fraction: float = 0.2) -> dict[str, "TableDataset"]:
        if not 0.0 < holdout_fraction < 1.0:
            raise ValueError("holdout_fraction must be between zero and one")
        grouped: dict[tuple[str, ...], list[tuple[str, ...]]] = defaultdict(list)
        for row in self.rows:
            grouped[row].append(row)
        group_digests = {
            group: hashlib.sha256(f"{seed}:{content_hash(list(group))}".encode()).hexdigest()
            for group in grouped
        }
        holdout_groups = {
            group
            for group, digest in group_digests.items()
            if int(digest[:16], 16) / 0xFFFFFFFFFFFFFFFF < holdout_fraction
        }
        if not holdout_groups:
            holdout_groups.add(min(group_digests, key=lambda group: group_digests[group]))
        if len(holdout_groups) == len(grouped):
            holdout_groups.remove(max(group_digests, key=lambda group: group_digests[group]))
        train_rows = [row for row in self.rows if row not in holdout_groups]
        holdout_rows = [row for row in self.rows if row in holdout_groups]
        if not train_rows or not holdout_rows:
            raise ValueError("split produced an empty partition")
        return {
            "train": TableDataset.from_rows(
                f"{self.name}:train",
                self.columns,
                train_rows,
                metadata={"parent": self.source_sha256, "seed": seed, "split_unit": "duplicate_row_group"},
            ),
            "holdout": TableDataset.from_rows(
                f"{self.name}:holdout",
                self.columns,
                holdout_rows,
                metadata={"parent": self.source_sha256, "seed": seed, "split_unit": "duplicate_row_group"},
            ),
        }


@dataclass(frozen=True, slots=True)
class GraphHypothesis:
    dataset_sha256: str
    columns: tuple[str, ...]
    order: tuple[str, ...]
    parents: Mapping[str, tuple[str, ...]]
    local_scores: Mapping[str, float]
    score: float
    algorithm: str
    max_parents: int

    def __post_init__(self) -> None:
        if isinstance(self.max_parents, bool) or not isinstance(self.max_parents, int) or not 1 <= self.max_parents <= 4:
            raise ValueError("graph max_parents must be an integer between 1 and 4")
        if not math.isfinite(self.score) or any(not math.isfinite(float(value)) for value in self.local_scores.values()):
            raise ValueError("graph scores must be finite")
        columns = _normalize_columns(self.columns)
        if set(self.order) != set(columns) or len(self.order) != len(columns):
            raise ValueError("graph order must contain every column exactly once")
        parents = {column: tuple(self.parents.get(column, ())) for column in columns}
        if any(parent not in columns or child not in columns or parent == child for child, values in parents.items() for parent in values):
            raise ValueError("graph parents must refer to distinct declared columns")
        if any(len(values) > self.max_parents for values in parents.values()):
            raise ValueError("graph parent sets exceed max_parents")
        if set(parents) != set(columns) or len(parents) != len(columns):
            raise ValueError("graph parents must cover every column")
        position = {column: index for index, column in enumerate(self.order)}
        if any(position[parent] >= position[child] for child, values in parents.items() for parent in values):
            raise ValueError("graph order must be topological for every parent edge")
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "order", tuple(self.order))
        object.__setattr__(self, "parents", parents)

    @property
    def edges(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (parent, child)
            for child in self.columns
            for parent in self.parents.get(child, ())
        )

    @property
    def graph_sha256(self) -> str:
        return content_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema": "anra-discovery-gym/causal-hypothesis/v1",
            "dataset_sha256": self.dataset_sha256,
            "columns": list(self.columns),
            "order": list(self.order),
            "parents": {column: list(self.parents.get(column, ())) for column in self.columns},
            "local_scores": {column: float(self.local_scores.get(column, 0.0)) for column in self.columns},
            "score": float(self.score),
            "algorithm": self.algorithm,
            "max_parents": int(self.max_parents),
        }
        return {**body, "graph_sha256": content_hash(body)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GraphHypothesis":
        if not isinstance(value, Mapping):
            raise TypeError("graph payload must be a mapping")
        body = dict(value)
        graph_hash = body.pop("graph_sha256", None)
        if not isinstance(graph_hash, str) or content_hash(body) != graph_hash:
            raise ValueError("graph hash does not match graph contents")
        return cls(
            dataset_sha256=str(body["dataset_sha256"]),
            columns=tuple(body["columns"]),
            order=tuple(body["order"]),
            parents={key: tuple(val) for key, val in body["parents"].items()},
            local_scores={key: float(val) for key, val in body["local_scores"].items()},
            score=float(body["score"]),
            algorithm=str(body["algorithm"]),
            max_parents=int(body["max_parents"]),
        )


@dataclass(frozen=True, slots=True)
class FittedMechanism:
    dataset_sha256: str
    graph_sha256: str
    columns: tuple[str, ...]
    parents: Mapping[str, tuple[str, ...]]
    counts: Mapping[str, Mapping[str, Mapping[str, int]]]
    majority: Mapping[str, int]

    def _key(self, child: str, assignment: Mapping[str, Any]) -> str:
        return "|".join(str(assignment[parent]) for parent in self.parents[child])

    def predict_proba(self, child: str, assignment: Mapping[str, Any]) -> float:
        if child not in self.columns:
            raise KeyError(child)
        key = self._key(child, assignment)
        bucket = self.counts[child].get(key, {})
        if not bucket:
            return float(self.majority.get(child, 0))
        total = sum(bucket.values())
        positive = bucket.get("1", 0)
        return (positive + 1.0) / (total + 2.0)

    def predict(self, child: str, assignment: Mapping[str, Any]) -> int:
        return int(self.predict_proba(child, assignment) >= 0.5)

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema": "anra-discovery-gym/fitted-mechanism/v1",
            "dataset_sha256": self.dataset_sha256,
            "graph_sha256": self.graph_sha256,
            "columns": list(self.columns),
            "parents": {column: list(values) for column, values in self.parents.items()},
            "counts": {column: {key: dict(values) for key, values in bucket.items()} for column, bucket in self.counts.items()},
            "majority": dict(self.majority),
        }
        return {**body, "mechanism_sha256": content_hash(body)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FittedMechanism":
        body = dict(value)
        mechanism_hash = body.pop("mechanism_sha256", None)
        if not isinstance(mechanism_hash, str) or content_hash(body) != mechanism_hash:
            raise ValueError("mechanism hash does not match mechanism contents")
        return cls(
            dataset_sha256=str(body["dataset_sha256"]),
            graph_sha256=str(body["graph_sha256"]),
            columns=tuple(body["columns"]),
            parents={key: tuple(val) for key, val in body["parents"].items()},
            counts={key: {inner: dict(values) for inner, values in bucket.items()} for key, bucket in body["counts"].items()},
            majority={key: int(val) for key, val in body["majority"].items()},
        )


def _fit_parents(dataset: TableDataset, parents: Mapping[str, Sequence[str]], graph_sha256: str) -> FittedMechanism:
    counts: dict[str, dict[str, Counter[str]]] = {column: defaultdict(Counter) for column in dataset.columns}
    majority: dict[str, int] = {}
    normalized_parents = {column: tuple(parents.get(column, ())) for column in dataset.columns}
    for row in dataset.rows:
        values = dict(zip(dataset.columns, row))
        for column in dataset.columns:
            key = "|".join(values[parent] for parent in normalized_parents[column])
            counts[column][key][values[column]] += 1
    for column in dataset.columns:
        totals: Counter[str] = Counter()
        for bucket in counts[column].values():
            totals.update(bucket)
        majority[column] = int(totals.most_common(1)[0][0]) if totals else 0
    frozen_counts = {
        column: {key: dict(values) for key, values in sorted(bucket.items())}
        for column, bucket in counts.items()
    }
    return FittedMechanism(
        dataset_sha256=dataset.source_sha256,
        graph_sha256=graph_sha256,
        columns=dataset.columns,
        parents=normalized_parents,
        counts=frozen_counts,
        majority=majority,
    )


def fit_mechanism(dataset: TableDataset, graph: GraphHypothesis) -> FittedMechanism:
    if graph.dataset_sha256 != dataset.source_sha256:
        raise ValueError("graph and dataset identities do not match")
    return _fit_parents(dataset, graph.parents, graph.graph_sha256)


def _leave_one_group_out_folds(dataset: TableDataset, seed: int) -> list[tuple[TableDataset, tuple[tuple[str, ...], ...]]]:
    grouped: dict[tuple[str, ...], list[tuple[str, ...]]] = defaultdict(list)
    for row in dataset.rows:
        grouped[row].append(row)
    groups = [grouped[row] for row in sorted(grouped)]
    if len(groups) < 2:
        raise ValueError("cross-validation needs at least two distinct row groups")
    offset = seed % len(groups)
    groups = groups[offset:] + groups[:offset]
    folds: list[tuple[TableDataset, tuple[tuple[str, ...], ...]]] = []
    for index, holdout_rows in enumerate(groups):
        holdout_set = set(holdout_rows)
        train_rows = [row for row in dataset.rows if row not in holdout_set]
        folds.append((TableDataset.from_rows(f"{dataset.name}:cv-train:{index}", dataset.columns, train_rows), tuple(holdout_rows)))
    return folds


def cross_validate_mechanism(dataset: TableDataset, *, max_parents: int = 2, seed: int = 0) -> dict[str, Any]:
    aggregates: dict[str, dict[str, float | int]] = {
        column: {"rows": 0, "correct": 0, "brier_sum": 0.0} for column in dataset.columns
    }
    folds: list[dict[str, Any]] = []
    for index, (train, holdout_rows) in enumerate(_leave_one_group_out_folds(dataset, seed)):
        graph = discover_graph(train, max_parents=max_parents)
        model = fit_mechanism(train, graph)
        fold_metrics: dict[str, Any] = {"fold": index, "graph_sha256": graph.graph_sha256, "mechanism_sha256": model.to_dict()["mechanism_sha256"]}
        for child in dataset.columns:
            rows = [dict(zip(dataset.columns, row)) for row in holdout_rows]
            labels = [int(row[child]) for row in rows]
            probabilities = [model.predict_proba(child, row) for row in rows]
            correct = sum(model.predict(child, row) == label for row, label in zip(rows, labels))
            brier_sum = sum((probability - label) ** 2 for probability, label in zip(probabilities, labels))
            aggregates[child]["rows"] += len(rows)
            aggregates[child]["correct"] += correct
            aggregates[child]["brier_sum"] += brier_sum
            fold_metrics[child] = {"rows": len(rows), "accuracy": correct / len(rows), "brier": brier_sum / len(rows)}
        folds.append(fold_metrics)
    by_child = {
        child: {
            "rows": int(values["rows"]),
            "accuracy": float(values["correct"]) / int(values["rows"]),
            "brier": float(values["brier_sum"]) / int(values["rows"]),
        }
        for child, values in aggregates.items()
    }
    return {
        "schema": "anra-discovery-gym/mechanism-validation/v1",
        "dataset_sha256": dataset.source_sha256,
        "folds": folds,
        "by_child": by_child,
        "claim_boundary": "held-out discrete prediction diagnostic, not causal identification",
    }


def predictive_benchmark(
    dataset: TableDataset,
    *,
    max_parents: int = 2,
    seed: int = 0,
) -> dict[str, Any]:
    model_names = ("majority", "full-conditional", "graph-mechanism")
    totals = {
        model: {column: {"rows": 0, "correct": 0, "brier_sum": 0.0, "log_loss_sum": 0.0} for column in dataset.columns}
        for model in model_names
    }
    folds: list[dict[str, Any]] = []
    for index, (train, holdout_rows) in enumerate(_leave_one_group_out_folds(dataset, seed)):
        graph = discover_graph(train, max_parents=max_parents)
        models = {
            "majority": _fit_parents(
                train,
                {column: () for column in dataset.columns},
                content_hash({"model": "majority", "dataset_sha256": train.source_sha256}),
            ),
            "full-conditional": _fit_parents(
                train,
                {column: tuple(value for value in dataset.columns if value != column) for column in dataset.columns},
                content_hash({"model": "full-conditional", "dataset_sha256": train.source_sha256}),
            ),
            "graph-mechanism": fit_mechanism(train, graph),
        }
        fold: dict[str, Any] = {"fold": index, "graph_sha256": graph.graph_sha256, "models": {}}
        for model_name, model in models.items():
            model_metrics: dict[str, Any] = {}
            for child in dataset.columns:
                rows = [dict(zip(dataset.columns, row)) for row in holdout_rows]
                labels = [int(row[child]) for row in rows]
                probabilities = [min(max(model.predict_proba(child, row), 1e-12), 1.0 - 1e-12) for row in rows]
                correct = sum(model.predict(child, row) == label for row, label in zip(rows, labels))
                brier_sum = sum((probability - label) ** 2 for probability, label in zip(probabilities, labels))
                log_loss_sum = sum(
                    -(label * math.log(probability) + (1 - label) * math.log(1.0 - probability))
                    for probability, label in zip(probabilities, labels)
                )
                bucket = totals[model_name][child]
                bucket["rows"] += len(rows)
                bucket["correct"] += correct
                bucket["brier_sum"] += brier_sum
                bucket["log_loss_sum"] += log_loss_sum
                model_metrics[child] = {"accuracy": correct / len(rows), "brier": brier_sum / len(rows), "log_loss": log_loss_sum / len(rows)}
            fold["models"][model_name] = model_metrics
        folds.append(fold)
    by_model: dict[str, Any] = {}
    for model_name in model_names:
        by_child = {}
        for child, values in totals[model_name].items():
            rows = int(values["rows"])
            by_child[child] = {
                "rows": rows,
                "accuracy": float(values["correct"]) / rows,
                "brier": float(values["brier_sum"]) / rows,
                "log_loss": float(values["log_loss_sum"]) / rows,
            }
        by_model[model_name] = {
            "by_child": by_child,
            "mean_accuracy": sum(value["accuracy"] for value in by_child.values()) / len(by_child),
            "mean_brier": sum(value["brier"] for value in by_child.values()) / len(by_child),
            "mean_log_loss": sum(value["log_loss"] for value in by_child.values()) / len(by_child),
        }
    return {
        "schema": "anra-discovery-gym/predictive-benchmark/v1",
        "dataset_sha256": dataset.source_sha256,
        "folds": folds,
        "models": by_model,
        "winner_by_mean_accuracy": max(model_names, key=lambda name: by_model[name]["mean_accuracy"]),
        "claim_boundary": "grouped predictive comparison; it does not establish causality or autonomous discovery",
    }


def _column_cardinality(
    rows: Sequence[tuple[str, ...]],
    columns: Sequence[str],
    column: str,
) -> int:
    index = tuple(columns).index(column)
    return len({row[index] for row in rows})


def _conditional_score(
    rows: Sequence[tuple[str, ...]],
    columns: Sequence[str],
    target: str,
    parents: Sequence[str],
) -> float:
    column_indexes = {column: index for index, column in enumerate(columns)}
    target_index = column_indexes[target]
    target_values = {row[target_index] for row in rows}
    parent_indexes = [column_indexes[parent] for parent in parents]
    parent_values = {parent: _column_cardinality(rows, columns, parent) for parent in parents}
    groups: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    for row in rows:
        groups[tuple(row[index] for index in parent_indexes)][row[target_index]] += 1
    log_likelihood = 0.0
    for counts in groups.values():
        total = sum(counts.values())
        denominator = total + len(target_values)
        for count in counts.values():
            log_likelihood += count * math.log((count + 1.0) / denominator)
    parameter_count = max(1, (len(target_values) - 1) * max(1, math.prod(parent_values.values())))
    return log_likelihood - 0.5 * math.log(len(rows)) * parameter_count


def _parent_options(columns: Sequence[str], target: str, allowed: set[str], max_parents: int) -> list[tuple[str, ...]]:
    candidates = [column for column in columns if column != target and column in allowed]
    options: list[tuple[str, ...]] = []
    for size in range(min(max_parents, len(candidates)) + 1):
        options.extend(tuple(value) for value in itertools.combinations(candidates, size))
    return options


def _would_cycle(edges: set[tuple[str, str]], source: str, target: str) -> bool:
    pending = [source]
    while pending:
        current = pending.pop()
        if current == target:
            return True
        pending.extend(child for parent, child in edges if parent == current)
    return False


def _greedy_graph(dataset: TableDataset, max_parents: int) -> GraphHypothesis:
    columns = dataset.columns
    parents = {column: () for column in columns}
    rows = dataset.rows
    while True:
        best: tuple[float, str, str, tuple[str, ...]] | None = None
        edges = {(parent, child) for child, values in parents.items() for parent in values}
        for source in columns:
            for target in columns:
                if source == target or source in parents[target] or _would_cycle(edges, target, source):
                    continue
                proposed = tuple(sorted((*parents[target], source)))
                if len(proposed) > max_parents:
                    continue
                improvement = _conditional_score(rows, columns, target, proposed) - _conditional_score(rows, columns, target, parents[target])
                candidate = (improvement, source, target, proposed)
                if improvement > 1e-12 and (best is None or candidate[:3] > best[:3]):
                    best = candidate
        if best is None:
            break
        _, source, target, proposed = best
        parents[target] = proposed
    order = _topological_order(columns, parents)
    local_scores = {column: _conditional_score(rows, columns, column, parents[column]) for column in columns}
    return GraphHypothesis(
        dataset_sha256=dataset.source_sha256,
        columns=columns,
        order=order,
        parents=parents,
        local_scores=local_scores,
        score=sum(local_scores.values()),
        algorithm="greedy-bic/v1",
        max_parents=max_parents,
    )


def _topological_order(columns: tuple[str, ...], parents: Mapping[str, Sequence[str]]) -> tuple[str, ...]:
    remaining = set(columns)
    order: list[str] = []
    while remaining:
        ready = sorted(column for column in remaining if not (set(parents.get(column, ())) & remaining))
        if not ready:
            raise ValueError("parent graph contains a cycle")
        order.append(ready[0])
        remaining.remove(ready[0])
    return tuple(order)


def discover_graph(dataset: TableDataset, *, max_parents: int = 2, max_exact_columns: int = MAX_EXACT_COLUMNS) -> GraphHypothesis:
    if not 1 <= max_parents <= 4:
        raise ValueError("max_parents must be between 1 and 4")
    if len(dataset.columns) > max_exact_columns:
        return _greedy_graph(dataset, max_parents)
    columns = dataset.columns
    rows = dataset.rows
    states: dict[int, tuple[float, tuple[str, ...], dict[str, tuple[str, ...]]]] = {0: (0.0, (), {})}
    for _ in range(len(columns)):
        next_states = dict(states)
        for mask, (score, order, parents) in states.items():
            selected = {column for column in columns if mask & (1 << columns.index(column))}
            for column in columns:
                if column in selected:
                    continue
                options = _parent_options(columns, column, selected, max_parents)
                best_local: tuple[float, tuple[str, ...]] | None = None
                for option in options:
                    local = _conditional_score(rows, columns, column, option)
                    if best_local is None or local > best_local[0] + 1e-12 or (
                        abs(local - best_local[0]) <= 1e-12 and len(option) < len(best_local[1])
                    ):
                        best_local = (local, option)
                if best_local is None:
                    continue
                new_mask = mask | (1 << columns.index(column))
                candidate = (score + best_local[0], order + (column,), {**parents, column: best_local[1]})
                current = next_states.get(new_mask)
                if current is None or candidate[0] > current[0] + 1e-12:
                    next_states[new_mask] = candidate
        states = next_states
    full_mask = (1 << len(columns)) - 1
    if full_mask not in states:
        raise ValueError("could not construct an acyclic graph")
    score, order, parents = states[full_mask]
    normalized_parents = {column: tuple(parents.get(column, ())) for column in columns}
    local_scores = {column: _conditional_score(rows, columns, column, normalized_parents[column]) for column in columns}
    return GraphHypothesis(
        dataset_sha256=dataset.source_sha256,
        columns=columns,
        order=order,
        parents=normalized_parents,
        local_scores=local_scores,
        score=score,
        algorithm="exact-dynamic-bic/v1",
        max_parents=max_parents,
    )


def stability_report(
    dataset: TableDataset,
    graph: GraphHypothesis,
    *,
    replicates: int = 16,
    max_parents: int = 2,
    seed: int = 0,
) -> dict[str, Any]:
    if not 1 <= replicates <= 128:
        raise ValueError("replicates must be between 1 and 128")
    rng = random.Random(seed)
    counts: Counter[str] = Counter()
    edge_counts: list[int] = []
    for replicate in range(replicates):
        rows = tuple(dataset.rows[rng.randrange(len(dataset.rows))] for _ in dataset.rows)
        sample = TableDataset.from_rows(f"{dataset.name}:bootstrap:{replicate}", dataset.columns, rows)
        candidate = discover_graph(sample, max_parents=max_parents)
        edges = candidate.edges
        edge_counts.append(len(edges))
        counts.update(f"{source}->{target}" for source, target in edges)
    baseline_edges = {f"{source}->{target}" for source, target in graph.edges}
    support = {edge: counts.get(edge, 0) / replicates for edge in sorted(baseline_edges | set(counts))}
    stable = [edge for edge in baseline_edges if support.get(edge, 0.0) >= 0.75]
    return {
        "schema": "anra-discovery-gym/stability/v1",
        "replicates": replicates,
        "seed": seed,
        "edge_support": support,
        "baseline_edge_count": len(baseline_edges),
        "stable_edge_count": len(stable),
        "topology_stability": len(stable) / len(baseline_edges) if baseline_edges else 1.0,
        "mean_edge_count": sum(edge_counts) / replicates,
        "interpretation": "bootstrap stability of an observational graph hypothesis, not causal proof",
    }


@dataclass(frozen=True, slots=True)
class Recommendation:
    recommendation_id: str
    kind: str
    target: str
    values: tuple[str, ...]
    cost: int
    priority: float
    rationale: str
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in {"randomized_intervention", "replicate_measure"}:
            raise ValueError("unsupported workbench recommendation kind")
        if not self.recommendation_id or not self.target or not self.rationale:
            raise ValueError("recommendation identity, target, and rationale are required")
        if isinstance(self.cost, bool) or not isinstance(self.cost, int) or self.cost <= 0:
            raise ValueError("recommendation cost must be a positive integer")
        if not math.isfinite(self.priority):
            raise ValueError("recommendation priority must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommendation_id": self.recommendation_id,
            "kind": self.kind,
            "target": self.target,
            "values": list(self.values),
            "cost": self.cost,
            "priority": self.priority,
            "rationale": self.rationale,
            "evidence": dict(self.evidence),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Recommendation":
        return cls(
            recommendation_id=str(value["recommendation_id"]),
            kind=str(value["kind"]),
            target=str(value["target"]),
            values=tuple(value["values"]),
            cost=int(value["cost"]),
            priority=float(value["priority"]),
            rationale=str(value["rationale"]),
            evidence=dict(value.get("evidence", {})),
        )


@dataclass(frozen=True, slots=True)
class ExperimentPlan:
    plan_id: str
    dataset_sha256: str
    graph_sha256: str
    recommendations: tuple[Recommendation, ...]
    budget: int | None = None
    planned_cost: int = 0
    claim_boundary: str = "observational hypotheses require randomized or interventional confirmation"

    def __post_init__(self) -> None:
        if not self.plan_id or not self.dataset_sha256 or not self.graph_sha256:
            raise ValueError("plan identities must be nonempty")
        if self.budget is not None and (isinstance(self.budget, bool) or not isinstance(self.budget, int) or self.budget < 0):
            raise ValueError("plan budget must be a nonnegative integer")
        if self.planned_cost < 0 or (self.budget is not None and self.planned_cost > self.budget):
            raise ValueError("planned cost exceeds the declared budget")
        if len({item.recommendation_id for item in self.recommendations}) != len(self.recommendations):
            raise ValueError("plan recommendation identities must be unique")
        if any(item.cost <= 0 for item in self.recommendations):
            raise ValueError("plan recommendation costs must be positive")

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema": "anra-discovery-gym/experiment-plan/v1",
            "plan_id": self.plan_id,
            "dataset_sha256": self.dataset_sha256,
            "graph_sha256": self.graph_sha256,
            "recommendations": [item.to_dict() for item in self.recommendations],
            "budget": self.budget,
            "planned_cost": self.planned_cost,
            "claim_boundary": self.claim_boundary,
        }
        return {**body, "plan_sha256": content_hash(body)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExperimentPlan":
        body = dict(value)
        plan_hash = body.pop("plan_sha256", None)
        if not isinstance(plan_hash, str) or content_hash(body) != plan_hash:
            raise ValueError("experiment plan hash does not match contents")
        return cls(
            plan_id=str(body["plan_id"]),
            dataset_sha256=str(body["dataset_sha256"]),
            graph_sha256=str(body["graph_sha256"]),
            recommendations=tuple(Recommendation.from_dict(item) for item in body["recommendations"]),
            budget=None if body.get("budget") is None else int(body["budget"]),
            planned_cost=int(body.get("planned_cost", 0)),
            claim_boundary=str(body["claim_boundary"]),
        )


def graph_recovery_metrics(graph: GraphHypothesis, true_edges: Iterable[tuple[str, str]]) -> dict[str, Any]:
    predicted = set(graph.edges)
    truth = {tuple(edge) for edge in true_edges}
    if any(source not in graph.columns or target not in graph.columns for source, target in truth):
        raise ValueError("true edges must refer to graph columns")
    true_positive = len(predicted & truth)
    precision = true_positive / len(predicted) if predicted else (1.0 if not truth else 0.0)
    recall = true_positive / len(truth) if truth else 1.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "schema": "anra-discovery-gym/graph-recovery/v1",
        "predicted_edges": sorted(predicted),
        "true_edges": sorted(truth),
        "true_positive": true_positive,
        "false_positive": len(predicted - truth),
        "false_negative": len(truth - predicted),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "interpretation": "fixture recovery diagnostic only; observational edge orientation is not causal proof",
    }


def benchmark_graph_discovery(
    cases: Sequence[tuple[TableDataset, Iterable[tuple[str, str]]]],
    *,
    max_parents: int = 2,
    stability_replicates: int = 8,
    seed: int = 0,
) -> dict[str, Any]:
    if not cases:
        raise ValueError("benchmark requires at least one case")
    rows: list[dict[str, Any]] = []
    for index, (dataset, true_edges) in enumerate(cases):
        graph = discover_graph(dataset, max_parents=max_parents)
        recovery = graph_recovery_metrics(graph, true_edges)
        stability = stability_report(dataset, graph, replicates=stability_replicates, max_parents=max_parents, seed=seed + index)
        rows.append(
            {
                "case_id": f"case-{index}",
                "dataset_sha256": dataset.source_sha256,
                "graph_sha256": graph.graph_sha256,
                "recovery": recovery,
                "stability": stability,
            }
        )
    return {
        "schema": "anra-discovery-gym/workbench-benchmark/v1",
        "cases": rows,
        "mean_precision": sum(float(row["recovery"]["precision"]) for row in rows) / len(rows),
        "mean_recall": sum(float(row["recovery"]["recall"]) for row in rows) / len(rows),
        "mean_f1": sum(float(row["recovery"]["f1"]) for row in rows) / len(rows),
        "mean_topology_stability": sum(float(row["stability"]["topology_stability"]) for row in rows) / len(rows),
        "independence_unit": "fixture",
        "claim_boundary": "graph recovery benchmark, not evidence of causal truth or autonomous discovery",
    }


def recommend_experiments(
    dataset: TableDataset,
    graph: GraphHypothesis,
    *,
    stability: Mapping[str, Any] | None = None,
    limit: int = 6,
    seed: int = 0,
    budget: int | None = None,
) -> ExperimentPlan:
    if graph.dataset_sha256 != dataset.source_sha256:
        raise ValueError("graph and dataset identities do not match")
    if not 1 <= limit <= 32:
        raise ValueError("limit must be between 1 and 32")
    if budget is not None and (isinstance(budget, bool) or not isinstance(budget, int) or budget < 0):
        raise ValueError("budget must be a nonnegative integer")
    stability_values = dict(stability or stability_report(dataset, graph, seed=seed))
    support = {str(key): float(value) for key, value in stability_values.get("edge_support", {}).items()}
    entropy = {column: float(dataset.profile()["columns"][column]["entropy_bits"]) for column in dataset.columns}
    out_degree = Counter(child for _, child in graph.edges)
    candidates: list[Recommendation] = []
    for parent, child in sorted(graph.edges):
        edge = f"{parent}->{child}"
        edge_support = support.get(edge, 0.0)
        values = tuple(sorted(dataset.value_counts(parent)))
        priority = 1.5 * (1.0 - edge_support) + 0.5 * entropy[parent] + 0.2 * out_degree[parent]
        rationale = f"test whether {parent} changes {child}; bootstrap edge support={edge_support:.3f}"
        payload = {"edge": edge, "edge_support": edge_support, "child": child}
        recommendation_id = "rec-" + content_hash({"kind": "randomized_intervention", "target": parent, "values": list(values), "evidence": payload})[:16]
        candidates.append(Recommendation(recommendation_id, "randomized_intervention", parent, values, 3, priority, rationale, payload))
    for column in sorted(dataset.columns, key=lambda item: (-entropy[item], item)):
        if len(candidates) >= limit:
            break
        payload = {"entropy_bits": entropy[column], "out_degree": out_degree[column]}
        recommendation_id = "rec-" + content_hash({"kind": "replicate_measure", "target": column, "evidence": payload})[:16]
        candidates.append(
            Recommendation(
                recommendation_id,
                "replicate_measure",
                column,
                (),
                1,
                0.25 + entropy[column] / 10.0,
                f"increase independent measurements for high-entropy variable {column}",
                payload,
            )
        )
    candidates.sort(key=lambda item: (-item.priority, item.recommendation_id))
    if budget is None:
        selected = tuple(candidates[:limit])
        planned_cost = sum(item.cost for item in selected)
    else:
        selected_list: list[Recommendation] = []
        planned_cost = 0
        for candidate in candidates:
            if len(selected_list) >= limit:
                break
            if planned_cost + candidate.cost <= budget:
                selected_list.append(candidate)
                planned_cost += candidate.cost
        if not selected_list and candidates and budget >= min(item.cost for item in candidates):
            candidate = min(candidates, key=lambda item: (item.cost, -item.priority, item.recommendation_id))
            selected_list.append(candidate)
            planned_cost = candidate.cost
        selected = tuple(selected_list)
    plan_body = {
        "dataset_sha256": dataset.source_sha256,
        "graph_sha256": graph.graph_sha256,
        "recommendations": [item.to_dict() for item in selected],
        "budget": budget,
        "planned_cost": planned_cost,
    }
    plan_id = "plan-" + content_hash(plan_body)[:20]
    return ExperimentPlan(plan_id, dataset.source_sha256, graph.graph_sha256, selected, budget, planned_cost)


@dataclass(frozen=True, slots=True)
class EvidenceEvent:
    sequence: int
    kind: str
    payload: Mapping[str, Any]
    previous_sha256: str
    event_sha256: str

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("event sequence must be nonnegative")
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("event kind must be nonempty")
        if not isinstance(self.payload, Mapping):
            raise TypeError("event payload must be a mapping")
        if not isinstance(self.previous_sha256, str) or len(self.previous_sha256) != 64:
            raise ValueError("event predecessor hash must be a SHA-256 hex digest")
        if not isinstance(self.event_sha256, str) or len(self.event_sha256) != 64:
            raise ValueError("event hash must be a SHA-256 hex digest")
        body = {
            "sequence": self.sequence,
            "kind": self.kind,
            "payload": dict(self.payload),
            "previous_sha256": self.previous_sha256,
        }
        if content_hash(body) != self.event_sha256:
            raise ValueError("event hash does not match event contents")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "kind": self.kind,
            "payload": dict(self.payload),
            "previous_sha256": self.previous_sha256,
            "event_sha256": self.event_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceEvent":
        return cls(
            sequence=int(value["sequence"]),
            kind=str(value["kind"]),
            payload=dict(value["payload"]),
            previous_sha256=str(value["previous_sha256"]),
            event_sha256=str(value["event_sha256"]),
        )


class EvidenceLedger:
    def __init__(self, events: Iterable[EvidenceEvent] = ()) -> None:
        self._events = list(events)
        self.verify()

    @property
    def events(self) -> tuple[EvidenceEvent, ...]:
        return tuple(self._events)

    def append(self, kind: str, payload: Mapping[str, Any]) -> EvidenceEvent:
        body = {
            "sequence": len(self._events),
            "kind": kind,
            "payload": dict(payload),
            "previous_sha256": self._events[-1].event_sha256 if self._events else GENESIS_HASH,
        }
        event = EvidenceEvent(
            sequence=body["sequence"],
            kind=body["kind"],
            payload=body["payload"],
            previous_sha256=body["previous_sha256"],
            event_sha256=content_hash(body),
        )
        self._events.append(event)
        return event

    def verify(self) -> dict[str, Any]:
        previous = GENESIS_HASH
        for index, event in enumerate(self._events):
            if event.sequence != index or event.previous_sha256 != previous:
                return {"status": "FAIL", "failure": f"chain break at {index}"}
            previous = event.event_sha256
        return {"status": "PASS", "event_count": len(self._events), "head_sha256": previous}

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "anra-discovery-gym/evidence-ledger/v1", "events": [event.to_dict() for event in self._events]}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceLedger":
        return cls(EvidenceEvent.from_dict(item) for item in value.get("events", ()))


@dataclass(frozen=True, slots=True)
class DiscoverySession:
    session_id: str
    dataset: TableDataset
    graph: GraphHypothesis
    plan: ExperimentPlan
    ledger: EvidenceLedger

    def __post_init__(self) -> None:
        if self.graph.dataset_sha256 != self.dataset.source_sha256:
            raise ValueError("session graph and dataset identities do not match")
        if self.plan.dataset_sha256 != self.dataset.source_sha256 or self.plan.graph_sha256 != self.graph.graph_sha256:
            raise ValueError("session plan identities do not match")

    @classmethod
    def create(
        cls,
        dataset: TableDataset,
        graph: GraphHypothesis,
        plan: ExperimentPlan,
        *,
        session_id: str | None = None,
    ) -> "DiscoverySession":
        identity = session_id or "session-" + content_hash({"dataset": dataset.source_sha256, "graph": graph.graph_sha256, "plan": plan.plan_id})[:20]
        return cls(identity, dataset, graph, plan, EvidenceLedger())

    def record(self, kind: str, payload: Mapping[str, Any]) -> EvidenceEvent:
        return self.ledger.append(kind, payload)

    def record_external_result(
        self,
        recommendation_id: str,
        *,
        executor_id: str,
        assignment: str | None,
        outcome: Any,
        cost: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> EvidenceEvent:
        recommendation = next((item for item in self.plan.recommendations if item.recommendation_id == recommendation_id), None)
        if recommendation is None:
            raise KeyError(f"unknown recommendation: {recommendation_id}")
        if not isinstance(executor_id, str) or not executor_id:
            raise ValueError("executor identity must be nonempty")
        if recommendation.kind == "randomized_intervention":
            if assignment not in recommendation.values:
                raise ValueError("external assignment is not one of the planned values")
        if isinstance(cost, bool) or not isinstance(cost, int) or cost < recommendation.cost:
            raise ValueError("external result cost is below the planned cost")
        return self.record(
            "external_result",
            {
                "recommendation_id": recommendation_id,
                "executor_id": executor_id,
                "assignment": assignment,
                "outcome": outcome,
                "cost": cost,
                "metadata": dict(metadata or {}),
            },
        )

    def completed_recommendations(self) -> frozenset[str]:
        return frozenset(
            str(event.payload["recommendation_id"])
            for event in self.ledger.events
            if event.kind == "external_result" and "recommendation_id" in event.payload
        )

    def next_recommendation(self) -> Recommendation | None:
        completed = self.completed_recommendations()
        return next((item for item in self.plan.recommendations if item.recommendation_id not in completed), None)

    def replay(self) -> tuple[dict[str, Any], ...]:
        verification = self.ledger.verify()
        if verification["status"] != "PASS":
            raise ValueError(verification["failure"])
        return tuple(event.to_dict() for event in self.ledger.events)

    def report(self) -> dict[str, Any]:
        next_recommendation = self.next_recommendation()
        return {
            "schema": WORKBENCH_SCHEMA,
            "session_id": self.session_id,
            "dataset": self.dataset.profile(),
            "readiness": self.dataset.readiness(),
            "graph": self.graph.to_dict(),
            "plan": self.plan.to_dict(),
            "ledger": self.ledger.verify(),
            "observed_events": len(self.ledger.events),
            "completed_recommendations": len(self.completed_recommendations()),
            "next_recommendation": next_recommendation.to_dict() if next_recommendation else None,
            "next_step": "execute the highest-priority recommendation and append its observed result" if next_recommendation else "plan complete; review evidence and create a new frozen plan before further experiments",
            "claim_boundary": "the workbench produces hypotheses and plans; it does not turn observational associations or unverified interventions into causal facts",
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": WORKBENCH_SCHEMA,
            "session_id": self.session_id,
            "dataset": self.dataset.to_dict(),
            "graph": self.graph.to_dict(),
            "plan": self.plan.to_dict(),
            "ledger": self.ledger.to_dict(),
        }

    def save(self, path: Path) -> None:
        write_json_exclusive(path, self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DiscoverySession":
        if value.get("schema") != WORKBENCH_SCHEMA:
            raise ValueError("unsupported workbench session schema")
        return cls(
            session_id=str(value["session_id"]),
            dataset=TableDataset.from_dict(value["dataset"]),
            graph=GraphHypothesis.from_dict(value["graph"]),
            plan=ExperimentPlan.from_dict(value["plan"]),
            ledger=EvidenceLedger.from_dict(value["ledger"]),
        )

    @classmethod
    def load(cls, path: Path) -> "DiscoverySession":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def load_dataset(path: Path) -> TableDataset:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return TableDataset.from_csv(path)
    if suffix == ".json":
        return TableDataset.from_json(path)
    if suffix == ".jsonl":
        return TableDataset.from_jsonl(path)
    raise ValueError("workbench input must be a .csv, .json, or .jsonl file")


def qualification_receipt(
    dataset: TableDataset,
    *,
    protocol_id: str,
    checks: Mapping[str, bool],
    seed: int = 0,
) -> dict[str, Any]:
    if not protocol_id or not checks:
        raise ValueError("qualification requires a protocol identity and checks")
    body = {
        "schema": "anra-discovery-gym/dataset-qualification/v1",
        "status": "QUALIFIED" if all(bool(value) for value in checks.values()) else "UNQUALIFIED",
        "dataset_sha256": dataset.source_sha256,
        "protocol_id": protocol_id,
        "protocol_sha256": content_hash({"protocol_id": protocol_id, "seed": seed}),
        "checks": {str(key): bool(value) for key, value in checks.items()},
        "seed": seed,
    }
    return {**body, "qualification_sha256": content_hash(body)}


def inventory_readiness(
    receipts: Sequence[Mapping[str, Any]],
    *,
    qualification_protocol: str | None = None,
) -> dict[str, Any]:
    required = {"dataset_sha256", "namespace", "lineage_sha256", "tokenizer_sha256", "example_count", "qualification_status"}
    normalized: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, receipt in enumerate(receipts):
        missing = required - set(receipt)
        if missing:
            errors.append(f"receipt-{index}:missing-{','.join(sorted(missing))}")
            continue
        normalized.append(dict(receipt))
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for receipt in normalized:
        by_hash[str(receipt["dataset_sha256"])].append(receipt)
    duplicate_hashes = sorted(identity for identity, values in by_hash.items() if len(values) > 1)
    lineage_conflicts = []
    for identity, values in by_hash.items():
        signatures = {
            (
                str(value["namespace"]),
                str(value["lineage_sha256"]),
                str(value["tokenizer_sha256"]),
            )
            for value in values
        }
        if len(signatures) > 1:
            lineage_conflicts.append(identity)
    qualified_hashes = {
        identity
        for identity, values in by_hash.items()
        if any(
            value.get("qualification_status", value.get("status")) == "QUALIFIED"
            and (qualification_protocol is None or value.get("qualification_protocol", value.get("protocol_id")) == qualification_protocol)
            for value in values
        )
    }
    hard_blockers = list(errors)
    qualification_blockers = []
    if not normalized:
        hard_blockers.append("no dataset receipts supplied")
    if len(qualified_hashes) < len(by_hash):
        qualification_blockers.append("not every unique dataset has an exact qualification receipt")
    if lineage_conflicts:
        hard_blockers.append("overlapping receipts disagree on lineage or tokenizer identity")
    if duplicate_hashes:
        hard_blockers.append("duplicate dataset identities must be reconciled before counting supply")
    blockers = hard_blockers + qualification_blockers + ["external experiment executor is not connected"]
    warnings = [f"duplicate receipt identity: {identity}" for identity in duplicate_hashes] if not lineage_conflicts else []
    if hard_blockers:
        state = "BLOCKED"
    elif qualified_hashes:
        state = "QUALIFIED"
    else:
        state = "VERIFIED"
    return {
        "schema": "anra-discovery-gym/data-inventory/v1",
        "state": state,
        "receipt_count": len(receipts),
        "valid_receipt_count": len(normalized),
        "unique_dataset_count": len(by_hash),
        "qualified_unique_dataset_count": len(qualified_hashes),
        "example_count_sum": sum(int(value["example_count"]) for value in normalized),
        "duplicate_dataset_sha256": duplicate_hashes,
        "lineage_conflicts": lineage_conflicts,
        "blockers": blockers,
        "qualification_blockers": qualification_blockers,
        "warnings": warnings,
        "runnable": False,
        "next_state": "QUALIFIED" if state == "VERIFIED" else "BLOCKED" if state == "BLOCKED" else "RUNNABLE_CANDIDATE",
    }


def audit_workbench_report(report: Mapping[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    try:
        dataset_profile = report["dataset"]
        graph = GraphHypothesis.from_dict(report["graph"])
        plan = ExperimentPlan.from_dict(report["plan"])
        readiness = dict(report["readiness"])
        stability = dict(report["stability"])
        checks["identity_alignment"] = (
            dataset_profile["source_sha256"] == graph.dataset_sha256
            and plan.dataset_sha256 == graph.dataset_sha256
            and plan.graph_sha256 == graph.graph_sha256
        )
        checks["plan_integrity"] = len({item.recommendation_id for item in plan.recommendations}) == len(plan.recommendations)
        declared_columns = set(dataset_profile["columns"])
        checks["recommendation_targets_declared"] = all(
            item.target in declared_columns
            and (item.kind != "randomized_intervention" or set(item.values) <= set(dataset_profile["columns"][item.target]["counts"]))
            for item in plan.recommendations
        )
        checks["stability_integrity"] = all(0.0 <= float(value) <= 1.0 for value in stability.get("edge_support", {}).values())
        checks["readiness_is_explicit"] = readiness.get("state") == "VERIFIED" and readiness.get("runnable") is False
        checks["claim_boundary_present"] = bool(report.get("claim_boundary"))
    except (KeyError, TypeError, ValueError):
        checks["identity_alignment"] = False
        checks["plan_integrity"] = False
        checks["recommendation_targets_declared"] = False
        checks["stability_integrity"] = False
        checks["readiness_is_explicit"] = False
        checks["claim_boundary_present"] = False
    return {
        "schema": "anra-discovery-gym/workbench-audit/v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }


def build_workbench_report(
    dataset: TableDataset,
    *,
    max_parents: int = 2,
    stability_replicates: int = 8,
    seed: int = 0,
    budget: int | None = 8,
) -> dict[str, Any]:
    graph = discover_graph(dataset, max_parents=max_parents)
    stability = stability_report(dataset, graph, replicates=stability_replicates, max_parents=max_parents, seed=seed)
    plan = recommend_experiments(dataset, graph, stability=stability, seed=seed, budget=budget)
    report = {
        "schema": WORKBENCH_SCHEMA,
        "dataset": dataset.profile(),
        "readiness": dataset.readiness(),
        "graph": graph.to_dict(),
        "stability": stability,
        "plan": plan.to_dict(),
        "utility_status": "ready_for_external_observation_or_intervention",
        "claim_boundary": "observational graph hypotheses and experiment priorities are not causal conclusions",
    }
    report["audit"] = audit_workbench_report(report)
    return report
