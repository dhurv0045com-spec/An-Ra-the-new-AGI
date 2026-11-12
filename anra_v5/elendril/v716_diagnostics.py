"""V7.16 frozen-model mechanism diagnostics.

Three CPU-side instruments over frozen checkpoints and their evaluation
reports: a trace-diff auditor that pinpoints the first divergent decision
between paired conditions, deterministic linear probes with a clearly labeled
proxy mutual-information estimate for I(H_L; resolved value) versus
I(H_L; action legality), and aggregation helpers for the preregistered
E01-E12 evaluation family. No instrument reads evaluator truth into the model
path, and none claims exact mutual information.
"""
import math

import torch

from .codec import digest

TRACE_DIFF_SCHEMA = "elendril-v716-trace-diff-v1"
PROBE_SCHEMA = "elendril-v716-linear-probe-v1"
AGGREGATION_SCHEMA = "elendril-v716-e-family-aggregation-v1"
PROXY_STATUS = "PROXY_ESTIMATE_NOT_EXACT_MI"


def first_divergence(actions_a, actions_b):
    """First decision index where two recorded action traces disagree.

    Traces may end at different lengths; a missing decision counts as
    divergence at that step.
    """
    actions_a = tuple(actions_a)
    actions_b = tuple(actions_b)
    if not actions_a and not actions_b:
        return None
    for step in range(max(len(actions_a), len(actions_b))):
        left = actions_a[step] if step < len(actions_a) else "<ended>"
        right = actions_b[step] if step < len(actions_b) else "<ended>"
        if left != right:
            return {"step": step, "a": left, "b": right}
    return None


def trace_diff_report(pairs):
    """Diff many labeled pairs; pairs are (label_a, actions_a, label_b, actions_b)."""
    reports = []
    for label_a, actions_a, label_b, actions_b in pairs:
        divergence = first_divergence(actions_a, actions_b)
        reports.append({
            "schema": TRACE_DIFF_SCHEMA,
            "label_a": str(label_a), "label_b": str(label_b),
            "divergence": divergence,
            "identical": divergence is None,
            "steps_compared": max(len(tuple(actions_a)), len(tuple(actions_b))),
        })
    return reports


class LinearProbe:
    """Deterministic multinomial logistic probe over final hidden states.

    Full-batch gradient descent from a zero initialization, L2-regularized, on
    CPU tensors. No shuffling, no dropout, no hidden layers: exactly what the
    E09 contract describes, and cheap enough for frozen-model audits.
    """

    def __init__(self, *, classes, l2=1e-2, steps=400, learning_rate=0.5):
        classes = tuple(classes)
        if not classes or len(set(classes)) != len(classes):
            raise ValueError("probe requires unique nonempty class labels")
        if type(steps) is not int or not 1 <= steps <= 20_000:
            raise ValueError("probe steps must be a bounded positive integer")
        if type(l2) not in (int, float) or not math.isfinite(l2) or l2 < 0:
            raise ValueError("L2 penalty must be finite and nonnegative")
        if type(learning_rate) not in (int, float) or not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("probe learning rate must be positive and finite")
        self.classes = classes
        self.l2 = float(l2)
        self.steps = steps
        self.learning_rate = float(learning_rate)
        self.weights = None

    def _tensors(self, features, labels=None):
        features = torch.tensor([list(row) for row in features], dtype=torch.float32)
        if features.ndim != 2 or not features.shape[0]:
            raise ValueError("probe features must be a nonempty [rows, width] table")
        if not torch.isfinite(features).all():
            raise FloatingPointError("nonfinite probe features")
        if labels is None:
            return features, None
        labels = tuple(labels)
        if len(labels) != features.shape[0] or any(label not in self.classes for label in labels):
            raise ValueError("probe labels must match rows and registered classes")
        targets = torch.tensor([self.classes.index(label) for label in labels])
        return features, targets

    def fit(self, features, labels):
        features, targets = self._tensors(features, labels)
        width = features.shape[1]
        weights = torch.zeros(len(self.classes), width)
        bias = torch.zeros(len(self.classes))
        for _ in range(self.steps):
            logits = features @ weights.T + bias
            probabilities = torch.softmax(logits, dim=-1)
            probabilities[range(features.shape[0]), targets] -= 1.
            gradient = probabilities.T @ features / features.shape[0] + self.l2 * weights
            bias_gradient = probabilities.sum(dim=0) / features.shape[0]
            weights = weights - self.learning_rate * gradient
            bias = bias - self.learning_rate * bias_gradient
        self.weights, self.bias = weights, bias
        return self

    def scores(self, features):
        if self.weights is None:
            raise ValueError("probe must be fitted before scoring")
        features, _ = self._tensors(features)
        return features @ self.weights.T + self.bias

    def predict(self, features):
        return [self.classes[int(index)] for index in self.scores(features).argmax(dim=-1)]

    def accuracy(self, features, labels):
        predictions = self.predict(features)
        return sum(prediction == label for prediction, label in zip(predictions, labels)) / len(labels)


def _entropy(counts):
    total = sum(counts.values())
    if not total:
        return 0.
    return -sum((value / total) * math.log2(value / total)
                for value in counts.values() if value)


def proxy_mutual_information(features, labels, *, classes, split=0.5, l2=1e-2, steps=400):
    """Held-out proxy estimate of I(Y; probe prediction), explicitly not exact MI.

    The probe is fit on one deterministic half and evaluated on the other;
    the reported quantity is the empirical plug-in estimate
    H(Y) - H(Y | prediction) on the held-out half. Estimation noise can push
    the value slightly negative; the raw value is reported with its status so
    nothing is silently clamped.
    """
    features = [list(row) for row in features]
    labels = tuple(labels)
    if len(features) != len(labels) or not features:
        raise ValueError("proxy MI requires aligned nonempty features and labels")
    if not 0. < split < 1.:
        raise ValueError("split fraction must be in (0, 1)")
    order = list(range(len(features)))
    order.sort(key=lambda index: digest([index, labels[index], "v716-probe-split"]))
    cut = max(1, min(len(features) - 1, int(split * len(features))))
    train_indices, test_indices = order[:cut], order[cut:]
    train_features = [features[index] for index in train_indices]
    train_labels = [labels[index] for index in train_indices]
    test_features = [features[index] for index in test_indices]
    test_labels = [labels[index] for index in test_indices]
    probe = LinearProbe(classes=classes, l2=l2, steps=steps).fit(train_features, train_labels)
    predictions = probe.predict(test_features)
    outcome_counts = {}
    for label in test_labels:
        outcome_counts[label] = outcome_counts.get(label, 0) + 1
    joint = {}
    for prediction, label in zip(predictions, test_labels):
        bucket = joint.setdefault(prediction, {})
        bucket[label] = bucket.get(label, 0) + 1
    h_y = _entropy(outcome_counts)
    h_y_given_prediction = sum(
        (sum(bucket.values()) / len(test_labels)) * _entropy(bucket)
        for bucket in joint.values())
    return {"schema": PROBE_SCHEMA, "status": PROXY_STATUS,
            "n_train": len(train_labels), "n_test": len(test_labels),
            "classes": list(classes), "heldout_accuracy": probe.accuracy(test_features, test_labels),
            "h_y_bits": h_y, "h_y_given_prediction_bits": h_y_given_prediction,
            "mi_bits_proxy": h_y - h_y_given_prediction}


def condition_metrics(report):
    """Extract the standard metric group from one evaluation report, or None."""
    if not isinstance(report, dict):
        raise ValueError("reports are dictionaries")
    evaluation = report.get("evaluation", report)
    metrics = evaluation.get("metrics") if isinstance(evaluation, dict) else None
    if not isinstance(metrics, dict):
        return None
    keys = ("episodes", "correct", "accuracy", "nonnull_accuracy", "invalid",
            "verified_queries", "query_precision", "query_recall",
            "utility_numerator", "utility_denominator")
    return {key: metrics[key] for key in keys if key in metrics}


def summarize_conditions(reports):
    if not isinstance(reports, dict) or not reports:
        raise ValueError("condition summaries require a report mapping")
    return {key: condition_metrics(report) for key, report in sorted(reports.items())}


def row_reversal_delta(metrics_original, metrics_reversed):
    """E04 delta: reversed-minus-canonical accuracy plus the recorded baselines."""
    if not metrics_original or not metrics_reversed:
        return {"status": "NOT_RUN"}
    delta = metrics_reversed["accuracy"] - metrics_original["accuracy"]
    return {"schema": AGGREGATION_SCHEMA, "accuracy_original": metrics_original["accuracy"],
            "accuracy_reversed": metrics_reversed["accuracy"], "delta": delta,
            "falsified_row_reversal": delta < -0.25 * metrics_original["accuracy"]}


def hop_accuracy(reports, *, prefix="anti/hop-", hops=(1, 2, 3, 4)):
    """E05 anti-shortcut hop accuracy; missing conditions are recorded, not dropped."""
    result = {}
    for hop in hops:
        metrics = condition_metrics(reports.get(f"{prefix}{hop}", {}))
        result[str(hop)] = None if metrics is None else metrics["accuracy"]
    return {"schema": AGGREGATION_SCHEMA, "hops": result}


def stale_memory_score(reports, *, key="memory/stale-corrected"):
    """E03 stale-memory correction score against the 28/48 blocked-CE record."""
    metrics = condition_metrics(reports.get(key, {}))
    if metrics is None:
        return {"status": "NOT_RUN", "baseline": "28/48"}
    return {"schema": AGGREGATION_SCHEMA, "correct": metrics["correct"],
            "episodes": metrics["episodes"], "baseline": "28/48",
            "no_regression": metrics["correct"] >= 28}


def invalid_actions_total(reports, *, decoder_suffixes=("raw", "grammar")):
    """E02 invalid action counts split by decoder across all supplied reports."""
    totals = {suffix: 0 for suffix in decoder_suffixes}
    seen = {suffix: 0 for suffix in decoder_suffixes}
    for key, report in reports.items():
        metrics = condition_metrics(report)
        if metrics is None or "invalid" not in metrics:
            continue
        for suffix in decoder_suffixes:
            if key.endswith(f"|{suffix}") or key.endswith(f"/{suffix}") or key.endswith(f"-{suffix}"):
                totals[suffix] += metrics["invalid"]
                seen[suffix] += 1
    return {"schema": AGGREGATION_SCHEMA, "invalid": totals, "reports_counted": seen,
            "recorded_reference": "861 raw across 6,720 records in V7.15 S8"}
