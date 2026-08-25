"""Canonical trainer for self-model policy v3 (leakage-safe, deterministic).

Given the MIXED-CAUSAL-v1 full intervention-outcome matrix, a fixed seed,
a fixed feature schema, and fixed action costs, this script deterministically
regenerates output/self_model_v3.json.

Feature schema v3.1 (12 observed-only features; NO gold/family/verifier):
  0  n_candidates
  1  output_arity
  2  format_code           {prose:0, table:1, list:2}
  3  raw_top2_margin
  4  norm_top2_margin
  5  raw_spread_std
  6  norm_spread_std
  7  raw_norm_same_pick    (pick agreement)
  8  free_matches_raw_pick
  9  free_matches_norm_pick
 10  normalization_applicable (bool)
 11  max_query_lift_gap    (target-INDEPENDENT query sensitivity:
                            max_i normalized_i − second-best normalized_i)

Actions: NO_CHANGE, CONSTRAINED, NORMALIZED  (NORM_EXACT removed as
dominated — identical output to NORMALIZED at higher cost).

Utility rule: argmax_a P(success | state, a) − λ·cost(a), λ = 0.25.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

FEATURE_NAMES = [
    "n_candidates", "output_arity", "format_code",
    "raw_top2_margin", "norm_top2_margin",
    "raw_spread_std", "norm_spread_std",
    "raw_norm_same_pick", "free_matches_raw_pick", "free_matches_norm_pick",
    "normalization_applicable", "max_query_lift_gap",
]
FORMAT_CODES = {"prose": 0.0, "table": 1.0, "list": 2.0}
ACTIONS = ["NO_CHANGE", "CONSTRAINED", "NORMALIZED"]
COSTS = {"NO_CHANGE": 0, "CONSTRAINED": 1, "NORMALIZED": 2}
LAMBDA = 0.25


def observed_features(row: dict) -> list[float]:
    """Build features from EXPLICIT observed fields only.

    Reads: observed.{n_candidates, output_arity, format_name},
    raw_pick_code/norm_pick_code/free_out_code, raw_scores/norm_scores.
    Never reads gold_code/family_analysis/*_ok/*_rank_of_gold.
    """
    o = row["observed"]
    raw = row.get("raw_scores") or []
    norm = row.get("norm_scores") or []

    def top2(s):
        s = sorted(s)
        return s[-1] - s[-2] if len(s) >= 2 else 0.0

    def std(s):
        if not s:
            return 0.0
        m = sum(s) / len(s)
        return (sum((x - m) ** 2 for x in s) / len(s)) ** 0.5

    # target-independent query sensitivity: gap between best and second-best
    # normalized score (how sharply ONE candidate's query-conditioned lift
    # dominates — no reference to which is gold or to any target index)
    ns = sorted(norm, reverse=True)
    lift_gap = ns[0] - ns[1] if len(ns) >= 2 else 0.0

    free_code = row.get("free_out_code")
    same_pick = bool(raw and row.get("raw_pick_code") == row.get("norm_pick_code"))
    return [
        float(o["n_candidates"]),
        float(o["output_arity"]),
        FORMAT_CODES.get(o["format_name"], 3.0),
        top2(raw), top2(norm), std(raw), std(norm),
        float(same_pick),
        float(free_code is not None and free_code == row.get("raw_pick_code")),
        float(free_code is not None and free_code == row.get("norm_pick_code")),
        float(bool(raw)),
        float(lift_gap),
    ]


def extract_training_data(matrix_path: Path):
    r = json.loads(matrix_path.read_text(encoding="utf-8"))
    X_by_action = {a: [] for a in ACTIONS}
    y_by_action = {a: [] for a in ACTIONS}
    for row in r["per_task_rows"]:
        f = observed_features(row)
        for a in ACTIONS:
            if a in row["actions"]:
                X_by_action[a].append(f)
                y_by_action[a].append(
                    1 if row["actions"][a]["verifier_pass"] else 0)
    return X_by_action, y_by_action


def train(X_by_action, y_by_action, seed: int = 31, epochs: int = 400,
          lr: float = 0.05) -> dict:
    torch.manual_seed(seed)
    models = {}
    for act in ACTIONS:
        X = torch.tensor(X_by_action[act], dtype=torch.float32)
        y = torch.tensor(y_by_action[act], dtype=torch.float32).unsqueeze(1)
        w = torch.zeros(X.shape[1], requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=lr)
        final_loss = None
        for _ in range(epochs):
            opt.zero_grad()
            p = torch.sigmoid(X @ w.unsqueeze(1) + b)
            loss = torch.nn.functional.binary_cross_entropy(p, y)
            loss.backward()
            opt.step()
            final_loss = loss.item()
        models[act] = {
            "weights": [round(x, 6) for x in w.tolist()],
            "bias": round(b.item(), 6),
            "train_examples": len(X),
            "train_positives": int(sum(y_by_action[act])),
            "final_train_loss": round(final_loss, 6),
        }
    return models


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", default="output/mixed_causal_matrix_v2.json")
    ap.add_argument("--out", default="output/self_model_v3.json")
    ap.add_argument("--seed", type=int, default=31)
    args = ap.parse_args()

    matrix = Path(args.matrix)
    X_by_action, y_by_action = extract_training_data(matrix)
    models = train(X_by_action, y_by_action, seed=args.seed)

    total = sum(m["train_examples"] for m in models.values())
    receipt = {
        "schema": "anra-self-model-v3/v2",
        "canonical_trainer": "scripts/train_self_model_v3.py",
        "training_matrix": str(args.matrix),
        "matrix_sha256": hashlib.sha256(
            matrix.read_bytes()).hexdigest(),
        "seed": args.seed,
        "epochs": 400,
        "optimizer": "Adam(lr=0.05)",
        "solver": "logistic regression per action, BCE loss",
        "regularization": "none",
        "feature_names": FEATURE_NAMES,
        "actions": ACTIONS,
        "costs": COSTS,
        "lambda": LAMBDA,
        "thresholds": {a: 0.5 for a in ACTIONS},
        "utility_rule": ("argmax_a P(success|state,a) - lambda*cost(a); "
                         "ties broken by lower cost"),
        "total_training_examples": total,
        "models": models,
    }
    blob = json.dumps(receipt, sort_keys=True).encode("utf-8")
    receipt["parameter_sha256"] = hashlib.sha256(blob).hexdigest()
    Path(args.out).write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"policy written to {args.out}")
    print(f"parameter_sha256: {receipt['parameter_sha256']}")
    print(f"seed={args.seed} examples={total} actions={ACTIONS}")


if __name__ == "__main__":
    main()
