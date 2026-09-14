"""VOCAB-PRESSURE-001 development-only mechanism diagnostic.

This is a prospective diagnostic adjunct to frozen FORMATION-MUX-001 Science S5.
It adds no training arm, never reads sealed rows, and cannot change the frozen
S5 verdict. It localizes endpoint extra-vocabulary competition versus
training-time formation damage using completed CS-MECH-002 checkpoints.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anra_v5 import formation_mux_model_v3 as fxm
from anra_v5 import formation_mux_train_v5 as train
from v5_experiments import formation_mux_protocol_v5 as proto
from v5_experiments.formation_mux_surface_v5 import load_public_surface

SCHEMA = "anra.formation-mux-vocab-pressure-diagnostic/v1"
DIAGNOSTIC = "VOCAB-PRESSURE-001"
SHARED_VOCAB = int(fxm.SHARED_VOCAB)
PHYSICAL_VOCAB = int(fxm.PHYSICAL_VOCAB)
MATERIAL_RESCUE_FLOOR = 0.05
FOCUS_ARM = "M2_EXTRA_FROZEN"


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(x) for x in values)
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def token_pressure_metrics(logits: Any, target: int, *, torch: Any) -> dict[str, float]:
    """Exact per-token denominator decomposition for a target in shared rows."""
    target = int(target)
    if target < 0 or target >= SHARED_VOCAB:
        raise RuntimeError(f"diagnostic target escaped shared vocabulary: {target}")
    if int(logits.numel()) != PHYSICAL_VOCAB:
        raise RuntimeError(
            f"diagnostic expected {PHYSICAL_VOCAB} logits, got {int(logits.numel())}"
        )
    x = logits.float()
    shared = x[:SHARED_VOCAB]
    extra = x[SHARED_VOCAB:]
    full_lse = torch.logsumexp(x, dim=0)
    shared_lse = torch.logsumexp(shared, dim=0)
    extra_lse = torch.logsumexp(extra, dim=0)
    target_logit = x[target]
    extra_mass = torch.exp(extra_lse - full_lse)
    tax = full_lse - shared_lse
    best_extra = torch.max(extra)
    rank_full = 1 + int(torch.sum(x > target_logit).item())
    rank_shared = 1 + int(torch.sum(shared > target_logit).item())
    return {
        "extra_probability_mass": float(extra_mass.item()),
        "denominator_tax_nats": float(tax.item()),
        "target_margin_vs_best_extra": float((target_logit - best_extra).item()),
        "target_rank_full": float(rank_full),
        "target_rank_shared": float(rank_shared),
        "extra_is_argmax": float(int(torch.argmax(x).item() >= SHARED_VOCAB)),
    }


def _next_logits(model: Any, prefix: list[int], *, torch: Any, device: Any) -> Any:
    from v5_model.core import packed_layout

    current = torch.tensor([prefix], dtype=torch.long, device=device)
    segment = torch.zeros_like(current)
    positions, mask = packed_layout(segment, torch_module=torch)
    return model(current, positions, mask)[0, -1].float()


def _greedy_exact(
    model: Any,
    rows: list[Mapping[str, Any]],
    arm: str,
    *,
    shared_only: bool,
    torch: Any,
    device: Any,
) -> float:
    was_training = bool(model.training)
    model.eval()
    complete = 0
    with torch.no_grad():
        for row in rows:
            prompt, answer = train._encode_row(row, proto.EXPERIMENT_A, arm)
            expected = [int(x) for x in answer if int(x) != 3]
            generated: list[int] = []
            stopped = False
            for _ in range(proto.MAX_GENERATION_TOKENS):
                logits = _next_logits(
                    model, prompt + generated, torch=torch, device=device
                )
                if shared_only:
                    logits = logits.clone()
                    logits[SHARED_VOCAB:] = float("-inf")
                nxt = int(torch.argmax(logits).item())
                if nxt == 3:
                    stopped = True
                    break
                if nxt == 0:
                    break
                generated.append(nxt)
            if generated == expected and stopped:
                complete += 1
    if was_training:
        model.train()
    return complete / max(len(rows), 1)


def _teacher_forced_pressure(
    model: Any,
    rows: list[Mapping[str, Any]],
    arm: str,
    *,
    torch: Any,
    device: Any,
) -> dict[str, float]:
    was_training = bool(model.training)
    model.eval()
    extra_mass: list[float] = []
    taxes: list[float] = []
    margins: list[float] = []
    full_ranks: list[float] = []
    shared_ranks: list[float] = []
    extra_argmax: list[float] = []
    with torch.no_grad():
        for row in rows:
            prompt, answer = train._encode_row(row, proto.EXPERIMENT_A, arm)
            prefix = list(prompt)
            for target in answer:
                logits = _next_logits(model, prefix, torch=torch, device=device)
                metrics = token_pressure_metrics(logits, int(target), torch=torch)
                extra_mass.append(metrics["extra_probability_mass"])
                taxes.append(metrics["denominator_tax_nats"])
                margins.append(metrics["target_margin_vs_best_extra"])
                full_ranks.append(metrics["target_rank_full"])
                shared_ranks.append(metrics["target_rank_shared"])
                extra_argmax.append(metrics["extra_is_argmax"])
                prefix.append(int(target))
    if was_training:
        model.train()
    n = max(len(taxes), 1)
    return {
        "teacher_forced_target_tokens": len(taxes),
        "mean_extra_probability_mass": sum(extra_mass) / n,
        "p95_extra_probability_mass": _percentile(extra_mass, 0.95),
        "mean_denominator_tax_nats": sum(taxes) / n,
        "p95_denominator_tax_nats": _percentile(taxes, 0.95),
        "mean_target_margin_vs_best_extra": sum(margins) / n,
        "mean_target_rank_full": sum(full_ranks) / n,
        "mean_target_rank_shared": sum(shared_ranks) / n,
        "mean_rank_improvement_shared_only": (
            sum(full_ranks) - sum(shared_ranks)
        ) / n,
        "extra_argmax_rate": sum(extra_argmax) / n,
    }


def _geometry(model: Any, *, torch: Any) -> dict[str, float]:
    with torch.no_grad():
        weight = model.embedding.weight.detach().float()
        shared_norm = torch.linalg.vector_norm(weight[:SHARED_VOCAB], dim=1)
        extra_norm = torch.linalg.vector_norm(weight[SHARED_VOCAB:], dim=1)
        shared_mean = float(shared_norm.mean().item())
        extra_mean = float(extra_norm.mean().item())
        return {
            "mean_shared_row_l2_norm": shared_mean,
            "mean_extra_row_l2_norm": extra_mean,
            "extra_to_shared_mean_norm_ratio": extra_mean / max(shared_mean, 1e-30),
            "max_extra_row_l2_norm": float(extra_norm.max().item()),
            "max_shared_row_l2_norm": float(shared_norm.max().item()),
        }


def classify_focus(per_seed: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    rescues = [
        float(row["counterfactual_rescue_exact_valid_eos"])
        for row in per_seed.values()
    ]
    if len(rescues) != len(proto.SEED_BUNDLES):
        raise RuntimeError("focus classification requires all four seed bundles")
    mean_rescue = sum(rescues) / len(rescues)
    positive = sum(x > 0.0 for x in rescues)
    if mean_rescue >= MATERIAL_RESCUE_FLOOR and positive >= 3:
        verdict = "MATERIAL_ENDPOINT_COMPETITION"
        meaning = (
            "Unused extra rows remain a material inference-time competitor at "
            "the endpoint. Future work may test output gating/partitioning, but "
            "this diagnostic does not promote such a treatment."
        )
    elif mean_rescue < MATERIAL_RESCUE_FLOOR:
        verdict = "SMALL_ENDPOINT_COMPETITION"
        meaning = (
            "Endpoint masking does not recover a material amount of exact "
            "performance. If frozen S5 shows an M2/M3 effect, that pattern is "
            "more consistent with training-time denominator/formation damage."
        )
    else:
        verdict = "MIXED_ENDPOINT_COMPETITION"
        meaning = (
            "Endpoint competition is seed-sensitive or borderline; do not "
            "change production output space from this diagnostic."
        )
    return {
        "focus_arm": FOCUS_ARM,
        "mean_counterfactual_rescue_exact_valid_eos": mean_rescue,
        "positive_rescue_seed_bundles": positive,
        "material_rescue_floor": MATERIAL_RESCUE_FLOOR,
        "classification": verdict,
        "meaning": meaning,
    }


def run_diagnostic(
    *,
    public_path: Path,
    out: Path,
    torch: Any,
    device: Any | None = None,
) -> dict[str, Any]:
    public = load_public_surface(public_path)
    if "sealed" in public.get("splits", {}):
        raise RuntimeError("SEALED_FIREWALL_BREACH: diagnostic received sealed rows")
    identity_rows = [
        row for row in public["splits"]["development"]
        if row.get("family") == "identity"
    ]
    if len(identity_rows) != 80:
        raise RuntimeError(
            f"VOCAB-PRESSURE expected 80 identity dev rows, got {len(identity_rows)}"
        )

    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    arms: dict[str, Any] = {}
    for arm in proto.ARMS_A:
        per_seed: dict[str, Any] = {}
        for bundle in proto.SEED_BUNDLES:
            label = f"S{proto.SEED_BUNDLES.index(bundle) + 1}"
            checkpoint = out / proto.EXPERIMENT_A / arm / label / "resume.pt"
            if not checkpoint.exists():
                raise RuntimeError(f"VOCAB-PRESSURE missing checkpoint: {checkpoint}")
            model = train.load_model_for_evaluation(
                checkpoint,
                experiment=proto.EXPERIMENT_A,
                arm=arm,
                seed_bundle=bundle,
                data_manifest_sha256=str(public["sha256"]),
                torch=torch,
                device=device,
            )
            full_exact = _greedy_exact(
                model, identity_rows, arm,
                shared_only=False, torch=torch, device=device,
            )
            shared_exact = _greedy_exact(
                model, identity_rows, arm,
                shared_only=True, torch=torch, device=device,
            )
            if shared_exact + 1e-12 < full_exact:
                raise RuntimeError(
                    "shared-only counterfactual unexpectedly reduced exact performance"
                )
            pressure = _teacher_forced_pressure(
                model, identity_rows, arm, torch=torch, device=device
            )
            per_seed[str(bundle)] = {
                "full_vocab_identity_exact_valid_eos": full_exact,
                "shared_only_counterfactual_identity_exact_valid_eos": shared_exact,
                "counterfactual_rescue_exact_valid_eos": shared_exact - full_exact,
                **pressure,
                **_geometry(model, torch=torch),
            }
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        keys = [
            "full_vocab_identity_exact_valid_eos",
            "shared_only_counterfactual_identity_exact_valid_eos",
            "counterfactual_rescue_exact_valid_eos",
            "mean_extra_probability_mass",
            "mean_denominator_tax_nats",
            "mean_target_margin_vs_best_extra",
            "mean_target_rank_full",
            "mean_target_rank_shared",
            "extra_argmax_rate",
            "extra_to_shared_mean_norm_ratio",
        ]
        means = {
            key: sum(float(row[key]) for row in per_seed.values()) / len(per_seed)
            for key in keys
        }
        arms[arm] = {"per_seed": per_seed, "mean_across_seeds": means}

    focus = classify_focus(arms[FOCUS_ARM]["per_seed"])
    result = {
        "schema": SCHEMA,
        "campaign": "FORMATION-MUX-001",
        "diagnostic": DIAGNOSTIC,
        "experiment": proto.EXPERIMENT_A,
        "science_commit": "c15ad8beb409537db42d075684ea54847a074ebd",
        "surface_sha256": str(public["sha256"]),
        "status": "COMPLETE",
        "diagnostic_only": True,
        "can_change_frozen_s5_verdict": False,
        "sealed_rows_used": False,
        "development_family": "identity",
        "development_rows": len(identity_rows),
        "physical_vocabulary": PHYSICAL_VOCAB,
        "shared_vocabulary": SHARED_VOCAB,
        "arms": arms,
        "focus_interpretation": focus,
        "architecture_targets": [
            "D7 vocabulary field",
            "D8 output-head field",
            "DECIDE-OUTPUT-SPACE",
        ],
        "claim_ceiling": (
            "Development-only mechanism localization. No production vocabulary, "
            "tokenizer, scale, cognition, or AGI authorization; frozen S5 "
            "primary and sealed verdicts remain authoritative."
        ),
    }
    path = out / proto.EXPERIMENT_A / "VOCAB_PRESSURE_DIAGNOSTIC.json"
    _atomic_json(path, result)
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--surface", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args(argv)
    import torch

    result = run_diagnostic(
        public_path=args.surface,
        out=args.out,
        torch=torch,
        device=torch.device(args.device),
    )
    print(
        "VOCAB-PRESSURE-001 "
        + json.dumps(
            {
                "status": result["status"],
                "focus": result["focus_interpretation"],
                "sealed_rows_used": result["sealed_rows_used"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
