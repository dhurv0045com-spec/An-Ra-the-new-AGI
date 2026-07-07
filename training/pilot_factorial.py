"""The pre-registered pilot factorial (MASTER_UPGRADE Part 1, Stream A).

Converts the pilot-science factorial — optimizer/MoE/MTP/QK-Norm/SWA/V4 cells
on the 50M→150M ladder, curriculum-order cells, and the pilot-gated moonshots
M1/M3/M5 — into signed launch manifests, three seeds each. Every cell's
predicted outcome is registered in the forecast ledger *before* its manifest
is built, and every manifest passes the pre-launch timestamp audit before it
is returned. Predictions use the honest literature ranges from the plan
(Muon 1.3–1.6×, upcycled-MoE 1.5–2.5×), not the optimistic ones.

Law 1: pilot cells train from scratch on their own branch; no manifest here
references the earned frontier checkpoint as a mutable source.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

from anra.anra_paths import ROOT

from training.forecast_ledger import (
    FORECAST_LEDGER,
    audit_pre_launch,
    register_forecast,
)
from training.launch_manifest import build_launch_manifest, sign_manifest
from training.v2_runtime import active_tokenizer_path

PILOT_SEEDS = (1301, 2602, 3903)
PILOT_SCHEDULE = {
    "kind": "cosine_with_warmup",
    "warmup_fraction": 0.02,
    "min_lr": 1e-5,
}
PILOT_DATA_MANIFESTS = ["output/v2/data_manifests/tokenizer_v3.json"]
EXPECTED_TOKENS_BY_SCALE = {"50m": 500_000_000, "150m": 1_500_000_000}


@dataclass(frozen=True)
class PilotCell:
    cell_id: str
    scale: str  # "50m" | "150m"
    axes: dict[str, str]
    metric: str
    predicted_low: float
    predicted_high: float
    gate: str
    rationale: str
    optimizer: str = "adamw"
    moonshot: bool = False
    blocked_on: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)


def _cell(
    cell_id: str,
    scale: str,
    axes: dict[str, str],
    metric: str,
    low: float,
    high: float,
    gate: str,
    rationale: str,
    **extra: object,
) -> PilotCell:
    optimizer = "muon" if axes.get("optimizer") == "muon" else "adamw"
    return PilotCell(
        cell_id=cell_id,
        scale=scale,
        axes=axes,
        metric=metric,
        predicted_low=low,
        predicted_high=high,
        gate=gate,
        rationale=rationale,
        optimizer=optimizer,
        **extra,  # type: ignore[arg-type]
    )


PILOT_FACTORIAL: tuple[PilotCell, ...] = (
    # --- 150M primary factorial -------------------------------------------
    _cell(
        "p150-baseline",
        "150m",
        {"optimizer": "adamw", "moe": "off", "mtp": "off", "tokenizer": "v3"},
        "token_efficiency_x",
        0.95,
        1.05,
        "Anchors the local scaling law; loss within ±5% of predicted curve.",
        "Dense AdamW reference cell every multiplier is measured against.",
    ),
    _cell(
        "p150-muon",
        "150m",
        {"optimizer": "muon", "moe": "off", "mtp": "off", "tokenizer": "v3"},
        "token_efficiency_x",
        1.3,
        1.6,
        "Adopt if >=1.2x matched-token efficiency vs p150-baseline, 3 seeds.",
        "Honest literature range for Muon at this scale (not v2's 2x claim).",
    ),
    _cell(
        "p150-moe",
        "150m",
        {"optimizer": "adamw", "moe": "upcycle-8r1s-top2", "mtp": "off", "tokenizer": "v3"},
        "capability_per_active_flop_x",
        1.5,
        2.5,
        "Adopt if >=1.3x capability per active FLOP at matched compute.",
        "Sparse upcycling 8 routed + 1 shared, top-2, aux-loss-free.",
    ),
    _cell(
        "p150-mtp",
        "150m",
        {"optimizer": "adamw", "moe": "off", "mtp": "on", "tokenizer": "v3"},
        "token_efficiency_x",
        1.05,
        1.2,
        "Adopt if >=1.05x and the head later parity-gates speculative decode.",
        "Multi-token-prediction head; modest training gain, serving upside.",
    ),
    _cell(
        "p150-qknorm-off",
        "150m",
        {"optimizer": "adamw", "moe": "off", "mtp": "off", "qk_norm": "off", "tokenizer": "v3"},
        "val_loss_delta_pct",
        0.0,
        3.0,
        "Keep QK-Norm if removing it is >=1% worse or unstable at high LR.",
        "Ablation cell: measures what QK-Norm actually buys us.",
    ),
    _cell(
        "p150-swa-full",
        "150m",
        {
            "optimizer": "adamw",
            "moe": "off",
            "mtp": "off",
            "attention": "full-only",
            "tokenizer": "v3",
        },
        "quality_ratio_swa_vs_full",
        0.98,
        1.01,
        "Keep the 3:1 SWA hybrid if quality >=0.98x full attention.",
        "Ablation cell: confirms the SWA hybrid costs nothing it shouldn't.",
    ),
    _cell(
        "p150-v4tok",
        "150m",
        {"optimizer": "adamw", "moe": "off", "mtp": "off", "tokenizer": "v4-campaign-candidate"},
        "effective_compute_x",
        1.3,
        1.9,
        "Adopt V4 if fertility win converts to >=1.3x effective compute.",
        "Fertility-to-compute conversion pilot; candidate comes from the "
        ">=50MB campaign corpus (Stream B).",
        blocked_on="stream-b-canonical-v4",
    ),
    _cell(
        "p150-muon-moe",
        "150m",
        {"optimizer": "muon", "moe": "upcycle-8r1s-top2", "mtp": "off", "tokenizer": "v3"},
        "token_efficiency_x",
        1.9,
        3.5,
        "Interaction >= 0.9x the product of solo gains, else effects overlap.",
        "First-order interaction: do the two big multipliers compose?",
    ),
    _cell(
        "p150-muon-mtp",
        "150m",
        {"optimizer": "muon", "moe": "off", "mtp": "on", "tokenizer": "v3"},
        "token_efficiency_x",
        1.4,
        1.9,
        "Interaction >= 0.9x the product of solo gains.",
        "Optimizer x MTP interaction cell.",
    ),
    _cell(
        "p150-moe-mtp",
        "150m",
        {"optimizer": "adamw", "moe": "upcycle-8r1s-top2", "mtp": "on", "tokenizer": "v3"},
        "capability_per_active_flop_x",
        1.6,
        2.8,
        "Interaction >= 0.9x the product of solo gains.",
        "MoE x MTP interaction cell.",
    ),
    _cell(
        "p150-muon-moe-mtp",
        "150m",
        {"optimizer": "muon", "moe": "upcycle-8r1s-top2", "mtp": "on", "tokenizer": "v3"},
        "token_efficiency_x",
        2.1,
        4.0,
        "Stack must beat the best pair by >=10% or the third axis is dropped.",
        "Three-way stack; candidate campaign configuration if it holds.",
    ),
    _cell(
        "p150-fullstack",
        "150m",
        {
            "optimizer": "muon",
            "moe": "upcycle-8r1s-top2",
            "mtp": "on",
            "tokenizer": "v4-campaign-candidate",
        },
        "effective_compute_x",
        2.7,
        6.0,
        "The winning configuration must reproduce across all 3 seeds within 5%.",
        "Everything-on cell: the campaign-config candidate.",
        blocked_on="stream-b-canonical-v4",
    ),
    # --- 50M ladder replicas (scaling-law anchors) ------------------------
    _cell(
        "p050-baseline",
        "50m",
        {"optimizer": "adamw", "moe": "off", "mtp": "off", "tokenizer": "v3"},
        "token_efficiency_x",
        0.95,
        1.05,
        "Anchors the 50M point of the local scaling law.",
        "Ladder anchor; cheap replica of p150-baseline.",
    ),
    _cell(
        "p050-muon",
        "50m",
        {"optimizer": "muon", "moe": "off", "mtp": "off", "tokenizer": "v3"},
        "token_efficiency_x",
        1.25,
        1.6,
        "Muon gain must hold within 15% across the 50M->150M ladder.",
        "Scale-consistency check for the Muon effect.",
    ),
    _cell(
        "p050-moe",
        "50m",
        {"optimizer": "adamw", "moe": "upcycle-8r1s-top2", "mtp": "off", "tokenizer": "v3"},
        "capability_per_active_flop_x",
        1.4,
        2.5,
        "MoE gain must hold within 20% across the ladder.",
        "Scale-consistency check for the MoE effect.",
    ),
    _cell(
        "p050-mtp",
        "50m",
        {"optimizer": "adamw", "moe": "off", "mtp": "on", "tokenizer": "v3"},
        "token_efficiency_x",
        1.0,
        1.2,
        "MTP gain must not invert across the ladder.",
        "Scale-consistency check for the MTP effect.",
    ),
    _cell(
        "p050-fullstack",
        "50m",
        {
            "optimizer": "muon",
            "moe": "upcycle-8r1s-top2",
            "mtp": "on",
            "tokenizer": "v4-campaign-candidate",
        },
        "effective_compute_x",
        2.3,
        6.0,
        "Full stack must not regress vs the best 50M pair.",
        "Cheap early warning for the p150-fullstack cell.",
        blocked_on="stream-b-canonical-v4",
    ),
    # --- Curriculum-order science (2.5) ------------------------------------
    _cell(
        "c050-code-first",
        "50m",
        {"curriculum": "code-before-prose", "optimizer": "adamw", "tokenizer": "v3"},
        "heldout_delta_pct",
        0.0,
        5.0,
        "Adopt the order if held-out delta >=2% vs the mixed baseline.",
        "Order is free to change; literature says up to ~5% final capability.",
    ),
    _cell(
        "c050-math-ramp",
        "50m",
        {"curriculum": "math-density-ramp", "optimizer": "adamw", "tokenizer": "v3"},
        "heldout_delta_pct",
        0.0,
        5.0,
        "Adopt the order if held-out delta >=2% vs the mixed baseline.",
        "Ramp math density late; tests interference vs consolidation.",
    ),
    _cell(
        "c050-identity-late",
        "50m",
        {"curriculum": "identity-mix-late", "optimizer": "adamw", "tokenizer": "v3"},
        "heldout_delta_pct",
        0.0,
        5.0,
        "Adopt the order if held-out delta >=2% vs the mixed baseline.",
        "Delays the identity mix; tests whether early identity data costs "
        "general capability.",
    ),
    # --- Moonshots (pilot-gated; never on the critical path) ---------------
    _cell(
        "m1-ssm-hybrid",
        "150m",
        {"backbone": "attn-ssm-1to3", "optimizer": "adamw", "tokenizer": "v3"},
        "long_context_throughput_x",
        1.5,
        2.5,
        ">=0.98x short-context capability AND >=1.5x long-context throughput; "
        "any short-context regression >2% shelves it.",
        "M1: Mamba-2-class SSM blocks replace the SWA layers (Jamba/Zamba "
        "pattern). New-architecture branch, never surgery on the lineage.",
        moonshot=True,
    ),
    _cell(
        "m3-latent-reasoning",
        "150m",
        {"reasoning": "latent-recurrent", "optimizer": "adamw", "tokenizer": "v3"},
        "reasoning_suite_score_x",
        1.0,
        1.2,
        ">=1.15x reasoning score at matched inference FLOPs vs <think> "
        "baseline; below 1.05x it is shelved.",
        "M3: COCONUT-style continuous thought; must decisively beat the "
        "legible token-space baseline to justify opacity.",
        moonshot=True,
    ),
    _cell(
        "m5-retriever-head",
        "150m",
        {"retriever": "two-tower-trained", "optimizer": "adamw", "tokenizer": "v3"},
        "recall_at5_delta_pct",
        0.0,
        15.0,
        ">=+10% recall@5 over the hybrid baseline on held-out episodes; "
        "killed if training pairs <20k.",
        "M5: trained retriever head behind the S3 interface.",
        moonshot=True,
        blocked_on="ledger-training-pairs-20k",
    ),
)


def factorial_summary(cells: tuple[PilotCell, ...] = PILOT_FACTORIAL) -> dict[str, object]:
    return {
        "cells": len(cells),
        "moonshots": sum(1 for cell in cells if cell.moonshot),
        "blocked": sorted(cell.cell_id for cell in cells if cell.blocked_on),
        "scales": sorted({cell.scale for cell in cells}),
        "seeds_per_cell": len(PILOT_SEEDS),
    }


def build_pilot_launch_manifests(
    output_dir: str | Path,
    *,
    owner_authorized: bool,
    key: str | None = None,
    seeds: tuple[int, ...] = PILOT_SEEDS,
    cells: tuple[PilotCell, ...] = PILOT_FACTORIAL,
    ledger_path: str | Path = FORECAST_LEDGER,
) -> list[dict[str, object]]:
    """Register forecasts, then emit one signed manifest per cell (3 seeds).

    Order is the contract: the forecast-ledger entry is appended *before*
    ``build_launch_manifest`` stamps ``created_at``, and every signed manifest
    must pass :func:`audit_pre_launch` before it is returned.
    """
    if len(seeds) != 3:
        raise ValueError("The pilot factorial pre-registers exactly three seeds per cell")
    if not owner_authorized:
        raise PermissionError("Pilot launch manifests require explicit owner authorization")
    if not (key or os.environ.get("ANRA_MANIFEST_SIGNING_KEY", "")):
        # Checked before any forecast is appended so a failed run cannot
        # leave orphan pre-registrations in the canonical ledger.
        raise PermissionError("ANRA_MANIFEST_SIGNING_KEY is required to sign a launch manifest.")
    root = Path(output_dir)
    tokenizer_hash = hashlib.sha256(active_tokenizer_path().read_bytes()).hexdigest()
    manifests: list[dict[str, object]] = []
    for cell in cells:
        forecast = register_forecast(
            cell_id=cell.cell_id,
            metric=cell.metric,
            predicted_low=cell.predicted_low,
            predicted_high=cell.predicted_high,
            gate=cell.gate,
            seeds=list(seeds),
            rationale=cell.rationale,
            path=ledger_path,
        )
        manifest = build_launch_manifest(
            model_profile=f"pilot-{cell.scale}",
            extension_profile="none",
            tokenizer_hash=tokenizer_hash,
            data_manifests=list(PILOT_DATA_MANIFESTS),
            stage=f"pilot-factorial-{cell.scale}",
            optimizer=cell.optimizer,
            batch_size=32,
            accumulation=8,
            schedule=dict(PILOT_SCHEDULE),
            seeds=list(seeds),
            checkpoint_source="scratch",
            expected_tokens=EXPECTED_TOKENS_BY_SCALE[cell.scale],
            runtime_estimate_hours=None,
            owner_authorized=owner_authorized,
            worker_id=f"pilot-{cell.cell_id}",
            worker_role="pilot_cell",
            artifact_path=str(
                ROOT / "output" / "v2" / "campaigns" / "pilots" / "artifacts" / f"{cell.cell_id}.pt"
            ),
            checkpoint_read_only=True,
        )
        manifest["pilot_cell_id"] = cell.cell_id
        manifest["pilot_axes"] = dict(cell.axes)
        manifest["pilot_scale"] = cell.scale
        manifest["moonshot"] = cell.moonshot
        manifest["blocked_on"] = cell.blocked_on
        manifest["forecast_id"] = forecast["forecast_id"]
        signed = sign_manifest(manifest, root / "cells" / f"{cell.cell_id}.json", key=key)
        audit = audit_pre_launch(signed, path=ledger_path)
        if not audit["passed"]:  # pragma: no cover - audit raises on failure
            raise PermissionError(f"Pre-launch audit failed for {cell.cell_id}")
        manifests.append(signed)
    return manifests


def main() -> int:
    """Owner entry point: emit the full pre-registered pilot manifest set.

    Requires ANRA_MANIFEST_SIGNING_KEY in the environment and explicit
    --owner-authorized; forecasts land in the canonical ledger first.
    """
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Build the pre-registered pilot factorial.")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "output" / "v2" / "campaigns" / "pilots"),
    )
    parser.add_argument("--owner-authorized", action="store_true")
    args = parser.parse_args()

    manifests = build_pilot_launch_manifests(
        args.output_dir,
        owner_authorized=args.owner_authorized,
    )
    print(
        json.dumps(
            {
                "factorial": factorial_summary(),
                "manifests_written": len(manifests),
                "output_dir": str(args.output_dir),
                "forecast_ledger": str(FORECAST_LEDGER),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
