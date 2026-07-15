"""Execute local-safe moonshot paths without fabricating campaign evidence."""

from __future__ import annotations

from dataclasses import asdict

import torch
from multimodal.vision import InHouseVisionEncoder, VisionSoftTokenProjector
from retrieval.protocols import RetrievalQuery
from retrieval.trained import TwoTowerRetriever
from robotics.rollout import rollout_actions
from robotics.world_model import PredictiveWorldModel
from runtime.experience_ledger import content_hash
from self_modification.proposal_ladder import SelfDevelopmentProposal, evaluate_proposal_only

from training.formal_proof_pilot import run_formal_proof_pilot
from training.moonshot_architectures import LatentReasoningChannel, StateSpaceMixer
from training.moonshot_pilots import MOONSHOT_PILOTS, evaluate_moonshot_pilot

_EXTERNAL_BLOCKERS: dict[str, tuple[str, ...]] = {
    "m1": (
        "canonical 181M matched three-run training artifacts",
        "matched short-context capability evaluation",
        "matched long-context throughput benchmark",
    ),
    "m2": (
        "Gate 6 text-core completion",
        "licensed image and image-text training corpora",
        "in-house vision training compute",
        "held-out 5k retrieval and 200-item vision-QA suites",
    ),
    "m3": (
        "canonical 181M matched three-run latent and token-thinking checkpoints",
        "matched-inference-FLOP reasoning evaluation",
    ),
    "m4": (
        "trained world-model checkpoint",
        "held-out simulation transitions",
        "held-out ledger tool-call transitions and majority baseline",
    ),
    "m5": (
        "at least 20,000 verified query-memory training pairs",
        "trained two-tower checkpoint",
        "held-out recall@5 comparison against the hybrid baseline",
    ),
    "m7": (
        "10 human-approved merged pull requests",
        "signed sovereignty-gate records",
        "revert-window audit",
    ),
}


def _m1_smoke() -> dict[str, object]:
    values = torch.randn(2, 16, 8)
    output = StateSpaceMixer(8)(values)
    finite = bool(torch.isfinite(output).all())
    return {
        "passed": output.shape == values.shape and finite,
        "shape": list(output.shape),
        "finite": finite,
    }


def _m2_smoke() -> dict[str, object]:
    images = torch.randn(2, 3, 16, 16)
    encoder = InHouseVisionEncoder(width=8, patch_size=4)
    projector = VisionSoftTokenProjector(8, 12)
    tokens = projector(encoder(images))
    finite = bool(torch.isfinite(tokens).all())
    return {
        "passed": tokens.shape == (2, 16, 12) and finite,
        "shape": list(tokens.shape),
        "finite": finite,
        "external_weights_loaded": False,
    }


def _m3_smoke() -> dict[str, object]:
    values = torch.randn(2, 8, 8)
    output = LatentReasoningChannel(8, latent_steps=4)(values)
    finite = bool(torch.isfinite(output).all())
    return {
        "passed": output.shape == (2, 8) and finite,
        "shape": list(output.shape),
        "finite": finite,
    }


def _m4_smoke() -> dict[str, object]:
    report = rollout_actions(
        PredictiveWorldModel(state_dim=8, action_dim=4, hidden_dim=16),
        {"position": 0},
        ({"move": "left"}, {"move": "right"}),
        max_uncertainty=10.0,
    )
    rollout_steps = len(report["steps"])  # type: ignore[arg-type]
    return {
        "passed": report["offline_only"] is True and rollout_steps == 2,
        "offline_only": report["offline_only"] is True,
        "rollout_steps": rollout_steps,
    }


def _m5_smoke() -> dict[str, object]:
    retriever = TwoTowerRetriever(4)
    with torch.no_grad():
        identity = torch.eye(4)
        retriever.query_head.weight.copy_(identity)
        retriever.document_head.weight.copy_(identity)
    retriever.index(
        (
            ("alpha", "alpha", torch.tensor([1.0, 0.0, 0.0, 0.0])),
            ("beta", "beta", torch.tensor([0.0, 1.0, 0.0, 0.0])),
        )
    )
    hits = retriever.search(RetrievalQuery("alpha", limit=1, vector=[1, 0, 0, 0]))
    passed = bool(hits) and hits[0].id == "alpha"
    return {"passed": passed, "protocol_search_passed": passed}


def _m7_smoke() -> dict[str, object]:
    outcome = evaluate_proposal_only(
        SelfDevelopmentProposal("local-readiness", "tests", 0.01, True)
    )
    passed = (
        outcome["eligible_for_human_review"] is True
        and outcome["auto_apply"] is False
        and outcome["merged"] is False
    )
    return {
        "passed": passed,
        "eligible_for_human_review": outcome["eligible_for_human_review"] is True,
        "auto_apply": outcome["auto_apply"] is True,
        "merged": outcome["merged"] is True,
    }


def execute_local_moonshot_paths() -> dict[str, object]:
    """Run every safe local path and return acceptance evidence only when valid."""
    torch.manual_seed(1301)
    smoke_results = {
        "m1": _m1_smoke(),
        "m2": _m2_smoke(),
        "m3": _m3_smoke(),
        "m4": _m4_smoke(),
        "m5": _m5_smoke(),
        "m7": _m7_smoke(),
    }
    formal_report = run_formal_proof_pilot()
    m6_metrics = dict(formal_report["metrics"])  # type: ignore[arg-type]
    m6_gate = evaluate_moonshot_pilot("m6", m6_metrics)
    rows: list[dict[str, object]] = []
    for pilot in MOONSHOT_PILOTS:
        if pilot.moonshot_id == "m6":
            rows.append(
                {
                    "moonshot_id": "m6",
                    "title": pilot.title,
                    "local_path_executed": True,
                    "local_smoke_passed": True,
                    "acceptance_status": "passed" if m6_gate["passed"] else "failed",
                    "acceptance_metrics": m6_metrics,
                    "blockers": [],
                    "formal_report_hash": formal_report["report_hash"],
                }
            )
            continue
        smoke = smoke_results[pilot.moonshot_id]
        smoke_passed = smoke.get("passed") is True
        rows.append(
            {
                "moonshot_id": pilot.moonshot_id,
                "title": pilot.title,
                "local_path_executed": True,
                "local_smoke_passed": smoke_passed,
                "smoke_evidence": smoke,
                "acceptance_status": "blocked",
                "acceptance_metrics": {},
                "missing_metrics": list(pilot.required_metrics),
                "blockers": list(_EXTERNAL_BLOCKERS[pilot.moonshot_id]),
            }
        )
    report: dict[str, object] = {
        "schema_version": 1,
        "suite": "moonshot_local_execution",
        "all_local_paths_executed": all(row["local_path_executed"] for row in rows),
        "all_local_smokes_passed": all(row["local_smoke_passed"] for row in rows),
        "acceptance_evidence": {"m6": m6_metrics} if m6_gate["passed"] else {},
        "formal_proof_report": formal_report,
        "rows": rows,
        "pilot_contracts": [asdict(pilot) for pilot in MOONSHOT_PILOTS],
    }
    report["report_hash"] = content_hash(report)
    return report
