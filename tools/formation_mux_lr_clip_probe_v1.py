"""LR/clip choke triage probe (engineering diagnostic, NOT science).

Runs two short engineering-only M0_STANDARD lanes (fresh seeds 93001/93002,
300 updates each, frozen LR) and reports clip_fraction + pre-clip gradient
norms. Motivation: every S5 CS-MECH arm recorded clip_fraction = 1.0, so the
substrate may be optimization-choked rather than mechanism-null.

Contract (architecture, not hack):
- No frozen-science change: same model/train/protocol/surface code and LR.
- No sealed reads, no verdict change, no promotion/authorization.
- Non-blocking advisory only: CHOKED / FLOWING / MIXED + recommendation text.
- Requires an existing PUBLIC_SURFACE_MANIFEST.json (from any prior session).
- Runs anywhere (T4 ~30 min; CPU slower but free quota); device recorded.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

SCHEMA = "anra.lr-clip-probe/v1"
EXPERIMENT = "CS-MECH-002"
ARM = "M0_STANDARD"
SEEDS = (93001, 93002)
UPDATES = 300


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _lane_summary(arm_result: Mapping[str, Any]) -> dict[str, Any]:
    diags = list(arm_result.get("diagnostics", []))
    norms = [float(d["grad_norm_pre_clip"]) for d in diags if d.get("grad_norm_pre_clip") is not None]
    return {
        "updates": int(arm_result.get("updates", 0)),
        "clip_fraction": float(arm_result.get("clip_fraction", 1.0)),
        "mean_grad_norm_pre_clip": (sum(norms) / len(norms)) if norms else None,
        "max_grad_norm_pre_clip": max(norms) if norms else None,
        "formation": arm_result.get("formation"),
    }


def run_probe(*, repo: Path, out: Path, torch: Any, device: Any) -> dict[str, Any]:
    started = time.monotonic()
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    from anra_v5 import formation_mux_train_v5 as train
    from v5_experiments.formation_mux_surface_v5 import load_public_surface

    manifest_path = out / "PUBLIC_SURFACE_MANIFEST.json"
    if not manifest_path.exists():
        raise RuntimeError(
            "probe requires PUBLIC_SURFACE_MANIFEST.json on disk; "
            "run one GPU train session first (or attach its Output) so the "
            "frozen S5 surface exists"
        )
    surface = load_public_surface(manifest_path)
    lanes: dict[str, Any] = {}
    for seed in SEEDS:
        label = f"CAL{seed}"
        print(f"PROBE {ARM} seed={seed} target={UPDATES} updates on {device} ...", flush=True)
        body = train.train_arm(
            experiment=EXPERIMENT,
            arm=ARM,
            seed_bundle=seed,
            surface=surface,
            out_dir=out / "lr_clip_probe",
            torch=torch,
            device=device,
            engineering_only=True,
            a_updates_override=UPDATES,
            b_token_budget_override=None,
            progress=lambda m: print("PROBE " + m, flush=True),
        )
        if body.get("status") != "COMPLETE":
            raise RuntimeError(f"probe lane seed={seed} did not complete: {body.get('status')}")
        result_path = out / "lr_clip_probe" / EXPERIMENT / ARM / label / "ARM_RESULT.json"
        arm_result = json.loads(result_path.read_text(encoding="utf-8"))
        lanes[str(seed)] = _lane_summary(arm_result)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # Architecture (quota exactness): the probe is one-shot triage with no
    # resume contract, so drop its resume.pt checkpoints (rerunnable in ~30
    # min) AND every receipt that points at them. Leaving LATEST_PROGRESS /
    # CHECKPOINT_RECEIPT behind a deleted checkpoint would be a dangling
    # custody claim. ARM_RESULT.json survives: it is the probe's evidence and
    # it rides the normal result bundle free. Storage-preflight slot math
    # stays exact for official arms.
    for ckpt in sorted((out / "lr_clip_probe").rglob("resume.pt")):
        ckpt.unlink()
    for name in ("LATEST_PROGRESS.json", "CHECKPOINT_RECEIPT.json"):
        for receipt in sorted((out / "lr_clip_probe").rglob(name)):
            receipt.unlink()
    for progress_dir in sorted((out / "lr_clip_probe").rglob("progress")):
        shutil.rmtree(progress_dir, ignore_errors=True)
    fracs = [float(v["clip_fraction"]) for v in lanes.values()]
    if min(fracs) >= 0.99:
        verdict, recommendation = (
            "CHOKED",
            "Both probe lanes clipped ~every update at frozen LR=1e-3. "
            "Treat the S5 NULLs as possibly optimization-choked; open the S6 "
            "amendment (LR/schedule bracket) before any 8h mechanism re-run.",
        )
    elif max(fracs) <= 0.90:
        verdict, recommendation = (
            "FLOWING",
            "Clipping is intermittent at frozen LR. The S5 floor is unlikely "
            "to be pure clip-choke; prioritize the metric-resolution audit + "
            "baseline-capability gate over optimizer work.",
        )
    else:
        verdict, recommendation = (
            "MIXED",
            "Clipping is seed-sensitive. Carry both hypotheses (choke + "
            "formation) into the baseline gate; record per-seed clip traces.",
        )
    receipt = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "arm": ARM,
        "updates_per_lane": UPDATES,
        "engineering_only": True,
        "sealed_touched": False,
        "device": str(device),
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "lanes": lanes,
        "verdict": verdict,
        "recommendation": recommendation,
        "wall_seconds": time.monotonic() - started,
    }
    _atomic_json(out / "LR_CLIP_PROBE.json", receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    import torch

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    receipt = run_probe(repo=args.repo.resolve(), out=args.out.resolve(), torch=torch, device=device)
    print("LR_CLIP_PROBE verdict:", receipt["verdict"], flush=True)
    print("LR_CLIP_PROBE recommendation:", receipt["recommendation"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
