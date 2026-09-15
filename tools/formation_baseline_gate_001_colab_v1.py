"""FORMATION-BASELINE-GATE-001: 2–3 hour Colab T4 capability gate.

Post-outcome diagnostic only. It does not alter FORMATION-MUX-001 S5 verdicts.
The gate asks whether the canonical M0 control can leave the identity floor
reliably enough to justify another expensive mechanism campaign.

Stage A runs two full six-family M0 controls to 3000 updates:
  * 73012: historical S5 late-learning seed
  * 91001: fresh diagnostic seed

If Stage A is decisively floor-limited, Stage B spends the remaining budget on
an identity-only fresh-seed probe to distinguish mixture interference from a
substrate/optimization failure. Otherwise Stage B runs a third fresh full-mixture
control (91002) to test reproducibility.

All training is engineering-only and development-only. No sealed score is read,
reported, or used for any decision. Checkpoints are preserved for exact resume.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "anra.formation-baseline-gate/v1"
SCIENCE_COMMIT = "c15ad8beb409537db42d075684ea54847a074ebd"
EXPECTED_PUBLIC_SURFACE_SHA = "f1d5200bd05bc28ede97af114b49f616ca72b24af74b7fc530a2cb084db4259c"
EXPERIMENT = "CS-MECH-002"
ARM = "M0_STANDARD"
HISTORICAL_SEED = 73012
FRESH_SEEDS = (91001, 91002)
FULL_UPDATES = 3000
IDENTITY_ONLY_UPDATES = 2000
SURFACE_SEED = 73011


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _git_head(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def _seed_label(proto: Any, seed: int) -> str:
    return f"S{proto.SEED_BUNDLES.index(seed) + 1}" if seed in proto.SEED_BUNDLES else f"CAL{seed}"


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _identity_rows(surface: Mapping[str, Any], split: str) -> list[dict[str, Any]]:
    rows = [dict(r) for r in surface["splits"][split] if r.get("family") == "identity"]
    if split == "development" and len(rows) != 80:
        raise RuntimeError(f"expected 80 identity development rows, got {len(rows)}")
    return rows


def _next_logits(model: Any, prefix: list[int], *, torch: Any, device: Any) -> Any:
    from v5_model.core import packed_layout

    current = torch.tensor([prefix], dtype=torch.long, device=device)
    segment = torch.zeros_like(current)
    positions, mask = packed_layout(segment, torch_module=torch)
    return model(current, positions, mask)[0, -1].float()


def _teacher_forced_identity(model: Any, rows: list[Mapping[str, Any]], *, base_train: Any,
                             proto: Any, torch: Any, device: Any) -> dict[str, Any]:
    was_training = bool(model.training)
    model.eval()
    correct = total = 0
    ranks: list[int] = []
    margins: list[float] = []
    by_position: dict[int, list[int]] = {}
    with torch.no_grad():
        for row in rows:
            prompt, answer = base_train._encode_row(row, proto.EXPERIMENT_A, ARM)
            prefix = list(prompt)
            for pos, target in enumerate(answer):
                target = int(target)
                logits = _next_logits(model, prefix, torch=torch, device=device)
                pred = int(torch.argmax(logits).item())
                hit = int(pred == target)
                correct += hit
                total += 1
                by_position.setdefault(pos, []).append(hit)
                target_logit = logits[target]
                ranks.append(1 + int(torch.sum(logits > target_logit).item()))
                other = logits.clone()
                other[target] = -float("inf")
                margins.append(float((target_logit - torch.max(other)).item()))
                prefix.append(target)
    if was_training:
        model.train()
    ordered_ranks = sorted(ranks)
    return {
        "target_token_accuracy": correct / max(total, 1),
        "mean_target_rank": _mean([float(x) for x in ranks]),
        "median_target_rank": float(ordered_ranks[len(ordered_ranks) // 2]) if ordered_ranks else None,
        "mean_target_margin": _mean(margins),
        "target_tokens": total,
        "accuracy_by_output_position": {
            str(k): _mean([float(x) for x in v]) for k, v in sorted(by_position.items())
        },
    }


def _greedy_identity(model: Any, rows: list[Mapping[str, Any]], *, shared_only: bool,
                     base_train: Any, proto: Any, fxm: Any, torch: Any, device: Any) -> dict[str, Any]:
    was_training = bool(model.training)
    model.eval()
    complete = content = eos = 0
    by_length: dict[int, list[int]] = {}
    with torch.no_grad():
        for row in rows:
            prompt, answer = base_train._encode_row(row, proto.EXPERIMENT_A, ARM)
            expected = [int(x) for x in answer if int(x) != 3]
            generated: list[int] = []
            stopped = False
            for _ in range(proto.MAX_GENERATION_TOKENS):
                logits = _next_logits(model, prompt + generated, torch=torch, device=device)
                if shared_only:
                    logits = logits.clone()
                    logits[fxm.SHARED_VOCAB:] = -float("inf")
                nxt = int(torch.argmax(logits).item())
                if nxt == 3:
                    stopped = True
                    break
                if nxt == 0:
                    break
                generated.append(nxt)
            exact = int(generated == expected and stopped)
            complete += exact
            content += int(generated == expected)
            eos += int(stopped)
            by_length.setdefault(len(expected), []).append(exact)
    if was_training:
        model.train()
    n = max(len(rows), 1)
    return {
        "exact_valid_eos": complete / n,
        "content_exact": content / n,
        "eos_rate": eos / n,
        "exact_by_answer_length": {
            str(k): _mean([float(x) for x in v]) for k, v in sorted(by_length.items())
        },
        "n": len(rows),
    }


def _diagnose_checkpoint(*, checkpoint: Path, seed: int, surface: Mapping[str, Any],
                         base_train: Any, proto: Any, fxm: Any, torch: Any, device: Any) -> dict[str, Any]:
    model = base_train.load_model_for_evaluation(
        checkpoint,
        experiment=proto.EXPERIMENT_A,
        arm=ARM,
        seed_bundle=seed,
        data_manifest_sha256=str(surface["sha256"]),
        torch=torch,
        device=device,
    )
    rows = _identity_rows(surface, "development")
    full = _greedy_identity(
        model, rows, shared_only=False, base_train=base_train, proto=proto, fxm=fxm,
        torch=torch, device=device,
    )
    shared = _greedy_identity(
        model, rows, shared_only=True, base_train=base_train, proto=proto, fxm=fxm,
        torch=torch, device=device,
    )
    teacher = _teacher_forced_identity(
        model, rows, base_train=base_train, proto=proto, torch=torch, device=device,
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "full_vocab": full,
        "shared_only": shared,
        "shared_only_rescue_exact": shared["exact_valid_eos"] - full["exact_valid_eos"],
        "teacher_forced": teacher,
    }


def _identity_only_surface(public: Mapping[str, Any]) -> dict[str, Any]:
    surface = copy.deepcopy(dict(public))
    surface["schema"] = "anra.formation-baseline-gate-identity-only/v1"
    surface["splits"] = {
        "training": _identity_rows(public, "training"),
        "development": _identity_rows(public, "development"),
    }
    surface["diagnostic_only"] = True
    surface["source_public_surface_sha256"] = str(public["sha256"])
    surface["sealed_rows_persisted"] = False
    surface.pop("sha256", None)
    surface["sha256"] = _sha({k: v for k, v in surface.items() if k != "sha256"})
    return surface


def _run_full_control(*, seed: int, out: Path, public: Mapping[str, Any], train_v5: Any,
                      base_train: Any, proto: Any, fxm: Any, torch: Any, device: Any) -> dict[str, Any]:
    lane = out / "FULL_MIXTURE" / f"SEED_{seed}"
    print(f"\n=== FULL MIXTURE M0 seed={seed} target={FULL_UPDATES} updates ===", flush=True)
    started = time.monotonic()
    result = train_v5.train_arm(
        experiment=proto.EXPERIMENT_A,
        arm=ARM,
        seed_bundle=seed,
        surface=public,
        out_dir=lane,
        torch=torch,
        device=device,
        engineering_only=True,
        a_updates_override=FULL_UPDATES,
        progress=lambda msg: print(msg, flush=True),
    )
    label = _seed_label(proto, seed)
    checkpoint = lane / proto.EXPERIMENT_A / ARM / label / "resume.pt"
    if not checkpoint.exists():
        raise RuntimeError(f"expected checkpoint missing: {checkpoint}")
    diag = _diagnose_checkpoint(
        checkpoint=checkpoint, seed=seed, surface=public, base_train=base_train,
        proto=proto, fxm=fxm, torch=torch, device=device,
    )
    row = {
        "lane": "FULL_MIXTURE",
        "seed": seed,
        "target_updates": FULL_UPDATES,
        "formation": result["formation"],
        "clip_fraction": result.get("clip_fraction"),
        "training_timing": result.get("timing", {}),
        "arm_wall_seconds_this_invocation": result.get("wall_seconds"),
        "operator_wall_seconds_this_lane": time.monotonic() - started,
        "checkpoint": str(checkpoint),
        **diag,
    }
    _atomic_json(lane / "GATE_DIAGNOSTIC.json", row)
    return row


def _run_identity_only_probe(*, seed: int, out: Path, public: Mapping[str, Any],
                             base_train: Any, proto: Any, fxm: Any, torch: Any, device: Any) -> dict[str, Any]:
    surface = _identity_only_surface(public)
    lane = out / "IDENTITY_ONLY_PROBE" / f"SEED_{seed}"
    print(f"\n=== IDENTITY-ONLY M0 seed={seed} target={IDENTITY_ONLY_UPDATES} updates ===", flush=True)
    started = time.monotonic()
    result = base_train.train_arm(
        experiment=proto.EXPERIMENT_A,
        arm=ARM,
        seed_bundle=seed,
        surface=surface,
        out_dir=lane,
        torch=torch,
        device=device,
        engineering_only=True,
        a_updates_override=IDENTITY_ONLY_UPDATES,
        progress=lambda msg: print(msg, flush=True),
    )
    label = _seed_label(proto, seed)
    checkpoint = lane / proto.EXPERIMENT_A / ARM / label / "resume.pt"
    if not checkpoint.exists():
        raise RuntimeError(f"expected identity-only checkpoint missing: {checkpoint}")
    diag = _diagnose_checkpoint(
        checkpoint=checkpoint, seed=seed, surface=surface, base_train=base_train,
        proto=proto, fxm=fxm, torch=torch, device=device,
    )
    row = {
        "lane": "IDENTITY_ONLY_PROBE",
        "seed": seed,
        "target_updates": IDENTITY_ONLY_UPDATES,
        "surface_sha256": surface["sha256"],
        "source_public_surface_sha256": public["sha256"],
        "formation": result["formation"],
        "clip_fraction": result.get("clip_fraction"),
        "training_timing": result.get("timing", {}),
        "arm_wall_seconds_this_invocation": result.get("wall_seconds"),
        "operator_wall_seconds_this_lane": time.monotonic() - started,
        "checkpoint": str(checkpoint),
        **diag,
    }
    _atomic_json(lane / "GATE_DIAGNOSTIC.json", row)
    return row


def _stage_a_floor(rows: list[Mapping[str, Any]]) -> bool:
    endpoints = [float(r["full_vocab"]["exact_valid_eos"]) for r in rows]
    token_acc = [float(r["teacher_forced"]["target_token_accuracy"]) for r in rows]
    rescues = [float(r["shared_only_rescue_exact"]) for r in rows]
    return max(endpoints) < 0.10 and _mean(token_acc) < 0.60 and _mean(rescues) < 0.05


def _stage_a_ceiling(rows: list[Mapping[str, Any]]) -> bool:
    return min(float(r["full_vocab"]["exact_valid_eos"]) for r in rows) > 0.85


def _final_decision(full_rows: list[Mapping[str, Any]], identity_probe: Mapping[str, Any] | None) -> dict[str, Any]:
    endpoints = [float(r["full_vocab"]["exact_valid_eos"]) for r in full_rows]
    token_acc = [float(r["teacher_forced"]["target_token_accuracy"]) for r in full_rows]
    rescues = [float(r["shared_only_rescue_exact"]) for r in full_rows]

    if identity_probe is not None:
        p_exact = float(identity_probe["full_vocab"]["exact_valid_eos"])
        p_token = float(identity_probe["teacher_forced"]["target_token_accuracy"])
        if p_exact >= 0.50 or p_token >= 0.90:
            return {
                "decision": "NO_GO_MIXTURE_INTERFERENCE_FIRST",
                "expensive_mechanism_campaign_worth_running": False,
                "reason": "Full-mixture controls stayed at the floor, but the identity-only probe learned strongly. Fix mixture/curriculum or effective identity exposure before spending on TIE-ROLE.",
            }
        if p_token >= 0.75:
            return {
                "decision": "NO_GO_IDENTITY_ONLY_PARTIAL",
                "expensive_mechanism_campaign_worth_running": False,
                "reason": "Even identity-only training produced only partial token-level formation. Baseline optimization/realization needs repair before mechanism science.",
            }
        return {
            "decision": "NO_GO_SUBSTRATE_FORMATION",
            "expensive_mechanism_campaign_worth_running": False,
            "reason": "Both full-mixture controls and a dedicated identity-only probe failed to form the primary capability. Diagnose optimization/capacity before any large mechanism campaign.",
        }

    if sum(x >= 0.85 for x in endpoints) >= 2:
        return {
            "decision": "NO_GO_CEILING_RECALIBRATE",
            "expensive_mechanism_campaign_worth_running": False,
            "reason": "The control moved too close to ceiling; the old paired endpoint threshold would lose discrimination. Recalibrate exposure before mechanism testing.",
        }

    fresh_endpoints = [
        float(r["full_vocab"]["exact_valid_eos"]) for r in full_rows if int(r["seed"]) in FRESH_SEEDS
    ]
    if (
        len(full_rows) >= 3
        and len(fresh_endpoints) == 2
        and min(fresh_endpoints) >= 0.30
        and 0.30 <= _mean(endpoints) <= 0.80
        and _mean(token_acc) >= 0.80
    ):
        return {
            "decision": "GO_MECHANISM_CAMPAIGN_WORTH_IT",
            "expensive_mechanism_campaign_worth_running": True,
            "reason": "The canonical full-mixture control now forms identity away from floor/ceiling across matched diagnostic seeds, including fresh-seed evidence. A new prospectively frozen mechanism campaign is worth the compute.",
        }

    if _mean(rescues) >= 0.10:
        return {
            "decision": "NO_GO_OUTPUT_COMPETITION_FIRST",
            "expensive_mechanism_campaign_worth_running": False,
            "reason": "Shared-only decoding materially rescues identity. Localize output-space competition before running TIE-ROLE.",
        }
    if _mean(token_acc) >= 0.80 and _mean(endpoints) < 0.30:
        return {
            "decision": "NO_GO_REALIZATION_OR_METRIC_FIRST",
            "expensive_mechanism_campaign_worth_running": False,
            "reason": "Teacher-forced token learning is strong but free sequence exact remains weak. The next bottleneck is autoregressive realization/measurement, not another architecture factorial.",
        }
    if max(endpoints) >= 0.15 or _mean(token_acc) >= 0.65:
        return {
            "decision": "BORDERLINE_SMALL_OPTIMIZATION_PROBE_FIRST",
            "expensive_mechanism_campaign_worth_running": False,
            "reason": "Formation exists but is still too seed-sensitive or weak. Spend one small optimization/exposure probe before any 8+ hour mechanism campaign.",
        }
    return {
        "decision": "NO_GO_FLOOR",
        "expensive_mechanism_campaign_worth_running": False,
        "reason": "The canonical control remains near the primary floor after extended exposure. Do not launch the expensive mechanism campaign yet.",
    }


def _package(out: Path) -> dict[str, Any]:
    results_zip = out / "FORMATION_BASELINE_GATE_001_RESULTS.zip"
    checkpoints_zip = out / "FORMATION_BASELINE_GATE_001_CHECKPOINTS.zip"
    for p in (results_zip, checkpoints_zip):
        if p.exists():
            p.unlink()

    def excluded(p: Path) -> bool:
        return p.name in {results_zip.name, checkpoints_zip.name} or p.suffix == ".sha256"

    with zipfile.ZipFile(results_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(out.rglob("*")):
            if not p.is_file() or excluded(p) or p.name == "resume.pt":
                continue
            zf.write(p, p.relative_to(out))

    with zipfile.ZipFile(checkpoints_zip, "w", zipfile.ZIP_STORED) as zf:
        for p in sorted(out.rglob("resume.pt")):
            zf.write(p, p.relative_to(out))
        for p in sorted(out.rglob("CHECKPOINT_RECEIPT.json")):
            zf.write(p, p.relative_to(out))
        for p in sorted(out.rglob("ARM_RESULT.json")):
            zf.write(p, p.relative_to(out))

    receipts = {}
    for label, p in (("results", results_zip), ("checkpoints", checkpoints_zip)):
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        Path(str(p) + ".sha256").write_text(f"{digest}  {p.name}\n", encoding="utf-8")
        receipts[label] = {"path": str(p), "sha256": digest, "bytes": p.stat().st_size}
    return receipts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prereg", type=Path, required=True)
    args = parser.parse_args(argv)

    repo = args.repo.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    prereg = json.loads(args.prereg.read_text(encoding="utf-8"))
    if prereg.get("schema") != "anra.formation-baseline-gate-preregistration/v1":
        raise RuntimeError("unexpected preregistration schema")
    if prereg.get("science_reference_commit") != SCIENCE_COMMIT:
        raise RuntimeError("preregistration science commit mismatch")
    if _git_head(repo) != SCIENCE_COMMIT:
        raise RuntimeError(f"repo must be checked out at frozen S5 science commit {SCIENCE_COMMIT}")

    import sys
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required; in Colab select a T4 GPU runtime")
    device = torch.device("cuda:0")

    from v5_data.corpus_loading import _load_tokenizer
    from v5_experiments.formation_mux_surface_v5 import build_public_surface
    from v5_experiments import formation_mux_protocol_v5 as proto
    from anra_v5 import formation_mux_train_v5 as train_v5
    from anra_v5 import formation_mux_train_v2 as base_train
    from anra_v5 import formation_mux_model_v3 as fxm

    # formation_mux_train_v5 intentionally binds base_train to S5 model/protocol.
    if base_train.proto is not proto or base_train.fxm is not fxm:
        raise RuntimeError("S5 train binding failed")

    tokenizer, _ = _load_tokenizer(repo)
    public = build_public_surface(seed=SURFACE_SEED, tokenizer=tokenizer)
    if str(public["sha256"]) != EXPECTED_PUBLIC_SURFACE_SHA:
        raise RuntimeError(f"public surface drift: {public['sha256']} != {EXPECTED_PUBLIC_SURFACE_SHA}")
    if "sealed" in public.get("splits", {}):
        raise RuntimeError("sealed firewall breach: public surface contains sealed rows")

    # Speed optimization: evaluate only the preregistered primary identity family.
    # Training still uses the full six-family S5 mixture. This is diagnostic-only.
    original_eval = base_train.evaluate_development
    def identity_only_eval(model: Any, dev_rows: list[Mapping[str, Any]], experiment: str,
                           arm: str, *, torch: Any, device: Any) -> dict[str, Any]:
        identity = [r for r in dev_rows if r.get("family") == "identity"]
        return original_eval(model, identity, experiment, arm, torch=torch, device=device)
    base_train.evaluate_development = identity_only_eval
    train_v5.evaluate_development = identity_only_eval

    execution = {
        "schema": SCHEMA,
        "status": "RUNNING",
        "science_reference_commit": SCIENCE_COMMIT,
        "public_surface_sha256": public["sha256"],
        "public_surface_seed": SURFACE_SEED,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "official_science": False,
        "diagnostic_only": True,
        "sealed_scores_used": False,
        "sealed_rows_persisted": False,
        "full_mixture_target_updates": FULL_UPDATES,
        "identity_only_target_updates": IDENTITY_ONLY_UPDATES,
        "stage_a_seeds": [HISTORICAL_SEED, FRESH_SEEDS[0]],
        "adaptive_stage_b_seed": FRESH_SEEDS[1],
        "preregistration_sha256": hashlib.sha256(args.prereg.read_bytes()).hexdigest(),
    }
    _atomic_json(out / "EXECUTION_BINDING.json", execution)

    full_rows: list[dict[str, Any]] = []
    for seed in (HISTORICAL_SEED, FRESH_SEEDS[0]):
        full_rows.append(_run_full_control(
            seed=seed, out=out, public=public, train_v5=train_v5, base_train=base_train,
            proto=proto, fxm=fxm, torch=torch, device=device,
        ))

    identity_probe = None
    stage_a = {
        "floor": _stage_a_floor(full_rows),
        "ceiling": _stage_a_ceiling(full_rows),
        "mean_exact": _mean([float(r["full_vocab"]["exact_valid_eos"]) for r in full_rows]),
        "mean_teacher_forced_token_accuracy": _mean([float(r["teacher_forced"]["target_token_accuracy"]) for r in full_rows]),
        "mean_shared_only_rescue": _mean([float(r["shared_only_rescue_exact"]) for r in full_rows]),
    }
    _atomic_json(out / "STAGE_A_DECISION.json", stage_a)

    if stage_a["ceiling"]:
        print("\nStage A is already at ceiling; skipping third training lane.", flush=True)
    elif stage_a["floor"]:
        print("\nStage A is decisively floor-limited; using remaining budget on identity-only disambiguation.", flush=True)
        identity_probe = _run_identity_only_probe(
            seed=FRESH_SEEDS[1], out=out, public=public, base_train=base_train,
            proto=proto, fxm=fxm, torch=torch, device=device,
        )
    else:
        print("\nStage A shows formation signal; using remaining budget on a third fresh full-mixture replication.", flush=True)
        full_rows.append(_run_full_control(
            seed=FRESH_SEEDS[1], out=out, public=public, train_v5=train_v5,
            base_train=base_train, proto=proto, fxm=fxm, torch=torch, device=device,
        ))

    decision = _final_decision(full_rows, identity_probe)
    result = {
        "schema": SCHEMA,
        "status": "COMPLETE",
        "science_reference_commit": SCIENCE_COMMIT,
        "source_s5_formal_verdicts_unchanged": {"CS-MECH-002": "NULL", "REP-FORM-003A": "NULL"},
        "official_science": False,
        "diagnostic_only": True,
        "sealed_scores_used": False,
        "sealed_rows_persisted": False,
        "public_surface_sha256": public["sha256"],
        "stage_a": stage_a,
        "full_mixture_controls": full_rows,
        "identity_only_probe": identity_probe,
        **decision,
        "wall_seconds_this_invocation": time.monotonic() - started,
        "next_rule": "Only GO_MECHANISM_CAMPAIGN_WORTH_IT licenses design of a new prospectively frozen mechanism campaign. It does not revive or modify the old S5 verdict or execute the old TIE-ROLE frontier.",
    }
    _atomic_json(out / "BASELINE_GATE_RESULT.json", result)
    packages = _package(out)
    _atomic_json(out / "PACKAGE_RECEIPT.json", packages)

    print("\n" + "#" * 78)
    print("FORMATION-BASELINE-GATE-001 COMPLETE")
    print("DECISION:", result["decision"])
    print("EXPENSIVE MECHANISM CAMPAIGN WORTH RUNNING:", result["expensive_mechanism_campaign_worth_running"])
    print("REASON:", result["reason"])
    print("WALL HOURS THIS INVOCATION:", round(result["wall_seconds_this_invocation"] / 3600.0, 3))
    print("RESULTS ZIP:", packages["results"]["path"])
    print("CHECKPOINTS ZIP:", packages["checkpoints"]["path"])
    print("#" * 78, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
