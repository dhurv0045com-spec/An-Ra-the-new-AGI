"""TIE-ROLE-PILOT-001: direct 2–3 hour compute gate for TIE-ROLE-FRONTIER-001.

This is a diagnostic compute-triage experiment, not part of the official
TIE-ROLE frontier. It directly tests the frontier's primary latent contrast:
T3_BALANCED_X4_X025 versus T0_CANONICAL on two fresh seeds.

If both fresh matched pairs show a sufficiently large, positive formation-AUC
and endpoint effect, the pilot emits RUN_FULL_TIE_ROLE. Otherwise it emits
DO_NOT_RUN_FULL_TIE_ROLE. Ambiguous evidence is deliberately a NO-GO for the
expensive 24-arm frontier.

No sealed rows are loaded or scored. S5 verdicts are immutable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "anra.tie-role-pilot/v1"
SCIENCE_COMMIT = "c15ad8beb409537db42d075684ea54847a074ebd"
PUBLIC_SURFACE_SHA = "f1d5200bd05bc28ede97af114b49f616ca72b24af74b7fc530a2cb084db4259c"
EXPERIMENT = "TIE-ROLE-001"
CONTROL = "T0_CANONICAL"
TREATMENT = "T3_BALANCED_X4_X025"
SEEDS = (92001, 92002)
UPDATES = 2000
BATCH_ROWS = 16
MIN_AUC_DELTA = 0.05
MIN_ENDPOINT_DELTA = 0.10
MAX_ENDPOINT_FOR_DISCRIMINATION = 0.90


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _model_sha(model: Any) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


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
    segments = torch.zeros_like(current)
    positions, mask = packed_layout(segments, torch_module=torch)
    return model(current, positions, mask)[0, -1].float()


def _teacher_forced(model: Any, rows: list[Mapping[str, Any]], *, train: Any, proto: Any, torch: Any, device: Any) -> dict[str, Any]:
    model.eval()
    correct = 0
    total = 0
    ranks: list[int] = []
    margins: list[float] = []
    by_position: dict[int, list[int]] = {}
    with torch.no_grad():
        for row in rows:
            prompt, answer = train._encode_row(row, proto.EXPERIMENT_A, CONTROL)
            prefix = list(prompt)
            for pos, target in enumerate(answer):
                target = int(target)
                logits = _next_logits(model, prefix, torch=torch, device=device)
                hit = int(int(torch.argmax(logits).item()) == target)
                correct += hit
                total += 1
                by_position.setdefault(pos, []).append(hit)
                target_logit = logits[target]
                ranks.append(1 + int(torch.sum(logits > target_logit).item()))
                other = logits.clone()
                other[target] = -float("inf")
                margins.append(float((target_logit - torch.max(other)).item()))
                prefix.append(target)
    return {
        "target_token_accuracy": correct / max(total, 1),
        "mean_target_rank": _mean([float(x) for x in ranks]),
        "median_target_rank": float(sorted(ranks)[len(ranks) // 2]) if ranks else None,
        "mean_target_margin": _mean(margins),
        "accuracy_by_output_position": {str(k): _mean([float(x) for x in v]) for k, v in sorted(by_position.items())},
        "target_tokens": total,
    }


def _greedy(model: Any, rows: list[Mapping[str, Any]], *, train: Any, proto: Any, torch: Any, device: Any) -> dict[str, Any]:
    model.eval()
    complete = content = eos = 0
    by_length: dict[int, list[int]] = {}
    with torch.no_grad():
        for row in rows:
            prompt, answer = train._encode_row(row, proto.EXPERIMENT_A, CONTROL)
            expected = [int(x) for x in answer if int(x) != 3]
            generated: list[int] = []
            stopped = False
            for _ in range(proto.MAX_GENERATION_TOKENS):
                nxt = int(torch.argmax(_next_logits(model, prompt + generated, torch=torch, device=device)).item())
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
    n = max(len(rows), 1)
    return {
        "exact_valid_eos": complete / n,
        "content_exact": content / n,
        "eos_rate": eos / n,
        "exact_by_answer_length": {str(k): _mean([float(x) for x in v]) for k, v in sorted(by_length.items())},
        "n": len(rows),
    }


def _gradient_decomposition(model: Any, rows: list[Mapping[str, Any]], *, train: Any, proto: Any, fxm: Any, torch: Any, device: Any) -> dict[str, Any]:
    from v5_model.core import packed_layout
    from v5_objectives.causal_lm import causal_lm_loss

    tokens, segments, eligible, supervised, processed = train.build_batch(
        rows, proto.EXPERIMENT_A, CONTROL, torch=torch, device=device
    )
    original = fxm.get_gradient_scales(model)

    def component(input_scale: float, output_scale: float) -> Any:
        model.zero_grad(set_to_none=True)
        fxm.set_gradient_scales(model, input_scale=input_scale, output_scale=output_scale)
        positions, mask = packed_layout(segments, torch_module=torch)
        logits = model(tokens, positions, mask)
        loss, _ = causal_lm_loss(
            logits, tokens, segments, bos_id=2, pad_id=0, eligible=eligible, torch_module=torch
        )
        loss.backward()
        grad = model.embedding.weight.grad
        if grad is None:
            raise RuntimeError("missing tied-matrix gradient")
        return grad.detach().float().clone()

    g_in = component(1.0, 0.0)
    g_out = component(0.0, 1.0)
    fxm.set_gradient_scales(model, input_scale=original[0], output_scale=original[1])
    model.zero_grad(set_to_none=True)

    n_in = float(torch.linalg.vector_norm(g_in).item())
    n_out = float(torch.linalg.vector_norm(g_out).item())
    combined = g_in + g_out
    n_combined = float(torch.linalg.vector_norm(combined).item())
    cosine = float((g_in * g_out).sum().item()) / max(n_in * n_out, 1e-30)
    cancellation = 1.0 - n_combined / max(n_in + n_out, 1e-30)
    return {
        "rows": len(rows),
        "supervised_tokens": int(supervised),
        "processed_tokens": int(processed),
        "input_gradient_norm": n_in,
        "output_gradient_norm": n_out,
        "output_to_input_gradient_norm_ratio": n_out / max(n_in, 1e-30),
        "input_output_gradient_cosine": cosine,
        "combined_gradient_norm": n_combined,
        "gradient_cancellation_fraction": cancellation,
    }


def _preflight_forward_equivalence(*, seed: int, public: Mapping[str, Any], train: Any, proto: Any, fxm: Any, torch: Any, device: Any) -> dict[str, Any]:
    rows = _identity_rows(public, "development")[:BATCH_ROWS]
    torch.manual_seed(int(seed))
    m0 = fxm.build_model(seed, CONTROL, torch=torch, device=device)
    h0 = _model_sha(m0)
    torch.manual_seed(int(seed))
    m3 = fxm.build_model(seed, TREATMENT, torch=torch, device=device)
    h3 = _model_sha(m3)
    if h0 != h3:
        raise RuntimeError(f"matched initialization mismatch seed={seed}: {h0} != {h3}")
    tokens, segments, _, _, _ = train.build_batch(rows, proto.EXPERIMENT_A, CONTROL, torch=torch, device=device)
    from v5_model.core import packed_layout
    positions, mask = packed_layout(segments, torch_module=torch)
    with torch.no_grad():
        y0 = m0(tokens, positions, mask).float()
        y3 = m3(tokens, positions, mask).float()
    max_abs = float(torch.max(torch.abs(y0 - y3)).item())
    del m0, m3
    torch.cuda.empty_cache()
    if max_abs != 0.0:
        raise RuntimeError(f"forward-equivalence failure seed={seed}: max_abs={max_abs}")
    return {"seed": seed, "matched_initial_model_sha256": h0, "forward_max_abs_difference": max_abs, "passed": True}


def _train_lane(*, seed: int, arm: str, out: Path, public: Mapping[str, Any], train: Any, proto: Any, fxm: Any, torch: Any, device: Any) -> dict[str, Any]:
    lane = out / "TRAIN" / f"SEED_{seed}" / arm
    lane.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    result = train.train_arm(
        experiment=proto.EXPERIMENT_A,
        arm=arm,
        seed_bundle=seed,
        surface=public,
        out_dir=lane,
        torch=torch,
        device=device,
        engineering_only=True,
        a_updates_override=UPDATES,
        progress=lambda msg: print(msg, flush=True),
    )
    label = f"CAL{seed}"
    checkpoint = lane / proto.EXPERIMENT_A / arm / label / "resume.pt"
    if not checkpoint.exists():
        raise RuntimeError(f"checkpoint missing: {checkpoint}")
    model = train.load_model_for_evaluation(
        checkpoint,
        experiment=proto.EXPERIMENT_A,
        arm=arm,
        seed_bundle=seed,
        data_manifest_sha256=str(public["sha256"]),
        torch=torch,
        device=device,
    )
    rows = _identity_rows(public, "development")
    greedy = _greedy(model, rows, train=train, proto=proto, torch=torch, device=device)
    teacher = _teacher_forced(model, rows, train=train, proto=proto, torch=torch, device=device)
    grad = _gradient_decomposition(model, rows[:BATCH_ROWS], train=train, proto=proto, fxm=fxm, torch=torch, device=device)
    summary = {
        "seed": seed,
        "arm": arm,
        "updates": int(result["updates"]),
        "formation": result["formation"],
        "clip_fraction": result.get("clip_fraction"),
        "timing": result.get("timing", {}),
        "wall_seconds": float(result.get("wall_seconds", time.monotonic() - started)),
        "checkpoint": str(checkpoint),
        "greedy_identity": greedy,
        "teacher_forced_identity": teacher,
        "gradient_diagnostic": grad,
    }
    _atomic_json(lane / "PILOT_LANE_RESULT.json", summary)
    del model
    torch.cuda.empty_cache()
    return summary


def _pairwise(control: Mapping[str, Any], treatment: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "seed": int(control["seed"]),
        "formation_auc_control": float(control["formation"]["formation_auc"]),
        "formation_auc_treatment": float(treatment["formation"]["formation_auc"]),
        "formation_auc_delta_treatment_minus_control": float(treatment["formation"]["formation_auc"]) - float(control["formation"]["formation_auc"]),
        "endpoint_control": float(control["formation"]["endpoint"]),
        "endpoint_treatment": float(treatment["formation"]["endpoint"]),
        "endpoint_delta_treatment_minus_control": float(treatment["formation"]["endpoint"]) - float(control["formation"]["endpoint"]),
        "teacher_forced_delta": float(treatment["teacher_forced_identity"]["target_token_accuracy"]) - float(control["teacher_forced_identity"]["target_token_accuracy"]),
        "gradient_output_to_input_ratio_control": float(control["gradient_diagnostic"]["output_to_input_gradient_norm_ratio"]),
        "gradient_output_to_input_ratio_treatment": float(treatment["gradient_diagnostic"]["output_to_input_gradient_norm_ratio"]),
        "gradient_cosine_control": float(control["gradient_diagnostic"]["input_output_gradient_cosine"]),
        "gradient_cosine_treatment": float(treatment["gradient_diagnostic"]["input_output_gradient_cosine"]),
    }


def _decision(pairs: list[Mapping[str, Any]]) -> dict[str, Any]:
    auc = [float(p["formation_auc_delta_treatment_minus_control"]) for p in pairs]
    endpoint = [float(p["endpoint_delta_treatment_minus_control"]) for p in pairs]
    t0 = [float(p["endpoint_control"]) for p in pairs]
    t3 = [float(p["endpoint_treatment"]) for p in pairs]

    if len(pairs) != 2:
        return {"decision": "DO_NOT_RUN_FULL_TIE_ROLE", "expensive_experiment_worth_running": False, "reason": "The pilot requires exactly two fresh matched pairs; incomplete evidence is a NO-GO."}

    ceiling = max(t0 + t3) >= MAX_ENDPOINT_FOR_DISCRIMINATION
    strong = all(x >= MIN_AUC_DELTA for x in auc) and all(x >= MIN_ENDPOINT_DELTA for x in endpoint)
    no_negative = all(x >= 0.0 for x in endpoint)

    if not ceiling and strong and no_negative:
        return {
            "decision": "RUN_FULL_TIE_ROLE",
            "expensive_experiment_worth_running": True,
            "reason": "Both fresh matched seeds show a large positive treatment effect on both formation AUC and final identity exactness, while the regime remains below ceiling. The full preregistered TIE-ROLE frontier is worth its compute.",
        }

    if ceiling:
        return {
            "decision": "DO_NOT_RUN_FULL_TIE_ROLE",
            "expensive_experiment_worth_running": False,
            "reason": "At least one pilot arm is too close to the identity ceiling for the existing endpoint-gap design to discriminate the mechanism reliably. Recalibrate the exposure/metric before spending on the full frontier.",
        }

    if any(x < 0.0 for x in endpoint):
        return {
            "decision": "DO_NOT_RUN_FULL_TIE_ROLE",
            "expensive_experiment_worth_running": False,
            "reason": "The balanced treatment loses on final identity in at least one fresh matched seed. The evidence is not strong enough to justify the full 24-arm frontier.",
        }

    if max(auc) < MIN_AUC_DELTA or max(endpoint) < MIN_ENDPOINT_DELTA:
        return {
            "decision": "DO_NOT_RUN_FULL_TIE_ROLE",
            "expensive_experiment_worth_running": False,
            "reason": "The pilot does not show a large enough reproducible effect to justify the full 24-arm campaign. Preserve the pilot and test a smaller bottleneck instead.",
        }

    return {
        "decision": "DO_NOT_RUN_FULL_TIE_ROLE",
        "expensive_experiment_worth_running": False,
        "reason": "Pilot evidence is mixed or below the predeclared compute-justification threshold. Ambiguous evidence is deliberately treated as NO-GO.",
    }


def _package(out: Path) -> dict[str, Any]:
    results_zip = out / "TIE_ROLE_PILOT_001_RESULTS.zip"
    checkpoints_zip = out / "TIE_ROLE_PILOT_001_CHECKPOINTS.zip"
    for p in (results_zip, checkpoints_zip):
        if p.exists():
            p.unlink()

    with zipfile.ZipFile(results_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(out.rglob("*")):
            if not p.is_file() or p.name in {results_zip.name, checkpoints_zip.name} or p.name == "resume.pt" or p.suffix == ".sha256":
                continue
            zf.write(p, p.relative_to(out))

    with zipfile.ZipFile(checkpoints_zip, "w", zipfile.ZIP_STORED) as zf:
        for p in sorted(out.rglob("resume.pt")):
            zf.write(p, p.relative_to(out))
        for p in sorted(out.rglob("CHECKPOINT_RECEIPT.json")):
            zf.write(p, p.relative_to(out))
        for p in sorted(out.rglob("PILOT_LANE_RESULT.json")):
            zf.write(p, p.relative_to(out))

    receipts = {}
    for name, p in (("results", results_zip), ("checkpoints", checkpoints_zip)):
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        sha_path = Path(str(p) + ".sha256")
        sha_path.write_text(f"{digest}  {p.name}\n", encoding="utf-8")
        receipts[name] = {"path": str(p), "sha256": digest, "bytes": p.stat().st_size}
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
    if prereg.get("schema") != "anra.tie-role-pilot-preregistration/v1":
        raise RuntimeError("unexpected pilot preregistration schema")
    if prereg.get("relationship_to_s5", {}).get("frozen_science_commit") != SCIENCE_COMMIT:
        raise RuntimeError("pilot science commit mismatch")
    if _git_head(repo) != SCIENCE_COMMIT:
        raise RuntimeError(f"execution tree must be frozen S5 science commit {SCIENCE_COMMIT}")

    import sys
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required; select a Colab T4 GPU")
    device = torch.device("cuda:0")

    from v5_data.corpus_loading import _load_tokenizer
    from v5_experiments.formation_mux_surface_v5 import build_public_surface
    from v5_experiments import tie_role_protocol_v1 as proto
    from anra_v5 import tie_role_model_v1 as fxm
    from anra_v5 import tie_role_train_v1 as train
    from anra_v5 import formation_mux_train_v2 as base_train

    # Rebind the audited base transaction to the TIE-ROLE protocol/model.
    train._bind()
    if base_train.proto is not proto or base_train.fxm is not fxm:
        raise RuntimeError("TIE-ROLE training binding failed")

    tokenizer, _ = _load_tokenizer(repo)
    public = build_public_surface(seed=73011, tokenizer=tokenizer)
    if str(public["sha256"]) != PUBLIC_SURFACE_SHA:
        raise RuntimeError(f"public surface drift: {public['sha256']} != {PUBLIC_SURFACE_SHA}")
    if "sealed" in public.get("splits", {}):
        raise RuntimeError("SEALED_FIREWALL_BREACH")

    # Speed optimization: training remains the full six-family mixture;
    # development evaluation is only the preregistered identity primary.
    original_eval = base_train.evaluate_development
    def identity_eval(model: Any, dev_rows: list[Mapping[str, Any]], experiment: str, arm: str, *, torch: Any, device: Any) -> dict[str, Any]:
        rows = [r for r in dev_rows if r.get("family") == "identity"]
        return original_eval(model, rows, experiment, arm, torch=torch, device=device)
    base_train.evaluate_development = identity_eval

    binding = {
        "schema": SCHEMA,
        "science_commit": SCIENCE_COMMIT,
        "public_surface_sha256": public["sha256"],
        "experiment": EXPERIMENT,
        "control": CONTROL,
        "treatment": TREATMENT,
        "seeds": list(SEEDS),
        "updates": UPDATES,
        "official_science": False,
        "sealed_scores_used": False,
        "sealed_rows_persisted": False,
        "preregistration_sha256": hashlib.sha256(args.prereg.read_bytes()).hexdigest(),
    }
    _atomic_json(out / "EXECUTION_BINDING.json", binding)

    for seed in SEEDS:
        eq = _preflight_forward_equivalence(seed=seed, public=public, train=train, proto=proto, fxm=fxm, torch=torch, device=device)
        _atomic_json(out / "PREFLIGHT" / f"FORWARD_EQUIVALENCE_{seed}.json", eq)

    rows_by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for seed in SEEDS:
        rows_by_seed[seed] = {}
        for arm in (CONTROL, TREATMENT):
            rows_by_seed[seed][arm] = _train_lane(
                seed=seed, arm=arm, out=out, public=public, train=train,
                proto=proto, fxm=fxm, torch=torch, device=device,
            )

    pairs = [_pairwise(rows_by_seed[s][CONTROL], rows_by_seed[s][TREATMENT]) for s in SEEDS]
    decision = _decision(pairs)

    result = {
        "schema": SCHEMA,
        "status": "COMPLETE",
        "experiment": "TIE-ROLE-PILOT-001",
        "decision_target": "TIE-ROLE-FRONTIER-001",
        "science_commit": SCIENCE_COMMIT,
        "public_surface_sha256": public["sha256"],
        "sealed_scores_used": False,
        "sealed_rows_persisted": False,
        "pairs": pairs,
        "decision": decision["decision"],
        "expensive_experiment_worth_running": decision["expensive_experiment_worth_running"],
        "reason": decision["reason"],
        "thresholds": {
            "pilot_min_auc_delta": MIN_AUC_DELTA,
            "pilot_min_endpoint_delta": MIN_ENDPOINT_DELTA,
            "max_endpoint_for_discrimination": MAX_ENDPOINT_FOR_DISCRIMINATION,
        },
        "wall_seconds": time.monotonic() - started,
        "next_action": (
            "Run the frozen TIE-ROLE-FRONTIER-001 campaign unchanged."
            if decision["expensive_experiment_worth_running"]
            else "Do not spend the full frontier compute; preserve this result and choose the smallest next bottleneck experiment."
        ),
    }
    _atomic_json(out / "TIE_ROLE_PILOT_RESULT.json", result)
    packages = _package(out)
    _atomic_json(out / "PACKAGE_RECEIPT.json", packages)

    print("\n" + "#" * 78)
    print("TIE-ROLE-PILOT-001 COMPLETE")
    print("NEXT EXPENSIVE EXPERIMENT:", result["decision"])
    print("WORTH RUNNING:", result["expensive_experiment_worth_running"])
    print("REASON:", result["reason"])
    print("WALL HOURS:", round(result["wall_seconds"] / 3600.0, 3))
    print("RESULTS ZIP:", packages["results"]["path"])
    print("CHECKPOINTS ZIP:", packages["checkpoints"]["path"])
    print("#" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
