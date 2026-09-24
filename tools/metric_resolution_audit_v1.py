"""METRIC-RES-001: checkpoint-only identity-resolution audit.

Re-scores the preserved FORMATION-MUX-001 S5 checkpoints at token resolution
so the three frozen CS-MECH-002 mechanism contrasts can be read on an
instrument with dynamic range, and so the branch learns whether identity is
forming-but-unmeasured or genuinely absent.

Architecture (why this is not a hack):
- No training. No optimizer reconstruction. No backward. Weights are loaded
  read-only through the audited model-only restore path.
- The subject is bit-identical to the executed S5 run. Any separation between
  arms here is attributable to instrument resolution alone, which is exactly
  the causal claim.
- Termination is a positive control: S5 measured 1.00 exact, so a valid
  instrument must reproduce it. Failure to reproduce invalidates the run.
- Sealed rows are never loaded, regenerated, scored, or persisted.
- Colab's Drive mount drops around the 10h mark with errno 107, so results
  are written to local disk first and synced with bounded retry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

SCHEMA = "anra.metric-resolution-audit/v1"
PREREG = "docs/cymek/experiments/METRIC-RES-001/PREREGISTRATION.json"
SURFACE_SHA = "f1d5200bd05bc28ede97af114b49f616ca72b24af74b7fc530a2cb084db4259c"
SCIENCE_COMMIT = "c15ad8beb409537db42d075684ea54847a074ebd"
CONTROL = "M0_STANDARD"
CONTRASTS = (
    ("M1_EXTRA_NO_DECAY", "extra_row_weight_decay"),
    ("M2_EXTRA_FROZEN", "extra_row_trainability"),
    ("M3_EXTRA_FROZEN_MASKED", "extra_row_denominator"),
)
TERMINATION_CONTROL_MIN = 0.90
DECISION_ORDER = (
    "INSTRUMENT_INVALID",
    "FORMING_BUT_UNMEASURED",
    "OPTIMIZATION_CHOKED",
    "OUTPUT_COMPETITION",
    "MIXTURE_OR_DATA_LIMITED",
    "GENUINELY_ABSENT",
    "INCONCLUSIVE",
)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _prereg(repo: Path) -> dict[str, Any]:
    path = repo / PREREG
    if not path.exists():
        raise RuntimeError(f"preregistration missing: {PREREG}")
    body = json.loads(path.read_text(encoding="utf-8"))
    if body.get("schema") != "anra.metric-resolution-audit-preregistration/v1":
        raise RuntimeError("unexpected preregistration schema")
    if body.get("written_before_any_audit_outcome") is not True:
        raise RuntimeError("preregistration is not prospective")
    if str(body["fixed_variables"]["surface_sha256"]) != SURFACE_SHA:
        raise RuntimeError("preregistration surface identity mismatch")
    declared = tuple(k for k in body["decision_tree"] if k != "note")
    if declared != DECISION_ORDER:
        # The preregistration promises a first-match order. If someone reorders
        # or adds a branch there, the code must be updated in the same change
        # or the run stops rather than silently applying a different tree.
        raise RuntimeError(
            "preregistration decision order does not match the implemented order: "
            f"declared={declared} implemented={DECISION_ORDER}"
        )
    return body


def _verify_surface(public: Mapping[str, Any]) -> None:
    splits = public.get("splits", {})
    if "sealed" in splits:
        raise RuntimeError("SEALED_FIREWALL_BREACH: audit surface contains sealed rows")
    if str(public.get("sha256")) != SURFACE_SHA:
        raise RuntimeError(f"public surface drift: {public.get('sha256')} != {SURFACE_SHA}")


def greedy_with_lcp(model: Any, rows: list[Mapping[str, Any]], *, torch: Any,
                    device: Any, max_tokens: int = 32) -> dict[str, Any]:
    """Candidate-free greedy generation reporting longest-common-prefix.

    The endpoint S5 used is all-or-nothing sequence exact. A model that emits
    the first two of three answer tokens scores 0.0 there. LCP exposes that
    partial realization without changing the subject.
    """
    from v5_model.core import packed_layout

    was_training = model.training
    model.eval()
    exact = complete = stopped_count = 0
    lcp_sum = 0.0
    expected_sum = 0
    per_row: list[dict[str, Any]] = []
    with torch.no_grad():
        for index, row in enumerate(rows):
            prompt = [2, *[int(x) for x in row["prompt_ids"]]]
            expected = [int(x) for x in row["answer_ids"] if int(x) != 3]
            generated: list[int] = []
            stopped = False
            for _ in range(max_tokens):
                current = torch.tensor([prompt + generated], dtype=torch.long, device=device)
                positions, mask = packed_layout(torch.zeros_like(current), torch_module=torch)
                nxt = int(torch.argmax(model(current, positions, mask)[0, -1]).item())
                if nxt == 3:
                    stopped = True
                    break
                if nxt == 0:
                    break
                generated.append(nxt)
            lcp = 0
            for got, want in zip(generated, expected):
                if got != want:
                    break
                lcp += 1
            lcp_sum += lcp
            expected_sum += max(len(expected), 1)
            if generated == expected:
                exact += 1
                if stopped:
                    complete += 1
            stopped_count += int(stopped)
            per_row.append({
                "index": index,
                "expected": expected,
                "generated": generated,
                "lcp": lcp,
                "stopped": stopped,
                "exact": generated == expected,
                "complete_exact": generated == expected and stopped,
            })
    if was_training:
        model.train()
    n = max(len(rows), 1)
    return {
        "content_exact": exact / n,
        "complete_exact_with_valid_stop": complete / n,
        "eos_rate": stopped_count / n,
        "lcp_ratio": lcp_sum / max(expected_sum, 1),
        "mean_lcp_tokens": lcp_sum / n,
        "n": len(rows),
        "per_row": per_row,
    }


def _family_rows(public: Mapping[str, Any], family: str) -> list[dict[str, Any]]:
    rows = [dict(r) for r in public["splits"]["development"] if r.get("family") == family]
    if not rows:
        raise RuntimeError(f"development surface missing family {family}")
    return rows


def audit_arm(*, checkpoint: Path, arm: str, seed_label: str, public: dict[str, Any],
              repo: Path, device: Any, torch: Any, out: Path) -> dict[str, Any]:
    from anra_v5 import formation_mux_train_v5 as train
    from anra_v5 import formation_diag as diag
    from v5_experiments import formation_mux_protocol_v5 as proto

    seed_bundle = proto.SEED_BUNDLES[int(seed_label[1:]) - 1]
    if not checkpoint.exists():
        raise RuntimeError(f"checkpoint missing: {checkpoint}")
    model = train.load_model_for_evaluation(
        checkpoint,
        experiment=proto.EXPERIMENT_A,
        arm=arm,
        seed_bundle=seed_bundle,
        data_manifest_sha256=SURFACE_SHA,
        torch=torch,
        device=device,
    )
    identity_rows = _family_rows(public, "identity")
    termination_rows = _family_rows(public, "termination")
    try:
        identity_gen = greedy_with_lcp(model, identity_rows, torch=torch, device=device)
        termination_gen = greedy_with_lcp(model, termination_rows, torch=torch, device=device)
        teacher = diag.teacher_forced_diagnostics(model, identity_rows, torch=torch, device=device)
        ce = diag.ce_by_family(model, list(public["splits"]["development"]), torch=torch, device=device)
        rescue = diag.full_vs_shared_rescue(model, identity_rows, torch=torch, device=device)
        body = torch.load(checkpoint, map_location="cpu", weights_only=False)
        updates = int(body.get("updates", 0))
        clip_events = int(body.get("clip_events", 0))
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    receipt = {
        "schema": "anra.metric-resolution-arm/v1",
        "experiment": proto.EXPERIMENT_A,
        "arm": arm,
        "seed_label": seed_label,
        "seed_bundle": seed_bundle,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha_file(checkpoint),
        "updates": updates,
        "processed_tokens": int(body.get("processed_tokens", 0)),
        "clip_fraction": clip_events / max(updates, 1),
        "identity_generation": identity_gen,
        "termination_generation": termination_gen,
        "identity_teacher_forced": teacher,
        "ce_by_family": ce["ce_by_family"],
        "output_competition_rescue": rescue,
    }
    _atomic_json(out / f"ARM_{arm}_{seed_label}.json", receipt)
    print(
        f"ARM {arm}/{seed_label} tok_acc={teacher['token_accuracy']} "
        f"exact={identity_gen['complete_exact_with_valid_stop']} "
        f"lcp={identity_gen['lcp_ratio']} term={termination_gen['complete_exact_with_valid_stop']}",
        flush=True,
    )
    return receipt


def decide(receipts: list[dict[str, Any]], prereg: dict[str, Any]) -> dict[str, Any]:
    control = [r for r in receipts if r["arm"] == CONTROL]
    if not control:
        raise RuntimeError("no control-arm receipts")
    token_acc = _mean([float(r["identity_teacher_forced"]["token_accuracy"]) for r in control])
    exact = _mean([float(r["identity_generation"]["complete_exact_with_valid_stop"]) for r in control])
    lcp = _mean([float(r["identity_generation"]["lcp_ratio"]) for r in control])
    rescue = _mean([float(r["output_competition_rescue"]["rescue"]) for r in control])
    term = _mean([float(r["termination_generation"]["complete_exact_with_valid_stop"]) for r in control])
    clip = _mean([float(r["clip_fraction"]) for r in control])
    measured = {
        "mean_identity_token_accuracy": round(token_acc, 4),
        "mean_identity_exact_valid_eos": round(exact, 4),
        "mean_identity_lcp_ratio": round(lcp, 4),
        "mean_shared_only_rescue": round(rescue, 4),
        "mean_termination_exact_valid_eos": round(term, 4),
        "mean_control_clip_fraction": round(clip, 4),
        "control_seeds": len(control),
    }
    if term < TERMINATION_CONTROL_MIN:
        decision = "INSTRUMENT_INVALID"
    elif token_acc >= 0.80 and exact < 0.30 and lcp >= 0.50:
        decision = "FORMING_BUT_UNMEASURED"
    elif token_acc < 0.30 and clip >= 0.99:
        decision = "OPTIMIZATION_CHOKED"
    elif rescue >= 0.10 and token_acc < 0.80:
        decision = "OUTPUT_COMPETITION"
    elif 0.30 <= token_acc < 0.80:
        decision = "MIXTURE_OR_DATA_LIMITED"
    elif token_acc < 0.30:
        decision = "GENUINELY_ABSENT"
    else:
        decision = "INCONCLUSIVE"
    return {
        "schema": "anra.metric-resolution-decision/v1",
        "decision": decision,
        "measured": measured,
        "rule_source": PREREG,
    }


def readjudicate(receipts: list[dict[str, Any]]) -> dict[str, Any]:
    by = {(r["arm"], r["seed_label"]): r for r in receipts}
    labels = sorted({r["seed_label"] for r in receipts})
    rows = []
    for treatment, name in CONTRASTS:
        deltas = {}
        for label in labels:
            t = by.get((treatment, label))
            c = by.get((CONTROL, label))
            if t is None or c is None:
                continue
            deltas[label] = (
                float(t["identity_teacher_forced"]["token_accuracy"])
                - float(c["identity_teacher_forced"]["token_accuracy"])
            )
        if not deltas:
            continue
        values = list(deltas.values())
        mean = _mean(values)
        signs = [1 if v > 0 else (-1 if v < 0 else 0) for v in values]
        pos, neg = signs.count(1), signs.count(-1)
        consistent = pos >= 3 or neg >= 3
        if mean >= 0.10 and pos >= 3:
            verdict = "SUCCESS"
        elif mean <= -0.10 and neg >= 3:
            verdict = "REVERSE_EFFECT"
        elif abs(mean) < 0.05:
            verdict = "NULL"
        else:
            verdict = "INCONCLUSIVE"
        rows.append({
            "contrast": name,
            "treatment": treatment,
            "control": CONTROL,
            "paired_token_accuracy_deltas": deltas,
            "mean_delta": round(mean, 4),
            "sign_consistent_3of4": bool(consistent),
            "verdict": verdict,
        })
    return {"schema": "anra.metric-resolution-contrast/v1", "contrasts": rows}


def package(out: Path) -> dict[str, Any]:
    bundle = out.parent / "METRIC_RES_001_RESULTS.zip"
    if bundle.exists():
        bundle.unlink()
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out.rglob("*.json")):
            zf.write(path, path.relative_to(out.parent))
    digest = _sha_file(bundle)
    Path(str(bundle) + ".sha256").write_text(f"{digest}  {bundle.name}\n", encoding="utf-8")
    return {"path": str(bundle), "sha256": digest, "bytes": bundle.stat().st_size}


def sync_to(target: Path | None, out: Path) -> dict[str, Any]:
    """Bounded-retry copy out of the runtime filesystem.

    Colab's Drive mount drops around the 10h mark with errno 107 even while
    the VM is healthy. Writing locally first and retrying the copy means a
    mount blip cannot destroy completed receipts.
    """
    if target is None:
        return {"synced": False, "reason": "no target supplied"}
    target.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for attempt in range(4):
        try:
            shutil.copytree(out, target, dirs_exist_ok=True)
            return {"synced": True, "attempts": attempt + 1, "target": str(target)}
        except OSError as exc:
            errors.append(f"attempt {attempt + 1}: {exc}")
            time.sleep(2.0 * (attempt + 1))
    return {"synced": False, "errors": errors, "target": str(target)}


def _aggregate(*, repo: Path, out: Path, started: float,
               sync_target: Path | None, skip_package: bool) -> int:
    """CPU-only stage: apply the frozen decision tree and package.

    Runs on a free Accelerator=None session. Torch is not imported at all, so
    this stage cannot touch a GPU context or consume quota.
    """
    prereg = _prereg(repo)
    receipts = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(out.glob("ARM_*.json"))
    ]
    if not receipts:
        print("METRIC-RES AGGREGATE: no ARM_*.json receipts found", file=sys.stderr, flush=True)
        return 4
    shards = [json.loads(p.read_text(encoding="utf-8"))
              for p in sorted(out.glob("SHARD_RECEIPT.json"))]
    if shards and any(s.get("training_steps") != 0 for s in shards):
        print("METRIC-RES AGGREGATE: shard receipt claims training steps; fail closed",
              file=sys.stderr, flush=True)
        return 4
    if shards and any(s.get("sealed_touched") is not False for s in shards):
        print("METRIC-RES AGGREGATE: shard receipt claims sealed touch; fail closed",
              file=sys.stderr, flush=True)
        return 4

    decision = decide(receipts, prereg)
    readjudication = readjudicate(receipts)
    result = {
        "schema": "anra.metric-resolution-result/v1",
        "experiment": "METRIC-RES-001",
        "science_reference_commit": SCIENCE_COMMIT,
        "preregistration": PREREG,
        "preregistration_sha256": _sha_file(repo / PREREG),
        "surface_sha256": SURFACE_SHA,
        "arms_audited": sorted({r["arm"] for r in receipts}),
        "seed_labels_audited": sorted({r["seed_label"] for r in receipts}),
        "arm_receipts": len(receipts),
        "decision": decision,
        "contrast_readjudication": readjudication,
        "training_steps": 0,
        "sealed_touched": False,
        "claim_ceiling": prereg["claim_ceiling"],
        "wall_seconds": time.monotonic() - started,
    }
    _atomic_json(out / "DECISION.json", result)
    sync = sync_to(sync_target, out)
    _atomic_json(out / "SYNC_RECEIPT.json", sync)
    packaged = None if skip_package else package(out)
    if packaged is not None:
        _atomic_json(out / "PACKAGE_RECEIPT.json", packaged)
    print("METRIC-RES DECISION:", decision["decision"], flush=True)
    print("METRIC-RES MEASURED:", json.dumps(decision["measured"]), flush=True)
    for row in readjudication["contrasts"]:
        print(
            f"CONTRAST {row['contrast']}: {row['verdict']} "
            f"mean_delta={row['mean_delta']} consistent={row['sign_consistent_3of4']}",
            flush=True,
        )
    if packaged is not None:
        print("METRIC-RES PACKAGE:", json.dumps(packaged), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--checkpoints", type=Path, required=True)
    p.add_argument("--arms", default=",".join([CONTROL, *(c[0] for c in CONTRASTS)]))
    p.add_argument("--seed-labels", default="S1,S2,S3,S4")
    p.add_argument("--device", default="cuda")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--sync-target", type=Path, default=None)
    p.add_argument("--skip-package", action="store_true")
    p.add_argument("--aggregate-only", action="store_true")
    args = p.parse_args(argv)

    repo = args.repo.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    if args.aggregate_only:
        return _aggregate(repo=repo, out=out, started=started,
                          sync_target=args.sync_target,
                          skip_package=args.skip_package)

    try:
        prereg = _prereg(repo)
        sys.path.insert(0, str(repo))
        os.chdir(repo)
        import torch
        from v5_experiments.formation_mux_surface_v5 import load_public_surface

        public = load_public_surface(repo / "artifacts" / "METRIC_RES_001" / "PUBLIC_SURFACE_MANIFEST.json") \
            if (repo / "artifacts" / "METRIC_RES_001" / "PUBLIC_SURFACE_MANIFEST.json").exists() \
            else load_public_surface(args.checkpoints.parent / "PUBLIC_SURFACE_MANIFEST.json")
        _verify_surface(public)

        device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; select GPU runtime")

        arms = [a.strip() for a in args.arms.split(",") if a.strip()]
        labels = [s.strip() for s in args.seed_labels.split(",") if s.strip()]
        mine = [
            (arm, label)
            for arm in arms
            for label in labels
            if labels.index(label) % args.num_shards == args.shard_index
        ]
        if not mine:
            print("METRIC-RES shard has no assigned arms", flush=True)
            return 0
        receipts = []
        for arm, label in mine:
            checkpoint = args.checkpoints / "CS-MECH-002" / arm / label / "resume.pt"
            receipts.append(audit_arm(
                checkpoint=checkpoint, arm=arm, seed_label=label, public=public,
                repo=repo, device=device, torch=torch, out=out,
            ))
        _atomic_json(out / "SHARD_RECEIPT.json", {
            "schema": "anra.metric-resolution-shard/v1",
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "arms": [a for a, _ in mine],
            "seed_labels": sorted({l for _, l in mine}),
            "device": str(device),
            "wall_seconds": time.monotonic() - started,
            "training_steps": 0,
            "sealed_touched": False,
        })
        print("METRIC-RES ARM PASS COMPLETE", flush=True)
        return 0
    except Exception as exc:
        _atomic_json(out / "METRIC_RES_FAILURE.json", {
            "schema": "anra.metric-resolution-failure/v1",
            "exception": type(exc).__name__,
            "message": str(exc),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        print(f"METRIC-RES FAIL-CLOSED: {exc}", file=sys.stderr, flush=True)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
