from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
BASE_RUNNER_PATH = REPO / "experiments" / "ARK-011" / "run_ark011.py"
PLAN_COMMIT_SHA = "c992a8c864366c209ee4cd764836dc47a6fc8622"
BASE_ARK011_RUNNER_COMMIT = "baec1687ed43d4c0dc420b4f329df5cf266d8645"
RESULTS_DIR = Path("/content/arkenstone_ark011s_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SESSION_START = time.time()
RUNNER_HEAD = "UNKNOWN"
RUNNER_SHA256 = "UNKNOWN"


def _load_base():
    spec = importlib.util.spec_from_file_location("ark011_base", BASE_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load ARK-011 base runner")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


base = _load_base()


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_json(value) -> str:
    return sha_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))


def git_head() -> str:
    return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()


def minutes_used() -> float:
    return (time.time() - SESSION_START) / 60.0


def minutes_left(budget: float) -> float:
    return budget - minutes_used()


def save_json(name: str, payload: dict) -> Path:
    out = dict(payload)
    out["experiment_id"] = out.get("experiment_id", "ARK-011S")
    out["screen_plan_commit_sha"] = PLAN_COMMIT_SHA
    out["base_ark011_runner_commit"] = BASE_ARK011_RUNNER_COMMIT
    out["runner_commit_sha"] = RUNNER_HEAD
    out["runner_source_sha256"] = RUNNER_SHA256
    out["device"] = str(base.DEVICE) if base.DEVICE is not None else "uninitialized"
    out["torch"] = torch.__version__
    body = dict(out)
    body.pop("receipt_sha256", None)
    out["receipt_sha256"] = sha_json(body)
    path = RESULTS_DIR / name
    path.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    print("saved:", path, flush=True)
    return path


def package_results(download: bool) -> Path:
    out = RESULTS_DIR / "ARKENSTONE_ARK011S_RESULTS.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(RESULTS_DIR.glob("*.json")):
            zf.write(p, p.name)
    print("RESULT ZIP:", out, flush=True)
    if download:
        try:
            from google.colab import files
            files.download(str(out))
        except Exception as exc:
            print("auto-download skipped:", repr(exc), flush=True)
    return out


def summarize(events: list[dict]) -> dict:
    qualified = [e for e in events if e.get("status") == "FORK_EXECUTED" and e.get("sealed_primary_qualified")]
    positives = 0
    reverses = 0
    low_materially_worse = 0
    rows = []
    for e in qualified:
        h = e["HIGH_CONTINUE"]["sealed_retention"]
        l = e["SWITCH_LOW"]["sealed_retention"]
        high_rec = bool(h.get("recollapsed"))
        low_rec = bool(l.get("recollapsed"))
        if high_rec and not low_rec:
            positives += 1
        if (not high_rec) and low_rec:
            reverses += 1
        if float(l.get("AREA", 0.0)) + 0.05 < float(h.get("AREA", 0.0)):
            low_materially_worse += 1
        rows.append({
            "continuation_seed": e["continuation_seed"],
            "high_recollapsed": high_rec,
            "low_recollapsed": low_rec,
            "high_RET90": h.get("RET90"),
            "low_RET90": l.get("RET90"),
            "high_AREA": h.get("AREA"),
            "low_AREA": l.get("AREA"),
            "high_FINAL": h.get("FINAL"),
            "low_FINAL": l.get("FINAL"),
        })
    if not qualified:
        verdict = "SCREEN_NO_EVENT"
    elif reverses > 0 or low_materially_worse > 0:
        verdict = "SCREEN_DIRECTION_NEGATIVE"
    elif positives > 0:
        verdict = "SCREEN_DIRECTION_POSITIVE"
    else:
        verdict = "SCREEN_NO_EVENT"
    return {
        "sealed_qualified_forks": len(qualified),
        "positive_discordants": positives,
        "reverse_discordants": reverses,
        "low_materially_worse_area_count": low_materially_worse,
        "rows": rows,
        "verdict": verdict,
        "claim_level": "SCREENING_NOT_REPLICATION",
    }


def smoke_test() -> None:
    manifest = base.load_manifest()
    train = [(p, a) for p, a in manifest["train"]]
    test = [(p, a) for p, a in manifest["test"]]
    control, sealed, split_manifest = base.build_control_sealed_split(test)
    assert control and sealed and not (set(control) & set(sealed))
    idx = base.generate_continuation_indices(2702, 16, 4, len(train))
    assert base.order_sha256(idx) == base.order_sha256(base.generate_continuation_indices(2702, 16, 4, len(train)))
    vocab = base.CompactVocab()
    torch.manual_seed(909)
    torch.cuda.manual_seed_all(909)
    model = base.Micro(vocab.size, 128).to(base.dev())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    before = base.parameter_sha(model)
    base.train_step(model, opt, vocab, train[:8])
    after = base.parameter_sha(model)
    assert before != after
    snap = base.snapshot_state(model, opt)
    _, restored, _ = base.load_fork(snap, 1e-3)
    assert base.model_state_equal(model.state_dict(), restored.state_dict())
    save_json("SMOKE_TEST.json", {
        "status": "PASS",
        "source_manifest_sha256": base.CANONICAL_T2_SHA,
        "control_sha256": split_manifest["control_sha256"],
        "sealed_sha256": split_manifest["sealed_sha256"],
        "snapshot_reload_exact": True,
        "one_step_parameter_mutation": True,
    })
    print("ARK-011S SMOKE TEST PASS", flush=True)


def full_screen(budget_minutes: float) -> dict:
    manifest = base.load_manifest()
    train = [(p, a) for p, a in manifest["train"]]
    test = [(p, a) for p, a in manifest["test"]]
    control, sealed, split_manifest = base.build_control_sealed_split(test)
    save_json("ARK-011S_TASK_MANIFEST.json", split_manifest)

    print("\n=== ACQUIRE seed 909 ===", flush=True)
    acq = base.acquire(909, train, control, max_steps=28000, eval_every=200)
    acq_public = {k: v for k, v in acq.items() if k != "snapshot"}
    events: list[dict] = []
    save_json("ARK-011S_PARTIAL.json", {"acquisition": acq_public, "events": events, "minutes_used": minutes_used()})
    if acq.get("status") != "ACQUIRED":
        result = {
            "status": "BLOCKED_BY_ACQUISITION",
            "acquisition": acq_public,
            "events": events,
            "summary": summarize(events),
            "minutes_used": minutes_used(),
        }
        save_json("ARK-011S_RESULT.json", result)
        return result

    for order_seed in [2702, 2703]:
        if minutes_left(budget_minutes) < 7.0:
            events.append({"continuation_seed": order_seed, "status": "BUDGET_BLOCKED_BEFORE_ORDER"})
            break

        print(f"\n=== SCREEN order {order_seed} ===", flush=True)
        indices = base.generate_continuation_indices(order_seed, 10000, 64, len(train))
        order_hash = base.order_sha256(indices)

        collapse = base.run_to_threshold(
            phase_name=f"SCREEN_COLLAPSE o{order_seed}", snapshot=acq["snapshot"], indices=indices,
            offset=0, max_steps=4000, train=train, control=control, lr=1e-3, bar=0.90, below=True,
        )
        if collapse.get("status") != "TRIGGERED":
            events.append({
                "continuation_seed": order_seed, "continuation_order_sha256": order_hash,
                "status": "NO_CONTROL_COLLAPSE",
                "collapse": {k: v for k, v in collapse.items() if k != "snapshot"},
            })
            save_json("ARK-011S_PARTIAL.json", {"acquisition": acq_public, "events": events, "minutes_used": minutes_used()})
            continue

        collapse_offset = int(collapse["confirmation_absolute_step"])
        recovery = base.run_to_threshold(
            phase_name=f"SCREEN_RECOVERY o{order_seed}", snapshot=collapse["snapshot"], indices=indices,
            offset=collapse_offset, max_steps=3000, train=train, control=control, lr=1e-3, bar=0.90, below=False,
        )
        if recovery.get("status") != "TRIGGERED":
            events.append({
                "continuation_seed": order_seed, "continuation_order_sha256": order_hash,
                "status": "NO_CONTROL_RECOVERY",
                "collapse": {k: v for k, v in collapse.items() if k != "snapshot"},
                "recovery": {k: v for k, v in recovery.items() if k != "snapshot"},
            })
            save_json("ARK-011S_PARTIAL.json", {"acquisition": acq_public, "events": events, "minutes_used": minutes_used()})
            continue

        recovery_offset = int(recovery["confirmation_absolute_step"])
        rvocab, rmodel, _ = base.load_fork(recovery["snapshot"], 1e-3)
        sealed_at_fork, _ = base.greedy_exact(rmodel, rvocab, sealed, base.dev())
        control_at_fork, _ = base.greedy_exact(rmodel, rvocab, control, base.dev())
        recovery_flat = base.flat_params(rmodel).detach().cpu()
        recovery_sha = base.parameter_sha(rmodel)
        del rmodel
        torch.cuda.empty_cache()

        high = base.run_retention_fork(
            snapshot=recovery["snapshot"], recovery_flat_cpu=recovery_flat, indices=indices,
            offset=recovery_offset, train=train, control=control, sealed=sealed, lr=1e-3,
            steps=3000, eval_every=200,
        )
        low = base.run_retention_fork(
            snapshot=recovery["snapshot"], recovery_flat_cpu=recovery_flat, indices=indices,
            offset=recovery_offset, train=train, control=control, sealed=sealed, lr=1e-5,
            steps=3000, eval_every=200,
        )
        events.append({
            "acquisition_seed": 909,
            "continuation_seed": order_seed,
            "continuation_order_sha256": order_hash,
            "status": "FORK_EXECUTED",
            "control_at_recovery_fork": control_at_fork,
            "sealed_at_recovery_fork": sealed_at_fork,
            "sealed_primary_qualified": sealed_at_fork >= 0.90,
            "recovery_parameter_sha256": recovery_sha,
            "collapse": {k: v for k, v in collapse.items() if k != "snapshot"},
            "recovery": {k: v for k, v in recovery.items() if k != "snapshot"},
            "HIGH_CONTINUE": high,
            "SWITCH_LOW": low,
        })
        save_json("ARK-011S_PARTIAL.json", {"acquisition": acq_public, "events": events, "minutes_used": minutes_used()})

    summary = summarize(events)
    result = {
        "status": "EXECUTED_SCREEN",
        "source_manifest_sha256": base.CANONICAL_T2_SHA,
        "acquisition": acq_public,
        "events": events,
        "summary": summary,
        "minutes_used": minutes_used(),
        "budget_minutes": budget_minutes,
    }
    save_json("ARK-011S_RESULT.json", result)
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--budget-minutes", type=float, default=25.0)
    p.add_argument("--expected-head", type=str, default="")
    return p.parse_args()


def main() -> int:
    global SESSION_START, RUNNER_HEAD, RUNNER_SHA256
    args = parse_args()
    SESSION_START = time.time()
    base.init_cuda()
    RUNNER_HEAD = git_head()
    RUNNER_SHA256 = sha_bytes(Path(__file__).read_bytes())
    if args.expected_head and RUNNER_HEAD != args.expected_head:
        raise RuntimeError(f"HEAD {RUNNER_HEAD} != expected {args.expected_head}")
    print("SCREEN PLAN:", PLAN_COMMIT_SHA, flush=True)
    print("RUNNER HEAD:", RUNNER_HEAD, flush=True)
    print("BUDGET MINUTES:", args.budget_minutes, flush=True)

    if args.smoke_test:
        smoke_test()
        package_results(download=False)
        return 0

    try:
        full_screen(float(args.budget_minutes))
        return 0
    except Exception as exc:
        save_json("FAILURE_RECEIPT.json", {
            "status": "FAILED",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "minutes_used": minutes_used(),
        })
        raise
    finally:
        package_results(download=True)


if __name__ == "__main__":
    raise SystemExit(main())
