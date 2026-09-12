from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO / "experiments" / "ARK-001"))

from experiments.lib import ark_tasks
from run_ark001 import CompactVocab, Micro, greedy_exact, loss_and_positions

PLAN_SHA = "1d5fc08000614e740b7c93d87e8d233c903bcddf"
ADDENDUM_SHA = "70fa2764fc0df79d4bf4b0b16e6620bc87278aa9"
CANONICAL_T2_SHA = "0dd9305697045b0fbf4e7f268b46a4d7276e4794af5d78b60e999df914ae4236"
RESULTS_DIR = Path("/content/arkenstone_ark011_results")
DEFAULT_BUDGET_MINUTES = 180
SESSION_START = time.time()
DEVICE: torch.device | None = None
RUNNER_HEAD = "UNKNOWN"
RUNNER_SOURCE_SHA256 = "UNKNOWN"


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_json(value) -> str:
    return sha_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))


def file_sha256(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def git_head() -> str:
    return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()


def minutes_used() -> float:
    return (time.time() - SESSION_START) / 60.0


def minutes_left(budget_minutes: int) -> float:
    return float(budget_minutes) - minutes_used()


def init_cuda() -> torch.device:
    global DEVICE
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not detected. In Colab choose Runtime -> Change runtime type -> T4 GPU.")
    DEVICE = torch.device("cuda")
    print("DEVICE:", torch.cuda.get_device_name(0), "| torch", torch.__version__, flush=True)
    return DEVICE


def dev() -> torch.device:
    if DEVICE is None:
        raise RuntimeError("CUDA device has not been initialized")
    return DEVICE


def sync() -> None:
    if DEVICE is not None and DEVICE.type == "cuda":
        torch.cuda.synchronize()


def cpu_tree(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: cpu_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [cpu_tree(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(cpu_tree(v) for v in obj)
    return copy.deepcopy(obj)


def snapshot_state(model, optimizer) -> dict:
    return {
        "model": cpu_tree(model.state_dict()),
        "optimizer": cpu_tree(optimizer.state_dict()),
        "torch_rng": torch.get_rng_state().cpu(),
        "cuda_rng": [x.cpu() for x in torch.cuda.get_rng_state_all()],
    }


def flat_params(model) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def model_state_equal(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k].detach().cpu(), b[k].detach().cpu()) for k in a)


def parameter_sha(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_json(name: str, payload: dict) -> Path:
    out = dict(payload)
    out["plan_commit_sha"] = PLAN_SHA
    out["preexecution_addendum_sha"] = ADDENDUM_SHA
    out["runner_commit_sha"] = RUNNER_HEAD
    out["runner_source_sha256"] = RUNNER_SOURCE_SHA256
    out["device"] = str(DEVICE) if DEVICE is not None else "uninitialized"
    out["torch"] = torch.__version__
    body = dict(out)
    body.pop("receipt_sha256", None)
    out["receipt_sha256"] = sha_json(body)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    path.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    print("saved:", path, flush=True)
    return path


def package_results(download: bool = True) -> Path:
    zip_path = RESULTS_DIR / "ARKENSTONE_ARK011_RESULTS.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(RESULTS_DIR.glob("*.json")):
            zf.write(path, path.name)
    print("RESULT ZIP:", zip_path, flush=True)
    if download:
        try:
            from google.colab import files
            files.download(str(zip_path))
        except Exception as exc:
            print("auto-download skipped:", repr(exc), flush=True)
    return zip_path


def load_manifest() -> dict:
    manifest = ark_tasks.load_or_build_manifest(str(REPO / "experiments" / "ARK-002B" / "TASK_MANIFEST.json"))
    actual = manifest.get("split_sha256")
    if actual != CANONICAL_T2_SHA:
        raise RuntimeError(f"canonical T2 manifest drift: {actual} != {CANONICAL_T2_SHA}")
    if len(manifest.get("train", [])) != 500:
        raise RuntimeError("canonical T2 train count drift")
    return manifest


def operand_a_tens(prompt: str) -> int:
    a = int(prompt.split("+")[0].strip())
    return a // 10


def row_key(row) -> str:
    prompt, answer = row
    return hashlib.sha256((prompt + "\0" + answer).encode("utf-8")).hexdigest()


def build_control_sealed_split(test_rows) -> tuple[list[tuple[str, str]], list[tuple[str, str]], dict]:
    rows = [(str(p), str(a)) for p, a in test_rows]
    groups: dict[int, list[tuple[str, str]]] = {6: [], 7: []}
    for row in rows:
        band = operand_a_tens(row[0])
        if band not in groups:
            raise RuntimeError(f"unexpected T2 test band: {band}")
        groups[band].append(row)

    control: list[tuple[str, str]] = []
    sealed: list[tuple[str, str]] = []
    band_counts = {}
    for band in sorted(groups):
        ordered = sorted(groups[band], key=row_key)
        c = [row for i, row in enumerate(ordered) if i % 2 == 0]
        s = [row for i, row in enumerate(ordered) if i % 2 == 1]
        control.extend(c)
        sealed.extend(s)
        band_counts[str(band)] = {"source": len(ordered), "control": len(c), "sealed": len(s)}

    source_set = set(rows)
    control_set = set(control)
    sealed_set = set(sealed)
    if control_set & sealed_set:
        raise RuntimeError("CONTROL/SEALED overlap")
    if control_set | sealed_set != source_set:
        raise RuntimeError("CONTROL/SEALED union does not reproduce source OOD set")
    if len(control) + len(sealed) != len(rows):
        raise RuntimeError("CONTROL/SEALED count mismatch")

    split_manifest = {
        "schema": "arkenstone-ark011-control-sealed/v1",
        "source_task": "t2-no-carry-add",
        "source_split_sha256": CANONICAL_T2_SHA,
        "algorithm": "within operand-A tens band 6/7, sort by sha256(prompt + NUL + answer); even rank CONTROL, odd rank SEALED",
        "band_counts": band_counts,
        "control_count": len(control),
        "sealed_count": len(sealed),
        "control_sha256": sha_json(control),
        "sealed_sha256": sha_json(sealed),
        "assignment_sha256": sha_json({"control": control, "sealed": sealed}),
        "control": [list(x) for x in control],
        "sealed": [list(x) for x in sealed],
    }
    return control, sealed, split_manifest


def detect_sustained(evals: list[tuple[int, float]], bar: float, consecutive: int = 3, below: bool = False):
    streak = 0
    onset = None
    for step, value in evals:
        hit = value < bar if below else value >= bar
        if hit:
            if streak == 0:
                onset = step
            streak += 1
            if streak >= consecutive:
                return onset, step
        else:
            streak = 0
            onset = None
    return None, None


def generate_continuation_indices(order_seed: int, n_batches: int, batch_size: int, pool_size: int):
    rng = torch.Generator().manual_seed(order_seed)
    return torch.randint(0, pool_size, (n_batches, batch_size), generator=rng).tolist()


def order_sha256(indices) -> str:
    return sha_bytes(json.dumps(indices, separators=(",", ":")).encode("utf-8"))


def load_fork(snapshot: dict, lr: float):
    vocab = CompactVocab()
    model = Micro(vocab.size, 128).to(dev())
    model.load_state_dict({k: v.to(dev()) for k, v in snapshot["model"].items()})
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    optimizer.load_state_dict(copy.deepcopy(snapshot["optimizer"]))
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(dev())
    for group in optimizer.param_groups:
        group["lr"] = lr
    torch.set_rng_state(snapshot["torch_rng"].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in snapshot["cuda_rng"]])
    return vocab, model, optimizer


def train_step(model, optimizer, vocab, rows) -> int:
    loss, count = loss_and_positions(model, vocab, rows, dev())
    if not torch.isfinite(loss):
        raise RuntimeError("nonfinite loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return int(count.item())


def acquire(seed: int, train, control, max_steps: int = 28000, eval_every: int = 200) -> dict:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    vocab = CompactVocab()
    model = Micro(vocab.size, 128).to(dev())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    rng = torch.Generator().manual_seed(seed)
    control_evals: list[tuple[int, float]] = []
    trajectory = []
    supervised_tokens = 0
    started = time.time()

    for step in range(1, max_steps + 1):
        idx = torch.randint(0, len(train), (64,), generator=rng)
        rows = [train[int(i)] for i in idx]
        loss, count = loss_and_positions(model, vocab, rows, dev())
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite acquisition loss seed={seed} step={step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        supervised_tokens += int(count.item())

        if step == 1 or step % eval_every == 0:
            train_exact, _ = greedy_exact(model, vocab, train[:100], dev())
            control_exact, _ = greedy_exact(model, vocab, control, dev())
            control_evals.append((step, control_exact))
            onset, confirm = detect_sustained(control_evals, 0.90, 3)
            trajectory.append({
                "step": step,
                "train_exact": train_exact,
                "control_exact": control_exact,
                "loss": float(loss.detach().item()),
            })
            print(f"[ACQ s{seed}] step={step} train={train_exact:.3f} control={control_exact:.3f}", flush=True)
            if confirm is not None:
                sync()
                return {
                    "status": "ACQUIRED",
                    "seed": seed,
                    "onset_step": onset,
                    "confirmation_step": confirm,
                    "supervised_tokens": supervised_tokens,
                    "trajectory": trajectory,
                    "snapshot": snapshot_state(model, optimizer),
                }
        if time.time() - started > 1800:
            return {
                "status": "ACQUISITION_WALLTIME_STOP",
                "seed": seed,
                "supervised_tokens": supervised_tokens,
                "trajectory": trajectory,
            }

    return {
        "status": "NO_CONTROL_G90",
        "seed": seed,
        "supervised_tokens": supervised_tokens,
        "trajectory": trajectory,
    }


def run_to_threshold(
    *,
    phase_name: str,
    snapshot: dict,
    indices,
    offset: int,
    max_steps: int,
    train,
    control,
    lr: float,
    bar: float,
    below: bool,
    eval_every: int = 200,
) -> dict:
    if offset + max_steps > len(indices):
        raise RuntimeError(f"{phase_name}: continuation tail too short")
    vocab, model, optimizer = load_fork(snapshot, lr)
    evals: list[tuple[int, float]] = []
    trajectory = []
    supervised_tokens = 0

    for rel_step in range(1, max_steps + 1):
        rows = [train[i] for i in indices[offset + rel_step - 1]]
        loss, count = loss_and_positions(model, vocab, rows, dev())
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite {phase_name} loss step={rel_step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        supervised_tokens += int(count.item())

        if rel_step % eval_every == 0:
            control_exact, _ = greedy_exact(model, vocab, control, dev())
            evals.append((rel_step, control_exact))
            onset, confirm = detect_sustained(evals, bar, 3, below=below)
            trajectory.append({
                "relative_step": rel_step,
                "absolute_continuation_step": offset + rel_step,
                "control_exact": control_exact,
                "loss": float(loss.detach().item()),
            })
            print(
                f"[{phase_name}] rel={rel_step} abs={offset+rel_step} control={control_exact:.3f}",
                flush=True,
            )
            if confirm is not None:
                sync()
                return {
                    "status": "TRIGGERED",
                    "onset_relative_step": onset,
                    "confirmation_relative_step": confirm,
                    "onset_absolute_step": offset + onset,
                    "confirmation_absolute_step": offset + confirm,
                    "steps_used": confirm,
                    "supervised_tokens": supervised_tokens,
                    "trajectory": trajectory,
                    "snapshot": snapshot_state(model, optimizer),
                    "control_at_confirmation": control_exact,
                }

    return {
        "status": "NO_TRIGGER",
        "steps_used": max_steps,
        "supervised_tokens": supervised_tokens,
        "trajectory": trajectory,
    }


def retention_metrics(trajectory: list[dict], key: str) -> dict:
    if not trajectory:
        return {"status": "EMPTY"}
    vals = [float(x[key]) for x in trajectory]
    onset, confirm = detect_sustained(
        [(int(x["step"]), float(x[key])) for x in trajectory], 0.90, 3, below=True
    )
    return {
        "RET90": sum(v >= 0.90 for v in vals) / len(vals),
        "RET50": sum(v >= 0.50 for v in vals) / len(vals),
        "AREA": sum(vals) / len(vals),
        "FINAL": vals[-1],
        "PEAK": max(vals),
        "T_RECOLLAPSE_ONSET": onset,
        "T_RECOLLAPSE_CONFIRM": confirm,
        "recollapsed": confirm is not None,
    }


def run_retention_fork(
    *,
    snapshot: dict,
    recovery_flat_cpu: torch.Tensor,
    indices,
    offset: int,
    train,
    control,
    sealed,
    lr: float,
    steps: int = 6000,
    eval_every: int = 200,
) -> dict:
    if offset + steps > len(indices):
        raise RuntimeError("retention continuation tail too short")
    vocab, model, optimizer = load_fork(snapshot, lr)
    recovery_flat = recovery_flat_cpu.to(dev())
    trajectory = []
    supervised_tokens = 0

    for step in range(1, steps + 1):
        rows = [train[i] for i in indices[offset + step - 1]]
        loss, count = loss_and_positions(model, vocab, rows, dev())
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite retention loss step={step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        supervised_tokens += int(count.item())

        if step % eval_every == 0:
            control_exact, _ = greedy_exact(model, vocab, control, dev())
            sealed_exact, _ = greedy_exact(model, vocab, sealed, dev())
            current = flat_params(model)
            l2 = float((current - recovery_flat).norm().item())
            relative = l2 / max(float(recovery_flat.norm().item()), 1e-12)
            trajectory.append({
                "step": step,
                "absolute_continuation_step": offset + step,
                "control_exact": control_exact,
                "sealed_exact": sealed_exact,
                "l2_displacement": l2,
                "relative_displacement": relative,
                "loss": float(loss.detach().item()),
            })

    return {
        "lr": lr,
        "supervised_tokens": supervised_tokens,
        "trajectory": trajectory,
        "control_retention": retention_metrics(trajectory, "control_exact"),
        "sealed_retention": retention_metrics(trajectory, "sealed_exact"),
        "final_parameter_sha256": parameter_sha(model),
    }


def summarize(results: list[dict]) -> dict:
    executed = [r for r in results if r.get("status") == "FORK_EXECUTED"]
    qualified = [r for r in executed if float(r.get("sealed_at_recovery_fork", 0.0)) >= 0.90]

    high_collapses = sum(bool(r["HIGH_CONTINUE"]["sealed_retention"]["recollapsed"]) for r in qualified)
    low_collapses = sum(bool(r["SWITCH_LOW"]["sealed_retention"]["recollapsed"]) for r in qualified)
    n = len(qualified)
    risk_diff = (low_collapses - high_collapses) / n if n else None
    high_bad_low_good = sum(
        bool(r["HIGH_CONTINUE"]["sealed_retention"]["recollapsed"])
        and not bool(r["SWITCH_LOW"]["sealed_retention"]["recollapsed"])
        for r in qualified
    )
    reverse = sum(
        not bool(r["HIGH_CONTINUE"]["sealed_retention"]["recollapsed"])
        and bool(r["SWITCH_LOW"]["sealed_retention"]["recollapsed"])
        for r in qualified
    )

    grouped = {}
    for seed in sorted({int(r["acquisition_seed"]) for r in qualified}):
        rows = [r for r in qualified if int(r["acquisition_seed"]) == seed]
        grouped[str(seed)] = {
            "n": len(rows),
            "high_recollapses": sum(bool(r["HIGH_CONTINUE"]["sealed_retention"]["recollapsed"]) for r in rows),
            "low_recollapses": sum(bool(r["SWITCH_LOW"]["sealed_retention"]["recollapsed"]) for r in rows),
        }

    distinct_seeds = len(grouped)
    if n < 4 or distinct_seeds < 2:
        verdict = "INCONCLUSIVE_LOW_RECOVERY_EVENT_RATE"
    elif high_collapses < 2:
        verdict = "INCONCLUSIVE_LOW_RECOLLAPSE_RATE"
    else:
        nonreversed_groups = sum(v["low_recollapses"] <= v["high_recollapses"] for v in grouped.values())
        majority_nonreversed = nonreversed_groups >= (distinct_seeds // 2 + 1)
        if risk_diff is not None and risk_diff <= -0.30 and majority_nonreversed:
            verdict = "SUPPORTED_ADAPTIVE_PROTECTION"
        else:
            verdict = "NOT_SUPPORTED"

    return {
        "event_opportunities_recorded": len(results),
        "forks_executed": len(executed),
        "sealed_qualified_forks": n,
        "independent_acquisition_seeds_in_primary": distinct_seeds,
        "high_recollapses": high_collapses,
        "low_recollapses": low_collapses,
        "risk_difference_low_minus_high": risk_diff,
        "high_recollapse_low_stable": high_bad_low_good,
        "high_stable_low_recollapse": reverse,
        "by_acquisition_seed": grouped,
        "verdict": verdict,
    }


def smoke_test() -> None:
    print("\n=== ARK-011 SMOKE TEST ===", flush=True)
    manifest = load_manifest()
    train = [(p, a) for p, a in manifest["train"]]
    test = [(p, a) for p, a in manifest["test"]]
    control, sealed, split_manifest = build_control_sealed_split(test)

    assert control and sealed
    assert not (set(control) & set(sealed))
    assert set(control) | set(sealed) == set(test)

    a = generate_continuation_indices(9991, 12, 4, len(train))
    a2 = generate_continuation_indices(9991, 12, 4, len(train))
    b = generate_continuation_indices(9992, 12, 4, len(train))
    assert order_sha256(a) == order_sha256(a2)
    assert order_sha256(a) != order_sha256(b)

    vocab = CompactVocab()
    torch.manual_seed(9191)
    torch.cuda.manual_seed_all(9191)
    model = Micro(vocab.size, 128).to(dev())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    rows = train[:8]
    loss, count = loss_and_positions(model, vocab, rows, dev())
    assert torch.isfinite(loss) and int(count.item()) > 0
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    snap = snapshot_state(model, optimizer)

    _, restored, _ = load_fork(snap, 1e-3)
    assert model_state_equal(model.state_dict(), restored.state_dict())

    vocab1, m1, o1 = load_fork(snap, 1e-3)
    vocab2, m2, o2 = load_fork(snap, 1e-3)
    batch = train[:8]
    train_step(m1, o1, vocab1, batch)
    train_step(m2, o2, vocab2, batch)
    assert parameter_sha(m1) == parameter_sha(m2)

    ce, _ = greedy_exact(restored, vocab, control[:12], dev())
    se, _ = greedy_exact(restored, vocab, sealed[:12], dev())
    assert 0.0 <= ce <= 1.0 and 0.0 <= se <= 1.0

    save_json(
        "SMOKE_TEST.json",
        {
            "experiment_id": "ARK-011_SMOKE",
            "status": "PASS",
            "source_split_sha256": CANONICAL_T2_SHA,
            "control_sha256": split_manifest["control_sha256"],
            "sealed_sha256": split_manifest["sealed_sha256"],
            "assignment_sha256": split_manifest["assignment_sha256"],
            "continuation_hash_reproducible": True,
            "snapshot_reload_exact": True,
            "equal_lr_equal_batch_fork_equivalence": True,
        },
    )
    print("ARK-011 SMOKE TEST PASS", flush=True)


def full_campaign(budget_minutes: int) -> dict:
    print("\n=== ARK-011 FULL CAMPAIGN ===", flush=True)
    manifest = load_manifest()
    train = [(p, a) for p, a in manifest["train"]]
    source_test = [(p, a) for p, a in manifest["test"]]
    control, sealed, split_manifest = build_control_sealed_split(source_test)
    save_json("ARK-011_TASK_MANIFEST.json", split_manifest)

    acquisition_seeds = [1313, 1414, 1515]
    order_seeds = [5701, 5702, 5703, 5704]
    acquisitions = []
    results = []

    for seed in acquisition_seeds:
        if minutes_left(budget_minutes) < 45:
            print("budget gate: no new acquisition", flush=True)
            break
        acq = acquire(seed, train, control)
        acquisitions.append({k: v for k, v in acq.items() if k not in {"snapshot"}})
        save_json("ARK-011_PARTIAL.json", {"acquisitions": acquisitions, "results": results})

        if acq["status"] != "ACQUIRED":
            for order_seed in order_seeds:
                results.append({
                    "acquisition_seed": seed,
                    "continuation_seed": order_seed,
                    "status": "BLOCKED_BY_ACQUISITION",
                })
            save_json("ARK-011_PARTIAL.json", {"acquisitions": acquisitions, "results": results})
            continue

        for order_seed in order_seeds:
            if minutes_left(budget_minutes) < 18:
                results.append({
                    "acquisition_seed": seed,
                    "continuation_seed": order_seed,
                    "status": "BUDGET_BLOCKED_BEFORE_ORDER",
                })
                save_json("ARK-011_PARTIAL.json", {"acquisitions": acquisitions, "results": results})
                continue

            print(f"\n--- source s{seed} order{order_seed} ---", flush=True)
            indices = generate_continuation_indices(order_seed, 16000, 64, len(train))
            order_hash = order_sha256(indices)

            collapse = run_to_threshold(
                phase_name=f"COLLAPSE s{seed} o{order_seed}",
                snapshot=acq["snapshot"],
                indices=indices,
                offset=0,
                max_steps=6000,
                train=train,
                control=control,
                lr=1e-3,
                bar=0.90,
                below=True,
            )
            if collapse["status"] != "TRIGGERED":
                results.append({
                    "acquisition_seed": seed,
                    "continuation_seed": order_seed,
                    "continuation_order_sha256": order_hash,
                    "status": "NO_CONTROL_COLLAPSE",
                    "collapse": {k: v for k, v in collapse.items() if k != "snapshot"},
                })
                save_json("ARK-011_PARTIAL.json", {"acquisitions": acquisitions, "results": results})
                continue

            collapse_offset = int(collapse["confirmation_absolute_step"])
            recovery = run_to_threshold(
                phase_name=f"RECOVERY s{seed} o{order_seed}",
                snapshot=collapse["snapshot"],
                indices=indices,
                offset=collapse_offset,
                max_steps=4000,
                train=train,
                control=control,
                lr=1e-3,
                bar=0.90,
                below=False,
            )
            if recovery["status"] != "TRIGGERED":
                results.append({
                    "acquisition_seed": seed,
                    "continuation_seed": order_seed,
                    "continuation_order_sha256": order_hash,
                    "status": "NO_CONTROL_RECOVERY",
                    "collapse": {k: v for k, v in collapse.items() if k != "snapshot"},
                    "recovery": {k: v for k, v in recovery.items() if k != "snapshot"},
                })
                save_json("ARK-011_PARTIAL.json", {"acquisitions": acquisitions, "results": results})
                continue

            recovery_offset = int(recovery["confirmation_absolute_step"])
            rvocab, rmodel, _ = load_fork(recovery["snapshot"], 1e-3)
            sealed_at_fork, _ = greedy_exact(rmodel, rvocab, sealed, dev())
            control_at_fork, _ = greedy_exact(rmodel, rvocab, control, dev())
            recovery_flat = flat_params(rmodel).detach().cpu()
            recovery_parameter_sha = parameter_sha(rmodel)
            del rmodel
            torch.cuda.empty_cache()

            high = run_retention_fork(
                snapshot=recovery["snapshot"],
                recovery_flat_cpu=recovery_flat,
                indices=indices,
                offset=recovery_offset,
                train=train,
                control=control,
                sealed=sealed,
                lr=1e-3,
                steps=6000,
            )
            low = run_retention_fork(
                snapshot=recovery["snapshot"],
                recovery_flat_cpu=recovery_flat,
                indices=indices,
                offset=recovery_offset,
                train=train,
                control=control,
                sealed=sealed,
                lr=1e-5,
                steps=6000,
            )

            event = {
                "acquisition_seed": seed,
                "continuation_seed": order_seed,
                "continuation_order_sha256": order_hash,
                "status": "FORK_EXECUTED",
                "sealed_at_recovery_fork": sealed_at_fork,
                "control_at_recovery_fork": control_at_fork,
                "sealed_primary_qualified": sealed_at_fork >= 0.90,
                "recovery_parameter_sha256": recovery_parameter_sha,
                "acquisition": {
                    "control_g90_onset_step": acq["onset_step"],
                    "control_g90_confirmation_step": acq["confirmation_step"],
                    "supervised_tokens": acq["supervised_tokens"],
                },
                "collapse": {k: v for k, v in collapse.items() if k != "snapshot"},
                "recovery": {k: v for k, v in recovery.items() if k != "snapshot"},
                "HIGH_CONTINUE": high,
                "SWITCH_LOW": low,
            }
            results.append(event)
            print(
                f"[ARK-011] s{seed} o{order_seed} sealed_at_fork={sealed_at_fork:.3f} "
                f"HIGH_recollapse={high['sealed_retention']['recollapsed']} "
                f"LOW_recollapse={low['sealed_retention']['recollapsed']}",
                flush=True,
            )
            save_json("ARK-011_PARTIAL.json", {"acquisitions": acquisitions, "results": results})

    summary = summarize(results)
    payload = {
        "experiment_id": "ARK-011",
        "status": "EXECUTED_OR_PARTIAL_BUDGETED",
        "source_manifest_sha256": CANONICAL_T2_SHA,
        "controller_eval_split": split_manifest,
        "acquisitions": acquisitions,
        "results": results,
        "summary": summary,
        "minutes_used": minutes_used(),
    }
    save_json("ARK-011_RESULT.json", payload)
    return payload


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--budget-minutes", type=int, default=DEFAULT_BUDGET_MINUTES)
    parser.add_argument("--expected-head", type=str, default="")
    return parser.parse_args()


def main() -> int:
    global SESSION_START, RUNNER_HEAD, RUNNER_SOURCE_SHA256
    args = parse_args()
    SESSION_START = time.time()
    init_cuda()
    RUNNER_HEAD = git_head()
    RUNNER_SOURCE_SHA256 = file_sha256(Path(__file__))
    print("PLAN:", PLAN_SHA, flush=True)
    print("ADDENDUM:", ADDENDUM_SHA, flush=True)
    print("RUNNER HEAD:", RUNNER_HEAD, flush=True)
    print("RUNNER SHA256:", RUNNER_SOURCE_SHA256, flush=True)
    if args.expected_head and RUNNER_HEAD != args.expected_head:
        raise RuntimeError(f"checked-out HEAD {RUNNER_HEAD} != expected {args.expected_head}")

    if args.smoke_test:
        smoke_test()
        package_results(download=False)
        return 0

    try:
        full_campaign(int(args.budget_minutes))
        return 0
    except Exception as exc:
        save_json(
            "FAILURE_RECEIPT.json",
            {
                "experiment_id": "ARK-011",
                "status": "FAILED",
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc(),
                "minutes_used": minutes_used(),
            },
        )
        raise
    finally:
        package_results(download=True)


if __name__ == "__main__":
    raise SystemExit(main())
