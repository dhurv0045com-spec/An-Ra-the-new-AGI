from __future__ import annotations

import argparse
import heapq
import math
import os
import shutil
import signal
import sys
import threading
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anra_paths import ROOT, V2_TOKENIZER_FILE, get_v2_checkpoint, inject_all_paths
from training.anra_optimizer import build_optimizer
from training.eval_v2 import run_compact_eval
from training.mixed_precision import MixedPrecisionTrainer
from training.scheduler import get_cosine_schedule_with_warmup
from training.v2_config import V2_MODEL, V2_TRAINING
from training.v2_data_mix import V2ConversationDataset, build_v2_training_examples
from training.v2_runtime import (
    atomic_save,
    build_v2_model,
    load_checkpoint,
    load_or_build_v2_tokenizer,
    model_summary,
    sync_to_drive,
    sync_v2_artifacts,
    v2_report_path,
    write_json,
)

inject_all_paths()

EARLY_STATUS_STEPS = {1, 2, 5, 10, 20, 50, 100}
HARD_EXAMPLE_KEEP = 16


EMERGENCY_SAVE_TIMEOUT_SECONDS = 20.0
_SAVE_COMPONENT_ORDER = ("model", "optimizer", "scheduler", "scaler")


def _utc_iso(ts: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() if ts is None else ts))


def _build_checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    mp: MixedPrecisionTrainer,
    global_step: int,
    epoch: int,
    best_loss: float,
    sessions_completed: int,
    mix_report: object,
) -> dict[str, object]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": mp.state_dict(),
        "step": global_step,
        "global_step": global_step,
        "epoch": epoch,
        "best_loss": best_loss,
        "sessions_completed": sessions_completed,
        "model_config": model.model_config(),
        "mix_report": mix_report.to_dict(),
    }


def _emergency_save_with_timeout(payload: dict[str, object], ckpt_path: Path) -> bool:
    status: dict[str, object] = {"ok": False, "error": None}

    def _save() -> None:
        try:
            ordered_payload = {key: payload[key] for key in _SAVE_COMPONENT_ORDER}
            ordered_payload.update({k: v for k, v in payload.items() if k not in ordered_payload})
            atomic_save(ordered_payload, ckpt_path, drive_dir=None)
            status["ok"] = True
        except Exception as exc:
            status["error"] = repr(exc)

    worker = threading.Thread(target=_save, name="anra-emergency-save", daemon=True)
    worker.start()
    worker.join(timeout=EMERGENCY_SAVE_TIMEOUT_SECONDS)
    if worker.is_alive():
        print(
            f"[build_brain] emergency save timeout after {EMERGENCY_SAVE_TIMEOUT_SECONDS:.1f}s; process exit continues",
            flush=True,
        )
        return False
    if not bool(status["ok"]):
        print(f"[build_brain] emergency save failed: {status['error']}", flush=True)
        return False
    print("[build_brain] emergency save completed", flush=True)
    return True



def _resolve_checkpoint_path(checkpoint_path: str) -> Path:
    path = Path(checkpoint_path)
    raw = checkpoint_path.replace("\\", "/")
    if os.name == "nt" and raw.startswith("/tmp/"):
        local_tmp = ROOT / "output" / "tmp" / path.name
        print(f"[build_brain] remapping temporary checkpoint path to {local_tmp}", flush=True)
        return local_tmp
    return path if path.is_absolute() else (ROOT / path)


def _prepare_resume_target(checkpoint_path: Path, resume_from: str | None) -> None:
    if checkpoint_path.exists():
        return
    candidate = None
    if resume_from:
        candidate = _resolve_checkpoint_path(resume_from)
        if not candidate.exists():
            drive_copy = DRIVE_V2_CHECKPOINTS / candidate.name
            candidate = drive_copy if drive_copy.exists() else None
    if candidate is None:
        drive_copy = DRIVE_V2_CHECKPOINTS / checkpoint_path.name
        candidate = drive_copy if drive_copy.exists() else None
    if candidate is not None and candidate.exists():
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate, checkpoint_path)
        print(f"[build_brain] restored checkpoint: {candidate} -> {checkpoint_path}", flush=True)


def _weighted_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    *,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, seq_len, channels = logits.shape
    per_token = F.cross_entropy(
        logits.view(bsz * seq_len, channels),
        targets.view(bsz * seq_len),
        reduction="none",
    ).view(bsz, seq_len)
    effective_weights = weights * (targets != pad_id).float()
    sample_losses = (per_token * effective_weights).sum(dim=1) / effective_weights.sum(dim=1).clamp_min(1.0)
    return sample_losses.mean(), sample_losses


def train_anra_v2(
    *,
    data_path: str,
    checkpoint_path: str = "anra_v2_brain.pt",
    resume_from: str | None = None,
    batch_size: int = V2_TRAINING.batch_size,
    block_size: int = V2_MODEL.block_size,
    max_minutes: int = V2_TRAINING.session_minutes,
    answer_loss_weight: float = V2_TRAINING.answer_loss_weight,
    max_examples: int | None = None,
    own_ratio: float | None = None,
    identity_ratio: float | None = None,
    teacher_ratio: float | None = None,
    symbolic_ratio: float | None = None,
    replay_ratio: float | None = None,
) -> dict[str, object]:
    dataset_path = Path(data_path)
    tokenizer = load_or_build_v2_tokenizer(dataset_path=dataset_path)
    examples, mix_report = build_v2_training_examples(
        dataset_path=dataset_path,
        max_examples=max_examples,
        own_ratio=own_ratio,
        identity_ratio=identity_ratio,
        teacher_ratio=teacher_ratio,
        symbolic_ratio=symbolic_ratio,
        replay_ratio=replay_ratio,
    )
    write_json(v2_report_path("mix_report"), mix_report.to_dict())
    ds = V2ConversationDataset(
        examples,
        tokenizer,
        block_size,
        answer_loss_weight=answer_loss_weight,
    )
    if len(ds) == 0:
        raise RuntimeError("V2ConversationDataset produced zero training windows.")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_v2_model(vocab_size=tokenizer.vocab_size, block_size=block_size).to(device)
    mp = MixedPrecisionTrainer(device=device)
    optimizer = build_optimizer(model, lr=3e-4)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=100,
        total_steps=50_000,
    )

    ckpt_path = get_v2_checkpoint("brain")
    ckpt: dict[str, object] = {}
    global_step = 0
    epoch = 0
    best_loss = float("inf")

    registration_ts = time.time()
    signal_state: dict[str, object] = {
        "registered_at": registration_ts,
        "registered_at_iso": _utc_iso(registration_ts),
        "triggered": False,
        "signal": None,
        "emergency_save_completed": None,
    }

    def _handle_sigterm(sig_num: int, _frame: object) -> None:
        signal_state["triggered"] = True
        signal_state["signal"] = sig_num
        print(
            f"[build_brain] SIGTERM handler invoked (signal={sig_num}) at {_utc_iso()}.",
            flush=True,
        )
        sessions_completed = int(ckpt.get("sessions_completed", 0) + 1) if "ckpt" in locals() else 1
        payload = _build_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            mp=mp,
            global_step=global_step,
            epoch=epoch,
            best_loss=best_loss,
            sessions_completed=sessions_completed,
            mix_report=mix_report,
        )
        ok = _emergency_save_with_timeout(payload, ckpt_path)
        signal_state["emergency_save_completed"] = ok
        print(f"[build_brain] SIGTERM emergency save status={ok}", flush=True)
        raise SystemExit(128 + sig_num)

    signal.signal(signal.SIGTERM, _handle_sigterm)
    print(
        f"[build_brain] SIGTERM handler registered at {signal_state['registered_at_iso']} (pre-training).",
        flush=True,
    )

    start_step = 0
    best_loss = float("inf")
    session_start_loss = float("inf")

    # ── AUTO-RESUME ──────────────────────────────────────────────────────────────
    if ckpt_path.exists():
        print(f"[Resume] Found checkpoint: {ckpt_path}", flush=True)
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt.get("model", ckpt.get("model_state_dict", {})))
            optimizer.load_state_dict(ckpt.get("optimizer", ckpt.get("optimizer_state_dict", {})))
            scheduler_state = ckpt.get("scheduler", ckpt.get("scheduler_state_dict"))
            if scheduler_state:
                scheduler.load_state_dict(scheduler_state)
            scaler_state = ckpt.get("scaler", ckpt.get("scaler_state_dict"))
            if scaler_state:
                mp.load_state_dict(scaler_state)
            start_step = int(ckpt.get("step", ckpt.get("global_step", 0)))
            best_loss = float(ckpt.get("best_loss", float("inf")))
            session_start_loss = best_loss
            print(f"[Resume] Resuming from step={start_step}  best_loss={best_loss:.4f}", flush=True)
        except Exception as e:
            print(f"[Resume] Checkpoint load failed ({e}) — starting from scratch", flush=True)
            start_step = 0
            best_loss = float("inf")
            session_start_loss = float("inf")
    else:
        print("[Resume] No checkpoint found — starting from scratch", flush=True)
    # ─────────────────────────────────────────────────────────────────────────────

    global_step = start_step
    epoch = 0

    start = time.time()
    end_at = start + max_minutes * 60
    initial_step = start_step
    session_step = 0
    optimizer.zero_grad(set_to_none=True)
    rolling_loss = 0.0
    rolling_count = 0
    accum_micro_steps = 0
    last_avg_loss = best_loss if math.isfinite(best_loss) else 0.0
    first_batch_wall = None
    hard_examples: list[tuple[float, int]] = []
    answer_weighted_tokens = 0.0
    total_target_tokens = 0.0

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0.0
    summary = model_summary(model)
    eff_batch = batch_size * V2_TRAINING.grad_accum_steps

    print("", flush=True)
    print("=" * 62, flush=True)
    print("  AN-RA V2 TRAINING SESSION", flush=True)
    print("=" * 62, flush=True)
    print(f"  GPU          : {gpu_name} ({gpu_mem:.1f} GB)", flush=True)
    print(f"  Parameters   : {summary['parameters']:,}", flush=True)
    print(
        f"  Micro batch  : {batch_size}  |  Grad accum : {V2_TRAINING.grad_accum_steps}  |  Eff batch : {eff_batch}",
        flush=True,
    )
    print(f"  Session time : {max_minutes} minutes", flush=True)
    print(
        f"  Resuming     : step {global_step:,}  |  best loss {best_loss if math.isfinite(best_loss) else float('inf'):.4f}",
        flush=True,
    )
    print(f"  Checkpoint   : {ckpt_path}", flush=True)
    print(f"  Data mix     : {mix_report.realized_counts}", flush=True)
    print("=" * 62, flush=True)
    print("", flush=True)

    while time.time() < end_at:
        epoch += 1
        for xb, yb, wb, sample_idx in loader:
            if first_batch_wall is None:
                first_batch_wall = time.time()
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)
            with mp.autocast():
                logits, _ = model(xb)
                batch_loss, sample_losses = _weighted_loss(
                    logits,
                    yb,
                    wb,
                    pad_id=tokenizer.pad_token_id,
                )
                loss = batch_loss / V2_TRAINING.grad_accum_steps

            mp.backward(loss)
            rolling_loss += float(loss.item() * V2_TRAINING.grad_accum_steps)
            rolling_count += 1
            accum_micro_steps += 1
            answer_weighted_tokens += float((wb > 1.0).sum().item())
            total_target_tokens += float((yb != tokenizer.pad_token_id).sum().item())

            for sample_loss, example_index in zip(sample_losses.detach().cpu().tolist(), sample_idx.tolist()):
                entry = (float(sample_loss), int(example_index))
                if len(hard_examples) < HARD_EXAMPLE_KEEP:
                    heapq.heappush(hard_examples, entry)
                elif entry[0] > hard_examples[0][0]:
                    heapq.heapreplace(hard_examples, entry)

            if accum_micro_steps >= V2_TRAINING.grad_accum_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                mp.step(optimizer)
                mp.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                session_step += 1
                accum_micro_steps = 0

                avg_loss = rolling_loss / max(1, rolling_count)
                loss_val = avg_loss
                last_avg_loss = avg_loss
                best_loss = min(best_loss, avg_loss) if math.isfinite(best_loss) else avg_loss

                elapsed_min = (time.time() - start) / 60.0
                if session_step % 10 == 0:
                    print(
                        f"  step={global_step:6d}"
                        f"  loss={loss_val:.4f}"
                        f"  best={best_loss:.4f}"
                        f"  elapsed={elapsed_min:.1f}m",
                        flush=True,
                    )

                if global_step in EARLY_STATUS_STEPS or global_step % 200 == 0:
                    elapsed_min = (time.time() - start) / 60.0
                    remaining_min = max(0.0, (end_at - time.time()) / 60.0)
                    startup_note = ""
                    if global_step in EARLY_STATUS_STEPS and first_batch_wall is not None:
                        startup_note = f"  startup={(first_batch_wall - start):.1f}s"
                    print(
                        f"  step={global_step:6d}  loss={avg_loss:.4f}  best={best_loss:.4f}  "
                        f"elapsed={elapsed_min:.1f}m  remaining={remaining_min:.1f}m{startup_note}",
                        flush=True,
                    )

            if time.time() >= end_at:
                break

    if accum_micro_steps > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        mp.step(optimizer)
        mp.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        session_step += 1
        avg_loss = rolling_loss / max(1, rolling_count)
        last_avg_loss = avg_loss
        best_loss = min(best_loss, avg_loss) if math.isfinite(best_loss) else avg_loss
        print(
            f"  step={global_step:6d}  loss={avg_loss:.4f}  best={best_loss:.4f}  "
            f"elapsed={(time.time() - start) / 60.0:.1f}m  remaining={max(0.0, (end_at - time.time()) / 60.0):.1f}m"
            f"  partial_accum={accum_micro_steps}/{V2_TRAINING.grad_accum_steps}",
            flush=True,
        )

    if global_step > initial_step and global_step % 200 != 0:
        elapsed_min = (time.time() - start) / 60.0
        remaining_min = max(0.0, (end_at - time.time()) / 60.0)
        print(
            f"  step={global_step:6d}  loss={last_avg_loss:.4f}  best={best_loss:.4f}  "
            f"elapsed={elapsed_min:.1f}m  remaining={remaining_min:.1f}m",
            flush=True,
        )

    payload = _build_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        mp=mp,
        global_step=global_step,
        epoch=epoch,
        best_loss=best_loss,
        sessions_completed=(int(ckpt.get("sessions_completed", 0) + 1) if "ckpt" in locals() else 1),
        mix_report=mix_report,
    )
    atomic_save(payload, ckpt_path, drive_dir=None)

    metrics = {
        "generated_at": time.time(),
        "elapsed_minutes": round((time.time() - start) / 60.0, 2),
        "session_minutes_target": max_minutes,
        "global_step": global_step,
        "epoch": epoch,
        "best_loss": round(best_loss, 4),
        "last_avg_loss": round(last_avg_loss, 4),
        "effective_batch_size": eff_batch,
        "grad_accum_steps": V2_TRAINING.grad_accum_steps,
        "answer_loss_weight": answer_loss_weight,
        "answer_supervision_ratio": round(ds.answer_supervision_ratio, 4),
        "reply_token_ratio_seen": round(answer_weighted_tokens / max(1.0, total_target_tokens), 4),
        "model_config": model.model_config(),
        "checkpoint_path": str(ckpt_path),
        "mix_report": mix_report.to_dict(),
        "signal_handler": signal_state,
    }
    write_json(v2_report_path("metrics"), metrics)

    hard_examples_report = [
        {
            "loss": round(loss_value, 4),
            "sample_index": sample_index,
            "preview": ds.snippet(sample_index),
        }
        for loss_value, sample_index in sorted(hard_examples, key=lambda item: item[0], reverse=True)
    ]
    write_json(
        v2_report_path("hard_examples"),
        {
            "generated_at": time.time(),
            "answer_loss_weight": answer_loss_weight,
            "examples": hard_examples_report,
        },
    )

    eval_summary = run_compact_eval(model, tokenizer, device=device, output=True)
    sync_v2_artifacts(
        ckpt_path,
        tokenizer_path=V2_TOKENIZER_FILE,
        extra_paths=[
            v2_report_path("metrics"),
            v2_report_path("hard_examples"),
            v2_report_path("eval_summary"),
            v2_report_path("mix_report"),
        ],
    )
    sync_to_drive("brain")
    sync_to_drive("tokenizer")
    sync_to_drive("eval_summary")

    elapsed_total = time.time() - start
    print("", flush=True)
    print("=" * 62, flush=True)
    print("  V2 SESSION COMPLETE", flush=True)
    print("=" * 62, flush=True)
    print(f"  Steps this session : {global_step - initial_step:,}", flush=True)
    print(f"  Total steps        : {global_step:,}", flush=True)
    print(f"  Best loss          : {best_loss:.4f}", flush=True)
    print(f"  Eval score         : {float(eval_summary.get('overall_score', 0.0)):.4f}", flush=True)
    print(f"  Time elapsed       : {elapsed_total / 60:.1f} minutes", flush=True)
    print(f"  Checkpoint saved   : {ckpt_path}", flush=True)
    print("  Drive synced       : yes", flush=True)
    print("=" * 62, flush=True)
    print("", flush=True)

    # ── SESSION SUMMARY ──────────────────────────────────────────────────────────
    print("\n" + "=" * 50, flush=True)
    print("SESSION COMPLETE", flush=True)
    print(f"  Steps this session : {session_step}", flush=True)
    print(f"  Total steps ever   : {global_step}", flush=True)
    print(f"  Loss at start      : {session_start_loss:.4f}", flush=True)
    print(f"  Best loss achieved : {best_loss:.4f}", flush=True)
    if session_start_loss != float("inf"):
        improvement = session_start_loss - best_loss
        direction = "improved" if improvement > 0 else "no improvement"
        print(f"  Improvement        : {improvement:+.4f}  ({direction})", flush=True)
    print("=" * 50 + "\n", flush=True)
    # ─────────────────────────────────────────────────────────────────────────────

    return {
        "checkpoint_path": str(ckpt_path),
        "global_step": global_step,
        "best_loss": best_loss,
        "eval_summary": eval_summary,
        "mix_report": mix_report.to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonical An-Ra base trainer")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--checkpoint_path", default="anra_v2_brain.pt")
    parser.add_argument("--resume_from", default=None)
    parser.add_argument("--batch_size", type=int, default=V2_TRAINING.batch_size)
    parser.add_argument("--block_size", type=int, default=V2_MODEL.block_size)
    parser.add_argument("--max_minutes", type=int, default=V2_TRAINING.session_minutes)
    parser.add_argument("--answer_loss_weight", type=float, default=V2_TRAINING.answer_loss_weight)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--own_ratio", type=float, default=None)
    parser.add_argument("--identity_ratio", type=float, default=None)
    parser.add_argument("--teacher_ratio", type=float, default=None)
    parser.add_argument("--symbolic_ratio", type=float, default=None)
    parser.add_argument("--replay_ratio", type=float, default=None)
    args = parser.parse_args()
    result = train_anra_v2(
        data_path=args.data_path,
        checkpoint_path=args.checkpoint_path,
        resume_from=args.resume_from,
        batch_size=args.batch_size,
        block_size=args.block_size,
        max_minutes=args.max_minutes,
        answer_loss_weight=args.answer_loss_weight,
        max_examples=args.max_examples,
        own_ratio=args.own_ratio,
        identity_ratio=args.identity_ratio,
        teacher_ratio=args.teacher_ratio,
        symbolic_ratio=args.symbolic_ratio,
        replay_ratio=args.replay_ratio,
    )
    print(result, flush=True)


if __name__ == "__main__":
    main()
