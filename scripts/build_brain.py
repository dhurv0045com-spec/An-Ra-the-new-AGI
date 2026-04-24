from __future__ import annotations

import argparse
import heapq
import json
import os
import pickle
import platform
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anra_paths import DRIVE_CHECKPOINTS, ROOT, get_checkpoint, get_tokenizer_file, inject_all_paths

inject_all_paths()

from anra_brain import CausalTransformer
from training.anra_optimizer import build_optimizer
from training.mixed_precision import MixedPrecisionTrainer
from training.scheduler import get_cosine_schedule_with_warmup

GRAD_ACCUM_STEPS = 8
MAX_MINUTES = 30
ANSWER_LOSS_WEIGHT = 1.75
HARD_EXAMPLE_KEEP = 12
EARLY_STATUS_STEPS = {1, 2, 5, 10, 20, 50, 100}


class TextDataset(Dataset):
    def __init__(self, data_tensor: torch.Tensor, block_size: int):
        self.data = data_tensor
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size - 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[index : index + self.block_size]
        y = self.data[index + 1 : index + self.block_size + 1]
        return x, y


class ConversationDataset(Dataset):
    """
    Aligns training windows to H: conversation boundaries.

    Every sample starts at the beginning of an H: turn so the model
    learns complete exchanges: question -> response. Falls back to
    uniform sampling if too few conversation boundaries are found.
    """

    def __init__(self, text: str, tokenizer, block_size: int, *, answer_loss_weight: float = ANSWER_LOSS_WEIGHT):
        import re

        self.text = text
        self.block_size = block_size
        self.answer_loss_weight = float(max(1.0, answer_loss_weight))
        self.tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        n = len(self.tokens)
        boundary_re = re.compile(r"(?:^|\n)H: ")
        starts = [m.start() if m.start() == 0 else m.start() + 1 for m in boundary_re.finditer(text)]

        self.starts: list[int] = []
        self.reply_starts: list[int] = []
        self.reply_ends: list[int] = []
        weighted_targets = 0

        for idx, start in enumerate(starts):
            next_start = starts[idx + 1] if idx + 1 < len(starts) else len(text)
            response_start = text.find("ANRA:", start, next_start)
            response_start = response_start if response_start != -1 else next_start
            sample_start = min(start, n - block_size - 2)
            if sample_start < 0:
                continue
            if sample_start + block_size + 1 > n:
                continue
            self.starts.append(sample_start)
            self.reply_starts.append(response_start)
            self.reply_ends.append(next_start)
            target_start = sample_start + 1
            target_end = sample_start + block_size + 1
            overlap_start = max(response_start, target_start)
            overlap_end = min(next_start, target_end)
            if overlap_end > overlap_start:
                weighted_targets += overlap_end - overlap_start

        if len(self.starts) < 200:
            self.starts = []
            self.reply_starts = []
            self.reply_ends = []
            weighted_targets = 0
            stride = max(1, block_size // 2)
            for idx in range(0, n - block_size - 1, stride):
                response_start = text.find("ANRA:", idx, min(len(text), idx + block_size + 1))
                response_start = response_start if response_start != -1 else idx + block_size + 1
                response_end = min(len(text), idx + block_size + 1)
                self.starts.append(idx)
                self.reply_starts.append(response_start)
                self.reply_ends.append(response_end)
                target_start = idx + 1
                target_end = idx + block_size + 1
                overlap_start = max(response_start, target_start)
                overlap_end = min(response_end, target_end)
                if overlap_end > overlap_start:
                    weighted_targets += overlap_end - overlap_start

        total_targets = max(1, len(self.starts) * block_size)
        self.answer_supervision_ratio = weighted_targets / total_targets
        print(
            f"[ConversationDataset] {len(self.starts)} training windows from {len(starts)} H: boundaries",
            flush=True,
        )
        print(
            f"[ConversationDataset] answer supervision ratio={self.answer_supervision_ratio:.2%} "
            f"(weight={self.answer_loss_weight:.2f})",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        start = self.starts[index]
        x = self.tokens[start : start + self.block_size]
        y = self.tokens[start + 1 : start + self.block_size + 1]
        weights = torch.ones(self.block_size, dtype=torch.float32)
        target_start = start + 1
        target_end = start + self.block_size + 1
        overlap_start = max(self.reply_starts[index], target_start)
        overlap_end = min(self.reply_ends[index], target_end)
        if overlap_end > overlap_start:
            weights[overlap_start - target_start : overlap_end - target_start] = self.answer_loss_weight
        return x, y, weights, index

    def snippet(self, index: int, max_chars: int = 220) -> str:
        start = self.starts[index]
        end = min(len(self.text), start + max_chars)
        return self.text[start:end].replace("\n", "\\n")


def _atomic_save(payload: dict, output_path: Path, drive_dir: Path | None = None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(output_path)
    if drive_dir is not None:
        try:
            drive_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output_path, drive_dir / output_path.name)
        except Exception:
            pass


def _sync_required_artifacts(checkpoint_path: Path) -> None:
    """Keep required runtime artifacts mirrored to Drive."""
    try:
        DRIVE_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint_path, DRIVE_CHECKPOINTS / checkpoint_path.name)
        tok = get_tokenizer_file()
        if tok.exists():
            drive_tokenizer = DRIVE_CHECKPOINTS.parent / "tokenizer.pkl"
            drive_tokenizer.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tok, drive_tokenizer)
    except Exception:
        pass


def _resolve_checkpoint_path(checkpoint_path: str) -> Path:
    path = Path(checkpoint_path)
    raw = checkpoint_path.replace("\\", "/")
    if os.name == "nt" and raw.startswith("/tmp/"):
        local_tmp = ROOT / "output" / "tmp" / path.name
        print(
            f"[build_brain] remapping temporary checkpoint path to {local_tmp}",
            flush=True,
        )
        return local_tmp
    return path


def train_anra_brain(
    data_path: str,
    checkpoint_path: str = "anra_brain.pt",
    batch_size: int = 64,
    block_size: int = 256,
    max_minutes: int = MAX_MINUTES,
    answer_loss_weight: float = ANSWER_LOSS_WEIGHT,
) -> None:
    repo = ROOT
    text = Path(data_path).read_text(encoding="utf-8", errors="replace")

    # Auto-generate tokenizer.pkl if missing.
    import pickle as _pickle
    import shutil as _shutil

    from tokenizer.char_tokenizer import CharTokenizer as _CharTok

    _tok_path = get_tokenizer_file()
    if not _tok_path.exists():
        _drive_tok = DRIVE_CHECKPOINTS.parent / "tokenizer.pkl"
        if _drive_tok.exists() and _drive_tok.stat().st_size > 100:
            _tok_path.parent.mkdir(parents=True, exist_ok=True)
            _shutil.copy2(_drive_tok, _tok_path)
            print("[build_brain] tokenizer.pkl restored from Drive", flush=True)
        else:
            print(f"[build_brain] Building tokenizer from {data_path} ...", flush=True)
            _raw = Path(data_path).read_text(encoding="utf-8", errors="replace")
            _new_tok = _CharTok(_raw)
            _tok_path.parent.mkdir(parents=True, exist_ok=True)
            with open(_tok_path, "wb") as _f:
                _pickle.dump(_new_tok, _f)
            try:
                _drive_dir = DRIVE_CHECKPOINTS.parent
                _drive_dir.mkdir(parents=True, exist_ok=True)
                _shutil.copy2(_tok_path, _drive_dir / "tokenizer.pkl")
                print(
                    f"[build_brain] tokenizer.pkl built + mirrored to Drive. vocab_size={_new_tok.vocab_size}",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"[build_brain] tokenizer.pkl built. vocab_size={_new_tok.vocab_size} (Drive: {exc})",
                    flush=True,
                )

    with open(get_tokenizer_file(), "rb") as f:
        tokenizer = pickle.load(f)

    ds = ConversationDataset(text, tokenizer, block_size, answer_loss_weight=answer_loss_weight)
    if len(ds) == 0:
        raise RuntimeError("ConversationDataset produced zero training windows.")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CausalTransformer(tokenizer.vocab_size, 256, 4, 4, block_size).to(device)

    compile_enabled = os.environ.get("ANRA_ENABLE_COMPILE", "").strip().lower() in {"1", "true", "yes", "on"}
    if compile_enabled and torch.cuda.is_available() and platform.system().lower() != "windows":
        try:
            from typing import cast

            print("[build_brain] torch.compile enabled", flush=True)
            model = cast(CausalTransformer, torch.compile(model))
        except Exception as exc:
            print(f"[build_brain] torch.compile unavailable ({exc}) - continuing without compile", flush=True)
    else:
        print("[build_brain] torch.compile disabled for faster startup", flush=True)

    mp = MixedPrecisionTrainer(device=device)
    optimizer = build_optimizer(model, lr=3e-4)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=200,
        total_steps=50_000,
    )

    ckpt_path = _resolve_checkpoint_path(checkpoint_path)
    global_step = 0
    epoch = 0
    best_loss = float("inf")

    if not ckpt_path.exists():
        fallback = get_checkpoint()
        if fallback and fallback.exists():
            shutil.copy2(fallback, ckpt_path)
            print(f"[build_brain] restored checkpoint: {fallback} -> {ckpt_path}", flush=True)
        else:
            drive_copy = DRIVE_CHECKPOINTS / ckpt_path.name
            if drive_copy.exists():
                shutil.copy2(drive_copy, ckpt_path)
                print(f"[build_brain] restored checkpoint from drive: {drive_copy}", flush=True)

    if ckpt_path.exists():
        try:
            state = torch.load(ckpt_path, map_location=device, weights_only=False)
            if isinstance(state, dict) and "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"], strict=False)
                try:
                    optimizer.load_state_dict(state["optimizer_state_dict"])
                except Exception:
                    pass
                global_step = int(state.get("global_step", 0))
                epoch = int(state.get("epoch", 0))
                best_loss = float(state.get("best_loss", float("inf")))
                print(
                    f"[build_brain] Resumed: step={global_step} epoch={epoch} best={best_loss:.4f}",
                    flush=True,
                )
            else:
                model.load_state_dict(state, strict=False)
                print(f"[build_brain] Loaded raw state dict from {ckpt_path}", flush=True)
        except Exception as exc:
            print(f"[build_brain] Checkpoint corrupted ({exc}) - training from scratch", flush=True)
            ckpt_path.unlink(missing_ok=True)

    start = time.time()
    end_at = start + max_minutes * 60
    initial_step = global_step
    optimizer.zero_grad(set_to_none=True)
    rolling_loss = 0.0
    rolling_count = 0
    last_avg_loss = best_loss
    reply_weighted_tokens = 0.0
    total_target_tokens = 0.0
    hard_examples: list[tuple[float, int]] = []
    first_batch_wall = None

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    gpu_mem = (
        torch.cuda.get_device_properties(0).total_memory / 1e9
        if torch.cuda.is_available()
        else 0.0
    )
    param_count = sum(p.numel() for p in model.parameters())
    eff_batch = batch_size * GRAD_ACCUM_STEPS

    print("", flush=True)
    print("=" * 58, flush=True)
    print("  AN-RA TRAINING SESSION", flush=True)
    print("=" * 58, flush=True)
    print(f"  GPU          : {gpu_name} ({gpu_mem:.1f} GB)", flush=True)
    print(f"  Parameters   : {param_count:,}", flush=True)
    print(
        f"  Micro batch  : {batch_size}  |  Grad accum : {GRAD_ACCUM_STEPS}  |  Eff batch : {eff_batch}",
        flush=True,
    )
    print(f"  Session time : {max_minutes} minutes", flush=True)
    print(f"  Resuming     : step {global_step:,}  |  best loss {best_loss:.4f}", flush=True)
    print(f"  Checkpoint   : {ckpt_path}", flush=True)
    print("=" * 58, flush=True)
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
                bsz, seq_len, channels = logits.shape
                per_token_loss = F.cross_entropy(
                    logits.view(bsz * seq_len, channels),
                    yb.view(bsz * seq_len),
                    reduction="none",
                ).view(bsz, seq_len)
                sample_losses = (per_token_loss * wb).sum(dim=1) / wb.sum(dim=1).clamp_min(1.0)
                loss = sample_losses.mean() / GRAD_ACCUM_STEPS

            mp.backward(loss)
            rolling_loss += float(loss.item() * GRAD_ACCUM_STEPS)
            rolling_count += 1
            reply_weighted_tokens += float((wb > 1.0).sum().item())
            total_target_tokens += float(wb.numel())

            for sample_loss, sample_index in zip(sample_losses.detach().cpu().tolist(), sample_idx.tolist()):
                entry = (float(sample_loss), int(sample_index))
                if len(hard_examples) < HARD_EXAMPLE_KEEP:
                    heapq.heappush(hard_examples, entry)
                elif entry[0] > hard_examples[0][0]:
                    heapq.heapreplace(hard_examples, entry)

            if rolling_count % GRAD_ACCUM_STEPS == 0:
                mp.step(optimizer)
                mp.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                avg_loss = rolling_loss / max(1, rolling_count)
                last_avg_loss = avg_loss
                best_loss = min(best_loss, avg_loss)

                if global_step in EARLY_STATUS_STEPS or global_step % 200 == 0:
                    elapsed_min = (time.time() - start) / 60.0
                    remaining_min = max(0.0, (end_at - time.time()) / 60.0)
                    avg_loss = rolling_loss / max(1, rolling_count)
                    startup_note = ""
                    if global_step in EARLY_STATUS_STEPS and first_batch_wall is not None:
                        startup_note = f"  startup={(first_batch_wall - start):.1f}s"
                    print(
                        f"  step={global_step:6d}  loss={avg_loss:.4f}  "
                        f"best={best_loss:.4f}  elapsed={elapsed_min:.1f}m  remaining={remaining_min:.1f}m"
                        f"{startup_note}",
                        flush=True,
                    )

            if time.time() >= end_at:
                break

    if global_step > initial_step and global_step % 200 != 0:
        elapsed_min = (time.time() - start) / 60.0
        remaining_min = max(0.0, (end_at - time.time()) / 60.0)
        print(
            f"  step={global_step:6d}  loss={last_avg_loss:.4f}  "
            f"best={best_loss:.4f}  elapsed={elapsed_min:.1f}m  remaining={remaining_min:.1f}m",
            flush=True,
        )

    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
        "epoch": epoch,
        "best_loss": best_loss,
    }
    _atomic_save(payload, ckpt_path, drive_dir=DRIVE_CHECKPOINTS)
    _sync_required_artifacts(ckpt_path)

    report = {
        "elapsed_minutes": round((time.time() - start) / 60.0, 2),
        "session_minutes_target": max_minutes,
        "global_step": global_step,
        "epoch": epoch,
        "best_loss": best_loss,
        "effective_batch_size": batch_size * GRAD_ACCUM_STEPS,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "answer_loss_weight": answer_loss_weight,
        "answer_supervision_ratio": round(ds.answer_supervision_ratio, 4),
        "reply_token_ratio_seen": round(reply_weighted_tokens / max(1.0, total_target_tokens), 4),
        "model_config": {
            "vocab_size": tokenizer.vocab_size,
            "n_embd": 256,
            "n_head": 4,
            "n_layer": 4,
            "block_size": block_size,
        },
        "scheduler": {"type": "cosine_warmup", "warmup_steps": 200, "total_steps": 50_000},
        "precision": {"amp_enabled": True, "optimizer": "Muon_or_AdamW_fallback"},
        "compiled_model": compile_enabled,
        "checkpoint_path": str(ckpt_path),
    }
    (repo / "output").mkdir(parents=True, exist_ok=True)
    (repo / "output" / "session_train_metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    hard_examples_report = [
        {
            "loss": round(loss_value, 4),
            "sample_index": sample_index,
            "preview": ds.snippet(sample_index),
        }
        for loss_value, sample_index in sorted(hard_examples, key=lambda item: item[0], reverse=True)
    ]
    (repo / "output" / "hard_examples.json").write_text(
        json.dumps(
            {
                "generated_at": time.time(),
                "answer_loss_weight": answer_loss_weight,
                "examples": hard_examples_report,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    elapsed_total = time.time() - start
    print("", flush=True)
    print("=" * 58, flush=True)
    print("  SESSION COMPLETE", flush=True)
    print("=" * 58, flush=True)
    print(f"  Steps this session : {global_step - initial_step:,}", flush=True)
    print(f"  Total steps        : {global_step:,}", flush=True)
    print(f"  Best loss          : {best_loss:.4f}", flush=True)
    print(f"  Time elapsed       : {elapsed_total / 60:.1f} minutes", flush=True)
    print(f"  Checkpoint saved   : {ckpt_path}", flush=True)
    print("  Drive synced       : yes", flush=True)
    print(f"  Next session       : resumes from step {global_step:,}", flush=True)
    print("=" * 58, flush=True)
    print("", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--checkpoint_path", default="anra_brain.pt")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--max_minutes", type=int, default=MAX_MINUTES)
    parser.add_argument("--answer_loss_weight", type=float, default=ANSWER_LOSS_WEIGHT)
    args = parser.parse_args()
    train_anra_brain(
        data_path=args.data_path,
        checkpoint_path=args.checkpoint_path,
        batch_size=args.batch_size,
        block_size=args.block_size,
        max_minutes=args.max_minutes,
        answer_loss_weight=args.answer_loss_weight,
    )
