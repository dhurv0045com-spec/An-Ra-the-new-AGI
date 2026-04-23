from __future__ import annotations

import argparse
import pickle
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anra_paths import ROOT, inject_all_paths, get_tokenizer_file
inject_all_paths()

from anra_brain import CausalTransformer
from training.anra_optimizer import build_optimizer
from training.checkpoint import CheckpointManager, CheckpointMeta
from training.mixed_precision import MixedPrecisionTrainer
from training.scheduler import get_cosine_schedule_with_warmup

GRAD_ACCUM_STEPS = 8
MAX_MINUTES = 30
SAVE_EVERY = 500
PRINT_EVERY = 100


class TextDataset(Dataset):
    def __init__(self, data_tensor: torch.Tensor, block_size: int):
        self.data = data_tensor
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size - 1

    def __getitem__(self, index: int):
        x = self.data[index: index + self.block_size]
        y = self.data[index + 1: index + self.block_size + 1]
        return x, y


def _atomic_save(payload: dict, output_path: Path, drive_dir: Path | None = None):
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(output_path)
    if drive_dir is not None:
        drive_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_path, drive_dir / output_path.name)


def train_anra_brain(data_path: str, checkpoint_path: str = "anra_brain.pt", batch_size: int = 64, block_size: int = 128):
    repo = ROOT
    text = Path(data_path).read_text(encoding="utf-8", errors="replace")
    with open(get_tokenizer_file(), "rb") as f:
        tokenizer = pickle.load(f)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    ds = TextDataset(data, block_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CausalTransformer(tokenizer.vocab_size, 256, 4, 4, block_size).to(device)
    try:
        from typing import cast
        model = cast(CausalTransformer, torch.compile(model))
    except Exception:
        pass

    mp = MixedPrecisionTrainer(device=device)
    optimizer = build_optimizer(model, lr=3e-4)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps=200, total_steps=50_000)

    ckpt_mgr = CheckpointManager(checkpoint_dir=str(repo / "checkpoints"), keep_last_n=3)
    ckpt_path = Path(checkpoint_path)
    global_step = 0
    epoch = 0
    best_loss = float("inf")

    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"], strict=False)
            if "optimizer_state_dict" in state:
                optimizer.load_state_dict(state["optimizer_state_dict"])
            global_step = int(state.get("global_step", 0))
            epoch = int(state.get("epoch", 0))
            best_loss = float(state.get("best_loss", float("inf")))
        else:
            model.load_state_dict(state, strict=False)

    start = time.time()
    end_at = start + MAX_MINUTES * 60
    optimizer.zero_grad(set_to_none=True)
    rolling_loss = 0.0
    rolling_count = 0

    while time.time() < end_at:
        epoch += 1
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            with mp.autocast():
                logits, _ = model(xb)
                b, t, c = logits.shape
                loss = F.cross_entropy(logits.view(b * t, c), yb.view(b * t)) / GRAD_ACCUM_STEPS
            mp.backward(loss)
            rolling_loss += float(loss.item() * GRAD_ACCUM_STEPS)
            rolling_count += 1

            if rolling_count % GRAD_ACCUM_STEPS == 0:
                mp.step(optimizer)
                mp.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                avg_loss = rolling_loss / max(1, rolling_count)
                best_loss = min(best_loss, avg_loss)

                if global_step % PRINT_EVERY == 0:
                    print(f"step={global_step} loss={avg_loss:.4f} best={best_loss:.4f}")

                if global_step % SAVE_EVERY == 0:
                    payload = {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "global_step": global_step,
                        "epoch": epoch,
                        "best_loss": best_loss,
                    }
                    _atomic_save(payload, ckpt_path)
                    ckpt_mgr.save(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=mp.scaler,
                        meta=CheckpointMeta(
                            epoch=epoch,
                            global_step=global_step,
                            train_loss=avg_loss,
                            val_loss=avg_loss,
                            best_val_loss=best_loss,
                            model_config={"vocab_size": tokenizer.vocab_size, "n_embd": 256, "n_head": 4, "n_layer": 4, "block_size": block_size},
                        ),
                    )
            if time.time() >= end_at:
                break

    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
        "epoch": epoch,
        "best_loss": best_loss,
    }
    _atomic_save(payload, ckpt_path)
    print(f"saved {ckpt_path} at step={global_step} epoch={epoch} best={best_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--checkpoint_path", default="anra_brain.pt")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--block_size", type=int, default=128)
    args = parser.parse_args()
    train_anra_brain(
        data_path=args.data_path,
        checkpoint_path=args.checkpoint_path,
        batch_size=args.batch_size,
        block_size=args.block_size,
    )
