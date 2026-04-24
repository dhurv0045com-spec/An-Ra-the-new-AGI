from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from anra_paths import ROOT, DRIVE_CHECKPOINTS, get_tokenizer_file
from anra_brain import CausalTransformer


def _atomic_save(payload: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _copy_to_drive(path: Path) -> None:
    try:
        DRIVE_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, DRIVE_CHECKPOINTS / path.name)
    except Exception:
        pass


def _load_model(base_model: str, device: torch.device):
    import pickle
    with open(get_tokenizer_file(), "rb") as f:
        tok = pickle.load(f)
    model = CausalTransformer(tok.vocab_size, 256, 4, 4, 128).to(device)
    state = torch.load(base_model, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    return model, tok


def main() -> None:
    ap = argparse.ArgumentParser(description="Ouroboros phased trainer")
    ap.add_argument("--base_model", default="anra_brain_identity.pt")
    ap.add_argument("--output", default="anra_ouroboros.pt")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = _load_model(args.base_model, device)
    text = (ROOT / "training_data" / "anra_dataset_v6_1.txt").read_text(encoding="utf-8", errors="replace")
    ids = torch.tensor(tok.encode(text), dtype=torch.long)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    best_loss = float("inf")
    seq_len = 128

    for step in range(args.steps):
        idx = torch.randint(0, len(ids) - seq_len - 1, (args.batch_size,))
        batch = torch.stack([ids[i:i + seq_len + 1] for i in idx]).to(device)
        x, y = batch[:, :-1], batch[:, 1:]

        optimizer.zero_grad(set_to_none=True)
        logits, loss_lm = model(x, y)
        assert loss_lm is not None

        # Phased auxiliary behavior (A/B/C schedule)
        aux = torch.tensor(0.0, device=device)
        if 1000 <= step < 3000:
            # pass consistency proxy: adjacent token stability
            aux = ((logits[:, 1:, :] - logits[:, :-1, :]) ** 2).mean() * 0.10
        elif step >= 3000:
            p = torch.softmax(logits, dim=-1)
            entropy = -(p * torch.log(p + 1e-10)).sum(-1).mean()
            consistency = ((logits[:, 1:, :] - logits[:, :-1, :]) ** 2).mean()
            aux = consistency * 0.05 + entropy * 0.05

        loss = loss_lm + aux
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        best_loss = min(best_loss, float(loss.item()))

        if (step + 1) % 500 == 0:
            out = Path(args.output)
            payload = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": step + 1,
                "best_loss": best_loss,
                "n_passes": 3,
            }
            _atomic_save(payload, out)
            _copy_to_drive(out)
            print(f"step={step+1} loss={loss.item():.4f} best={best_loss:.4f}")

    final = Path(args.output)
    _atomic_save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": args.steps,
            "best_loss": best_loss,
            "n_passes": 3,
        },
        final,
    )
    _copy_to_drive(final)
    print(f"saved {final}")


if __name__ == "__main__":
    main()
