"""
45G / Step 32 — Model Save / Load / Export
============================================
Complete model persistence layer.

  save_checkpoint  — full model state + metadata to a single .pt file
  load_checkpoint  — restore model in one call; strict=False for partial loads
  export_onnx      — export to ONNX for inference serving / mobile / edge
  list_checkpoints — scan directory and show version history
  CheckpointMetadata — structured versioning with arch config + training stats

File format:
  {
      "model_state":   OrderedDict,      # model.state_dict()
      "metadata":      dict,             # arch, dataset, date, stats
      "config":        dict,             # GenerationConfig defaults
      "tokenizer":     dict | None,      # vocab if tokenizer is serialisable
      "torch_version": str,
      "format_version": int,             # bumped on breaking changes
  }
"""

from __future__ import annotations

import json
import os
import datetime
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

FORMAT_VERSION = 1


# ─────────────────────────────────────────────
# Metadata schema
# ─────────────────────────────────────────────

@dataclass
class CheckpointMetadata:
    """
    Versioned metadata attached to every saved checkpoint.
    Serialised as plain dict inside the .pt file.
    """
    # Architecture
    model_class: str = "unknown"
    d_model: int = 0
    n_layers: int = 0
    n_heads: int = 0
    d_ff: int = 0
    vocab_size: int = 0
    max_seq_len: int = 0

    # Training
    dataset: str = "unknown"
    epoch: int = 0
    global_step: int = 0
    train_loss: float = float("inf")
    val_loss: float = float("inf")
    best_val_loss: float = float("inf")
    tokens_seen: int = 0

    # Provenance
    date_trained: str = field(
        default_factory=lambda: datetime.datetime.utcnow().isoformat()
    )
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CheckpointMetadata":
        valid = {k: v for k, v in d.items()
                 if k in CheckpointMetadata.__dataclass_fields__}
        return CheckpointMetadata(**valid)


# ─────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    path: str | Path,
    metadata: Optional[CheckpointMetadata] = None,
    tokenizer=None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    config: Optional[Dict[str, Any]] = None,
    overwrite: bool = True,
) -> Path:
    """
    Serialise model (and optionally optimizer + tokenizer) to a single file.

    Args:
        model:      The nn.Module to save.
        path:       Destination .pt path.
        metadata:   CheckpointMetadata instance with training provenance.
        tokenizer:  Optional tokenizer; saved if it exposes a to_dict() method.
        optimizer:  Optional optimizer state (for training resumption).
        config:     Arbitrary dict of generation defaults.
        overwrite:  If False, appends a timestamp suffix instead of clobbering.

    Returns:
        Resolved Path of the saved file.
    """
    path = Path(path)
    if not overwrite and path.exists():
        stem, suffix = path.stem, path.suffix
        ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = path.with_name(f"{stem}_{ts}{suffix}")

    path.parent.mkdir(parents=True, exist_ok=True)

    # Serialise tokenizer if possible
    tok_dict = None
    if tokenizer is not None:
        if hasattr(tokenizer, "to_dict"):
            tok_dict = tokenizer.to_dict()
        elif hasattr(tokenizer, "vocab"):
            tok_dict = {"vocab": list(tokenizer.vocab)}

    payload: Dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "torch_version":  torch.__version__,
        "model_state":    model.state_dict(),
        "metadata":       (metadata or CheckpointMetadata()).to_dict(),
        "config":         config or {},
        "tokenizer":      tok_dict,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()

    torch.save(payload, path)
    size_mb = path.stat().st_size / 1_048_576
    print(f"[model_io] Saved → {path}  ({size_mb:.1f} MB)")
    return path


# ─────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────

def load_checkpoint(
    model: nn.Module,
    path: str | Path,
    device: Optional[str] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    strict: bool = True,
) -> CheckpointMetadata:
    """
    Load model weights from a checkpoint.

    Args:
        model:    nn.Module to populate — must match the saved architecture
                  when strict=True.
        path:     Checkpoint .pt file.
        device:   Map location ("cpu", "cuda", etc.). Auto-detected if None.
        optimizer: If provided, optimizer state is also restored.
        strict:   Passed to load_state_dict. False allows partial loading.

    Returns:
        CheckpointMetadata from the checkpoint for inspection.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(path, map_location=device, weights_only=False)

    # Forward-compat: warn on version mismatch but don't crash
    saved_ver = payload.get("format_version", 0)
    if saved_ver != FORMAT_VERSION:
        print(f"[model_io] Warning: checkpoint format v{saved_ver}, "
              f"current v{FORMAT_VERSION}")

    missing, unexpected = model.load_state_dict(
        payload["model_state"], strict=strict
    )
    if missing:
        print(f"[model_io] Missing keys ({len(missing)}): {missing[:5]} ...")
    if unexpected:
        print(f"[model_io] Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")

    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
        print("[model_io] Optimizer state restored.")

    meta = CheckpointMetadata.from_dict(payload.get("metadata", {}))
    size_mb = path.stat().st_size / 1_048_576
    print(f"[model_io] Loaded ← {path}  ({size_mb:.1f} MB)")
    print(f"           epoch={meta.epoch}  val_loss={meta.val_loss:.4f}  "
          f"date={meta.date_trained[:10]}")
    return meta


# ─────────────────────────────────────────────
# ONNX export
# ─────────────────────────────────────────────

def export_onnx(
    model: nn.Module,
    path: str | Path,
    seq_len: int = 16,
    vocab_size: int = 256,
    opset: int = 17,
    dynamic_axes: bool = True,
) -> Path:
    """
    Export model to ONNX for deployment (TensorRT, ONNX Runtime, mobile, etc.).

    Args:
        model:         nn.Module to export.
        path:          Output .onnx path.
        seq_len:       Sequence length for the example input trace.
        vocab_size:    Vocabulary size for dummy input generation.
        opset:         ONNX opset version.
        dynamic_axes:  Allow variable batch and sequence dimensions.

    Returns:
        Path of the written .onnx file.
    """
    try:
        import onnx  # optional dependency
    except ImportError:
        raise ImportError("pip install onnx onnxruntime to use export_onnx()")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    dummy_input = torch.randint(0, min(vocab_size, 64), (1, seq_len))

    axes = {"input_ids": {0: "batch", 1: "seq_len"},
            "logits":    {0: "batch", 1: "seq_len"}} if dynamic_axes else {}

    torch.onnx.export(
        model,
        dummy_input,
        str(path),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes=axes if dynamic_axes else None,
        opset_version=opset,
        do_constant_folding=True,
    )

    size_mb = path.stat().st_size / 1_048_576
    print(f"[model_io] ONNX export → {path}  ({size_mb:.1f} MB)")
    return path


# ─────────────────────────────────────────────
# Checkpoint directory scanner
# ─────────────────────────────────────────────

def list_checkpoints(directory: str | Path) -> List[Dict[str, Any]]:
    """
    Scan a directory for .pt checkpoint files, read their metadata,
    and return a list sorted by val_loss (ascending).

    Useful for picking the best checkpoint for inference.
    """
    directory = Path(directory)
    rows = []
    for ckpt in sorted(directory.glob("*.pt")):
        try:
            payload = torch.load(ckpt, map_location="cpu", weights_only=False)
            meta = payload.get("metadata", {})
            rows.append({
                "file":       ckpt.name,
                "size_mb":    round(ckpt.stat().st_size / 1_048_576, 1),
                "epoch":      meta.get("epoch", "?"),
                "val_loss":   meta.get("val_loss", float("inf")),
                "train_loss": meta.get("train_loss", float("inf")),
                "date":       meta.get("date_trained", "")[:10],
                "notes":      meta.get("notes", ""),
            })
        except Exception as e:
            rows.append({"file": ckpt.name, "error": str(e)})

    rows.sort(key=lambda r: r.get("val_loss", float("inf")))
    return rows


def print_checkpoint_table(directory: str | Path) -> None:
    """Print a human-readable table of available checkpoints."""
    rows = list_checkpoints(directory)
    if not rows:
        print(f"No checkpoints found in {directory}")
        return
    header = f"{'File':<35} {'MB':>6} {'Epoch':>6} {'ValLoss':>9} {'Date':<12}"
    print(header)
    print("─" * len(header))
    for r in rows:
        if "error" in r:
            print(f"  {r['file']}  ERROR: {r['error']}")
        else:
            print(f"{r['file']:<35} {r['size_mb']:>6.1f} "
                  f"{str(r['epoch']):>6} {r['val_loss']:>9.4f} {r['date']:<12}")


# ─────────────────────────────────────────────
# Smoke-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile
    import torch.nn as nn

    class _TinyLM(nn.Module):
        def __init__(self): super().__init__(); self.fc = nn.Linear(16, 64)
        def forward(self, x): return self.fc(x.float())

    model = _TinyLM()
    meta  = CheckpointMetadata(
        model_class="TinyLM", d_model=16, vocab_size=64,
        epoch=5, train_loss=0.42, val_loss=0.55,
        dataset="test_corpus", notes="smoke-test"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = Path(tmpdir) / "model_v1.pt"

        # Save
        saved_path = save_checkpoint(model, ckpt, metadata=meta)
        assert saved_path.exists()
        print("save_checkpoint ✓")

        # Load
        model2 = _TinyLM()
        loaded_meta = load_checkpoint(model2, ckpt, device="cpu")
        assert loaded_meta.epoch == 5
        assert loaded_meta.dataset == "test_corpus"
        # Verify weights transferred exactly
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            assert torch.allclose(p1, p2), f"Weight mismatch: {n1}"
        print("load_checkpoint ✓")

        # List
        rows = list_checkpoints(tmpdir)
        assert len(rows) == 1 and rows[0]["epoch"] == 5
        print_checkpoint_table(tmpdir)
        print("list_checkpoints ✓")

    print("\nStep 32 — save / load / export ✓")
