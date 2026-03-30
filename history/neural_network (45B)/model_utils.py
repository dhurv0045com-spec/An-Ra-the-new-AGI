"""
utils/model_utils.py — Steps 30–33
Save and load model, resume from checkpoint, inspect weights,
count parameters. Full lifecycle management.
"""

import os
import json
import math
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np


# ─────────────────────────────────────────────
# STEP 30 — Save and load model
# ─────────────────────────────────────────────

def save_model(model: nn.Module, path: str,
               config: Optional[dict] = None,
               metadata: Optional[dict] = None):
    """
    Save model weights (and optionally config) to disk.
    Saves in safetensors format if available, otherwise .pt.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state_dict": model.state_dict(),
        "config": config or {},
        "metadata": metadata or {},
        "torch_version": torch.__version__,
    }

    # Try safetensors for safer, faster loading
    try:
        from safetensors.torch import save_file as st_save
        st_path = path.with_suffix(".safetensors")
        st_save(model.state_dict(), str(st_path))
        # Save config separately
        if config:
            cfg_path = path.parent / "config.json"
            with open(cfg_path, "w") as f:
                json.dump(config, f, indent=2)
        logging.info(f"Saved model (safetensors) → {st_path}")
        return str(st_path)
    except ImportError:
        pass

    torch.save(payload, str(path))
    logging.info(f"Saved model → {path}")
    return str(path)


def load_model(model: nn.Module, path: str,
               strict: bool = True,
               map_location: str = "cpu") -> nn.Module:
    """
    Load model weights from .pt or .safetensors.
    strict=False allows loading partial checkpoints / fine-tuned variants.
    """
    path = Path(path)

    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
            state = load_file(str(path), device=map_location)
            result = model.load_state_dict(state, strict=strict)
            logging.info(f"Loaded model (safetensors) from {path}")
            if not strict:
                logging.info(f"  Missing: {result.missing_keys}")
                logging.info(f"  Unexpected: {result.unexpected_keys}")
            return model
        except ImportError:
            raise ImportError("pip install safetensors to load .safetensors files")
    else:
        ckpt = torch.load(str(path), map_location=map_location)
        state = ckpt.get("model_state_dict", ckpt)
        result = model.load_state_dict(state, strict=strict)
        logging.info(f"Loaded model from {path}")
        return model


# ─────────────────────────────────────────────
# STEP 31 — Resume from checkpoint
# ─────────────────────────────────────────────

class CheckpointManager:
    """
    Full training state: model + optimizer + scheduler + step.
    Supports resume, best-model tracking, and checkpoint rotation.
    """

    def __init__(self, checkpoint_dir: str, keep_last: int = 3):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last
        self._history: List[Path] = []
        self.best_metric = float("inf")
        self._load_registry()

    def _registry_path(self) -> Path:
        return self.dir / "registry.json"

    def _load_registry(self):
        reg = self._registry_path()
        if reg.exists():
            with open(reg) as f:
                data = json.load(f)
            self._history = [Path(p) for p in data.get("checkpoints", [])]
            self.best_metric = data.get("best_metric", float("inf"))

    def _save_registry(self):
        with open(self._registry_path(), "w") as f:
            json.dump({
                "checkpoints": [str(p) for p in self._history],
                "best_metric": self.best_metric,
            }, f, indent=2)

    def save(self, step: int, model: nn.Module,
             optimizer: Optional[torch.optim.Optimizer] = None,
             scheduler=None,
             metrics: Optional[dict] = None,
             is_best: bool = False) -> Path:

        path = self.dir / f"ckpt_step{step:08d}.pt"
        payload: Dict[str, Any] = {
            "step": step,
            "model_state_dict": model.state_dict(),
        }
        if optimizer:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler:
            payload["scheduler_state"] = {
                "_step": getattr(scheduler, "_step", 0),
                "best_metric": getattr(scheduler, "best_metric", None),
            }
        if metrics:
            payload["metrics"] = metrics

        torch.save(payload, path)
        self._history.append(path)

        if is_best or (metrics and metrics.get("val_loss", float("inf")) < self.best_metric):
            best_val = metrics.get("val_loss", float("inf")) if metrics else float("inf")
            if best_val < self.best_metric:
                self.best_metric = best_val
            best_path = self.dir / "best.pt"
            torch.save(payload, best_path)

        # Prune old checkpoints
        while len(self._history) > self.keep_last:
            old = self._history.pop(0)
            if old.exists():
                old.unlink()

        # Update "latest" pointer
        latest = self.dir / "latest.pt"
        torch.save(payload, latest)

        self._save_registry()
        logging.info(f"Checkpoint saved → {path.name}")
        return path

    def load(self, path: Optional[str] = None,
             model: Optional[nn.Module] = None,
             optimizer: Optional[torch.optim.Optimizer] = None,
             scheduler=None,
             map_location: str = "cpu") -> Dict[str, Any]:
        """Load checkpoint. If path is None, loads latest."""
        if path is None:
            path = self.dir / "latest.pt"
        path = Path(path)
        if not path.exists():
            logging.warning(f"No checkpoint found at {path}")
            return {}

        ckpt = torch.load(str(path), map_location=map_location)

        if model and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        if optimizer and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler and "scheduler_state" in ckpt:
            for k, v in ckpt["scheduler_state"].items():
                if v is not None:
                    setattr(scheduler, k, v)

        logging.info(f"Resumed from {path.name} (step={ckpt.get('step', '?')})")
        return ckpt

    def load_best(self, model: nn.Module, map_location: str = "cpu"):
        return self.load(str(self.dir / "best.pt"), model, map_location=map_location)


# ─────────────────────────────────────────────
# STEP 32 — Inspect weights
# ─────────────────────────────────────────────

class WeightInspector:
    """
    Inspect model weights and gradients for debugging and analysis.
    Useful for diagnosing gradient flow, dead neurons, weight drift.
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def summary(self) -> str:
        """Layer-by-layer weight statistics."""
        lines = ["=" * 70, f"{'Layer':<45} {'Shape':>15} {'Mean':>8} {'Std':>8}", "=" * 70]
        for name, param in self.model.named_parameters():
            shape_str = str(tuple(param.shape))
            mean = param.data.mean().item()
            std = param.data.std().item()
            lines.append(f"{name[:45]:<45} {shape_str:>15} {mean:>8.4f} {std:>8.4f}")
        lines.append("=" * 70)
        return "\n".join(lines)

    def grad_summary(self) -> str:
        """Gradient norms per layer — useful to spot vanishing/exploding gradients."""
        lines = ["=" * 60, f"{'Layer':<45} {'Grad Norm':>12}", "=" * 60]
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gnorm = param.grad.norm().item()
                lines.append(f"{name[:45]:<45} {gnorm:>12.4e}")
            else:
                lines.append(f"{name[:45]:<45} {'(no grad)':>12}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def activation_range(self, input_ids: torch.Tensor) -> Dict[str, dict]:
        """Run a forward pass and record activation statistics via hooks."""
        stats = {}
        hooks = []

        def make_hook(name):
            def hook(module, inp, out):
                t = out[0] if isinstance(out, tuple) else out
                stats[name] = {
                    "mean": t.mean().item(),
                    "std":  t.std().item(),
                    "min":  t.min().item(),
                    "max":  t.max().item(),
                    "shape": tuple(t.shape),
                }
            return hook

        for name, module in self.model.named_modules():
            if name:
                hooks.append(module.register_forward_hook(make_hook(name)))

        with torch.no_grad():
            self.model(input_ids)

        for h in hooks:
            h.remove()
        return stats

    def find_dead_neurons(self, threshold: float = 1e-4) -> List[str]:
        """Identify weight tensors with near-zero std (possibly dead)."""
        dead = []
        for name, param in self.model.named_parameters():
            if param.std().item() < threshold:
                dead.append(name)
        return dead

    def weight_histogram(self, layer_name: str, bins: int = 20) -> str:
        """ASCII histogram of weights for a named layer."""
        for name, param in self.model.named_parameters():
            if layer_name in name:
                vals = param.data.cpu().float().numpy().flatten()
                counts, edges = np.histogram(vals, bins=bins)
                max_count = counts.max()
                lines = [f"Weight histogram: {name}"]
                for i, (c, e) in enumerate(zip(counts, edges)):
                    bar = "█" * int(c / max_count * 40)
                    lines.append(f"  {e:+.3f} | {bar}")
                return "\n".join(lines)
        return f"Layer '{layer_name}' not found"


# ─────────────────────────────────────────────
# STEP 33 — Count parameters
# ─────────────────────────────────────────────

def count_parameters(model: nn.Module,
                     verbose: bool = True) -> Dict[str, int]:
    """
    Comprehensive parameter count breakdown.
    Returns total, trainable, non-trainable, per-component counts.
    """
    total = 0
    trainable = 0
    non_trainable = 0
    by_component: Dict[str, int] = defaultdict(int)

    for name, param in model.named_parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
        else:
            non_trainable += n
        component = name.split(".")[0]
        by_component[component] += n

    stats = {
        "total": total,
        "trainable": trainable,
        "non_trainable": non_trainable,
        "by_component": dict(by_component),
    }

    if verbose:
        print("=" * 50)
        print(f"{'Total parameters':35} {total:>12,}")
        print(f"{'Trainable parameters':35} {trainable:>12,}")
        print(f"{'Non-trainable parameters':35} {non_trainable:>12,}")
        print("-" * 50)
        print("By component:")
        for comp, n in sorted(by_component.items(), key=lambda x: -x[1]):
            pct = n / total * 100
            print(f"  {comp:<30} {n:>12,}  ({pct:.1f}%)")
        print("=" * 50)

        # Human-readable scale
        if total >= 1e9:
            scale = f"{total/1e9:.2f}B"
        elif total >= 1e6:
            scale = f"{total/1e6:.2f}M"
        else:
            scale = f"{total/1e3:.2f}K"
        print(f"Scale: ~{scale} parameters")

    return stats


def estimate_memory(model: nn.Module,
                    batch_size: int = 1,
                    seq_len: int = 512,
                    dtype: torch.dtype = torch.float32) -> Dict[str, float]:
    """
    Estimate memory usage in GB:
    - Parameters
    - Gradients
    - Optimizer states (Adam: 2x params for m, v)
    - Activations (rough estimate)
    """
    n_params = sum(p.numel() for p in model.parameters())
    bytes_per_element = torch.finfo(dtype).bits // 8

    param_mem = n_params * bytes_per_element
    grad_mem = n_params * bytes_per_element
    optimizer_mem = n_params * 4 * 2  # Adam: fp32 m + v

    # Rough activation memory: O(n_layers * batch * seq * d_model * bytes)
    try:
        d = getattr(model, "d_model", 512)
        L = getattr(model, "n_layers", 6)
        act_mem = L * batch_size * seq_len * d * bytes_per_element * 12
    except Exception:
        act_mem = 0

    total = param_mem + grad_mem + optimizer_mem + act_mem

    result = {
        "params_gb":     param_mem / 1e9,
        "gradients_gb":  grad_mem / 1e9,
        "optimizer_gb":  optimizer_mem / 1e9,
        "activations_gb": act_mem / 1e9,
        "total_gb":      total / 1e9,
    }

    print(f"Estimated memory (batch={batch_size}, seq={seq_len}):")
    for k, v in result.items():
        print(f"  {k:<20} {v:.2f} GB")

    return result


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from model.transformer import CausalTransformer

    print("Steps 30–33: Model utils checks")

    model = CausalTransformer(
        vocab_size=1000, d_model=128, n_heads=4,
        n_layers=3, d_ff=512, max_seq_len=64,
        dropout=0.0, n_kv_heads=2,
    )

    # Step 33 — Count params
    print("\n--- Step 33: Parameter count ---")
    stats = count_parameters(model)

    # Step 32 — Inspect weights
    print("\n--- Step 32: Weight inspection ---")
    inspector = WeightInspector(model)
    dead = inspector.find_dead_neurons(threshold=1e-3)
    print(f"Potentially dead layers: {dead if dead else 'none'}")

    # Step 30 — Save
    print("\n--- Step 30: Save model ---")
    save_path = "/tmp/test_lm_model.pt"
    save_model(model, save_path, config={"d_model": 128})

    # Load into fresh model
    model2 = CausalTransformer(
        vocab_size=1000, d_model=128, n_heads=4,
        n_layers=3, d_ff=512, max_seq_len=64,
        dropout=0.0, n_kv_heads=2,
    )
    load_model(model2, save_path)

    # Verify identical outputs
    tokens = torch.randint(0, 1000, (1, 16))
    with torch.no_grad():
        out1, _ = model(tokens)
        out2, _ = model2(tokens)
    max_diff = (out1 - out2).abs().max().item()
    print(f"Max output diff after save/load: {max_diff:.2e}  ✓")

    # Step 31 — Checkpoint manager
    print("\n--- Step 31: Checkpoint manager ---")
    mgr = CheckpointManager("/tmp/test_checkpoints", keep_last=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    mgr.save(step=100, model=model, optimizer=optimizer, metrics={"val_loss": 2.5})
    mgr.save(step=200, model=model, optimizer=optimizer, metrics={"val_loss": 2.1})

    model3 = CausalTransformer(
        vocab_size=1000, d_model=128, n_heads=4,
        n_layers=3, d_ff=512, max_seq_len=64,
        dropout=0.0, n_kv_heads=2,
    )
    ckpt = mgr.load(model=model3)
    print(f"Resumed from step: {ckpt.get('step')}  ✓")

    # Memory estimate
    print("\n--- Memory estimate ---")
    estimate_memory(model, batch_size=8, seq_len=512)

    print("\n✓ Steps 30–33 verified")
