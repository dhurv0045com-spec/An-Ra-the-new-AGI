"""
lora/lora.py — Low-Rank Adaptation (LoRA) Fine-Tuning

Fine-tune a large frozen model by training only small rank-decomposition
matrices injected into the attention projections.

MATH:
    Original weight: W  shape (out, in)
    LoRA delta:      ΔW = B @ A   where A: (r, in),  B: (out, r),  r << min(in, out)
    Forward:         y = x @ (W + ΔW).T  =  x @ W.T  +  x @ A.T @ B.T

    Only A and B are trained. W is frozen.
    Parameter savings: instead of out*in params, train r*(in+out) params.
    For d_model=512, r=8: 512*512=262144 → 8*(512+512)=8192  (32× fewer)

Why this matters:
    You can fine-tune a 350M param model on a CPU in minutes.
    New skills load without touching the base weights.
    Multiple LoRA adapters can be swapped without reloading the model.
"""

import numpy as np
import math, os, pickle, json, time
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, asdict, field


CKPT_DIR = Path("checkpoints/lora")
CKPT_DIR.mkdir(parents=True, exist_ok=True)


# ── LoRA Parameter ─────────────────────────────────────────────────────────────

class LoRAParam:
    """
    A LoRA adapter for one weight matrix.
    Stores A (down-projection) and B (up-projection).
    B is initialized to zero so ΔW = 0 at init — no change to base model.
    A is initialized with random Gaussian values scaled by 1/√r.
    """

    def __init__(self, in_features: int, out_features: int,
                 rank: int = 8, alpha: float = 16.0):
        """
        Args:
            in_features:  input dimension of the weight being adapted
            out_features: output dimension
            rank:         LoRA rank r — lower = fewer params, less capacity
            alpha:        scaling factor (ΔW is scaled by alpha/rank)
        """
        self.r     = rank
        self.alpha = alpha
        self.scale = alpha / rank    # final scaling applied to ΔW

        # A: (r, in_features) — initialized random, captures input directions
        self.A = np.random.randn(rank, in_features).astype(np.float32) / math.sqrt(rank)
        self.B = np.zeros((out_features, rank), dtype=np.float32)  # zero init

        # Gradients
        self.dA = np.zeros_like(self.A)
        self.dB = np.zeros_like(self.B)

        # Cache for backward
        self._x  = None    # input to forward
        self._Ax = None    # A @ x.T intermediate

    def delta_W(self) -> np.ndarray:
        """Compute ΔW = scale * B @ A — the low-rank weight update."""
        return self.scale * (self.B @ self.A)     # (out, in)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the LoRA contribution: x @ ΔW.T = x @ A.T @ B.T * scale
        Args:
            x: (..., in_features)
        Returns:
            (..., out_features)  — add this to the frozen weight's output
        """
        self._x  = x
        Ax       = x @ self.A.T                   # (..., r)
        self._Ax = Ax
        return self.scale * (Ax @ self.B.T)       # (..., out_features)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Accumulate gradients for A and B.
        Returns gradient w.r.t. x (to pass further back).
        """
        x  = self._x
        Ax = self._Ax

        # dB: gradient w.r.t. B
        # dL/dB = scale * dout.T @ Ax  (reshaped to 2D)
        dout_2d = dout.reshape(-1, dout.shape[-1])    # (N, out)
        Ax_2d   = Ax.reshape(-1, self.r)              # (N, r)
        x_2d    = x.reshape(-1, x.shape[-1])          # (N, in)

        self.dB += self.scale * dout_2d.T @ Ax_2d     # (out, r)

        # dA: gradient w.r.t. A
        # dL/dA = scale * B.T @ dout.T @ x
        dAx = (self.scale * dout_2d @ self.B)         # (N, r)
        self.dA += dAx.T @ x_2d                       # (r, in)

        # Gradient w.r.t. input x
        return (dAx @ self.A)                         # (N, in) → original shape

    def zero_grad(self):
        self.dA[:] = 0.0
        self.dB[:] = 0.0

    def trainable_params(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Return (param, grad) pairs for optimizer."""
        return [(self.A, self.dA), (self.B, self.dB)]

    def param_count(self) -> int:
        return self.A.size + self.B.size


# ── LoRA-adapted Linear layer ──────────────────────────────────────────────────

class LoRALinear:
    """
    A frozen Linear layer with an injected LoRA adapter.
    The base weights W are never updated.
    Only A and B are trained.

    forward:  y = x @ W.T  +  lora.forward(x)
    """

    def __init__(self, base_W: np.ndarray, base_b: Optional[np.ndarray] = None,
                 rank: int = 8, alpha: float = 16.0):
        """
        Args:
            base_W: frozen weight matrix (out, in) — copy from pretrained model
            base_b: frozen bias (out,) or None
            rank:   LoRA rank
            alpha:  LoRA scaling factor
        """
        self.W    = base_W.copy().astype(np.float32)  # frozen — never touched
        self.b    = base_b.copy().astype(np.float32) if base_b is not None else None
        self.lora = LoRAParam(base_W.shape[1], base_W.shape[0], rank, alpha)
        self._x   = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        base_out = x @ self.W.T                       # frozen contribution
        if self.b is not None:
            base_out = base_out + self.b
        lora_out = self.lora.forward(x)               # trainable contribution
        return base_out + lora_out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        # Base weights are frozen — no grad update for W, b
        # LoRA backward accumulates dA, dB
        dx_lora = self.lora.backward(dout)
        dx_base = dout @ self.W                       # gradient through frozen W
        return dx_base + dx_lora

    def forward_as_patch(self, base_linear, x: np.ndarray) -> np.ndarray:
        """
        Forward pass designed to replace base_linear.forward.
        Sets base_linear._x_cache so backward still works correctly.
        """
        base_linear._x_cache = x        # keep base Linear's backward working
        self._x = x
        base_out = x @ self.W.T
        if self.b is not None:
            base_out = base_out + self.b
        lora_out = self.lora.forward(x)
        return base_out + lora_out
        """
        Merge LoRA delta into base weights permanently.
        Returns merged W. Used when deploying a fine-tuned adapter.
        """
        return self.W + self.lora.delta_W()

    def trainable_params(self):
        return self.lora.trainable_params()


# ── LoRA Optimizer (AdamW on LoRA params only) ─────────────────────────────────

class LoRAAdamW:
    """AdamW optimizer that only touches LoRA parameters, not the frozen base."""

    def __init__(self, adapters: List[LoRAParam], lr=1e-3,
                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.adapters = adapters
        self.lr       = lr
        self.b1, self.b2 = betas
        self.eps      = eps
        self.wd       = weight_decay
        self.t        = 0
        # Moments for each (A, B) pair across all adapters
        self.m: Dict[int, List[np.ndarray]] = {}
        self.v: Dict[int, List[np.ndarray]] = {}
        for i, a in enumerate(adapters):
            self.m[i] = [np.zeros_like(a.A), np.zeros_like(a.B)]
            self.v[i] = [np.zeros_like(a.A), np.zeros_like(a.B)]

    def step(self):
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        for i, adapter in enumerate(self.adapters):
            for j, (param, grad) in enumerate([(adapter.A, adapter.dA),
                                                (adapter.B, adapter.dB)]):
                self.m[i][j] = self.b1 * self.m[i][j] + (1.0 - self.b1) * grad
                self.v[i][j] = self.b2 * self.v[i][j] + (1.0 - self.b2) * grad**2
                m_hat = self.m[i][j] / bc1
                v_hat = self.v[i][j] / bc2
                param -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.wd * param)

    def zero_grad(self):
        for a in self.adapters:
            a.zero_grad()

    def set_lr(self, lr: float):
        self.lr = lr


# ── LoRA Adapter Collection ────────────────────────────────────────────────────

@dataclass
class LoRAConfig:
    rank:             int   = 8
    alpha:            float = 16.0
    target_layers:    List[str] = field(default_factory=lambda: ["Wq", "Wv"])
    dropout_rate:     float = 0.05
    task_name:        str   = "default"
    base_model_name:  str   = ""
    created_at:       str   = ""
    trained_steps:    int   = 0
    final_loss:       Optional[float] = None


class LoRAAdapter:
    """
    A full set of LoRA adapters for all targeted layers in a TransformerLM.
    Supports save/load, merging, and swapping between task-specific adapters.
    """

    def __init__(self, config: LoRAConfig):
        self.config   = config
        self.layers:  Dict[str, LoRALinear] = {}   # key: "block_i_layername"
        self._injected = False

    def inject(self, model) -> int:
        """
        Inject LoRA adapters into a TransformerLM's attention projections.
        Returns the number of LoRA params added.

        After injection:
        - model.blocks[i].attn.Wq  (and Wv) are wrapped with LoRALinear
        - All other weights remain frozen
        """
        total_lora_params = 0

        for i, block in enumerate(model.blocks):
            for layer_name in self.config.target_layers:
                linear = getattr(block.attn, layer_name, None)
                if linear is None:
                    continue

                key = f"block_{i}_{layer_name}"
                lora_linear = LoRALinear(
                    base_W = linear.W.data,
                    base_b = linear.b.data if linear.b else None,
                    rank   = self.config.rank,
                    alpha  = self.config.alpha,
                )
                self.layers[key] = lora_linear
                total_lora_params += lora_linear.lora.param_count()

        self._injected = True
        return total_lora_params

    def all_adapters(self) -> List[LoRAParam]:
        return [ll.lora for ll in self.layers.values()]

    def forward_with_lora(self, model, ids: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Run forward pass using LoRA-adapted weights.
        Temporarily patches attention layer forward methods.
        """
        if not self._injected:
            return model.forward(ids, training=training)

        # Patch each targeted linear layer
        original_forwards = {}
        for i, block in enumerate(model.blocks):
            for layer_name in self.config.target_layers:
                linear = getattr(block.attn, layer_name, None)
                if linear is None:
                    continue
                key = f"block_{i}_{layer_name}"
                lora_linear = self.layers[key]
                original_forwards[key] = linear.forward
                # Closure to capture correct linear and lora_linear references
                def make_patch(ll, lin):
                    return lambda x: ll.forward_as_patch(lin, x)
                linear.forward = make_patch(lora_linear, linear)

        # Run the model
        output = model.forward(ids, training=training)

        # Restore original forwards
        for i, block in enumerate(model.blocks):
            for layer_name in self.config.target_layers:
                linear = getattr(block.attn, layer_name, None)
                if linear is None:
                    continue
                key = f"block_{i}_{layer_name}"
                if key in original_forwards:
                    linear.forward = original_forwards[key]

        return output

    def save(self, path: str):
        """Save all LoRA adapter weights + config."""
        data = {
            "config":  asdict(self.config),
            "weights": {
                key: {"A": ll.lora.A.tolist(), "B": ll.lora.B.tolist()}
                for key, ll in self.layers.items()
            }
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "LoRAAdapter":
        with open(path) as f:
            data = json.load(f)
        config  = LoRAConfig(**data["config"])
        adapter = cls(config)
        for key, weights in data["weights"].items():
            # Reconstruct LoRALinear with dummy base weights (shape only)
            A = np.array(weights["A"], dtype=np.float32)
            B = np.array(weights["B"], dtype=np.float32)
            dummy_W = np.zeros((B.shape[0], A.shape[1]), dtype=np.float32)
            ll = LoRALinear(dummy_W, rank=config.rank, alpha=config.alpha)
            ll.lora.A = A
            ll.lora.B = B
            adapter.layers[key] = ll
        return adapter

    def param_count(self) -> int:
        return sum(ll.lora.param_count() for ll in self.layers.values())

    def merge_into_model(self, model):
        """
        Permanently merge LoRA weights into model base weights.
        Makes the adapter permanent — faster inference, no longer swappable.
        """
        for i, block in enumerate(model.blocks):
            for layer_name in self.config.target_layers:
                linear = getattr(block.attn, layer_name, None)
                if linear is None:
                    continue
                key = f"block_{i}_{layer_name}"
                if key in self.layers:
                    merged_W = self.layers[key].merge_weights()
                    linear.W.data = merged_W


# ── LoRA Training Loop ─────────────────────────────────────────────────────────

def train_lora(
    model,
    tokenizer,
    examples:   List[Dict[str, str]],
    rank:       int   = 8,
    alpha:      float = 16.0,
    lr:         float = 2e-4,
    max_steps:  int   = 300,
    batch_size: int   = 4,
    seq_len:    int   = 32,
    task_name:  str   = "finetune",
    save_path:  Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fine-tune a TransformerLM with LoRA.
    Only trains the injected A and B matrices. Base weights frozen.

    Returns:
        dict with final_loss, steps, adapter_path, param_counts
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from myai_v2 import cross_entropy_loss, TextDataset, cosine_lr_schedule, softmax

    # Build config and adapter
    config  = LoRAConfig(
        rank=rank, alpha=alpha,
        target_layers=["Wq", "Wv"],
        task_name=task_name,
        base_model_name=getattr(model, 'VERSION', ''),
        created_at=__import__('datetime').datetime.utcnow().isoformat(),
    )
    adapter = LoRAAdapter(config)
    lora_params = adapter.inject(model)

    # Freeze all base params
    # (they still exist but optimizer won't touch them)
    all_adapters = adapter.all_adapters()
    optimizer    = LoRAAdamW(all_adapters, lr=lr)

    # Encode training data
    all_ids = []
    for ex in examples:
        text = ex.get("input", "") + " " + ex.get("output", "")
        all_ids.extend(tokenizer.encode(text, add_special=True))

    if len(all_ids) < seq_len + 1:
        return {"error": "Not enough training data", "min_tokens": seq_len + 1}

    dataset = TextDataset(all_ids, seq_len)

    print(f"\n  LoRA Fine-tuning: task='{task_name}'")
    print(f"  Base params:  {model.count_params():,} (frozen)")
    print(f"  LoRA params:  {lora_params:,}  (rank={rank}, α={alpha})")
    print(f"  Ratio:        {lora_params/model.count_params()*100:.2f}% of base")
    print(f"  Examples:     {len(examples)}  ({len(all_ids)} tokens)")
    print(f"  Steps:        {max_steps}")

    loss_history = []
    t0 = time.time()

    for step in range(1, max_steps + 1):
        lr_sched = cosine_lr_schedule(step, max_steps // 10, max_steps, lr, lr * 0.01)
        optimizer.set_lr(lr_sched)

        x, y     = dataset.get_batch(batch_size)

        # Forward with LoRA adapters active
        logits   = adapter.forward_with_lora(model, x, training=True)

        loss, dlogits = cross_entropy_loss(logits, y)
        loss_history.append(float(loss))

        # Backward through model
        optimizer.zero_grad()
        model.backward(dlogits)

        # Gradient clipping on LoRA params only
        lora_grads = np.concatenate([a.dA.ravel() for a in all_adapters] +
                                     [a.dB.ravel() for a in all_adapters])
        gnorm = np.linalg.norm(lora_grads)
        if gnorm > 1.0:
            scale = 1.0 / gnorm
            for a in all_adapters:
                a.dA *= scale
                a.dB *= scale

        optimizer.step()

        if step % 50 == 0:
            avg  = float(np.mean(loss_history[-50:]))
            ppl  = math.exp(min(avg, 20))
            elapsed = time.time() - t0
            print(f"  step {step:4d}/{max_steps} | loss {avg:.4f} | "
                  f"ppl {ppl:.2f} | lr {lr_sched:.2e} | {elapsed:.1f}s")

    config.trained_steps = max_steps
    config.final_loss    = float(np.mean(loss_history[-20:]))

    # Save adapter
    if save_path is None:
        save_path = str(CKPT_DIR / f"lora_{task_name}_{int(time.time())}.json")
    adapter.save(save_path)

    print(f"\n  LoRA training complete.")
    print(f"  Final loss: {config.final_loss:.4f}")
    print(f"  Adapter saved: {save_path}")
    print(f"  Params trained: {lora_params:,} ({lora_params/model.count_params()*100:.2f}% of model)")

    return {
        "final_loss":    config.final_loss,
        "steps":         max_steps,
        "lora_params":   lora_params,
        "base_params":   model.count_params(),
        "adapter_path":  save_path,
        "loss_history":  loss_history[::10],
    }


# ── Demo / test ────────────────────────────────────────────────────────────────

def demo_lora():
    """Show LoRA working end-to-end on a small model."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from myai_v2 import TransformerLM, Tokenizer, CORPUS, generate

    print("\n╔══════════════════════════════════════════╗")
    print("║      LoRA Fine-Tuning Demo               ║")
    print("╚══════════════════════════════════════════╝\n")

    # Build small model
    tokenizer = Tokenizer()
    tokenizer.build_vocab([CORPUS], max_vocab=2048)
    model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=64, n_heads=2, n_layers=2, d_ff=128
    )

    # Before fine-tuning
    print("  Before LoRA fine-tuning:")
    before = generate(model, tokenizer, "attention", max_new=15, temperature=0.8)
    print(f"  → {before}\n")

    # Fine-tune on domain-specific examples
    examples = [
        {"input": "what is lora",
         "output": "LoRA is low-rank adaptation for fine-tuning large language models efficiently."},
        {"input": "explain lora",
         "output": "LoRA injects trainable rank decomposition matrices into frozen model weights."},
        {"input": "lora parameters",
         "output": "LoRA trains only A and B matrices where delta W equals B times A scaled."},
        {"input": "why use lora",
         "output": "LoRA reduces trainable parameters by 99 percent while preserving model quality."},
    ] * 10

    result = train_lora(
        model, tokenizer, examples,
        rank=4, alpha=8, max_steps=100, task_name="lora_demo"
    )

    print(f"\n  Training stats:")
    print(f"    Base params:  {result['base_params']:,}")
    print(f"    LoRA params:  {result['lora_params']:,}  ({result['lora_params']/result['base_params']*100:.1f}%)")
    print(f"    Final loss:   {result['final_loss']:.4f}")


if __name__ == "__main__":
    demo_lora()
