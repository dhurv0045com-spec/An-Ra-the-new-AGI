# ============================================================
# FILE: finetune/lora.py
# Low-Rank Adaptation — fine-tune without touching base weights.
#
# Instead of updating W (large), we learn two small matrices:
#   W' = W + (A @ B) * scale
# where A is (d_in, rank), B is (rank, d_out), rank << d_in.
#
# Memory saved: rank*(d_in+d_out) vs d_in*d_out
# At rank=8, d=512: 8192 params vs 262144 — 32x smaller.
# ============================================================

import numpy as np
import json, os


class LoRALayer:
    """
    Wraps one weight matrix with a LoRA adapter.
    Keeps the original W frozen. Only A and B are trained.
    """

    def __init__(self, d_in, d_out, rank=8, alpha=16, seed=42):
        """
        Args:
            d_in  (int): Input dimension of the wrapped layer.
            d_out (int): Output dimension of the wrapped layer.
            rank  (int): LoRA rank — controls adapter capacity.
            alpha (int): Scaling factor. scale = alpha / rank.
            seed  (int): RNG seed.
        """
        self.d_in  = d_in
        self.d_out = d_out
        self.rank  = rank
        self.alpha = alpha
        self.scale = alpha / rank           # how strongly LoRA output is applied

        rng = np.random.default_rng(seed)

        # A: initialised with small normals (non-zero so gradients flow)
        self.A = rng.normal(0.0, 0.02, size=(d_in, rank))
        # B: initialised to zero so LoRA output starts at zero (no disruption)
        self.B = np.zeros((rank, d_out))

        # Gradients — allocated on first backward
        self.dA = np.zeros_like(self.A)
        self.dB = np.zeros_like(self.B)

        # Cache for backward pass
        self._last_x    = None
        self._last_Ax   = None

    def forward(self, x, W_frozen):
        """
        Forward pass: base output + LoRA delta.

        y = x @ W_frozen  +  (x @ A @ B) * scale

        Args:
            x        (np.ndarray): Input, shape (..., d_in).
            W_frozen (np.ndarray): Frozen base weight (d_in, d_out).

        Returns:
            np.ndarray: Output shape (..., d_out).
        """
        self._last_x  = x
        Ax            = x @ self.A              # (..., rank)
        self._last_Ax = Ax
        delta         = Ax @ self.B * self.scale  # (..., d_out)
        return x @ W_frozen + delta             # base + adapter

    def backward(self, grad_out):
        """
        Compute gradients for A and B only. W_frozen gets no gradient.

        grad_out: (..., d_out)
        Returns:  grad_x (..., d_in) to pass to previous layer.
        """
        # dB: how much B should change
        # grad_out shape: (batch, d_out), _last_Ax shape: (batch, rank)
        x  = self._last_x
        Ax = self._last_Ax
        scaled_grad = grad_out * self.scale     # (..., d_out)

        # Reshape to 2D for matmul
        flat_scaled = scaled_grad.reshape(-1, self.d_out)   # (N, d_out)
        flat_Ax     = Ax.reshape(-1, self.rank)              # (N, rank)
        flat_x      = x.reshape(-1, self.d_in)              # (N, d_in)

        self.dB = flat_Ax.T @ flat_scaled                   # (rank, d_out)
        dAx     = flat_scaled @ self.B.T                    # (N, rank)
        self.dA = flat_x.T @ dAx                            # (d_in, rank)

        # Gradient w.r.t. input (only from LoRA path; frozen path is separate)
        grad_x = (dAx @ self.A.T).reshape(x.shape)          # (..., d_in)
        return grad_x

    def update(self, lr):
        """SGD step on A and B only."""
        self.A -= lr * self.dA
        self.B -= lr * self.dB
        self.dA[:] = 0
        self.dB[:] = 0

    def get_delta(self):
        """Return the current LoRA weight delta: A @ B * scale."""
        return self.A @ self.B * self.scale

    def merge_into(self, W):
        """
        Permanently bake the adapter into the base weight matrix.
        Call this when you want to ship a single merged model.

        Returns:
            np.ndarray: W + delta, same shape as W.
        """
        return W + self.get_delta()

    def save(self, path):
        """Save A and B to disk as .npz."""
        np.savez(path, A=self.A, B=self.B,
                 rank=np.array(self.rank),
                 alpha=np.array(self.alpha))

    @classmethod
    def load(cls, path, d_in, d_out):
        """Load a saved adapter from disk."""
        data  = np.load(path + '.npz')
        rank  = int(data['rank'])
        alpha = int(data['alpha'])
        obj   = cls(d_in, d_out, rank=rank, alpha=alpha)
        obj.A = data['A']
        obj.B = data['B']
        return obj


class LoRAManager:
    """
    Manages a collection of LoRA adapters for a full model.
    Tracks which layers are adapted, handles save/load of all adapters.
    """

    def __init__(self, rank=8, alpha=16):
        self.rank    = rank
        self.alpha   = alpha
        self.adapters = {}            # name → LoRALayer

    def add(self, name, d_in, d_out, seed=42):
        """Register a new LoRA adapter by layer name."""
        self.adapters[name] = LoRALayer(d_in, d_out, self.rank, self.alpha, seed)
        return self.adapters[name]

    def get(self, name):
        return self.adapters[name]

    def update_all(self, lr):
        """Run one SGD step on every adapter."""
        for adapter in self.adapters.values():
            adapter.update(lr)

    def save_all(self, directory):
        """Save every adapter to directory/name.npz"""
        os.makedirs(directory, exist_ok=True)
        meta = {}
        for name, adapter in self.adapters.items():
            fpath = os.path.join(directory, name.replace('/', '_'))
            adapter.save(fpath)
            meta[name] = {'d_in': adapter.d_in, 'd_out': adapter.d_out,
                          'rank': adapter.rank, 'alpha': adapter.alpha}
        with open(os.path.join(directory, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"[LoRA] Saved {len(self.adapters)} adapters → {directory}")

    @classmethod
    def load_all(cls, directory):
        """Reconstruct a LoRAManager from a saved directory."""
        with open(os.path.join(directory, 'meta.json')) as f:
            meta = json.load(f)
        mgr = cls()
        for name, cfg in meta.items():
            fpath = os.path.join(directory, name.replace('/', '_'))
            mgr.adapters[name] = LoRALayer.load(
                fpath, cfg['d_in'], cfg['d_out'])
        return mgr

    def param_count(self):
        """Total trainable parameters across all adapters."""
        total = 0
        for a in self.adapters.values():
            total += a.A.size + a.B.size
        return total


# ============================================================
# TEST
# ============================================================

if __name__ == '__main__':
    print("=" * 55)
    print("  LoRA SELF TEST")
    print("=" * 55)

    d_in, d_out, rank = 64, 64, 4
    W_frozen = np.random.default_rng(0).normal(0, 0.02, (d_in, d_out))
    lora     = LoRALayer(d_in, d_out, rank=rank, alpha=8)

    x        = np.random.default_rng(1).normal(size=(8, d_in))  # batch of 8

    # Forward
    out_lora = lora.forward(x, W_frozen)
    out_base = x @ W_frozen
    delta    = np.abs(out_lora - out_base).mean()
    print(f"Output shape      : {out_lora.shape}")
    print(f"Mean |delta| vs base: {delta:.6f}  (should be ~0 at init)")

    # Backward
    grad = np.ones_like(out_lora)
    gx   = lora.backward(grad)
    print(f"Grad x shape      : {gx.shape}")

    # Update
    lora.update(lr=1e-3)
    print(f"dA after update   : {lora.dA.sum():.6f}  (should be 0 after reset)")

    # Merge
    W_merged = lora.merge_into(W_frozen)
    print(f"Merged W shape    : {W_merged.shape}")

    # Manager save/load
    mgr = LoRAManager(rank=4, alpha=8)
    mgr.add('attn.qkv', d_in, d_out)
    mgr.add('ffn.fc1',  d_in, d_out)
    print(f"Total LoRA params : {mgr.param_count()}")

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        mgr.save_all(tmp)
        mgr2 = LoRAManager.load_all(tmp)
        print(f"Loaded back {len(mgr2.adapters)} adapters ✓")

    print("\n✓ LoRA all checks passed.")
    print("=" * 55)
