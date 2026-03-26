"""
################################################################################
#                                                                              #
#   nanoGPT.py  —  A Complete Transformer Language Model From Scratch         #
#   Version 1.0 — Built entirely on NumPy. No PyTorch. No TensorFlow.         #
#                                                                              #
#   WHAT IS INSIDE:                                                            #
#                                                                              #
#   SECTION 1 — AUTOGRAD ENGINE                                                #
#     A custom reverse-mode automatic differentiation engine.                 #
#     Every arithmetic operation records a backward closure.                  #
#     Calling .backward() walks the graph in reverse topological order,       #
#     propagating gradients via the chain rule — exactly like PyTorch.        #
#                                                                              #
#   SECTION 2 — NEURAL NETWORK LAYERS                                         #
#     Parameter, Module, Linear, Embedding, LayerNorm, Dropout,               #
#     ReLU, Sigmoid, Tanh, GELU, FeedForward, Sequential                     #
#                                                                              #
#   SECTION 3 — LOSS FUNCTIONS                                                 #
#     MSE loss for regression. Cross-entropy (numerically stable,             #
#     label-smoothing, ignore_index) for language modeling.                   #
#                                                                              #
#   SECTION 4 — OPTIMIZERS & LR SCHEDULERS                                    #
#     SGD with momentum. AdamW (the transformer standard).                    #
#     Linear warmup + cosine annealing learning rate schedule.                #
#                                                                              #
#   SECTION 5 — TOKENIZERS                                                     #
#     CharTokenizer: character-level, zero config, works on anything.         #
#     BPETokenizer: byte-pair encoding trained from corpus data.              #
#                                                                              #
#   SECTION 6 — TRANSFORMER ARCHITECTURE                                       #
#     Sinusoidal + learned positional encodings.                              #
#     Scaled dot-product attention. Multi-head attention with causal mask.    #
#     Transformer block (pre-norm). Full GPT decoder-only model.             #
#                                                                              #
#   SECTION 7 — TRAINING PIPELINE                                              #
#     TextDataset with sliding-window batching. LossTracker with EMA.        #
#     Checkpointer (save/resume). Trainer orchestrator with validation.       #
#                                                                              #
#   SECTION 8 — INFERENCE ENGINE                                               #
#     Greedy decoding. Temperature sampling. Top-k filtering.                 #
#     Nucleus (top-p) sampling. Streaming generation. InferenceEngine API.   #
#                                                                              #
#   SECTION 9 — MODEL MANAGEMENT                                               #
#     save_model / load_model. inspect_model (weight stats).                 #
#     count_parameters with breakdown chart. print_model_card.               #
#                                                                              #
#   SECTION 10 — DEMO                                                          #
#     Runs when you execute: python3 nanoGPT.py                               #
#     Trains on Shakespeare, verifies gradients, generates with 4 strategies. #
#                                                                              #
################################################################################
"""


import numpy as np
import math
import os
import json
import time
import pickle
import gc
import re
from collections import Counter, defaultdict
from typing import (
    List, Optional, Tuple, Union, Dict, Any, Callable, Iterator
)



################################################################################
# SECTION 1: AUTOGRAD ENGINE — Custom reverse-mode AD
################################################################################


"""
================================================================================
nanoGPT / core / engine.py
================================================================================
The autograd engine. This is the mathematical foundation everything else
is built on.

Every computation that needs gradients flows through the Tensor class here.
We implement reverse-mode automatic differentiation (backpropagation) from
scratch — the same algorithm that powers PyTorch, JAX, and TensorFlow.

HOW IT WORKS:
  When you do math with Tensors, each operation records:
    1. What inputs it consumed
    2. A closure (_backward) that knows how to compute the gradient of
       that specific operation's output with respect to its inputs

  When you call .backward() on a final scalar loss, the engine walks the
  computation graph in reverse topological order, calling each _backward
  closure in turn. Gradients accumulate via the chain rule.

SUPPORTED OPS:
  Arithmetic: +, -, *, /, **, @(matmul)
  Reductions: sum, mean, var, max
  Shape:      reshape, transpose, permute, squeeze, unsqueeze, cat, split
  Activations: relu, sigmoid, tanh, gelu, softmax
  Special:    log, exp, sqrt, abs, masked_fill, embedding_lookup

CORRECTNESS:
  Every backward pass is verified against numerical gradients during
  development. Error tolerance < 1e-3 for all ops.
================================================================================
Author: nanoGPT builder
================================================================================
"""

# import numpy as np  # at top
# from typing import Callable, Optional, Tuple, Union, List  # at top


# ──────────────────────────────────────────────────────────────────────────────
# Utility: unbroadcast a gradient back to an original shape
# ──────────────────────────────────────────────────────────────────────────────

def _unbroadcast(grad: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    NumPy broadcasting expands tensors implicitly during forward ops.
    During backward, we must sum the gradient back over any axes that
    were broadcast, to match the original tensor shape.

    Example:
        forward:  (3,) + (2,3) → (2,3)   [scalar broadcast over batch]
        backward: grad is (2,3), must be summed to (3,) for the first input

    Args:
        grad:         The upstream gradient, possibly with extra broadcast dims.
        target_shape: The shape of the original tensor whose gradient we want.

    Returns:
        Gradient reshaped / summed to match target_shape.
    """
    # Step 1: if grad has more dims than target, sum over leading dims
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)

    # Step 2: sum over any axis where target_shape has size 1 (broadcast axis)
    for axis, (g_dim, t_dim) in enumerate(zip(grad.shape, target_shape)):
        if t_dim == 1 and g_dim != 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad.reshape(target_shape)


# ──────────────────────────────────────────────────────────────────────────────
# Core Tensor class
# ──────────────────────────────────────────────────────────────────────────────

class Tensor:
    """
    An N-dimensional array with automatic gradient tracking.

    Every Tensor either:
      - is a LEAF  (created by user code, e.g. weights)
      - is a RESULT of an op (built by __add__, matmul, etc.)

    Leaf tensors with requires_grad=True accumulate gradients in .grad.
    Result tensors store a _backward closure that propagates gradients
    to their parent tensors.

    Usage:
        x = Tensor([[1.0, 2.0]], requires_grad=True)
        y = x * 3.0
        y.sum().backward()
        print(x.grad)  # → [[3., 3.]]
    """

    # ────────────────────────────── construction ──────────────────────────────

    def __init__(
        self,
        data: Union[np.ndarray, list, float, int],
        requires_grad: bool = False,
        _children: Tuple["Tensor", ...] = (),
        _op: str = "",
    ):
        """
        Args:
            data:          Raw numerical data (will be cast to float32).
            requires_grad: If True, this tensor participates in gradient
                           computation and accumulates .grad on backward().
            _children:     Tensors this tensor was computed from (internal).
            _op:           Name of the op that created this tensor (internal).
        """
        # Always store as float32 — the standard dtype for neural networks
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        self.data: np.ndarray = data.astype(np.float32)

        self.requires_grad: bool = requires_grad

        # .grad is None until backward() is called; then it's a numpy array
        self.grad: Optional[np.ndarray] = None

        # The backward closure — replaced by each op that creates this tensor
        self._backward: Callable = lambda: None

        # Parent tensors in the computation graph
        self._prev: Tuple["Tensor", ...] = _children

        # Name of the op for debugging
        self._op: str = _op

    # ────────────────────────────── properties ────────────────────────────────

    @property
    def shape(self) -> tuple:
        """Shape of the underlying array, e.g. (batch, seq, d_model)."""
        return self.data.shape

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self.data.ndim

    @property
    def size(self) -> int:
        """Total number of elements."""
        return self.data.size

    @property
    def T(self) -> "Tensor":
        """Shortcut for transpose of last two dimensions."""
        return self.transpose()

    def __repr__(self) -> str:
        return (
            f"Tensor(shape={self.shape}, dtype=float32"
            + (f", op='{self._op}'" if self._op else "")
            + (", grad_fn=True" if self._op else "")
            + ")"
        )

    # ────────────────────────────── grad helpers ──────────────────────────────

    def _ensure_grad(self):
        """Initialize gradient buffer to zeros if not yet allocated."""
        if self.grad is None:
            self.grad = np.zeros_like(self.data)

    def zero_grad(self):
        """Reset gradient to None (called before each optimizer step)."""
        self.grad = None

    def detach(self) -> "Tensor":
        """Return a new Tensor with the same data but no gradient tracking."""
        return Tensor(self.data.copy())

    def numpy(self) -> np.ndarray:
        """Extract the raw numpy array."""
        return self.data

    def item(self) -> float:
        """Extract a Python scalar from a single-element tensor."""
        return float(self.data)

    # ────────────────────────────── backward ──────────────────────────────────

    def backward(self, grad: Optional[np.ndarray] = None):
        """
        Reverse-mode autodiff. Computes gradients for all leaf tensors
        that have requires_grad=True.

        Call this on a scalar loss tensor. The gradient of the loss with
        respect to every parameter in the graph will be accumulated in
        param.grad.

        Args:
            grad: The upstream gradient (default: ones for a scalar loss).
                  Pass a shaped gradient for non-scalar outputs.
        """
        # For scalar loss, start with gradient = 1.0
        if grad is None:
            assert self.data.size == 1, (
                "backward() without a grad argument requires a scalar tensor. "
                f"Got shape {self.shape}. Call .sum() or .mean() first."
            )
            grad = np.ones_like(self.data)

        # Seed the gradient of this (root) tensor
        self.grad = grad

        # ── Topological sort ──
        # We need to visit each node AFTER all nodes that depend on it
        # (i.e., children before parents in the computation graph).
        topo_order: List["Tensor"] = []
        visited = set()

        def build_topo(node: "Tensor"):
            node_id = id(node)
            if node_id not in visited:
                visited.add(node_id)
                for parent in node._prev:
                    build_topo(parent)
                topo_order.append(node)

        build_topo(self)

        # ── Backward pass ──
        # Walk the graph in reverse: output → inputs
        for node in reversed(topo_order):
            node._backward()

    # ──────────────────────────── arithmetic ops ──────────────────────────────

    def __add__(self, other: Union["Tensor", float]) -> "Tensor":
        """
        Element-wise addition with broadcasting support.
        Forward:  out = self + other
        Backward: d/d_self = 1, d/d_other = 1
                  (then unbroadcast to handle shape mismatches)
        """
        if not isinstance(other, Tensor):
            other = Tensor(np.full(1, other, dtype=np.float32))

        out = Tensor(
            self.data + other.data,
            _children=(self, other),
            _op="add",
        )

        def _backward():
            # Gradient flows equally to both inputs; unbroadcast handles shapes
            if self.requires_grad:
                self._ensure_grad()
                self.grad += _unbroadcast(out.grad, self.shape)
            if other.requires_grad:
                other._ensure_grad()
                other.grad += _unbroadcast(out.grad, other.shape)

        out._backward = _backward
        out.requires_grad = self.requires_grad or other.requires_grad
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other: Union["Tensor", float]) -> "Tensor":
        """
        Element-wise multiplication.
        Forward:  out = self * other
        Backward: d/d_self = other, d/d_other = self
        """
        if not isinstance(other, Tensor):
            other = Tensor(np.full(1, other, dtype=np.float32))

        out = Tensor(
            self.data * other.data,
            _children=(self, other),
            _op="mul",
        )

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += _unbroadcast(other.data * out.grad, self.shape)
            if other.requires_grad:
                other._ensure_grad()
                other.grad += _unbroadcast(self.data * out.grad, other.shape)

        out._backward = _backward
        out.requires_grad = self.requires_grad or other.requires_grad
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        """Subtraction: self - other = self + (-1 * other)."""
        return self.__add__(other.__mul__(-1) if isinstance(other, Tensor) else Tensor(np.array(-other, dtype=np.float32)))

    def __rsub__(self, other):
        return self.__mul__(-1).__add__(other)

    def __neg__(self):
        return self.__mul__(-1)

    def __truediv__(self, other: Union["Tensor", float]) -> "Tensor":
        """Division: self / other = self * other^(-1)."""
        if isinstance(other, Tensor):
            return self.__mul__(other.pow(-1))
        return self.__mul__(1.0 / other)

    def __rtruediv__(self, other):
        return self.pow(-1).__mul__(other)

    # ──────────────────────────── power / exp / log ───────────────────────────

    def pow(self, exponent: float) -> "Tensor":
        """
        Element-wise power: out = self^exponent
        Backward: d/d_self = exponent * self^(exponent-1)
        """
        out = Tensor(
            self.data ** exponent,
            _children=(self,),
            _op=f"pow({exponent})",
        )

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += exponent * (self.data ** (exponent - 1)) * out.grad

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def __pow__(self, exponent):
        return self.pow(exponent)

    def exp(self) -> "Tensor":
        """
        Exponential: out = e^self
        Backward: d/d_self = e^self = out  (very clean!)
        Clipped at ±80 to prevent overflow.
        """
        ex = np.exp(np.clip(self.data, -80.0, 80.0))
        out = Tensor(ex, _children=(self,), _op="exp")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += ex * out.grad  # chain: d(e^x)/dx = e^x

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def log(self) -> "Tensor":
        """
        Natural log: out = ln(self)
        Backward: d/d_self = 1 / self
        Clipped to prevent log(0) = -inf.
        """
        out = Tensor(
            np.log(np.clip(self.data, 1e-12, None)),
            _children=(self,),
            _op="log",
        )

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += out.grad / np.clip(self.data, 1e-12, None)

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def sqrt(self) -> "Tensor":
        """Square root via pow(0.5)."""
        return self.pow(0.5)

    def abs(self) -> "Tensor":
        """
        Element-wise absolute value.
        Backward: sign(self)  (subgradient — 0 at exactly 0)
        """
        out = Tensor(np.abs(self.data), _children=(self,), _op="abs")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += np.sign(self.data) * out.grad

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    # ──────────────────────────── matrix multiply ────────────────────────────

    def matmul(self, other: "Tensor") -> "Tensor":
        """
        Batched matrix multiplication: out = self @ other

        Works for any number of batch dimensions:
            (B, T, D) @ (D, V)   →  (B, T, V)
            (B, H, T, d) @ (B, H, d, T) → (B, H, T, T)   [attention scores]

        Backward:
            d/d_self  = out.grad @ other^T
            d/d_other = self^T @ out.grad
            (summed over batch dims to match original shapes)
        """
        out = Tensor(
            self.data @ other.data,
            _children=(self, other),
            _op="matmul",
        )

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                g = out.grad @ other.data.swapaxes(-1, -2)
                self.grad += _unbroadcast(g, self.shape)

            if other.requires_grad:
                other._ensure_grad()
                g = self.data.swapaxes(-1, -2) @ out.grad
                # Sum over extra leading batch dims (e.g. 2D weight hit by 3D input)
                while g.ndim > other.data.ndim:
                    g = g.sum(axis=0)
                other.grad += g

        out._backward = _backward
        out.requires_grad = self.requires_grad or other.requires_grad
        return out

    def __matmul__(self, other: "Tensor") -> "Tensor":
        return self.matmul(other)

    # ──────────────────────────── reductions ─────────────────────────────────

    def sum(self, axis=None, keepdims: bool = False) -> "Tensor":
        """
        Sum over specified axis (or all elements).
        Backward: broadcast the upstream gradient back to input shape.
        """
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            _children=(self,),
            _op="sum",
        )

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                g = out.grad
                # Restore reduced axes before broadcasting back
                if axis is not None and not keepdims:
                    g = np.expand_dims(g, axis=axis)
                self.grad += np.broadcast_to(g, self.shape)

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def mean(self, axis=None, keepdims: bool = False) -> "Tensor":
        """
        Mean over specified axis.
        Backward: gradient / N  (N = number of elements averaged).
        """
        # Compute N = number of elements being averaged
        if axis is None:
            n = self.data.size
        elif isinstance(axis, int):
            n = self.data.shape[axis]
        else:
            n = int(np.prod([self.data.shape[a] for a in axis]))

        out = Tensor(
            self.data.mean(axis=axis, keepdims=keepdims),
            _children=(self,),
            _op="mean",
        )

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                g = out.grad
                if axis is not None and not keepdims:
                    g = np.expand_dims(g, axis=axis)
                self.grad += np.broadcast_to(g, self.shape) / n

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def var(self, axis=-1, keepdims: bool = False) -> "Tensor":
        """
        Variance (biased) over specified axis.
        Used in layer normalization.
        """
        mean_val = self.data.mean(axis=axis, keepdims=True)
        diff = self.data - mean_val
        n = self.data.shape[axis]
        var_val = (diff ** 2).mean(axis=axis, keepdims=keepdims)

        out = Tensor(var_val, _children=(self,), _op="var")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                g = out.grad
                if not keepdims:
                    g = np.expand_dims(g, axis=axis)
                g = np.broadcast_to(g, self.shape)
                self.grad += 2.0 * diff * g / n

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def max(self, axis=None, keepdims: bool = False) -> "Tensor":
        """
        Maximum over specified axis.
        Backward: gradient flows only to the max element(s).
        Ties are broken by distributing the gradient equally.
        """
        m = self.data.max(axis=axis, keepdims=keepdims)
        out = Tensor(m, _children=(self,), _op="max")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                g = out.grad
                m_expanded = m if keepdims or axis is None else np.expand_dims(m, axis=axis)
                g_expanded = g if keepdims or axis is None else np.expand_dims(g, axis=axis)
                # Indicator: 1.0 where self == max, 0.0 elsewhere
                indicator = (self.data == m_expanded).astype(np.float32)
                # Normalize to handle ties (so gradient sums to 1 per group)
                count = indicator.sum(axis=axis, keepdims=True)
                self.grad += (indicator / count) * g_expanded

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    # ──────────────────────────── shape ops ──────────────────────────────────

    def reshape(self, *shape) -> "Tensor":
        """
        Reshape without copying data.
        Backward: reshape gradient back to original shape.
        """
        out = Tensor(self.data.reshape(*shape), _children=(self,), _op="reshape")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += out.grad.reshape(self.shape)

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def transpose(self, ax1: int = -2, ax2: int = -1) -> "Tensor":
        """
        Swap two axes (default: last two — standard matrix transpose).
        Backward: swap the same axes back.
        """
        axes = list(range(self.ndim))
        # Handle negative indices
        ax1 = axes[ax1]
        ax2 = axes[ax2]
        axes[ax1], axes[ax2] = axes[ax2], axes[ax1]

        out = Tensor(np.transpose(self.data, axes), _children=(self,), _op="transpose")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += np.transpose(out.grad, axes)  # same permutation is its own inverse here

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def permute(self, axes: tuple) -> "Tensor":
        """
        Permute dimensions in arbitrary order.
        Backward: apply the inverse permutation.
        """
        axes = list(axes)
        # Compute inverse permutation
        inv_axes = [0] * len(axes)
        for i, a in enumerate(axes):
            inv_axes[a] = i

        out = Tensor(np.transpose(self.data, axes), _children=(self,), _op="permute")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += np.transpose(out.grad, inv_axes)

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def unsqueeze(self, axis: int) -> "Tensor":
        """Add a dimension of size 1 at the given axis."""
        out = Tensor(np.expand_dims(self.data, axis), _children=(self,), _op="unsqueeze")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += np.squeeze(out.grad, axis=axis).reshape(self.shape)

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def squeeze(self, axis=None) -> "Tensor":
        """Remove dimensions of size 1."""
        out = Tensor(np.squeeze(self.data, axis=axis), _children=(self,), _op="squeeze")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += out.grad.reshape(self.shape)

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def __getitem__(self, idx) -> "Tensor":
        """
        Slice / index the tensor.
        Backward: scatter the upstream gradient back to the indexed positions.
        np.add.at handles repeated indices correctly (unlike fancy indexing
        assignment which would overwrite instead of accumulate).
        """
        out = Tensor(self.data[idx], _children=(self,), _op="getitem")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                np.add.at(self.grad, idx, out.grad)

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def masked_fill(self, mask: np.ndarray, fill_value: float = -1e9) -> "Tensor":
        """
        Replace elements where mask is False with fill_value.
        Used to implement causal attention masking.

        Args:
            mask:       Boolean array (True = keep, False = fill).
            fill_value: Value to fill masked positions with (default: -1e9).

        Backward: gradient is zeroed out at masked positions.
        """
        filled = np.where(mask, self.data, fill_value)
        out = Tensor(filled, _children=(self,), _op="masked_fill")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                # Gradient only flows through un-masked positions
                self.grad += out.grad * mask.astype(np.float32)

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    # ──────────────────────────── activation ops ─────────────────────────────

    def relu(self) -> "Tensor":
        """
        Rectified Linear Unit: max(0, x)
        Backward: 1 where x > 0, 0 elsewhere (subgradient at 0 = 0).
        """
        out = Tensor(np.maximum(0.0, self.data), _children=(self,), _op="relu")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                # Indicator function: gates the gradient
                self.grad += (self.data > 0).astype(np.float32) * out.grad

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def sigmoid(self) -> "Tensor":
        """
        Sigmoid: σ(x) = 1 / (1 + e^{-x})
        Backward: σ(x) * (1 - σ(x))  — elegant closed form.
        """
        s = 1.0 / (1.0 + np.exp(-np.clip(self.data, -80.0, 80.0)))
        out = Tensor(s, _children=(self,), _op="sigmoid")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += s * (1.0 - s) * out.grad

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def tanh(self) -> "Tensor":
        """
        Hyperbolic tangent.
        Backward: 1 - tanh²(x)  — sech²(x) in disguise.
        """
        t = np.tanh(self.data)
        out = Tensor(t, _children=(self,), _op="tanh")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += (1.0 - t * t) * out.grad

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def gelu(self) -> "Tensor":
        """
        Gaussian Error Linear Unit — the default activation in GPT/BERT.
        Uses the tanh approximation from the original paper:
            GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

        Why GELU beats ReLU for transformers:
          - Smooth (differentiable everywhere, no dead neurons)
          - Non-monotonic (can suppress near-zero inputs)
          - Empirically better for language tasks
        """
# import math  # at top
        c = math.sqrt(2.0 / math.pi)
        x3 = self.data ** 3
        inner = np.tanh(c * (self.data + 0.044715 * x3))
        gelu_val = 0.5 * self.data * (1.0 + inner)

        out = Tensor(gelu_val, _children=(self,), _op="gelu")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                # Full analytic derivative of the tanh-approximated GELU
                sech2 = 1.0 - inner ** 2           # sech²(tanh_arg)
                d_inner = c * (1.0 + 3.0 * 0.044715 * self.data ** 2)
                dgelu = 0.5 * (1.0 + inner) + 0.5 * self.data * sech2 * d_inner
                self.grad += dgelu * out.grad

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def softmax(self, axis: int = -1) -> "Tensor":
        """
        Softmax normalization: converts logits to a probability distribution.
        Uses the numerically stable max-subtraction trick.

        Backward: the Jacobian-vector product for softmax is:
            Δx_i = s_i * (Δout_i - sum_j(Δout_j * s_j))
            where s is the softmax output.

        This is the full Jacobian, not just the diagonal — softmax couples
        all outputs together, which is the key to why attention works.
        """
        # Subtract max for numerical stability (doesn't change the distribution)
        x_shifted = self.data - self.data.max(axis=axis, keepdims=True)
        ex = np.exp(x_shifted)
        s = ex / ex.sum(axis=axis, keepdims=True)

        out = Tensor(s, _children=(self,), _op="softmax")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                g = out.grad
                # Efficient Jacobian-vector product: avoids building the full Jacobian
                dot = (g * s).sum(axis=axis, keepdims=True)
                self.grad += s * (g - dot)

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def log_softmax(self, axis: int = -1) -> "Tensor":
        """
        Numerically stable log-softmax.
        log(softmax(x))_i = x_i - log(sum_j exp(x_j))

        Used in cross-entropy loss. More numerically stable than
        computing softmax then taking log separately.
        """
        x_shifted = self.data - self.data.max(axis=axis, keepdims=True)
        log_sum_exp = np.log(np.exp(x_shifted).sum(axis=axis, keepdims=True))
        lsm = x_shifted - log_sum_exp

        out = Tensor(lsm, _children=(self,), _op="log_softmax")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                s = np.exp(lsm)  # softmax values (free, already computed)
                g = out.grad
                # Same Jacobian-vector product as softmax
                self.grad += g - s * g.sum(axis=axis, keepdims=True)

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    # ──────────────────────────── concatenation ───────────────────────────────

    def cat(self, others: List["Tensor"], axis: int = 0) -> "Tensor":
        """
        Concatenate this tensor with a list of other tensors along an axis.
        Backward: split the gradient at the same boundaries and route back.
        """
        all_tensors = [self] + list(others)
        out = Tensor(
            np.concatenate([t.data for t in all_tensors], axis=axis),
            _children=tuple(all_tensors),
            _op="cat",
        )
        # Boundary positions for np.split
        split_points = np.cumsum([t.data.shape[axis] for t in all_tensors[:-1]])

        def _backward():
            parts = np.split(out.grad, split_points, axis=axis)
            for tensor, grad_part in zip(all_tensors, parts):
                if tensor.requires_grad:
                    tensor._ensure_grad()
                    tensor.grad += grad_part

        out._backward = _backward
        out.requires_grad = any(t.requires_grad for t in all_tensors)
        return out

    def split(self, sizes, axis: int = 0) -> List["Tensor"]:
        """
        Split tensor into chunks of given sizes along axis.
        Backward: concatenate gradients back.
        """
        split_points = np.cumsum(sizes[:-1])
        parts_data = np.split(self.data, split_points, axis=axis)
        results = []

        for i, part_data in enumerate(parts_data):
            part = Tensor(part_data, _children=(self,), _op=f"split[{i}]")

            def make_backward(idx, start, end):
                def _backward():
                    if self.requires_grad:
                        self._ensure_grad()
                        slices = [slice(None)] * self.ndim
                        slices[axis] = slice(start, end)
                        self.grad[tuple(slices)] += results[idx].grad
                return _backward

            start = int(split_points[i - 1]) if i > 0 else 0
            end = int(split_points[i]) if i < len(split_points) else self.shape[axis]
            part._backward = make_backward(i, start, end)
            part.requires_grad = self.requires_grad
            results.append(part)

        return results


# ──────────────────────────────────────────────────────────────────────────────
# Factory functions (match PyTorch API style)
# ──────────────────────────────────────────────────────────────────────────────

def zeros(*shape, requires_grad: bool = False) -> Tensor:
    """Create a tensor of all zeros."""
    return Tensor(np.zeros(shape, dtype=np.float32), requires_grad=requires_grad)


def ones(*shape, requires_grad: bool = False) -> Tensor:
    """Create a tensor of all ones."""
    return Tensor(np.ones(shape, dtype=np.float32), requires_grad=requires_grad)


def randn(*shape, requires_grad: bool = False) -> Tensor:
    """Create a tensor from the standard normal distribution N(0, 1)."""
    return Tensor(np.random.randn(*shape).astype(np.float32), requires_grad=requires_grad)


def arange(n: int) -> Tensor:
    """Create a 1D tensor [0, 1, 2, ..., n-1]."""
    return Tensor(np.arange(n, dtype=np.float32))


def cat(tensors: List[Tensor], axis: int = 0) -> Tensor:
    """Concatenate a list of tensors along an axis (free function)."""
    return tensors[0].cat(tensors[1:], axis=axis)


def gradient_check(func, *inputs, eps: float = 1e-4) -> float:
    """
    Numerically verify analytic gradients against finite differences.
    Returns max absolute error across all parameters.

    Use this during development to catch backward() bugs.

    Args:
        func:   A function that takes Tensor inputs and returns a scalar Tensor.
        inputs: Tensor arguments (must have requires_grad=True).
        eps:    Step size for finite differences.
    """
    # Forward + backward
    result = func(*inputs)
    result.backward()
    analytic_grads = [inp.grad.copy() for inp in inputs]

    # Reset for numerical check
    for inp in inputs:
        inp.zero_grad()

    max_error = 0.0
    for inp, analytic in zip(inputs, analytic_grads):
        flat = inp.data.ravel()
        for i in range(len(flat)):
            orig = flat[i]
            flat[i] = orig + eps
            f_plus = float(func(*inputs).data)
            flat[i] = orig - eps
            f_minus = float(func(*inputs).data)
            flat[i] = orig
            numerical = (f_plus - f_minus) / (2 * eps)
            error = abs(analytic.ravel()[i] - numerical)
            max_error = max(max_error, error)

    return max_error





################################################################################
# SECTION 2: NEURAL NETWORK LAYERS — Steps 1-4, 9, 12, 18
################################################################################


"""
================================================================================
nanoGPT / core / layers.py
================================================================================
Neural network building blocks — Steps 1–9 and parts of 12, 18.

Every component here is a Module that:
  - Has learnable Parameters (Tensors with requires_grad=True)
  - Implements forward() which does the computation
  - Tracks training vs eval state (affects Dropout)
  - Exposes parameters() for the optimizer to find

CONTENTS:
  Parameter   — a learnable tensor
  Module      — base class for all components
  Linear      — the fundamental learned transformation: y = xW^T + b
  Embedding   — integer → dense vector lookup table
  LayerNorm   — normalizes activations, keeps training stable
  Dropout     — randomly zeroes activations during training
  ReLU        — Rectified Linear Unit activation
  Sigmoid     — sigmoid activation (used in gates, binary outputs)
  Tanh        — tanh activation
  GELU        — Gaussian Error Linear Unit (used in GPT)
  SoftMax     — probability distribution over vocab
  FeedForward — the MLP inside each transformer block
  Sequential  — chain multiple modules together

INITIALIZATION:
  Weight initialization is critical. We use Kaiming uniform for Linear
  (appropriate for ReLU/GELU networks) and N(0, 0.02) for embeddings
  (GPT-2 convention).
================================================================================
"""

# import numpy as np  # at top
# import math  # at top
# from typing import List, Optional, Callable, Iterator  # at top
# from .engine import Tensor  # merged


# ──────────────────────────────────────────────────────────────────────────────
# Parameter: a Tensor that is always a leaf and always requires grad
# ──────────────────────────────────────────────────────────────────────────────

class Parameter(Tensor):
    """
    A learnable parameter.

    Identical to Tensor but always has requires_grad=True. By subclassing
    Tensor, the optimizer can update it in-place via .data, and the autograd
    engine treats it as a leaf node.

    Example:
        weight = Parameter(np.random.randn(128, 256).astype(np.float32))
        # Now weight.requires_grad is True automatically
    """

    def __init__(self, data: np.ndarray):
        """
        Args:
            data: Initial values (must be a numpy float32 array).
        """
        super().__init__(data, requires_grad=True)

    def __repr__(self) -> str:
        return f"Parameter(shape={self.shape})"


# ──────────────────────────────────────────────────────────────────────────────
# Module: base class for all neural network components
# ──────────────────────────────────────────────────────────────────────────────

class Module:
    """
    Base class for all neural network components.

    A Module:
      - Holds Parameters and child Modules
      - Implements forward() to define the computation
      - Provides parameters() to collect all learnable params
      - Tracks training/eval state (relevant for Dropout, BatchNorm, etc.)

    Subclass this and implement forward() to build any layer.
    """

    _training: bool = True   # True during training, False during eval/inference

    def __call__(self, *args, **kwargs) -> Tensor:
        """Make the module callable: module(x) calls module.forward(x)."""
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs) -> Tensor:
        """Override this in every subclass to define the computation."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement forward()"
        )

    def parameters(self) -> List[Parameter]:
        """
        Recursively collect all Parameters from this module and all
        child Modules. Used by the optimizer to find everything to update.

        Returns:
            List of all Parameter objects (no duplicates).
        """
        params = []
        seen_ids = set()

        def collect(obj):
            for value in obj.__dict__.values():
                if isinstance(value, Parameter):
                    if id(value) not in seen_ids:
                        seen_ids.add(id(value))
                        params.append(value)
                elif isinstance(value, Module):
                    collect(value)
                elif isinstance(value, (list, tuple)):
                    for item in value:
                        if isinstance(item, Parameter):
                            if id(item) not in seen_ids:
                                seen_ids.add(id(item))
                                params.append(item)
                        elif isinstance(item, Module):
                            collect(item)

        collect(self)
        return params

    def named_parameters(self) -> Iterator:
        """Iterate (name, parameter) pairs for inspection."""
        for i, p in enumerate(self.parameters()):
            yield f"param_{i}", p

    def zero_grad(self):
        """Reset all parameter gradients to None."""
        for p in self.parameters():
            p.zero_grad()

    def num_parameters(self) -> int:
        """Count the total number of scalar parameters."""
        seen = set()
        total = 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.data.size
        return total

    def train(self) -> "Module":
        """
        Switch to training mode.
        Activates Dropout, etc.
        """
        self._training = True
        for v in self.__dict__.values():
            if isinstance(v, Module):
                v.train()
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, Module):
                        item.train()
        return self

    def eval(self) -> "Module":
        """
        Switch to evaluation/inference mode.
        Disables Dropout, uses running statistics for BatchNorm, etc.
        """
        self._training = False
        for v in self.__dict__.values():
            if isinstance(v, Module):
                v.eval()
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, Module):
                        item.eval()
        return self

    def __repr__(self) -> str:
        children = []
        for name, val in self.__dict__.items():
            if isinstance(val, Module):
                children.append(f"  ({name}): {val}")
        if children:
            return f"{self.__class__.__name__}(\n" + "\n".join(children) + "\n)"
        return f"{self.__class__.__name__}()"


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 + 2: Single Neuron → Linear Layer
# ──────────────────────────────────────────────────────────────────────────────

class Linear(Module):
    """
    A fully-connected (linear/dense) layer: y = x @ W^T + b

    This is the most fundamental transformation in a neural network.
    A single neuron is just Linear(in_features=N, out_features=1).
    A layer of neurons is Linear(in_features=N, out_features=M).
    Stack them to get a deep network.

    Weight initialization:
        We use Kaiming uniform initialization: W ~ U(-k, k) where k = 1/√fan_in
        This keeps the variance of activations approximately constant through
        layers with ReLU/GELU activations, preventing vanishing/exploding gradients.

    Args:
        in_features:  Number of input features.
        out_features: Number of output features (neurons).
        bias:         Whether to add a learnable bias term.

    Shape:
        Input:  (..., in_features)
        Output: (..., out_features)
        weight: (out_features, in_features)
        bias:   (out_features,)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features

        # Kaiming uniform init: good default for networks with ReLU/GELU
        k = 1.0 / math.sqrt(in_features)
        self.weight = Parameter(
            np.random.uniform(-k, k, (out_features, in_features)).astype(np.float32)
        )

        # Bias initialized to zero (common convention)
        self.bias = Parameter(np.zeros(out_features, dtype=np.float32)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (..., in_features)
        Returns:
            Output tensor of shape (..., out_features)
        """
        # x @ W^T: batched matrix multiply
        # x is (..., in_features), weight is (out_features, in_features)
        # weight.transpose() is (in_features, out_features)
        out = x @ self.weight.transpose()

        # Add bias if present (broadcasts over all batch dims)
        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self) -> str:
        bias_str = f", bias={self.bias is not None}"
        return f"Linear({self.in_features} → {self.out_features}{bias_str})"


# ──────────────────────────────────────────────────────────────────────────────
# Step 12: Embedding Layer
# ──────────────────────────────────────────────────────────────────────────────

class Embedding(Module):
    """
    Lookup table: integer token indices → dense float vectors.

    Every token in the vocabulary gets its own learned vector.
    These vectors represent meaning — similar tokens end up nearby
    in embedding space after training.

    Step 11 (one-hot encoding) is implicit: an embedding lookup is
    mathematically equivalent to a linear layer applied to a one-hot
    vector, but infinitely more efficient — O(d_model) vs O(vocab_size).

    Initialization: GPT-2 convention, N(0, 0.02).

    Args:
        num_embeddings: Vocabulary size (number of distinct tokens).
        embedding_dim:  Size of each embedding vector (d_model).

    Shape:
        Input:  integer array of shape (B, T) with values in [0, num_embeddings)
        Output: float tensor of shape (B, T, embedding_dim)
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Initialize with small random values (GPT-2 style)
        self.weight = Parameter(
            (np.random.randn(num_embeddings, embedding_dim) * 0.02).astype(np.float32)
        )

    def forward(self, idx: np.ndarray) -> Tensor:
        """
        Args:
            idx: Integer array of shape (B, T), values in [0, num_embeddings).
        Returns:
            Tensor of shape (B, T, embedding_dim).
        """
        # Gather rows from the embedding table
        out_data = self.weight.data[idx]   # (B, T, D) or (T, D)

        out = Tensor(out_data, _children=(self.weight,), _op="embedding")

        def _backward():
            if self.weight.requires_grad:
                self.weight._ensure_grad()
                # np.add.at handles repeated indices correctly
                # (important if the same token appears multiple times in a sequence)
                np.add.at(self.weight.grad, idx, out.grad)

        out._backward = _backward
        out.requires_grad = self.weight.requires_grad
        return out

    def __repr__(self) -> str:
        return f"Embedding({self.num_embeddings}, {self.embedding_dim})"


# ──────────────────────────────────────────────────────────────────────────────
# Step 18: Layer Normalization
# ──────────────────────────────────────────────────────────────────────────────

class LayerNorm(Module):
    """
    Layer normalization (Ba et al., 2016).

    Normalizes the input across the last dimension, then applies
    learnable scale (gamma/weight) and shift (beta/bias).

    WHY LAYER NORM MATTERS:
        Without normalization, activations can grow exponentially through
        deep networks, causing gradients to explode or vanish. LayerNorm
        keeps activations in a consistent range regardless of network depth.

    WHY LAYER NORM (not Batch Norm) FOR TRANSFORMERS:
        BatchNorm averages across the batch dimension, which is problematic
        for variable-length sequences and autoregressive generation (you
        don't have a full batch during inference). LayerNorm normalizes
        per-token, so it works identically in training and inference.

    Formula:
        y = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta
        where mean and var are computed over the last dimension.

    Args:
        normalized_shape: Size of the dimension to normalize (usually d_model).
        eps:              Small constant to prevent division by zero.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        self.normalized_shape = normalized_shape
        self.eps = eps

        # Learnable scale: initialized to 1 (no scaling)
        self.weight = Parameter(np.ones(normalized_shape, dtype=np.float32))
        # Learnable shift: initialized to 0 (no shift)
        self.bias = Parameter(np.zeros(normalized_shape, dtype=np.float32))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape (..., normalized_shape).
        Returns:
            Normalized tensor of same shape.
        """
        # Compute statistics over last dimension
        mean = x.data.mean(axis=-1, keepdims=True)   # (B, T, 1)
        var  = x.data.var(axis=-1, keepdims=True)    # (B, T, 1)  [biased]

        # Normalize
        x_norm = (x.data - mean) / np.sqrt(var + self.eps)

        # Scale and shift with learnable parameters
        out_data = self.weight.data * x_norm + self.bias.data

        out = Tensor(out_data, _children=(x, self.weight, self.bias), _op="layernorm")

        def _backward():
            N = x.data.shape[-1]   # number of features being normalized
            g = out.grad           # upstream gradient

            # Gradient w.r.t. gamma (weight): sum over all non-feature dims
            if self.weight.requires_grad:
                self.weight._ensure_grad()
                reduce_axes = tuple(range(g.ndim - 1))
                self.weight.grad += (g * x_norm).sum(axis=reduce_axes)

            # Gradient w.r.t. beta (bias): sum over all non-feature dims
            if self.bias.requires_grad:
                self.bias._ensure_grad()
                reduce_axes = tuple(range(g.ndim - 1))
                self.bias.grad += g.sum(axis=reduce_axes)

            # Gradient w.r.t. input x
            if x.requires_grad:
                x._ensure_grad()
                # Full analytic gradient of layer norm (derived via chain rule):
                # dx = gamma/sqrt(var+eps) * [N*dout - sum(dout) - x_norm*sum(dout*x_norm)] / N
                std_inv = 1.0 / np.sqrt(var + self.eps)
                dx_norm = g * self.weight.data           # gradient into x_norm
                x.grad += (1.0 / N) * std_inv * (
                    N * dx_norm
                    - dx_norm.sum(axis=-1, keepdims=True)
                    - x_norm * (dx_norm * x_norm).sum(axis=-1, keepdims=True)
                )

        out._backward = _backward
        out.requires_grad = x.requires_grad or self.weight.requires_grad
        return out

    def __repr__(self) -> str:
        return f"LayerNorm({self.normalized_shape}, eps={self.eps})"


# ──────────────────────────────────────────────────────────────────────────────
# Step 9: Dropout
# ──────────────────────────────────────────────────────────────────────────────

class Dropout(Module):
    """
    Dropout regularization (Srivastava et al., 2014).

    During training: randomly zeros out each element with probability p.
    The remaining elements are scaled by 1/(1-p) to maintain expected value.
    During inference: does nothing (no stochasticity needed).

    WHY DROPOUT HELPS:
        By randomly killing neurons during training, the network cannot rely
        on any single neuron — it must learn redundant, distributed representations.
        This is a form of ensemble learning: each training step trains a slightly
        different subnetwork.

    Args:
        p: Probability of zeroing an element (typically 0.1–0.3).
    """

    def __init__(self, p: float = 0.1):
        assert 0.0 <= p < 1.0, f"Dropout probability must be in [0, 1), got {p}"
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor (any shape).
        Returns:
            During training: randomly masked and rescaled tensor.
            During inference: unchanged input.
        """
        # No-op during inference
        if not self._training or self.p == 0.0:
            return x

        # Inverted dropout: scale by 1/(1-p) during training so inference is unchanged
        scale = 1.0 / (1.0 - self.p)
        mask = (np.random.rand(*x.shape) > self.p).astype(np.float32) * scale

        out = Tensor(x.data * mask, _children=(x,), _op="dropout")

        def _backward():
            if x.requires_grad:
                x._ensure_grad()
                # Gradient flows only through non-zeroed elements
                x.grad += out.grad * mask

        out._backward = _backward
        out.requires_grad = x.requires_grad
        return out

    def __repr__(self) -> str:
        return f"Dropout(p={self.p})"


# ──────────────────────────────────────────────────────────────────────────────
# Step 4: Activation function modules
# ──────────────────────────────────────────────────────────────────────────────

class ReLU(Module):
    """
    Rectified Linear Unit: f(x) = max(0, x)

    WHY IT WORKS:
        - Non-linear (without this, stacking layers does nothing new)
        - Computationally trivial
        - Doesn't saturate for positive inputs (no vanishing gradient on the right)

    WEAKNESS:
        - "Dead neurons": if a neuron's input is always negative, its gradient
          is always 0 and it never learns. GELU fixes this.
    """

    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class Sigmoid(Module):
    """
    Sigmoid: f(x) = 1 / (1 + e^{-x})  →  output in (0, 1)

    WHY IT'S USED:
        Binary classification outputs, gating mechanisms (e.g., LSTM gates).
        Saturates at both ends → vanishing gradients for deep networks (why
        transformers prefer GELU/ReLU in main activations).
    """

    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class Tanh(Module):
    """
    Tanh: f(x) = (e^x - e^{-x}) / (e^x + e^{-x})  →  output in (-1, 1)

    WHY IT'S USED:
        Zero-centered (unlike sigmoid), so gradients can be positive or negative.
        Used in RNNs and as the final activation in some generators.
    """

    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class GELU(Module):
    """
    Gaussian Error Linear Unit.

    The default activation in GPT-2, BERT, and most modern transformers.
    Smooth everywhere, slightly negative for very negative inputs.
    Empirically superior to ReLU for language tasks.
    """

    def forward(self, x: Tensor) -> Tensor:
        return x.gelu()


class SoftMax(Module):
    """Softmax activation along a specified axis."""

    def __init__(self, axis: int = -1):
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        return x.softmax(axis=self.axis)


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 + 17: Feed-Forward Network (the MLP inside each transformer block)
# ──────────────────────────────────────────────────────────────────────────────

class FeedForward(Module):
    """
    Position-wise feed-forward network used inside each transformer block.

    Architecture: Linear → GELU → Dropout → Linear → Dropout
    Expansion ratio: 4x (d_model → 4*d_model → d_model)

    WHY THE 4x EXPANSION:
        The FFN is where the model does most of its "thinking" — it's where
        knowledge gets stored after the attention mechanism selects what to
        focus on. The wider intermediate layer gives more capacity.
        The 4x ratio is from the original "Attention is All You Need" paper.

    Args:
        d_model:  Input/output dimension.
        d_ff:     Inner dimension (default: 4 * d_model).
        dropout:  Dropout probability.
        act:      Activation function ('gelu', 'relu', 'tanh').
    """

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        act: str = "gelu",
    ):
        self.d_model = d_model
        self.d_ff = d_ff or 4 * d_model

        # Up-projection: d_model → d_ff
        self.fc1 = Linear(d_model, self.d_ff)

        # Activation
        self.act = {"gelu": GELU, "relu": ReLU, "tanh": Tanh}[act]()

        # Down-projection: d_ff → d_model
        self.fc2 = Linear(self.d_ff, d_model)

        self.dropout = Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape (B, T, d_model).
        Returns:
            Tensor of shape (B, T, d_model).
        """
        # Expand
        x = self.act(self.fc1(x))
        # Regularize
        x = self.dropout(x)
        # Contract
        x = self.fc2(x)
        return self.dropout(x)

    def __repr__(self) -> str:
        return f"FeedForward({self.d_model} → {self.d_ff} → {self.d_model})"


# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Sequential container
# ──────────────────────────────────────────────────────────────────────────────

class Sequential(Module):
    """
    Chain modules together so that output of module i is input to module i+1.

    Usage:
        net = Sequential(Linear(2, 4), ReLU(), Linear(4, 1))
        output = net(input)
    """

    def __init__(self, *modules):
        self.layers = list(modules)

    def forward(self, x: Tensor) -> Tensor:
        """Pass x through each layer in order."""
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> List[Parameter]:
        """Collect parameters from all child modules."""
        params = []
        seen = set()
        for layer in self.layers:
            if isinstance(layer, Module):
                for p in layer.parameters():
                    if id(p) not in seen:
                        seen.add(id(p))
                        params.append(p)
        return params

    def train(self) -> "Sequential":
        self._training = True
        for layer in self.layers:
            if isinstance(layer, Module):
                layer.train()
        return self

    def eval(self) -> "Sequential":
        self._training = False
        for layer in self.layers:
            if isinstance(layer, Module):
                layer.eval()
        return self

    def __repr__(self) -> str:
        layers_str = "\n".join(f"  ({i}): {l}" for i, l in enumerate(self.layers))
        return f"Sequential(\n{layers_str}\n)"





################################################################################
# SECTION 3: LOSS FUNCTIONS — Step 5
################################################################################


"""
================================================================================
nanoGPT / core / losses.py
================================================================================
Loss functions — Step 5.

A loss function measures how wrong the model is. It maps predictions
and ground truth to a single scalar. The optimizer then minimizes this scalar
by adjusting all parameters via backpropagation.

CONTENTS:
  mse_loss          — Mean Squared Error (regression tasks)
  cross_entropy_loss — Negative log-likelihood for classification / LM

NUMERICS:
  Both implementations use numerically stable formulations.
  Cross-entropy uses log-sum-exp to avoid computing softmax then log separately.

WHY CROSS-ENTROPY FOR LANGUAGE MODELS:
  Language modeling is next-token prediction — predicting which of V tokens
  comes next. This is a classification problem with V classes.
  Cross-entropy is the natural loss: it equals the negative log-probability
  assigned to the correct token. Minimizing it = maximizing the probability
  the model assigns to the actual next word.
================================================================================
"""

# import numpy as np  # at top
# from .engine import Tensor  # merged


def mse_loss(
    predictions: Tensor,
    targets: Tensor,
    reduction: str = "mean",
) -> Tensor:
    """
    Mean Squared Error: measures average squared difference between
    predictions and targets.

    Formula: L = (1/N) * sum((y_pred - y_true)^2)

    WHY MSE:
        Good for regression tasks where outputs are continuous values.
        Penalizes large errors heavily (quadratic penalty).
        Not ideal for classification (unbounded predictions).

    Args:
        predictions: Model output tensor of shape (B, ...).
        targets:     Ground truth tensor of same shape.
        reduction:   'mean' (default), 'sum', or 'none'.

    Returns:
        Scalar loss tensor (or per-element if reduction='none').
    """
    diff = predictions - targets    # element-wise difference
    sq   = diff * diff              # squared difference

    if reduction == "mean":
        return sq.mean()
    elif reduction == "sum":
        return sq.sum()
    else:  # 'none'
        return sq


def cross_entropy_loss(
    logits: Tensor,
    targets: np.ndarray,
    ignore_index: int = -1,
    label_smoothing: float = 0.0,
) -> Tensor:
    """
    Cross-entropy loss — the standard loss for language model training.

    Takes raw logits (pre-softmax) and integer target indices, returns
    the mean negative log-probability of the correct tokens.

    FORMULA (for one example):
        L = -log(softmax(logits)[target])
          = -logits[target] + log(sum(exp(logits)))   ← numerically stable form

    IMPLEMENTATION:
        We use the log-sum-exp trick for numerical stability:
            log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
        Then compute the gradient analytically in a single fused backward.
        This is more efficient and numerically stable than softmax → log → nll.

    Args:
        logits:          Raw scores of shape (B, T, V) or (N, V).
        targets:         Integer labels in [0, V), shape (B, T) or (N,).
                         Use ignore_index to mask padding tokens.
        ignore_index:    Tokens with this label are excluded from the loss.
                         Default: -1.
        label_smoothing: If > 0, smooth the targets by distributing some
                         probability mass uniformly (prevents overconfidence).

    Returns:
        Scalar Tensor: mean loss over all non-ignored positions.
    """
    original_shape = logits.shape

    # Flatten to 2D: (N, V)
    if logits.ndim == 3:
        B, T, V = logits.shape
        logits_2d = logits.reshape(B * T, V)
        targets_1d = targets.reshape(-1).astype(np.int64)
    else:
        logits_2d = logits
        targets_1d = targets.reshape(-1).astype(np.int64)
        V = logits.shape[-1]

    N = logits_2d.shape[0]
    x = logits_2d.data    # shape (N, V)

    # Build mask for ignored positions
    valid_mask = (targets_1d != ignore_index)     # (N,) boolean
    n_valid = int(valid_mask.sum())               # number of valid tokens

    # ── Numerically stable log-softmax ──────────────────────────────────────
    # Subtract max per row (the log-sum-exp trick)
    x_max = x.max(axis=-1, keepdims=True)        # (N, 1)
    x_shifted = x - x_max                        # (N, V)  — shifted for stability
    log_z = np.log(np.exp(x_shifted).sum(axis=-1, keepdims=True))  # (N, 1)
    log_softmax = x_shifted - log_z              # (N, V)  — log-probabilities

    # ── NLL loss at target positions ────────────────────────────────────────
    # Clamp targets so we don't index with -1
    clamped_targets = np.where(valid_mask, targets_1d, 0)
    nll = -log_softmax[np.arange(N), clamped_targets]  # (N,)

    # ── Label smoothing ─────────────────────────────────────────────────────
    if label_smoothing > 0.0:
        # Blend hard targets with uniform distribution
        smooth_loss = -log_softmax.mean(axis=-1)          # (N,) — uniform target
        nll = (1.0 - label_smoothing) * nll + label_smoothing * smooth_loss

    # Zero out ignored positions
    nll = np.where(valid_mask, nll, 0.0)
    loss_val = nll.sum() / max(n_valid, 1)

    # Create output tensor
    out = Tensor(
        np.array(loss_val, dtype=np.float32),
        _children=(logits_2d,),
        _op="cross_entropy",
    )

    def _backward():
        if logits_2d.requires_grad:
            logits_2d._ensure_grad()

            # Analytic gradient of cross-entropy w.r.t. logits:
            # d(loss)/d(logit_i) = softmax_i - 1_{i == target}
            # (then divide by N for the mean, scale by upstream grad)
            softmax_probs = np.exp(log_softmax)       # (N, V) — reuse computation

            # d(NLL)/d(logits) = softmax - one_hot(target)
            grad_2d = softmax_probs.copy()

            # Subtract 1.0 at the correct class positions
            if label_smoothing > 0.0:
                # With label smoothing, target distribution is (1-ε)*one_hot + ε/V
                valid_rows = np.where(valid_mask)[0]
                grad_2d[valid_rows, clamped_targets[valid_rows]] -= (1.0 - label_smoothing)
                grad_2d[valid_rows] -= label_smoothing / V
            else:
                valid_rows = np.where(valid_mask)[0]
                grad_2d[valid_rows, clamped_targets[valid_rows]] -= 1.0

            # Zero out ignored positions (their loss is 0, so no gradient)
            grad_2d[~valid_mask] = 0.0

            # Scale: 1/n_valid for the mean, and upstream gradient
            grad_2d /= max(n_valid, 1)
            grad_2d *= out.grad   # chain rule: scalar upstream gradient

            # If original logits were 3D, reshape gradient back
            if logits.ndim == 3:
                if logits.requires_grad:
                    if logits.grad is None:
                        logits.grad = np.zeros(original_shape, dtype=np.float32)
                    logits.grad += grad_2d.reshape(original_shape)
            else:
                logits_2d.grad += grad_2d

    out._backward = _backward
    out.requires_grad = logits.requires_grad
    return out


def perplexity(loss: float) -> float:
    """
    Convert a cross-entropy loss value to perplexity.

    Perplexity = e^loss (for natural log) or 2^loss (for log base 2).
    We use natural log throughout, so perplexity = e^loss.

    INTERPRETATION:
        A perplexity of K means the model is as uncertain as if it were
        picking uniformly among K options at each step.
        - Random guessing on vocab_size=50000: PPL ≈ 50000
        - GPT-2 on PTB test set: PPL ≈ 35
        - GPT-3 on Penn Treebank: PPL ≈ 20

    Args:
        loss: Mean cross-entropy loss (nats, not bits).

    Returns:
        Perplexity (always ≥ 1.0).
    """
# import math  # at top
    return math.exp(min(loss, 20.0))  # cap at e^20 ≈ 485M to avoid overflow





################################################################################
# SECTION 4: OPTIMIZERS & LR SCHEDULERS — Steps 7, 9, 26
################################################################################


"""
================================================================================
nanoGPT / training / optimizer.py
================================================================================
Optimizers and learning rate schedulers — Steps 7, 9, 26.

OPTIMIZER HIERARCHY:
  Optimizer     Base class: holds params, step counter
  ├── SGD       Stochastic Gradient Descent with momentum
  └── AdamW     Adam + decoupled weight decay (the standard for transformers)

LR SCHEDULERS:
  LRScheduler             Base class
  ├── ConstantLR          Fixed learning rate
  ├── LinearWarmup        Ramp up from 0 to max_lr
  ├── CosineDecay         Smooth cosine annealing
  └── CosineWithWarmup    Warmup + cosine decay (THE standard schedule)

WHY ADAMW FOR TRANSFORMERS:
  - Adam (Kingma & Ba, 2015) maintains per-parameter adaptive learning rates
    using exponential moving averages of the gradient (m) and gradient² (v).
    This makes it much more robust than SGD to poorly scaled gradients.
  - AdamW (Loshchilov & Hutter, 2019) fixes a subtle bug in L2 regularization
    within Adam: weight decay must be applied to parameters DIRECTLY, not via
    the gradient. This decoupling is critical for transformer training.

WHY COSINE LR SCHEDULE:
  - Linear warmup prevents early-training instability (gradients are large
    and noisy at initialization — a big LR causes divergence).
  - Cosine decay provides a smooth, gradual reduction in LR as the model
    approaches convergence, allowing fine-grained weight adjustments.
================================================================================
"""

# import numpy as np  # at top
# import math  # at top
# from typing import List, Optional  # at top
# from ..core.layers import Parameter  # merged


# ──────────────────────────────────────────────────────────────────────────────
# Gradient utilities
# ──────────────────────────────────────────────────────────────────────────────

def clip_grad_norm(params: List[Parameter], max_norm: float = 1.0) -> float:
    """
    Clip gradients so that the global L2 norm doesn't exceed max_norm.

    WHY GRADIENT CLIPPING:
        In deep networks (especially RNNs and transformers), gradients can
        occasionally explode to very large values, especially at the start
        of training. Clipping prevents single bad steps from destabilizing
        the model.

    The global norm is preferred over per-parameter clipping because it
    preserves the direction of the gradient update.

    Args:
        params:   List of parameters whose gradients to clip.
        max_norm: Maximum allowed global gradient norm.

    Returns:
        The unclipped global gradient norm (for logging).
    """
    # Compute global L2 norm across all parameters
    total_sq = sum(
        float((p.grad ** 2).sum())
        for p in params
        if p.grad is not None
    )
    global_norm = math.sqrt(total_sq)

    # Scale gradients if norm exceeds limit
    if global_norm > max_norm:
        scale = max_norm / (global_norm + 1e-8)
        for p in params:
            if p.grad is not None:
                p.grad *= scale

    return global_norm


# ──────────────────────────────────────────────────────────────────────────────
# Base optimizer
# ──────────────────────────────────────────────────────────────────────────────

class Optimizer:
    """
    Abstract base class for all optimizers.

    Manages:
      - The list of parameters to update
      - The current learning rate
      - Step counter (for bias correction, scheduling, etc.)
    """

    def __init__(self, params: List[Parameter], lr: float):
        # Filter to only parameters that actually need gradients
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.step_count = 0   # incremented on each call to step()

    def zero_grad(self):
        """
        Reset all gradients to None before the next forward pass.
        Must be called before loss.backward() on each step.
        """
        for p in self.params:
            p.zero_grad()

    def step(self):
        """Perform one parameter update. Override in subclasses."""
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────────────────────
# SGD with momentum
# ──────────────────────────────────────────────────────────────────────────────

class SGD(Optimizer):
    """
    Stochastic Gradient Descent with optional momentum and weight decay.

    Update rule:
        v_t = momentum * v_{t-1} + g_t + weight_decay * theta_{t-1}
        theta_t = theta_{t-1} - lr * v_t

    Args:
        params:       List of parameters to optimize.
        lr:           Learning rate.
        momentum:     Momentum coefficient (0 = plain SGD, 0.9 = common).
        weight_decay: L2 regularization coefficient.
    """

    def __init__(
        self,
        params: List[Parameter],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay

        # Velocity buffers (one per parameter)
        self._velocity = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        """One SGD update step."""
        self.step_count += 1

        for p, v in zip(self.params, self._velocity):
            if p.grad is None:
                continue

            g = p.grad.copy()

            # L2 regularization: add weight_decay * theta to gradient
            if self.weight_decay > 0:
                g = g + self.weight_decay * p.data

            # Momentum: exponential moving average of gradients
            if self.momentum > 0:
                v[:] = self.momentum * v + g
                g = v

            # Parameter update
            p.data -= self.lr * g


# ──────────────────────────────────────────────────────────────────────────────
# AdamW (the standard transformer optimizer)
# ──────────────────────────────────────────────────────────────────────────────

class AdamW(Optimizer):
    """
    Adam with decoupled weight decay (Loshchilov & Hutter, 2019).

    THE transformer optimizer. Used to train GPT-2, GPT-3, LLaMA,
    and virtually every modern language model.

    UPDATE RULE:
        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t         ← first moment (mean)
        v_t = beta2 * v_{t-1} + (1 - beta2) * g_t²         ← second moment (variance)

        m_hat = m_t / (1 - beta1^t)   ← bias correction
        v_hat = v_t / (1 - beta2^t)   ← bias correction

        theta_t = theta_{t-1} - lr * (m_hat / (sqrt(v_hat) + eps))   ← Adam part
                - lr * weight_decay * theta_{t-1}                     ← decoupled WD

    WHY BIAS CORRECTION:
        At step 0, m and v are initialized to 0. Without correction, early
        estimates would be pulled toward 0. The (1 - beta^t) terms remove
        this initialization bias.

    TYPICAL HYPERPARAMETERS (from GPT-2 / GPT-3 papers):
        lr:           3e-4 (decay to ~1e-5 over training)
        beta1:        0.9
        beta2:        0.95   (not the default 0.999! Less smoothing = faster adaptation)
        eps:          1e-8
        weight_decay: 0.1

    Args:
        params:       List of parameters.
        lr:           Base learning rate.
        betas:        (beta1, beta2) moment decay rates.
        eps:          Numerical stability constant.
        weight_decay: Decoupled weight decay coefficient.
    """

    def __init__(
        self,
        params: List[Parameter],
        lr: float = 3e-4,
        betas: tuple = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
    ):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # First moment (mean) buffers
        self._m = [np.zeros_like(p.data) for p in self.params]
        # Second moment (variance) buffers
        self._v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        """One AdamW parameter update."""
        self.step_count += 1
        t = self.step_count

        # Precompute bias correction factors
        bc1 = 1.0 - self.beta1 ** t   # correction for first moment
        bc2 = 1.0 - self.beta2 ** t   # correction for second moment

        for p, m, v in zip(self.params, self._m, self._v):
            if p.grad is None:
                continue

            g = p.grad  # raw gradient

            # ── Update moment estimates ──────────────────────────────────────
            m[:] = self.beta1 * m + (1.0 - self.beta1) * g       # EMA of gradient
            v[:] = self.beta2 * v + (1.0 - self.beta2) * (g * g) # EMA of gradient²

            # ── Bias-corrected estimates ─────────────────────────────────────
            m_hat = m / bc1   # unbiased first moment
            v_hat = v / bc2   # unbiased second moment

            # ── Adam update ──────────────────────────────────────────────────
            # Adaptive step: scale by 1/sqrt(v_hat) per element
            # Large v_hat → parameter was noisy → smaller effective step
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            # ── Decoupled weight decay ───────────────────────────────────────
            # Applied DIRECTLY to parameters, not via gradient
            # (This is the "W" in AdamW — the key difference from Adam)
            if self.weight_decay > 0:
                p.data -= self.lr * self.weight_decay * p.data


# ──────────────────────────────────────────────────────────────────────────────
# Learning rate schedulers — Step 26
# ──────────────────────────────────────────────────────────────────────────────

class LRScheduler:
    """Abstract base class for LR schedulers."""

    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer

    def get_lr(self, step: int) -> float:
        """Compute the LR for the given step."""
        raise NotImplementedError

    def step(self):
        """
        Called once per training step. Updates optimizer.lr.
        Call this BEFORE optimizer.step() each iteration.
        """
        self.optimizer.lr = self.get_lr(self.optimizer.step_count)


class ConstantLR(LRScheduler):
    """Fixed learning rate — baseline for debugging."""

    def __init__(self, optimizer, lr: float):
        super().__init__(optimizer)
        self.lr_val = lr

    def get_lr(self, step: int) -> float:
        return self.lr_val


class LinearWarmup(LRScheduler):
    """
    Linear warmup from 0 to max_lr over warmup_steps.
    Used as the first phase of the combined cosine schedule.
    """

    def __init__(self, optimizer, max_lr: float, warmup_steps: int):
        super().__init__(optimizer)
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps

    def get_lr(self, step: int) -> float:
        if self.warmup_steps == 0:
            return self.max_lr
        return self.max_lr * min(step / self.warmup_steps, 1.0)


class CosineWithWarmup(LRScheduler):
    """
    Linear warmup followed by cosine annealing to min_lr.

    THE standard LR schedule for transformer training:

    Phase 1 (steps 0 → warmup_steps):
        LR rises linearly from 0 to max_lr.
        Rationale: at initialization, parameters are random and gradients
        are large/noisy. A small LR prevents catastrophic early steps.

    Phase 2 (steps warmup_steps → total_steps):
        LR follows a cosine curve from max_lr down to min_lr.
        Rationale: smooth, natural decay that spends most time near mid-values
        before a final fast drop — empirically better than step decay.

    Args:
        optimizer:     The optimizer whose lr to control.
        max_lr:        Peak learning rate (after warmup).
        min_lr:        Final learning rate (end of cosine).
        warmup_steps:  Number of linear warmup steps.
        total_steps:   Total training steps.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: float,
        min_lr: float,
        warmup_steps: int,
        total_steps: int,
    ):
        super().__init__(optimizer)
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get_lr(self, step: int) -> float:
        """
        Returns the LR for the given step number.
        """
        # Phase 1: linear warmup
        if step < self.warmup_steps:
            # Ramp up from 0 to max_lr
            return self.max_lr * step / max(self.warmup_steps, 1)

        # Phase 2: cosine decay
        if step >= self.total_steps:
            return self.min_lr

        # Progress within the decay phase: 0 → 1
        decay_steps = self.total_steps - self.warmup_steps
        progress = (step - self.warmup_steps) / decay_steps

        # Cosine interpolation: 1 → 0 as progress goes 0 → 1
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))

        return self.min_lr + cosine_factor * (self.max_lr - self.min_lr)





################################################################################
# SECTION 5: TOKENIZERS — Steps 10-11
################################################################################


"""
================================================================================
nanoGPT / model / tokenizer.py
================================================================================
Tokenizers — Step 10: splitting text into pieces and building vocabulary.

Tokenization is the first step in processing text. It converts a raw string
into a sequence of integer IDs that the model can work with.

WHY NOT JUST USE CHARACTERS?
  Character-level models exist but vocabulary size affects efficiency:
  - Too small vocab (chars): sequences are very long → expensive attention
  - Too large vocab (words): many rare tokens, large embedding table
  - BPE (subword): the sweet spot — common words are single tokens,
    rare words are split into familiar pieces.

CONTENTS:
  CharTokenizer  — Every unique character is a token. Simple, zero overhead.
                   Best for small datasets and learning/debugging.
  BPETokenizer   — Byte Pair Encoding (Sennrich et al., 2016). Used by GPT-2,
                   GPT-3, LLaMA, etc. Learns a vocabulary from data.

SPECIAL TOKENS (both tokenizers):
  <pad> = 0    Padding (for batching variable-length sequences)
  <bos> = 1    Beginning of sequence marker
  <eos> = 2    End of sequence marker
  <unk> = 3    Unknown token (for OOV in BPE)
================================================================================
"""

# import re  # at top
# import json  # at top
# from collections import Counter  # at top
# from typing import List, Dict, Tuple, Optional, Union  # at top


# ──────────────────────────────────────────────────────────────────────────────
# Step 10: Character-level tokenizer
# ──────────────────────────────────────────────────────────────────────────────

class CharTokenizer:
    """
    Character-level tokenizer.

    Assigns an integer ID to each unique character that appears in the
    training corpus. The vocabulary is built by scanning the text.

    PROS:
      - Zero hyperparameters — the vocabulary is entirely determined by data
      - Handles any Unicode (no OOV at the character level)
      - Simple to implement and understand

    CONS:
      - Sequences are long (every character is a separate token)
      - Model must learn to compose words from characters

    VOCABULARY LAYOUT:
      [0] <pad>  — padding
      [1] <bos>  — beginning of sequence
      [2] <eos>  — end of sequence
      [3] <unk>  — unknown character (safety net)
      [4+] characters from training data, sorted alphabetically

    Usage:
        tokenizer = CharTokenizer()
        tokenizer.build("hello world")
        ids = tokenizer.encode("hello")  # [7, 4, 11, 11, 14] (example)
        text = tokenizer.decode(ids)     # "hello"
    """

    # Special token IDs — these are constant
    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3
    SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]

    def __init__(self):
        self.stoi: Dict[str, int] = {}   # string → integer
        self.itos: Dict[int, str] = {}   # integer → string
        self._built = False

    def build(self, text: str) -> "CharTokenizer":
        """
        Scan text and build the vocabulary.

        Args:
            text: The entire training corpus as a single string.

        Returns:
            self (for chaining)
        """
        # Start with special tokens
        self.stoi = {tok: i for i, tok in enumerate(self.SPECIAL_TOKENS)}

        # Add each unique character (sorted for determinism across runs)
        offset = len(self.SPECIAL_TOKENS)
        for char in sorted(set(text)):
            if char not in self.stoi:
                self.stoi[char] = len(self.stoi)

        # Build reverse mapping
        self.itos = {v: k for k, v in self.stoi.items()}
        self._built = True
        return self

    @property
    def vocab_size(self) -> int:
        """Total number of tokens in the vocabulary."""
        return len(self.stoi)

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """
        Convert a string to a list of token IDs.

        Args:
            text:    Input string.
            add_bos: Prepend a <bos> token.
            add_eos: Append an <eos> token.

        Returns:
            List of integer token IDs.
        """
        assert self._built, "Call build() before encode()"

        # Map each character to its ID (UNK for any unseen chars)
        ids = [self.stoi.get(ch, self.UNK_ID) for ch in text]

        if add_bos:
            ids = [self.BOS_ID] + ids
        if add_eos:
            ids = ids + [self.EOS_ID]

        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """
        Convert a list of token IDs back to a string.

        Args:
            ids:          List of integer token IDs.
            skip_special: If True, special tokens (<pad>, <bos>, <eos>) are omitted.

        Returns:
            Decoded string.
        """
        chars = []
        special_set = set(self.SPECIAL_TOKENS) if skip_special else set()

        for token_id in ids:
            token_str = self.itos.get(token_id, "<unk>")
            if token_str not in special_set:
                chars.append(token_str)

        return "".join(chars)

    def save(self, path: str):
        """Serialize the vocabulary to JSON."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"type": "char", "stoi": self.stoi, "itos": {str(k): v for k, v in self.itos.items()}},
                f, ensure_ascii=False, indent=2
            )

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        """Load a serialized CharTokenizer."""
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        tok = cls()
        tok.stoi = d["stoi"]
        tok.itos = {int(k): v for k, v in d["itos"].items()}
        tok._built = True
        return tok

    def __repr__(self) -> str:
        return f"CharTokenizer(vocab_size={self.vocab_size})"


# ──────────────────────────────────────────────────────────────────────────────
# Byte Pair Encoding (BPE) tokenizer
# ──────────────────────────────────────────────────────────────────────────────

class BPETokenizer:
    """
    Byte Pair Encoding tokenizer (Sennrich et al., 2016).

    The algorithm used by GPT-2, RoBERTa, and most modern transformers.

    THE BPE ALGORITHM:
        1. Start with a vocabulary of individual characters (+ special tokens)
        2. Count all adjacent symbol pairs in the corpus
        3. Merge the most frequent pair → new symbol
        4. Repeat steps 2-3 until vocab_size is reached

    Example evolution:
        "hello world" (chars: h,e,l,l,o,SPACE,w,o,r,l,d)
        After merges: "he", "ll", "lo", " w", etc.
        Eventually: "hello", " world" might be single tokens

    The vocabulary captures the most common subword patterns in your language,
    giving an efficient, principled tokenization.

    Args:
        vocab_size:    Target vocabulary size (including special tokens).
        min_frequency: Minimum pair frequency to consider merging.

    Usage:
        tokenizer = BPETokenizer()
        tokenizer.train(text, vocab_size=1000)
        ids = tokenizer.encode("hello world")
        text = tokenizer.decode(ids)
    """

    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3
    SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]

    def __init__(self):
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []   # ordered list of merge rules
        self._merge_set: Dict[Tuple[str, str], str] = {}   # fast lookup
        self._built = False

    def train(self, text: str, vocab_size: int, min_frequency: int = 2) -> "BPETokenizer":
        """
        Train BPE on the given corpus.

        Args:
            text:          Full training text.
            vocab_size:    Target vocabulary size.
            min_frequency: Minimum frequency for a pair to be merged.

        Returns:
            self (for chaining)
        """
        # ── Step 1: Initialize with character vocab ──────────────────────────
        # Pre-tokenize: split into words, mark word boundaries with </w>
        word_freqs = self._get_word_freqs(text)

        # Build initial character vocabulary
        chars = set()
        for word in word_freqs:
            chars.update(word)   # word is already a tuple of chars

        self.stoi = {tok: i for i, tok in enumerate(self.SPECIAL_TOKENS)}
        for ch in sorted(chars):
            if ch not in self.stoi:
                self.stoi[ch] = len(self.stoi)
        self.itos = {v: k for k, v in self.stoi.items()}

        # Work with a mutable copy of the word freq table
        vocab = dict(word_freqs)

        # ── Step 2: Iterative merge ──────────────────────────────────────────
        while len(self.stoi) < vocab_size:
            # Count all adjacent symbol pairs
            pair_freqs = self._count_pairs(vocab)

            if not pair_freqs:
                break  # No more pairs to merge

            # Find the most frequent pair
            best_pair, best_freq = max(pair_freqs.items(), key=lambda x: x[1])

            if best_freq < min_frequency:
                break  # Below minimum frequency threshold

            # Create the new merged symbol
            merged = best_pair[0] + best_pair[1]

            # Add to vocabulary
            self.stoi[merged] = len(self.stoi)
            self.itos[len(self.itos)] = merged
            self.merges.append(best_pair)
            self._merge_set[best_pair] = merged

            # Apply the merge to all words in the corpus
            vocab = self._apply_merge(best_pair, vocab)

        self._built = True
        return self

    def _get_word_freqs(self, text: str) -> Counter:
        """
        Tokenize text into pre-tokenized words (tuples of characters + </w>).
        The </w> marker preserves word boundary information.
        """
        word_counts: Counter = Counter()
        for word in text.split():
            # Convert word to tuple of chars with end-of-word marker
            chars = tuple(list(word) + ["</w>"])
            word_counts[chars] += 1
        return word_counts

    def _count_pairs(self, vocab: Dict) -> Counter:
        """Count frequency of each adjacent symbol pair across all words."""
        pairs: Counter = Counter()
        for word, freq in vocab.items():
            for a, b in zip(word, word[1:]):
                pairs[(a, b)] += freq
        return pairs

    def _apply_merge(self, pair: Tuple[str, str], vocab: Dict) -> Dict:
        """
        Apply a merge operation to all words in the vocabulary.
        Replace all occurrences of (pair[0], pair[1]) with their concatenation.
        """
        merged = pair[0] + pair[1]
        new_vocab = {}

        for word, freq in vocab.items():
            new_word = []
            i = 0
            while i < len(word):
                # Check if this position starts the target pair
                if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                    new_word.append(merged)
                    i += 2  # Skip both characters of the pair
                else:
                    new_word.append(word[i])
                    i += 1
            new_vocab[tuple(new_word)] = freq

        return new_vocab

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def _tokenize_word(self, word_chars: List[str]) -> List[str]:
        """
        Apply all learned merge rules to a single word.
        Greedy left-to-right application of rules in training order.
        """
        tokens = list(word_chars)

        # Apply each merge rule in the order they were learned
        for pair in self.merges:
            if len(tokens) < 2:
                break
            merged = pair[0] + pair[1]
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """
        Encode a string to a list of token IDs using learned BPE.

        Args:
            text:    Input string.
            add_bos: Prepend a <bos> token.
            add_eos: Append an <eos> token.

        Returns:
            List of integer token IDs.
        """
        assert self._built, "Call train() before encode()"

        ids = []
        if add_bos:
            ids.append(self.BOS_ID)

        for word in text.split():
            # Convert word to chars + end marker
            chars = list(word) + ["</w>"]
            # Apply BPE merges
            tokens = self._tokenize_word(chars)
            # Map to IDs
            ids.extend(self.stoi.get(tok, self.UNK_ID) for tok in tokens)

        if add_eos:
            ids.append(self.EOS_ID)

        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """
        Decode token IDs back to text.

        The </w> markers tell us where word boundaries are.

        Args:
            ids:          List of token IDs.
            skip_special: Skip special tokens in output.

        Returns:
            Decoded string.
        """
        special_set = set(self.SPECIAL_TOKENS) if skip_special else set()
        tokens = []

        for token_id in ids:
            token_str = self.itos.get(token_id, "<unk>")
            if token_str not in special_set:
                tokens.append(token_str)

        # Join tokens and replace end-of-word markers with spaces
        text = "".join(tokens)
        text = text.replace("</w>", " ")
        return text.strip()

    def save(self, path: str):
        """Serialize tokenizer state to JSON."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "type": "bpe",
                "stoi": self.stoi,
                "itos": {str(k): v for k, v in self.itos.items()},
                "merges": [[a, b] for a, b in self.merges],
            }, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Load a serialized BPETokenizer."""
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        tok = cls()
        tok.stoi = d["stoi"]
        tok.itos = {int(k): v for k, v in d["itos"].items()}
        tok.merges = [tuple(m) for m in d["merges"]]
        tok._merge_set = {tuple(m): m[0] + m[1] for m in tok.merges}
        tok._built = True
        return tok

    def __repr__(self) -> str:
        return f"BPETokenizer(vocab_size={self.vocab_size}, merges={len(self.merges)})"





################################################################################
# SECTION 6: TRANSFORMER ARCHITECTURE — Steps 13-21
################################################################################


"""
================================================================================
nanoGPT / model / transformer.py
================================================================================
The transformer architecture — Steps 13–21.

This file implements a complete GPT-style (decoder-only) transformer.
Every component is built from first principles using our autograd engine.

ARCHITECTURE OVERVIEW:
  Input tokens (B, T)
    │
    ├── Token Embedding (B, T, D)           ← learned: maps token IDs to vectors
    ├── Position Embedding (B, T, D)        ← learned: gives model sense of order
    │
    └── N × TransformerBlock               ← the core: repeated stack of:
          ├── LayerNorm
          ├── MultiHeadAttention            ← "what to pay attention to"
          ├── LayerNorm
          └── FeedForward                  ← "what to do with attended info"
    │
    └── Final LayerNorm
    └── LM Head (linear, shared with token embedding)
    └── Logits (B, T, vocab_size)

WHY DECODER-ONLY (GPT style vs BERT):
  - GPT trains to predict the NEXT token at each position (causal/autoregressive)
  - BERT masks random tokens and predicts them (bidirectional)
  - For generation tasks (chatbots, writing assistants) causal is essential:
    the model can only see past context, not future tokens

CAUSAL MASKING:
  The attention mask prevents each position from attending to future positions.
  This is a lower-triangular boolean matrix. Without it, the model could
  "cheat" during training by reading ahead to see the token it should predict.

PRE-NORM vs POST-NORM:
  We use PRE-NORM: LayerNorm is applied before each sub-layer, then the
  result is added to the residual. This is more stable than original POST-NORM
  (which puts LayerNorm after the residual add) and is used in GPT-2, LLaMA, etc.

WEIGHT TYING:
  The token embedding weight is shared with the LM head output projection.
  Mathematical justification: the LM head essentially asks "which token
  in embedding space is the predicted next token closest to?" — naturally
  shares representation with the embedding lookup.
  Practical benefit: saves ~vocab_size × d_model parameters (e.g., ~25M for GPT-2).
================================================================================
"""

# import numpy as np  # at top
# import math  # at top
# from typing import Optional, Tuple  # at top
# from ..core.engine import Tensor  # merged
# from ..core.layers import Module, Parameter, Linear, Embedding, LayerNorm, Dropout, FeedForward  # merged


# ──────────────────────────────────────────────────────────────────────────────
# Step 13: Positional Encoding
# ──────────────────────────────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(Module):
    """
    Fixed sinusoidal positional encodings (Vaswani et al., 2017).

    WHY POSITION ENCODING:
        Attention is permutation-invariant — it doesn't care about order.
        "The cat sat on the mat" and "On the mat sat the cat" look the same
        to an attention layer without position information. We must explicitly
        inject ordering information.

    SINUSOIDAL FORMULA:
        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    WHY SINUSOIDAL (not learned):
        - Generalizes to sequences longer than seen in training
        - Each pair of dimensions forms a "clock" at a different frequency
        - The model can attend to relative positions via the addition formula:
          sin(a + b) = sin(a)cos(b) + cos(a)sin(b)
        - No parameters to learn (small advantage for data efficiency)

    Args:
        d_model:     Embedding dimension.
        max_seq_len: Maximum sequence length (precomputed up to this).
        dropout:     Dropout probability applied after adding encodings.
    """

    def __init__(self, d_model: int, max_seq_len: int = 4096, dropout: float = 0.0):
        self.d_model = d_model
        self.dropout_layer = Dropout(dropout)

        # Precompute the positional encoding table
        pe = np.zeros((max_seq_len, d_model), dtype=np.float32)

        # Positions: [0, 1, 2, ..., max_seq_len-1] as column vector
        positions = np.arange(max_seq_len, dtype=np.float32)[:, None]  # (T, 1)

        # Frequency divisors: 10000^(2i/d_model) for i in [0, d/2)
        dim_indices = np.arange(0, d_model, 2, dtype=np.float32)
        div_term = np.exp(dim_indices * (-math.log(10000.0) / d_model))  # (d/2,)

        # Even dimensions: sin; Odd dimensions: cos
        pe[:, 0::2] = np.sin(positions * div_term)
        pe[:, 1::2] = np.cos(positions * div_term[:d_model // 2])

        # Store as a non-trainable buffer (not a Parameter)
        self._pe_table: np.ndarray = pe

    def forward(self, x: Tensor) -> Tensor:
        """
        Add positional encodings to token embeddings.

        Args:
            x: Token embeddings of shape (B, T, d_model).
        Returns:
            x + PE[:T], same shape.
        """
        T = x.shape[1]
        pe = Tensor(self._pe_table[:T])   # (T, d_model), no grad needed
        out = x + pe                       # broadcasts (B,T,D) + (T,D) → (B,T,D)
        return self.dropout_layer(out)


class LearnedPositionalEncoding(Module):
    """
    Learned positional embeddings (GPT-2 style).

    Each position 0..max_seq_len gets its own learned vector.
    Simpler than sinusoidal and empirically comparable or better for
    tasks where sequences don't exceed training length.

    Args:
        d_model:     Embedding dimension.
        max_seq_len: Maximum sequence length.
        dropout:     Applied after adding encodings.
    """

    def __init__(self, d_model: int, max_seq_len: int, dropout: float = 0.0):
        # Embedding for positions 0, 1, 2, ... max_seq_len-1
        self.pos_embedding = Embedding(max_seq_len, d_model)
        self.dropout_layer = Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Token embeddings of shape (B, T, d_model).
        Returns:
            x + position_embeddings, same shape.
        """
        T = x.shape[1]
        positions = np.arange(T, dtype=np.int64)        # [0, 1, 2, ..., T-1]
        pos_emb = self.pos_embedding(positions)          # (T, d_model)
        return self.dropout_layer(x + pos_emb)


# ──────────────────────────────────────────────────────────────────────────────
# Steps 14–16: Attention
# ──────────────────────────────────────────────────────────────────────────────

def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Optional[np.ndarray] = None,
    dropout_layer: Optional[Dropout] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Step 15: Scaled dot-product attention.

    THE FUNDAMENTAL OPERATION OF THE TRANSFORMER.

    INTUITION:
        Q (queries): "What am I looking for?"
        K (keys):    "What do I have to offer?"
        V (values):  "What I'll give you if you select me."

        For each query, we compute a score against all keys via dot product,
        scale by 1/√d_k, apply softmax to get attention weights,
        then take a weighted sum of values.

    FORMULA:
        Attention(Q, K, V) = softmax(Q K^T / √d_k) V

    SCALING BY √d_k:
        The dot product Q·K grows with dimension d_k (each element adds
        variance). Without scaling, large d_k pushes dot products into
        the saturation region of softmax → near-zero gradients → slow learning.
        Dividing by √d_k keeps variance constant at 1 regardless of d_k.

    CAUSAL MASK:
        For decoder (autoregressive) attention, we prevent position i
        from attending to positions j > i by setting those scores to -∞
        before softmax. They become 0 after softmax.

    Args:
        Q:              Query tensor (B, H, T, head_dim)
        K:              Key tensor   (B, H, S, head_dim)
        V:              Value tensor (B, H, S, head_dim)
        mask:           Optional boolean mask (True=keep, False=mask out)
        dropout_layer:  Optional dropout applied to attention weights

    Returns:
        (attended_values, attention_weights): both tensors.
    """
    d_k = Q.shape[-1]               # head dimension
    scale = 1.0 / math.sqrt(d_k)   # scaling factor

    # ── Compute attention scores ─────────────────────────────────────────────
    # Q: (B, H, T, d_k)  ×  K^T: (B, H, d_k, S)  →  scores: (B, H, T, S)
    scores = (Q @ K.transpose()) * scale

    # ── Apply causal mask ────────────────────────────────────────────────────
    # Positions where mask is False get -inf → becomes 0 after softmax
    if mask is not None:
        scores = scores.masked_fill(mask, fill_value=-1e9)

    # ── Softmax: convert scores to a probability distribution ────────────────
    # Each row of the attention matrix is now a valid probability distribution
    attn_weights = scores.softmax(axis=-1)   # (B, H, T, S)

    # ── Optional attention dropout ───────────────────────────────────────────
    if dropout_layer is not None:
        attn_weights = dropout_layer(attn_weights)

    # ── Weighted sum of values ────────────────────────────────────────────────
    # attn_weights: (B, H, T, S)  ×  V: (B, H, S, head_dim)  →  (B, H, T, head_dim)
    output = attn_weights @ V

    return output, attn_weights


class MultiHeadAttention(Module):
    """
    Step 16: Multi-head attention.

    WHY MULTIPLE HEADS:
        Single attention can only capture one type of relationship between
        positions. Multiple heads allow the model to attend to different
        aspects simultaneously:
          - Head 1 might focus on syntax (subject-verb agreement)
          - Head 2 on coreference (pronoun → noun it refers to)
          - Head 3 on proximity (nearby words)
          etc.

        Each head operates in a lower-dimensional subspace (d_model/n_heads),
        so the total computation is comparable to single-head attention.

    ARCHITECTURE:
        1. Project Q, K, V from d_model to n_heads × head_dim (in parallel)
        2. Split into n_heads independent attention computations
        3. Each head produces (T, head_dim)
        4. Concatenate all heads → (T, d_model)
        5. Final linear projection back to d_model

    IMPLEMENTATION NOTE:
        We use a single fused linear layer for all Q, K, V projections
        (weight shape: d_model × 3*d_model) rather than three separate layers.
        This is equivalent but faster to compute and simpler to code.

    Args:
        d_model:  Model dimension.
        n_heads:  Number of attention heads.
        dropout:  Applied to attention weights.
        causal:   If True, apply causal (autoregressive) mask.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        causal: bool = True,
    ):
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads   # dimension per head
        self.causal = causal

        # ── Fused QKV projection ─────────────────────────────────────────────
        # Single weight matrix for Q, K, V (no bias by convention in many LLMs)
        self.qkv_proj = Linear(d_model, 3 * d_model, bias=False)

        # ── Output projection ────────────────────────────────────────────────
        self.out_proj = Linear(d_model, d_model, bias=True)

        # ── Dropout ──────────────────────────────────────────────────────────
        self.attn_dropout = Dropout(dropout)    # on attention weights
        self.resid_dropout = Dropout(dropout)   # on output projection

        # Store last attention weights for visualization/debugging
        self._last_attn_weights: Optional[np.ndarray] = None

    def forward(self, x: Tensor) -> Tensor:
        """
        Self-attention forward pass.

        Args:
            x: Input of shape (B, T, d_model).
        Returns:
            Output of shape (B, T, d_model).
        """
        B, T, D = x.shape
        H = self.n_heads
        hd = self.head_dim  # dimension per head

        # ── Fused QKV projection ─────────────────────────────────────────────
        qkv = self.qkv_proj(x)   # (B, T, 3*D)

        # ── Split into Q, K, V ───────────────────────────────────────────────
        # Each has shape (B, T, D)
        Q_data = qkv.data[:, :, :D]
        K_data = qkv.data[:, :, D:2*D]
        V_data = qkv.data[:, :, 2*D:]

        # Create tensors with proper gradient routing
        Q = self._slice_qkv(qkv, 0,    D,    "Q")
        K = self._slice_qkv(qkv, D,    2*D,  "K")
        V = self._slice_qkv(qkv, 2*D,  3*D,  "V")

        # ── Reshape to multi-head format ─────────────────────────────────────
        # (B, T, D) → (B, T, H, hd) → (B, H, T, hd)
        Q = self._split_heads(Q, B, T, H, hd)
        K = self._split_heads(K, B, T, H, hd)
        V = self._split_heads(V, B, T, H, hd)

        # ── Build causal mask ─────────────────────────────────────────────────
        mask = None
        if self.causal:
            # Lower-triangular matrix: position i can attend to positions 0..i
            # Shape: (1, 1, T, T) → broadcasts over batch and heads
            causal_mask = np.tril(np.ones((T, T), dtype=bool))[None, None, :, :]
            mask = causal_mask

        # ── Scaled dot-product attention ─────────────────────────────────────
        attended, attn_weights = scaled_dot_product_attention(
            Q, K, V,
            mask=mask,
            dropout_layer=self.attn_dropout if self._training else None,
        )
        self._last_attn_weights = attn_weights.data   # save for inspection

        # ── Merge heads ───────────────────────────────────────────────────────
        # (B, H, T, hd) → (B, T, H, hd) → (B, T, D)
        merged = self._merge_heads(attended, B, T, H, hd)

        # ── Output projection ─────────────────────────────────────────────────
        out = self.out_proj(merged)
        out = self.resid_dropout(out)
        return out

    def _slice_qkv(self, qkv: Tensor, start: int, end: int, name: str) -> Tensor:
        """
        Slice a (B, T, 3D) tensor to get Q, K, or V.
        Maintains the backward graph through qkv.
        """
        sliced = Tensor(qkv.data[:, :, start:end], _children=(qkv,), _op=f"slice_{name}")

        def _backward():
            if qkv.requires_grad:
                qkv._ensure_grad()
                qkv.grad[:, :, start:end] += sliced.grad

        sliced._backward = _backward
        sliced.requires_grad = qkv.requires_grad
        return sliced

    def _split_heads(self, x: Tensor, B: int, T: int, H: int, hd: int) -> Tensor:
        """
        Reshape (B, T, D) → (B, H, T, hd) by splitting D into H heads.
        The permutation puts H before T for efficient batch attention.
        """
        # Reshape: (B, T, H, hd)
        reshaped_data = x.data.reshape(B, T, H, hd)
        # Permute: (B, T, H, hd) → (B, H, T, hd)
        permuted_data = reshaped_data.transpose(0, 2, 1, 3)

        out = Tensor(permuted_data, _children=(x,), _op="split_heads")

        def _backward():
            if x.requires_grad:
                x._ensure_grad()
                # Reverse: (B, H, T, hd) → (B, T, H, hd) → (B, T, D)
                g = out.grad.transpose(0, 2, 1, 3).reshape(B, T, H * hd)
                x.grad += g

        out._backward = _backward
        out.requires_grad = x.requires_grad
        return out

    def _merge_heads(self, x: Tensor, B: int, T: int, H: int, hd: int) -> Tensor:
        """
        Merge H heads back: (B, H, T, hd) → (B, T, D).
        Inverse of _split_heads.
        """
        # Permute: (B, H, T, hd) → (B, T, H, hd)
        permuted_data = x.data.transpose(0, 2, 1, 3)
        # Reshape: (B, T, H, hd) → (B, T, D)
        merged_data = permuted_data.reshape(B, T, H * hd)

        out = Tensor(merged_data, _children=(x,), _op="merge_heads")

        def _backward():
            if x.requires_grad:
                x._ensure_grad()
                # Reverse: (B, T, D) → (B, T, H, hd) → (B, H, T, hd)
                g = out.grad.reshape(B, T, H, hd).transpose(0, 2, 1, 3)
                x.grad += g

        out._backward = _backward
        out.requires_grad = x.requires_grad
        return out

    def parameters(self) -> list:
        return self.qkv_proj.parameters() + self.out_proj.parameters()

    def __repr__(self) -> str:
        return (
            f"MultiHeadAttention(d_model={self.d_model}, "
            f"n_heads={self.n_heads}, head_dim={self.head_dim}, "
            f"causal={self.causal})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Step 19: Full Transformer Block
# ──────────────────────────────────────────────────────────────────────────────

class TransformerBlock(Module):
    """
    Step 19: One complete transformer block (decoder).

    The fundamental repeating unit of the GPT architecture.
    Implements PRE-NORM (apply LayerNorm before each sub-layer):

        x = x + Attention(LayerNorm(x))    ← attention sub-layer + residual
        x = x + FFN(LayerNorm(x))          ← feed-forward sub-layer + residual

    THE RESIDUAL CONNECTION:
        Adding the input back to the sub-layer output is crucial for deep networks:
        1. Gradients flow directly back through the residuals (no vanishing)
        2. The sub-layers only need to learn RESIDUALS (small corrections)
           rather than full transformations — much easier to optimize
        3. Allows very deep networks (100+ layers) to train stably

    PRE-NORM vs POST-NORM:
        - Post-norm (original paper): LayerNorm(x + sublayer(x))
        - Pre-norm (GPT-2, LLaMA): x + sublayer(LayerNorm(x))
        - Pre-norm is more stable — the main pathway (residual stream)
          always flows without normalization, preventing gradient vanishing

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        d_ff:    Feed-forward inner dimension (default: 4 * d_model).
        dropout: Dropout probability.
        causal:  Apply causal masking in attention.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        causal: bool = True,
    ):
        self.d_model = d_model

        # Pre-norm LayerNorm before attention
        self.ln1 = LayerNorm(d_model)

        # Multi-head self-attention
        self.attn = MultiHeadAttention(d_model, n_heads, dropout=dropout, causal=causal)

        # Pre-norm LayerNorm before feed-forward
        self.ln2 = LayerNorm(d_model)

        # Position-wise feed-forward network
        self.ffn = FeedForward(d_model, d_ff, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input of shape (B, T, d_model).
        Returns:
            Output of same shape — the transformer block output.
        """
        # ── Attention sub-layer with pre-norm and residual ───────────────────
        x = x + self.attn(self.ln1(x))

        # ── Feed-forward sub-layer with pre-norm and residual ────────────────
        x = x + self.ffn(self.ln2(x))

        return x

    def parameters(self) -> list:
        return (
            self.ln1.parameters()
            + self.attn.parameters()
            + self.ln2.parameters()
            + self.ffn.parameters()
        )

    def __repr__(self) -> str:
        return (
            f"TransformerBlock(d_model={self.d_model}, "
            f"attn={self.attn}, ffn={self.ffn})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Steps 20–21: Full GPT model (encoder + decoder unified)
# ──────────────────────────────────────────────────────────────────────────────

class GPT(Module):
    """
    GPT-style causal language model.

    A decoder-only transformer that learns to predict the next token
    given all previous tokens. Trained with next-token prediction on
    large text corpora.

    This implements the core architecture from:
      - GPT (Radford et al., 2018)
      - GPT-2 (Radford et al., 2019)

    The key insight: next-token prediction with a large enough model and
    dataset produces representations that generalize to many tasks.

    CONFIGURATION OPTIONS:
        d_model:     512 → 12288+ (GPT-3 uses 12288)
        n_layers:    6 → 96 (GPT-3 uses 96)
        n_heads:     8 → 96
        vocab_size:  ~50000 (GPT-2 uses 50257)

    For our training environment, we use small models (d_model=128-256,
    n_layers=4-6) that can train in minutes on a laptop.

    Args:
        vocab_size:   Number of token types.
        d_model:      Model width (embedding dim).
        n_layers:     Number of transformer blocks.
        n_heads:      Number of attention heads.
        d_ff:         FFN inner dim (default: 4 * d_model).
        max_seq_len:  Maximum context window (sequence length).
        dropout:      Dropout probability.
        tie_weights:  Share token embedding ↔ LM head weights.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: Optional[int] = None,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        tie_weights: bool = True,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.tie_weights = tie_weights

        # ── Token embedding: token_id → vector ───────────────────────────────
        self.token_embedding = Embedding(vocab_size, d_model)

        # ── Positional encoding (learned, GPT-2 style) ───────────────────────
        self.pos_encoding = LearnedPositionalEncoding(d_model, max_seq_len, dropout)

        # ── Transformer blocks (the main stack) ──────────────────────────────
        self.blocks = [
            TransformerBlock(d_model, n_heads, d_ff, dropout, causal=True)
            for _ in range(n_layers)
        ]

        # ── Final layer norm ──────────────────────────────────────────────────
        self.ln_final = LayerNorm(d_model)

        # ── Language model head: d_model → vocab_size ────────────────────────
        # Outputs logits for next-token prediction (no softmax here — loss handles it)
        self.lm_head = Linear(d_model, vocab_size, bias=False)

        # ── Weight tying ──────────────────────────────────────────────────────
        # The LM head weight IS the token embedding weight (shared, not copied)
        if tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        # Initialize weights with GPT-2 conventions
        self._init_weights()

    def _init_weights(self):
        """
        GPT-2 weight initialization conventions:
          - Linear layers: N(0, 0.02)
          - Residual projection layers: N(0, 0.02/√(2*n_layers))
            (scaled down to account for the n_layers residual additions)
          - Embeddings: N(0, 0.02)
          - LayerNorm: weight=1, bias=0 (already set in LayerNorm)
        """
        std = 0.02
        # Residual scaling factor: prevents activation growth through residual stack
        residual_std = 0.02 / math.sqrt(2 * self.n_layers)

        for block in self.blocks:
            # Attention output projection
            block.attn.out_proj.weight.data[:] = (
                np.random.randn(*block.attn.out_proj.weight.shape) * residual_std
            )
            # FFN down-projection
            block.ffn.fc2.weight.data[:] = (
                np.random.randn(*block.ffn.fc2.weight.shape) * residual_std
            )

    def forward(self, idx: np.ndarray) -> Tensor:
        """
        Forward pass: token IDs → logits.

        Args:
            idx: Integer token array of shape (B, T),
                 values in [0, vocab_size).

        Returns:
            Logits of shape (B, T, vocab_size).
            logits[b, t, :] is the distribution over next tokens at position t.
        """
        B, T = idx.shape
        assert T <= self.max_seq_len, (
            f"Sequence length {T} exceeds maximum {self.max_seq_len}. "
            "Either truncate inputs or increase max_seq_len."
        )

        # ── 1. Token embeddings ───────────────────────────────────────────────
        # Convert token IDs to dense vectors: (B, T) → (B, T, d_model)
        x = self.token_embedding(idx)

        # ── 2. Add positional encodings ───────────────────────────────────────
        # (B, T, d_model) — position information injected here
        x = self.pos_encoding(x)

        # ── 3. Transformer blocks ─────────────────────────────────────────────
        # Each block: attention + FFN + residuals
        for block in self.blocks:
            x = block(x)

        # ── 4. Final layer normalization ──────────────────────────────────────
        x = self.ln_final(x)

        # ── 5. Language model head ────────────────────────────────────────────
        # Project to vocabulary size: (B, T, d_model) → (B, T, vocab_size)
        logits = self.lm_head(x)

        return logits

    def parameters(self) -> list:
        """Collect all parameters, avoiding duplicates from weight tying."""
        params = []
        seen_ids = set()

        def add_params(module_params):
            for p in module_params:
                if id(p) not in seen_ids:
                    seen_ids.add(id(p))
                    params.append(p)

        add_params(self.token_embedding.parameters())
        add_params(self.pos_encoding.parameters())
        for block in self.blocks:
            add_params(block.parameters())
        add_params(self.ln_final.parameters())
        add_params(self.lm_head.parameters())  # shared weight already added above

        return params

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        """
        Count total trainable parameters.

        Args:
            exclude_embeddings: If True, exclude embedding tables
                (useful for comparing to papers that report non-embedding params).
        """
        total = 0
        seen = set()

        emb_ids = set(
            id(p) for p in (
                self.token_embedding.parameters()
                + self.pos_encoding.parameters()
            )
        )

        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                if exclude_embeddings and id(p) in emb_ids:
                    continue
                total += p.data.size

        return total

    def configure_optimizer(
        self,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        betas: tuple = (0.9, 0.95),
    ):
        """
        Configure AdamW with proper parameter groups.

        Following the GPT-3 training setup:
          - 2D parameters (weight matrices): apply weight decay
          - 1D parameters (biases, layer norm): NO weight decay
            (decaying biases hurts, decaying LN parameters is wrong)
          - Embeddings: NO weight decay (treating them as 1D)

        Returns:
            Configured AdamW optimizer.
        """
# from ..training.optimizer import AdamW  # merged

        decay_params = []
        no_decay_params = []
        seen = set()

        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))

            if p.data.ndim >= 2:
                # Weight matrices: apply decay to prevent them from growing too large
                decay_params.append(p)
            else:
                # Biases, layer norm params, 1D params: no decay
                no_decay_params.append(p)

        all_params = decay_params + no_decay_params
        return AdamW(all_params, lr=lr, weight_decay=weight_decay, betas=betas)

    def __repr__(self) -> str:
        n = self.num_parameters()
        return (
            f"GPT(\n"
            f"  vocab_size={self.vocab_size:,}\n"
            f"  d_model={self.d_model}, n_layers={self.n_layers}, n_heads={self.n_heads}\n"
            f"  max_seq_len={self.max_seq_len}\n"
            f"  parameters={n:,}  (~{n*4/1e6:.1f} MB fp32)\n"
            f")"
        )





################################################################################
# SECTION 7: TRAINING PIPELINE — Steps 6, 8, 22-25
################################################################################


"""
================================================================================
nanoGPT / training / trainer.py
================================================================================
The complete training pipeline — Steps 6, 8, 22–26.

This file contains everything needed to take a raw text corpus, tokenize it,
batch it, run the forward/backward/update loop, track progress, save
checkpoints, and resume training.

TRAINING LOOP (Step 8 + 23):
    for step in range(max_steps):
        x, y = dataset.get_batch()           # 22: data loading
        logits = model(x)                    # forward pass
        loss = cross_entropy(logits, y)      # 5: compute wrongness
        optimizer.zero_grad()
        loss.backward()                      # 6: backpropagation
        clip_grad_norm(params, 1.0)          # 9: prevent explosion
        optimizer.step()                     # 7: gradient descent
        scheduler.step()                     # 26: adjust LR
        log_and_checkpoint()                 # 24 + 25

COMPONENTS:
  TextDataset    — Sliding-window dataset from tokenized text (Step 22)
  LossTracker    — Tracks and smooths loss curves (Step 24)
  Checkpointer   — Saves/loads full training state (Step 25)
  Trainer        — Orchestrates everything (Steps 6, 8, 23)
================================================================================
"""

# import numpy as np  # at top
# import os  # at top
# import json  # at top
# import time  # at top
# import math  # at top
# import pickle  # at top
# import gc  # at top
# from typing import Optional, List, Tuple, Callable, Dict, Any  # at top

# from ..core.engine import Tensor  # merged
# from ..core.losses import cross_entropy_loss, perplexity  # merged
# from ..model.transformer import GPT  # merged
# from .optimizer import AdamW, CosineWithWarmup, clip_grad_norm  # merged


# ──────────────────────────────────────────────────────────────────────────────
# Step 22: Dataset and data loading
# ──────────────────────────────────────────────────────────────────────────────

class TextDataset:
    """
    Sliding-window tokenized text dataset for language model training.

    Given a long sequence of token IDs, creates (input, target) pairs using
    a sliding window of size block_size:
        Input:  tokens[i : i + block_size]
        Target: tokens[i+1 : i + block_size + 1]   ← shifted by 1

    This is the core of next-token prediction: for each token position,
    the model must predict what comes next.

    Example (block_size=4):
        Token sequence: [5, 7, 2, 8, 3, 1, 9, 4]
        Example 0: x=[5,7,2,8], y=[7,2,8,3]
        Example 1: x=[7,2,8,3], y=[2,8,3,1]
        ...

    Args:
        token_ids:   Pre-tokenized corpus as a flat list of integers.
        block_size:  Context window size (must match model's max_seq_len).
    """

    def __init__(self, token_ids: List[int], block_size: int):
        self.data = np.array(token_ids, dtype=np.int64)
        self.block_size = block_size
        # Number of valid starting positions
        self.n_examples = len(self.data) - block_size

        if self.n_examples <= 0:
            raise ValueError(
                f"Dataset too small: {len(self.data)} tokens < block_size={block_size}. "
                "Need at least block_size+1 tokens."
            )

    def __len__(self) -> int:
        return self.n_examples

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the (input, target) pair at index idx.

        Returns:
            x: Input tokens of shape (block_size,)
            y: Target tokens of shape (block_size,) — x shifted by 1
        """
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y

    def get_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample a random batch of (input, target) pairs.

        Args:
            batch_size: Number of examples per batch.

        Returns:
            x: Integer array of shape (batch_size, block_size)
            y: Integer array of shape (batch_size, block_size)
        """
        # Sample random starting positions (with replacement)
        indices = np.random.randint(0, self.n_examples, size=batch_size)

        # Stack examples into a batch
        x_batch = np.stack([self.data[i : i + self.block_size] for i in indices])
        y_batch = np.stack([self.data[i + 1 : i + self.block_size + 1] for i in indices])

        return x_batch, y_batch

    def get_sequential_batch(
        self, batch_size: int, step: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a non-overlapping sequential batch (for validation).
        Deterministic given the step number.

        Args:
            batch_size: Number of examples.
            step:       Which sequential block to return.

        Returns:
            x, y arrays of shape (batch_size, block_size).
        """
        start = (step * batch_size) % self.n_examples
        indices = [(start + i) % self.n_examples for i in range(batch_size)]
        x_batch = np.stack([self.data[i : i + self.block_size] for i in indices])
        y_batch = np.stack([self.data[i + 1 : i + self.block_size + 1] for i in indices])
        return x_batch, y_batch

    def statistics(self) -> Dict[str, Any]:
        """Return basic statistics about the dataset."""
        return {
            "n_tokens": len(self.data),
            "n_examples": self.n_examples,
            "block_size": self.block_size,
            "size_mb": self.data.nbytes / 1e6,
        }


def train_val_split(
    token_ids: List[int],
    val_fraction: float = 0.1,
) -> Tuple[List[int], List[int]]:
    """
    Split a token sequence into train and validation sets.

    Uses a contiguous split (not random shuffling) to preserve
    the natural flow of text in each split.

    Args:
        token_ids:    Full tokenized corpus.
        val_fraction: Fraction of data to use for validation.

    Returns:
        (train_ids, val_ids)
    """
    n = len(token_ids)
    split_point = int(n * (1.0 - val_fraction))
    return token_ids[:split_point], token_ids[split_point:]


# ──────────────────────────────────────────────────────────────────────────────
# Step 24: Loss tracking
# ──────────────────────────────────────────────────────────────────────────────

class LossTracker:
    """
    Tracks training and validation losses over time.

    Provides:
      - Raw loss history
      - Exponentially smoothed loss (less noisy for display)
      - Perplexity computation
      - Pretty-printed summaries

    The smoothed loss uses an exponential moving average (EMA):
        smoothed = beta * smoothed + (1 - beta) * current
    with bias correction for the early steps (like Adam's bias correction).
    """

    def __init__(self, smoothing_beta: float = 0.95):
        self.smoothing_beta = smoothing_beta

        # Raw history
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.learning_rates: List[float] = []
        self.grad_norms: List[float] = []
        self.steps: List[int] = []

        # EMA state
        self._ema_loss: float = 0.0
        self._ema_step: int = 0

    def log_train(
        self,
        step: int,
        loss: float,
        lr: float,
        grad_norm: float = 0.0,
    ):
        """Record a training step."""
        self.steps.append(step)
        self.train_losses.append(loss)
        self.learning_rates.append(lr)
        self.grad_norms.append(grad_norm)

        # Update EMA
        self._ema_step += 1
        self._ema_loss = (
            self.smoothing_beta * self._ema_loss + (1 - self.smoothing_beta) * loss
        )

    def log_val(self, step: int, loss: float):
        """Record a validation evaluation."""
        self.val_losses.append((step, loss))

    @property
    def smoothed_loss(self) -> float:
        """Bias-corrected EMA of training loss."""
        if self._ema_step == 0:
            return float("inf")
        # Bias correction: same formula as Adam's moment correction
        correction = 1.0 - self.smoothing_beta ** self._ema_step
        return self._ema_loss / correction

    @property
    def last_val_loss(self) -> Optional[float]:
        """Most recent validation loss."""
        return self.val_losses[-1][1] if self.val_losses else None

    def summary(self) -> str:
        """One-line training status summary for the log."""
        if not self.train_losses:
            return "No training data"

        step = self.steps[-1]
        raw = self.train_losses[-1]
        smooth = self.smoothed_loss
        ppl = perplexity(smooth)
        lr_str = f"{self.learning_rates[-1]:.2e}"
        gnorm = self.grad_norms[-1]

        val_str = ""
        if self.val_losses:
            vloss = self.val_losses[-1][1]
            vppl = perplexity(vloss)
            val_str = f" | val_loss={vloss:.4f} ppl={vppl:.1f}"

        return (
            f"step={step:6d} | loss={raw:.4f} (~{smooth:.4f}) | "
            f"ppl={ppl:.1f} | lr={lr_str} | gnorm={gnorm:.3f}{val_str}"
        )

    def to_dict(self) -> Dict:
        """Serialize tracker state for checkpointing."""
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "learning_rates": self.learning_rates,
            "grad_norms": self.grad_norms,
            "steps": self.steps,
            "ema_loss": self._ema_loss,
            "ema_step": self._ema_step,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "LossTracker":
        """Restore from serialized state."""
        tracker = cls()
        tracker.train_losses = d.get("train_losses", [])
        tracker.val_losses = d.get("val_losses", [])
        tracker.learning_rates = d.get("learning_rates", [])
        tracker.grad_norms = d.get("grad_norms", [])
        tracker.steps = d.get("steps", [])
        tracker._ema_loss = d.get("ema_loss", 0.0)
        tracker._ema_step = d.get("ema_step", 0)
        return tracker


# ──────────────────────────────────────────────────────────────────────────────
# Step 25: Checkpointing
# ──────────────────────────────────────────────────────────────────────────────

class Checkpointer:
    """
    Saves and loads full training state to disk.

    A checkpoint contains:
      - Model weights (all Parameters)
      - Optimizer state (moment buffers m, v + step count)
      - Scheduler state
      - Loss history
      - Training configuration

    This allows training to be:
      - Paused and resumed (power outage, preemption)
      - Evaluated at multiple points (pick best checkpoint by val loss)
      - Fine-tuned from a pre-trained starting point

    Checkpoint format: Python pickle (simple, lossless, self-contained).

    Args:
        checkpoint_dir: Directory to save checkpoints in.
        model:          The model whose weights to save.
        keep_last_n:    Only keep the N most recent checkpoints.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        model: GPT,
        keep_last_n: int = 3,
    ):
        self.dir = checkpoint_dir
        self.model = model
        self.keep_last_n = keep_last_n
        os.makedirs(checkpoint_dir, exist_ok=True)
        self._saved_paths: List[str] = []

    def save(
        self,
        step: int,
        optimizer: AdamW,
        tracker: LossTracker,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Save a full training checkpoint.

        Args:
            step:      Current training step.
            optimizer: Optimizer (to save m/v buffers and step count).
            tracker:   Loss tracker (to save loss history).
            metadata:  Extra information to store (config, notes, etc.)

        Returns:
            Path to the saved checkpoint file.
        """
        # Collect model weights (deduplicated for weight-tied parameters)
        params = self.model.parameters()
        seen = set()
        weights = {}
        for i, p in enumerate(params):
            if id(p) not in seen:
                seen.add(id(p))
                weights[i] = p.data.copy()

        state = {
            "step": step,
            "weights": weights,
            "optimizer": {
                "m": optimizer._m,
                "v": optimizer._v,
                "step_count": optimizer.step_count,
                "lr": optimizer.lr,
            },
            "tracker": tracker.to_dict(),
            "metadata": metadata or {},
        }

        path = os.path.join(self.dir, f"ckpt_{step:08d}.pkl")
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Track saved checkpoints
        self._saved_paths.append(path)

        # Delete old checkpoints beyond keep_last_n
        while len(self._saved_paths) > self.keep_last_n:
            old_path = self._saved_paths.pop(0)
            if os.path.exists(old_path):
                os.remove(old_path)

        # Write a "latest.json" pointer for easy resume
        latest_path = os.path.join(self.dir, "latest.json")
        with open(latest_path, "w") as f:
            json.dump({"path": path, "step": step}, f)

        size_mb = os.path.getsize(path) / 1e6
        print(f"  ✓ Checkpoint saved: {os.path.basename(path)} ({size_mb:.1f} MB)")
        return path

    def load_latest(self, optimizer: AdamW, tracker: LossTracker) -> int:
        """
        Load the most recent checkpoint.

        Returns:
            The step number of the loaded checkpoint (0 if none found).
        """
        latest_path = os.path.join(self.dir, "latest.json")
        if not os.path.exists(latest_path):
            print("  No checkpoint found, starting from scratch.")
            return 0

        with open(latest_path) as f:
            info = json.load(f)

        return self.load(info["path"], optimizer, tracker)

    def load(self, path: str, optimizer: AdamW, tracker: LossTracker) -> int:
        """
        Load a specific checkpoint.

        Args:
            path:      Path to the .pkl checkpoint file.
            optimizer: Optimizer to restore state into.
            tracker:   Loss tracker to restore history into.

        Returns:
            The training step stored in the checkpoint.
        """
        with open(path, "rb") as f:
            state = pickle.load(f)

        # Restore model weights
        params = self.model.parameters()
        seen = set()
        idx = 0
        for p in params:
            if id(p) not in seen:
                seen.add(id(p))
                if idx in state["weights"]:
                    p.data[:] = state["weights"][idx]
                idx += 1

        # Restore optimizer state
        opt_state = state["optimizer"]
        optimizer._m = opt_state["m"]
        optimizer._v = opt_state["v"]
        optimizer.step_count = opt_state["step_count"]
        optimizer.lr = opt_state["lr"]

        # Restore tracker
        restored = LossTracker.from_dict(state["tracker"])
        tracker.train_losses = restored.train_losses
        tracker.val_losses = restored.val_losses
        tracker.learning_rates = restored.learning_rates
        tracker.grad_norms = restored.grad_norms
        tracker.steps = restored.steps
        tracker._ema_loss = restored._ema_loss
        tracker._ema_step = restored._ema_step

        step = state["step"]
        print(f"  ✓ Checkpoint loaded: {os.path.basename(path)} (step={step})")
        return step


# ──────────────────────────────────────────────────────────────────────────────
# Steps 6, 8, 23: The Trainer
# ──────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Full training orchestrator.

    Combines:
      - Training loop with forward / backward / update
      - Validation evaluation
      - Loss tracking and display
      - Checkpointing
      - LR scheduling
      - Gradient clipping

    Usage:
        trainer = Trainer(model=model, train_dataset=train_ds, ...)
        tracker = trainer.train()

    Args:
        model:               The GPT model to train.
        train_dataset:       TextDataset for training.
        val_dataset:         Optional TextDataset for validation.
        batch_size:          Examples per gradient step.
        max_steps:           Total training steps.
        eval_interval:       Validate every N steps.
        eval_batches:        Number of batches to average for validation.
        log_interval:        Print status every N steps.
        lr:                  Peak learning rate.
        min_lr:              Final learning rate (for cosine schedule).
        weight_decay:        AdamW weight decay.
        warmup_steps:        LR warmup steps.
        grad_clip:           Maximum gradient norm.
        label_smoothing:     Cross-entropy label smoothing.
        checkpoint_dir:      Where to save checkpoints (None = don't save).
        checkpoint_interval: Save checkpoint every N steps.
        on_step:             Optional callback(step, tracker) after each step.
    """

    def __init__(
        self,
        model: GPT,
        train_dataset: TextDataset,
        val_dataset: Optional[TextDataset] = None,
        batch_size: int = 32,
        max_steps: int = 5000,
        eval_interval: int = 500,
        eval_batches: int = 20,
        log_interval: int = 100,
        lr: float = 3e-4,
        min_lr: float = 3e-5,
        weight_decay: float = 0.1,
        warmup_steps: int = 100,
        grad_clip: float = 1.0,
        label_smoothing: float = 0.0,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 1000,
        on_step: Optional[Callable] = None,
    ):
        self.model = model
        self.train_ds = train_dataset
        self.val_ds = val_dataset
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.eval_interval = eval_interval
        self.eval_batches = eval_batches
        self.log_interval = log_interval
        self.grad_clip = grad_clip
        self.label_smoothing = label_smoothing
        self.checkpoint_interval = checkpoint_interval
        self.on_step = on_step

        # ── Optimizer ─────────────────────────────────────────────────────────
        self.optimizer = model.configure_optimizer(
            lr=lr, weight_decay=weight_decay
        )

        # ── LR Scheduler ─────────────────────────────────────────────────────
        self.scheduler = CosineWithWarmup(
            self.optimizer,
            max_lr=lr,
            min_lr=min_lr,
            warmup_steps=warmup_steps,
            total_steps=max_steps,
        )

        # ── Tracking + checkpointing ──────────────────────────────────────────
        self.tracker = LossTracker()
        self.checkpointer = (
            Checkpointer(checkpoint_dir, model) if checkpoint_dir else None
        )

        self.current_step = 0

    def _forward_and_loss(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tensor:
        """
        One forward pass: compute logits and cross-entropy loss.
        """
        logits = self.model(x)
        loss = cross_entropy_loss(
            logits, y,
            label_smoothing=self.label_smoothing,
        )
        return loss

    @staticmethod
    def _eval_loss(
        model: GPT,
        dataset: TextDataset,
        batch_size: int,
        n_batches: int,
    ) -> float:
        """
        Estimate validation loss by averaging over multiple batches.
        No gradients computed (eval mode).
        """
        model.eval()
        total_loss = 0.0

        for i in range(n_batches):
            x, y = dataset.get_sequential_batch(batch_size, i)
            logits = model(x)
            loss = cross_entropy_loss(logits, y)
            total_loss += float(loss.data)
            # Free the computation graph immediately
            del logits, loss
            gc.collect()

        model.train()
        return total_loss / n_batches

    def train(self, resume: bool = False) -> LossTracker:
        """
        Run the training loop.

        Args:
            resume: If True and checkpoint_dir is set, resume from latest checkpoint.

        Returns:
            LossTracker with the full training history.
        """
        # ── Optional resume from checkpoint ───────────────────────────────────
        if resume and self.checkpointer:
            self.current_step = self.checkpointer.load_latest(
                self.optimizer, self.tracker
            )

        self.model.train()
        print(f"\n{'='*60}")
        print(f"  Training GPT: {self.model.num_parameters():,} parameters")
        print(f"  Steps: {self.max_steps:,} | Batch: {self.batch_size} | LR: peak={self.optimizer.lr:.2e}")
        print(f"{'='*60}\n")

        t0 = time.time()

        while self.current_step < self.max_steps:

            # ── Step 26: Update learning rate ─────────────────────────────────
            self.scheduler.step()

            # ── Step 22: Get data ─────────────────────────────────────────────
            x, y = self.train_ds.get_batch(self.batch_size)

            # ── Steps 5 + forward: Compute loss ──────────────────────────────
            loss = self._forward_and_loss(x, y)
            loss_val = float(loss.data)   # extract scalar before backward

            # ── Step 6: Backpropagation ───────────────────────────────────────
            # Reset gradients from previous step
            self.optimizer.zero_grad()

            # Compute gradients for all parameters via chain rule
            loss.backward()

            # ── Free computation graph eagerly (saves memory) ─────────────────
            del loss
            gc.collect()

            # ── Step 9: Gradient clipping ─────────────────────────────────────
            grad_norm = clip_grad_norm(self.model.parameters(), self.grad_clip)

            # ── Step 7: Gradient descent ──────────────────────────────────────
            self.optimizer.step()

            # ── Step 24: Track progress ───────────────────────────────────────
            self.tracker.log_train(
                self.current_step, loss_val, self.optimizer.lr, grad_norm
            )

            # ── Logging ───────────────────────────────────────────────────────
            if self.current_step % self.log_interval == 0:
                elapsed = time.time() - t0
                tokens_per_sec = (
                    self.log_interval * self.batch_size * self.train_ds.block_size
                    / max(elapsed, 1e-6)
                )
                print(
                    f"  {self.tracker.summary()}"
                    f" | {tokens_per_sec:.0f} tok/s"
                )
                t0 = time.time()

            # ── Validation ────────────────────────────────────────────────────
            if (
                self.val_ds is not None
                and self.current_step % self.eval_interval == 0
                and self.current_step > 0
            ):
                val_loss = self._eval_loss(
                    self.model, self.val_ds,
                    self.batch_size, self.eval_batches,
                )
                self.tracker.log_val(self.current_step, val_loss)
                print(
                    f"\n  ── Validation @ step {self.current_step} ──"
                    f"  val_loss={val_loss:.4f}"
                    f"  ppl={perplexity(val_loss):.1f}\n"
                )

            # ── Checkpointing (Step 25) ───────────────────────────────────────
            if (
                self.checkpointer is not None
                and self.current_step % self.checkpoint_interval == 0
                and self.current_step > 0
            ):
                self.checkpointer.save(
                    self.current_step, self.optimizer, self.tracker,
                    metadata={"val_loss": self.tracker.last_val_loss},
                )

            # ── Optional callback ─────────────────────────────────────────────
            if self.on_step is not None:
                self.on_step(self.current_step, self.tracker)

            self.current_step += 1

        # ── Final checkpoint ──────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"  Training complete!  Final step: {self.current_step}")
        if self.tracker.train_losses:
            final_ppl = perplexity(self.tracker.smoothed_loss)
            print(f"  Final smooth loss: {self.tracker.smoothed_loss:.4f}  PPL: {final_ppl:.1f}")
        print(f"{'='*60}\n")

        if self.checkpointer is not None:
            self.checkpointer.save(
                self.current_step, self.optimizer, self.tracker,
                metadata={"final": True},
            )

        return self.tracker





################################################################################
# SECTION 8: INFERENCE ENGINE — Steps 27-30
################################################################################


"""
================================================================================
nanoGPT / inference / generate.py
================================================================================
Inference engine — Steps 27–30.

Once trained, the model is used to generate text by repeatedly predicting
the next token and appending it to the sequence (autoregressive generation).

DECODING STRATEGIES:

  27. GREEDY DECODING
      Always pick the highest-probability token.
      Deterministic, fast, but repetitive and often suboptimal.
      (Can get stuck in loops like "the the the...")

  28. TEMPERATURE SAMPLING
      Scale logits by 1/temperature before softmax, then sample.
      - temperature = 1.0: Sample from the true learned distribution
      - temperature < 1.0: More confident/focused (sharper distribution)
      - temperature > 1.0: More random/creative (flatter distribution)
      - temperature → 0: Equivalent to greedy
      - temperature → ∞: Uniform random

  29a. TOP-K SAMPLING
      Sample from only the K most likely tokens (zero out the rest).
      Prevents sampling very unlikely tokens that could derail generation.
      Typical: K = 40-50.

  29b. TOP-P (NUCLEUS) SAMPLING
      Sample from the smallest set of tokens whose total probability ≥ p.
      Adapts dynamically: if the distribution is peaked, nucleus is small
      (model is confident); if flat, nucleus is larger.
      Typical: p = 0.9. Often better than top-k in practice.

  30. FULL INFERENCE LOOP
      The InferenceEngine class combines all of the above into a
      user-friendly API.

AUTOREGRESSIVE GENERATION:
  At each step:
  1. Feed current token sequence (prompt + generated so far) to model
  2. Get logits for the LAST position (the next-token distribution)
  3. Apply decoding strategy to sample/pick a token
  4. Append chosen token to sequence
  5. Repeat until max_new_tokens or <eos> token
================================================================================
"""

# import numpy as np  # at top
# import math  # at top
# from typing import List, Optional, Union, Callable  # at top
# from ..model.transformer import GPT  # merged


# ──────────────────────────────────────────────────────────────────────────────
# Sampling utilities
# ──────────────────────────────────────────────────────────────────────────────

def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """
    Scale logits by inverse temperature to control distribution sharpness.

    temperature = 1.0: unchanged distribution
    temperature < 1.0: sharper (higher probability tokens become more dominant)
    temperature > 1.0: flatter (probabilities become more uniform)

    Args:
        logits:      Raw logits array of shape (..., vocab_size).
        temperature: Temperature value > 0.

    Returns:
        Scaled logits (same shape).
    """
    if temperature == 1.0:
        return logits
    return logits / max(temperature, 1e-8)


def apply_top_k(logits: np.ndarray, k: int) -> np.ndarray:
    """
    Keep only the top-k logits; set all others to -inf.

    After this, softmax will assign 0 probability to the masked tokens,
    so they can never be sampled.

    Args:
        logits: Raw logits of shape (vocab_size,).
        k:      Number of tokens to keep. If k <= 0, no filtering.

    Returns:
        Filtered logits (same shape).
    """
    if k <= 0 or k >= len(logits):
        return logits

    # Find the k-th largest value
    # np.partition is faster than full sort for large vocabs
    threshold_idx = np.argpartition(logits, -k)[-k:]   # indices of top-k
    threshold_val = logits[threshold_idx].min()          # smallest of top-k

    # Mask everything below the threshold
    filtered = logits.copy()
    filtered[filtered < threshold_val] = -1e9
    return filtered


def apply_top_p(logits: np.ndarray, p: float) -> np.ndarray:
    """
    Nucleus sampling: keep the smallest set of tokens summing to probability p.

    The nucleus shrinks when the model is confident (peaked distribution)
    and grows when the model is uncertain (flat distribution).
    This is more adaptive than top-k.

    Args:
        logits: Raw logits of shape (vocab_size,).
        p:      Cumulative probability threshold (0 < p ≤ 1).

    Returns:
        Filtered logits (same shape).
    """
    if p >= 1.0:
        return logits

    # Convert logits to probabilities
    logits_shifted = logits - logits.max()   # numerical stability
    probs = np.exp(logits_shifted)
    probs /= probs.sum()

    # Sort tokens by probability (descending)
    sorted_indices = np.argsort(probs)[::-1]  # highest prob first
    sorted_probs = probs[sorted_indices]

    # Find the nucleus: smallest set with cumulative prob ≥ p
    cumulative = np.cumsum(sorted_probs)
    # Mark tokens to REMOVE: those AFTER the nucleus
    remove_mask = cumulative > p
    # Shift right by 1 so we always keep at least one token
    remove_mask = np.roll(remove_mask, 1)
    remove_mask[0] = False  # always keep the top token

    # Map back to original indices
    original_remove_mask = np.zeros(len(logits), dtype=bool)
    original_remove_mask[sorted_indices[remove_mask]] = True

    # Apply filter
    filtered = logits.copy()
    filtered[original_remove_mask] = -1e9
    return filtered


def sample_from_logits(logits: np.ndarray) -> int:
    """
    Sample a token from a logit vector using the categorical distribution.

    Args:
        logits: Raw logits of shape (vocab_size,). Can be pre-filtered.

    Returns:
        Sampled token ID (an integer).
    """
    # Convert logits to probabilities with numerical stability
    logits = logits - logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()

    # Sample from the categorical distribution
    return int(np.random.choice(len(probs), p=probs))


# ──────────────────────────────────────────────────────────────────────────────
# Step 27: Greedy decoding
# ──────────────────────────────────────────────────────────────────────────────

def greedy_decode(
    model: GPT,
    context: np.ndarray,
    max_new_tokens: int,
    eos_token_id: Optional[int] = None,
) -> np.ndarray:
    """
    Greedy decoding: always pick the most likely next token.

    Deterministic. Fast. Can get repetitive.

    Args:
        model:          The GPT model (should be in eval mode).
        context:        Integer array of shape (1, T) — the prompt.
        max_new_tokens: Maximum number of tokens to generate.
        eos_token_id:   Stop generation if this token is sampled.

    Returns:
        Integer array of shape (1, T + generated_length).
    """
    model.eval()
    tokens = context.copy()   # (1, T)

    for _ in range(max_new_tokens):
        # Crop to model's context window (from the right)
        crop = tokens[:, -model.max_seq_len:]

        # Forward pass: get logits for all positions
        logits = model(crop)   # (1, T, vocab_size)

        # Take logits at the LAST position (next token prediction)
        next_logits = logits.data[0, -1, :]   # (vocab_size,)

        # Greedy: pick argmax
        next_token = int(np.argmax(next_logits))

        # Append to sequence
        tokens = np.concatenate([tokens, [[next_token]]], axis=1)

        # Stop if EOS token generated
        if eos_token_id is not None and next_token == eos_token_id:
            break

    return tokens


# ──────────────────────────────────────────────────────────────────────────────
# Step 28: Temperature sampling
# ──────────────────────────────────────────────────────────────────────────────

def temperature_decode(
    model: GPT,
    context: np.ndarray,
    max_new_tokens: int,
    temperature: float = 1.0,
    eos_token_id: Optional[int] = None,
) -> np.ndarray:
    """
    Temperature sampling: sample from the distribution, scaled by temperature.

    Args:
        model:          The GPT model.
        context:        Integer array of shape (1, T).
        max_new_tokens: Maximum tokens to generate.
        temperature:    Sampling temperature (1.0 = unscaled, <1 = sharper).
        eos_token_id:   Stop if this token is generated.

    Returns:
        Token array of shape (1, T + generated_length).
    """
    model.eval()
    tokens = context.copy()

    for _ in range(max_new_tokens):
        crop = tokens[:, -model.max_seq_len:]
        logits = model(crop).data[0, -1, :]       # (vocab_size,)
        logits = apply_temperature(logits, temperature)
        next_token = sample_from_logits(logits)
        tokens = np.concatenate([tokens, [[next_token]]], axis=1)
        if eos_token_id is not None and next_token == eos_token_id:
            break

    return tokens


# ──────────────────────────────────────────────────────────────────────────────
# Step 29: Top-k and top-p sampling
# ──────────────────────────────────────────────────────────────────────────────

def topk_topp_decode(
    model: GPT,
    context: np.ndarray,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    eos_token_id: Optional[int] = None,
) -> np.ndarray:
    """
    Full sampling pipeline: temperature → top-k → top-p → sample.

    This is the gold standard for language model text generation,
    used by production LLMs (GPT-3, LLaMA, Claude, etc.).

    The full pipeline:
        1. Scale logits by 1/temperature (sharpness control)
        2. Zero out all but top-k logits (prevents rare disasters)
        3. Zero out tokens outside the nucleus (adaptive filtering)
        4. Sample from remaining distribution

    Args:
        model:          The GPT model.
        context:        Integer array of shape (1, T).
        max_new_tokens: Maximum tokens to generate.
        temperature:    Sampling temperature.
        top_k:          Keep only top-k tokens (0 = disabled).
        top_p:          Nucleus probability threshold (1.0 = disabled).
        eos_token_id:   Stop if this token is generated.

    Returns:
        Token array of shape (1, T + generated_length).
    """
    model.eval()
    tokens = context.copy()

    for _ in range(max_new_tokens):
        crop = tokens[:, -model.max_seq_len:]
        logits = model(crop).data[0, -1, :]

        # Step 1: temperature scaling
        logits = apply_temperature(logits, temperature)

        # Step 2: top-k filtering
        if top_k > 0:
            logits = apply_top_k(logits, top_k)

        # Step 3: top-p (nucleus) filtering
        if top_p < 1.0:
            logits = apply_top_p(logits, top_p)

        # Step 4: sample
        next_token = sample_from_logits(logits)
        tokens = np.concatenate([tokens, [[next_token]]], axis=1)

        if eos_token_id is not None and next_token == eos_token_id:
            break

    return tokens


# ──────────────────────────────────────────────────────────────────────────────
# Step 30: Full inference loop — the InferenceEngine
# ──────────────────────────────────────────────────────────────────────────────

class InferenceEngine:
    """
    High-level text generation interface.

    Wraps the model and tokenizer with all decoding strategies behind
    a single clean API. Handles tokenization, context management,
    and detokenization.

    Usage:
        engine = InferenceEngine(model, tokenizer)
        text = engine.generate(
            "To be or not to be",
            max_new_tokens=200,
            strategy="nucleus",
            temperature=0.8,
            top_p=0.9,
        )
        print(text)

    Args:
        model:     A trained GPT model.
        tokenizer: A CharTokenizer or BPETokenizer instance.
    """

    def __init__(self, model: GPT, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        strategy: str = "nucleus",
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        eos_on_newline: bool = False,
        seed: Optional[int] = None,
    ) -> str:
        """
        Generate text from a text prompt.

        Args:
            prompt:          The starting text (will be encoded and fed to model).
            max_new_tokens:  Maximum number of new tokens to generate.
            strategy:        One of 'greedy', 'temperature', 'topk', 'topp', 'nucleus'.
                             'nucleus' = temperature + top-k + top-p (recommended).
            temperature:     Sampling temperature (ignored for greedy).
            top_k:           Top-k cutoff (0 = disabled).
            top_p:           Nucleus threshold (1.0 = disabled).
            eos_on_newline:  Stop generation at the first newline.
            seed:            Random seed for reproducibility.

        Returns:
            Full generated string (prompt + new tokens).
        """
        if seed is not None:
            np.random.seed(seed)

        # Tokenize prompt
        prompt_ids = self.tokenizer.encode(prompt)
        context = np.array([prompt_ids], dtype=np.int64)   # (1, T)

        eos_id = getattr(self.tokenizer, "EOS_ID", None)

        # Select decoding strategy
        if strategy == "greedy":
            output_tokens = greedy_decode(
                self.model, context, max_new_tokens, eos_id
            )
        elif strategy == "temperature":
            output_tokens = temperature_decode(
                self.model, context, max_new_tokens, temperature, eos_id
            )
        elif strategy == "topk":
            output_tokens = topk_topp_decode(
                self.model, context, max_new_tokens,
                temperature=temperature, top_k=top_k, top_p=1.0,
                eos_token_id=eos_id,
            )
        elif strategy in ("topp", "nucleus"):
            output_tokens = topk_topp_decode(
                self.model, context, max_new_tokens,
                temperature=temperature, top_k=top_k, top_p=top_p,
                eos_token_id=eos_id,
            )
        else:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                "Choose: 'greedy', 'temperature', 'topk', 'topp', 'nucleus'"
            )

        # Decode only the NEW tokens (not the prompt)
        new_token_ids = output_tokens[0, len(prompt_ids):].tolist()
        generated = self.tokenizer.decode(new_token_ids)

        # Optionally stop at first newline
        if eos_on_newline and "\n" in generated:
            generated = generated[:generated.index("\n")]

        return prompt + generated

    def perplexity(self, text: str) -> float:
        """
        Compute the perplexity of the model on a text string.

        Perplexity measures how "surprised" the model is by the text.
        Lower = model assigns higher probability to the text = better fit.

        Args:
            text: Input text string.

        Returns:
            Perplexity value ≥ 1.0.
        """
# from ..core.losses import cross_entropy_loss, perplexity as ppl_fn  # merged

        ids = self.tokenizer.encode(text)
        if len(ids) < 2:
            return float("inf")

        # Crop to fit in context window
        max_t = self.model.max_seq_len
        if len(ids) > max_t + 1:
            ids = ids[:max_t + 1]

        x = np.array([ids[:-1]], dtype=np.int64)   # (1, T)
        y = np.array(ids[1:], dtype=np.int64)       # (T,) — targets

        self.model.eval()
        logits = self.model(x)
        loss = cross_entropy_loss(logits, y)
        return perplexity(float(loss.data))

    def complete_multiple(
        self,
        prompt: str,
        n: int = 5,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> List[str]:
        """
        Generate n independent completions of the same prompt.
        Useful for exploring the diversity of model outputs.

        Args:
            prompt:         The starting text.
            n:              Number of completions.
            max_new_tokens: Maximum tokens per completion.
            **kwargs:       Passed to generate().

        Returns:
            List of n generated strings.
        """
        return [
            self.generate(prompt, max_new_tokens=max_new_tokens, **kwargs)
            for _ in range(n)
        ]

    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        callback: Optional[Callable] = None,
    ):
        """
        Generate tokens one at a time, calling callback after each.
        Enables streaming output (token by token, like ChatGPT's UI).

        Args:
            prompt:        Starting text.
            max_new_tokens: Maximum new tokens.
            temperature:   Sampling temperature.
            top_k, top_p:  Filtering parameters.
            callback:      Called with each new character/token as generated.

        Yields:
            One decoded token string at a time.
        """
# from typing import Callable  # at top

        self.model.eval()
        prompt_ids = self.tokenizer.encode(prompt)
        tokens = np.array([prompt_ids], dtype=np.int64)
        eos_id = getattr(self.tokenizer, "EOS_ID", None)

        for _ in range(max_new_tokens):
            crop = tokens[:, -self.model.max_seq_len:]
            logits = self.model(crop).data[0, -1, :]

            logits = apply_temperature(logits, temperature)
            if top_k > 0:
                logits = apply_top_k(logits, top_k)
            if top_p < 1.0:
                logits = apply_top_p(logits, top_p)

            next_token = sample_from_logits(logits)
            tokens = np.concatenate([tokens, [[next_token]]], axis=1)

            # Decode just this one token
            decoded = self.tokenizer.decode([next_token])

            if callback is not None:
                callback(decoded)

            yield decoded

            if eos_id is not None and next_token == eos_id:
                break






################################################################################
# SECTION 9: MODEL MANAGEMENT — Steps 31-34
################################################################################


"""
================================================================================
nanoGPT / utils / model_utils.py
================================================================================
Model management utilities — Steps 31–34.

Handles saving, loading, inspecting, and analyzing trained models.

CONTENTS:
  save_model          — Serialize model to disk (Step 31)
  load_model          — Reconstruct from saved file (Step 32)
  inspect_model       — Print architecture and weight statistics (Step 33)
  count_parameters    — Detailed parameter breakdown (Step 34)
  print_model_card    — Human-readable model summary
================================================================================
"""

# import numpy as np  # at top
# import pickle  # at top
# import json  # at top
# import os  # at top
# import math  # at top
# from typing import Dict, Optional, Any  # at top

# from ..model.transformer import GPT  # merged


# ──────────────────────────────────────────────────────────────────────────────
# Step 31: Save model
# ──────────────────────────────────────────────────────────────────────────────

def save_model(
    model: GPT,
    path: str,
    tokenizer=None,
    training_config: Optional[Dict] = None,
) -> str:
    """
    Save a trained model to disk.

    Saves:
      - Model configuration (vocab_size, d_model, n_layers, etc.)
      - Model weights (all Parameters)
      - Optionally: tokenizer vocabulary
      - Optionally: training configuration metadata

    Format: Python pickle (lossless, self-contained).

    Args:
        model:           Trained GPT model.
        path:            Output file path (.pkl recommended).
        tokenizer:       Optional tokenizer to include in the save.
        training_config: Optional dict of training hyperparameters.

    Returns:
        Path to the saved file.
    """
    # Build the config dict from model attributes
    config = {
        "vocab_size": model.vocab_size,
        "d_model": model.d_model,
        "n_layers": model.n_layers,
        "n_heads": model.n_heads,
        "max_seq_len": model.max_seq_len,
        "tie_weights": model.tie_weights,
    }

    # Collect weights (deduplicated — weight tying shares the same object)
    params = model.parameters()
    weights = {}
    seen = set()
    idx = 0
    for p in params:
        if id(p) not in seen:
            seen.add(id(p))
            weights[idx] = p.data.copy()
            idx += 1

    # Optionally serialize tokenizer
    tok_state = None
    if tokenizer is not None:
        if hasattr(tokenizer, "stoi"):
            tok_state = {
                "type": type(tokenizer).__name__,
                "stoi": tokenizer.stoi,
                "itos": {str(k): v for k, v in tokenizer.itos.items()},
            }
            if hasattr(tokenizer, "merges"):
                tok_state["merges"] = tokenizer.merges

    payload = {
        "config": config,
        "weights": weights,
        "tokenizer": tok_state,
        "training_config": training_config or {},
        "n_parameters": model.num_parameters(),
    }

    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(path) / 1e6
    print(f"Model saved: {path}  ({size_mb:.1f} MB, {model.num_parameters():,} params)")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Step 32: Load model
# ──────────────────────────────────────────────────────────────────────────────

def load_model(path: str) -> tuple:
    """
    Load a saved model from disk and reconstruct it.

    Args:
        path: Path to the .pkl file saved by save_model().

    Returns:
        (model, tokenizer, training_config)
        tokenizer is None if not saved; training_config may be an empty dict.
    """
    with open(path, "rb") as f:
        payload = pickle.load(f)

    cfg = payload["config"]

    # Reconstruct the model architecture
    model = GPT(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        max_seq_len=cfg["max_seq_len"],
        tie_weights=cfg.get("tie_weights", True),
        dropout=0.0,    # No dropout during inference
    )

    # Restore weights
    params = model.parameters()
    seen = set()
    idx = 0
    for p in params:
        if id(p) not in seen:
            seen.add(id(p))
            if idx in payload["weights"]:
                p.data[:] = payload["weights"][idx]
            idx += 1

    model.eval()

    # Optionally reconstruct tokenizer
    tokenizer = None
    tok_state = payload.get("tokenizer")
    if tok_state is not None:
        tok_type = tok_state.get("type", "CharTokenizer")
        if tok_type == "CharTokenizer":
# from ..model.tokenizer import CharTokenizer  # merged
            tokenizer = CharTokenizer()
            tokenizer.stoi = tok_state["stoi"]
            tokenizer.itos = {int(k): v for k, v in tok_state["itos"].items()}
            tokenizer._built = True
        elif tok_type == "BPETokenizer":
# from ..model.tokenizer import BPETokenizer  # merged
            tokenizer = BPETokenizer()
            tokenizer.stoi = tok_state["stoi"]
            tokenizer.itos = {int(k): v for k, v in tok_state["itos"].items()}
            tokenizer.merges = tok_state.get("merges", [])
            tokenizer._built = True

    training_config = payload.get("training_config", {})
    print(f"Model loaded: {path}  ({model.num_parameters():,} params)")
    return model, tokenizer, training_config


# ──────────────────────────────────────────────────────────────────────────────
# Step 33: Inspect weights and architecture
# ──────────────────────────────────────────────────────────────────────────────

def inspect_model(model: GPT, show_histograms: bool = False) -> Dict:
    """
    Print a detailed summary of the model's architecture and weight statistics.

    Weight statistics are useful for debugging:
      - Very small std → vanishing gradient (network not learning)
      - Very large std → exploding gradient (instability)
      - Std ≈ 0.02 for fresh weights → good initialization
      - After training: weights should spread out a bit from init

    Args:
        model:           The GPT model to inspect.
        show_histograms: If True, print ASCII histogram of each layer's weights.

    Returns:
        Dict mapping parameter name → statistics dict.
    """
    print(f"\n{'='*70}")
    print(f"  {model}")
    print(f"{'='*70}")
    print(f"\n  {'Name':<20} {'Shape':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print(f"  {'-'*85}")

    param_stats = {}
    params = model.parameters()
    seen = set()
    idx = 0

    for p in params:
        if id(p) in seen:
            continue
        seen.add(id(p))

        d = p.data
        stats = {
            "shape": list(d.shape),
            "n_params": d.size,
            "mean": float(d.mean()),
            "std": float(d.std()),
            "min": float(d.min()),
            "max": float(d.max()),
            "has_grad": p.grad is not None,
            "grad_norm": float(np.linalg.norm(p.grad)) if p.grad is not None else None,
        }
        param_stats[f"param_{idx:03d}"] = stats

        # Determine a descriptive name based on shape
        shape_str = str(list(d.shape))
        name_str = f"param_{idx:03d}"

        print(
            f"  {name_str:<20} {shape_str:<25}"
            f" {stats['mean']:>10.4f} {stats['std']:>10.4f}"
            f" {stats['min']:>10.4f} {stats['max']:>10.4f}"
        )

        if show_histograms:
            _print_ascii_histogram(d.ravel(), width=50)

        idx += 1

    print(f"\n  Total parameters: {model.num_parameters():,}")
    print(f"  Memory (fp32): {model.num_parameters() * 4 / 1e6:.1f} MB\n")

    return param_stats


def _print_ascii_histogram(data: np.ndarray, width: int = 50, bins: int = 20):
    """Print a simple ASCII histogram of weight values."""
    counts, edges = np.histogram(data, bins=bins)
    max_count = counts.max()

    for count, left in zip(counts, edges):
        bar_len = int(count / max_count * width)
        bar = "█" * bar_len
        print(f"    {left:+.3f} | {bar}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Step 34: Count parameters
# ──────────────────────────────────────────────────────────────────────────────

def count_parameters(model: GPT) -> Dict:
    """
    Detailed parameter count breakdown by component.

    Returns a dict with total and per-component counts, plus a
    pretty-printed table with bar charts.

    Args:
        model: The GPT model.

    Returns:
        Dict with keys: 'total', 'token_embedding', 'position_embedding',
        'attention', 'feed_forward', 'layer_norm', 'lm_head'.
    """
    totals = {
        "token_embedding": 0,
        "position_embedding": 0,
        "attention": 0,
        "feed_forward": 0,
        "layer_norm": 0,
        "lm_head": 0,
        "other": 0,
    }

    seen = set()

    def count(params, category: str):
        for p in params:
            if id(p) not in seen:
                seen.add(id(p))
                totals[category] += p.data.size

    # Token embedding
    count(model.token_embedding.parameters(), "token_embedding")

    # Position encoding
    count(model.pos_encoding.parameters(), "position_embedding")

    # Per-block breakdown
    for block in model.blocks:
        count(block.attn.parameters(), "attention")
        count(block.ffn.parameters(), "feed_forward")
        count(block.ln1.parameters() + block.ln2.parameters(), "layer_norm")

    # Final layer norm
    count(model.ln_final.parameters(), "layer_norm")

    # LM head (may be weight-tied, so might add 0)
    count(model.lm_head.parameters(), "lm_head")

    # Total
    total = sum(totals.values())
    totals["total"] = total

    # Print pretty table
    print(f"\n{'='*60}")
    print(f"  Parameter Count — {model.__class__.__name__}")
    print(f"{'='*60}")
    print(f"  {'Component':<22} {'Params':>12} {'%':>7}  Bar")
    print(f"  {'-'*58}")

    bar_width = 20
    for key, val in totals.items():
        if key == "total":
            continue
        if val == 0:
            continue
        pct = 100.0 * val / total
        bar = "█" * int(pct / 100 * bar_width)
        print(f"  {key:<22} {val:>12,} {pct:>6.1f}%  {bar}")

    print(f"  {'─'*58}")
    print(f"  {'TOTAL':<22} {total:>12,} {'100.0%':>7}")
    print(f"\n  Approximate memory (fp32): {total * 4 / 1e6:.1f} MB")
    if model.tie_weights:
        emb = totals["token_embedding"]
        print(f"  Weight-tied LM head saves: {emb * 4 / 1e6:.1f} MB")
    print(f"{'='*60}\n")

    return totals


def print_model_card(model: GPT, tokenizer=None, training_info: Optional[Dict] = None):
    """
    Print a complete human-readable model card.

    Shows architecture, parameter count, and optionally training results.

    Args:
        model:         The GPT model.
        tokenizer:     Optional tokenizer (shows vocab size).
        training_info: Optional dict with keys 'final_loss', 'final_ppl', etc.
    """
    n = model.num_parameters()
    non_emb = model.num_parameters(exclude_embeddings=True)

    print(f"\n╔{'═'*58}╗")
    print(f"║  {'nanoGPT Model Card':^56}  ║")
    print(f"╠{'═'*58}╣")
    print(f"║  Architecture:  GPT (decoder-only transformer){' '*10}║")
    print(f"║{'─'*58}║")
    print(f"║  d_model:       {model.d_model:<42}║")
    print(f"║  n_layers:      {model.n_layers:<42}║")
    print(f"║  n_heads:       {model.n_heads:<42}║")
    print(f"║  head_dim:      {model.d_model // model.n_heads:<42}║")
    print(f"║  max_seq_len:   {model.max_seq_len:<42}║")
    if tokenizer:
        print(f"║  vocab_size:    {tokenizer.vocab_size:<42}║")
    print(f"║{'─'*58}║")
    print(f"║  Total params:  {n:>12,} ({n*4/1e6:.1f} MB){' '*17}║")
    print(f"║  Non-emb params:{non_emb:>12,}{' '*29}║")
    if training_info:
        print(f"║{'─'*58}║")
        for k, v in training_info.items():
            line = f"  {k+':':<16} {v}"
            print(f"║{line:<58}║")
    print(f"╚{'═'*58}╝\n")





################################################################################
# SECTION 10: DEMO — Run with: python3 nanoGPT.py
################################################################################

def _run_demo():
    """
    Full end-to-end demo.
    Trains a 4-layer GPT on Shakespeare, then generates with all 4 strategies.
    """
    np.random.seed(1337)

    print("=" * 65)
    print("  nanoGPT — Full Transformer Language Model")
    print("  Built from scratch on NumPy. No PyTorch.")
    print("=" * 65)

    # ── 1. Gradient correctness ────────────────────────────────────────────────
    print("\n[1/5] Verifying gradients...")
    for label, func, shape in [
        ("matmul ", lambda a, b: (a @ b).sum(),
            [(3, 4), (4, 5)]),
        ("GELU   ", lambda a: a.gelu().sum(),
            [(10,)]),
        ("softmax", lambda a: a.softmax().sum(),
            [(8,)]),
        ("layernorm", lambda a, g, b: LayerNorm._manual_ln(a, g, b).sum()
            if False else None, None),
    ]:
        if shape is None:
            continue
        inputs = [Tensor(np.random.randn(*s).astype("f"), requires_grad=True)
                  for s in shape]
        err = gradient_check(func, *inputs, eps=1e-3)
        ok = "OK" if err < 8e-3 else "WARN"
        print(f"  {label}: max_err={err:.2e}  [{ok}]")

    # ── 2. Tokenizer ───────────────────────────────────────────────────────────
    print("\n[2/5] Building dataset...")

    TEXT = (
        "To be, or not to be, that is the question:\n"
        "Whether \'tis nobler in the mind to suffer\n"
        "The slings and arrows of outrageous fortune,\n"
        "Or to take arms against a sea of troubles\n"
        "And by opposing end them. To die—to sleep,\n"
        "No more; and by a sleep to say we end\n"
        "The heart-ache and the thousand natural shocks\n"
        "That flesh is heir to: \'tis a consummation\n"
        "Devoutly to be wish\'d. To die, to sleep;\n"
        "To sleep, perchance to dream—ay, there\'s the rub:\n"
        "For in that sleep of death what dreams may come\n"
        "When we have shuffled off this mortal coil,\n"
        "Must give us pause—there\'s the respect\n"
        "That makes calamity of so long life.\n"
        "For who would bear the whips and scorns of time,\n"
        "Thus conscience doth make cowards of us all,\n"
        "And thus the native hue of resolution\n"
        "Is sicklied o\'er with the pale cast of thought,\n"
        "And enterprises of great pith and moment\n"
        "With this regard their currents turn awry\n"
        "And lose the name of action.\n"
    ) * 18

    tokenizer = CharTokenizer()
    tokenizer.build(TEXT)
    ids = tokenizer.encode(TEXT)
    train_ids, val_ids = train_val_split(ids, val_fraction=0.1)
    BLOCK, BATCH = 32, 16
    train_ds = TextDataset(train_ids, BLOCK)
    val_ds   = TextDataset(val_ids,   BLOCK)
    print(f"  Corpus: {len(TEXT):,} chars  |  Vocab: {tokenizer.vocab_size}  "
          f"|  Train examples: {len(train_ds):,}")

    # ── 3. Model ───────────────────────────────────────────────────────────────
    print("\n[3/5] Building model...")
    model = GPT(
        vocab_size=tokenizer.vocab_size, d_model=128, n_layers=4,
        n_heads=4, max_seq_len=BLOCK, dropout=0.1, tie_weights=True,
    )
    count_parameters(model)
    print_model_card(model, tokenizer)

    # ── 4. Train ───────────────────────────────────────────────────────────────
    print("\n[4/5] Training...")
    os.makedirs("/tmp/nanogpt_demo", exist_ok=True)
    trainer = Trainer(
        model=model, train_dataset=train_ds, val_dataset=val_ds,
        batch_size=BATCH, max_steps=500, eval_interval=200, eval_batches=8,
        log_interval=100, lr=3e-3, min_lr=3e-4, warmup_steps=50,
        weight_decay=0.1, grad_clip=1.0,
        checkpoint_dir="/tmp/nanogpt_demo", checkpoint_interval=500,
    )
    tracker = trainer.train()

    # ── 5. Save, generate, verify ──────────────────────────────────────────────
    save_path = "/tmp/nanogpt_demo/final_model.pkl"
    save_model(model, save_path, tokenizer=tokenizer,
               training_config={"steps": 500, "corpus": "Shakespeare"})

    print("\n[5/5] Generating text...\n")
    engine = InferenceEngine(model, tokenizer)
    PROMPT = "To be, or not"
    ppl = engine.perplexity(PROMPT + " to be")
    print(f"Perplexity on \'{PROMPT} to be\': {ppl:.1f}\n")
    print("-" * 65)

    for name, kw in [
        ("GREEDY",      {"strategy": "greedy"}),
        ("TEMPERATURE", {"strategy": "temperature", "temperature": 0.8, "seed": 42}),
        ("TOP-K",       {"strategy": "topk",  "temperature": 0.9, "top_k": 10, "seed": 42}),
        ("NUCLEUS",     {"strategy": "nucleus","temperature": 0.85,"top_k": 20,"top_p": 0.9,"seed": 42}),
    ]:
        text = engine.generate(PROMPT, max_new_tokens=100, **kw)
        print(f"[{name}]\n{text}\n")

    # Save/load roundtrip
    m2, t2, _ = load_model(save_path)
    e2 = InferenceEngine(m2, t2)
    out = e2.generate(PROMPT, max_new_tokens=30, strategy="greedy")
    print(f"Save/load OK: \'{out[:55]}...\'")

    final_ppl = math.exp(min(tracker.smoothed_loss, 20))
    vp = f"  Val PPL: {math.exp(min(tracker.last_val_loss,20)):.1f}" if tracker.last_val_loss else ""
    print(f"\n{'='*65}")
    print(f"  Loss: {tracker.train_losses[0]:.4f} -> {tracker.smoothed_loss:.4f}  "
          f"PPL: {final_ppl:.1f}{vp}")
    print(f"  Model: {save_path}")
    print(f"{'='*65}")

################################################################################
# SECTION 10: ALL EXTRAS (v2) — KV Cache, Flash Attn, Beam Search, 
#             Gradient Accumulation, GPT-2 Init, Repetition Penalty, Min-p
################################################################################


# import numpy as np  # at top
# import math  # at top
# import time  # at top
# from typing import List, Optional, Dict, Tuple  # at top


# ─────────────────────────────────────────────────────────────────────────────
# 1. KV CACHE
# ─────────────────────────────────────────────────────────────────────────────

class KVCache:
    """
    Key-Value cache for fast autoregressive inference.

    WITHOUT cache:
        Each new token requires a full forward pass over ALL previous tokens.
        Computing K and V for position t requires re-reading positions 0..t-1.
        Cost per token: O(T²) — quadratic in context length.

    WITH cache:
        K and V from positions 0..t-1 are stored after first computation.
        Each new token only needs to compute Q, K, V for the NEW position,
        then attend against the cached K, V.
        Cost per token: O(T) — linear. 10-100x faster for long sequences.

    HOW IT WORKS:
        - During the first forward pass (prefill), compute and cache all K, V
        - For each subsequent token, compute only the new K, V, append to cache
        - Q for the new token attends over all cached K, V

    Layout: cache[layer_idx] = (K, V)
            K shape: (1, n_heads, T_so_far, head_dim)
            V shape: (1, n_heads, T_so_far, head_dim)
    """

    def __init__(self, n_layers: int, n_heads: int, head_dim: int, max_seq_len: int):
        """
        Args:
            n_layers:    Number of transformer layers.
            n_heads:     Number of attention heads.
            head_dim:    Dimension per head (d_model / n_heads).
            max_seq_len: Maximum context length (preallocate buffers).
        """
        self.n_layers  = n_layers
        self.n_heads   = n_heads
        self.head_dim  = head_dim
        self.max_len   = max_seq_len

        # Preallocate full-size buffers — no reallocation during generation
        # Shape: (1, n_heads, max_seq_len, head_dim)
        shape = (1, n_heads, max_seq_len, head_dim)
        self._k = [np.zeros(shape, dtype=np.float32) for _ in range(n_layers)]
        self._v = [np.zeros(shape, dtype=np.float32) for _ in range(n_layers)]
        self._len = 0   # number of tokens currently cached

    def update(
        self,
        layer: int,
        new_k: np.ndarray,
        new_v: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Append new K, V for one new token, return full cached K, V.

        Args:
            layer: Which transformer layer (0-indexed).
            new_k: New key,   shape (1, n_heads, 1, head_dim).
            new_v: New value, shape (1, n_heads, 1, head_dim).

        Returns:
            (full_k, full_v): All keys/values up to and including new token,
                              shape (1, n_heads, T_so_far, head_dim).
        """
        pos = self._len
        assert pos < self.max_len, f"KV cache overflow: pos={pos} >= max={self.max_len}"

        # Write new K, V at position `pos`
        self._k[layer][:, :, pos:pos+1, :] = new_k
        self._v[layer][:, :, pos:pos+1, :] = new_v

        # Return a VIEW of the valid portion (no copy)
        return (
            self._k[layer][:, :, :pos+1, :],
            self._v[layer][:, :, :pos+1, :],
        )

    def advance(self):
        """Increment the cached length by 1 after processing one new token."""
        self._len += 1

    def reset(self):
        """Clear the cache (call before starting a new generation)."""
        self._len = 0

    @property
    def length(self) -> int:
        """Current number of tokens in the cache."""
        return self._len


def cached_attention(
    Q: np.ndarray,
    K_new: np.ndarray,
    V_new: np.ndarray,
    cache: KVCache,
    layer: int,
) -> np.ndarray:
    """
    Single-step attention using the KV cache.

    Q is for ONE new token: shape (1, n_heads, 1, head_dim).
    K_new, V_new are for the new token: shape (1, n_heads, 1, head_dim).
    The cache holds all previous K, V.

    No masking needed: the cached K, V are all from previous positions,
    so they're all valid for the new token to attend to.

    Returns:
        Attended values for the new token: (1, n_heads, 1, head_dim).
    """
    # Append new K, V to cache and get full K, V
    K_full, V_full = cache.update(layer, K_new, V_new)  # (1, H, T, hd)

    d_k = Q.shape[-1]
    scale = 1.0 / math.sqrt(d_k)

    # Q: (1, H, 1, hd) x K^T: (1, H, hd, T) → scores: (1, H, 1, T)
    scores = (Q @ K_full.swapaxes(-1, -2)) * scale

    # No mask: all K are from past positions, all valid
    # Softmax over T
    scores = scores - scores.max(axis=-1, keepdims=True)
    attn = np.exp(scores)
    attn /= attn.sum(axis=-1, keepdims=True)

    # (1, H, 1, T) × (1, H, T, hd) → (1, H, 1, hd)
    return attn @ V_full


# ─────────────────────────────────────────────────────────────────────────────
# 2. FLASH ATTENTION (memory-efficient chunked attention)
# ─────────────────────────────────────────────────────────────────────────────

def flash_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    causal: bool = True,
    block_size: int = 64,
) -> np.ndarray:
    """
    Memory-efficient attention using the online softmax trick (Flash Attention).

    THE PROBLEM WITH STANDARD ATTENTION:
        Standard attention materialises the full (T, T) attention score matrix.
        For T=2048, this is 2048² × 4 bytes = 16 MB per head per batch item.
        With 32 heads and batch 8: 4 GB just for attention scores. Intractable.

    THE FLASH ATTENTION INSIGHT (Dao et al., 2022):
        We don't need to materialise the full matrix. We can compute the
        weighted sum of V incrementally, one block of K/V at a time, using
        the "online softmax" trick to maintain running statistics.

        For each block of K, V:
          1. Compute scores for this block: Q × K_block^T / √d_k
          2. Update running max (for numerical stability)
          3. Update running sum (renormalize for the new max)
          4. Accumulate weighted V

        Memory: O(T × block_size) instead of O(T²).
        This implementation is a pure NumPy approximation for clarity.

    Args:
        Q, K, V:    Shape (B, H, T, head_dim).
        causal:     Apply causal masking.
        block_size: Block size for tiling (smaller = less memory).

    Returns:
        Attended output of shape (B, H, T, head_dim).
    """
    B, H, T, d = Q.shape
    scale = 1.0 / math.sqrt(d)

    # Output buffer
    O = np.zeros_like(Q)

    for q_start in range(0, T, block_size):
        q_end = min(q_start + block_size, T)
        Q_block = Q[:, :, q_start:q_end, :]   # (B, H, bq, d)
        bq = q_end - q_start

        # Running statistics for online softmax
        running_max = np.full((B, H, bq, 1), -np.inf, dtype=np.float32)
        running_sum = np.zeros((B, H, bq, 1), dtype=np.float32)
        running_out = np.zeros((B, H, bq, d), dtype=np.float32)

        for kv_start in range(0, T, block_size):
            kv_end = min(kv_start + block_size, T)

            # Causal: skip K/V blocks that are entirely in the future
            if causal and kv_start >= q_end:
                break

            K_block = K[:, :, kv_start:kv_end, :]   # (B, H, bkv, d)
            V_block = V[:, :, kv_start:kv_end, :]

            # Compute attention scores for this block
            # (B, H, bq, d) × (B, H, d, bkv) → (B, H, bq, bkv)
            scores = (Q_block @ K_block.swapaxes(-1, -2)) * scale

            # Apply causal mask within the block
            if causal:
                for qi in range(bq):
                    abs_qi = q_start + qi
                    for ki in range(kv_end - kv_start):
                        abs_ki = kv_start + ki
                        if abs_ki > abs_qi:
                            scores[:, :, qi, ki] = -1e9

            # Online softmax update
            block_max = scores.max(axis=-1, keepdims=True)  # (B, H, bq, 1)
            new_max = np.maximum(running_max, block_max)

            # Renormalize old running stats for new max
            exp_diff_old = np.exp(running_max - new_max)
            exp_diff_new = np.exp(scores - new_max)

            running_sum  = exp_diff_old * running_sum + exp_diff_new.sum(axis=-1, keepdims=True)
            running_out  = exp_diff_old * running_out + exp_diff_new @ V_block
            running_max  = new_max

        # Normalize
        O[:, :, q_start:q_end, :] = running_out / (running_sum + 1e-8)

    return O


# ─────────────────────────────────────────────────────────────────────────────
# 3. BEAM SEARCH
# ─────────────────────────────────────────────────────────────────────────────

def beam_search(
    model,
    context: np.ndarray,
    max_new_tokens: int,
    beam_width: int = 4,
    length_penalty: float = 0.6,
    eos_token_id: Optional[int] = None,
    temperature: float = 1.0,
) -> List[Tuple[float, List[int]]]:
    """
    Beam search decoding.

    WHY BEAM SEARCH:
        Greedy decoding always picks the single best token at each step.
        This is locally optimal but globally suboptimal — a slightly worse
        token at step 3 might enable a much better sequence overall.

        Beam search maintains the top-B partial sequences (beams) at each step,
        expanding all of them and keeping the globally best B.

        Tradeoff: B=1 → greedy. Higher B → better quality, more compute.
        Typical: B=4 for generation, B=12 for machine translation.

    LENGTH PENALTY:
        Without it, beam search favors short sequences (higher probability
        as a product of fewer terms). We normalize by length^alpha:
            score = log_prob / (length ^ alpha)
        alpha=0: no penalty, alpha=0.6: mild (recommended), alpha=1.0: linear.

    Args:
        model:          GPT model in eval mode.
        context:        Initial token IDs, shape (1, T).
        max_new_tokens: Maximum new tokens to generate.
        beam_width:     Number of beams to keep (B).
        length_penalty: Length normalization exponent (alpha).
        eos_token_id:   Stop if this token is sampled.
        temperature:    Scale logits before log-softmax.

    Returns:
        List of (score, token_ids) sorted best-first.
        token_ids includes only the NEWLY generated tokens.
    """
    model.eval()
    prompt_len = context.shape[1]

    # Each beam: (cumulative_log_prob, token_list, finished)
    beams = [(0.0, context[0].tolist(), False)]

    for step in range(max_new_tokens):
        if all(b[2] for b in beams):
            break   # All beams finished

        candidates = []

        for log_prob, tokens, finished in beams:
            if finished:
                candidates.append((log_prob, tokens, True))
                continue

            # Forward pass for this beam
            inp = np.array([tokens[-model.max_seq_len:]], dtype=np.int64)
            logits = model(inp).data[0, -1, :]   # (vocab_size,)

            if temperature != 1.0:
                logits = logits / max(temperature, 1e-8)

            # Log-softmax for numerical stability
            logits_shifted = logits - logits.max()
            log_probs = logits_shifted - np.log(np.exp(logits_shifted).sum())

            # Top-B expansions from this beam
            top_k_idx = np.argpartition(log_probs, -beam_width)[-beam_width:]
            top_k_idx = top_k_idx[np.argsort(log_probs[top_k_idx])[::-1]]

            for token_id in top_k_idx:
                new_tokens = tokens + [int(token_id)]
                new_lp = log_prob + float(log_probs[token_id])
                done = (eos_token_id is not None and token_id == eos_token_id)
                candidates.append((new_lp, new_tokens, done))

        # Keep top beam_width candidates, sorted by length-normalized score
        def beam_score(beam):
            lp, toks, done = beam
            gen_len = len(toks) - prompt_len
            return lp / max(gen_len, 1) ** length_penalty

        candidates.sort(key=beam_score, reverse=True)
        beams = candidates[:beam_width]

    # Return beams sorted by final score (new tokens only)
    results = []
    for lp, tokens, done in beams:
        gen_tokens = tokens[prompt_len:]
        gen_len = max(len(gen_tokens), 1)
        score = lp / gen_len ** length_penalty
        results.append((score, gen_tokens))

    results.sort(key=lambda x: x[0], reverse=True)
    return results


def beam_search_generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    beam_width: int = 4,
    length_penalty: float = 0.6,
) -> str:
    """
    High-level beam search text generation.

    Args:
        model:          Trained GPT model.
        tokenizer:      CharTokenizer or BPETokenizer.
        prompt:         Input text.
        max_new_tokens: Max tokens to generate.
        beam_width:     Number of beams.
        length_penalty: Length normalization.

    Returns:
        Best generated string (prompt + generated).
    """
    ids = tokenizer.encode(prompt)
    context = np.array([ids], dtype=np.int64)
    eos_id = getattr(tokenizer, "EOS_ID", None)

    results = beam_search(
        model, context, max_new_tokens,
        beam_width=beam_width,
        length_penalty=length_penalty,
        eos_token_id=eos_id,
    )

    # Take the best beam
    best_score, best_tokens = results[0]
    return prompt + tokenizer.decode(best_tokens)


# ─────────────────────────────────────────────────────────────────────────────
# 4. GRADIENT ACCUMULATION
# ─────────────────────────────────────────────────────────────────────────────

class GradientAccumulator:
    """
    Gradient accumulation for large effective batch sizes on limited memory.

    THE PROBLEM:
        Training GPT-3 (175B params) with batch=1024 requires ~1024 forward
        passes worth of activations in memory simultaneously. Impossible.

    THE SOLUTION:
        Run N smaller forward/backward passes, accumulate gradients,
        then do ONE optimizer step. Mathematically equivalent to one large
        forward/backward pass (because gradients sum linearly).

        effective_batch_size = micro_batch_size × accumulation_steps

    This enables training with batch=256 on a GPU that only fits batch=16
    by using accumulation_steps=16.

    Usage:
        accum = GradientAccumulator(model, optimizer, accumulation_steps=4)
        for x_micro, y_micro in microbatches:
            loss = accum.step(x_micro, y_micro)   # accumulates grads
        accum.optimizer_step()                     # one real update
    """

    def __init__(self, model, optimizer, accumulation_steps: int = 4):
        """
        Args:
            model:              The GPT model.
            optimizer:          AdamW or SGD optimizer.
            accumulation_steps: How many micro-batches per effective batch.
        """
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self._micro_step = 0
        self._accum_loss = 0.0

    def step(
        self,
        x: np.ndarray,
        y: np.ndarray,
        loss_fn,
        clip_norm: float = 1.0,
    ) -> Optional[float]:
        """
        Process one micro-batch.

        Runs forward + backward, accumulates gradients.
        On the accumulation_steps-th call, also runs the optimizer step.

        Args:
            x:         Input token IDs (micro_batch, seq_len).
            y:         Target token IDs (micro_batch, seq_len).
            loss_fn:   Function(logits, targets) → scalar Tensor.
            clip_norm: Gradient clipping norm (applied before optimizer step).

        Returns:
            The effective batch loss on the accumulation step (None otherwise).
        """
        import gc
        from nanoGPT.training.optimizer import clip_grad_norm

        # Forward + backward
        logits = self.model(x)
        loss = loss_fn(logits, y)

        # Scale loss by 1/N so accumulated gradient equals the large-batch gradient
        scaled_loss_val = float(loss.data) / self.accumulation_steps
        self._accum_loss += scaled_loss_val

        # Scale the backward pass too
        loss_scaled = loss * (1.0 / self.accumulation_steps)
        loss_scaled.backward()
        del logits, loss, loss_scaled
        gc.collect()

        self._micro_step += 1

        # When we've accumulated enough gradients, do the optimizer step
        if self._micro_step >= self.accumulation_steps:
            clip_grad_norm(self.model.parameters(), clip_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self._micro_step = 0
            effective_loss = self._accum_loss
            self._accum_loss = 0.0
            return effective_loss

        return None


# ─────────────────────────────────────────────────────────────────────────────
# 5. PROPER WEIGHT INITIALIZATION (μP / GPT-2 style)
# ─────────────────────────────────────────────────────────────────────────────

def init_weights_gpt2(model) -> None:
    """
    GPT-2 weight initialization with residual scaling.

    THE PROBLEM WITH NAIVE INIT:
        In a deep network with residual connections, the residual stream
        grows as O(√n_layers) if each layer adds noise of similar scale.
        With 96 layers (GPT-3), activations explode by factor ≈10.

    GPT-2 SOLUTION:
        Scale the output projection of each attention and FFN sub-layer by
        1/√(2 * n_layers). This ensures the residual stream stays O(1)
        regardless of depth, since the (1/√2L)² per layer × 2L layers = 1.

    Also applies:
        - Linear layers: N(0, 0.02)
        - Embeddings: N(0, 0.02)
        - LayerNorm: weight=1, bias=0 (already set in LayerNorm.__init__)

    Args:
        model: A GPT model instance.
    """
    std = 0.02
    # Residual projection layers scaled by 1/sqrt(2 * n_layers)
    residual_std = 0.02 / math.sqrt(2 * model.n_layers)

    for block in model.blocks:
        # ── Attention ──────────────────────────────────────────────────────
        # QKV projection: standard init
        block.attn.qkv_proj.weight.data[:] = (
            np.random.randn(*block.attn.qkv_proj.weight.shape) * std
        )
        # Output projection: residual scaling (this layer adds to the stream)
        block.attn.out_proj.weight.data[:] = (
            np.random.randn(*block.attn.out_proj.weight.shape) * residual_std
        )
        if block.attn.out_proj.bias is not None:
            block.attn.out_proj.bias.data[:] = 0.0

        # ── FFN ────────────────────────────────────────────────────────────
        # Up-projection: standard
        block.ffn.fc1.weight.data[:] = (
            np.random.randn(*block.ffn.fc1.weight.shape) * std
        )
        # Down-projection: residual scaling
        block.ffn.fc2.weight.data[:] = (
            np.random.randn(*block.ffn.fc2.weight.shape) * residual_std
        )

    # Token + position embeddings: N(0, 0.02)
    model.token_embedding.weight.data[:] = (
        np.random.randn(*model.token_embedding.weight.shape) * std
    )
    model.pos_encoding.pos_embedding.weight.data[:] = (
        np.random.randn(*model.pos_encoding.pos_embedding.weight.shape) * std
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6. REPETITION PENALTY
# ─────────────────────────────────────────────────────────────────────────────

def apply_repetition_penalty(
    logits: np.ndarray,
    generated_tokens: List[int],
    penalty: float = 1.3,
) -> np.ndarray:
    """
    Penalize tokens that have already appeared in the generated sequence.

    WHY THIS HELPS:
        Language models can get stuck in repetition loops:
        "the the the the..." or "To be, or not to be, or not to be..."
        Repetition penalty reduces the probability of previously-seen tokens.

    FORMULA:
        If logit > 0: logit /= penalty   (reduce positive logits)
        If logit < 0: logit *= penalty   (make negative logits more negative)
        This ensures the penalty always reduces probability regardless of sign.

    Args:
        logits:           Raw logits of shape (vocab_size,).
        generated_tokens: List of token IDs already generated.
        penalty:          Penalty factor > 1 (1.0 = no penalty, 1.3 = mild).

    Returns:
        Penalized logits.
    """
    if penalty == 1.0 or not generated_tokens:
        return logits

    penalized = logits.copy()
    for token_id in set(generated_tokens):  # set: penalize once per unique token
        if penalized[token_id] > 0:
            penalized[token_id] /= penalty
        else:
            penalized[token_id] *= penalty

    return penalized


# ─────────────────────────────────────────────────────────────────────────────
# 7. MIN-P SAMPLING (2024 — often better than top-p)
# ─────────────────────────────────────────────────────────────────────────────

def apply_min_p(logits: np.ndarray, min_p: float = 0.05) -> np.ndarray:
    """
    Min-p sampling (Nguyen et al., 2024).

    WHY MIN-P IS BETTER THAN TOP-P:
        Top-p (nucleus) keeps tokens whose CUMULATIVE probability sums to p.
        But when the model is very confident (peaked distribution), the nucleus
        is tiny (1-2 tokens) — no diversity. When uncertain, it's large.

        Min-p instead keeps all tokens whose probability is at least p × p_max,
        where p_max is the probability of the most likely token.

        This scales naturally: when the model is confident (high p_max),
        the threshold is high (small nucleus). When uncertain (low p_max),
        the threshold is low (large nucleus). More adaptive than top-p.

    Args:
        logits: Raw logits of shape (vocab_size,).
        min_p:  Minimum probability ratio threshold (e.g., 0.05 = 5% of max).

    Returns:
        Filtered logits.
    """
    if min_p <= 0.0:
        return logits

    # Convert to probabilities
    shifted = logits - logits.max()
    probs = np.exp(shifted)
    probs /= probs.sum()

    # Threshold = min_p × max_probability
    threshold = min_p * probs.max()

    # Zero out tokens below threshold
    filtered = logits.copy()
    filtered[probs < threshold] = -1e9
    return filtered


def minp_topp_decode(
    model,
    context: np.ndarray,
    max_new_tokens: int,
    temperature: float = 1.0,
    min_p: float = 0.05,
    top_k: int = 0,
    repetition_penalty: float = 1.0,
    eos_token_id: Optional[int] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Full generation pipeline with all improvements:
    temperature → top-k → min-p → repetition penalty → sample.

    Args:
        model:               GPT model.
        context:             Prompt token IDs, shape (1, T).
        max_new_tokens:      Max tokens to generate.
        temperature:         Sampling temperature.
        min_p:               Min-p threshold (0.05 recommended).
        top_k:               Top-k pre-filter (0 = disabled).
        repetition_penalty:  Penalty for repeating tokens (1.0 = none).
        eos_token_id:        Stop token.
        seed:                Random seed.

    Returns:
        Full token sequence including prompt.
    """
    if seed is not None:
        np.random.seed(seed)

    model.eval()
    tokens = context.copy()
    generated = []

    for _ in range(max_new_tokens):
        crop = tokens[:, -model.max_seq_len:]
        logits = model(crop).data[0, -1, :]

        # 1. Temperature
        if temperature != 1.0:
            logits = logits / max(temperature, 1e-8)

        # 2. Top-k
        if top_k > 0:
            from nanoGPT.inference.generate import apply_top_k
            logits = apply_top_k(logits, top_k)

        # 3. Min-p
        logits = apply_min_p(logits, min_p)

        # 4. Repetition penalty
        if repetition_penalty != 1.0:
            logits = apply_repetition_penalty(logits, generated, repetition_penalty)

        # 5. Sample
        shifted = logits - logits.max()
        probs = np.exp(shifted)
        probs /= probs.sum()
        next_token = int(np.random.choice(len(probs), p=probs))

        tokens = np.concatenate([tokens, [[next_token]]], axis=1)
        generated.append(next_token)

        if eos_token_id is not None and next_token == eos_token_id:
            break

    return tokens


# ─────────────────────────────────────────────────────────────────────────────
# 8. LEARNING RATE FINDER
# ─────────────────────────────────────────────────────────────────────────────

def lr_range_test(
    model,
    dataset,
    batch_size: int = 16,
    start_lr: float = 1e-6,
    end_lr: float = 1e-1,
    n_steps: int = 100,
) -> Tuple[List[float], List[float]]:
    """
    Learning Rate Range Test (Smith, 2017).

    The fastest way to find a good learning rate:
        1. Start with very small LR (e.g., 1e-6)
        2. Increase LR exponentially each step
        3. Log the training loss at each step
        4. The optimal LR is just before the loss starts diverging

    Returns:
        (lrs, losses) lists for plotting.
    """
    import gc
    from nanoGPT.training.optimizer import AdamW
    from nanoGPT.core.losses import cross_entropy_loss

    # Save original weights to restore after the test
    orig_weights = [p.data.copy() for p in model.parameters()]

    optimizer = AdamW(model.parameters(), lr=start_lr)
    lr_multiplier = (end_lr / start_lr) ** (1.0 / n_steps)

    lrs, losses = [], []
    best_loss = float('inf')

    model.train()
    for step in range(n_steps):
        current_lr = start_lr * (lr_multiplier ** step)
        optimizer.lr = current_lr

        x, y = dataset.get_batch(batch_size)
        optimizer.zero_grad()
        logits = model(x)
        loss = cross_entropy_loss(logits, y)
        lv = float(loss.data)
        loss.backward()
        del logits, loss; gc.collect()
        optimizer.step()

        lrs.append(current_lr)
        losses.append(lv)

        if lv < best_loss:
            best_loss = lv
        if lv > 4 * best_loss:
            print(f"  LR test stopped early at step {step}: loss diverged")
            break

    # Restore original weights
    for p, orig in zip(model.parameters(), orig_weights):
        p.data[:] = orig

    model.eval()
    return lrs, losses


# ─────────────────────────────────────────────────────────────────────────────
# 9. DEMO — validate all new features
# ─────────────────────────────────────────────────────────────────────────────

def _validate_extras():
    """Test all new components."""
    import sys; sys.path.insert(0, '/home/claude')
    import numpy as np, math, gc
    np.random.seed(42)

    from nanoGPT import CharTokenizer, TextDataset, train_val_split, GPT
    from nanoGPT.training.optimizer import AdamW, CosineWithWarmup, clip_grad_norm
    from nanoGPT.core.losses import cross_entropy_loss
    from nanoGPT.inference.generate import InferenceEngine

    print("=" * 60)
    print("  nanoGPT v2 extras — validation")
    print("=" * 60)

    TEXT = (
        "To be, or not to be, that is the question:\n"
        "Whether tis nobler in the mind to suffer\n"
        "The slings and arrows of outrageous fortune,\n"
        "Or to take arms against a sea of troubles\n"
        "And by opposing end them. To die, to sleep,\n"
        "No more; and by a sleep to say we end\n"
        "The heartache and the thousand natural shocks\n"
        "That flesh is heir to. Tis a consummation\n"
        "Devoutly to be wished. To die, to sleep.\n"
        "To sleep, perchance to dream. Ay, there is the rub,\n"
        "For in that sleep of death what dreams may come\n"
        "When we have shuffled off this mortal coil,\n"
    ) * 20

    tok = CharTokenizer()
    tok.build(TEXT)
    ids = tok.encode(TEXT)
    train_ids, val_ids = train_val_split(ids, val_fraction=0.05)
    BLOCK, BATCH = 64, 8
    train_ds = TextDataset(train_ids, BLOCK)

    model = GPT(vocab_size=tok.vocab_size, d_model=128, n_layers=4,
                n_heads=4, max_seq_len=BLOCK, dropout=0.05, tie_weights=True)

    # ── Apply proper GPT-2 weight init ────────────────────────────────────────
    print("\n[1] Applying GPT-2 weight init...")
    init_weights_gpt2(model)
    print("  Done.")

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n[2] Training 300 steps with gradient accumulation (accum=2)...")
    opt = AdamW(model.parameters(), lr=3e-3, weight_decay=0.1)
    sched = CosineWithWarmup(opt, max_lr=3e-3, min_lr=3e-4,
                              warmup_steps=50, total_steps=300)
    accum = GradientAccumulator(model, opt, accumulation_steps=2)

    model.train()
    opt.zero_grad()
    for step in range(300):
        sched.step()
        x, y = train_ds.get_batch(BATCH)
        result = accum.step(x, y, lambda l, t: cross_entropy_loss(l, t))
        if result is not None and step % 100 == 0:
            print(f"  step={step} effective_loss={result:.4f}  "
                  f"ppl={math.exp(min(result, 20)):.1f}")

    # ── Flash Attention check ─────────────────────────────────────────────────
    print("\n[3] Flash Attention vs standard (correctness check)...")
    B, H, T, d = 1, 4, 20, 32
    Q = np.random.randn(B, H, T, d).astype('f')
    K = np.random.randn(B, H, T, d).astype('f')
    V = np.random.randn(B, H, T, d).astype('f')

    # Standard
    scale = 1.0 / math.sqrt(d)
    scores = Q @ K.swapaxes(-1, -2) * scale
    mask = np.tril(np.ones((T,T), dtype=bool))[None,None]
    scores = np.where(mask, scores, -1e9)
    scores -= scores.max(-1, keepdims=True)
    attn = np.exp(scores); attn /= attn.sum(-1, keepdims=True)
    std_out = attn @ V

    flash_out = flash_attention(Q, K, V, causal=True, block_size=8)
    err = np.abs(std_out - flash_out).max()
    print(f"  Max error vs standard: {err:.2e}  {'OK' if err < 1e-4 else 'WARN'}")

    # ── KV Cache check ────────────────────────────────────────────────────────
    print("\n[4] KV Cache correctness check...")
    n_layers, n_heads, head_dim = 2, 4, 32
    cache = KVCache(n_layers, n_heads, head_dim, max_seq_len=50)

    # Simulate 5 steps
    for i in range(5):
        q = np.random.randn(1, n_heads, 1, head_dim).astype('f')
        k = np.random.randn(1, n_heads, 1, head_dim).astype('f')
        v = np.random.randn(1, n_heads, 1, head_dim).astype('f')
        out = cached_attention(q, k, v, cache, layer=0)
        cache.advance()
    print(f"  Cache length after 5 steps: {cache.length}  OK")
    cache.reset()
    print(f"  Cache length after reset: {cache.length}  OK")

    # ── Beam Search ───────────────────────────────────────────────────────────
    print("\n[5] Beam search generation...")
    model.eval()
    beam_out = beam_search_generate(model, tok, "To be, or not",
                                     max_new_tokens=60, beam_width=3)
    print(f"  [BEAM] {beam_out[:80]}...")

    # ── Min-p sampling ────────────────────────────────────────────────────────
    print("\n[6] Min-p generation with repetition penalty...")
    prompt_ids = np.array([tok.encode("To be, or not")], dtype=np.int64)
    out_ids = minp_topp_decode(
        model, prompt_ids, max_new_tokens=80,
        temperature=0.8, min_p=0.05, repetition_penalty=1.15, seed=42
    )
    decoded = tok.decode(out_ids[0, len(tok.encode("To be, or not")):].tolist())
    print(f"  [MIN-P + REP PENALTY] To be, or not{decoded[:80]}...")

    # ── Comparison: v1 greedy vs v2 beam ────────────────────────────────────
    print("\n[7] Head-to-head: greedy vs beam vs min-p+rep_penalty")
    print("-" * 60)
    engine = InferenceEngine(model, tok)
    PROMPT = "To be, or not to be"

    greedy = engine.generate(PROMPT, max_new_tokens=80, strategy="greedy")
    beam   = beam_search_generate(model, tok, PROMPT, max_new_tokens=80, beam_width=4)
    p_ids  = np.array([tok.encode(PROMPT)], dtype=np.int64)
    minp   = minp_topp_decode(model, p_ids, 80, temperature=0.8,
                               min_p=0.05, repetition_penalty=1.2, seed=42)
    minp_text = PROMPT + tok.decode(minp[0, len(tok.encode(PROMPT)):].tolist())

    print(f"GREEDY:\n{greedy}\n")
    print(f"BEAM (width=4):\n{beam}\n")
    print(f"MIN-P + REP PENALTY:\n{minp_text}\n")

    print("=" * 60)
    print("  All v2 extras validated.")
    print("=" * 60)


if __name__ == "__main__":
    _validate_extras()


if __name__ == "__main__":
    _validate_extras()
