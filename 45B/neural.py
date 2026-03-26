"""
model/neural.py — Steps 2–8
Pure numpy foundation: layers, activations, losses, backprop, SGD.
This is the conceptual backbone before PyTorch takes over.
"""

import numpy as np
from typing import Callable, List, Tuple, Optional


# ─────────────────────────────────────────────
# STEP 2 — Layer of neurons
# ─────────────────────────────────────────────

class DenseLayer:
    """
    Fully-connected layer: y = activation(xW + b)
    Owns weights, bias, forward state, and gradients.
    """
    def __init__(self, in_features: int, out_features: int,
                 activation: str = "relu", seed: int = 42):
        rng = np.random.default_rng(seed)
        # He init for ReLU; Glorot for others
        if activation == "relu":
            scale = np.sqrt(2.0 / in_features)
        else:
            scale = np.sqrt(1.0 / in_features)

        self.W = rng.normal(0, scale, (in_features, out_features))
        self.b = np.zeros(out_features)
        self.activation_name = activation
        self.activation_fn, self.activation_grad = _get_activation(activation)

        # cache for backprop
        self._x = None
        self._z = None
        self._a = None

        # gradients
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        self._z = x @ self.W + self.b
        self._a = self.activation_fn(self._z)
        return self._a

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        # gradient through activation
        d_z = d_out * self.activation_grad(self._z)
        # gradients w.r.t. weights and bias
        self.dW = self._x.T @ d_z
        self.db = d_z.sum(axis=0)
        # gradient w.r.t. input — propagate upstream
        return d_z @ self.W.T

    @property
    def params(self):
        return [self.W, self.b]

    @property
    def grads(self):
        return [self.dW, self.db]


# ─────────────────────────────────────────────
# STEP 4 — Activation functions (and their gradients)
# ─────────────────────────────────────────────

def _sigmoid(z):
    return np.where(z >= 0, 1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))

def _sigmoid_grad(z):
    s = _sigmoid(z)
    return s * (1 - s)

def _relu(z):
    return np.maximum(0, z)

def _relu_grad(z):
    return (z > 0).astype(float)

def _gelu(z):
    # Gaussian Error Linear Unit — approximation used in GPT
    return 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3)))

def _gelu_grad(z):
    cdf = 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3)))
    pdf = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    return cdf + z * pdf

def _tanh(z):
    return np.tanh(z)

def _tanh_grad(z):
    return 1 - np.tanh(z)**2

def _linear(z):
    return z

def _linear_grad(z):
    return np.ones_like(z)

_ACTIVATIONS = {
    "sigmoid": (_sigmoid, _sigmoid_grad),
    "relu":    (_relu,    _relu_grad),
    "gelu":    (_gelu,    _gelu_grad),
    "tanh":    (_tanh,    _tanh_grad),
    "linear":  (_linear,  _linear_grad),
}

def _get_activation(name: str):
    if name not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(_ACTIVATIONS)}")
    return _ACTIVATIONS[name]


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


# ─────────────────────────────────────────────
# STEP 3 — Full network (forward pass)
# ─────────────────────────────────────────────

class NeuralNetwork:
    """
    Multi-layer perceptron with arbitrary depth.
    Stacks DenseLayer objects, handles full forward + backward passes.
    """
    def __init__(self, layer_sizes: List[int], activations: List[str]):
        assert len(activations) == len(layer_sizes) - 1
        self.layers: List[DenseLayer] = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                DenseLayer(layer_sizes[i], layer_sizes[i+1], activations[i], seed=i)
            )

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            d_out = layer.backward(d_out)
        return d_out

    @property
    def all_params_and_grads(self):
        pairs = []
        for layer in self.layers:
            for p, g in zip(layer.params, layer.grads):
                pairs.append((p, g))
        return pairs


# ─────────────────────────────────────────────
# STEP 5 — Loss functions
# ─────────────────────────────────────────────

class CrossEntropyLoss:
    """Numerically stable cross-entropy with softmax baked in."""
    def __init__(self, label_smoothing: float = 0.0):
        self.label_smoothing = label_smoothing
        self._logits = None
        self._probs  = None
        self._targets = None

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """
        logits:  (N, C) raw scores
        targets: (N,)   integer class indices
        """
        self._logits  = logits
        self._probs   = softmax(logits)
        self._targets = targets
        N, C = logits.shape

        if self.label_smoothing > 0:
            smooth = self.label_smoothing / C
            one_hot = np.full((N, C), smooth)
            one_hot[np.arange(N), targets] += (1 - self.label_smoothing)
        else:
            one_hot = np.zeros((N, C))
            one_hot[np.arange(N), targets] = 1.0

        log_probs = np.log(self._probs + 1e-12)
        loss = -(one_hot * log_probs).sum() / N
        return float(loss)

    def backward(self) -> np.ndarray:
        N, C = self._probs.shape
        d_logits = self._probs.copy()

        if self.label_smoothing > 0:
            smooth = self.label_smoothing / C
            one_hot = np.full((N, C), smooth)
            one_hot[np.arange(N), self._targets] += (1 - self.label_smoothing)
            d_logits -= one_hot
        else:
            d_logits[np.arange(N), self._targets] -= 1.0

        return d_logits / N


class MSELoss:
    def __init__(self):
        self._diff = None

    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        self._diff = pred - target
        return float(np.mean(self._diff**2))

    def backward(self) -> np.ndarray:
        N = self._diff.shape[0]
        return 2 * self._diff / N


# ─────────────────────────────────────────────
# STEP 7 — Gradient descent optimizers
# ─────────────────────────────────────────────

class SGD:
    def __init__(self, lr: float = 0.01, momentum: float = 0.0,
                 weight_decay: float = 0.0):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._velocity = {}

    def step(self, params_and_grads):
        for i, (param, grad) in enumerate(params_and_grads):
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param
            if self.momentum > 0:
                v = self._velocity.get(i, np.zeros_like(param))
                v = self.momentum * v - self.lr * grad
                self._velocity[i] = v
                param += v
            else:
                param -= self.lr * grad


class AdamW:
    """Adam with decoupled weight decay — the optimizer of choice for LLMs."""
    def __init__(self, lr: float = 3e-4, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8,
                 weight_decay: float = 0.01):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self._m = {}   # first moment
        self._v = {}   # second moment
        self._t = 0    # step count

    def step(self, params_and_grads):
        self._t += 1
        bc1 = 1 - self.beta1 ** self._t
        bc2 = 1 - self.beta2 ** self._t

        for i, (param, grad) in enumerate(params_and_grads):
            m = self._m.get(i, np.zeros_like(param))
            v = self._v.get(i, np.zeros_like(param))

            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * grad**2

            self._m[i] = m
            self._v[i] = v

            m_hat = m / bc1
            v_hat = v / bc2

            # weight decay applied directly to param, not gradient
            param -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps)
                                + self.weight_decay * param)


# ─────────────────────────────────────────────
# STEP 8 — Training loop
# ─────────────────────────────────────────────

class Trainer:
    """
    Minimal training loop for the numpy MLP.
    Used to validate the foundations before PyTorch takes over.
    """
    def __init__(self, model: NeuralNetwork, loss_fn, optimizer,
                 lr_schedule: Optional[Callable] = None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.history = {"loss": [], "step": []}

    def train_step(self, x: np.ndarray, y: np.ndarray) -> float:
        # forward
        logits = self.model.forward(x)
        loss = self.loss_fn.forward(logits, y)
        # backward
        d_out = self.loss_fn.backward()
        self.model.backward(d_out)
        # optimizer step
        self.optimizer.step(self.model.all_params_and_grads)
        return loss

    def fit(self, x: np.ndarray, y: np.ndarray,
            steps: int = 1000, batch_size: int = 32,
            log_every: int = 100) -> List[float]:
        N = x.shape[0]
        losses = []
        rng = np.random.default_rng(0)

        for step in range(steps):
            if self.lr_schedule:
                self.optimizer.lr = self.lr_schedule(step)

            idx = rng.integers(0, N, batch_size)
            xb, yb = x[idx], y[idx]
            loss = self.train_step(xb, yb)
            losses.append(loss)

            if step % log_every == 0:
                print(f"step {step:5d} | loss {loss:.4f} | lr {self.optimizer.lr:.2e}")

        self.history["loss"].extend(losses)
        return losses


# ─────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("Steps 2–8: numpy neural network smoke test")
    print("=" * 50)

    # XOR — a classic non-linearly-separable problem
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y = np.array([0, 1, 1, 0])

    net = NeuralNetwork([2, 16, 16, 2], ["relu", "relu", "linear"])
    loss_fn = CrossEntropyLoss()
    opt = AdamW(lr=1e-2)
    trainer = Trainer(net, loss_fn, opt)

    print("\nTraining XOR (2-class) for 2000 steps...")
    losses = trainer.fit(X, y, steps=2000, batch_size=4, log_every=500)

    logits = net.forward(X)
    preds = logits.argmax(axis=1)
    acc = (preds == y).mean()
    print(f"\nFinal predictions: {preds}  |  True labels: {y}")
    print(f"Accuracy: {acc*100:.1f}%  |  Final loss: {losses[-1]:.4f}")
    print("\n✓ Steps 2–8 verified")
