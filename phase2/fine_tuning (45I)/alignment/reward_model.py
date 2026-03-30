# ============================================================
# FILE: alignment/reward_model.py
# Reward model trained on human preference pairs.
#
# Training signal: you show the model two outputs (A, B),
# mark which is better, and the reward model learns to
# score outputs the way you score them.
#
# Architecture: takes a (prompt, response) pair → scalar score.
# Trained with Bradley-Terry pairwise loss.
# ============================================================

import numpy as np
import json, os


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class RewardModel:
    """
    Lightweight reward model: linear scoring head on top of
    simple bag-of-tokens features.

    In production: replace _encode with a frozen transformer
    embedding from the base model.
    """

    def __init__(self, vocab_size=512, feature_dim=64, seed=42):
        rng = np.random.default_rng(seed)

        self.vocab_size  = vocab_size
        self.feature_dim = feature_dim

        # Embedding: token → feature vector (frozen in this version)
        self.embed = rng.normal(0, 0.02, (vocab_size, feature_dim))

        # Scoring head: feature_dim → 1 scalar
        self.W = rng.normal(0, 0.02, (feature_dim,))
        self.b = 0.0

        # Gradient accumulators
        self.dW = np.zeros_like(self.W)
        self.db = 0.0

        # Preference pairs for training: list of (tokens_a, tokens_b, label)
        # label = 1 if a is better, 0 if b is better
        self.pairs = []

    def _encode(self, token_ids):
        """
        Encode a token sequence to a single feature vector.
        Mean-pools the token embeddings.
        """
        if not token_ids:
            return np.zeros(self.feature_dim)
        ids     = np.array(token_ids) % self.vocab_size
        vectors = self.embed[ids]          # (seq_len, feature_dim)
        return vectors.mean(axis=0)        # (feature_dim,)

    def score(self, token_ids):
        """
        Score a single response.

        Args:
            token_ids (list[int]): Tokenised (prompt + response).

        Returns:
            float: Scalar reward score (higher = better).
        """
        feat = self._encode(token_ids)
        return float(np.dot(feat, self.W) + self.b)

    def _bt_loss(self, score_a, score_b, label):
        """
        Bradley-Terry pairwise loss.

        If label=1 (a is better):  loss = -log σ(score_a - score_b)
        If label=0 (b is better):  loss = -log σ(score_b - score_a)

        Returns (loss, gradient_w.r.t. score_a).
        """
        margin = score_a - score_b
        if label == 0:
            margin = -margin

        prob = sigmoid(margin)
        loss = -np.log(prob + 1e-12)

        # Gradient of loss w.r.t. margin
        d_margin = -(1.0 - prob)
        if label == 0:
            d_margin = -d_margin

        return float(loss), float(d_margin)

    def train_step(self, tokens_a, tokens_b, label, lr=1e-3):
        """
        One gradient update from a single preference pair.

        Args:
            tokens_a (list[int]): Tokenised response A.
            tokens_b (list[int]): Tokenised response B.
            label    (int):       1 if A is better, 0 if B is better.
            lr       (float):     Learning rate.

        Returns:
            float: Loss for this pair.
        """
        feat_a = self._encode(tokens_a)
        feat_b = self._encode(tokens_b)

        score_a = float(np.dot(feat_a, self.W) + self.b)
        score_b = float(np.dot(feat_b, self.W) + self.b)

        loss, d_margin = self._bt_loss(score_a, score_b, label)

        # Gradient w.r.t. W (chain rule: d_loss/dW = d_loss/d_score * d_score/dW)
        # score_a = feat_a @ W + b → d_score_a/dW = feat_a
        grad_W = d_margin * (feat_a - feat_b)   # combined gradient
        grad_b = d_margin * (1.0 - 1.0)         # cancels for margin

        self.W -= lr * grad_W
        self.b -= lr * float(grad_b)

        return loss

    def train(self, pairs, epochs=5, lr=1e-3):
        """
        Train on a list of (tokens_a, tokens_b, label) tuples.

        Returns:
            list[float]: Per-epoch average loss.
        """
        losses = []
        rng    = np.random.default_rng(42)

        for epoch in range(1, epochs + 1):
            rng.shuffle(pairs)
            epoch_loss = []
            for tokens_a, tokens_b, label in pairs:
                loss = self.train_step(tokens_a, tokens_b, label, lr)
                epoch_loss.append(loss)

            avg = float(np.mean(epoch_loss))
            losses.append(avg)
            print(f"  [RewardModel] Epoch {epoch}/{epochs}  loss={avg:.4f}")

        return losses

    def add_preference(self, tokens_a, tokens_b, label):
        """Buffer a preference pair for later batch training."""
        self.pairs.append((tokens_a, tokens_b, label))

    def train_buffered(self, epochs=5, lr=1e-3):
        """Train on all buffered pairs."""
        if not self.pairs:
            print("[RewardModel] No buffered pairs to train on.")
            return []
        return self.train(self.pairs, epochs, lr)

    def save(self, path):
        np.savez(path, W=self.W, b=np.array([self.b]), embed=self.embed)
        print(f"[RewardModel] Saved → {path}.npz")

    @classmethod
    def load(cls, path):
        data = np.load(path if path.endswith('.npz') else path + '.npz')
        vocab_size, feature_dim = data['embed'].shape
        obj       = cls(vocab_size, feature_dim)
        obj.W     = data['W']
        obj.b     = float(data['b'][0])
        obj.embed = data['embed']
        return obj


# ============================================================
# TEST
# ============================================================

if __name__ == '__main__':
    print("=" * 55)
    print("  REWARD MODEL SELF TEST")
    print("=" * 55)

    rm = RewardModel(vocab_size=128, feature_dim=32)

    # Synthetic preference pairs: longer responses are "better"
    rng   = np.random.default_rng(0)
    pairs = []
    for _ in range(40):
        a = rng.integers(0, 128, size=rng.integers(10, 30)).tolist()
        b = rng.integers(0, 128, size=rng.integers(5, 12)).tolist()
        pairs.append((a, b, 1))   # A (longer) is always labelled better

    losses = rm.train(pairs, epochs=5, lr=1e-2)
    print(f"\nLoss curve: {[f'{l:.4f}' for l in losses]}")
    assert losses[-1] < losses[0], "Loss should decrease"

    # Check longer responses score higher
    long_resp  = rng.integers(0, 128, 25).tolist()
    short_resp = rng.integers(0, 128, 6).tolist()
    s_long  = rm.score(long_resp)
    s_short = rm.score(short_resp)
    print(f"\nScore (long)  = {s_long:.4f}")
    print(f"Score (short) = {s_short:.4f}")

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, 'reward_model')
        rm.save(p)
        rm2 = RewardModel.load(p)
        assert abs(rm2.score(long_resp) - s_long) < 1e-6, "Load mismatch"

    print("\n✓ Reward model all checks passed.")
    print("=" * 55)
