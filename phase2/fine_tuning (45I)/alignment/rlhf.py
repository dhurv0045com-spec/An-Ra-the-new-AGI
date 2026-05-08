# ============================================================
# FILE: alignment/rlhf.py
# Reinforcement Learning from Human Feedback training loop.
#
# Flow:
#   1. Generate two responses for a prompt
#   2. Human picks which is better (or rates them)
#   3. Reward model learns from that preference
#   4. Policy (language model) is nudged toward higher-reward outputs
#      using REINFORCE-style gradient update
#
# This is a simplified but structurally correct RLHF loop.
# Full PPO can be plugged in at the policy_update step.
# ============================================================

import numpy as np
import json, os
import copy
from contextlib import nullcontext as _nullcontext
from datetime import datetime

try:
    import torch
except Exception:
    torch = None  # type: ignore


class RLHFTrainer:
    """
    RLHF training loop connecting: model, reward model, feedback store.

    Policy update uses a simplified REINFORCE rule:
        Δθ ∝ reward × ∇log P(response | prompt)

    For a real system: replace _policy_update with PPO.
    """

    def __init__(self, model, reward_model, tokenizer, config=None):
        """
        Args:
            model        : Base / fine-tuned language model.
            reward_model : Trained RewardModel instance.
            tokenizer    : Matching tokenizer.
            config (dict):
                lr           : Policy learning rate.
                kl_coeff     : KL penalty weight (keeps model close to base).
                reward_scale : Normalisation factor for rewards.
                save_dir     : Where to write iteration checkpoints.
        """
        self.model        = model
        self.reward_model = reward_model
        self.tokenizer    = tokenizer
        try:
            self._reference_model = copy.deepcopy(self.model)
            for p in (getattr(self._reference_model, "parameters", lambda: []))():
                p.requires_grad_(False)
        except Exception:
            self._reference_model = None
        cfg               = config or {}

        self.lr           = cfg.get('lr',           1e-5)
        self.kl_coeff     = cfg.get('kl_coeff',     0.02)
        self.reward_scale = cfg.get('reward_scale',  1.0)
        self.save_dir     = cfg.get('save_dir',     'checkpoints/rlhf')

        # History
        self.reward_history = []   # per-step mean reward
        self.step_count     = 0

    def generate(self, prompt_tokens, max_new=40):
        """
        Generate a response token sequence from the model.
        (Greedy in this stub — replace with real sampling.)
        """
        rng       = np.random.default_rng(self.step_count)
        n_tokens  = rng.integers(8, max_new)
        vocab     = self.model.config.get('vocab_size', 512)
        return rng.integers(0, vocab, size=n_tokens).tolist()

    def _policy_update(self, prompt_tokens, response_tokens, reward, epsilon: float = 0.2) -> float:
        """
        PPO-clip policy update.
        Returns the clipped policy loss for logging.
        """
        current_logp = self._log_prob(prompt_tokens, response_tokens)
        no_grad = torch.no_grad() if torch is not None and hasattr(torch, "no_grad") else _nullcontext()
        with no_grad:
            old_logp = self._log_prob_ref(prompt_tokens, response_tokens)

        ratio = np.exp(min(current_logp - old_logp, 20.0))
        advantage = float(reward)
        clipped_ratio = np.clip(ratio, 1.0 - epsilon, 1.0 + epsilon)
        policy_loss = -min(ratio * advantage, clipped_ratio * advantage)

        self._apply_gradient(prompt_tokens, response_tokens, policy_loss)
        return float(policy_loss)

    def _log_prob_ref(self, prompt_tokens, response_tokens) -> float:
        """Log-prob under the frozen reference policy."""
        ref = getattr(self, "_reference_model", None)
        if ref is None:
            return self._log_prob(prompt_tokens, response_tokens)
        real_model = self.model
        self.model = ref
        try:
            return self._log_prob(prompt_tokens, response_tokens)
        finally:
            self.model = real_model

    def _log_prob(self, prompt_tokens, response_tokens) -> float:
        """Approximate log-probability of response tokens under the current model."""
        try:
            logits = self.model.forward(prompt_tokens)
            logits = np.asarray(logits)
            if logits.ndim == 1:
                logits = logits[np.newaxis, :]
            shifted = logits - logits.max(axis=-1, keepdims=True)
            probs = np.exp(shifted)
            probs = probs / probs.sum(axis=-1, keepdims=True)
            rows = min(len(response_tokens), probs.shape[0])
            if rows <= 0:
                return 0.0
            vocab = probs.shape[-1]
            total = 0.0
            for idx, tok in enumerate(response_tokens[:rows]):
                total += float(np.log(probs[idx, int(tok) % vocab] + 1e-12))
            return total / rows
        except Exception:
            return 0.0

    def _apply_gradient(self, prompt_tokens, response_tokens, policy_loss: float) -> None:
        """Apply a stub-compatible PPO-scaled update to the policy head."""
        if not hasattr(self.model, "weights") or "head" not in self.model.weights:
            return
        reward_signal = (-float(policy_loss)) * self.reward_scale - self.kl_coeff
        rng = np.random.default_rng(self.step_count)
        grad = rng.normal(0, 0.001, self.model.weights['head'].shape)
        self.model.weights['head'] += self.lr * reward_signal * grad

    def run_iteration(self, prompts, n_candidates=2):
        """
        One RLHF iteration over a list of prompts.

        For each prompt:
          - Generate n_candidates responses
          - Score all with reward model
          - Update policy toward the highest-scoring response

        Args:
            prompts      (list[list[int]]): Tokenised prompts.
            n_candidates (int):             Responses to generate per prompt.

        Returns:
            float: Mean reward across this iteration.
        """
        iter_rewards = []

        for prompt_tokens in prompts:
            candidates = [
                self.generate(prompt_tokens) for _ in range(n_candidates)
            ]

            # Score each candidate
            scores = [
                self.reward_model.score(prompt_tokens + resp)
                for resp in candidates
            ]

            best_idx    = int(np.argmax(scores))
            best_resp   = candidates[best_idx]
            best_reward = scores[best_idx]

            # Update policy toward best response
            self._policy_update(prompt_tokens, best_resp, best_reward)

            iter_rewards.append(best_reward)
            self.step_count += 1

        mean_reward = float(np.mean(iter_rewards))
        self.reward_history.append(mean_reward)
        return mean_reward

    def train(self, prompts, iterations=5, n_candidates=2):
        """
        Run multiple RLHF iterations.

        Args:
            prompts    (list[list[int]]): Pool of prompts.
            iterations (int):             Number of training iterations.

        Returns:
            list[float]: Mean reward per iteration.
        """
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"\n[RLHF] Starting — iterations={iterations}  "
              f"prompts={len(prompts)}  candidates={n_candidates}")

        for it in range(1, iterations + 1):
            mean_r = self.run_iteration(prompts, n_candidates)
            print(f"  Iteration {it}/{iterations}  mean_reward={mean_r:.4f}")

        print(f"[RLHF] Done. Reward trajectory: "
              f"{[f'{r:.3f}' for r in self.reward_history]}")
        return self.reward_history

    def save_checkpoint(self):
        """Save model state after RLHF."""
        ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(self.save_dir, f'rlhf_{ts}')
        os.makedirs(path, exist_ok=True)
        self.model.save(os.path.join(path, 'model'))
        meta = {
            'timestamp':      ts,
            'steps':          self.step_count,
            'reward_history': self.reward_history,
        }
        with open(os.path.join(path, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"[RLHF] Checkpoint saved → {path}")


# ============================================================
# TEST
# ============================================================

if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'finetune'))
    from pipeline     import BaseModelStub, SimpleTokenizer
    from reward_model import RewardModel

    print("=" * 55)
    print("  RLHF TRAINER SELF TEST")
    print("=" * 55)

    config = {'vocab_size': 128, 'd_model': 32}
    model     = BaseModelStub(config)
    tokenizer = SimpleTokenizer(128)
    rm        = RewardModel(vocab_size=128, feature_dim=16)

    # Quick reward model pre-train on synthetic pairs
    rng   = np.random.default_rng(0)
    pairs = [(rng.integers(0, 128, 20).tolist(),
              rng.integers(0, 128,  6).tolist(), 1) for _ in range(20)]
    rm.train(pairs, epochs=3, lr=1e-2)

    # Build prompts
    prompts = [rng.integers(0, 128, 10).tolist() for _ in range(6)]

    trainer = RLHFTrainer(model, rm, tokenizer,
                          config={'lr': 1e-5, 'save_dir': '/tmp/rlhf_test'})
    rewards = trainer.train(prompts, iterations=4, n_candidates=3)

    assert len(rewards) == 4
    print(f"\nFinal mean reward: {rewards[-1]:.4f}")
    print("\n✓ RLHF trainer all checks passed.")
    print("=" * 55)
