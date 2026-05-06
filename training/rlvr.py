"""RLVR - Reinforcement Learning from Verifiable Rewards.

Closed self-improvement loop. No human labels. No learned reward model.
The verifier is the ground truth: code either runs or it does not.

GRPO: Group Relative Policy Optimization.
  - Generate G completions per task
  - Score all G with verifier
  - Normalize advantages: (r - mean) / (std + eps)
  - Policy gradient + KL penalty against frozen reference
  - Backpropagate, clip, step
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

try:
    from identity.hal import HALModule
except Exception:
    HALModule = None

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - structural tests can still inspect this module.
    torch = None
    F = None


def _no_grad():
    if torch is not None:
        return torch.no_grad()

    def decorator(fn):
        return fn

    return decorator


@dataclass
class RLVRTask:
    prompt: str
    task_type: str  # "code" | "math" | "instruction" | "open"
    test_code: str = ""
    expected: str = ""
    task_id: str = ""


@dataclass
class RLVRStep:
    task: RLVRTask
    completions: list[str]
    rewards: list[float]
    advantages: list[float]
    loss: float
    mean_reward: float


class RLVRTrainer:
    """Reinforcement learning from verifier-scored completions."""

    def __init__(
        self,
        model,
        tokenizer,
        optimizer: torch.optim.Optimizer,
        verifier,
        hal: HALModule | None = None,
        G: int = 4,
        kl_coeff: float = 0.04,
        max_new_tokens: int = 256,
        grad_clip: float = 1.0,
        replay_pipeline=None,
        replay_min_reward: float = 0.5,
        entropy_bonus: float = 0.01,
    ) -> None:
        if torch is None:
            raise ImportError("RLVRTrainer requires torch.")
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.verifier = verifier
        self.hal = hal
        self.G = int(G)
        self.kl_coeff = float(kl_coeff)
        self.max_new_tokens = int(max_new_tokens)
        self.grad_clip = float(grad_clip)
        self.replay_pipeline = replay_pipeline
        self.replay_min_reward = float(replay_min_reward)
        self.entropy_bonus = float(entropy_bonus)

        self._ref_model = copy.deepcopy(model)
        for p in self._ref_model.parameters():
            p.requires_grad_(False)
        self._ref_model.eval()
        self._steps_since_sync = 0
        self._consecutive_failures: int = 0
        self._last_effective_kl = self.kl_coeff

    def sync_reference(self) -> None:
        """Refresh the KL anchor from the current policy."""
        self._ref_model.load_state_dict(self.model.state_dict())
        self._ref_model.eval()

    def _device(self) -> torch.device:
        return next(self.model.parameters()).device

    @_no_grad()
    def _generate_one(self, prompt_ids: list[int], temperature: float = 0.8) -> list[int]:
        """Sample tokens from prompt_ids. Returns completion IDs only."""
        special_ids = getattr(self.tokenizer, "special_ids", {})
        eos_id = special_ids.get("<eos>", -1)
        device = self._device()
        block = getattr(self.model, "block_size", 2048)
        generated = list(prompt_ids)

        self.model.eval()
        for _ in range(self.max_new_tokens):
            x = torch.tensor([generated[-block:]], dtype=torch.long, device=device)
            logits, _ = self.model(x)
            logits = logits[0, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            nxt = int(torch.multinomial(probs, 1).item())
            generated.append(nxt)
            if nxt == eos_id:
                break

        return generated[len(prompt_ids) :]

    def _generate_completions(self, prompt: str, n: int) -> list[str]:
        ids = self.tokenizer.encode(prompt)
        completions = []
        for _ in range(n):
            comp_ids = self._generate_one(ids)
            completions.append(self.tokenizer.decode(comp_ids))
        self.model.train()
        return completions

    def _compute_logprobs(self, model_to_use, prompt: str, completion: str) -> torch.Tensor:
        """Sum log-probs for completion tokens given prompt."""
        block = getattr(self.model, "block_size", 2048)
        device = self._device()
        p_ids = self.tokenizer.encode(prompt)
        c_ids = self.tokenizer.encode(completion)

        if not c_ids:
            return torch.tensor(0.0, device=device, requires_grad=model_to_use is self.model)

        full_ids = p_ids + c_ids
        start = max(0, len(full_ids) - block)
        all_ids = full_ids[start:]

        if len(all_ids) < 2:
            return torch.tensor(0.0, device=device, requires_grad=model_to_use is self.model)

        x = torch.tensor([all_ids[:-1]], dtype=torch.long, device=device)
        targets = torch.tensor([all_ids[1:]], dtype=torch.long, device=device)

        logits, _ = model_to_use(x)
        log_probs = F.log_softmax(logits, dim=-1)

        first_completion_pos = max(0, len(p_ids) - start)
        target_start = max(0, first_completion_pos - 1)
        if target_start >= log_probs.shape[1]:
            return torch.tensor(0.0, device=device, requires_grad=model_to_use is self.model)

        comp_lp = log_probs[0, target_start:, :]
        comp_tgt = targets[0, target_start:]
        return comp_lp.gather(1, comp_tgt.unsqueeze(1)).squeeze(1).sum()

    @_no_grad()
    def _completion_entropy(self, prompt: str, completion: str) -> float:
        """Mean next-token entropy over a completion under the current policy."""
        block = getattr(self.model, "block_size", 2048)
        device = self._device()
        p_ids = self.tokenizer.encode(prompt)
        c_ids = self.tokenizer.encode(completion)
        if not c_ids:
            return 0.0

        full_ids = p_ids + c_ids
        start = max(0, len(full_ids) - block)
        all_ids = full_ids[start:]
        if len(all_ids) < 2:
            return 0.0

        x = torch.tensor([all_ids[:-1]], dtype=torch.long, device=device)
        logits, _ = self.model(x)
        first_completion_pos = max(0, len(p_ids) - start)
        target_start = max(0, first_completion_pos - 1)
        if target_start >= logits.shape[1]:
            return 0.0

        comp_logits = logits[0, target_start:, :]
        probs = F.softmax(comp_logits, dim=-1)
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
        return float(entropy.mean().item())

    def train_step(self, task: RLVRTask, completions: list[str] | None = None) -> RLVRStep:
        """Generate, verify, compute GRPO loss, backpropagate, and step."""
        if completions is None:
            completions = self._generate_completions(task.prompt, self.G)
        else:
            completions = list(completions)

        rewards = []
        for c in completions:
            vr = self.verifier.score(
                task.task_type,
                code=c,
                test_code=task.test_code,
                expression=c,
                expected=task.expected,
                response=c,
                task=task.prompt,
            )
            reward = float(vr.score)
            if self.entropy_bonus:
                # AN: preserve exploration pressure so GRPO does not collapse into brittle low-entropy completions.
                reward += self.entropy_bonus * self._completion_entropy(task.prompt, c)
            rewards.append(reward)

        if self.hal is not None:
            mean_reward_now = sum(rewards) / max(1, len(rewards))
            self.hal.update(
                verifier_result=mean_reward_now,
                session_context={
                    "consecutive_failures": self._consecutive_failures,
                    "domain": getattr(task, "domain", ""),
                    "task_type": getattr(task, "task_type", ""),
                },
            )

        r = torch.tensor(rewards, dtype=torch.float32)
        mean_r = r.mean()
        std_r = r.std(unbiased=False) + 1e-8
        advantages = ((r - mean_r) / std_r).tolist()

        self.model.train()
        self.optimizer.zero_grad()

        device = self._device()
        policy_loss = torch.zeros((), device=device)
        kl_loss = torch.zeros((), device=device)

        for completion, advantage in zip(completions, advantages):
            lp_cur = self._compute_logprobs(self.model, task.prompt, completion)
            with torch.no_grad():
                lp_ref = self._compute_logprobs(self._ref_model, task.prompt, completion)

            policy_loss = policy_loss + (-float(advantage) * lp_cur)
            kl_loss = kl_loss + torch.clamp(lp_cur - lp_ref.detach(), min=0.0)

        group_size = max(1, len(completions))
        policy_loss = policy_loss / group_size
        kl_loss = kl_loss / group_size
        effective_kl = (
            self.hal.kl_coefficient(self.kl_coeff)
            if self.hal is not None
            else self.kl_coeff
        )
        self._last_effective_kl = float(effective_kl)
        total_loss = policy_loss + effective_kl * kl_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        mean_r_val = sum(rewards) / max(1, len(rewards))
        if mean_r_val < 0.35:
            self._consecutive_failures += 1
        else:
            self._consecutive_failures = 0
        self._steps_since_sync += 1
        if self._steps_since_sync >= 100:
            self.sync_reference()
            self._steps_since_sync = 0

        step = RLVRStep(
            task=task,
            completions=completions,
            rewards=rewards,
            advantages=advantages,
            loss=float(total_loss.item()),
            mean_reward=float(mean_r.item()),
        )
        if self.replay_pipeline is not None and float(mean_r.item()) < self.replay_min_reward:
            try:
                self.replay_pipeline.add_rlvr_step(step)
            except Exception:
                pass
        return step
