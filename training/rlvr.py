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
import time
from dataclasses import asdict, dataclass, field

try:
    from identity.hal import HALModule
except Exception:
    HALModule = None

try:
    from runtime.feedback_bus import record_verifier_feedback
except Exception:
    record_verifier_feedback = None

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
    policy_loss: float = 0.0
    kl_loss: float = 0.0
    effective_kl: float = 0.0
    output_lengths: list[int] = field(default_factory=list)
    verifier_pass_rate: float = 0.0
    replay_additions: int = 0
    reward_stats: dict[str, float] = field(default_factory=dict)
    dapo_config: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RLVRDapoConfig:
    overlong_penalty: float = 0.0
    overlong_token_limit: int = 256
    token_level_policy_loss: bool = False
    dynamic_sampling: bool = False
    max_dynamic_G: int = 8
    verifier_pass_threshold: float = 0.5


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
        dapo_config: RLVRDapoConfig | None = None,
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
        self.dapo_config = dapo_config or RLVRDapoConfig(overlong_token_limit=self.max_new_tokens)

        self._ref_model = copy.deepcopy(model)
        for p in self._ref_model.parameters():
            p.requires_grad_(False)
        self._ref_model.eval()
        self._steps_since_sync = 0
        self._consecutive_failures: int = 0
        self._last_effective_kl = self.kl_coeff
        self.last_step_report: dict[str, object] | None = None

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

    def _sample_count_for_task(self) -> int:
        if not self.dapo_config.dynamic_sampling:
            return self.G
        extra = min(max(0, self._consecutive_failures), max(0, self.dapo_config.max_dynamic_G - self.G))
        return max(1, self.G + extra)

    def _completion_token_count(self, completion: str) -> int:
        return len(self.tokenizer.encode(completion))

    def _shape_reward(self, reward: float, output_tokens: int) -> float:
        if self.dapo_config.overlong_penalty <= 0:
            return reward
        overflow = max(0, int(output_tokens) - int(self.dapo_config.overlong_token_limit))
        if overflow <= 0:
            return reward
        return reward - self.dapo_config.overlong_penalty * overflow

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

    def _loss_logprob(self, model_to_use, prompt: str, completion: str, output_tokens: int) -> torch.Tensor:
        logprob = self._compute_logprobs(model_to_use, prompt, completion)
        if self.dapo_config.token_level_policy_loss:
            return logprob / max(1, int(output_tokens))
        return logprob

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
            completions = self._generate_completions(task.prompt, self._sample_count_for_task())
        else:
            completions = list(completions)

        rewards = []
        raw_rewards = []
        output_lengths = []
        verifier_results = []
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
            verifier_results.append(vr)
            reward = float(vr.score)
            raw_rewards.append(reward)
            output_tokens = self._completion_token_count(c)
            output_lengths.append(output_tokens)
            reward = self._shape_reward(reward, output_tokens)
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
            try:
                from runtime.hal_telemetry import publish_hal_state

                publish_hal_state(self.hal, source="rlvr")
            except Exception:
                pass

        r = torch.tensor(rewards, dtype=torch.float32)
        mean_r = r.mean()
        std_r = r.std(unbiased=False) + 1e-8
        advantages = ((r - mean_r) / std_r).tolist()

        self.model.train()
        self.optimizer.zero_grad()

        device = self._device()
        policy_loss = torch.zeros((), device=device)
        kl_loss = torch.zeros((), device=device)
        kl_values = []

        for completion, advantage, output_tokens in zip(completions, advantages, output_lengths):
            lp_cur = self._loss_logprob(self.model, task.prompt, completion, output_tokens)
            with torch.no_grad():
                lp_ref = self._loss_logprob(self._ref_model, task.prompt, completion, output_tokens)

            policy_loss = policy_loss + (-float(advantage) * lp_cur)
            kl_gap = lp_cur - lp_ref.detach()
            kl_values.append(float(kl_gap.detach().item()))
            kl_loss = kl_loss + torch.clamp(kl_gap, min=0.0)

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

        verifier_pass_rate = sum(
            1 for reward in raw_rewards if reward >= self.dapo_config.verifier_pass_threshold
        ) / max(1, len(raw_rewards))
        reward_stats = {
            "raw_mean": float(sum(raw_rewards) / max(1, len(raw_rewards))),
            "shaped_mean": float(mean_r.item()),
            "min": float(min(rewards) if rewards else 0.0),
            "max": float(max(rewards) if rewards else 0.0),
            "kl_mean": float(sum(kl_values) / max(1, len(kl_values))),
            "output_tokens_mean": float(sum(output_lengths) / max(1, len(output_lengths))),
        }
        step = RLVRStep(
            task=task,
            completions=completions,
            rewards=rewards,
            advantages=advantages,
            loss=float(total_loss.item()),
            mean_reward=float(mean_r.item()),
            policy_loss=float(policy_loss.detach().item()),
            kl_loss=float(kl_loss.detach().item()),
            effective_kl=float(effective_kl),
            output_lengths=output_lengths,
            verifier_pass_rate=float(verifier_pass_rate),
            reward_stats=reward_stats,
            dapo_config=asdict(self.dapo_config),
        )
        replay_additions = 0
        if self.replay_pipeline is not None and float(mean_r.item()) < self.replay_min_reward:
            try:
                replay_additions = int(self.replay_pipeline.add_rlvr_step(step))
            except Exception:
                pass
        step.replay_additions = replay_additions
        self.last_step_report = self._step_report(step)
        if self._consecutive_failures >= 3:
            self._write_failure_replay(task, completions, step)
        if record_verifier_feedback is not None:
            for completion, vr in zip(completions, verifier_results):
                try:
                    if float(getattr(vr, "score", 0.0)) < self.replay_min_reward:
                        record_verifier_feedback(
                            prompt=task.prompt,
                            response=completion,
                            verifier_result=vr,
                            task_type=task.task_type,
                            hal=self.hal,
                        )
                except Exception:
                    pass
        return step

    def _step_report(self, step: RLVRStep) -> dict[str, object]:
        return {
            "generated_at": time.time(),
            "task_id": step.task.task_id,
            "task_type": step.task.task_type,
            "G": len(step.completions),
            "loss": step.loss,
            "policy_loss": step.policy_loss,
            "kl_loss": step.kl_loss,
            "effective_kl": step.effective_kl,
            "mean_reward": step.mean_reward,
            "verifier_pass_rate": step.verifier_pass_rate,
            "output_lengths": step.output_lengths,
            "reward_stats": step.reward_stats,
            "replay_additions": step.replay_additions,
            "dapo_config": step.dapo_config,
        }

    def write_last_step_report(self, output_path=None) -> dict[str, object]:
        if self.last_step_report is None:
            raise RuntimeError("No RLVR step has been run yet.")
        from training.v2_runtime import v2_report_path, write_json

        path = output_path or v2_report_path("rlvr_report")
        write_json(path, self.last_step_report)
        return self.last_step_report

    def _write_failure_replay(
        self,
        task: RLVRTask,
        completions: list[str],
        step: "RLVRStep",
    ) -> None:
        """
        Convert a real training failure into a FAILURE_REPLAY
        DFC training example. This closes the loop between
        RLVR and DFC training data.
        """
        import json
        from datetime import datetime, timezone
        from anra.anra_paths import TRAINING_DATA_DIR

        if not completions or not step.rewards:
            return
        worst_idx = int(min(range(len(step.rewards)), key=lambda i: step.rewards[i]))
        failed_attempt = completions[worst_idx]
        worst_reward = step.rewards[worst_idx]

        domain = getattr(task, "domain", "general")
        task_type = getattr(task, "task_type", "unknown")

        example = {
            "text": (
                f"<bos>"
                f"<task domain=\"{domain}\" type=\"failure_replay\">"
                f"{task.prompt}"
                f"</task>"
                f"<act>FAILED ATTEMPT: {failed_attempt[:500]}</act>"
                f"<obs>REWARD: {worst_reward:.3f} - below threshold 0.35. "
                f"Verifier rejected this completion.</obs>"
                f"<err>reward_delta: {worst_reward:.3f} - 0.35 = "
                f"{worst_reward - 0.35:.3f}</err>"
                f"<upd>This completion approach failed. "
                f"Consecutive failures: {self._consecutive_failures}. "
                f"The verifier requires a different approach.</upd>"
                f"<eos>"
            ),
            "domain": domain,
            "template": "failure_replay",
            "verified": False,
            "source": "live_rlvr",
            "timestamp": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
            "reward": worst_reward,
            "task_type": task_type,
        }

        dfc_path = TRAINING_DATA_DIR / "frontier_dfc.jsonl"
        dfc_path.parent.mkdir(parents=True, exist_ok=True)
        with dfc_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(example) + "\n")
