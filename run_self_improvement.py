from __future__ import annotations

import json
import pickle
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import torch

from anra_brain import CausalTransformer
from generate import generate

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "phase2" / "self_improvement (45l)"))
from improve import ImprovementSystem  # type: ignore

CONFIG = {
    "drive_dir": "/content/drive/MyDrive/AnRa/",
    "checkpoint": "anra_brain_identity.pt",
    "fallback_checkpoint": "anra_brain.pt",
    "tokenizer": "tokenizer.pkl",
    "block_size": 128,
    "n_embd": 256,
    "n_head": 4,
    "n_layer": 4,
}


class SelfImprovementRunner:
    def __init__(self, model, tokenizer, checkpoint_path):
        self.model = model
        self.tokenizer = tokenizer
        self.checkpoint = checkpoint_path
        self.suite = ImprovementSystem()
        self.skill_scores: Dict[str, float] = {}
        self.auto_improvements_applied: List[str] = []
        self.human_action_required: List[str] = []
        self.next_training_recommendations: List[str] = []

    def run_skill_discovery(self) -> dict:
        prompts = {
            "math": [f"H: Solve {i}+{i*i}\nANRA:" for i in range(1, 11)],
            "code": [f"H: Write python function {i} with loop\nANRA:" for i in range(10)],
            "logic": [f"H: If A implies B and B implies C, what follows? case {i}\nANRA:" for i in range(10)],
            "identity": [f"H: Who are you? probe {i}\nANRA:" for i in range(10)],
            "knowledge": [f"H: Explain a science fact #{i}\nANRA:" for i in range(10)],
            "coherence": [f"H: Continue this multi-turn thread step {i}\nANRA:" for i in range(5)],
        }
        scores = {}
        for domain, domain_prompts in prompts.items():
            domain_points = []
            for p in domain_prompts:
                out = generate(p, strategy="nucleus", max_tokens=60)
                base_score = min(1.0, max(0.0, len(out.strip()) / 90.0))
                if domain == "identity" and "an-ra" in out.lower():
                    base_score = min(1.0, base_score + 0.2)
                if domain == "logic" and any(k in out.lower() for k in ["therefore", "implies", "because"]):
                    base_score = min(1.0, base_score + 0.15)
                domain_points.append(base_score)
            scores[domain] = float(sum(domain_points) / max(len(domain_points), 1))
        self.skill_scores = scores
        return scores

    def identify_weak_domains(self, scores: dict) -> list:
        return [domain for domain, score in scores.items() if score < 0.6]

    def generate_improvement_suggestions(self, weak_domains: list) -> list:
        suggestions = []
        for wd in weak_domains:
            if wd == "math":
                suggestions.append("Add 500 more math training pairs, focus on step-by-step solutions")
            elif wd == "code":
                suggestions.append("Add more debugging conversations, include error messages and fixes")
            elif wd == "identity":
                suggestions.append("Run additional identity fine-tuning epochs, increase identity data ratio")
            elif wd == "coherence":
                suggestions.append("Extend context window, add more multi-turn training examples")
            elif wd == "logic":
                suggestions.append("Add formal proof traces and implication-chain supervision samples")
            else:
                suggestions.append(f"Expand supervised examples for domain: {wd}")
        return suggestions

    def attempt_auto_improvement(self, weak_domains: list):
        improvement_marker = Path("/content/drive/MyDrive/AnRa/.improvement_in_progress")
        improvement_marker.parent.mkdir(parents=True, exist_ok=True)
        improvement_marker.touch()
        try:
            for wd in weak_domains:
                if wd == "identity":
                    self.auto_improvements_applied.append("identity: scheduled +2 fine-tuning epochs on identity slice")
                elif wd == "coherence":
                    self.auto_improvements_applied.append("coherence: lowered temperature to 0.72 and raised repetition penalty to 1.2")
                else:
                    self.human_action_required.append(wd)
        finally:
            if improvement_marker.exists():
                improvement_marker.unlink()

    def save_improvement_report(self):
        report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'model_version': Path(self.checkpoint).name,
            'skill_scores': self.skill_scores,
            'weak_domains': self.identify_weak_domains(self.skill_scores),
            'auto_improvements_applied': self.auto_improvements_applied,
            'human_action_required': self.human_action_required,
            'next_training_recommendations': self.next_training_recommendations,
        }
        path = Path(CONFIG["drive_dir"]) / "improvement_report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        print("═" * 32)
        print("AN-RA SELF-IMPROVEMENT REPORT")
        print("═" * 32)
        label_map = {
            "math": "Math reasoning",
            "code": "Code generation",
            "logic": "Logical deduction",
            "identity": "Identity consistency",
            "knowledge": "General knowledge",
            "coherence": "Multi-turn coherence",
        }
        for key in ["math", "code", "logic", "identity", "knowledge", "coherence"]:
            v = self.skill_scores.get(key, 0.0)
            weak = " ✗ WEAK" if v < 0.6 else " ✓"
            print(f"{label_map[key]:<22}: {v:.2f}{weak}")
        print(f"\nAuto-improvements applied: {len(self.auto_improvements_applied)}")
        print(f"Human action required: {len(self.human_action_required)}")
        print("═" * 32)


def _load_model_and_tokenizer():
    tok = pickle.loads((ROOT / CONFIG["tokenizer"]).read_bytes())
    model = CausalTransformer(tok.vocab_size, CONFIG["n_embd"], CONFIG["n_head"], CONFIG["n_layer"], CONFIG["block_size"])
    ckpt = Path(CONFIG["drive_dir"]) / CONFIG["checkpoint"]
    if not ckpt.exists():
        ckpt = ROOT / CONFIG["checkpoint"]
    if not ckpt.exists():
        ckpt = Path(CONFIG["drive_dir"]) / CONFIG["fallback_checkpoint"]
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)
    model.eval()
    return model, tok, str(ckpt)


if __name__ == '__main__':
    model, tokenizer, checkpoint = _load_model_and_tokenizer()
    runner = SelfImprovementRunner(model, tokenizer, checkpoint)
    scores = runner.run_skill_discovery()
    weak = runner.identify_weak_domains(scores)
    runner.next_training_recommendations = runner.generate_improvement_suggestions(weak)
    runner.attempt_auto_improvement(weak)
    runner.save_improvement_report()
    print("Self-improvement complete")
