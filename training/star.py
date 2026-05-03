from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class STaRExample:
    prompt: str
    rationale: str
    answer: str
    score: float


@dataclass
class STaRLoop:
    model: object
    tokenizer: object
    verifier: object
    accepted: list[STaRExample] = field(default_factory=list)

    def generate(self, prompt: str, n: int = 4) -> list[str]:
        if hasattr(self.model, "generate"):
            out = self.model.generate(prompt, n=n)
            if isinstance(out, list):
                return [str(x) for x in out]
            return [str(out)]
        return [prompt] * n

    def step(self, prompt: str, task_type: str = "open") -> list[STaRExample]:
        candidates = self.generate(prompt)
        batch: list[STaRExample] = []
        for c in candidates:
            vr = self.verifier.score(task_type, task=prompt, response=c)
            ex = STaRExample(prompt=prompt, rationale=c, answer=c, score=vr.score)
            if vr.score >= 0.5:
                self.accepted.append(ex)
            batch.append(ex)
        return batch

    def replay_buffer(self) -> list[dict]:
        return [
            {"prompt": x.prompt, "rationale": x.rationale, "answer": x.answer, "score": x.score}
            for x in self.accepted
        ]
