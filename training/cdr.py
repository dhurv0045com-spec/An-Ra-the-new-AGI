"""Corrected-failure curriculum records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time


FAILURE_CATEGORIES = {
    "reasoning",
    "tool_selection",
    "planning",
    "memory",
    "identity_drift",
    "perception",
    "execution",
}


@dataclass(frozen=True)
class CorrectedFailure:
    prompt: str
    failed_output: str
    diagnosis: str
    corrected_target: str
    category: str
    verifier: str
    verified: bool
    provenance: dict[str, object] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


class CorrectedFailureCurriculum:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._queue: list[CorrectedFailure] = []
        self._metrics = {
            "captured": 0,
            "diagnosed": 0,
            "corrected": 0,
            "verified": 0,
            "replayed": 0,
            "retested": 0,
            "closed": 0,
        }

    def append(self, record: CorrectedFailure) -> None:
        if record.category not in FAILURE_CATEGORIES:
            raise ValueError(f"Unknown failure category: {record.category}")
        if not record.verified:
            raise ValueError("Unverified corrections cannot enter the curriculum.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(record), ensure_ascii=True) + "\n")

    def load(self) -> list[CorrectedFailure]:
        if not self.path.exists():
            return []
        return [
            CorrectedFailure(**json.loads(line))
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def capture_step_failure(
        self,
        *,
        input_tokens,
        target_tokens,
        predicted_tokens,
        loss: float,
        step: int,
        tokenizer,
        category: str = "reasoning",
    ) -> CorrectedFailure:
        def decode(value) -> str:
            tensor = value[0] if getattr(value, "ndim", 1) > 1 else value
            values = tensor.detach().cpu().tolist()
            pad = getattr(tokenizer, "pad_token_id", None)
            if pad is not None:
                values = [token for token in values if token != pad]
            return tokenizer.decode(values)

        record = CorrectedFailure(
            prompt=decode(input_tokens),
            failed_output=decode(predicted_tokens),
            diagnosis=f"high_loss_step:{step}:loss={float(loss):.6f}",
            corrected_target=decode(target_tokens),
            category=category if category in FAILURE_CATEGORIES else "reasoning",
            verifier="training_target",
            verified=True,
            provenance={"step": int(step), "loss": float(loss), "source": "training_step"},
        )
        self._queue.append(record)
        for key in ("captured", "diagnosed", "corrected", "verified"):
            self._metrics[key] += 1
        return record

    def capture_task_result(
        self,
        *,
        prompt: str,
        output: str,
        category: str,
        success: bool,
        diagnosis: str = "",
        corrected_target: str = "",
        verifier: str = "",
        verified: bool = False,
        provenance: dict[str, object] | None = None,
    ) -> CorrectedFailure | None:
        if success:
            return None
        record = CorrectedFailure(
            prompt=prompt,
            failed_output=output,
            diagnosis=diagnosis or "task_failed",
            corrected_target=corrected_target,
            category=category if category in FAILURE_CATEGORIES else "execution",
            verifier=verifier,
            verified=verified,
            provenance=provenance or {},
        )
        self._queue.append(record)
        self._metrics["captured"] += 1
        self._metrics["diagnosed"] += int(bool(record.diagnosis))
        self._metrics["corrected"] += int(bool(record.corrected_target))
        self._metrics["verified"] += int(bool(record.verified))
        return record

    def flush_to_dataset(self, replay_path: str | Path | None = None) -> int:
        target = Path(replay_path) if replay_path is not None else self.path
        ready = [
            record
            for record in self._queue
            if record.verified and record.corrected_target
        ]
        if not ready:
            return 0
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as stream:
            for record in ready:
                stream.write(
                    json.dumps(
                        {
                            "prompt": record.prompt,
                            "answer": record.corrected_target,
                            "bucket": "replay",
                            "source": f"cdr_{record.category}",
                            "weight": 1.5,
                            "verified": True,
                            "provenance": record.provenance,
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )
        ready_ids = {id(record) for record in ready}
        self._queue = [
            record for record in self._queue if id(record) not in ready_ids
        ]
        self._metrics["replayed"] += len(ready)
        self._metrics["closed"] += len(ready)
        return len(ready)

    def report(self) -> dict[str, object]:
        captured = int(self._metrics["captured"])
        closed = int(self._metrics["closed"])
        return {
            **self._metrics,
            "queued": len(self._queue),
            "closure_rate": closed / captured if captured else 0.0,
        }
