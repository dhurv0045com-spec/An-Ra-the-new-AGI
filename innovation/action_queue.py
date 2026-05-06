from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

from anra_paths import AIE_ACTION_QUEUE
from innovation.schema import Hypothesis, InnovationScore


def queue_actions(
    hypotheses: Iterable[Hypothesis],
    scores: dict[str, InnovationScore],
    *,
    path: str | Path = AIE_ACTION_QUEUE,
    threshold: float = 80.0,
) -> list[dict]:
    queued: list[dict] = []
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for hyp in hypotheses:
            score = scores.get(hyp.hyp_id)
            if score is None or float(score.total) < threshold:
                continue
            action = {
                "action_id": f"aie_{hyp.hyp_id}_{int(time.time())}",
                "created_at": time.time(),
                "status": "queued_for_human_review",
                "decision": score.decision,
                "score": score.to_dict(),
                "hypothesis": hyp.to_dict(),
                "required_gate": "sovereignty_audit",
                "verifier_path": hyp.verifier_path,
            }
            f.write(json.dumps(action, sort_keys=True) + "\n")
            queued.append(action)
    return queued

