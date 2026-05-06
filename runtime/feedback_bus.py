from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from anra_paths import EPG_PATH, FAILURE_REPLAY_DATASET, FALSIFICATION_LEDGER_PATH


def _score(result: Any) -> float:
    if isinstance(result, (int, float)):
        return max(0.0, min(1.0, float(result)))
    if isinstance(result, dict):
        return max(0.0, min(1.0, float(result.get("score", result.get("confidence", 0.0)) or 0.0)))
    return max(0.0, min(1.0, float(getattr(result, "score", getattr(result, "confidence", 0.0)) or 0.0)))


def _reason(result: Any) -> str:
    if isinstance(result, dict):
        return str(result.get("reason", result.get("error", "")))
    return str(getattr(result, "reason", getattr(result, "stderr", "")) or "")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def record_verifier_feedback(
    *,
    prompt: str,
    response: str,
    verifier_result: Any,
    task_type: str = "open",
    hal: Any = None,
    memory_router: Any = None,
    epg_path: str | Path = EPG_PATH,
    ledger_path: str | Path = FALSIFICATION_LEDGER_PATH,
    replay_path: str | Path = FAILURE_REPLAY_DATASET,
) -> dict[str, Any]:
    score = _score(verifier_result)
    reason = _reason(verifier_result) or ("verifier failed" if score < 0.35 else "verifier passed")
    passed = score >= 0.35
    label = "VERIFIED" if score >= 0.75 else "FALSIFIED" if not passed else "INFERRED"

    replay_text = (
        f"<hyp>{prompt}</hyp>\n"
        f"<act>{json.dumps({'task_type': task_type, 'response': response}, sort_keys=True)}</act>\n"
        f"<obs>{json.dumps({'score': score, 'passed': passed, 'reason': reason}, sort_keys=True)}</obs>\n"
        f"<err>{'' if passed else reason}</err>\n"
        f"<upd>{'Rerun with a verifier-grounded correction before accepting the claim.' if not passed else 'Maintain verified behavior.'}</upd>\n"
        f"<verify>{label}</verify>"
    )

    nodes: dict[str, Any] = {}
    try:
        from memory.experimental_proof_graph import ExperimentalProofGraph

        epg = ExperimentalProofGraph(epg_path)
        nodes = epg.record_experiment(
            hypothesis={"prompt": prompt, "task_type": task_type},
            action={"response": response},
            observation={"score": score, "reason": reason, "passed": passed},
            correction={"text": replay_text, "label": label} if not passed else None,
            memory={"dfc": replay_text, "template": "FAILURE_REPLAY" if not passed else "VERIFIER_TRACE"},
        )
    except Exception:
        nodes = {}

    try:
        from identity.falsification_ledger import FalsificationLedger

        FalsificationLedger(ledger_path, memory_router=memory_router).append(
            prompt,
            status=label,
            confidence=score,
            evidence=[{"task_type": task_type, "score": score, "reason": reason}],
            would_be_false_if=reason if not passed else "",
            next_verifier=task_type,
        )
    except Exception:
        pass

    if not passed:
        record = {
            "prompt": prompt,
            "target": replay_text,
            "source": "failure_replay",
            "score": 1.0 - score,
            "weight": 1.0 + (1.0 - score),
            "metadata": {"task_type": task_type, "reason": reason, "label": label},
        }
        try:
            from training.replay_pipeline import ReplayPipeline

            pipe = ReplayPipeline.load(replay_path, max_size=8192)
            pipe.add(**record)
            pipe.save(replay_path)
        except Exception:
            _append_jsonl(Path(replay_path), record)
        if hal is not None:
            try:
                hal.update(
                    verifier_result=score,
                    session_context={
                        "task_type": task_type,
                        "model_incoherence_self_detected": True,
                        "near_capability_boundary": True,
                    },
                    decay_turns=0,
                )
                from runtime.hal_telemetry import publish_hal_state

                publish_hal_state(hal, source="verifier_feedback")
            except Exception:
                pass
        if memory_router is not None:
            try:
                memory_router.write(
                    replay_text,
                    metadata={"kind": "failure_replay", "salience": 1.0, "task_type": task_type},
                    tier="episodic",
                )
            except Exception:
                pass

    return {"score": score, "label": label, "passed": passed, "reason": reason, "nodes": list(nodes)}

