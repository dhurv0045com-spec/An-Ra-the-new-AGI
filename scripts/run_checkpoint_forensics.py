# ruff: noqa: E402
"""Stream A checkpoint forensics driver (MASTER_PLAN Stage 1.1).

Runs the full forensic sequence against the real 500M checkpoint and writes
one evidence artifact per run:

1. Locate the checkpoint (explicit arg -> ANRA_CHECKPOINT_PATH -> canonical
   path). The artifact never enters git; a miss is reported as blocked, not
   as a failure.
2. Exact tensor accounting via the existing frontier checkpoint proof.
3. The 500 tokenizer probes, cross-checked against the frozen manifest.
4. Deterministic generation (greedy, seed 0, KV cache off) through the
   200-prompt recovery gate.
5. The Part 0 decision rule: exact load with coherence <80% reads as
   undertraining -> proceed to tokenizer V4 and the token campaign.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anra.anra_paths import OUTPUT_V2_DIR, ROOT

from scripts.check_frontier_checkpoint import _checkpoint_report
from scripts.freeze_baseline_hashes import freeze_tokenizer, resolve_checkpoint

FORENSICS_REPORT = OUTPUT_V2_DIR / "stream_a_forensics.json"
COHERENCE_RECOVERY_GATE = 0.80


def _frontier_generator() -> Callable[[str, str, int, str | None], object]:
    """Deterministic greedy generator over the real checkpoint (heavy import)."""
    from generate import GenerationConfig, clear_session_runtime_state, generate_traced

    call_index = 0

    def generator(prompt: str, mode: str, seed: int, ablation: str | None) -> object:
        nonlocal call_index
        call_index += 1
        session_id = f"forensics_gate_{call_index:04d}"
        try:
            return generate_traced(
                prompt,
                GenerationConfig(
                    strategy="greedy",
                    max_tokens=64,
                    seed=seed,
                    use_kv_cache=False,
                    mode=mode,
                    ablated_subsystem=ablation,
                    persist_adaptive_state=False,
                ),
                session_id=session_id,
            )
        finally:
            clear_session_runtime_state(session_id)

    return generator


def run_forensics(
    checkpoint: Path,
    *,
    run_generation: bool = False,
    generator: Callable[[str, str, int, str | None], object] | None = None,
) -> dict[str, object]:
    steps: dict[str, dict[str, object]] = {}

    available = checkpoint.exists()
    steps["locate_checkpoint"] = {
        "status": "passed" if available else "blocked",
        "path": str(checkpoint),
        "note": None
        if available
        else "Real checkpoint not on disk; set ANRA_CHECKPOINT_PATH or restore "
        "it to the canonical path (kept outside git by policy).",
    }

    if available:
        try:
            proof = _checkpoint_report(checkpoint)
            steps["tensor_accounting"] = {
                "status": "passed" if proof["ok"] else "failed",
                "proof": proof,
            }
        except Exception as error:  # noqa: BLE001 - forensic evidence, not control flow
            steps["tensor_accounting"] = {"status": "failed", "error": str(error)}
    else:
        steps["tensor_accounting"] = {"status": "blocked", "reason": "no checkpoint"}

    tokenizer = freeze_tokenizer()
    probes_ok = bool(
        tokenizer.get("status") == "frozen"
        and tokenizer.get("probe_match_vs_manifest") is not False
    )
    steps["tokenizer_probes"] = {
        "status": "passed" if probes_ok else "failed",
        "probe_count": tokenizer.get("probe_count"),
        "probe_sha256": tokenizer.get("probe_sha256"),
        "probe_match_vs_manifest": tokenizer.get("probe_match_vs_manifest"),
    }

    coherence_rate: float | None = None
    if not available:
        steps["recovery_gate"] = {"status": "blocked", "reason": "no checkpoint"}
    elif not run_generation and generator is None:
        steps["recovery_gate"] = {
            "status": "skipped",
            "reason": "generation not requested (pass --run-generation)",
        }
    else:
        from training.eval_v2 import run_recovery_prompt_gate

        gate_generator = generator or _frontier_generator()
        gate_report = run_recovery_prompt_gate(gate_generator)
        candidate = gate_report.get("candidate", {})
        candidate = candidate if isinstance(candidate, dict) else {}
        coherence_rate = float(candidate.get("coherence_rate", 0.0))
        steps["recovery_gate"] = {
            "status": "passed" if coherence_rate >= COHERENCE_RECOVERY_GATE else "failed",
            "coherence_rate": coherence_rate,
            "gate": COHERENCE_RECOVERY_GATE,
            "report": gate_report,
        }

    exact_load = bool(
        steps["tensor_accounting"].get("status") == "passed"
    )
    if exact_load and coherence_rate is not None and coherence_rate < COHERENCE_RECOVERY_GATE:
        verdict = (
            "undertraining: exact load with coherence below 80% -> "
            "proceed to V4 + token campaign"
        )
    elif exact_load and coherence_rate is not None:
        verdict = "checkpoint loads exactly and clears the recovery gate"
    elif not available:
        verdict = "incomplete: forensics blocked on the real checkpoint artifact"
    else:
        verdict = "incomplete: exact tensor accounting and the recovery gate must both pass"

    statuses = {step["status"] for step in steps.values()}
    return {
        "schema_version": 1,
        "generated_at": time.time(),
        "checkpoint": str(checkpoint),
        "steps": steps,
        "verdict": verdict,
        "complete": statuses <= {"passed"},
        "blocked": "blocked" in statuses,
        "failed": "failed" in statuses,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stream A checkpoint forensics.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--json-out", default=str(FORENSICS_REPORT))
    parser.add_argument(
        "--run-generation",
        action="store_true",
        help="Load the model and run the 200-prompt deterministic recovery gate.",
    )
    args = parser.parse_args()

    report = run_forensics(
        resolve_checkpoint(args.checkpoint),
        run_generation=args.run_generation,
    )
    output = Path(args.json_out)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({k: v for k, v in report.items() if k != "steps"}, indent=2, sort_keys=True))
    if report["failed"]:
        return 2
    if report["blocked"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
