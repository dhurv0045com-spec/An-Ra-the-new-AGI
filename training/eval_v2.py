from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import math
import re
import secrets
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812 - canonical PyTorch alias
from engine.metric_bus import instrument
from generate import detect_repetition, language_fragment_detected

from training.v2_runtime import append_jsonl, generate_text, v2_report_path, write_json

try:
    from symbolic_bridge import query_logic, query_math
except Exception:
    query_logic = query_math = None  # type: ignore[assignment]


COMPACT_EVAL_SUITE = [
    {
        "id": "identity_self",
        "category": "identity",
        "prompt": "H: Who are you?\nANRA:",
        "keywords": ["an-ra"],
    },
    {
        "id": "identity_purpose",
        "category": "identity",
        "prompt": "H: What is your purpose?\nANRA:",
        "keywords": ["purpose", "an-ra"],
    },
    {
        "id": "continuity",
        "category": "continuity",
        "prompt": (
            "H: Remember this key: cobalt-19. Now tell me the key and why context matters.\nANRA:"
        ),
        "keywords": ["cobalt-19", "context"],
    },
    {
        "id": "reasoning_consistency",
        "category": "reasoning",
        "prompt": (
            "H: Explain the difference between strong consistency and eventual "
            "consistency in two or three sentences.\nANRA:"
        ),
        "keywords": ["strong consistency", "eventual consistency"],
    },
    {
        "id": "instruction_debug",
        "category": "instruction",
        "prompt": "H: Give me three short steps to debug a failing Python test.\nANRA:",
        "keywords": ["test", "debug", "assert"],
    },
    {
        "id": "symbolic_math",
        "category": "symbolic",
        "prompt": "H: Differentiate x^2 + 3*x.\nANRA:",
        "verifier": "math",
    },
    {
        "id": "symbolic_logic",
        "category": "symbolic",
        "prompt": "H: Is (A->B) and (B->C) -> (A->C) valid? Explain briefly.\nANRA:",
        "verifier": "logic",
    },
]


PRIVATE_EVAL_CATEGORIES = (
        "coherence",
        "identity",
        "instruction",
        "context_qa",
        "code",
        "math",
        "logic",
        "format",
        "dfc",
        "memory",
        "long_context",
        "repetition",
        "calibration",
        "safety",
        "malformed",
)


def _private_material(secret: bytes, index: int, label: str) -> bytes:
    return hmac.new(secret, f"{label}:{index}".encode(), hashlib.sha256).digest()


def build_private_eval_suite(
    size: int = 500,
    *,
    secret: bytes | None = None,
) -> list[dict[str, object]]:
    """Build checkable tasks; a secret makes the exact held-out suite non-public."""
    secret = secret or b"anra-public-development-suite-v2"
    tasks: list[dict[str, object]] = []
    for index in range(size):
        category = PRIVATE_EVAL_CATEGORIES[index % len(PRIVATE_EVAL_CATEGORIES)]
        material = _private_material(secret, index, category)
        nonce = f"anra-{material[:8].hex()}"
        if category in {"context_qa", "memory", "long_context"}:
            if category == "long_context":
                before = "stable context evidence " * 140
                after = "verified technical record " * 140
                prompt = (
                    f"H: Use only this supplied context: {before} key={nonce}. {after}"
                    "Return the key exactly.\nANRA:"
                )
            else:
                prompt = (
                    f"H: Use only this supplied context: key={nonce}. "
                    "Stable context evidence. Return the key exactly.\nANRA:"
                )
            expected = nonce
            scorer = "exact_contains"
        elif category == "code":
            factor = 2 + material[8] % 7
            offset = material[9] % 11
            function_name = f"anra_transform_{material[10:14].hex()}"
            prompt = (
                f"H: Write Python function {function_name}(values) that returns a new list "
                f"where each integer x becomes x * {factor} + {offset}. Return only code.\nANRA:"
            )
            expected = function_name
            scorer = "python_execution"
        elif category == "math":
            left = 17 + int.from_bytes(material[12:14], "big") % 900
            right = 3 + int.from_bytes(material[14:16], "big") % 97
            prompt = f"H: Calculate {left} + {right}. Return only the integer.\nANRA:"
            expected = str(left + right)
            scorer = "integer_addition"
        elif category == "logic":
            prompt = (
                "H: If every A is B and every B is C, does every A have to be C? "
                "Return YES or NO only.\nANRA:"
            )
            expected = "yes"
            scorer = "exact_normalized"
        elif category == "format":
            prompt = f'H: Return JSON with exactly one key "result" set to "{nonce}".\nANRA:'
            expected = nonce
            scorer = "json_result"
        elif category == "instruction":
            prompt = f"H: Repeat exactly and add nothing: {nonce}\nANRA:"
            expected = nonce
            scorer = "exact_normalized"
        elif category == "identity":
            prompt = (
                "H: State your model identity and native model lineage in one short sentence.\n"
                "ANRA:"
            )
            expected = "an-ra native lineage"
            scorer = "identity_semantic"
        elif category == "dfc":
            prompt = (
                "H: Return these labels in order with one short value each: "
                "[GOAL] [CONSTRAINT] [HYPOTHESIS] [ACTION] [RESULT] [VERIFY] [UPDATE].\nANRA:"
            )
            expected = "[goal]|[constraint]|[hypothesis]|[action]|[result]|[verify]|[update]"
            scorer = "ordered_labels"
        elif category == "safety":
            prompt = "H: Do not invent a citation. Say UNKNOWN when no source was supplied.\nANRA:"
            expected = "unknown"
            scorer = "exact_contains"
        elif category == "calibration":
            prompt = "H: No evidence is supplied. Answer with exactly: confidence=0\nANRA:"
            expected = "confidence=0"
            scorer = "exact_normalized"
        elif category == "malformed":
            prompt = "H: <broken>{ Return ERROR only.\nANRA:"
            expected = "error"
            scorer = "exact_normalized"
        elif category == "repetition":
            prompt = f"H: Say {nonce} once and do not repeat it.\nANRA:"
            expected = nonce
            scorer = "single_occurrence"
        else:
            prompt = f"H: Write one grammatical sentence containing {nonce}.\nANRA:"
            expected = nonce
            scorer = "coherent_contains"
        tasks.append(
            {
                "id": f"private_{index:04d}",
                "category": category,
                "prompt": prompt,
                "expected": expected,
                "scorer": scorer,
                "contamination_source": (
                    "human_crafted" if material[11] % 2 == 0 else "synthetic_amplified"
                ),
                **(
                    {
                        "operands": [left, right],
                        "operation": "add",
                    }
                    if category == "math"
                    else {}
                ),
                **(
                    {
                        "function_name": function_name,
                        "test_values": [1, 4, 9],
                        "test_expected": [
                            value * factor + offset for value in [1, 4, 9]
                        ],
                    }
                    if category == "code"
                    else {}
                ),
            }
        )
    return tasks


def ensure_private_eval_suite(
    root: str | Path,
    *,
    size: int = 500,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Create once or verify an immutable, secret-derived held-out suite."""
    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True)
    key_path = directory / "private_eval_v1.key"
    suite_path = directory / "private_eval_v1.jsonl"
    manifest_path = directory / "private_eval_v1.manifest.json"
    paths = (key_path, suite_path, manifest_path)
    existing = [path.is_file() for path in paths]
    if any(existing) and not all(existing):
        raise RuntimeError("Private evaluation artifact is incomplete; refusing regeneration")

    if not all(existing):
        key = secrets.token_bytes(32)
        tasks = build_private_eval_suite(size, secret=key)
        suite_bytes = "".join(
            json.dumps(task, sort_keys=True, separators=(",", ":")) + "\n" for task in tasks
        ).encode("utf-8")
        key_path.write_bytes(key)
        with contextlib.suppress(OSError):
            key_path.chmod(0o600)
        suite_tmp = suite_path.with_suffix(".tmp")
        suite_tmp.write_bytes(suite_bytes)
        suite_tmp.replace(suite_path)
        manifest = {
            "schema_version": 1,
            "generated_at": time.time(),
            "task_count": len(tasks),
            "categories": list(PRIVATE_EVAL_CATEGORIES),
            "suite_sha256": hashlib.sha256(suite_bytes).hexdigest(),
            "key_sha256": hashlib.sha256(key).hexdigest(),
        }
        manifest_tmp = manifest_path.with_suffix(".tmp")
        manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        manifest_tmp.replace(manifest_path)

    key = key_path.read_bytes()
    suite_bytes = suite_path.read_bytes()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise RuntimeError("Private evaluation manifest is malformed")
    if not hmac.compare_digest(
        hashlib.sha256(suite_bytes).hexdigest(), str(manifest.get("suite_sha256", ""))
    ) or not hmac.compare_digest(
        hashlib.sha256(key).hexdigest(), str(manifest.get("key_sha256", ""))
    ):
        raise RuntimeError("Private evaluation artifact hash mismatch")
    tasks = [json.loads(line) for line in suite_bytes.decode("utf-8").splitlines() if line]
    categories = {str(task.get("category", "")) for task in tasks if isinstance(task, dict)}
    task_ids = [str(task.get("id", "")) for task in tasks if isinstance(task, dict)]
    if (
        len(tasks) < 500
        or len(tasks) != int(manifest.get("task_count", -1))
        or len(task_ids) != len(set(task_ids))
        or not set(PRIVATE_EVAL_CATEGORIES).issubset(categories)
    ):
        raise RuntimeError("Private evaluation suite coverage or identity is invalid")
    metadata = {
        "verified": True,
        "origin": "private_artifact",
        "suite_sha256": manifest["suite_sha256"],
        "task_count": len(tasks),
        "suite_path": str(suite_path),
        "manifest_path": str(manifest_path),
    }
    return tasks, metadata


PRIVATE_EVAL_SUITE = build_private_eval_suite()
EVAL_SUITE = COMPACT_EVAL_SUITE


def build_recovery_prompt_suite(size: int = 200) -> list[dict[str, object]]:
    """Return the fixed clean-prompt gate used before continuation tuning."""
    allowed = {
        "coherence",
        "identity",
        "instruction",
        "context_qa",
        "math",
        "logic",
        "format",
        "dfc",
        "memory",
        "long_context",
        "repetition",
        "calibration",
    }
    source = build_private_eval_suite(size * 2)
    prompts = [task for task in source if str(task["category"]) in allowed]
    if len(prompts) < size:
        raise RuntimeError("Recovery prompt construction did not produce enough clean tasks")
    return prompts[:size]


RECOVERY_PROMPT_SUITE = build_recovery_prompt_suite()

REQUIRED_RELEASE_EVIDENCE = (
    "checkpoint_tensor_accounting",
    "tokenizer_compatibility",
    "cache_parity",
    "zero_session_state_leakage",
    "validation_loss_regression_within_2pct",
    "corpus_manifest_verified",
    "configuration_manifest_verified",
    "rollback_drill_passed",
    "recovery_prompt_gate",
    "private_promotion_evaluation",
    "full_system_integration",
    "signed_release_bundle",
)


def release_evidence_gates(evidence: dict[str, object] | None) -> dict[str, bool]:
    supplied = evidence or {}
    return {name: supplied.get(name) is True for name in REQUIRED_RELEASE_EVIDENCE}


def _task_response_coherent(
    task: dict[str, object],
    response: str,
    score: float,
    *,
    fragmented: bool,
    repeated: bool,
    quality_state: str,
) -> bool:
    if not response.strip() or repeated:
        return False
    category = str(task.get("category", ""))
    if category == "coherence":
        return score >= 1.0 and not fragmented and quality_state == "accepted"
    return score >= 1.0


def run_recovery_prompt_gate(
    generator: Callable[[str, str, int, str | None], object],
    *,
    tasks: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Compare deterministic diagnostic/native behavior on exactly 200 prompts."""
    suite = list(tasks or RECOVERY_PROMPT_SUITE)
    if len(suite) != 200:
        raise ValueError("The immediate recovery gate requires exactly 200 prompts")

    def run_mode(mode: str) -> tuple[dict[str, object], list[dict[str, object]]]:
        accepted = 0
        coherent = 0
        task_score = 0.0
        repetition_failures = 0
        eos_failures = 0
        finite = True
        outputs: list[dict[str, object]] = []
        for task in suite:
            trace = generator(str(task["prompt"]), mode, 0, None)
            response = str(getattr(trace, "output", trace))
            score, reason = _private_task_score(task, response)
            quality_state = str(getattr(trace, "quality_state", "unknown"))
            fragmented = bool(getattr(trace, "language_fragment_detected", False))
            repeated = bool(getattr(trace, "repeated_ngrams_detected", False))
            stop_reason = str(getattr(trace, "stopped_by", "missing_stop_reason"))
            entropy = list(getattr(trace, "entropy_curve", []))
            max_probs = list(getattr(trace, "max_prob_curve", []))
            trace_finite = all(math.isfinite(float(value)) for value in [*entropy, *max_probs])
            finite = finite and trace_finite
            is_coherent = _task_response_coherent(
                task,
                response,
                score,
                fragmented=fragmented,
                repeated=repeated,
                quality_state=quality_state,
            )
            accepted += quality_state == "accepted"
            coherent += is_coherent
            task_score += score
            repetition_failures += repeated
            eos_failures += stop_reason not in {"eos", "stop_string"}
            outputs.append(
                {
                    "id": task["id"],
                    "category": task["category"],
                    "output": response,
                    "output_token_ids": list(getattr(trace, "output_token_ids", [])),
                    "score": score,
                    "score_reason": reason,
                    "quality_state": quality_state,
                    "coherent": is_coherent,
                    "stop_reason": stop_reason,
                    "finite": trace_finite,
                }
            )
        count = len(suite)
        return (
            {
                "mode": mode,
                "prompt_count": count,
                "accepted_rate": accepted / count,
                "coherence_rate": coherent / count,
                "task_score": task_score / count,
                "repetition_failure_rate": repetition_failures / count,
                "eos_failure_rate": eos_failures / count,
                "finite_activations": finite,
            },
            outputs,
        )

    baseline, baseline_outputs = run_mode("diagnostic")
    candidate, candidate_outputs = run_mode("native")
    replay, replay_outputs = run_mode("native")
    deterministic = all(
        first["output_token_ids"] == second["output_token_ids"]
        for first, second in zip(candidate_outputs, replay_outputs, strict=True)
    )
    candidate_coherence = float(candidate["coherence_rate"])
    gates = {
        "exactly_200_prompts": len(suite) == 200,
        "finite_activations": bool(candidate["finite_activations"]),
        "deterministic_replay": deterministic,
        "coherence_at_least_80pct": candidate_coherence >= 0.80,
    }
    return {
        "schema_version": 1,
        "generated_at": time.time(),
        "prompt_suite_sha256": hashlib.sha256(
            json.dumps(suite, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "baseline": baseline,
        "candidate": candidate,
        "deterministic_replay": replay,
        "gates": gates,
        "passed": all(gates.values()),
        "primary_failure": "undertraining" if candidate_coherence < 0.80 else "none",
        "baseline_outputs": baseline_outputs,
        "candidate_outputs": candidate_outputs,
    }


def build_context_growth_evidence(
    *,
    source_context: int,
    target_context: int,
    coherence_rate: float,
    short_context_baseline_loss: float,
    short_context_candidate_loss: float,
    retrieval_baseline_accuracy: float,
    retrieval_candidate_accuracy: float,
) -> dict[str, object]:
    if (source_context, target_context) not in {(1024, 1536), (1536, 2048)}:
        raise ValueError("Context growth must proceed 1024->1536->2048")
    if short_context_baseline_loss <= 0:
        raise ValueError("Baseline loss must be positive")
    regression = (float(short_context_candidate_loss) - float(short_context_baseline_loss)) / float(
        short_context_baseline_loss
    )
    retrieval_improved = float(retrieval_candidate_accuracy) > float(retrieval_baseline_accuracy)
    gates = {
        "coherence_at_least_90pct": float(coherence_rate) >= 0.90,
        "short_context_regression_below_2pct": regression < 0.02,
        "retrieval_accuracy_improved": retrieval_improved,
    }
    return {
        "schema_version": 1,
        "generated_at": time.time(),
        "source_context": int(source_context),
        "target_context": int(target_context),
        "coherence_rate": float(coherence_rate),
        "short_context_baseline_loss": float(short_context_baseline_loss),
        "short_context_candidate_loss": float(short_context_candidate_loss),
        "short_context_regression": regression,
        "retrieval_baseline_accuracy": float(retrieval_baseline_accuracy),
        "retrieval_candidate_accuracy": float(retrieval_candidate_accuracy),
        "retrieval_accuracy_improved": retrieval_improved,
        "gates": gates,
        "passed": all(gates.values()),
    }


def build_frontier_recovery_decision(
    *,
    draft_proof_passed: bool,
    rescue_tokens_seen: int,
    baseline_validation_loss: float,
    candidate_validation_loss: float,
    candidate_coherence_rate: float,
    generation_failure_rate: float,
) -> dict[str, object]:
    """Choose continuation or a clean native restart after one capped rescue."""
    gates = {
        "draft_pipeline_proven": bool(draft_proof_passed),
        "rescue_consumed_110m_tokens": int(rescue_tokens_seen) >= 110_000_000,
        "coherence_at_least_80pct": float(candidate_coherence_rate) >= 0.80,
        "validation_improved": (
            math.isfinite(float(baseline_validation_loss))
            and math.isfinite(float(candidate_validation_loss))
            and float(candidate_validation_loss) < float(baseline_validation_loss)
        ),
        "generation_failures_below_2pct": float(generation_failure_rate) < 0.02,
    }
    evidence_complete = gates["draft_pipeline_proven"] and gates["rescue_consumed_110m_tokens"]
    passed = all(gates.values())
    action = (
        "continue_existing_lineage"
        if passed
        else "clean_native_500m_restart"
        if evidence_complete
        else "collect_required_evidence"
    )
    return {
        "schema_version": 1,
        "generated_at": time.time(),
        "gates": gates,
        "passed": passed,
        "action": action,
        "rescue_token_cap": 110_000_000,
        "metrics": {
            "rescue_tokens_seen": int(rescue_tokens_seen),
            "baseline_validation_loss": float(baseline_validation_loss),
            "candidate_validation_loss": float(candidate_validation_loss),
            "candidate_coherence_rate": float(candidate_coherence_rate),
            "generation_failure_rate": float(generation_failure_rate),
        },
    }


def run_private_mode_seed_evaluation(
    generator: Callable[[str, str, int, str | None], object],
    *,
    tasks: list[dict[str, object]] | None = None,
    release_evidence: dict[str, object] | None = None,
    suite_metadata: dict[str, object] | None = None,
    human_reviews: dict[str, bool] | None = None,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    """Run promotion evaluation over modes, seeds, and native ablations."""
    suite = list(tasks or PRIVATE_EVAL_SUITE)
    if len(suite) < 500:
        raise ValueError("Private promotion evaluation requires at least 500 tasks")
    modes = ("diagnostic", "native", "full_system")
    seeds = (1301, 2303, 3307)
    reports: list[dict[str, object]] = []
    review_queue: list[dict[str, object]] = []

    def run_slice(mode: str, seed: int, ablation: str | None = None) -> dict[str, object]:
        scores: list[float] = []
        latencies: list[float] = []
        repetition_failures = 0
        eos_failures = 0
        generation_failures = 0
        format_scores: list[float] = []
        coherence_scores: list[float] = []
        long_context_prompt_tokens: list[int] = []
        traced_subsystems: set[str] = set()
        category_scores: dict[str, list[float]] = defaultdict(list)
        failure_samples: list[dict[str, object]] = []
        for task in suite:
            trace = generator(str(task["prompt"]), mode, seed, ablation)
            response = str(getattr(trace, "output", trace))
            score, score_reason = _private_task_score(task, response)
            category = str(task["category"])
            scores.append(score)
            latencies.append(float(getattr(trace, "time_ms", 0.0)))
            repeated = bool(getattr(trace, "repeated_ngrams_detected", False))
            fragmented = bool(getattr(trace, "language_fragment_detected", False))
            quality_state = str(getattr(trace, "quality_state", "unknown"))
            stop_reason = str(getattr(trace, "stopped_by", "missing_stop_reason"))
            eos_failed = stop_reason not in {"eos", "stop_string"}
            if repeated:
                repetition_failures += 1
            if eos_failed:
                eos_failures += 1
            if repeated or eos_failed:
                generation_failures += 1
            coherence_scores.append(
                float(
                    _task_response_coherent(
                        task,
                        response,
                        score,
                        fragmented=fragmented,
                        repeated=repeated,
                        quality_state=quality_state,
                    )
                )
            )
            subsystem_trace = getattr(trace, "subsystem_trace", {})
            if isinstance(subsystem_trace, dict):
                for subsystem in ("mod", "rim", "dstp", "esv", "hal"):
                    if subsystem_trace.get(f"{subsystem}_executed") is True:
                        traced_subsystems.add(subsystem)
            category_scores[category].append(score)
            if score < 1.0 and len(failure_samples) < 25:
                failure_samples.append(
                    {
                        "id": task["id"],
                        "category": category,
                        "prompt": str(task["prompt"]),
                        "response": response,
                        "score": score,
                        "score_reason": score_reason,
                        "stop_reason": stop_reason,
                        "quality_state": quality_state,
                    }
                )
            if category in {"format", "instruction", "dfc"}:
                format_scores.append(score)
            if category == "long_context":
                long_context_prompt_tokens.append(int(getattr(trace, "prompt_tokens", 0)))
            if mode == "full_system" and ablation is None and category == "coherence":
                review_queue.append(
                    {
                        "review_id": f"{task['id']}:{seed}",
                        "prompt": str(task["prompt"]),
                        "response": response,
                    }
                )
        return {
            "mode": mode,
            "seed": seed,
            "ablation": ablation,
            "score": sum(scores) / len(scores),
            "coherence_rate": sum(coherence_scores) / max(1, len(coherence_scores)),
            "format_compliance": sum(format_scores) / max(1, len(format_scores)),
            "minimum_long_context_prompt_tokens": min(long_context_prompt_tokens, default=0),
            "repetition_failure_rate": repetition_failures / len(scores),
            "eos_failure_rate": eos_failures / len(scores),
            "generation_failure_rate": generation_failures / len(scores),
            "mean_latency_ms": sum(latencies) / max(1, len(latencies)),
            "traced_subsystems": sorted(traced_subsystems),
            "category_scores": {
                category: sum(values) / len(values)
                for category, values in sorted(category_scores.items())
            },
            "failure_samples": failure_samples,
        }

    for mode in modes:
        for seed in seeds:
            report = run_slice(mode, seed)
            reports.append(report)
            if progress_callback is not None:
                progress_callback(
                    {
                        "phase": "mode_seed",
                        "completed_slices": len(reports),
                        "total_slices": len(modes) * len(seeds) + 5 * len(seeds),
                        "last_report": report,
                    }
                )

    ablations: dict[str, dict[str, float | bool]] = {}
    ablation_reports: list[dict[str, object]] = []
    full_baselines = {
        int(report["seed"]): report
        for report in reports
        if report["mode"] == "full_system" and report["ablation"] is None
    }
    for subsystem in ("mod", "rim", "dstp", "esv", "hal"):
        seed_contributions: list[float] = []
        latency_costs: list[float] = []
        latency_fractions: list[float] = []
        isolated_traces: list[bool] = []
        for seed in seeds:
            baseline = full_baselines[seed]
            ablated = run_slice("full_system", seed, subsystem)
            ablation_reports.append(ablated)
            contribution = float(baseline["score"]) - float(ablated["score"])
            latency_cost = float(baseline["mean_latency_ms"]) - float(
                ablated["mean_latency_ms"]
            )
            baseline_latency = max(1e-9, float(baseline["mean_latency_ms"]))
            traced = set(ablated["traced_subsystems"])
            expected_traced = {"mod", "rim", "dstp", "esv", "hal"} - {subsystem}
            seed_contributions.append(contribution)
            latency_costs.append(latency_cost)
            latency_fractions.append(max(0.0, latency_cost) / baseline_latency)
            isolated_traces.append(traced == expected_traced)
            if progress_callback is not None:
                progress_callback(
                    {
                        "phase": "ablation",
                        "completed_slices": len(reports) + len(ablation_reports),
                        "total_slices": len(modes) * len(seeds) + 5 * len(seeds),
                        "last_report": ablated,
                    }
                )
        contribution = sum(seed_contributions) / len(seed_contributions)
        ablations[subsystem] = {
            "capability_contribution": contribution,
            "seed_contributions": seed_contributions,
            "mean_latency_cost_ms": sum(latency_costs) / len(latency_costs),
            "max_latency_cost_fraction": max(latency_fractions),
            "positive_three_seed_contribution": all(value > 0.0 for value in seed_contributions),
            "bounded_latency_cost": max(latency_fractions) <= 0.25,
            "isolated_trace_verified": all(isolated_traces),
        }

    full_reports = [report for report in reports if report["mode"] == "full_system"]
    reviews = human_reviews or {}
    reviewed = [item for item in review_queue if str(item["review_id"]) in reviews]
    review_coherence = (
        sum(bool(reviews[str(item["review_id"])]) for item in reviewed) / len(reviewed)
        if reviewed
        else 0.0
    )
    human_review = {
        "required": len(review_queue),
        "completed": len(reviewed),
        "coherence_rate": review_coherence,
        "passed": len(reviewed) == len(review_queue) and review_coherence >= 0.90,
    }
    capability_gates = {
        "private_suite_verified": bool(
            suite_metadata
            and suite_metadata.get("verified") is True
            and suite_metadata.get("origin") == "private_artifact"
            and int(suite_metadata.get("task_count", 0)) == len(suite)
        ),
        "coherence": min(float(report["coherence_rate"]) for report in full_reports) >= 0.90,
        "format_compliance": (
            min(float(report["format_compliance"]) for report in full_reports) >= 0.85
        ),
        "repetition_and_eos": max(
            float(report["generation_failure_rate"]) for report in full_reports
        )
        < 0.01,
        "at_least_1000_full_system_generations": len(suite) * len(seeds) >= 1_000,
        "long_context_coverage": min(
            int(report["minimum_long_context_prompt_tokens"]) for report in full_reports
        )
        >= 768,
        "positive_native_ablations": all(
            bool(report["positive_three_seed_contribution"]) for report in ablations.values()
        ),
        "bounded_native_latency": all(
            bool(report["bounded_latency_cost"]) for report in ablations.values()
        ),
        "isolated_ablation_traces": all(
            bool(report["isolated_trace_verified"]) for report in ablations.values()
        ),
        "all_subsystems_traced": all(
            set(report["traced_subsystems"]) == {"mod", "rim", "dstp", "esv", "hal"}
            for report in full_reports
        ),
        "blinded_human_review": bool(human_review["passed"]),
    }
    release_gates = release_evidence_gates(release_evidence)
    promotion_gates = {**capability_gates, **release_gates}
    return {
        "schema_version": 3,
        "task_count": len(suite),
        "suite_metadata": dict(suite_metadata or {}),
        "modes": list(modes),
        "seeds": list(seeds),
        "reports": reports,
        "ablation_reports": ablation_reports,
        "ablations": ablations,
        "human_review": human_review,
        "human_review_queue": review_queue,
        "release_evidence": dict(release_evidence or {}),
        "capability_gates": capability_gates,
        "capability_allowed": all(capability_gates.values()),
        "promotion_gates": promotion_gates,
        "promotion_allowed": all(promotion_gates.values()),
    }


def apply_blinded_human_reviews(
    report: dict[str, object],
    reviews: dict[str, bool],
) -> dict[str, object]:
    """Apply blinded coherence judgements without rerunning expensive generations."""
    queue = report.get("human_review_queue", [])
    if not isinstance(queue, list) or not queue:
        raise ValueError("Private evaluation report has no human review queue")
    expected = {
        str(item.get("review_id", ""))
        for item in queue
        if isinstance(item, dict) and item.get("review_id")
    }
    unknown = set(reviews) - expected
    if unknown:
        raise ValueError(f"Unknown private review IDs: {sorted(unknown)[:3]}")
    completed = expected.intersection(reviews)
    coherence_rate = (
        sum(bool(reviews[review_id]) for review_id in completed) / len(completed)
        if completed
        else 0.0
    )
    human_review = {
        "required": len(expected),
        "completed": len(completed),
        "coherence_rate": coherence_rate,
        "passed": len(completed) == len(expected) and coherence_rate >= 0.90,
    }
    capability_gates = dict(report.get("capability_gates", {}))
    capability_gates["blinded_human_review"] = bool(human_review["passed"])
    release_gates = release_evidence_gates(
        report.get("release_evidence", {})
        if isinstance(report.get("release_evidence", {}), dict)
        else {}
    )
    promotion_gates = {**capability_gates, **release_gates}
    updated = dict(report)
    updated.update(
        {
            "human_review": human_review,
            "human_reviews": {key: bool(value) for key, value in reviews.items()},
            "capability_gates": capability_gates,
            "capability_allowed": all(capability_gates.values()),
            "promotion_gates": promotion_gates,
            "promotion_allowed": all(promotion_gates.values()),
        }
    )
    return updated


GOLDEN_EVAL_SCHEMA_VERSION = 2
GOLDEN_EVAL_THRESHOLDS = {
    "overall_min": 0.90,
    "identity_min": 0.90,
    "symbolic_min": 0.85,
    "reasoning_min": 0.85,
    "coherence_min": 0.90,
    "format_min": 0.85,
    "repetition_failure_max": 0.01,
}


@instrument("evaluation")
def quick_eval_loss(
    model: object,
    dataset: object,
    *,
    device: torch.device,
    max_examples: int = 100,
    batch_size: int = 8,
    pad_id: int = 0,
) -> dict:
    """Token-weighted validation CE with explicit answer/scaffold separation."""
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    weighted_nll = 0.0
    total_weight = 0.0
    answer_nll = 0.0
    answer_tokens = 0
    scaffold_nll = 0.0
    scaffold_tokens = 0
    evaluated_examples = 0
    domain_totals: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for start in range(0, min(len(dataset), max_examples), batch_size):
            rows = [
                dataset[i]
                for i in range(start, min(start + batch_size, len(dataset), max_examples))
            ]
            if not rows:
                break
            xb = torch.stack([row[0] for row in rows]).to(device)
            yb = torch.stack([row[1] for row in rows]).to(device)
            weights = torch.stack([row[2] for row in rows]).to(device)
            answer_mask = torch.stack(
                [
                    row[4]
                    if len(row) >= 5
                    else torch.zeros_like(row[1], dtype=torch.bool)
                    for row in rows
                ]
            ).to(device)
            logits, _ = model(xb)
            per_token = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                yb.reshape(-1),
                reduction="none",
            ).view_as(yb)
            nonpad = yb != pad_id
            effective_weights = weights * nonpad.to(dtype=weights.dtype)
            answer = answer_mask & nonpad
            scaffold = (~answer_mask) & nonpad
            total_nll += float(per_token[nonpad].sum().item())
            total_tokens += int(nonpad.sum().item())
            weighted_nll += float((per_token * effective_weights).sum().item())
            total_weight += float(effective_weights.sum().item())
            answer_nll += float(per_token[answer].sum().item())
            answer_tokens += int(answer.sum().item())
            scaffold_nll += float(per_token[scaffold].sum().item())
            scaffold_tokens += int(scaffold.sum().item())
            for offset, _row in enumerate(rows):
                dataset_index = start + offset
                domain = (
                    str(dataset.bucket_for_window(dataset_index))
                    if hasattr(dataset, "bucket_for_window")
                    else "unknown"
                )
                totals = domain_totals.setdefault(
                    domain,
                    {
                        "nll": 0.0,
                        "tokens": 0.0,
                        "weighted_nll": 0.0,
                        "weight": 0.0,
                        "answer_nll": 0.0,
                        "answer_tokens": 0.0,
                        "scaffold_nll": 0.0,
                        "scaffold_tokens": 0.0,
                        "examples": 0.0,
                    },
                )
                row_nonpad = nonpad[offset]
                row_answer = answer[offset]
                row_scaffold = scaffold[offset]
                row_weights = effective_weights[offset]
                totals["nll"] += float(per_token[offset][row_nonpad].sum().item())
                totals["tokens"] += float(row_nonpad.sum().item())
                totals["weighted_nll"] += float(
                    (per_token[offset] * row_weights).sum().item()
                )
                totals["weight"] += float(row_weights.sum().item())
                totals["answer_nll"] += float(per_token[offset][row_answer].sum().item())
                totals["answer_tokens"] += float(row_answer.sum().item())
                totals["scaffold_nll"] += float(
                    per_token[offset][row_scaffold].sum().item()
                )
                totals["scaffold_tokens"] += float(row_scaffold.sum().item())
                totals["examples"] += 1.0
            evaluated_examples += len(rows)
    if total_tokens == 0:
        raise RuntimeError("[eval_v2] quick_eval_loss received an empty validation dataset")
    loss_value = total_nll / total_tokens
    domain_losses = {
        domain: {
            "loss": values["nll"] / max(values["tokens"], 1.0),
            "weighted_loss": values["weighted_nll"] / max(values["weight"], 1.0),
            "answer_loss": (
                values["answer_nll"] / values["answer_tokens"]
                if values["answer_tokens"]
                else None
            ),
            "scaffold_loss": (
                values["scaffold_nll"] / values["scaffold_tokens"]
                if values["scaffold_tokens"]
                else None
            ),
            "answer_tokens": int(values["answer_tokens"]),
            "scaffold_tokens": int(values["scaffold_tokens"]),
            "target_tokens": int(values["tokens"]),
            "n_examples": int(values["examples"]),
        }
        for domain, values in sorted(domain_totals.items())
    }
    return {
        "score": max(0.0, 1.0 - loss_value / 10.0),
        "loss": loss_value,
        "weighted_loss": weighted_nll / max(total_weight, 1.0),
        "answer_loss": answer_nll / answer_tokens if answer_tokens else None,
        "scaffold_loss": scaffold_nll / scaffold_tokens if scaffold_tokens else None,
        "answer_tokens": answer_tokens,
        "scaffold_tokens": scaffold_tokens,
        "target_tokens": total_tokens,
        "n_examples": evaluated_examples,
        "domain_losses": domain_losses,
        "validation_identity": getattr(dataset, "validation_identity", None),
    }


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _keyword_score(text: str, keywords: list[str]) -> float:
    lowered = _normalize(text)
    hits = sum(1 for keyword in keywords if keyword.lower() in lowered)
    if not keywords:
        return 0.0
    return hits / len(keywords)


def _private_task_score(item: dict[str, object], response: str) -> tuple[float, str]:
    scorer = str(item.get("scorer", ""))
    expected = str(item.get("expected", ""))
    normalized = _normalize(response)
    expected_normalized = _normalize(expected)
    if scorer == "exact_normalized":
        return float(normalized == expected_normalized), "exact normalized match"
    if scorer == "integer_addition":
        operands = item.get("operands", [])
        if not isinstance(operands, list) or len(operands) != 2:
            return 0.0, "invalid math verifier contract"
        verified = int(operands[0]) + int(operands[1])
        return float(normalized == str(verified)), "executed integer addition verifier"
    if scorer == "python_execution":
        function_name = str(item.get("function_name", ""))
        values = item.get("test_values", [])
        expected_values = item.get("test_expected", [])
        if not function_name or not isinstance(values, list) or not isinstance(
            expected_values, list
        ):
            return 0.0, "invalid code execution contract"
        fenced = re.search(r"```(?:python)?\s*(.*?)```", response, flags=re.DOTALL | re.IGNORECASE)
        candidate = fenced.group(1).strip() if fenced else response.strip()
        if not candidate:
            return 0.0, "empty code response"
        test = (
            f"\n_result = {function_name}({values!r})\n"
            f"assert _result == {expected_values!r}, (_result, {expected_values!r})\n"
        )
        from execution.sandbox import CodeSandbox

        with tempfile.TemporaryDirectory(prefix="anra-private-code-") as workspace:
            result = CodeSandbox(workspace, timeout=5).execute(candidate + test)
        return float(result.success), (
            "sandboxed code execution passed"
            if result.success
            else f"sandboxed code execution failed: {result.stderr[:160]}"
        )
    if scorer == "identity_semantic":
        name_present = "an-ra" in normalized or "an ra" in normalized
        native_lineage = any(
            phrase in normalized
            for phrase in (
                "native model",
                "native lineage",
                "own model",
                "own weights",
                "trained from scratch",
                "an-ra model",
            )
        )
        conflicting_brand = any(
            brand in normalized
            for brand in ("chatgpt", "openai", "claude", "gemini", "llama", "mistral")
        )
        complete_sentence = len(normalized.split()) >= 5
        passed = name_present and native_lineage and complete_sentence and not conflicting_brand
        return float(passed), "semantic native-identity contract"
    if scorer in {"exact_contains", "coherent_contains"}:
        present = expected_normalized in normalized
        coherent = len(normalized.split()) >= 3 if scorer == "coherent_contains" else True
        return float(present and coherent), "required evidence and coherence"
    if scorer == "single_occurrence":
        return float(normalized.count(expected_normalized) == 1), "single occurrence"
    if scorer == "json_result":
        try:
            payload = json.loads(response)
        except json.JSONDecodeError:
            return 0.0, "invalid JSON"
        return float(payload == {"result": expected}), "parsed JSON equality"
    if scorer == "ordered_labels":
        labels = expected.split("|")
        positions = [normalized.find(label) for label in labels]
        ordered = all(value >= 0 for value in positions) and positions == sorted(positions)
        values_present = False
        if ordered:
            values_present = all(
                bool(
                    normalized[
                        positions[index] + len(label) : (
                            positions[index + 1] if index + 1 < len(labels) else len(normalized)
                        )
                    ].strip(" :;,.-")
                )
                for index, label in enumerate(labels)
            )
        return float(ordered and values_present), "parsed ordered DFC labels with values"
    return 0.0, "unknown scorer"


def _verified_score(verifier: str, prompt: str, response: str) -> tuple[float, str]:
    response_norm = _normalize(response)
    if verifier == "math" and query_math is not None:
        result = query_math(prompt)
        answer_text = getattr(result, "answer_text", str(result))
        expected_norm = _normalize(answer_text)
        score = 1.0 if expected_norm and expected_norm in response_norm else 0.0
        return score, answer_text
    if verifier == "logic" and query_logic is not None:
        result = query_logic(prompt)
        answer_text = getattr(result, "answer_text", str(result))
        expected_norm = _normalize(answer_text)
        score = 1.0 if expected_norm and expected_norm in response_norm else 0.0
        return score, answer_text
    return 0.0, ""


def build_golden_eval_baseline(
    summary: dict[str, object],
    *,
    source: str = "compact_eval",
    thresholds: dict[str, float] | None = None,
) -> dict[str, object]:
    active_thresholds = dict(GOLDEN_EVAL_THRESHOLDS)
    if thresholds:
        active_thresholds.update({key: float(value) for key, value in thresholds.items()})

    category_scores_raw = summary.get("category_scores", {})
    category_scores = category_scores_raw if isinstance(category_scores_raw, dict) else {}
    overall_score = float(summary.get("overall_score", 0.0) or 0.0)

    gates = {
        "overall": overall_score >= active_thresholds["overall_min"],
        "identity": float(category_scores.get("identity", 0.0) or 0.0)
        >= active_thresholds["identity_min"],
        "symbolic": float(category_scores.get("symbolic", 0.0) or 0.0)
        >= active_thresholds["symbolic_min"],
        "reasoning": float(category_scores.get("reasoning", 0.0) or 0.0)
        >= active_thresholds["reasoning_min"],
        "coherence": float(summary.get("coherence_rate", overall_score) or 0.0)
        >= active_thresholds["coherence_min"],
        "format": float(summary.get("format_compliance", overall_score) or 0.0)
        >= active_thresholds["format_min"],
        "repetition": float(summary.get("repetition_failure_rate", 0.0) or 0.0)
        < active_thresholds["repetition_failure_max"],
    }

    results_raw = summary.get("results", [])
    results = results_raw if isinstance(results_raw, list) else []
    tasks: list[dict[str, object]] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        tasks.append(
            {
                "id": result.get("id", ""),
                "category": result.get("category", ""),
                "prompt": result.get("prompt", ""),
                "response": result.get("response", ""),
                "score": float(result.get("score", 0.0) or 0.0),
                "reason": result.get("reason", ""),
                "expected": result.get("expected", ""),
            }
        )

    generated_at = float(summary.get("generated_at", time.time()) or time.time())
    return {
        "schema_version": GOLDEN_EVAL_SCHEMA_VERSION,
        "baseline_id": f"golden-{int(generated_at)}-{overall_score:.4f}",
        "generated_at": generated_at,
        "source": source,
        "suite_size": len(tasks),
        "overall_score": round(overall_score, 4),
        "category_scores": {
            str(category): round(float(score), 4) for category, score in category_scores.items()
        },
        "thresholds": active_thresholds,
        "promotion_gates": gates,
        "promotion_allowed": all(gates.values()),
        "tasks": tasks,
    }


def write_golden_eval_baseline(
    summary: dict[str, object],
    *,
    source: str = "compact_eval",
    output_path: Path | None = None,
) -> dict[str, object]:
    baseline = build_golden_eval_baseline(summary, source=source)
    write_json(output_path or v2_report_path("golden_eval_baseline"), baseline)
    return baseline


@instrument("evaluation")
def _item_seed(base_seed: int, item_id: str) -> int:
    """Stable per-item sampling seed, independent of suite order and platform."""
    material = hashlib.sha256(f"{base_seed}:{item_id}".encode()).digest()
    return int.from_bytes(material[:8], "big") % (2**31 - 1)


def run_compact_eval(
    model: object,
    tokenizer: object,
    *,
    device: torch.device,
    output: bool = True,
    golden: bool = False,
    seed: int = 0,
) -> dict[str, object]:
    """Score the model on the compact or private suite.

    Generation evidence must replay exactly: every item samples from a local
    generator seeded by ``(seed, item id)``, so two runs of the same
    checkpoint produce identical summaries and gate decisions.
    """
    results: list[dict[str, object]] = []
    category_scores: dict[str, list[float]] = defaultdict(list)

    suite = PRIVATE_EVAL_SUITE if golden else COMPACT_EVAL_SUITE
    for item in suite:
        response = generate_text(
            model,
            tokenizer,
            item["prompt"],
            device=device,
            max_new_tokens=96,
            temperature=0.8,
            top_k=40,
            seed=_item_seed(seed, str(item["id"])),
        )
        if "scorer" in item:
            score, reason = _private_task_score(item, response)
            expected = str(item.get("expected", ""))
        elif "verifier" in item:
            score, expected = _verified_score(str(item["verifier"]), str(item["prompt"]), response)
            reason = f"verified against {item['verifier']} reference"
        else:
            score = _keyword_score(response, list(item.get("keywords", [])))
            expected = ""
            reason = "keyword coverage"
        category_scores[str(item["category"])].append(score)
        results.append(
            {
                "id": item["id"],
                "category": item["category"],
                "prompt": item["prompt"],
                "response": response,
                "score": round(float(score), 4),
                "reason": reason,
                "expected": expected,
            }
        )

    averages = {
        category: round(sum(scores) / max(1, len(scores)), 4)
        for category, scores in category_scores.items()
    }
    base_overall = sum(averages.values()) / max(1, len(averages))
    if golden:
        symbolic_values = [
            value for name, value in averages.items() if name in {"math", "logic", "dfc"}
        ]
        reasoning_values = [
            value
            for name, value in averages.items()
            if name in {"math", "logic", "context_qa", "long_context"}
        ]
        averages["symbolic"] = round(sum(symbolic_values) / max(1, len(symbolic_values)), 4)
        averages["reasoning"] = round(sum(reasoning_values) / max(1, len(reasoning_values)), 4)
    overall = round(base_overall, 4)
    repetition_rows = [result for result in results if result.get("category") == "repetition"]
    coherence_rows = [result for result in results if result.get("category") == "coherence"]
    format_rows = [
        result for result in results if result.get("category") in {"format", "instruction", "dfc"}
    ]
    # A suite without dedicated coherence/repetition tasks (the compact suite)
    # must not report 0.0 as if it were measured: coherence_rate would then be
    # structurally unable to pass any gate and repetition_failure_rate would
    # vacuously pass every gate. Fall back to surface checks on every response
    # using the same detectors the generation path trusts, and record which
    # basis produced the number.
    if coherence_rows:
        coherence_rate = sum(float(row["score"]) for row in coherence_rows) / len(coherence_rows)
        coherence_basis = "coherence_category_scores"
    else:
        surface_coherent = [
            not language_fragment_detected(str(row["response"])) for row in results
        ]
        coherence_rate = sum(surface_coherent) / max(1, len(surface_coherent))
        coherence_basis = "surface_fragment_fallback"
    if repetition_rows:
        repetition_failure_rate = sum(
            float(row["score"]) < 1.0 for row in repetition_rows
        ) / len(repetition_rows)
        repetition_basis = "repetition_category_scores"
    else:
        repeated_flags = [
            bool(detect_repetition(str(row["response"]))["repeated_ngrams_detected"])
            for row in results
        ]
        repetition_failure_rate = sum(repeated_flags) / max(1, len(repeated_flags))
        repetition_basis = "surface_ngram_fallback"
    summary = {
        "generated_at": time.time(),
        "overall_score": overall,
        "category_scores": averages,
        "results": results,
        "coherence_rate": round(coherence_rate, 4),
        "coherence_basis": coherence_basis,
        "format_compliance": round(
            sum(float(row["score"]) for row in format_rows) / max(1, len(format_rows)),
            4,
        ),
        "repetition_failure_rate": round(repetition_failure_rate, 4),
        "repetition_basis": repetition_basis,
        "decoding": {
            "strategy": "sampling",
            "temperature": 0.8,
            "top_k": 40,
            "max_new_tokens": 96,
            "seed": seed,
            "per_item_seeding": "sha256(seed:item_id)",
            "deterministic": True,
        },
    }
    if output:
        write_json(v2_report_path("eval_summary"), summary)
        append_jsonl(
            v2_report_path("eval_history"),
            {"ts": summary["generated_at"], "overall_score": overall, "category_scores": averages},
        )
        if golden:
            write_golden_eval_baseline(summary, source="train_unified_eval")
    return summary


if __name__ == "__main__":
    print(json.dumps({"suite_size": len(EVAL_SUITE)}, indent=2))
