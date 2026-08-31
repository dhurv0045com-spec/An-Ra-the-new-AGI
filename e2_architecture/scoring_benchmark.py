"""Bounded real-tokenizer/random-P35 audit of E0 candidate scoring.

This probe performs no training.  It resizes the existing exact middle-P35
constructor to each actual local 16k/24k/32k tokenizer vocabulary, initializes
random weights from one declared seed, and computes suffix-only teacher-forced
candidate log-probabilities.  Candidate order is rotated independently of the
model-facing prompt so position leakage and CPU/CUDA disagreement are visible.

Vocabulary changes alter embedding/output parameters.  Results are null-scoring
and device-conformance evidence, never an iso-parameter tokenizer comparison.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import math
import platform
import statistics
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from e0_cognition.contracts import CausalCase, HiddenTruth, Split
from e0_cognition.scoring_certification import (
    CandidateLogprobAdapter,
    CandidateTrace,
    ScoreMode,
    predict_case,
    score_case,
)

from .block_benchmark import _build_model, shape_arms
from .plan import StaticArm


VOCABULARIES = (16_384, 24_576, 32_768)
PARITY_MAX_ABSOLUTE_ERROR = 0.05
PARITY_RELATIVE_RMS_ERROR = 1e-3


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _prompt_key(model_view: Mapping[str, str]) -> str:
    if set(model_view) != {"context", "query", "prompt"}:
        raise ValueError("unexpected E0 model-view schema")
    return _canonical_sha256(dict(model_view))


def _candidate_groups() -> tuple[tuple[str, str, str], ...]:
    """Cognition-relevant surfaces with varied bytes, tokens, and first tokens."""

    return (
        (
            " value_00000 = -381.90172564e-07; count=91,700,021",
            " entity_00000_edge_case_90172564 -> entity_00000_result",
            " DV-9017-2564 maps to FR90172564-G.",
        ),
        (
            " Δx=77.31005892e+04; नमूना-1; 東京; 🧪",
            " x\t:=\tvalue_1\nnext_line_1()",
            " ∀x∈S_1: P_310(x) ⇒ ∃y R_05892(x,y)",
        ),
        (
            " alpha_beta_gamma::node-17/revision_0042",
            " 00000000000000000000000000000127",
            " rollback(transaction_id=QZ-19-ALPHA, semantic_time=2048)",
        ),
        (
            " user.name+experiment@example.invalid",
            " C:\\models\\an-ra\\checkpoint_00000042.pt",
            " sha256:09fcc3c176c932cd102d45c3a2c74149",
        ),
        (
            " if (counter <= 17) { state = previous_state; }",
            " SELECT value FROM state_log WHERE logical_time <= 17 ORDER BY priority DESC;",
            " {\"event\":\"rollback\",\"time\":17,\"priority\":5}",
        ),
        (
            " relation(A_17, B_42) and relation(B_42, C_99)",
            " compose[edge_17 → edge_42 → edge_99]",
            " answer := <MISSING> unless evidence(node_99)",
        ),
    )


def build_null_cases() -> tuple[CausalCase, ...]:
    """Rotate candidate position while keeping each group prompt and answer fixed."""

    cases: list[CausalCase] = []
    hidden = HiddenTruth((), (), (), ())
    for group, candidates in enumerate(_candidate_groups()):
        answer = candidates[group % 3]
        for rotation in range(3):
            ordered = candidates[rotation:] + candidates[:rotation]
            cases.append(
                CausalCase(
                    case_id=f"p35-null-{group:02d}-{rotation}",
                    family="random_weight_scoring_null",
                    split=Split.DEVELOPMENT,
                    domain="local-device-null",
                    template_id="p35.random-weight.rotation.v1",
                    seed=94_001 + group,
                    facts=(f"Null-scoring surface group {group}; no semantic model claim.",),
                    query=f"Select the evaluator-hidden candidate for null group {group}.",
                    answer=answer,
                    candidates=ordered,
                    difficulty=(("candidates", 3),),
                    surface_axes=(("candidate_rotation", str(rotation)),),
                    provenance=(("rotation_group", str(group)),),
                    hidden=hidden,
                )
            )
    return tuple(cases)


@dataclass(frozen=True, slots=True)
class ScoringConfig:
    device: str
    seed: int = 94_101
    batch_size: int = 6

    def assert_valid(self) -> None:
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if self.seed < 0 or self.batch_size <= 0:
            raise ValueError("invalid seed or batch size")


class RecordedAdapter(CandidateLogprobAdapter):
    """Read-only adapter over teacher-forced traces emitted by one model run."""

    def __init__(
        self,
        traces: Mapping[tuple[str, str], CandidateTrace],
        *,
        identity_sha256: str,
    ) -> None:
        self.traces = dict(traces)
        self._identity_sha256 = identity_sha256

    @property
    def identity_sha256(self) -> str:
        return self._identity_sha256

    def trace(
        self,
        model_view: Mapping[str, str],
        candidate: str,
        candidate_position: int,
    ) -> CandidateTrace:
        del candidate_position
        key = (_prompt_key(model_view), candidate)
        if key not in self.traces:
            raise ValueError("teacher-forced trace is missing for prompt/candidate")
        return self.traces[key]


def _load_tokenizer(path: Path) -> Any:
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError("the tokenizers package is required for the device scoring audit") from exc
    payload = gzip.decompress(path.read_bytes()).decode("utf-8")
    return Tokenizer.from_str(payload)


def _middle_arm(vocabulary_size: int) -> StaticArm:
    middle = next(arm for arm in shape_arms() if arm.name == "middle")
    model = replace(middle.model, vocabulary_size=vocabulary_size)
    model.assert_valid()
    return StaticArm(
        name=f"middle-vocab-{vocabulary_size}",
        group="shape",
        factors=(("shape", "middle"), ("vocabulary_size", str(vocabulary_size))),
        model=model,
    )


def _unique_jobs(
    cases: Sequence[CausalCase], tokenizer: Any
) -> tuple[list[dict[str, object]], dict[tuple[str, str], tuple[int, ...]]]:
    jobs: list[dict[str, object]] = []
    tokenized: dict[tuple[str, str], tuple[int, ...]] = {}
    seen: set[tuple[str, str]] = set()
    for case in cases:
        prompt_key = _prompt_key(case.model_view())
        prompt = case.prompt()
        for candidate in case.candidates:
            key = (prompt_key, candidate)
            encoding = tokenizer.encode(prompt + candidate, add_special_tokens=False)
            full_ids = tuple(encoding.ids)
            boundary = len(prompt)
            crossing = [
                offset for offset in encoding.offsets if offset[0] < boundary < offset[1]
            ]
            if crossing:
                raise ValueError("a tokenizer token crosses the prompt/candidate boundary")
            suffix_start = next(
                (index for index, offset in enumerate(encoding.offsets) if offset[0] >= boundary),
                len(full_ids),
            )
            prompt_ids = full_ids[:suffix_start]
            candidate_ids = full_ids[suffix_start:]
            if not prompt_ids:
                raise ValueError("model-facing prompt tokenized to an empty sequence")
            if not candidate_ids:
                raise ValueError("candidate tokenized to an empty sequence")
            if (
                tokenizer.decode(list(full_ids), skip_special_tokens=False) != prompt + candidate
                or tokenizer.decode(list(prompt_ids), skip_special_tokens=False) != prompt
                or tokenizer.decode(list(candidate_ids), skip_special_tokens=False) != candidate
            ):
                raise ValueError("full-sequence prompt/candidate tokenization does not round-trip")
            previous = tokenized.setdefault(key, candidate_ids)
            if previous != candidate_ids:
                raise ValueError("candidate tokenization changed across position rotations")
            if key in seen:
                continue
            seen.add(key)
            jobs.append(
                {
                    "prompt_key": prompt_key,
                    "candidate": candidate,
                    "prompt_ids": prompt_ids,
                    "candidate_ids": candidate_ids,
                    "input_ids": full_ids,
                }
            )
    return jobs, tokenized


def _teacher_forced_traces(
    *,
    torch: Any,
    model: Any,
    jobs: Sequence[dict[str, object]],
    device: Any,
    batch_size: int,
) -> tuple[dict[tuple[str, str], CandidateTrace], float]:
    traces: dict[tuple[str, str], CandidateTrace] = {}
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(jobs), batch_size):
            batch = jobs[offset : offset + batch_size]
            maximum = max(len(job["input_ids"]) for job in batch)
            input_ids = torch.zeros((len(batch), maximum), dtype=torch.long, device=device)
            for row, job in enumerate(batch):
                values = torch.tensor(job["input_ids"], dtype=torch.long, device=device)
                input_ids[row, : len(values)] = values
            logits = model(input_ids)
            for row, job in enumerate(batch):
                prompt_length = len(job["prompt_ids"])
                candidate_ids = tuple(job["candidate_ids"])
                positions = torch.arange(
                    prompt_length - 1,
                    prompt_length - 1 + len(candidate_ids),
                    device=device,
                )
                suffix_logits = logits[row, positions].float()
                target_ids = torch.tensor(candidate_ids, dtype=torch.long, device=device)
                values = suffix_logits.gather(1, target_ids[:, None]).squeeze(1)
                logprobs = values - torch.logsumexp(suffix_logits, dim=-1)
                trace = CandidateTrace(
                    candidate_ids,
                    tuple(float(value) for value in logprobs.cpu().tolist()),
                )
                trace.assert_valid()
                traces[(str(job["prompt_key"]), str(job["candidate"]))] = trace
            del logits, input_ids
    if device.type == "cuda":
        torch.cuda.synchronize()
    return traces, time.perf_counter() - started


def _winner(rows: Sequence[Any]) -> Any:
    predicted = predict_case(rows)
    return next(row for row in rows if row.candidate == predicted)


def _bias_summary(
    cases: Sequence[CausalCase],
    adapter: RecordedAdapter,
    mode: ScoreMode,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    first_position = shortest_bytes = fewest_tokens = correct = 0
    rotations: dict[str, list[str]] = {}
    first_tokens: dict[int, int] = {}
    candidate_wins: dict[str, int] = {}
    score_rows: list[dict[str, object]] = []
    for case in cases:
        rows = score_case(case, adapter, mode)
        selected = _winner(rows)
        first_position += selected.candidate_position == 0
        shortest_bytes += selected.utf8_bytes == min(row.utf8_bytes for row in rows)
        fewest_tokens += len(selected.token_ids) == min(len(row.token_ids) for row in rows)
        correct += selected.candidate == case.answer
        first_tokens[selected.token_ids[0]] = first_tokens.get(selected.token_ids[0], 0) + 1
        candidate_wins[selected.candidate] = candidate_wins.get(selected.candidate, 0) + 1
        group = dict(case.provenance)["rotation_group"]
        rotations.setdefault(group, []).append(selected.candidate)
        score_rows.append(
            {
                "case_id": case.case_id,
                "prediction": selected.candidate,
                "answer": case.answer,
                "scores": {row.candidate: row.score for row in rows},
            }
        )
    total = len(cases)
    stable = sum(len(set(values)) == 1 for values in rotations.values())
    return (
        {
            "cases": total,
            "effective_prompt_groups": len(rotations),
            "first_position_selection_rate": first_position / total,
            "shortest_utf8_selection_rate": shortest_bytes / total,
            "fewest_token_selection_rate": fewest_tokens / total,
            "maximum_first_token_selection_share": max(first_tokens.values()) / total,
            "rotation_stability_rate": stable / len(rotations),
            "balanced_hidden_answer_accuracy": correct / total,
            "selected_first_token_histogram": {
                str(key): value for key, value in sorted(first_tokens.items())
            },
            "candidate_win_histogram": dict(sorted(candidate_wins.items())),
        },
        score_rows,
    )


def run_benchmark(
    config: ScoringConfig,
    *,
    artifact_directory: Path,
) -> dict[str, object]:
    config.assert_valid()
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment-dependent
        return {"schema": "esoes-e2-p35-scoring-null/v1", "status": "BLOCKED_TORCH", "reason": str(exc)}
    if config.device == "cuda" and not torch.cuda.is_available():
        return {
            "schema": "esoes-e2-p35-scoring-null/v1",
            "status": "BLOCKED_CUDA",
            "torch_version": torch.__version__,
        }

    cases = build_null_cases()
    case_identity = _canonical_sha256(
        [case.canonical(include_hidden=True) for case in cases]
    )
    device = torch.device(config.device)
    rows: list[dict[str, object]] = []
    all_finite = True
    all_roundtrip = True
    all_position_invariant = True
    for vocabulary_size in VOCABULARIES:
        artifact = artifact_directory / f"tokenizer-{vocabulary_size}.json.gz"
        tokenizer = _load_tokenizer(artifact)
        actual_vocabulary = tokenizer.get_vocab_size()
        if actual_vocabulary != vocabulary_size:
            raise ValueError(
                f"tokenizer vocabulary mismatch: {actual_vocabulary} != {vocabulary_size}"
            )
        jobs, tokenized = _unique_jobs(cases, tokenizer)
        arm = _middle_arm(vocabulary_size)
        torch.manual_seed(config.seed)
        model = _build_model(
            torch,
            arm,
            maximum_sequence_length=max(len(job["input_ids"]) for job in jobs),
        ).to(device=device, dtype=torch.float32)
        model.eval()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count != arm.model.parameter_receipt().total:
            raise RuntimeError("resized P35 parameter count mismatch")
        traces, elapsed = _teacher_forced_traces(
            torch=torch,
            model=model,
            jobs=jobs,
            device=device,
            batch_size=config.batch_size,
        )
        adapter_identity = _canonical_sha256(
            {
                "schema": "esoes-e2-recorded-p35-scorer/v1",
                "device": config.device,
                "model_sha256": arm.model.sha256(),
                "tokenizer_sha256": _sha256_file(artifact),
                "seed": config.seed,
                "source_sha256": _sha256_file(Path(__file__)),
            }
        )
        adapter = RecordedAdapter(traces, identity_sha256=adapter_identity)
        mode_summaries: dict[str, object] = {}
        mode_scores: dict[str, object] = {}
        for mode in ScoreMode:
            summary, scores = _bias_summary(cases, adapter, mode)
            mode_summaries[mode.value] = summary
            mode_scores[mode.value] = scores
            all_position_invariant &= (
                summary["first_position_selection_rate"] == 1 / 3
                and summary["rotation_stability_rate"] == 1.0
            )
        finite = all(
            math.isfinite(value)
            for trace in traces.values()
            for value in trace.token_logprobs
        )
        roundtrip = len(tokenized) == len(traces) == len(jobs)
        all_finite &= finite
        all_roundtrip &= roundtrip
        rows.append(
            {
                "vocabulary_size": vocabulary_size,
                "tokenizer_artifact": artifact.name,
                "tokenizer_sha256": _sha256_file(artifact),
                "model_sha256": arm.model.sha256(),
                "parameters": parameter_count,
                "adapter_identity_sha256": adapter_identity,
                "unique_prompt_candidate_jobs": len(jobs),
                "maximum_sequence_tokens": max(len(job["input_ids"]) for job in jobs),
                "elapsed_seconds": elapsed,
                "finite": finite,
                "roundtrip_and_trace_coverage": roundtrip,
                "candidate_tokenizations": [
                    {
                        "prompt_sha256": key[0],
                        "candidate": key[1],
                        "token_ids": list(tokenized[key]),
                        "token_logprobs": list(traces[key].token_logprobs),
                    }
                    for key in sorted(tokenized)
                ],
                "bias_by_mode": mode_summaries,
                "scores_by_mode": mode_scores,
            }
        )
        del model, adapter, traces
        gc.collect()
        if config.device == "cuda":
            torch.cuda.empty_cache()

    checks = {
        "actual_tokenizers_all_loaded": len(rows) == len(VOCABULARIES),
        "all_teacher_forced_scores_finite": all_finite,
        "prompt_candidate_roundtrip_and_trace_coverage": all_roundtrip,
        "candidate_position_rotation_invariant": all_position_invariant,
        "vocabulary_parameter_counts_explicit_and_distinct": (
            len({row["parameters"] for row in rows}) == len(VOCABULARIES)
        ),
        "no_training_performed": True,
        "not_claimed_iso_parameter": True,
        "promotion_remains_unauthorized": True,
    }
    return {
        "schema": "esoes-e2-p35-scoring-null/v1",
        "status": "PASS_LOCAL_NULL_DEVICE" if all(checks.values()) else "FAIL",
        "scope": "random-weight exact middle-P35 suffix-only scoring; no training",
        "implementation_sha256": _sha256_file(Path(__file__)),
        "model_constructor_sha256": _sha256_file(
            Path(__file__).with_name("block_benchmark.py")
        ),
        "e0_scoring_contract_sha256": _sha256_file(
            Path(__file__).parents[1] / "e0_cognition/scoring_certification.py"
        ),
        "config": asdict(config),
        "device_name": (
            torch.cuda.get_device_name(0) if config.device == "cuda" else platform.processor()
        ),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "case_sha256": case_identity,
        "cases": len(cases),
        "effective_prompt_groups": len(cases) // 3,
        "checks": checks,
        "rows": rows,
        "promotion_authorized": False,
        "production_scoring_mode": None,
        "limitations": [
            "Random weights measure scorer/device bias, not cognition or tokenizer quality.",
            "Vocabulary changes alter tied embedding/output parameters; this is not iso-parameter learning evidence.",
            "The eager local CPU/CUDA constructor is not target TPU/XLA evidence.",
            "Six effective prompt groups are a bounded null canary, not a powered model comparison.",
        ],
    }


def _flatten_scores(receipt: Mapping[str, object]) -> dict[tuple[int, str, str, str], float]:
    flattened: dict[tuple[int, str, str, str], float] = {}
    for row in receipt["rows"]:
        vocabulary = row["vocabulary_size"]
        for mode, cases in row["scores_by_mode"].items():
            for case in cases:
                for candidate, score in case["scores"].items():
                    flattened[(vocabulary, mode, case["case_id"], candidate)] = score
    return flattened


def compare_receipts(cpu: Mapping[str, object], cuda: Mapping[str, object]) -> dict[str, object]:
    for receipt, device in ((cpu, "cpu"), (cuda, "cuda")):
        if receipt.get("status") != "PASS_LOCAL_NULL_DEVICE":
            raise ValueError(f"{device} scoring receipt is not passing")
        if receipt["config"]["device"] != device:
            raise ValueError("scoring receipt device identity mismatch")
    fixed = ("implementation_sha256", "model_constructor_sha256", "e0_scoring_contract_sha256", "case_sha256")
    if any(cpu[key] != cuda[key] for key in fixed):
        raise ValueError("CPU/CUDA scoring receipts do not bind the same source/cases")
    if cpu["config"]["seed"] != cuda["config"]["seed"]:
        raise ValueError("CPU/CUDA scoring seeds differ")
    cpu_rows = {row["vocabulary_size"]: row for row in cpu["rows"]}
    cuda_rows = {row["vocabulary_size"]: row for row in cuda["rows"]}
    if set(cpu_rows) != set(cuda_rows) or any(
        cpu_rows[v]["tokenizer_sha256"] != cuda_rows[v]["tokenizer_sha256"]
        or cpu_rows[v]["model_sha256"] != cuda_rows[v]["model_sha256"]
        for v in cpu_rows
    ):
        raise ValueError("CPU/CUDA tokenizer or model identities differ")

    left, right = _flatten_scores(cpu), _flatten_scores(cuda)
    if set(left) != set(right):
        raise ValueError("CPU/CUDA score keys differ")
    errors = [abs(left[key] - right[key]) for key in left]
    reference_rms = math.sqrt(statistics.fmean(value * value for value in left.values()))
    error_rms = math.sqrt(statistics.fmean(value * value for value in errors))
    maximum = max(errors)
    relative_rms = error_rms / max(reference_rms, 1e-12)
    prediction_mismatches = 0
    comparison_rows: list[dict[str, object]] = []
    for vocabulary in VOCABULARIES:
        for mode in ScoreMode:
            cpu_cases = {
                row["case_id"]: row["prediction"]
                for row in cpu_rows[vocabulary]["scores_by_mode"][mode.value]
            }
            cuda_cases = {
                row["case_id"]: row["prediction"]
                for row in cuda_rows[vocabulary]["scores_by_mode"][mode.value]
            }
            mismatches = sum(cpu_cases[key] != cuda_cases[key] for key in cpu_cases)
            prediction_mismatches += mismatches
            comparison_rows.append(
                {
                    "vocabulary_size": vocabulary,
                    "score_mode": mode.value,
                    "cases": len(cpu_cases),
                    "prediction_mismatches": mismatches,
                    "cpu_bias": cpu_rows[vocabulary]["bias_by_mode"][mode.value],
                    "cuda_bias": cuda_rows[vocabulary]["bias_by_mode"][mode.value],
                }
            )
    checks = {
        "source_case_model_tokenizer_identities_match": True,
        "score_keys_match": set(left) == set(right),
        "all_scores_finite": all(math.isfinite(value) for value in (*left.values(), *right.values())),
        "maximum_absolute_error_within_limit": maximum <= PARITY_MAX_ABSOLUTE_ERROR,
        "relative_rms_error_within_limit": relative_rms <= PARITY_RELATIVE_RMS_ERROR,
        "predictions_match_exactly": prediction_mismatches == 0,
        "promotion_remains_unauthorized": True,
    }
    return {
        "schema": "esoes-e2-p35-scoring-parity/v1",
        "status": "PASS_LOCAL_CPU_CUDA_NULL_PARITY" if all(checks.values()) else "FAIL",
        "scope": "paired CPU/CUDA random-weight P35 scorer parity; no training",
        "implementation_sha256": cpu["implementation_sha256"],
        "case_sha256": cpu["case_sha256"],
        "cpu_receipt_sha256": _canonical_sha256(cpu),
        "cuda_receipt_sha256": _canonical_sha256(cuda),
        "thresholds": {
            "maximum_absolute_error": PARITY_MAX_ABSOLUTE_ERROR,
            "relative_rms_error": PARITY_RELATIVE_RMS_ERROR,
        },
        "metrics": {
            "scores_compared": len(left),
            "maximum_absolute_error": maximum,
            "relative_rms_error": relative_rms,
            "prediction_mismatches": prediction_mismatches,
        },
        "checks": checks,
        "comparisons": comparison_rows,
        "production_scoring_mode": None,
        "promotion_authorized": False,
        "limitations": [
            "Parity and null-bias evidence do not select a tokenizer, architecture, or scoring mode.",
            "The three vocabulary arms have different parameter counts.",
            "No TPU/XLA scorer has been tested.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--device", choices=("cpu", "cuda"), required=True)
    run.add_argument("--artifact-directory", type=Path, required=True)
    run.add_argument("--seed", type=int, default=94_101)
    run.add_argument("--batch-size", type=int, default=6)
    run.add_argument("--output", type=Path, required=True)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--cpu", type=Path, required=True)
    compare.add_argument("--cuda", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        result = run_benchmark(
            ScoringConfig(device=args.device, seed=args.seed, batch_size=args.batch_size),
            artifact_directory=args.artifact_directory,
        )
    else:
        cpu = json.loads(args.cpu.read_text(encoding="utf-8"))
        cuda = json.loads(args.cuda.read_text(encoding="utf-8"))
        result = compare_receipts(cpu, cuda)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": result["status"]}, sort_keys=True))
    return 0 if str(result["status"]).startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
