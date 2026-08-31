"""Fail-closed certification for candidate log-likelihood adapters.

The certificate in this module tests the evaluator/model boundary, not model
quality.  A model adapter receives only the model-facing prompt, one candidate,
and its evaluator-assigned position.  It returns the log-probability of each
candidate suffix token; prompt-token likelihoods must never enter the score.

Deterministic controls prove the aggregation and bias detectors before any P35
comparison is trusted.  The committed receipt deliberately leaves random-weight
P35 x real-tokenizer device evidence pending: fake logits can certify plumbing,
but cannot certify a production scoring mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from .contracts import CausalCase, HiddenTruth, Split
from .evaluation_generators import build_evaluation_suite
from .metrics import measure_pair_behavior, selection_eligible


class ScoreMode(str, Enum):
    SUM = "sum"
    TOKEN_NORMALIZED = "token_normalized"
    BYTE_NORMALIZED = "byte_normalized"


@dataclass(frozen=True, slots=True)
class CandidateTrace:
    """Candidate-suffix token evidence returned by a model adapter."""

    token_ids: tuple[int, ...]
    token_logprobs: tuple[float, ...]

    def assert_valid(self) -> None:
        if not self.token_ids:
            raise ValueError("candidate suffix must contain at least one token")
        if len(self.token_ids) != len(self.token_logprobs):
            raise ValueError("candidate token ids and log-probabilities differ in length")
        if any(token < 0 for token in self.token_ids):
            raise ValueError("candidate token ids must be nonnegative")
        if any(not math.isfinite(value) or value > 1e-9 for value in self.token_logprobs):
            raise ValueError("candidate token log-probabilities must be finite and nonpositive")


class CandidateLogprobAdapter(Protocol):
    """The only interface a future model scorer may expose to E0."""

    @property
    def identity_sha256(self) -> str: ...

    def trace(
        self,
        model_view: Mapping[str, str],
        candidate: str,
        candidate_position: int,
    ) -> CandidateTrace: ...


@dataclass(frozen=True, slots=True)
class CandidateScore:
    candidate: str
    candidate_position: int
    utf8_bytes: int
    token_ids: tuple[int, ...]
    token_logprobs: tuple[float, ...]
    score: float


def aggregate_candidate_log_likelihood(
    trace: CandidateTrace,
    candidate: str,
    mode: ScoreMode,
) -> float:
    """Aggregate suffix-only token log-probabilities under a declared policy."""

    trace.assert_valid()
    byte_count = len(candidate.encode("utf-8"))
    if byte_count <= 0:
        raise ValueError("empty candidates are not scoreable")
    total = math.fsum(trace.token_logprobs)
    if mode is ScoreMode.SUM:
        return total
    if mode is ScoreMode.TOKEN_NORMALIZED:
        return total / len(trace.token_ids)
    if mode is ScoreMode.BYTE_NORMALIZED:
        return total / byte_count
    raise AssertionError(f"unhandled score mode: {mode}")


def score_case(
    case: CausalCase,
    adapter: CandidateLogprobAdapter,
    mode: ScoreMode,
) -> tuple[CandidateScore, ...]:
    """Score candidates without exposing evaluator-only truth to the adapter."""

    identity = adapter.identity_sha256
    if len(identity) != 64 or any(character not in "0123456789abcdef" for character in identity):
        raise ValueError("candidate scorer identity must be a lowercase SHA-256")
    model_view = case.model_view()
    rows: list[CandidateScore] = []
    for position, candidate in enumerate(case.candidates):
        trace = adapter.trace(model_view, candidate, position)
        trace.assert_valid()
        rows.append(
            CandidateScore(
                candidate=candidate,
                candidate_position=position,
                utf8_bytes=len(candidate.encode("utf-8")),
                token_ids=trace.token_ids,
                token_logprobs=trace.token_logprobs,
                score=aggregate_candidate_log_likelihood(trace, candidate, mode),
            )
        )
    if len({row.candidate for row in rows}) != len(rows):
        raise ValueError("candidate strings must be unique within a case")
    return tuple(rows)


def predict_case(rows: Sequence[CandidateScore]) -> str:
    if not rows:
        raise ValueError("cannot predict from an empty candidate set")
    return min(rows, key=lambda row: (-row.score, row.candidate)).candidate


def _prompt_key(model_view: Mapping[str, str]) -> str:
    if set(model_view) != {"context", "query", "prompt"}:
        raise ValueError("adapter received an invalid model-view schema")
    encoded = json.dumps(dict(model_view), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _identity(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class DeterministicControlAdapter:
    """Evaluator-only oracle/broken/random controls with deterministic fake logits."""

    def __init__(
        self,
        *,
        policy: str,
        token_ids: Mapping[str, tuple[int, ...]],
        targets_by_prompt: Mapping[str, str] | None = None,
        seed: int = 0,
    ) -> None:
        allowed = {"target", "random_token_logits"}
        if policy not in allowed:
            raise ValueError(f"unknown deterministic control policy: {policy}")
        self.policy = policy
        self.token_ids = dict(token_ids)
        self.targets_by_prompt = dict(targets_by_prompt or {})
        self.seed = seed
        self._identity = _identity(
            {
                "schema": "esoes-e0-deterministic-control/v1",
                "policy": policy,
                "token_ids": {key: list(value) for key, value in sorted(self.token_ids.items())},
                "targets": sorted(self.targets_by_prompt.items()),
                "seed": seed,
            }
        )

    @property
    def identity_sha256(self) -> str:
        return self._identity

    def _ids(self, candidate: str) -> tuple[int, ...]:
        if candidate in self.token_ids:
            return self.token_ids[candidate]
        raw = candidate.encode("utf-8")
        return tuple(1 + byte for byte in raw) or (1,)

    def trace(
        self,
        model_view: Mapping[str, str],
        candidate: str,
        candidate_position: int,
    ) -> CandidateTrace:
        del candidate_position
        prompt_key = _prompt_key(model_view)
        token_ids = self._ids(candidate)
        if self.policy == "target":
            if prompt_key not in self.targets_by_prompt:
                raise ValueError("target control has no evaluator-side target for prompt")
            favored = candidate == self.targets_by_prompt[prompt_key]
            value = -0.001 if favored else -20.0
            return CandidateTrace(token_ids, tuple(value for _ in token_ids))

        values = []
        for index, _ in enumerate(token_ids):
            digest = hashlib.sha256(
                f"{self.seed}:{prompt_key}:{candidate}:{index}".encode()
            ).digest()
            unit = int.from_bytes(digest[:8], "big") / (2**64 - 1)
            values.append(-0.1 - 4.9 * unit)
        return CandidateTrace(token_ids, tuple(values))


def _empty_hidden() -> HiddenTruth:
    return HiddenTruth((), (), (), ())


def build_bias_probe_cases(*, groups: int = 32) -> tuple[CausalCase, ...]:
    """Balanced position rotations with byte/token/first-token axes separated."""

    if groups <= 0:
        raise ValueError("bias probe groups must be positive")
    base = ("x", "medium", "the-longest-candidate")
    cases: list[CausalCase] = []
    for group in range(groups):
        answer = base[group % len(base)]
        for rotation in range(len(base)):
            candidates = base[rotation:] + base[:rotation]
            cases.append(
                CausalCase(
                    case_id=f"scoring-bias-{group:03d}-{rotation}",
                    family="scoring_adapter_bias_probe",
                    split=Split.DEVELOPMENT,
                    domain="scoring-contract",
                    template_id="scoring.bias.rotation.v1",
                    seed=91_001 + group,
                    facts=(f"Probe group {group} contains evaluator-hidden truth.",),
                    query=f"Return the correct probe value for group {group}.",
                    answer=answer,
                    candidates=candidates,
                    difficulty=(("candidates", 3),),
                    surface_axes=(("candidate_rotation", str(rotation)),),
                    provenance=(("rotation_group", str(group)),),
                    hidden=_empty_hidden(),
                )
            )
    return tuple(cases)


def _targets(cases: Sequence[CausalCase]) -> dict[str, str]:
    targets: dict[str, str] = {}
    for case in cases:
        key = _prompt_key(case.model_view())
        previous = targets.setdefault(key, case.answer)
        if previous != case.answer:
            raise ValueError("identical model-facing prompts have conflicting targets")
    return targets


def _token_map() -> dict[str, tuple[int, ...]]:
    # The three axes deliberately select different candidates:
    # shortest bytes=x, fewest tokens=medium, preferred first token=longest.
    return {
        "x": (101, 102, 103, 104),
        "medium": (201,),
        "the-longest-candidate": (777, 301),
    }


def _broken_targets(cases: Sequence[CausalCase], kind: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for case in cases:
        if kind == "candidate_position":
            target = case.candidates[0]
        elif kind == "answer_length":
            target = min(case.candidates, key=lambda value: (len(value.encode()), value))
        elif kind == "tokenization":
            target = min(case.candidates, key=lambda value: (len(_token_map()[value]), value))
        elif kind == "first_token":
            target = next(value for value in case.candidates if _token_map()[value][0] == 777)
        else:
            raise ValueError(f"unknown broken control: {kind}")
        # Position rotations share a prompt. Only position is allowed to vary.
        key = _prompt_key(case.model_view())
        if kind == "candidate_position":
            key = f"{key}:{case.case_id}"
        result[key] = target
    return result


class PositionControlAdapter(DeterministicControlAdapter):
    """Broken control that explicitly reads the candidate position."""

    def __init__(self, token_ids: Mapping[str, tuple[int, ...]]) -> None:
        super().__init__(policy="target", token_ids=token_ids, targets_by_prompt={})
        self._identity = _identity({"schema": "broken-position/v1", "token_ids": token_ids})

    def trace(
        self,
        model_view: Mapping[str, str],
        candidate: str,
        candidate_position: int,
    ) -> CandidateTrace:
        _prompt_key(model_view)
        ids = self._ids(candidate)
        value = -0.001 if candidate_position == 0 else -20.0
        return CandidateTrace(ids, tuple(value for _ in ids))


@dataclass(frozen=True, slots=True)
class BiasProfile:
    cases: int
    first_position_rate: float
    shortest_utf8_rate: float
    fewest_token_rate: float
    preferred_first_token_rate: float
    rotation_stability_rate: float


def bias_profile(
    cases: Sequence[CausalCase],
    adapter: CandidateLogprobAdapter,
    mode: ScoreMode,
    *,
    preferred_first_token_id: int = 777,
) -> BiasProfile:
    if not cases:
        raise ValueError("bias profile requires cases")
    first = shortest = fewest = preferred = 0
    rotation_groups: dict[str, list[str]] = {}
    for case in cases:
        rows = score_case(case, adapter, mode)
        prediction = predict_case(rows)
        selected = next(row for row in rows if row.candidate == prediction)
        first += int(selected.candidate_position == 0)
        shortest_bytes = min(row.utf8_bytes for row in rows)
        shortest += int(selected.utf8_bytes == shortest_bytes)
        fewest_tokens = min(len(row.token_ids) for row in rows)
        fewest += int(len(selected.token_ids) == fewest_tokens)
        preferred += int(selected.token_ids[0] == preferred_first_token_id)
        provenance = dict(case.provenance)
        rotation_groups.setdefault(provenance["rotation_group"], []).append(prediction)
    stable = sum(len(set(values)) == 1 for values in rotation_groups.values())
    total = len(cases)
    return BiasProfile(
        cases=total,
        first_position_rate=first / total,
        shortest_utf8_rate=shortest / total,
        fewest_token_rate=fewest / total,
        preferred_first_token_rate=preferred / total,
        rotation_stability_rate=stable / len(rotation_groups),
    )


def _predictions(
    cases: Sequence[CausalCase],
    adapter: CandidateLogprobAdapter,
    mode: ScoreMode,
) -> dict[str, str]:
    return {case.case_id: predict_case(score_case(case, adapter, mode)) for case in cases}


def implementation_sha256() -> str:
    """Hash source semantics independently of Git checkout newline policy."""

    normalized = Path(__file__).read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def build_scoring_certificate() -> dict[str, object]:
    modes = tuple(ScoreMode)
    probes = build_bias_probe_cases(groups=32)
    token_ids = _token_map()
    oracle = DeterministicControlAdapter(
        policy="target", token_ids=token_ids, targets_by_prompt=_targets(probes)
    )
    position = PositionControlAdapter(token_ids)
    broken = {
        name: DeterministicControlAdapter(
            policy="target",
            token_ids=token_ids,
            targets_by_prompt=_broken_targets(probes, name),
        )
        for name in ("answer_length", "tokenization", "first_token")
    }
    random_adapter = DeterministicControlAdapter(
        policy="random_token_logits", token_ids=token_ids, seed=92_001
    )

    direct_trace = CandidateTrace((1, 2), (-2.0, -2.0))
    aggregation_canary = {
        mode.value: aggregate_candidate_log_likelihood(direct_trace, "four", mode)
        for mode in modes
    }
    expected_aggregation = {"sum": -4.0, "token_normalized": -2.0, "byte_normalized": -1.0}

    oracle_profiles = {
        mode.value: asdict(bias_profile(probes, oracle, mode)) for mode in modes
    }
    broken_profiles = {
        "candidate_position": asdict(bias_profile(probes, position, ScoreMode.TOKEN_NORMALIZED)),
        "answer_length": asdict(
            bias_profile(probes, broken["answer_length"], ScoreMode.TOKEN_NORMALIZED)
        ),
        "tokenization": asdict(
            bias_profile(probes, broken["tokenization"], ScoreMode.TOKEN_NORMALIZED)
        ),
        "first_token": asdict(
            bias_profile(probes, broken["first_token"], ScoreMode.TOKEN_NORMALIZED)
        ),
    }
    random_profiles = {
        mode.value: asdict(bias_profile(probes, random_adapter, mode)) for mode in modes
    }
    repeated_random_profiles = {
        mode.value: asdict(bias_profile(probes, random_adapter, mode)) for mode in modes
    }

    suite = build_evaluation_suite(Split.DEVELOPMENT, seed=93_001, groups_per_family=2)
    suite_oracle = DeterministicControlAdapter(
        policy="target",
        token_ids={},
        targets_by_prompt=_targets(suite.cases),
    )
    oracle_predictions = _predictions(suite.cases, suite_oracle, ScoreMode.TOKEN_NORMALIZED)
    pair_measurement = measure_pair_behavior(suite, oracle_predictions)
    suite_random = DeterministicControlAdapter(
        policy="random_token_logits",
        token_ids={},
        seed=93_002,
    )
    random_predictions = _predictions(suite.cases, suite_random, ScoreMode.TOKEN_NORMALIZED)
    random_pair_measurement = measure_pair_behavior(suite, random_predictions)
    eligible = [case for case in suite.cases if selection_eligible(case)]
    oracle_selection_accuracy = sum(
        oracle_predictions[case.case_id] == case.answer for case in eligible
    ) / len(eligible)

    checks = {
        "aggregation_formulas_exact": aggregation_canary == expected_aggregation,
        "oracle_all_modes_exact": all(
            _predictions(probes, oracle, mode)
            == {case.case_id: case.answer for case in probes}
            for mode in modes
        ),
        "candidate_position_bias_detected": (
            broken_profiles["candidate_position"]["first_position_rate"] == 1.0
            and broken_profiles["candidate_position"]["rotation_stability_rate"] == 0.0
        ),
        "answer_length_bias_detected": (
            broken_profiles["answer_length"]["shortest_utf8_rate"] == 1.0
        ),
        "tokenization_bias_detected": (
            broken_profiles["tokenization"]["fewest_token_rate"] == 1.0
        ),
        "first_token_bias_detected": (
            broken_profiles["first_token"]["preferred_first_token_rate"] == 1.0
        ),
        "random_control_deterministic": random_profiles == repeated_random_profiles,
        "random_pair_metrics_computed": (
            random_pair_measurement.sensitivity_total == pair_measurement.sensitivity_total
            and random_pair_measurement.invariance_total == pair_measurement.invariance_total
        ),
        "oracle_selection_exact": oracle_selection_accuracy == 1.0,
        "sensitivity_metrics_exact": (
            pair_measurement.sensitivity_total > 0
            and pair_measurement.sensitivity_both_correct == pair_measurement.sensitivity_total
            and pair_measurement.sensitivity_correct_flip == pair_measurement.sensitivity_total
        ),
        "invariance_metrics_exact": (
            pair_measurement.invariance_total > 0
            and pair_measurement.invariance_both_correct == pair_measurement.invariance_total
            and pair_measurement.invariance_stable == pair_measurement.invariance_total
        ),
        "device_evidence_not_fabricated": True,
        "promotion_remains_unauthorized": True,
    }
    passed = all(checks.values())
    return {
        "schema": "esoes-e0-scoring-adapter-certificate/v1",
        "status": "CONTRACT_PASS_DEVICE_PENDING" if passed else "FAIL",
        "scope": "deterministic scorer plumbing and bias detection; no model quality claim",
        "implementation_sha256": implementation_sha256(),
        "suite": {
            "sha256": suite.sha256(),
            "cases": len(suite.cases),
            "pairs": len(suite.pairs),
            "split": suite.split.value,
            "generator_version": suite.generator_version,
        },
        "aggregation_canary": aggregation_canary,
        "score_modes": [mode.value for mode in modes],
        "controls": {
            "oracle_identity_sha256": oracle.identity_sha256,
            "position_broken_identity_sha256": position.identity_sha256,
            "answer_length_broken_identity_sha256": broken["answer_length"].identity_sha256,
            "tokenization_broken_identity_sha256": broken["tokenization"].identity_sha256,
            "first_token_broken_identity_sha256": broken["first_token"].identity_sha256,
            "random_identity_sha256": random_adapter.identity_sha256,
        },
        "bias_probe": {
            "cases": len(probes),
            "groups": 32,
            "preferred_first_token_id": 777,
            "oracle": oracle_profiles,
            "broken": broken_profiles,
            "random_token_logits": random_profiles,
        },
        "pair_metrics": asdict(pair_measurement),
        "random_pair_metrics": asdict(random_pair_measurement),
        "oracle_selection_accuracy": oracle_selection_accuracy,
        "checks": checks,
        "random_weight_p35_device_evidence": {
            "status": "PENDING",
            "promotion_authorized": False,
            "required": [
                "exact P35 constructor and checkpoint/model identity",
                "real 16k/24k/32k tokenizer artifacts and tokenizer hashes",
                "suffix-only teacher-forced log-probability adapter",
                "CPU/CUDA score parity within preregistered tolerance",
                "null-weight answer-length, first-token, position, and tokenization gates",
            ],
            "reason": (
                "This repository has no production P35 candidate-logprob adapter. "
                "Deterministic fake logits certify only the scorer contract."
            ),
        },
        "production_scoring_mode": None,
        "promotion_authorized": False,
        "fail_closed": True,
        "execution_scope": "platform-independent deterministic controls",
        "limitations": [
            "Fake token ids and logits do not measure a neural model or tokenizer quality.",
            "No aggregation mode is promoted until exact random-weight P35 and real-tokenizer null audits pass.",
            "The development suite is not an externally custodied sealed fixture.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/e0/scoring_adapter_certificate.json"),
    )
    args = parser.parse_args()
    receipt = build_scoring_certificate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": receipt["status"]}, sort_keys=True))
    return 0 if receipt["status"] == "CONTRACT_PASS_DEVICE_PENDING" else 1


if __name__ == "__main__":
    raise SystemExit(main())
