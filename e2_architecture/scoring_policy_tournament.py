"""Execute and aggregate the frozen development-only scorer tournament.

Each shard owns one (device, vocabulary, model seed) cell.  Fresh fixtures are
intentionally unreachable here: fresh execution requires a separately committed
selection receipt after development closes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from e0_cognition.scoring_certification import CandidateTrace

from .block_benchmark import _build_model
from .scoring_benchmark import _middle_arm, _teacher_forced_traces
from .scoring_policy import (
    DEVELOPMENT_SEEDS,
    EQUIVALENCE_MARGIN,
    HOLM_FAMILYWISE_ALPHA,
    INDEPENDENT_TRIPLETS,
    MAX_GPU_HOURS,
    NULL_CONTEXTS_PER_PANEL,
    PER_SEED_MARGIN,
    VOCABULARIES,
    CandidateEvidence,
    Policy,
    score_contextual_calibration,
    score_independent_policy,
    select,
)
from .scoring_policy_fixture import (
    _load_tokenizer,
    _neutral_anchors,
    _sha256_file,
    _suffix_tokens,
    materialize_cases,
    redacted_case_identity,
    rotation_schedule,
    verify_rotation_schedule,
)


SHARD_SCHEMA = "esoes-e2-scoring-policy-shard/v1"
AGGREGATE_SCHEMA = "esoes-e2-scoring-policy-development/v1"
POLICIES = tuple(Policy)
T_CRITICAL_90_DF4 = 2.131_846_786


def rotation_order(candidates: Sequence[str], rotation: tuple[int, ...]) -> tuple[str, ...]:
    """Present ``candidates`` under a position-rotation permutation.

    ``rotation`` maps position -> candidate index (the fixture schedule's
    definition). This is the ONLY path through which candidate position can
    influence downstream selection; every rotation view must be materialized
    and scored for a receipt to claim the rotation contract.
    """

    if sorted(rotation) != [0, 1, 2] or len(candidates) != len(rotation):
        raise ValueError("rotation must be a permutation of the candidate positions")
    return tuple(candidates[rotation[position]] for position in range(len(candidates)))


def _rotation_geometry(
    scores: Mapping[str, float], candidates: Sequence[str], rotation: tuple[int, ...]
) -> dict[str, object]:
    """Execute one rotation view: present candidates in rotated position
    order, select a winner, and record BOTH the winner's intrinsic role and
    the position it occupied. A position-biased selector betrays itself
    here: the winning role changes across rotations."""

    if len(rotation) != len(candidates):
        raise ValueError("rotation width must match the candidate count")
    ordered = rotation_order(candidates, rotation)
    winner = select({candidate: scores[candidate] for candidate in ordered})
    return {
        "rotation": list(rotation),
        "presented_order": list(ordered),
        "winner_role": candidates.index(winner),
        "winner_position": ordered.index(winner),
    }


def _assert_rotation_geometry(geometry: Sequence[Mapping[str, object]],
                              candidates: Sequence[str]) -> None:
    """Fail-closed: three distinct permutations with full position coverage,
    and a rotation-stable winning role. Deleted, duplicated, or doctored
    rotations cannot pass."""

    if len(geometry) != 3:
        raise ValueError("rotation geometry requires exactly three executed rotations")
    seen = [tuple(item["rotation"]) for item in geometry]
    if len(set(seen)) != 3 or any(sorted(item) != [0, 1, 2] for item in seen):
        raise ValueError("rotation permutations are missing, duplicated, or malformed")
    for candidate_position in range(len(candidates)):
        occupied = [item["rotation"].index(candidate_position) for item in geometry]
        if sorted(occupied) != [0, 1, 2]:
            raise ValueError("a candidate did not occupy every position exactly once")
    roles = {item["winner_role"] for item in geometry}
    if len(roles) != 1:
        raise ValueError("winning role is not rotation-stable")


def _source_sha256() -> str:
    normalized = Path(__file__).read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode()).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _rank(scores: Mapping[str, float]) -> tuple[str, ...]:
    if len(scores) < 2 or any(not math.isfinite(value) for value in scores.values()):
        raise ValueError("ranking requires finite scores")
    ordered = sorted(scores, key=lambda candidate: (-scores[candidate], candidate))
    if any(
        math.isclose(scores[left], scores[right], rel_tol=0.0, abs_tol=1e-12)
        for left, right in zip(ordered, ordered[1:])
    ):
        raise ValueError("ranking contains an unresolved tie")
    return tuple(ordered)


def _evidence(
    candidates: Sequence[str],
    target: Mapping[str, CandidateTrace],
    neutral: Sequence[Mapping[str, CandidateTrace]],
) -> tuple[CandidateEvidence, ...]:
    if len(neutral) != NULL_CONTEXTS_PER_PANEL:
        raise ValueError("a neutral panel must contain exactly four contexts")
    return tuple(
        CandidateEvidence(
            candidate,
            target[candidate],
            tuple(context[candidate] for context in neutral),
        )
        for candidate in candidates
    )


def _scores(items: Sequence[CandidateEvidence], policy: Policy) -> dict[str, float]:
    if policy is Policy.CONTEXTUAL_CALIBRATION:
        return score_contextual_calibration(items)
    return {item.candidate: score_independent_policy(item, policy) for item in items}


def _build_jobs(
    tokenizer: Any,
    cases: Sequence[Mapping[str, object]],
    anchors: Sequence[Sequence[int]],
) -> tuple[list[dict[str, object]], dict[int, dict[str, object]]]:
    jobs: list[dict[str, object]] = []
    metadata: dict[int, dict[str, object]] = {}
    for case in cases:
        group = int(case["group"])
        prompt = str(case["prompt"])
        candidates = tuple(str(value) for value in case["candidates"])
        decoy = str(case["decoy"])
        all_candidates = (*candidates, decoy)
        target_tokens: dict[str, tuple[int, ...]] = {}
        prompt_length: int | None = None
        for candidate in all_candidates:
            candidate_ids, current_prompt_length = _suffix_tokens(tokenizer, prompt, candidate)
            if prompt_length is None:
                prompt_length = current_prompt_length
            elif current_prompt_length != prompt_length:
                raise ValueError("candidate changed target prompt token length")
            full = tuple(tokenizer.encode(prompt + candidate, add_special_tokens=False).ids)
            target_tokens[candidate] = candidate_ids
            jobs.append(
                {
                    "prompt_key": f"target:{group}",
                    "candidate": candidate,
                    "prompt_ids": full[: -len(candidate_ids)],
                    "candidate_ids": candidate_ids,
                    "input_ids": full,
                }
            )
        assert prompt_length is not None
        for panel, panel_anchors in enumerate(anchors):
            if len(panel_anchors) != NULL_CONTEXTS_PER_PANEL:
                raise ValueError("neutral anchor panel width drifted")
            for context, anchor in enumerate(panel_anchors):
                neutral_prompt = (int(anchor),) * prompt_length
                for candidate in all_candidates:
                    candidate_ids = target_tokens[candidate]
                    jobs.append(
                        {
                            "prompt_key": f"neutral:{panel}:{context}:{group}",
                            "candidate": candidate,
                            "prompt_ids": neutral_prompt,
                            "candidate_ids": candidate_ids,
                            "input_ids": neutral_prompt + candidate_ids,
                        }
                    )
        metadata[group] = {
            "candidates": candidates,
            "decoy": decoy,
            "surface_family": int(case["surface_family"]),
            "hidden_answer_role": int(case["hidden_answer_role"]),
            "prompt_tokens": prompt_length,
        }
    return jobs, metadata


def _synthetic_checks() -> dict[str, float]:
    base = {
        "a": CandidateTrace((1,), (-4.0,)),
        "b": CandidateTrace((2,), (-4.0,)),
        "c": CandidateTrace((3,), (-4.0,)),
    }
    neutral = tuple(dict(base) for _ in range(NULL_CONTEXTS_PER_PANEL))
    candidates = ("a", "b", "c")
    schedule = rotation_schedule(len(candidates))[0]
    recovered = swapped = 0
    # Inject the -1e-6 target into EVERY role (a, b, c): the preregistration
    # requires the synthetic intervention to rotate through all three roles.
    for role in range(3):
        for policy in POLICIES:
            injected = dict(base)
            injected[candidates[role]] = CandidateTrace((role + 1,), (-1e-6,))
            recovered += select(_scores(_evidence(candidates, injected, neutral), policy)) == candidates[role]
    for policy in POLICIES:
        swapped_target = dict(base)
        swapped_target["c"] = CandidateTrace((3,), (-1e-6,))
        swapped += select(_scores(_evidence(candidates, swapped_target, neutral), policy)) == "c"
    total_injections = len(candidates) * len(POLICIES)

    # Negative control for the rotation gates themselves: construct a
    # deliberately position-biased selector (the candidate presented at
    # position 0 always wins) and require the gate to REJECT it — unstable
    # winning role across rotations, first-position rate 3/3. If the gate
    # passed this biased selector, the rotation contract would be vacuous
    # and the shard must fail.
    biased_caught = True
    for policy in POLICIES:
        items = _evidence(candidates, base, neutral)
        scores = _scores(items, policy)
        biased = []
        for rotation in schedule:
            ordered = rotation_order(candidates, rotation)
            winner = ordered[0]  # the bias: first-presented candidate wins
            biased.append({
                "rotation": list(rotation),
                "presented_order": list(ordered),
                "winner_role": candidates.index(winner),
                "winner_position": ordered.index(winner),
            })
        try:
            _assert_rotation_geometry(biased, candidates)
            caught = False  # gate accepted a biased selector: gates are broken
        except ValueError:
            caught = True
        biased_caught &= caught
    return {
        "injection_recovery": recovered / total_injections,
        "swap_recovery": swapped / len(POLICIES),
        "all_three_roles_injected": True,
        "position_bias_negative_control_caught": float(biased_caught),
    }


def run_shard(
    *,
    artifact_directory: Path,
    fixture_receipt_path: Path,
    vocabulary: int,
    seed: int,
    device_name: str,
    batch_size: int,
    group_limit: int | None,
) -> dict[str, object]:
    if vocabulary not in VOCABULARIES or seed not in DEVELOPMENT_SEEDS:
        raise ValueError("shard is outside frozen development cells")
    if device_name not in {"cpu", "cuda"} or batch_size <= 0:
        raise ValueError("invalid shard device or batch size")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        return {"schema": SHARD_SCHEMA, "status": "BLOCKED_TORCH", "reason": str(exc)}
    if device_name == "cuda" and not torch.cuda.is_available():
        return {"schema": SHARD_SCHEMA, "status": "BLOCKED_CUDA"}

    fixture_receipt = json.loads(fixture_receipt_path.read_text(encoding="utf-8"))
    all_tokenizers = {
        size: _load_tokenizer(artifact_directory / f"tokenizer-{size}.json.gz")
        for size in VOCABULARIES
    }
    if any(tokenizer.get_vocab_size() != size for size, tokenizer in all_tokenizers.items()):
        raise ValueError("tokenizer vocabulary mismatch")
    artifact = artifact_directory / f"tokenizer-{vocabulary}.json.gz"
    tokenizer = all_tokenizers[vocabulary]
    cases = materialize_cases("development", all_tokenizers)
    identity = redacted_case_identity(cases)
    expected_identity = str(fixture_receipt["development"]["fixture_sha256"])
    if identity != expected_identity:
        raise ValueError("materialized development fixture identity drifted")
    if group_limit is not None:
        if group_limit <= 0 or group_limit > INDEPENDENT_TRIPLETS:
            raise ValueError("group limit is outside fixture")
        cases = cases[:group_limit]

    tokenizer_sha = _sha256_file(artifact)
    anchors = _neutral_anchors(tokenizer_sha, vocabulary)
    jobs, metadata = _build_jobs(tokenizer, cases, anchors)
    maximum_length = max(len(job["input_ids"]) for job in jobs)
    torch.manual_seed(seed)
    if device_name == "cuda":
        torch.cuda.manual_seed_all(seed)
    device = torch.device(device_name)
    dtype = torch.bfloat16 if device_name == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    model = _build_model(torch, _middle_arm(vocabulary), maximum_sequence_length=maximum_length)
    model = model.to(device=device, dtype=dtype).eval()
    actual_parameters = sum(parameter.numel() for parameter in model.parameters())
    expected_parameters = _middle_arm(vocabulary).model.parameter_receipt().total
    if actual_parameters != expected_parameters:
        raise RuntimeError("exact middle-P35 constructor parameter mismatch")
    traces, elapsed = _teacher_forced_traces(
        torch=torch, model=model, jobs=jobs, device=device, batch_size=batch_size
    )
    if device_name == "cuda" and elapsed / 3600 > MAX_GPU_HOURS:
        raise RuntimeError("single shard exceeded the frozen total GPU-hour abort budget")

    rows: list[dict[str, object]] = []
    for group, meta in sorted(metadata.items()):
        candidates = tuple(str(value) for value in meta["candidates"])
        decoy = str(meta["decoy"])
        all_candidates = (*candidates, decoy)
        target = {candidate: traces[(f"target:{group}", candidate)] for candidate in all_candidates}
        neutral_panels = []
        for panel in range(2):
            neutral_panels.append(
                tuple(
                    {
                        candidate: traces[(f"neutral:{panel}:{context}:{group}", candidate)]
                        for candidate in all_candidates
                    }
                    for context in range(NULL_CONTEXTS_PER_PANEL)
                )
            )
        policy_rows: dict[str, object] = {}
        schedule = rotation_schedule(1)[0]  # per-group schedule (3 rotations)
        for policy in POLICIES:
            panel_rows = []
            for panel in range(2):
                base = _scores(_evidence(candidates, target, neutral_panels[panel]), policy)
                with_decoy = _scores(_evidence(all_candidates, target, neutral_panels[panel]), policy)
                base_rank = _rank(base)
                decoy_rank = tuple(value for value in _rank(with_decoy) if value != decoy)
                # Execute ALL THREE position rotations on this panel: every
                # candidate is presented in every position exactly once. The
                # winning ROLE must be rotation-stable; the winner's POSITION
                # must distribute 1/3 across rotations for an unbiased
                # selection pipeline. Raw model traces are reused (legitimate:
                # prompts and candidate suffixes are rotation-invariant).
                geometry = [
                    _rotation_geometry(base, candidates, rotation)
                    for rotation in schedule
                ]
                _assert_rotation_geometry(geometry, candidates)
                panel_rows.append(
                    {
                        "winner_role": geometry[0]["winner_role"],
                        "ranking_roles": [candidates.index(value) for value in base_rank],
                        "ranking_with_decoy_roles": [all_candidates.index(value) for value in _rank(with_decoy)],
                        "scores": [base[candidate] for candidate in candidates],
                        "scores_with_decoy": [with_decoy[candidate] for candidate in all_candidates],
                        "decoy_shared_ranking_stable": decoy_rank == base_rank,
                        "rotation_geometry": geometry,
                        "first_position_wins": sum(
                            1 for item in geometry if item["winner_position"] == 0
                        ),
                    }
                )
            policy_rows[policy.value] = panel_rows
        rows.append(
            {
                "group": group,
                "surface_family": meta["surface_family"],
                "hidden_answer_role": meta["hidden_answer_role"],
                "policies": policy_rows,
            }
        )
    complete = len(rows) == INDEPENDENT_TRIPLETS
    return {
        "schema": SHARD_SCHEMA,
        "status": "PASS_EXECUTION" if complete else "PASS_SMOKE_ONLY",
        "implementation_sha256": _source_sha256(),
        "fixture_sha256": expected_identity,
        "tokenizer_sha256": tokenizer_sha,
        "configuration": {
            "split": "development",
            "vocabulary": vocabulary,
            "seed": seed,
            "device": device_name,
            "dtype": str(dtype).removeprefix("torch."),
            "batch_size": batch_size,
            "groups": len(rows),
        },
        "model": {"parameters": actual_parameters, "maximum_sequence_length": maximum_length},
        "execution": {"jobs": len(jobs), "seconds": elapsed, "gpu_hours": elapsed / 3600 if device_name == "cuda" else 0.0},
        "synthetic_checks": _synthetic_checks(),
        "rows": rows,
        "rows_sha256": _canonical_sha256(rows),
        "promotion_authorized": False,
    }


def _seed_rate(rows: Sequence[Mapping[str, object]], policy: str, axis: str) -> float:
    hits = 0
    for row in rows:
        winner = int(row["policies"][policy][0]["winner_role"])
        if axis == "shortest_utf8":
            hits += winner == 0
        elif axis == "fewest_tokens":
            hits += winner == 1
        elif axis == "marked_prefix":
            hits += winner == 2
        elif axis == "arbitrary_hidden_label":
            hits += winner == int(row["hidden_answer_role"])
        elif axis.startswith("surface_family_"):
            _, _, family_text, role_text = axis.split("_")
            family, role = int(family_text), int(role_text)
            if int(row["surface_family"]) == family:
                hits += winner == role
            else:
                continue
        else:
            raise ValueError("unknown preference axis")
    denominator = sum(
        axis.startswith("surface_family_") and int(row["surface_family"]) == int(axis.split("_")[2])
        or not axis.startswith("surface_family_")
        for row in rows
    )
    return hits / denominator


def _equivalence(seed_rates: Sequence[float]) -> dict[str, object]:
    if len(seed_rates) != len(DEVELOPMENT_SEEDS):
        raise ValueError("equivalence requires every frozen model-seed cluster")
    mean = statistics.fmean(seed_rates)
    standard_error = statistics.stdev(seed_rates) / math.sqrt(len(seed_rates))
    half_width = T_CRITICAL_90_DF4 * standard_error
    lower, upper = mean - half_width, mean + half_width
    if standard_error == 0:
        p_value = 0.0 if 1 / 3 - EQUIVALENCE_MARGIN < mean < 1 / 3 + EQUIVALENCE_MARGIN else 1.0
    else:
        lower_t = (mean - (1 / 3 - EQUIVALENCE_MARGIN)) / standard_error
        upper_t = (mean - (1 / 3 + EQUIVALENCE_MARGIN)) / standard_error
        p_value = max(1 - _student_t_cdf(lower_t, 4), _student_t_cdf(upper_t, 4))
    return {
        "seed_rates": list(seed_rates),
        "mean": mean,
        "confidence_interval_90": [lower, upper],
        "inside_equivalence_margin": lower > 1 / 3 - EQUIVALENCE_MARGIN and upper < 1 / 3 + EQUIVALENCE_MARGIN,
        "every_seed_inside_margin": all(abs(value - 1 / 3) <= PER_SEED_MARGIN for value in seed_rates),
        "tost_p_value": p_value,
    }


def _continued_beta(a: float, b: float, x: float) -> float:
    """Numerically stable continued fraction used by regularized beta."""

    maximum_iterations, epsilon, floor = 200, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1.0 / max(abs(d), floor) * (1 if d >= 0 else -1)
    value = d
    for index in range(1, maximum_iterations + 1):
        even = 2 * index
        coefficient = index * (b - index) * x / ((qam + even) * (a + even))
        d = 1.0 + coefficient * d
        d = d if abs(d) >= floor else floor
        c = 1.0 + coefficient / c
        c = c if abs(c) >= floor else floor
        d = 1.0 / d
        value *= d * c
        coefficient = -(a + index) * (qab + index) * x / ((a + even) * (qap + even))
        d = 1.0 + coefficient * d
        d = d if abs(d) >= floor else floor
        c = 1.0 + coefficient / c
        c = c if abs(c) >= floor else floor
        d = 1.0 / d
        delta = d * c
        value *= delta
        if abs(delta - 1.0) <= epsilon:
            return value
    raise ArithmeticError("incomplete beta did not converge")


def _regularized_beta(x: float, a: float, b: float) -> float:
    if not 0 <= x <= 1 or a <= 0 or b <= 0:
        raise ValueError("invalid regularized-beta arguments")
    if x in {0.0, 1.0}:
        return x
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1) / (a + b + 2):
        return front * _continued_beta(a, b, x) / a
    return 1 - front * _continued_beta(b, a, 1 - x) / b


def _student_t_cdf(value: float, degrees_of_freedom: int) -> float:
    if degrees_of_freedom <= 0 or not math.isfinite(value):
        raise ValueError("Student-t CDF requires finite value and positive df")
    beta = _regularized_beta(
        degrees_of_freedom / (degrees_of_freedom + value * value),
        degrees_of_freedom / 2,
        0.5,
    )
    return 1 - beta / 2 if value >= 0 else beta / 2


def _holm_decisions(named_p_values: Mapping[str, float]) -> dict[str, bool]:
    """Holm step-down rejection decisions for equivalence-null hypotheses."""

    ordered = sorted(named_p_values.items(), key=lambda item: (item[1], item[0]))
    decisions: dict[str, bool] = {name: False for name in named_p_values}
    still_rejecting = True
    total = len(ordered)
    for index, (name, p_value) in enumerate(ordered):
        if not 0 <= p_value <= 1:
            raise ValueError("Holm p-values must lie in [0, 1]")
        still_rejecting = still_rejecting and p_value <= HOLM_FAMILYWISE_ALPHA / (total - index)
        decisions[name] = still_rejecting
    return decisions


def aggregate(receipts: Sequence[Mapping[str, object]]) -> dict[str, object]:
    expected = {(device, vocabulary, seed) for device in ("cpu", "cuda") for vocabulary in VOCABULARIES for seed in DEVELOPMENT_SEEDS}
    cells: dict[tuple[str, int, int], Mapping[str, object]] = {}
    shard_implementations: set[str] = set()
    for receipt in receipts:
        if receipt.get("schema") != SHARD_SCHEMA or receipt.get("status") != "PASS_EXECUTION":
            raise ValueError("aggregate accepts only complete execution shards")
        config = receipt["configuration"]
        key = (str(config["device"]), int(config["vocabulary"]), int(config["seed"]))
        if key in cells:
            raise ValueError("duplicate tournament cell")
        cells[key] = receipt
        shard_implementations.add(str(receipt["implementation_sha256"]))
    if len(shard_implementations) > 1:
        raise ValueError("tournament shards came from different implementations")
    missing = sorted(expected - set(cells))
    extra = sorted(set(cells) - expected)
    if extra:
        raise ValueError("unexpected tournament cells")

    axes = (
        "shortest_utf8",
        "fewest_tokens",
        "marked_prefix",
        "arbitrary_hidden_label",
        *(f"surface_family_{family}_{role}" for family in range(6) for role in range(3)),
    )
    policy_results: dict[str, object] = {}
    for policy in (Policy.DOMAIN_PMI.value, Policy.CONTEXTUAL_CALIBRATION.value):
        vocabulary_rows: dict[str, object] = {}
        for vocabulary in VOCABULARIES:
            cuda = [cells.get(("cuda", vocabulary, seed)) for seed in DEVELOPMENT_SEEDS]
            if any(item is None for item in cuda):
                continue
            equivalence = {
                axis: _equivalence([_seed_rate(item["rows"], policy, axis) for item in cuda])
                for axis in axes
            }
            flat_rows = [row for item in cuda for row in item["rows"]]
            panel_agreement = statistics.fmean(
                row["policies"][policy][0]["ranking_roles"] == row["policies"][policy][1]["ranking_roles"]
                for row in flat_rows
            )
            decoy_stability = statistics.fmean(
                panel["decoy_shared_ranking_stable"]
                for row in flat_rows
                for panel in row["policies"][policy]
            )
            parity_strata = {
                f"panel_{panel}:{candidate_set}": {
                    "winner_mismatches": 0,
                    "maximum_absolute_error": 0.0,
                    "squared_error": 0.0,
                    "squared_reference": 0.0,
                    "scores": 0,
                }
                for panel in range(2)
                for candidate_set in ("base", "with_decoy")
            }
            parity_complete = True
            for seed in DEVELOPMENT_SEEDS:
                left, right = cells.get(("cpu", vocabulary, seed)), cells.get(("cuda", vocabulary, seed))
                if left is None or right is None:
                    parity_complete = False
                    continue
                for cpu_row, cuda_row in zip(left["rows"], right["rows"], strict=True):
                    for panel in range(2):
                        cpu_policy = cpu_row["policies"][policy][panel]
                        cuda_policy = cuda_row["policies"][policy][panel]
                        for candidate_set, score_key in (("base", "scores"), ("with_decoy", "scores_with_decoy")):
                            stratum = parity_strata[f"panel_{panel}:{candidate_set}"]
                            rank_key = "ranking_roles" if candidate_set == "base" else "ranking_with_decoy_roles"
                            stratum["winner_mismatches"] += cpu_policy[rank_key] != cuda_policy[rank_key]
                            for cpu_score, cuda_score in zip(cpu_policy[score_key], cuda_policy[score_key], strict=True):
                                error = float(cpu_score) - float(cuda_score)
                                stratum["maximum_absolute_error"] = max(stratum["maximum_absolute_error"], abs(error))
                                stratum["squared_error"] += error * error
                                stratum["squared_reference"] += float(cpu_score) * float(cpu_score)
                                stratum["scores"] += 1
            for stratum in parity_strata.values():
                stratum["relative_rms_error"] = (
                    math.sqrt(stratum.pop("squared_error") / max(stratum.pop("squared_reference"), 1e-30))
                    if stratum["scores"] else None
                )
            parity_winners = parity_complete and all(value["winner_mismatches"] == 0 for value in parity_strata.values())
            parity_absolute = parity_complete and all(value["maximum_absolute_error"] <= 0.05 for value in parity_strata.values())
            parity_rms = parity_complete and all(value["relative_rms_error"] is not None and value["relative_rms_error"] <= 0.001 for value in parity_strata.values())
            checks = {
                "equivalence_intervals": all(value["inside_equivalence_margin"] for value in equivalence.values()),
                "per_seed_margins": all(value["every_seed_inside_margin"] for value in equivalence.values()),
                "panel_ranking_agreement": panel_agreement >= 0.95,
                "decoy_stability": decoy_stability >= 0.99,
                "parity_complete": parity_complete,
                "parity_winners": parity_winners,
                "parity_absolute": parity_absolute,
                "parity_relative_rms": parity_rms,
            }
            vocabulary_rows[str(vocabulary)] = {
                "equivalence": equivalence,
                "panel_ranking_agreement": panel_agreement,
                "decoy_shared_ranking_stability": decoy_stability,
                "parity": parity_strata,
                "checks": checks,
            }
        policy_results[policy] = vocabulary_rows
    hypothesis_p_values = {
        f"{policy}:{vocabulary}:{axis}": detail["tost_p_value"]
        for policy, vocabulary_rows in policy_results.items()
        for vocabulary, result in vocabulary_rows.items()
        for axis, detail in result["equivalence"].items()
    }
    holm = _holm_decisions(hypothesis_p_values) if len(hypothesis_p_values) == 132 else {}
    holm_complete = len(holm) == 132 and all(holm.values())
    gpu_hours = math.fsum(float(value["execution"]["gpu_hours"]) for value in cells.values())
    complete = not missing
    synthetic_pass = all(
        value["synthetic_checks"] == {
            "injection_recovery": 1.0,
            "swap_recovery": 1.0,
            "all_three_roles_injected": True,
            "position_bias_negative_control_caught": 1.0,
        }
        for value in cells.values()
    )
    # Rotation geometry: fail closed on every executed row. Three distinct
    # permutations, full position coverage, rotation-stable winner role, and
    # a first-position selection rate equivalent to 1/3 (the preregistered
    # unbiased-pipeline contract). Deleted or duplicated rotations cannot
    # aggregate.
    rotation_gate_pass = True
    first_position_per_seed: dict[tuple[int, int], list[int]] = {}
    for receipt in cells.values():
        for row in receipt["rows"]:
            for policy_rows in row["policies"].values():
                for panel in policy_rows:
                    try:
                        candidates_length = len(panel["rotation_geometry"][0]["presented_order"])
                        _assert_rotation_geometry(panel["rotation_geometry"],
                                                  list(range(candidates_length)))
                    except (ValueError, KeyError, IndexError):
                        rotation_gate_pass = False
                    if "first_position_wins" in panel:
                        key = (int(receipt["configuration"]["vocabulary"]),
                               int(receipt["configuration"]["seed"]))
                        first_position_per_seed.setdefault(key, []).append(
                            int(panel["first_position_wins"]))
    first_position_equivalence: dict[str, dict[str, object]] = {}
    for (vocabulary, seed), wins in sorted(first_position_per_seed.items()):
        rate = sum(wins) / (3 * len(wins))  # each group contributes 3 rotations
        first_position_equivalence[f"{vocabulary}:{seed}"] = {
            "rate": round(rate, 4),
            "inside_margin": abs(rate - 1 / 3) <= PER_SEED_MARGIN,
        }
    cuda_expected = {
        ("cuda", vocabulary, seed)
        for vocabulary in VOCABULARIES
        for seed in DEVELOPMENT_SEEDS
    }
    cuda_complete = cuda_expected <= set(cells)
    nonparity_keys = {
        "equivalence_intervals",
        "per_seed_margins",
        "panel_ranking_agreement",
        "decoy_stability",
    }
    policy_survives_bias_screen = {
        policy: len(vocabulary_rows) == len(VOCABULARIES)
        and all(
            all(row["checks"][key] for key in nonparity_keys)
            for row in vocabulary_rows.values()
        )
        for policy, vocabulary_rows in policy_results.items()
    }
    early_policy_failure = (
        cuda_complete
        and synthetic_pass
        and not any(policy_survives_bias_screen.values())
    )
    all_checks = complete and synthetic_pass and holm_complete and rotation_gate_pass and gpu_hours <= MAX_GPU_HOURS and all(
        all(row["checks"].values()) for policy in policy_results.values() for row in policy.values()
    )
    return {
        "schema": AGGREGATE_SCHEMA,
        "status": (
            "PASS_DEVELOPMENT_ELIGIBILITY"
            if all_checks
            else "FAIL_DEVELOPMENT_POLICY"
            if early_policy_failure
            else "BLOCKED_OR_FAIL"
        ),
        "implementation_sha256": _source_sha256(),
        "shard_implementation_sha256": next(iter(shard_implementations), None),
        "cells_present": len(cells),
        "cells_required": len(expected),
        "missing_cells": [list(value) for value in missing],
        "gpu_hours": gpu_hours,
        "gpu_budget_pass": gpu_hours <= MAX_GPU_HOURS,
        "synthetic_interventions_pass": synthetic_pass,
        "rotation_geometry_pass": rotation_gate_pass,
        "first_position_equivalence": first_position_equivalence,
        "cuda_stage_complete": cuda_complete,
        "policy_survives_bias_screen": policy_survives_bias_screen,
        "parity_skipped_due_prior_failure": early_policy_failure and bool(missing),
        "policies": policy_results,
        "holm_familywise_alpha": HOLM_FAMILYWISE_ALPHA,
        "holm_gate_implemented": True,
        "holm_hypotheses": len(holm),
        "holm_all_equivalence_nulls_rejected": holm_complete,
        "holm_decisions": holm,
        "promotion_authorized": False,
        "limitations": [
            "This command cannot execute fresh fixtures.",
            "Parametric cluster-level TOST assumes the five seed rates are an adequate sampling model.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run-shard")
    run.add_argument("--artifact-directory", type=Path, required=True)
    run.add_argument("--fixture-receipt", type=Path, required=True)
    run.add_argument("--vocabulary", type=int, choices=VOCABULARIES, required=True)
    run.add_argument("--seed", type=int, choices=DEVELOPMENT_SEEDS, required=True)
    run.add_argument("--device", choices=("cpu", "cuda"), required=True)
    run.add_argument("--batch-size", type=int, default=32)
    run.add_argument("--group-limit", type=int)
    run.add_argument("--output", type=Path, required=True)
    combine = subparsers.add_parser("aggregate-development")
    combine.add_argument("--receipt", action="append", type=Path, required=True)
    combine.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run-shard":
        result = run_shard(
            artifact_directory=args.artifact_directory,
            fixture_receipt_path=args.fixture_receipt,
            vocabulary=args.vocabulary,
            seed=args.seed,
            device_name=args.device,
            batch_size=args.batch_size,
            group_limit=args.group_limit,
        )
    else:
        result = aggregate([json.loads(path.read_text(encoding="utf-8")) for path in args.receipt])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": result["status"]}, sort_keys=True))
    return 0 if str(result["status"]).startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
