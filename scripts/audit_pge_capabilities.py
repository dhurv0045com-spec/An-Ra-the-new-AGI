"""Identical, stateful capability audit for An-Ra V4 checkpoints.

This script deliberately separates free decoding, candidate selection, exact
realization, and query normalization.  It also measures loss on the held-out
validation shards from the exact continuation pack.  It never trains or
mutates a checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anra_core.executor import CoreExecutor
from anra_core.generate import generate


SCHEMA = "anra-pge-capability-audit/v1"
ENTITIES = (
    "talren", "vexora", "mirven", "quorath", "selmira", "dovarin",
    "kelvath", "norvex", "praxine", "ulmar", "zendri", "boreth",
    "calvix", "ferona", "galdren", "hestari",
)
VALUES = (
    "amber", "linen", "quartz", "maple", "silver", "violet", "copper",
    "coral", "marble", "bronze", "ivory", "orange", "purple", "yellow",
    "ocean", "river",
)
CREATIVE_PROMPTS = (
    "The moonlit library opened its doors and",
    "At the edge of the silent desert, a machine",
    "She unfolded the impossible map and discovered",
    "The last botanist on Mars recorded that",
)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _exact_value(text: str, expected: str) -> bool:
    words = _norm(text).split()
    return bool(words) and words[0] == expected


def _ngram_metrics(texts: list[str]) -> dict[str, float | int]:
    tokenized = [_norm(text).split() for text in texts]
    flat = [word for words in tokenized for word in words]
    grams2 = [tuple(words[i:i + 2]) for words in tokenized for i in range(len(words) - 1)]
    grams3 = [tuple(words[i:i + 3]) for words in tokenized for i in range(len(words) - 2)]

    def distinct(items) -> float:
        return len(set(items)) / len(items) if items else 0.0

    repeated3 = 1.0 - distinct(grams3) if grams3 else 0.0
    degenerate = 0
    for words in tokenized:
        local = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
        if local and 1.0 - distinct(local) >= 0.30:
            degenerate += 1
    return {
        "outputs": len(texts),
        "words": len(flat),
        "distinct_1": round(distinct(flat), 6),
        "distinct_2": round(distinct(grams2), 6),
        "distinct_3": round(distinct(grams3), 6),
        "repeated_3gram_ratio": round(repeated3, 6),
        "degenerate_outputs": degenerate,
    }


def _generate(executor: CoreExecutor, prompt: str, *, assisted: bool = False,
              max_new_tokens: int = 12, seed: int = 0) -> str:
    return generate(
        executor,
        executor.tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.8 if assisted else 0.0,
        top_p=0.92,
        seed=seed,
        repetition_penalty=1.15 if assisted else 1.0,
        no_repeat_ngram_size=4 if assisted else 0,
    )


@torch.inference_mode()
def _candidate_logprob(executor: CoreExecutor, prompt: str, candidate: str) -> float:
    tok = executor.tokenizer
    prompt_ids = [tok.bos_token_id, *tok.encode(prompt)]
    candidate_ids = tok.encode(candidate)
    ids = torch.tensor([prompt_ids + candidate_ids], dtype=torch.long)
    logits = executor.forward(ids).logits.float()
    start = len(prompt_ids) - 1
    selected = logits[0, start:start + len(candidate_ids)]
    targets = torch.tensor(candidate_ids, device=selected.device)
    return float(F.log_softmax(selected, dim=-1).gather(1, targets[:, None]).sum().item())


def _binding_cases() -> list[dict]:
    """Create a frozen 48-query, four-fact nonce binding set."""
    rng = random.Random(1701501)
    cases: list[dict] = []
    for block in range(12):
        entity_start = (block * 3) % len(ENTITIES)
        value_start = (block * 5) % len(VALUES)
        entities = [ENTITIES[(entity_start + i) % len(ENTITIES)] for i in range(4)]
        values = [VALUES[(value_start + i) % len(VALUES)] for i in range(4)]
        pairs = list(zip(entities, values))
        order = pairs[:]
        rng.shuffle(order)
        facts = " ".join(f"The seal for {entity} is {value}." for entity, value in order)
        # Candidate and query indices follow the byte-identical fact order.
        # Counterfactual normalization is allowed to change query identity
        # only; it must never rewrite facts, add a plan, or use gold.
        candidates = [value for _entity, value in order]
        query_entities = [entity for entity, _value in order]

        def prompt_for(query_entity: str) -> str:
            return (
                f"<k>{facts}</k>\n"
                f"<q>What is the seal for {query_entity}?</q>\n<answer>"
            )

        for query_index, (entity, value) in enumerate(order):
            raw = prompt_for(entity)
            cases.append({
                "id": f"binding-{block:02d}-{query_index}",
                "entity": entity,
                "expected": value,
                "candidates": candidates,
                "query_index": query_index,
                "query_entities": query_entities,
                "raw_prompt": raw,
                "counterfactual_prompts": {
                    str(index): prompt_for(other_entity)
                    for index, other_entity in enumerate(query_entities)
                    if index != query_index
                },
            })
    return cases


def _policy_transfer(rows: list[dict], policy_path: Path | None) -> dict | None:
    if policy_path is None:
        return None
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    means = policy["standardization"]["means"]
    stds = policy["standardization"]["stds"]
    models = policy["models"]
    costs = policy["costs"]
    action_counts = Counter()
    successes = regressions = total_cost = 0

    def top2(values: list[float]) -> float:
        ordered = sorted(values)
        return ordered[-1] - ordered[-2]

    def spread(values: list[float]) -> float:
        mean = sum(values) / len(values)
        return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5

    for row in rows:
        raw = row["raw_score_vector"]
        normalized = row["normalized_score_vector"]
        free_candidate = row["free_candidate"]
        raw_pick = row["raw_choice"]
        normalized_pick = row["normalized_choice"]
        raw_margin = top2(raw)
        normalized_margin = top2(normalized)
        normalized_sorted = sorted(normalized, reverse=True)
        features_raw = [
            4.0, 1.0, 0.0, raw_margin, normalized_margin,
            spread(raw), spread(normalized),
            float(raw_pick == normalized_pick),
            float(free_candidate is not None and free_candidate == raw_pick),
            float(free_candidate is not None and free_candidate == normalized_pick),
            1.0,
            normalized_sorted[0] - normalized_sorted[1],
            normalized_margin - raw_margin,
        ]
        features = [
            (value - mean) / std
            for value, mean, std in zip(features_raw, means, stds)
        ]

        def prediction(action: str) -> float:
            model = models[action]
            z = sum(weight * value for weight, value in zip(model["weights"], features)) + model["bias"]
            # Stable logistic for occasionally large out-of-distribution features.
            return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z))))

        utilities = {
            action: prediction(action) - policy["lambda"] * costs[action]
            for action in ("NO_CHANGE", "CONSTRAINED", "NORMALIZED")
        }
        action = min(utilities, key=lambda name: (-utilities[name], costs[name]))
        action_counts[action] += 1
        total_cost += costs[action]
        outcomes = {
            "NO_CHANGE": row["free_correct"],
            "CONSTRAINED": row["constrained_correct"],
            "NORMALIZED": row["normalized_correct"],
        }
        success = outcomes[action]
        successes += success
        regressions += action != "NO_CHANGE" and not success and row["free_correct"]
        row["policy_action"] = action
        row["policy_correct"] = success
        row["policy_utilities"] = utilities

    return {
        "policy_path": str(policy_path.resolve()),
        "policy_parameter_sha256": policy["parameter_sha256"],
        "successes": successes,
        "total": len(rows),
        "regressions": regressions,
        "cost": total_cost,
        "action_counts": dict(action_counts),
        "note": "Frozen SFT6-trained v7 policy; no fitting or threshold changes on PGE/SFT audit rows.",
    }


def _run_binding(executor: CoreExecutor, policy_path: Path | None = None) -> dict:
    rows = []
    counts = Counter()
    for case in _binding_cases():
        free = _generate(executor, case["raw_prompt"], max_new_tokens=8)
        # The causal contract: every counterfactual must differ only in its
        # query line, with the same fact block and answer marker.
        raw_context = "\n".join(case["raw_prompt"].splitlines()[:-2])
        for counterfactual in case["counterfactual_prompts"].values():
            assert "\n".join(counterfactual.splitlines()[:-2]) == raw_context
        assert len(case["counterfactual_prompts"]) == len(case["candidates"]) - 1
        raw_vector = [
            _candidate_logprob(executor, case["raw_prompt"], value)
            for value in case["candidates"]
        ]
        counterfactual_vectors = {
            query_index: [
                _candidate_logprob(executor, prompt, value)
                for value in case["candidates"]
            ]
            for query_index, prompt in case["counterfactual_prompts"].items()
        }
        normalized_vector = []
        for candidate_index, actual_score in enumerate(raw_vector):
            baselines = [
                scores[candidate_index]
                for _query_index, scores in sorted(counterfactual_vectors.items())
            ]
            normalized_vector.append(actual_score - sum(baselines) / len(baselines))
        raw_index = max(range(len(raw_vector)), key=raw_vector.__getitem__)
        normalized_index = max(
            range(len(normalized_vector)), key=normalized_vector.__getitem__
        )
        raw_choice = case["candidates"][raw_index]
        normalized_choice = case["candidates"][normalized_index]
        raw_scores = dict(zip(case["candidates"], raw_vector))
        normalized_scores = dict(zip(case["candidates"], normalized_vector))
        free_ok = _exact_value(free, case["expected"])
        raw_ok = raw_choice == case["expected"]
        normalized_ok = normalized_choice == case["expected"]
        counts["free"] += free_ok
        counts["raw"] += raw_ok
        # This is deterministic candidate emission, not another model decode.
        counts["constrained"] += raw_ok
        counts["normalized"] += normalized_ok
        counts["realization_failure"] += raw_ok and not free_ok
        counts["normalization_repair"] += normalized_ok and not raw_ok
        counts["normalization_harm"] += raw_ok and not normalized_ok
        free_words = _norm(free).split()
        mentioned = [value for value in case["candidates"] if value in free_words]
        rows.append({
            "id": case["id"],
            "expected": case["expected"],
            "free_output": free,
            "free_correct": free_ok,
            "raw_choice": raw_choice,
            "raw_correct": raw_ok,
            "constrained_output": raw_choice,
            "constrained_correct": raw_ok,
            "normalized_choice": normalized_choice,
            "normalized_correct": normalized_ok,
            "raw_scores": raw_scores,
            "normalized_scores": normalized_scores,
            "raw_score_vector": raw_vector,
            "normalized_score_vector": normalized_vector,
            "counterfactual_score_vectors": counterfactual_vectors,
            "free_candidate": mentioned[0] if len(mentioned) == 1 else None,
        })
    total = len(rows)
    policy = _policy_transfer(rows, policy_path)
    return {
        "definitions": {
            "FREE": "unassisted canonical greedy generation",
            "RAW": "argmax conditional log-probability among the four observed values",
            "CONSTRAINED": (
                "deterministically emit the RAW-selected observed value; "
                "no additional model forward pass"
            ),
            "COUNTERFACTUAL_NORMALIZATION": (
                "actual candidate logP minus its mean logP under every other query; "
                "fact block and realization path held identical"
            ),
            "realization_failure": "RAW selected gold but FREE did not begin with gold",
        },
        "total": total,
        "free_correct": counts["free"],
        "raw_correct": counts["raw"],
        "constrained_correct": counts["constrained"],
        "normalized_correct": counts["normalized"],
        "selection_failures": total - counts["raw"],
        "realization_failures": counts["realization_failure"],
        "normalization_repairs": counts["normalization_repair"],
        "normalization_harms": counts["normalization_harm"],
        "frozen_policy_transfer": policy,
        "rows": rows,
    }


def _run_primitives(executor: CoreExecutor) -> dict:
    copy_words = ("ember", "quartz", "linen", "cobalt", "willow", "violet")
    copy_rows = []
    for word in copy_words:
        prompt = f"<q>Echo exactly this word and nothing else: {word}</q>\n<answer>"
        output = _generate(executor, prompt, max_new_tokens=8)
        copy_rows.append({"word": word, "output": output, "correct": _exact_value(output, word)})

    context_rows = []
    for i, (entity, value) in enumerate(zip(ENTITIES[:8], VALUES[8:16])):
        nonce = f"NX-{7913 + i * 137}"
        prompt = (
            f"<k>The access token for {entity} is {nonce}. Its region is {value}.</k>\n"
            f"<q>What is the access token for {entity}? Return only the token.</q>\n<answer>"
        )
        output = _generate(executor, prompt, max_new_tokens=10)
        context_rows.append({
            "entity": entity, "expected": nonce, "output": output,
            "correct": _norm(output).split()[:2] == _norm(nonce).split()[:2]
            or _norm(nonce) in _norm(output),
        })

    # Two-hop composition: entity -> relay, relay -> destination.
    composition_rows = []
    for i in range(12):
        entity = ENTITIES[i]
        relay = ENTITIES[(i + 5) % len(ENTITIES)]
        destination = VALUES[(i + 7) % len(VALUES)]
        distractor = VALUES[(i + 11) % len(VALUES)]
        prompt = (
            f"<k>{entity} routes through {relay}. {relay} terminates at {destination}. "
            f"{ENTITIES[(i + 1) % len(ENTITIES)]} terminates at {distractor}.</k>\n"
            f"<q>Where does {entity} ultimately terminate? Return only the destination.</q>\n"
            "<answer>"
        )
        free = _generate(executor, prompt, max_new_tokens=10)
        assisted = _generate(executor, prompt, assisted=True, max_new_tokens=10, seed=100 + i)
        composition_rows.append({
            "id": f"composition-{i:02d}", "expected": destination,
            "free_output": free, "free_correct": _exact_value(free, destination),
            "assisted_output": assisted, "assisted_correct": _exact_value(assisted, destination),
        })

    return {
        "exact_copy": {"correct": sum(r["correct"] for r in copy_rows), "total": len(copy_rows), "rows": copy_rows},
        "nonce_context": {"correct": sum(r["correct"] for r in context_rows), "total": len(context_rows), "rows": context_rows},
        "composition": {
            "free_correct": sum(r["free_correct"] for r in composition_rows),
            "assisted_correct": sum(r["assisted_correct"] for r in composition_rows),
            "total": len(composition_rows), "rows": composition_rows,
        },
    }


def _run_language(executor: CoreExecutor) -> dict:
    raw = [_generate(executor, prompt, max_new_tokens=48) for prompt in CREATIVE_PROMPTS]
    assisted = [
        _generate(executor, prompt, assisted=True, max_new_tokens=48, seed=700 + i)
        for i, prompt in enumerate(CREATIVE_PROMPTS)
    ]
    return {
        "raw": {"metrics": _ngram_metrics(raw), "outputs": raw},
        "assisted": {"metrics": _ngram_metrics(assisted), "outputs": assisted},
    }


@torch.inference_mode()
def _run_validation(executor: CoreExecutor, validation_dir: Path,
                    windows_per_domain: int = 8, window_tokens: int = 256) -> dict:
    domains: dict[str, list[Path]] = {}
    for path in sorted(validation_dir.glob("*.npy")):
        domain = path.stem.rsplit("-", 1)[0]
        domains.setdefault(domain, []).append(path)
    result = {}
    total_nll = 0.0
    total_tokens = 0
    for domain, paths in domains.items():
        losses = []
        for index in range(windows_per_domain):
            path = paths[index % len(paths)]
            data = np.load(path, mmap_mode="r")
            max_start = max(0, len(data) - window_tokens - 1)
            start = 0 if max_start == 0 else ((index + 1) * 104729) % max_start
            seq = np.asarray(data[start:start + window_tokens + 1], dtype=np.int64)
            ids = torch.from_numpy(seq[:-1].copy())[None, :]
            targets = torch.from_numpy(seq[1:].copy()).to(executor.device)
            logits = executor.forward(ids).logits[0].float()
            loss_sum = F.cross_entropy(logits, targets, reduction="sum").item()
            losses.append(loss_sum / len(targets))
            total_nll += loss_sum
            total_tokens += len(targets)
        mean_loss = sum(losses) / len(losses)
        result[domain] = {
            "windows": len(losses), "tokens": len(losses) * window_tokens,
            "loss": round(mean_loss, 6), "perplexity": round(math.exp(min(mean_loss, 20)), 4),
        }
    overall = total_nll / total_tokens
    return {
        "protocol": f"{windows_per_domain} deterministic windows/domain x {window_tokens} targets",
        "overall_loss": round(overall, 6),
        "overall_perplexity": round(math.exp(min(overall, 20)), 4),
        "tokens": total_tokens,
        "domains": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--label", required=True)
    parser.add_argument("--validation-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-legacy-unverified", action="store_true")
    parser.add_argument("--policy")
    args = parser.parse_args()

    started = time.time()
    executor = CoreExecutor.from_checkpoint(
        args.checkpoint,
        device=args.device,
        allow_legacy_unverified=args.allow_legacy_unverified,
    )
    identity = executor.checkpoint_identity
    report = {
        "schema": SCHEMA,
        "label": args.label,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "identity": identity.to_dict(),
        "protocol": {
            "device": str(executor.device), "dtype": executor.dtype_str,
            "generation": "canonical stateful prefill + incremental decode",
            "raw_decode": "greedy, repetition_penalty=1.0, no_repeat_ngram=0",
            "assisted_decode": "temperature=0.8, top_p=0.92, repetition_penalty=1.15, no_repeat_ngram=4",
        },
        "validation": _run_validation(executor, Path(args.validation_dir)),
        "primitives": _run_primitives(executor),
        "binding": _run_binding(
            executor,
            Path(args.policy) if args.policy else None,
        ),
        "language": _run_language(executor),
    }
    report["elapsed_seconds"] = round(time.time() - started, 3)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {
        "label": args.label,
        "step": identity.global_step,
        "validation_loss": report["validation"]["overall_loss"],
        "copy": report["primitives"]["exact_copy"]["correct"],
        "context": report["primitives"]["nonce_context"]["correct"],
        "composition_free": report["primitives"]["composition"]["free_correct"],
        "binding_free": report["binding"]["free_correct"],
        "binding_raw": report["binding"]["raw_correct"],
        "binding_normalized": report["binding"]["normalized_correct"],
        "realization_failures": report["binding"]["realization_failures"],
        "elapsed_seconds": report["elapsed_seconds"],
        "output": str(output.resolve()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
