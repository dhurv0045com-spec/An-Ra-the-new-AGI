"""Compile tokenizer-robust null fixtures for the frozen scorer tournament."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "esoes-e2-scoring-policy-fixture/v1"
VOCABULARIES = (16_384, 24_576, 32_768)
GROUPS = 256
PANELS = 2
CONTEXTS_PER_PANEL = 4


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_sha256() -> str:
    normalized = Path(__file__).read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_tokenizer(path: Path) -> Any:
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("tokenizers is required to compile scoring-policy fixtures") from exc
    return Tokenizer.from_str(gzip.decompress(path.read_bytes()).decode("utf-8"))


def _surfaces(index: int, split: str) -> tuple[tuple[str, str, str], str, int]:
    digest = hashlib.sha256(f"{split}:{index}".encode()).hexdigest()
    family = index % 6
    shortest = f" {digest[:24]}"
    fewest_tokens = " evidence" * 6 + f" {index:06d}"
    marked_templates = (
        f" SELECT node_{index:06d} FROM state_log;",
        f" answer := rule_{index:06d}(context_state)",
        f" relation(A_{index:06d}, B_{index + 1:06d})",
        f" C:\\models\\checkpoint_{index:08d}.pt",
        f" Δstate_{index:06d}=previous_{index:06d}; 東京",
        f" rollback(transaction_{index:06d}, semantic_time=17)",
    )
    marked_prefix = marked_templates[family]
    decoy = f" decoy::{digest[24:48]}::{index:06d}"
    return (shortest, fewest_tokens, marked_prefix), decoy, family


def _prompt(split: str, index: int, family: int) -> str:
    marker = hashlib.sha256(f"prompt:{split}:{index}".encode()).hexdigest()[:16]
    return (
        f"Null calibration item {marker}; surface family {family}; task evidence withheld.\n"
        "Question: Choose the candidate supported by the unavailable evidence.\nAnswer:"
    )


def _suffix_tokens(tokenizer: Any, prompt: str, candidate: str) -> tuple[tuple[int, ...], int]:
    encoding = tokenizer.encode(prompt + candidate, add_special_tokens=False)
    boundary = len(prompt)
    if any(left < boundary < right for left, right in encoding.offsets):
        raise ValueError("token crosses the target prompt/candidate boundary")
    suffix_start = next(
        (position for position, (left, _) in enumerate(encoding.offsets) if left >= boundary),
        len(encoding.ids),
    )
    prompt_ids = tuple(encoding.ids[:suffix_start])
    candidate_ids = tuple(encoding.ids[suffix_start:])
    if not prompt_ids or not candidate_ids:
        raise ValueError("target prompt or candidate tokenization is empty")
    if tokenizer.decode(list(encoding.ids), skip_special_tokens=False) != prompt + candidate:
        raise ValueError("joint prompt/candidate tokenization does not round-trip")
    if tokenizer.decode(list(candidate_ids), skip_special_tokens=False) != candidate:
        raise ValueError("candidate suffix does not round-trip independently")
    return candidate_ids, len(prompt_ids)


def _neutral_anchors(tokenizer_sha256: str, vocabulary_size: int) -> tuple[tuple[int, ...], ...]:
    anchors: list[tuple[int, ...]] = []
    used: set[int] = set()
    lower = min(512, vocabulary_size // 4)
    span = vocabulary_size - lower
    for panel in range(PANELS):
        values: list[int] = []
        for context in range(CONTEXTS_PER_PANEL):
            nonce = 0
            while True:
                digest = hashlib.sha256(
                    f"{tokenizer_sha256}:{panel}:{context}:{nonce}".encode()
                ).digest()
                token = lower + int.from_bytes(digest[:8], "big") % span
                if token not in used:
                    used.add(token)
                    values.append(token)
                    break
                nonce += 1
        anchors.append(tuple(values))
    return tuple(anchors)


def _compile_split(
    split: str,
    tokenizers: Mapping[int, Any],
    tokenizer_hashes: Mapping[int, str],
) -> dict[str, object]:
    accepted: list[dict[str, object]] = []
    attempted = 0
    candidate_index = 0 if split == "development" else 1_000_000
    prompt_lengths: dict[int, list[int]] = {vocabulary: [] for vocabulary in VOCABULARIES}
    while len(accepted) < GROUPS and attempted < 100_000:
        index = candidate_index + attempted
        attempted += 1
        candidates, decoy, family = _surfaces(index, split)
        prompt = _prompt(split, index, family)
        if not (
            len(candidates[0].encode("utf-8"))
            < min(len(candidates[1].encode("utf-8")), len(candidates[2].encode("utf-8")))
        ):
            continue
        token_rows: dict[int, dict[str, object]] = {}
        valid = True
        for vocabulary, tokenizer in tokenizers.items():
            try:
                values = [_suffix_tokens(tokenizer, prompt, candidate) for candidate in (*candidates, decoy)]
            except ValueError:
                valid = False
                break
            token_counts = [len(item[0]) for item in values[:3]]
            first_tokens = [item[0][0] for item in values[:3]]
            if token_counts[1] != min(token_counts) or token_counts.count(token_counts[1]) != 1:
                valid = False
                break
            if len(set(first_tokens)) != 3 or len({item[1] for item in values}) != 1:
                valid = False
                break
            token_rows[vocabulary] = {
                "candidate_token_counts": token_counts,
                "candidate_first_tokens": first_tokens,
                "prompt_tokens": values[0][1],
                "decoy_tokens": len(values[3][0]),
            }
        if not valid:
            continue
        group = len(accepted)
        hidden_answer_role = group % 3
        accepted.append(
            {
                "group": group,
                "source_index": index,
                "surface_family": family,
                "hidden_answer_role": hidden_answer_role,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "candidate_sha256": [hashlib.sha256(value.encode("utf-8")).hexdigest() for value in candidates],
                "decoy_sha256": hashlib.sha256(decoy.encode("utf-8")).hexdigest(),
                "tokenizers": {str(key): value for key, value in sorted(token_rows.items())},
            }
        )
        for vocabulary, row in token_rows.items():
            prompt_lengths[vocabulary].append(int(row["prompt_tokens"]))
    if len(accepted) != GROUPS:
        raise RuntimeError(f"only compiled {len(accepted)} of {GROUPS} required groups")
    family_counts = {
        str(family): sum(item["surface_family"] == family for item in accepted)
        for family in range(6)
    }
    hidden_counts = {
        str(role): sum(item["hidden_answer_role"] == role for item in accepted)
        for role in range(3)
    }
    anchors = {
        str(vocabulary): [list(panel) for panel in _neutral_anchors(tokenizer_hashes[vocabulary], vocabulary)]
        for vocabulary in VOCABULARIES
    }
    checks = {
        "exact_group_count": len(accepted) == GROUPS,
        "surface_families_balanced_to_one": max(family_counts.values()) - min(family_counts.values()) <= 1,
        "hidden_labels_balanced_to_one": max(hidden_counts.values()) - min(hidden_counts.values()) <= 1,
        "unique_shortest_utf8_role": True,
        "unique_fewest_token_role_every_tokenizer": True,
        "distinct_first_token_roles_every_tokenizer": True,
        "three_rotations_declared": True,
        "neutral_panels_disjoint": all(
            set(value[0]).isdisjoint(value[1]) for value in anchors.values()
        ),
        "neutral_contexts_exact_length_constructible": all(prompt_lengths.values()),
    }
    return {
        "split": split,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "groups": len(accepted),
        "attempted_candidates": attempted,
        "fixture_sha256": _canonical_sha256(accepted),
        "surface_family_counts": family_counts,
        "hidden_answer_role_counts": hidden_counts,
        "prompt_token_ranges": {
            str(vocabulary): [min(lengths), max(lengths)]
            for vocabulary, lengths in prompt_lengths.items()
        },
        "neutral_anchor_token_ids": anchors,
        "checks": checks,
    }


def compile_fixtures(artifact_directory: Path) -> dict[str, object]:
    tokenizers: dict[int, Any] = {}
    tokenizer_hashes: dict[int, str] = {}
    for vocabulary in VOCABULARIES:
        artifact = artifact_directory / f"tokenizer-{vocabulary}.json.gz"
        tokenizer = _load_tokenizer(artifact)
        if tokenizer.get_vocab_size() != vocabulary:
            raise ValueError("tokenizer vocabulary does not match fixture contract")
        tokenizers[vocabulary] = tokenizer
        tokenizer_hashes[vocabulary] = _sha256_file(artifact)
    development = _compile_split("development", tokenizers, tokenizer_hashes)
    fresh = _compile_split("fresh", tokenizers, tokenizer_hashes)
    checks = {
        "tokenizers_loaded": len(tokenizers) == len(VOCABULARIES),
        "development_passes": development["status"] == "PASS",
        "fresh_passes": fresh["status"] == "PASS",
        "development_fresh_identities_differ": development["fixture_sha256"] != fresh["fixture_sha256"],
        "no_model_execution": True,
        "no_training_performed": True,
    }
    return {
        "schema": SCHEMA,
        "status": "PASS_FIXTURE_COMPILATION" if all(checks.values()) else "FAIL",
        "implementation_sha256": _source_sha256(),
        "tokenizers": {str(key): value for key, value in sorted(tokenizer_hashes.items())},
        "development": development,
        "fresh": fresh,
        "checks": checks,
        "promotion_authorized": False,
        "limitations": [
            "Fixture compilation checks measurement geometry only; no model was executed.",
            "Fresh fixture identity is committed but fresh model outcomes remain forbidden before immutable development selection.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compile_fixtures(args.artifact_directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": result["status"]}, sort_keys=True))
    return 0 if result["status"] == "PASS_FIXTURE_COMPILATION" else 1


if __name__ == "__main__":
    raise SystemExit(main())
