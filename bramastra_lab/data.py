"""Small, paired binding worlds and an explicit byte/terminal contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import random
import string

PAD, BOS, EOS, UNK = 0, 1, 2, 3
VOCAB_SIZE = 260


def encode(text: str) -> list[int]:
    return [value + 4 for value in text.encode("utf-8")]


def decode(ids: list[int]) -> str:
    if any(type(value) is not int or not 4 <= value < VOCAB_SIZE for value in ids):
        raise ValueError("only byte IDs may be decoded as text")
    return bytes(value - 4 for value in ids).decode("utf-8", errors="strict")


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class Example:
    world_id: str
    prompt: str
    answer: str
    query_index: int
    split: str


def build_worlds(
    *, seed: int, count: int, entities: int = 2, split: str = "train",
    style: str = "compact", exclude_ids: set[str] | None = None,
) -> list[Example]:
    """All queries for each independent world; identity ignores its rendering."""
    if count <= 0 or not 2 <= entities <= 10:
        raise ValueError("positive count and 2..10 entities required")
    if style not in {"compact", "alternate"}:
        raise ValueError("unknown rendering")
    if not split:
        raise ValueError("split must be named")
    rng = random.Random(seed)
    seen = set(exclude_ids or ())
    examples: list[Example] = []
    attempts = 0
    while len(examples) < count * entities:
        attempts += 1
        if attempts > count * 100 + 10_000:
            raise ValueError("could not obtain enough distinct worlds")
        keys = rng.sample(string.ascii_uppercase, entities)
        values = rng.sample(list(string.digits), entities)
        facts = list(zip(keys, values))
        identity = digest(sorted(facts))
        if identity in seen:
            continue
        seen.add(identity)
        for index, (key, answer) in enumerate(facts):
            if style == "compact":
                context = ";".join(f"{name}={value}" for name, value in facts)
                prompt = f"{context};Q={key};V="
            else:
                context = "|".join(f"{name}:{value}" for name, value in reversed(facts))
                prompt = f"{context}|lookup({key}):"
            examples.append(Example(identity, prompt, answer, index, split))
    return examples


def assert_disjoint(*groups: list[Example]) -> None:
    seen: set[str] = set()
    for group in groups:
        identities = {row.world_id for row in group}
        if seen.intersection(identities):
            raise ValueError("latent world crosses partitions")
        seen.update(identities)


def dataset_hash(examples: list[Example]) -> str:
    return digest([asdict(row) for row in examples])


def make_batch(examples: list[Example], *, max_length: int, device="cpu") -> dict:
    import torch

    if not examples or max_length < 3:
        raise ValueError("nonempty examples and a valid length required")
    tokens = torch.full((len(examples), max_length), PAD, dtype=torch.long)
    target_mask = torch.zeros_like(tokens, dtype=torch.bool)
    for index, row in enumerate(examples):
        if not row.answer:
            raise ValueError("answer must not be empty")
        prefix = [BOS] + encode(row.prompt)
        answer = encode(row.answer)
        sequence = prefix + answer + [EOS]
        if len(sequence) > max_length:
            raise ValueError("record exceeds context; truncation is forbidden")
        tokens[index, :len(sequence)] = torch.tensor(sequence)
        target_mask[index, len(prefix):len(sequence)] = True
    return {"tokens": tokens.to(device), "target_mask": target_mask.to(device)}
