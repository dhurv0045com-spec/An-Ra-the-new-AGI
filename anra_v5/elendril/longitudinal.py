"""Paired development contrasts for future acquisition-qualified studies."""

from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True, slots=True)
class Work:
    """Cumulative work for one fixed model and execution protocol."""

    training_tokens: int
    evaluation_positions: int
    memory_calls: int = 0
    tool_calls: int = 0

    def __post_init__(self) -> None:
        if any(type(count) is not int or count < 0 for count in self.counts()):
            raise ValueError("work counters must be nonnegative integers")

    def counts(self) -> tuple[int, ...]:
        return self.training_tokens, self.evaluation_positions, self.memory_calls, self.tool_calls


@dataclass(frozen=True, slots=True)
class Probe:
    fixture_sha256: str
    model_sha256: str
    seed: int
    work: Work
    clusters: tuple[tuple[str, int, int], ...]

    def __post_init__(self) -> None:
        for digest in (self.fixture_sha256, self.model_sha256):
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError("probe requires SHA-256 identities")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("probe requires a nonnegative seed")
        if not self.clusters or len({key for key, _, _ in self.clusters}) != len(self.clusters):
            raise ValueError("probe requires unique nonempty clusters")
        for key, correct, rows in self.clusters:
            if not key or type(correct) is not int or type(rows) is not int or not 0 <= correct <= rows or not rows:
                raise ValueError("invalid cluster counts")

    @classmethod
    def from_score(cls, *, fixture_sha256: str, model_sha256: str, seed: int,
                   work: Work, metrics: dict) -> "Probe":
        clusters = tuple((key, counts[0], counts[1]) for key, counts in sorted(metrics["clusters"].items()))
        return cls(fixture_sha256, model_sha256, seed, work, clusters)


def _paired_delta(reference: Probe, candidate: Probe) -> dict:
    if (reference.fixture_sha256, reference.model_sha256, reference.seed) != (
        candidate.fixture_sha256, candidate.model_sha256, candidate.seed
    ):
        raise ValueError("contrast requires the same fixture, model spec and seed")
    reference_counts = {key: (correct, rows) for key, correct, rows in reference.clusters}
    candidate_counts = {key: (correct, rows) for key, correct, rows in candidate.clusters}
    if {key: rows for key, (_, rows) in reference_counts.items()} != {
        key: rows for key, (_, rows) in candidate_counts.items()
    }:
        raise ValueError("contrast requires identical cluster coverage")
    deltas = {key: [candidate_counts[key][0] - correct, rows]
              for key, (correct, rows) in sorted(reference_counts.items())}
    mean = sum(Fraction(delta, rows) for delta, rows in deltas.values()) / len(deltas)
    return {"mean_cluster_delta": [mean.numerator, mean.denominator],
            "cluster_deltas": deltas, "n_clusters": len(deltas)}


def retention_delta(before_b: Probe, after_b: Probe) -> dict:
    """Measure change on A after subsequent training; B learnability is a separate gate."""
    if (after_b.work.training_tokens <= before_b.work.training_tokens
            or any(after < before for before, after in zip(before_b.work.counts(), after_b.work.counts()))):
        raise ValueError("retention requires cumulative work and an intervening training exposure")
    return _paired_delta(before_b, after_b)


def forward_transfer_delta(scratch_c: Probe, prior_then_c: Probe) -> dict:
    """Compare on C at the same cumulative resource counts including earlier stages."""
    if scratch_c.work != prior_then_c.work or not scratch_c.work.training_tokens:
        raise ValueError("forward transfer requires matched positive training and total work")
    return _paired_delta(scratch_c, prior_then_c)
