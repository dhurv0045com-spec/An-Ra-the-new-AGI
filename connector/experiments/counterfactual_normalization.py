"""TRUE counterfactual-query normalization — the ONE shared implementation.

Used by QIM evaluators and MIXED-CAUSAL runners alike. Do not duplicate
this math anywhere else.

For candidate value v_i with candidate index i in a task whose ACTUAL
query is query index a:

    actual_i   = logP(v_i | actual_query,  facts)      # facts byte-identical
    baseline_i = mean over all OTHER query indices j != a of
                 logP(v_i | counterfactual_query_j, facts)
    normalized_i = actual_i - baseline_i

Only the QUERY IDENTITY changes between scoring passes; the fact block is
byte-identical. Gold never enters this computation.

Historical note: an earlier mixed-causal runner computed
    adj_i = raw_i - mean(raw_j for j != i)
which is rank-preserving (argmax identical to raw up to ties) and was NOT
normalization. That formula is rejected by test_pseudo_normalization_rejected.
"""

from __future__ import annotations

from typing import Callable, Sequence


def normalize_scores(
    actual_query_index: int,
    raw_scores: Sequence[float],
    counterfactual_scores: dict[int, Sequence[float]],
) -> list[float]:
    """Compute true normalized scores for one target.

    actual_query_index : which query is the real one (a)
    raw_scores         : logP(v_i | actual query) per candidate i
    counterfactual_scores : {j: [logP(v_i | query_j) per candidate i]}
                            for every other query index j != a

    Returns normalized_i = actual_i - mean_j(logP(v_i | q_j)).
    """
    n = len(raw_scores)
    if not counterfactual_scores:
        raise ValueError("no counterfactual queries supplied")
    bad = [j for j in counterfactual_scores if j == actual_query_index]
    if bad:
        raise ValueError(
            f"counterfactual set must exclude the actual query {bad}")
    out = []
    for i in range(n):
        baselines = []
        for j, per_cand in sorted(counterfactual_scores.items()):
            if len(per_cand) != n:
                raise ValueError("counterfactual row length mismatch")
            baselines.append(per_cand[i])
        baseline = sum(baselines) / len(baselines)
        out.append(raw_scores[i] - baseline)
    return out


def build_counterfactual_queries(
    prompt_builder: Callable[[int], str],
    n_queries: int,
    actual_index: int,
) -> dict[int, str]:
    """Build legal counterfactual prompts: only query identity changes.

    prompt_builder(j) must return the full prompt whose FACT BLOCK is
    byte-identical across all j and whose queried entity is the j-th.
    Returns {j: prompt} for all j != actual_index.
    """
    if not 0 <= actual_index < n_queries:
        raise ValueError("actual_index out of range")
    return {j: prompt_builder(j) for j in range(n_queries) if j != actual_index}


def verify_byte_identical_context(prompts: dict[int, str]) -> None:
    """All prompts must share the same context block (everything before
    the final query line AND the trailing 'Answer:' line). Raises on
    mismatch. The query line itself is expected to differ — that is the
    single variable normalization is allowed to change."""
    def context_of(p: str) -> str:
        lines = p.splitlines()
        # drop trailing 'Answer:' and the query line before it
        assert lines[-1].strip() == "Answer:", "prompt must end with Answer:"
        return "\n".join(lines[:-2])

    prefixes = {context_of(p) for p in prompts.values()}
    if len(prefixes) != 1:
        raise AssertionError(
            f"context blocks differ across counterfactual prompts: "
            f"{len(prefixes)} distinct prefixes")


def rank(scores: Sequence[float], index: int) -> int:
    """Rank (1-based) of scores[index] among all scores."""
    s = scores[index]
    return 1 + sum(1 for j, x in enumerate(scores) if x > s)


def argmax(scores: Sequence[float]) -> int:
    best, bi = None, None
    for i, x in enumerate(scores):
        if best is None or x > best:
            best, bi = x, i
    return bi


def pseudo_normalization_rejected_example() -> tuple[list[float], list[float]]:
    """Return (raw, 'pseudo') demonstrating rank-preservation: the old
    formula subtracts the leave-one-out mean of the SAME vector, which is
    an affine transform of raw -> identical ranking."""
    raw = [0.3, -1.2, 2.5]
    n = len(raw)
    pseudo = []
    for i in range(n):
        others = [raw[j] for j in range(n) if j != i]
        pseudo.append(raw[i] - sum(others) / len(others))
    assert argmax(pseudo) == argmax(raw), "pseudo should be rank-preserving"
    return raw, pseudo
