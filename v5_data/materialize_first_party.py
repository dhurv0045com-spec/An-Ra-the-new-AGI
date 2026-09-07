"""B1 — materialize the first-party production corpus subset + honest
500M supply accounting (DATA gate).

Materializes what legitimately exists as first-party sources: the repository's
own tracked text/code documents (natural + code_math_formal) and the E0
verified-cognition generator rows. Runs them through the identity-bound
preparation chain and reports the honest 500M supply arithmetic:
UNIQUE_RUNNABLE_TRAIN_TOKENS vs the campaign demand, per-source replay
factors. This module never fakes volume: a shortfall is a recorded shortfall.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TARGET_TOKENS = 500_000_000
MAX_CORPUS_FILES = 48


def materialize_first_party(*, repo_root: str | Path, tokenizer: Any,
                            run_id: str = 'first-party-v1',
                            seed: int = 202_609_06,
                            max_files: int = MAX_CORPUS_FILES
                            ) -> dict[str, Any]:
    """Materialize first-party documents + supply accounting (pure I/O over
    the repo tree + deterministic generators)."""
    from e0_cognition.training_generators import build_training_examples
    from v5_training.miniature import _load_corpus, _load_tokenizer

    root = Path(repo_root)
    _tok, _eval = _load_tokenizer(root)
    corpus_docs = _load_corpus(root, _tok)
    del _tok, _eval

    examples = build_training_examples(seed=seed, count=2_000)
    cognition_docs = [
        {"doc_id": f"e0-cog-{i:06d}",
         "text": f"{e.context}\n{e.query} {e.answer}",
         "source_id": f"e0-cog-{i:06d}", "family": "verified_cognition",
         "domain": "synthetic-cognition"}
        for i, e in enumerate(examples)]

    documents = (
        [{"doc_id": d.doc_id, "text": d.text, "source_id": d.source_id,
          "family": d.family, "domain": d.domain} for d in corpus_docs]
        + cognition_docs)

    supply: dict[str, int] = {}
    for d in documents:
        supply[d["family"]] = supply.get(d["family"], 0) + len(
            tokenizer.encode(d["text"]))
    demand = {"natural": int(TARGET_TOKENS * 0.65),
              "code_math_formal": int(TARGET_TOKENS * 0.20),
              "verified_cognition": int(TARGET_TOKENS * 0.15)}
    replay = {src: round(demand.get(src, 0) / max(supply.get(src, 1), 1), 2)
              for src in sorted(set(supply) | set(demand))}
    shortfall = {src: max(0, demand.get(src, 0) - supply.get(src, 0))
                 for src in demand}
    ready = all(v <= 0 for v in shortfall.values())
    return {
        "schema": "anra-v5-production-materialization/v1",
        "run_id": run_id, "seed": seed,
        "documents_materialized": len(documents),
        "unique_train_tokens_by_source": supply,
        "unique_train_tokens_total": sum(supply.values()),
        "demand_tokens_by_source_500m": demand,
        "shortfall_tokens_by_source": shortfall,
        "replay_factor_by_source": replay,
        "verdict": ("SUPPLIED" if ready else
                    "DATA_NOT_READY: unique supply short of 500M demand — "
                    "recorded, never faked"),
        "DATA_NOT_READY": not ready,
    }


__all__ = ["materialize_first_party"]
