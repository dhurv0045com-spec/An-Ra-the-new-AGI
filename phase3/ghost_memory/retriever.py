"""
45P Ghost State Memory — similarity search over decompressed vectors.

Retrieval decompresses each stored blob to float32, L2-normalizes for cosine
similarity against the query embedding, then applies temporal decay to the
combined score before thresholding and top-K selection.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import decay as decay_mod
from .config import GhostConfig
from .quantizer import decompress_vector


def retrieve_memories(
    query_text: str,
    rows: Sequence[Tuple[int, float, str, str, bytes]],
    embed: Callable[[str], np.ndarray],
    config: GhostConfig,
    reference_time: Optional[float] = None,
) -> List[Dict[str, object]]:
    """
    Rank stored memories by cosine similarity to the query embedding times decay.

    Parameters
    ----------
    query_text:
        User (or system) query string to embed.
    rows:
        Sequence of (memory_id, created_at_unix, role, text, compressed_blob).
    embed:
        Produces a float32 vector of shape (embedding_dim,).
    config:
        Threshold, top_k, half-life, etc.

    Returns
    -------
    List of dicts with keys: memory_id, text, role, score, similarity, decay,
    sorted by score descending.
    """
    q = np.asarray(embed(query_text), dtype=np.float32).ravel()
    qn = np.linalg.norm(q, ord=2)
    if qn > 1e-12:
        q = q / qn
    now = reference_time if reference_time is not None else time.time()
    scored: List[Tuple[float, Dict[str, object]]] = []

    for mid, created_at, role, text, blob in rows:
        vec = decompress_vector(blob)
        vec = np.asarray(vec, dtype=np.float32).ravel()
        vn = np.linalg.norm(vec, ord=2)
        if vn > 1e-12:
            vec = vec / vn
        sim = float(np.dot(q, vec)) if q.shape == vec.shape else 0.0
        d = decay_mod.apply_decay(created_at, reference_time=now, half_life_days=config.decay_half_life_days)
        eff = float(sim * d)
        if sim < config.similarity_thresh:
            continue
        scored.append(
            (
                eff,
                {
                    "memory_id": int(mid),
                    "text": text,
                    "role": role,
                    "score": eff,
                    "similarity": sim,
                    "decay": d,
                },
            )
        )

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[: config.top_k]
    return [t[1] for t in top]
