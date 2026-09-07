"""Shared miniature data/tokenizer helpers for the production-path canaries.

Binds the frozen 24,576-entry tokenizer artifact and the real first-party
corpus (current-HEAD tracked text files, hashed as read) so the miniature and
the accelerator canaries run identical, provenance-bound data loading.
"""

from __future__ import annotations

from pathlib import Path

from v5_data.corpus_loading import (
    MAX_CORPUS_FILES,
    MAX_DOCUMENT_TOKENS,
    _HFTokenizerBackend,
    _canonical_json,
    _load_corpus,
    _load_tokenizer,
    _sha256_file,
)
from v5_data.manifest import Document
from v5_tokenizer.adapter import FrozenTokenizer, TokenizerIdentity


SPLITS = {"training": 0.7, "development": 0.2, "sealed": 0.05, "fresh": 0.05}

MINIATURE_EVAL_TASKS = [
    {
        "task_id": "mini-bind-001",
        "cluster_id": "mini-bind",
        "family": "query_binding",
        "split": "software_eval",
        "difficulty": "easy",
        "prompt": "The zibble is crimson. The woggle is blue. What color is the zibble?",
        "candidates": (" crimson", " blue"),
        "gold": " crimson",
    },
    {
        "task_id": "mini-bind-002",
        "cluster_id": "mini-bind",
        "family": "query_binding",
        "split": "software_eval",
        "difficulty": "easy",
        "prompt": "The zibble is crimson. The woggle is blue. What color is the woggle?",
        "candidates": (" crimson", " blue"),
        "gold": " blue",
    },
]



__all__ = [
    "MINIATURE_EVAL_TASKS",
    "SPLITS",
    "_HFTokenizerBackend",
    "_canonical_json",
    "_load_corpus",
    "_load_tokenizer",
    "_sha256_file",
    "_source_commit",
]


def _source_commit(repo: Path) -> str:
    import subprocess

    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        value = "0" * 40
    return value if len(value) == 40 and all(c in "0123456789abcdef" for c in value) else "0" * 40
