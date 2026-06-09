"""
anra/inference — Token generation and inference strategies.

Generation logic lives in generate.py.
This package provides the clean import path.
"""

from __future__ import annotations

from typing import Any


def generate_traced(*args: Any, **kwargs: Any) -> Any:
    from generate import generate_traced as _generate_traced

    return _generate_traced(*args, **kwargs)


def generate_stream(*args: Any, **kwargs: Any) -> Any:
    from generate import generate_stream as _generate_stream

    return _generate_stream(*args, **kwargs)


__all__ = ["generate_traced", "generate_stream"]
