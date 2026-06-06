"""
anra/inference — Token generation and inference strategies.

generate.py at the project root contains the generation logic.
This package provides the clean import path.

Usage:
    from anra.inference import generate_traced, generate_stream
"""

from __future__ import annotations


def generate_traced(*args, **kwargs):
    from generate import generate_traced as _generate_traced

    return _generate_traced(*args, **kwargs)


def generate_stream(*args, **kwargs):
    from generate import generate_stream as _generate_stream

    return _generate_stream(*args, **kwargs)

__all__ = ["generate_traced", "generate_stream"]
