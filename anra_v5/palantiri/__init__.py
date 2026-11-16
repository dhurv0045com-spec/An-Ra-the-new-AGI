"""Voice-friendly `palantiri` alias re-exporting the full `anra_v5.planthir` public API."""

from __future__ import annotations

import anra_v5.planthir as _planthir
from anra_v5.planthir import *  # noqa: F403

__all__ = list(_planthir.__all__)
