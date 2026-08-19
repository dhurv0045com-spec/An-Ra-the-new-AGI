from __future__ import annotations

import pytest

from anra_core.errors import RepresentationIncompatibleError
from anra_core.tokenizer import V4Tokenizer


def test_schema9_contract_without_status_marker_remains_compatible() -> None:
    tokenizer = V4Tokenizer.load_canonical()
    contract = tokenizer.identity(probe_count=8)

    # Historical schema-9 checkpoints carry the complete identity but no
    # later-added ``available`` marker.
    tokenizer.assert_checkpoint_contract(contract)


def test_explicit_unavailable_contract_is_rejected() -> None:
    tokenizer = V4Tokenizer.load_canonical()
    contract = tokenizer.identity(probe_count=8)
    contract["available"] = False

    with pytest.raises(RepresentationIncompatibleError, match="unavailable"):
        tokenizer.assert_checkpoint_contract(contract)
