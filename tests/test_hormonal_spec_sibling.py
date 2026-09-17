"""Sibling-spec receipt: experimental composite vs. frozen ModelSpec."""
from dataclasses import fields, replace

import pytest

from v5_contracts.model_spec import V5A_250M
from v5_contracts.model_spec_hormonal import V5A_250M_HORMONAL_V1


def test_sibling_spec_differs_from_frozen_modelspec():
    assert V5A_250M_HORMONAL_V1.family == "dense-decoder-transformer-hormonal-v1"
    assert V5A_250M_HORMONAL_V1.base == V5A_250M
    assert V5A_250M_HORMONAL_V1.sha256() != V5A_250M.sha256()
    base_total = V5A_250M.parameter_receipt().total
    receipt = V5A_250M_HORMONAL_V1.parameter_receipt()
    assert receipt["projection"] == V5A_250M.query_heads * 7
    assert receipt["total"] == base_total + V5A_250M.query_heads * 7


def test_frozen_spec_untouched_by_import():
    assert V5A_250M.family == "dense-decoder-transformer"
    assert V5A_250M.vocabulary_size == 24_576
    assert V5A_250M.layers == 26


def test_hash_binds_every_field():
    variant = replace(V5A_250M_HORMONAL_V1)
    assert variant.sha256() == V5A_250M_HORMONAL_V1.sha256()
    changed = replace(V5A_250M_HORMONAL_V1, family="dense-decoder-transformer-hormonal-v2")
    assert changed.sha256() != V5A_250M_HORMONAL_V1.sha256()
    with pytest.raises(TypeError):
        replace(V5A_250M_HORMONAL_V1, nonexistent_field=1)
