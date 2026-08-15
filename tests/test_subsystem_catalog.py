from __future__ import annotations

from runtime.subsystem_catalog import (
    build_subsystem_catalog,
    subsystem_records,
    validate_subsystem_catalog,
)


def test_subsystem_catalog_is_valid_and_dependency_closed() -> None:
    records = subsystem_records()
    assert validate_subsystem_catalog(records) == []
    assert len({record.subsystem_id for record in records}) == len(records)


def test_catalog_separates_execution_from_promotion_evidence() -> None:
    rows = {record.subsystem_id: record for record in subsystem_records()}
    assert rows["dense_v4"].lifecycle == "active"
    assert rows["dense_v4"].promotion_eligible is False
    assert rows["mtp"].lifecycle == "pilot"
    assert rows["moe"].lifecycle == "disabled"
    assert rows["tokenizer_v3"].lifecycle == "retired"
    assert rows["cross_colab_sparse_averaging"].lifecycle == "retired"


def test_catalog_exposes_exact_known_parameter_costs() -> None:
    rows = {record.subsystem_id: record for record in subsystem_records()}
    assert rows["dense_v4"].parameter_delta == 181_132_071
    assert rows["mtp"].parameter_delta == 1_607_424
    assert rows["moe"].parameter_delta == 941_488_128


def test_system_catalog_wire_shape_is_versioned() -> None:
    catalog = build_subsystem_catalog()
    assert catalog["schema"] == "anra-subsystem-catalog/v1"
    assert catalog["valid"] is True
    assert catalog["records"]

