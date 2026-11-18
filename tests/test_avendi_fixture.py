"""Acceptance tests for Avendi V8-Y2 P0 instrument, oracle parser, custody, and registry lifecycle."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import pytest

from anra_v5.elendril.avendi.oracle import (
    evaluate_reference_policies_on_prompt,
    parse_tokenized_public_observation,
    run_pilot_reference_check,
    verify_target_tokens_have_eos,
    walk_parsed_graph,
)
from anra_v5.elendril.avendi.protocol import (
    DESIGN_SHA,
    build_registry_document,
    load_registry_document,
    save_registry_document,
    update_registry_lifecycle,
)
from anra_v5.elendril.avendi.render import (
    RenderedRow,
    encode_prompt_and_target,
    erase_goal_token,
    extract_prompt_prefix_from_framed_tokens,
    mask_bridge_edges_in_prompt,
    render_avendi_prompt,
)
from anra_v5.elendril.avendi.worlds import (
    audit_avendi_custody_and_overlap,
    build_avendi_suite,
    evaluate_policies_on_families,
    plant_custody_collision_negative_control,
    run_destructive_input_checks,
    verify_p0_instrument,
)
from anra_v5.elendril.representation import UTF8, SpecialToken


def test_p0_oracle_truth_table_bijection_metadata_and_rejection() -> None:
    suite = build_avendi_suite(scale_factor=32)
    dev = suite["splits"]["dev"]

    # 1. Verify 4-cell crossover truth table (G, H, H, G) and independent oracle agreement
    for fam in (*dev["one_hop_families"], *dev["three_hop_families"]):
        cmap = fam.cell_map()
        assert cmap["00"].expected_answer == fam.endpoint_g
        assert cmap["10"].expected_answer == fam.endpoint_h
        assert cmap["01"].expected_answer == fam.endpoint_h
        assert cmap["11"].expected_answer == fam.endpoint_g
        for cell in fam.cells:
            parsed = parse_tokenized_public_observation(cell.prompt_tokens)
            walked = walk_parsed_graph(parsed)
            assert walked["action_text"] == cell.expected_answer

    # 2. Alias bijection changes surface tokens and maps back to canonical answer; non-A-P shift changes answer
    fam0 = dev["three_hop_families"][0]
    c00 = fam0.cell_map()["00"]
    p00 = parse_tokenized_public_observation(c00.prompt_tokens)
    # Permuting timestamps preserves gold answer while changing rank-h timestamp order
    rev_ticks = [r.tick for r in reversed(p00.rows)]
    perm_rows = [
        RenderedRow(subject=r.subject, value=r.value, tick=rev_ticks[i], source=r.source)
        for i, r in enumerate(p00.rows)
    ]
    perm_prompt = render_avendi_prompt(query_subject=c00.query_subject, hops=3, rows=perm_rows)
    perm_enc = encode_prompt_and_target(perm_prompt, c00.expected_answer)
    assert walk_parsed_graph(parse_tokenized_public_observation(perm_enc["prompt_tokens"]))["action_text"] == c00.expected_answer

    # Symbol bijection changes expected answer correspondingly
    from anra_v5.elendril.avendi.module import AvendiEnvironmentKernel, case_to_planthir_world
    kernel = AvendiEnvironmentKernel()
    w0 = case_to_planthir_world(c00)
    w_perm = kernel.permute_world(w0, shift=3)
    assert w_perm["hidden"]["expected"] != w0["hidden"]["expected"]
    assert walk_parsed_graph(parse_tokenized_public_observation(w_perm["prompt_tokens"]))["action_text"] == w_perm["hidden"]["expected"]

    # 3. Reject truncated prompt, overlength prompt, and malformed EOS
    with pytest.raises(ValueError, match="terminate at"):
        parse_tokenized_public_observation(c00.prompt_tokens[:-3])
    with pytest.raises(ValueError, match="exceeds bound"):
        parse_tokenized_public_observation([UTF8.bos_id] + [69] * 520)
    with pytest.raises(ValueError, match="final token is not EOS"):
        verify_target_tokens_have_eos((69, 70))


def test_p0_custody_collision_destructive_checks_and_reference_policies() -> None:
    p0 = verify_p0_instrument(scale_factor=32)
    assert p0["status"] == "QUALIFIED"
    assert p0["verification_row_status"] == "PASS"
    assert p0["custody_audit"]["status"] == "PASS"
    assert p0["planted_collision_negative_control"]["status"] == "PASS"
    assert p0["planted_collision_negative_control"]["caught_prompt_prefix_overlap"] is True
    assert p0["planted_collision_negative_control"]["caught_instantiated_state_overlap"] is True
    assert p0["destructive_checks"]["one_hop"]["status"] == "PASS"
    assert p0["destructive_checks"]["three_hop"]["status"] == "PASS"

    # Pilot 1801 exact reproduction
    pilot = p0["pilot_reference_1801"]
    assert pilot["all_four_correct"]["traversal"] == 256
    assert pilot["all_four_correct"]["last_object"] == 0
    assert pilot["all_four_correct"]["third_timestamp"] == 0
    assert pilot["all_four_correct"]["ignore_query_oracle_elsewhere"] == 0
    assert pilot["all_four_correct"]["ignore_bridge_oracle_elsewhere"] == 0


def test_registry_design_hash_survives_status_lifecycle_and_save_load(tmp_path: Path) -> None:
    doc = build_registry_document()
    orig_design_sha = doc["design_sha256"]
    orig_planthir_sha = doc["lifecycle"]["planthir_registration_sha256"]
    assert orig_design_sha == DESIGN_SHA

    updated = update_registry_lifecycle(
        doc,
        experiment_status="COMPLETE",
        result_ref="avendi-fixture-test",
        stage_statuses={"P0_instrument": "QUALIFIED", "P1_restoration": "BLOCKED"},
        scenario_updates={"01": [{"status": "TESTED_ON_P0", "evidence_id": "avendi-fixture-test"}]},
    )
    assert updated["design_sha256"] == orig_design_sha
    assert updated["lifecycle"]["planthir_registration_sha256"] == orig_planthir_sha
    assert updated["lifecycle"]["planthir_entry"]["registration_sha256"] == orig_planthir_sha
    assert len(updated["lifecycle"]["planthir_entry"]["status_events"]) == 2

    reg_path = tmp_path / "REGISTRY.json"
    save_registry_document(updated, reg_path)
    reloaded = load_registry_document(reg_path)
    assert reloaded["design_sha256"] == orig_design_sha
    assert reloaded["lifecycle"]["stage_statuses"]["P0_instrument"] == "QUALIFIED"
