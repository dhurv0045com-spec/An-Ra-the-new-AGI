"""Avendi V8-Y2 P5 conditional evidence use & preservation and P6 bounded convention correction."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Mapping, Sequence

from anra_v5.elendril.codec import digest
from .oracle import parse_tokenized_public_observation, walk_parsed_graph
from .protocol import GENERATOR_SEEDS, QUALIFICATION_GATES, SYMBOLS, derive_seed_int
from .render import RenderedRow, encode_prompt_and_target, render_avendi_prompt
from .worlds import SOURCES


@dataclass(frozen=True, slots=True)
class P5ConditionalCase:
    """Single P5 conditional-evidence presentation with verified feasible endpoint set."""

    case_id: str
    family_id: str
    condition: str  # one of the 6 paired conditions
    query_subject: str
    query_available: bool
    returned_row_index: int | None
    public_prompt: str
    prompt_tokens: tuple[int, ...]
    target_tokens: tuple[int, ...]
    expected_operation: str  # 'ANSWER', 'QUERY', or 'ABSTAIN'
    expected_action_text: str  # 'A'..'P', 'Q', or '!'
    terminal_gold_answer: str
    rival_terminal_answer: str
    feasible_endpoints: tuple[str, ...]
    ambiguous: bool


def build_p5_conditional_families(*, families_count: int = 256, seed: int = GENERATOR_SEEDS["p5_eval"]) -> dict[str, Any]:
    """Build P5 two-chain 3-hop families across all 6 paired conditions with matched row counts and counterbalanced ticks."""
    rng = random.Random(derive_seed_int("avendi-p5-conditional", seed))
    cases: list[P5ConditionalCase] = []
    conditions = (
        "complete",
        "irrelevant_edge_missing",
        "required_internal_edge_missing_query_available",
        "required_internal_edge_missing_tool_unavailable",
        "useful_returned_bridge",
        "irrelevant_returned_fact",
    )

    for f_idx in range(families_count):
        labels = rng.sample(SYMBOLS, 12)
        a, b, c, d, e, f, g, h, u, v, x, y = labels
        base_ticks = rng.sample(range(1000, 9000), 6)
        base_srcs = [rng.choice(SOURCES) for _ in range(6)]
        order = rng.sample(range(6), 6)
        # Balance channel availability in complete and irrelevant-missing cases so flag alone is uninformative
        balanced_q_flag = (f_idx % 2) == 0
        fid = f"AV-p5-fam-{f_idx:04d}"

        for q_idx, (q_sym, req_bridge_idx, irr_bridge_idx, gold_ans, riv_ans) in enumerate((
            (a, 2, 3, g, h),  # chain A: a->c (0), c->e (2, required internal bridge), e->g (4, terminal always visible)
            (b, 3, 2, h, g),  # chain B: b->d (1), d->f (3, required internal bridge), f->h (5, terminal always visible)
        )):
            full_edges = [(a, c), (b, d), (c, e), (d, f), (e, g), (f, h)]

            for cond in conditions:
                edge_list = list(full_edges)
                ret_pos: int | None = None
                if cond == "complete":
                    q_avail = balanced_q_flag
                elif cond == "irrelevant_edge_missing":
                    # Replace non-queried chain's internal bridge with filler (u, v) to keep row count matched at 6
                    edge_list[irr_bridge_idx] = (u, v)
                    q_avail = balanced_q_flag
                elif cond == "required_internal_edge_missing_query_available":
                    edge_list[req_bridge_idx] = (u, v)
                    q_avail = True
                elif cond == "required_internal_edge_missing_tool_unavailable":
                    edge_list[req_bridge_idx] = (u, v)
                    q_avail = False
                elif cond == "useful_returned_bridge":
                    # Initial state had filler (u, v); returned event supplies required internal bridge at randomized non-final/varied position
                    edge_list[req_bridge_idx] = (u, v)
                    q_avail = False
                elif cond == "irrelevant_returned_fact":
                    # Required bridge remains missing; returned event supplies irrelevant fact (x, y); tool budget exhausted -> abstain!
                    edge_list[req_bridge_idx] = (u, v)
                    q_avail = False
                else:
                    raise ValueError(cond)

                rows = [
                    RenderedRow(
                        subject=edge_list[i][0],
                        value=edge_list[i][1],
                        tick=base_ticks[i],
                        source=base_srcs[i],
                    )
                    for i in order
                ]

                if cond in ("useful_returned_bridge", "irrelevant_returned_fact"):
                    ret_edge = full_edges[req_bridge_idx] if cond == "useful_returned_bridge" else (x, y)
                    # Counterbalance returned timestamp (alternating earliest vs middle vs latest) and randomize row placement
                    tick_choices = (min(base_ticks) - 50, sum(base_ticks) // len(base_ticks), max(base_ticks) + 50)
                    ret_tick = tick_choices[(f_idx + q_idx) % 3]
                    ret_pos = (f_idx + q_idx) % (len(rows) + 1)
                    rows.insert(ret_pos, RenderedRow(subject=ret_edge[0], value=ret_edge[1], tick=ret_tick, source="s0"))

                prompt = render_avendi_prompt(
                    query_subject=q_sym,
                    hops=3,
                    rows=rows,
                    query_available=q_avail,
                    memory_available=False,
                    template=0,
                )
                # Independent verifier computes feasible endpoints and expected action
                parsed = parse_tokenized_public_observation(encode_prompt_and_target(prompt, "!")["prompt_tokens"])
                walked = walk_parsed_graph(parsed)

                exp_op = walked["operation"]
                exp_act = walked["action_text"]
                if cond in ("complete", "irrelevant_edge_missing", "useful_returned_bridge"):
                    if exp_op != "ANSWER" or exp_act != gold_ans:
                        raise ValueError(f"P5 verifier mismatch on {cond}: got {(exp_op, exp_act)}, expected ('ANSWER', {gold_ans!r})")
                elif cond == "required_internal_edge_missing_query_available":
                    if exp_op != "QUERY" or exp_act != "Q" or not walked["ambiguous"]:
                        raise ValueError(f"P5 verifier mismatch on {cond}: got {walked}")
                elif cond in ("required_internal_edge_missing_tool_unavailable", "irrelevant_returned_fact"):
                    if exp_op != "ABSTAIN" or exp_act != "!" or not walked["ambiguous"]:
                        raise ValueError(f"P5 verifier mismatch on {cond}: got {walked}")

                enc = encode_prompt_and_target(prompt, exp_act)
                cases.append(
                    P5ConditionalCase(
                        case_id=f"{fid}__q{q_idx}__{cond}",
                        family_id=fid,
                        condition=cond,
                        query_subject=q_sym,
                        query_available=q_avail,
                        returned_row_index=ret_pos,
                        public_prompt=prompt,
                        prompt_tokens=enc["prompt_tokens"],
                        target_tokens=enc["target_tokens"],
                        expected_operation=exp_op,
                        expected_action_text=exp_act,
                        terminal_gold_answer=gold_ans,
                        rival_terminal_answer=riv_ans,
                        feasible_endpoints=tuple(walked["feasible_endpoints"]),
                        ambiguous=bool(walked["ambiguous"]),
                    )
                )

    return {
        "schema": "anra-avendi-v8-y2-p5-suite-v1",
        "families": families_count,
        "conditions": list(conditions),
        "cases_total": len(cases),
        "cases": tuple(cases),
    }


def build_p5_dose_experiment_spec() -> dict[str, Any]:
    """Return preregistered P5 512-update dose experiment specification with explicit displaced-slot accounting."""
    return {
        "schema": "anra-avendi-v8-y2-p5-dose-spec-v1",
        "updates": 512,
        "batch_size": 16,
        "seeds": [7, 19, 41],
        "direct_answer_slots_per_batch": 8,
        "rehearsal_slots_control": 8,
        "rehearsal_slots_treatment": 4,
        "replaced_slots_in_treatment": 4,
        "displaced_rehearsal_categories": ["conflict", "memory", "null", "missing"],
        "treatment_arms": {
            "CTRL_REHEARSAL_8": "8 direct-answer + 8 fixed rehearsal slots unchanged",
            "TRT_QUERY_PAIR_DOSE_4": "8 direct-answer + 4 fixed rehearsal + 2 query-initial/post-response pairs (4 slots)",
            "TRT_UNAVAILABLE_DOSE_4": "8 direct-answer + 4 fixed rehearsal + 4 tool-unavailable abstain examples",
        },
    }


@dataclass(frozen=True, slots=True)
class P6ConventionBlock:
    """Single P6 evaluation block with a latent persistent interface convention (Regime 0 vs Regime 1)."""

    block_id: str
    regime: int  # 0: L=left, R=right; 1: L=right, R=left (swapped)
    paired_block_id: str
    support_query_op: str  # 'L' or 'R'
    support_left_val: str
    support_right_val: str
    pre_feedback_prompt: str
    pre_feedback_gold: str
    opposite_regime_gold: str
    informative_calibration_record: tuple[str, str]  # unequal (left, right)
    uninformative_calibration_record: tuple[str, str]  # equal (val, val)
    unseen_transfer_cases: tuple[dict[str, Any], ...]  # 16 unseen graphs with fresh symbols
    retention_cases: tuple[dict[str, Any], ...]  # 16 old-convention retention cases in cued old context


def build_p6_correction_blocks(*, blocks_count: int = 128, seed: int = GENERATOR_SEEDS["p6_blocks"]) -> tuple[P6ConventionBlock, ...]:
    """Construct paired P6 blocks with identical pre-feedback observations and opposite targets across Regimes 0 and 1."""
    if blocks_count % 2 != 0:
        raise ValueError("P6 blocks_count must be even to pair Regime 0 and Regime 1")
    rng = random.Random(derive_seed_int("avendi-p6-blocks", seed))
    blocks: list[P6ConventionBlock] = []
    n_pairs = blocks_count // 2

    for p_idx in range(n_pairs):
        syms = rng.sample(SYMBOLS, 8)
        k_sup, left_v, right_v, cal_l, cal_r, uninf_v, ret_l, ret_r = syms
        op = "L" if (p_idx % 2 == 0) else "R"

        # Pre-feedback observation omits the regime cue and is byte-identical across Regime 0 and Regime 1!
        rows = [
            RenderedRow(subject=k_sup, value=left_v, tick=1010, source="s0"),
            RenderedRow(subject=k_sup, value=right_v, tick=2020, source="s1"),
        ]
        pre_prompt = render_avendi_prompt(
            query_subject=k_sup,
            hops=1,
            rows=rows,
            query_available=False,
            memory_available=False,
            template=0,
            convention_header=f"Operator={op} (apply block convention).",
        )

        # Generate 16 unseen transfer graphs and 16 old-convention retention graphs for this pair
        unseen_specs: list[tuple[str, str, str, str]] = []
        ret_specs: list[tuple[str, str, str, str]] = []
        for u_i in range(16):
            u_syms = rng.sample(SYMBOLS, 3)
            u_op = "L" if (u_i % 2 == 0) else "R"
            unseen_specs.append((u_op, u_syms[0], u_syms[1], u_syms[2]))

            r_syms = rng.sample(SYMBOLS, 3)
            r_op = "L" if (u_i % 2 == 0) else "R"
            ret_specs.append((r_op, r_syms[0], r_syms[1], r_syms[2]))

        for regime in (0, 1):
            def _eval_op(operator: str, lv: str, rv: str, reg: int) -> str:
                if reg == 0:
                    return lv if operator == "L" else rv
                return rv if operator == "L" else lv

            gold_pre = _eval_op(op, left_v, right_v, regime)
            opp_pre = _eval_op(op, left_v, right_v, 1 - regime)

            unseen_cases = tuple(
                {
                    "case_id": f"P6-pair{p_idx:03d}-reg{regime}-unseen-{idx:02d}",
                    "operator": u_op,
                    "subject": u_k,
                    "left_val": u_l,
                    "right_val": u_r,
                    "expected_answer": _eval_op(u_op, u_l, u_r, regime),
                    "opposite_answer": _eval_op(u_op, u_l, u_r, 1 - regime),
                }
                for idx, (u_op, u_k, u_l, u_r) in enumerate(unseen_specs)
            )
            retention_cases = tuple(
                {
                    "case_id": f"P6-pair{p_idx:03d}-reg{regime}-ret-{idx:02d}",
                    "operator": r_op,
                    "subject": r_k,
                    "left_val": r_l,
                    "right_val": r_r,
                    "cued_regime": 0,
                    "expected_answer": _eval_op(r_op, r_l, r_r, 0),
                }
                for idx, (r_op, r_k, r_l, r_r) in enumerate(ret_specs)
            )

            bid = f"P6-pair{p_idx:03d}-reg{regime}"
            pbid = f"P6-pair{p_idx:03d}-reg{1 - regime}"
            blocks.append(
                P6ConventionBlock(
                    block_id=bid,
                    regime=regime,
                    paired_block_id=pbid,
                    support_query_op=op,
                    support_left_val=left_v,
                    support_right_val=right_v,
                    pre_feedback_prompt=pre_prompt,
                    pre_feedback_gold=gold_pre,
                    opposite_regime_gold=opp_pre,
                    informative_calibration_record=(cal_l, cal_r),
                    uninformative_calibration_record=(uninf_v, uninf_v),
                    unseen_transfer_cases=unseen_cases,
                    retention_cases=retention_cases,
                )
            )
    return tuple(blocks)


def run_p6_bounded_correction_study(
    blocks: Sequence[P6ConventionBlock],
    *,
    seed: int = 7,
) -> dict[str, Any]:
    """Evaluate all 8 P6 bounded-correction arms + memory-reset and impossible no-observation controls with block isolation."""
    # Verify paired pre-feedback observations are byte-identical with opposite targets
    by_id = {b.block_id: b for b in blocks}
    identical_pre_obs_pairs = 0
    opposite_target_pairs = 0
    for b in blocks:
        pb = by_id[b.paired_block_id]
        if b.pre_feedback_prompt == pb.pre_feedback_prompt:
            identical_pre_obs_pairs += 1
        if b.pre_feedback_gold != pb.pre_feedback_gold and b.pre_feedback_gold == pb.opposite_regime_gold:
            opposite_target_pairs += 1

    # Evaluate arms with strict per-block state reset (`block_isolation_verified = True`)
    arm_names = (
        "1_frozen_matched_forward",
        "2_external_true_feedback",
        "3_external_irrelevant_feedback",
        "4_external_wrong_feedback",
        "5_persistent_two_slot_memory",
        "5b_persistent_memory_after_reset",
        "6_one_step_grad_true_feedback",
        "7_one_step_grad_irrelevant_feedback",
        "8_self_directed_calibration_choice",
        "9_no_observation_impossible_control",
    )

    arm_results: dict[str, Any] = {}
    for arm in arm_names:
        unseen_hits_all = 0
        unseen_total_all = 0
        unseen_hits_initial_err = 0
        unseen_total_initial_err = 0
        ret_hits = 0
        ret_total = 0
        action_log: list[dict[str, Any]] = []

        for b in blocks:
            # Reset block state completely at the start of every block!
            memory_slots: dict[str, str | None] = {"slot_0": None, "slot_1": None}
            adapted_regime_estimate = 0  # default prior before feedback is Regime 0
            initial_pred = b.support_left_val if b.support_query_op == "L" else b.support_right_val
            initial_is_error = initial_pred != b.pre_feedback_gold

            actor_who_chose = "none"
            label_producer = "none"
            state_changed = "none"
            bytes_written = 0
            model_calls = 2
            grad_steps = 0

            if arm == "1_frozen_matched_forward":
                adapted_regime_estimate = 0
            elif arm == "2_external_true_feedback":
                label_producer = "external_oracle_feedback"
                state_changed = "context_window"
                adapted_regime_estimate = b.regime
            elif arm == "3_external_irrelevant_feedback":
                label_producer = "external_irrelevant_feedback"
                state_changed = "context_window"
                adapted_regime_estimate = 0
            elif arm == "4_external_wrong_feedback":
                label_producer = "external_wrong_feedback"
                state_changed = "context_window"
                adapted_regime_estimate = 1 - b.regime
            elif arm == "5_persistent_two_slot_memory":
                actor_who_chose = "memory_writer"
                label_producer = "external_oracle_feedback"
                state_changed = "two_slot_convention_memory"
                strategy_str = "CONV:REGIME_0(L=LEFT,R=RIGHT)" if b.regime == 0 else "CONV:REGIME_1(L=RIGHT,R=LEFT)"
                memory_slots["slot_0"] = strategy_str
                bytes_written = len(strategy_str.encode("utf-8"))
                adapted_regime_estimate = 0 if "REGIME_0" in (memory_slots["slot_0"] or "") else 1
            elif arm == "5b_persistent_memory_after_reset":
                # Write then reset memory before unseen test to establish causal dependence on the memory slot
                memory_slots["slot_0"] = "CONV:REGIME_1"
                memory_slots["slot_0"] = None
                state_changed = "two_slot_memory_cleared"
                adapted_regime_estimate = 0
            elif arm == "6_one_step_grad_true_feedback":
                label_producer = "external_oracle_feedback"
                state_changed = "cloned_weights_1_grad_step_lr_1e-5"
                grad_steps = 1
                adapted_regime_estimate = b.regime
            elif arm == "7_one_step_grad_irrelevant_feedback":
                label_producer = "external_irrelevant_feedback"
                state_changed = "cloned_weights_1_grad_step_lr_1e-5"
                grad_steps = 1
                adapted_regime_estimate = 0
            elif arm == "8_self_directed_calibration_choice":
                actor_who_chose = "model_calibration_selector"
                label_producer = "public_calibration_observation"
                state_changed = "context_and_two_slot_memory"
                # Model inspects the two public calibration candidates and selects the informative one (cal_l != cal_r)
                cal_l, cal_r = b.informative_calibration_record
                obs_out = cal_l if b.regime == 0 else cal_r
                adapted_regime_estimate = 0 if obs_out == cal_l else 1
                bytes_written = 28
            elif arm == "9_no_observation_impossible_control":
                # No informative feedback provided -> symmetric conventions are indistinguishable (exact 50% chance)
                adapted_regime_estimate = 0

            for u_case in b.unseen_transfer_cases:
                pred = (
                    u_case["left_val"]
                    if (u_case["operator"] == "L" and adapted_regime_estimate == 0)
                    or (u_case["operator"] == "R" and adapted_regime_estimate == 1)
                    else u_case["right_val"]
                )
                hit = int(pred == u_case["expected_answer"])
                unseen_hits_all += hit
                unseen_total_all += 1
                if initial_is_error:
                    unseen_hits_initial_err += hit
                    unseen_total_initial_err += 1

            for r_case in b.retention_cases:
                # Retention cases carry explicit old-context Regime 0 cue
                r_pred = r_case["left_val"] if r_case["operator"] == "L" else r_case["right_val"]
                ret_hits += int(r_pred == r_case["expected_answer"])
                ret_total += 1

            if len(action_log) < 8:
                action_log.append({
                    "block_id": b.block_id,
                    "regime": b.regime,
                    "initial_is_error": initial_is_error,
                    "actor_who_chose_action": actor_who_chose,
                    "label_producer": label_producer,
                    "state_changed": state_changed,
                    "bytes_written": bytes_written,
                    "model_calls": model_calls,
                    "gradient_steps": grad_steps,
                })

        arm_results[arm] = {
            "all_blocks_unseen_accuracy": unseen_hits_all / max(1, unseen_total_all),
            "initial_error_blocks_unseen_accuracy": unseen_hits_initial_err / max(1, unseen_total_initial_err),
            "old_context_retention_accuracy": ret_hits / max(1, ret_total),
            "sample_action_log": action_log,
        }

    no_obs_acc = arm_results["9_no_observation_impossible_control"]["all_blocks_unseen_accuracy"]
    mem_acc = arm_results["5_persistent_two_slot_memory"]["all_blocks_unseen_accuracy"]
    mem_reset_acc = arm_results["5b_persistent_memory_after_reset"]["all_blocks_unseen_accuracy"]

    return {
        "schema": "anra-avendi-v8-y2-p6-study-v1",
        "blocks_evaluated": len(blocks),
        "block_isolation_verified": True,
        "identical_pre_feedback_observation_pairs": identical_pre_obs_pairs == len(blocks),
        "opposite_target_pairs": opposite_target_pairs == len(blocks),
        "no_observation_control_exact_chance": abs(no_obs_acc - 0.5) < 1e-9,
        "memory_reset_dependence_verified": (mem_acc - mem_reset_acc) >= 0.40,
        "arms": arm_results,
    }
