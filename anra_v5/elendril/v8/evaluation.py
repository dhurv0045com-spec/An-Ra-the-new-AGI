"""V8 truth-firewalled interactive rollout, four-mode policy evaluation, and head diagnostics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import math
import random
from typing import Any, Callable

import torch

from anra_v5.elendril.representation import UTF8, SpecialToken
from anra_v5.elendril.training import batch_tensors, logits_for
from anra_v5.elendril.v6_evaluation import parse_action

from .contracts import POLICY_MODES, SCREEN_MAX_SEQUENCE, SYMBOLS, EvaluationOutcome, PolicyProposal, PublicInput, V8ModelSpec
from .data import render_public_v8
from .models import OP_TO_INDEX, algebraic_null_from_joint, compute_typed_nll
from .policies import make_policy
from .teacher import PublicFact, TeacherStep, teacher_decide
from .training import teacher_steps_to_typed_samples
from .verifier import verify_episode_trace


@dataclass(frozen=True, slots=True)
class EpisodeStepRecord:
    decision: int
    public_text_sha256: str
    raw_proposal: PolicyProposal
    executed_proposal: PolicyProposal
    ideal_operation: str
    ideal_argument: str | None
    diverged_from_raw: bool
    diverged_from_teacher: bool


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    world_id: str
    base_world_id: str
    cohort: str
    family: str
    form: str
    mode: str
    model_identity: str
    steps: tuple[EpisodeStepRecord, ...]
    outcome: EvaluationOutcome
    first_observable_divergence: dict[str, object] | None
    query_count: int
    retrieve_count: int
    model_calls: int
    evaluated_positions: int
    teacher_action_matches: int
    is_isolated_one_edge_probe: bool | None = None


def run_episode(
    policy: Any,
    world: dict[str, object],
    *,
    render_fn: Callable[..., str] = render_public_v8,
    max_actions: int = 4,
    extra_initial_facts: tuple[PublicFact, ...] = (),
) -> EpisodeRecord:
    """Execute one episode with strict actor/verifier truth separation."""

    initial_facts = tuple(PublicFact(**row) for row in world["initial_facts"]) + tuple(extra_initial_facts)
    ordered_facts = list(initial_facts)
    seen_facts = set(initial_facts)
    query_avail = bool(world["allowed_public_tools"]["query_available"])
    mem_avail = bool(world["allowed_public_tools"]["memory_available"])
    goal_subject = str(world["goal_subject"])
    hops = int(world["hops"])
    template_id = int(world["renderer_template"])
    alias_prefix = str(world["alias_prefix"])
    order_seed = int(world["row_order_seed"])

    steps: list[EpisodeStepRecord] = []
    executed_actions: list[tuple[str, str | None]] = []
    first_div: dict[str, object] | None = None
    queried = False
    retrieved = False
    query_count = 0
    retrieve_count = 0
    model_calls = 0
    evaluated_positions = 0
    teacher_matches = 0
    terminal_status = "BUDGET_EXHAUSTED"

    from hashlib import sha256

    for decision in range(max_actions):
        text = render_fn(
            ordered_facts=tuple(ordered_facts),
            goal_subject=goal_subject,
            hops=hops,
            query_available=query_avail,
            memory_available=mem_avail,
            template_id=template_id,
            alias_prefix=alias_prefix,
            order_seed=order_seed,
            decision=decision,
        )
        pub = PublicInput(
            text_utf8=text,
            query_available=query_avail,
            memory_available=mem_avail,
            action_budget=max_actions - decision,
        )
        ideal_op, ideal_arg = teacher_decide(tuple(ordered_facts), goal_subject, hops, query_avail, mem_avail)

        # Actor only receives PublicInput; never touches world truth or hidden state.
        exec_prop, raw_prop = policy.propose(pub)
        model_calls += max(exec_prop.model_calls, raw_prop.model_calls)
        evaluated_positions += max(exec_prop.evaluated_positions, raw_prop.evaluated_positions)

        div_raw = (exec_prop.operation, exec_prop.argument) != (raw_prop.operation, raw_prop.argument)
        div_teacher = (exec_prop.operation, exec_prop.argument) != (ideal_op, ideal_arg if ideal_op == "ANSWER" else None)
        if exec_prop.operation == "QUERY" and ideal_op == "QUERY":
            div_teacher = False
        if not div_teacher:
            teacher_matches += 1

        if first_div is None and (div_raw or div_teacher):
            reason = "mode_constraint" if div_raw else (
                "wrong_operation" if exec_prop.operation != ideal_op else "wrong_argument"
            )
            first_div = {
                "decision": decision,
                "reason": reason,
                "raw_op": raw_prop.operation,
                "raw_arg": raw_prop.argument,
                "executed_op": exec_prop.operation,
                "executed_arg": exec_prop.argument,
                "ideal_op": ideal_op,
                "ideal_arg": ideal_arg,
            }

        steps.append(
            EpisodeStepRecord(
                decision=decision,
                public_text_sha256=sha256(text.encode("utf-8")).hexdigest(),
                raw_proposal=raw_prop,
                executed_proposal=exec_prop,
                ideal_operation=ideal_op,
                ideal_argument=ideal_arg,
                diverged_from_raw=div_raw,
                diverged_from_teacher=div_teacher,
            )
        )

        op = exec_prop.operation
        arg = exec_prop.argument
        if op == "INVALID":
            executed_actions.append(("INVALID", None))
            terminal_status = "INVALID_ACTION"
            break
        if op == "QUERY":
            executed_actions.append(("QUERY", arg))
            query_count += 1
            if not query_avail:
                terminal_status = "UNAVAILABLE_QUERY"
                break
            if not queried:
                queried = True
                for row in world["hidden"]["query_payload"]:
                    f = PublicFact(**row)
                    if f not in seen_facts:
                        seen_facts.add(f)
                        ordered_facts.append(f)
            continue
        if op == "RETRIEVE":
            executed_actions.append(("RETRIEVE", None))
            retrieve_count += 1
            if not mem_avail:
                terminal_status = "UNAVAILABLE_RETRIEVE"
                break
            if not retrieved:
                retrieved = True
                for row in world["hidden"]["memory_payload"]:
                    f = PublicFact(**row)
                    if f not in seen_facts:
                        seen_facts.add(f)
                        ordered_facts.append(f)
            continue
        if op in ("ANSWER", "ABSTAIN"):
            executed_actions.append((op, arg))
            terminal_status = f"TERMINAL_{op}"
            break

    outcome = verify_episode_trace(world, executed_actions)
    if terminal_status not in ("INVALID_ACTION", "UNAVAILABLE_QUERY", "UNAVAILABLE_RETRIEVE"):
        terminal_status = outcome.terminal_status
    final_outcome = EvaluationOutcome(
        correct=outcome.correct and terminal_status not in ("INVALID_ACTION", "UNAVAILABLE_QUERY", "UNAVAILABLE_RETRIEVE"),
        terminal_status=terminal_status,
        charged_cost=outcome.charged_cost,
        verifier_identity=outcome.verifier_identity,
    )
    total_edges = (
        len(world["initial_facts"])
        + len(world["hidden"]["query_payload"])
        + len(world["hidden"]["memory_payload"])
    )
    is_iso = (
        bool(world["is_isolated_one_edge_probe"])
        if "is_isolated_one_edge_probe" in world
        else (total_edges == 1 and hops <= 1)
    )
    return EpisodeRecord(
        world_id=str(world["world_id"]),
        base_world_id=str(world.get("parent_world_id") or world.get("base_world_id") or world["world_id"]),
        cohort=str(world.get("scenario") or world.get("cohort") or "R09"),
        family=str(world["family"]),
        form=str(world.get("variant") or world.get("form") or "canonical"),
        mode=str(policy.mode),
        model_identity=str(policy.model_identity),
        steps=tuple(steps),
        outcome=final_outcome,
        first_observable_divergence=first_div,
        query_count=query_count,
        retrieve_count=retrieve_count,
        model_calls=model_calls,
        evaluated_positions=evaluated_positions,
        teacher_action_matches=teacher_matches,
        is_isolated_one_edge_probe=is_iso,
    )


def evaluate_worlds_by_mode(
    model: Any,
    v8_spec: V8ModelSpec,
    worlds: list[dict[str, object]] | tuple[dict[str, object], ...],
    *,
    model_identity: str,
    modes: tuple[str, ...] = POLICY_MODES,
    render_fn: Callable[..., str] = render_public_v8,
    copy_enabled_override: bool | None = None,
) -> dict[str, list[EpisodeRecord]]:
    """Evaluate a collection of worlds under all requested policy modes."""

    results: dict[str, list[EpisodeRecord]] = {}
    for mode in modes:
        pol = make_policy(
            model,
            v8_spec,
            model_identity=model_identity,
            mode=mode,
            copy_enabled_override=copy_enabled_override,
        )
        results[mode] = [run_episode(pol, w, render_fn=render_fn) for w in worlds]
    return results


def _is_isolated_episode(ep: EpisodeRecord, world_by_id: dict[str, dict[str, object]]) -> bool:
    flag = getattr(ep, "is_isolated_one_edge_probe", None)
    if isinstance(flag, bool):
        return flag
    w = world_by_id.get(ep.world_id)
    if w is not None:
        if "is_isolated_one_edge_probe" in w:
            return bool(w["is_isolated_one_edge_probe"])
        if "initial_facts" in w and "hidden" in w:
            hidden = w["hidden"]
            if isinstance(hidden, dict):
                total_edges = (
                    len(w["initial_facts"])  # type: ignore[arg-type]
                    + len(hidden.get("query_payload", ()))
                    + len(hidden.get("memory_payload", ()))
                )
                return total_edges == 1 and int(w.get("hops", 1)) <= 1  # type: ignore[arg-type]
    return str(ep.cohort).lower() in ("isolated_probe", "isolated_one_edge_probe")


def _summarize_episode_list(
    episodes: list[EpisodeRecord],
    world_by_id: dict[str, dict[str, object]],
) -> dict[str, object]:
    n = len(episodes)
    by_cohort: dict[str, list[EpisodeRecord]] = defaultdict(list)
    by_family: dict[str, list[EpisodeRecord]] = defaultdict(list)
    by_form: dict[str, list[EpisodeRecord]] = defaultdict(list)
    by_base_form: dict[str, dict[str, EpisodeRecord]] = defaultdict(dict)
    isolated_eps: list[EpisodeRecord] = []
    non_isolated_eps: list[EpisodeRecord] = []

    for ep in episodes:
        by_cohort[ep.cohort].append(ep)
        by_family[ep.family].append(ep)
        by_form[ep.form].append(ep)
        by_base_form[ep.base_world_id][ep.form] = ep
        if _is_isolated_episode(ep, world_by_id):
            isolated_eps.append(ep)
        else:
            non_isolated_eps.append(ep)

    # Linked world counterfactual consistency checks
    rel_pairs = 0
    rel_both_correct = 0
    irrel_pairs = 0
    irrel_stable = 0
    row_pairs = 0
    row_same_correct = 0
    alias_pairs = 0
    alias_same_correct = 0
    for forms in by_base_form.values():
        canon = forms.get("canonical")
        if canon is None:
            continue
        rel = forms.get("relevant_cf")
        if rel is not None:
            rel_pairs += 1
            if canon.outcome.correct and rel.outcome.correct:
                rel_both_correct += 1
        irr = forms.get("irrelevant_cf")
        if irr is not None:
            irrel_pairs += 1
            c_final = canon.steps[-1].executed_proposal if canon.steps else None
            i_final = irr.steps[-1].executed_proposal if irr.steps else None
            if (
                canon.outcome.correct == irr.outcome.correct
                and c_final is not None
                and i_final is not None
                and (c_final.operation, c_final.argument) == (i_final.operation, i_final.argument)
            ):
                irrel_stable += 1
        rsh = forms.get("row_shuffle")
        if rsh is not None:
            row_pairs += 1
            if canon.outcome.correct and rsh.outcome.correct:
                row_same_correct += 1
        fal = forms.get("fresh_alias")
        if fal is not None:
            alias_pairs += 1
            if canon.outcome.correct and fal.outcome.correct:
                alias_same_correct += 1

    # A18 fixed-policy query cost repricing (0 / 0.1 / 0.5 / 1.0)
    a18_repricing = {}
    for q_cost in (0.0, 0.1, 0.5, 1.0):
        utilities = [
            (1.0 if ep.outcome.correct else 0.0) - q_cost * ep.query_count - 0.02 * ep.retrieve_count
            for ep in episodes
        ]
        a18_repricing[str(q_cost)] = sum(utilities) / n if n else 0.0

    # F01 / F02 / F04 / G05 / G06 / G07 cross-cuts
    def _rate(eps: list[EpisodeRecord]) -> float | None:
        return (sum(1 for e in eps if e.outcome.correct) / len(eps)) if eps else None

    invalid_count = sum(1 for e in episodes if e.outcome.terminal_status in ("INVALID_ACTION", "BUDGET_EXHAUSTED"))
    overall_acc = _rate(episodes)
    return {
        "episodes": n,
        "total_episodes": n,
        "exact_episode_accuracy": overall_acc,
        "exact_accuracy": overall_acc,
        "non_isolated_count": len(non_isolated_eps),
        "non_isolated_exact_accuracy": _rate(non_isolated_eps),
        "isolated_probe_count": len(isolated_eps),
        "isolated_probe_exact_accuracy": _rate(isolated_eps),
        "invalid_or_budget_rate": (invalid_count / n) if n else None,
        "by_cohort": {k: {"n": len(v), "exact_accuracy": _rate(v)} for k, v in sorted(by_cohort.items())},
        "by_family": {k: {"n": len(v), "exact_accuracy": _rate(v)} for k, v in sorted(by_family.items())},
        "by_form": {k: {"n": len(v), "exact_accuracy": _rate(v)} for k, v in sorted(by_form.items())},
        "linked_counterfactuals": {
            "relevant_cf_pairs": rel_pairs,
            "relevant_cf_both_correct": (rel_both_correct / rel_pairs) if rel_pairs else None,
            "irrelevant_cf_pairs": irrel_pairs,
            "irrelevant_cf_stability": (irrel_stable / irrel_pairs) if irrel_pairs else None,
            "row_shuffle_both_correct": (row_same_correct / row_pairs) if row_pairs else None,
            "fresh_alias_both_correct": (alias_same_correct / alias_pairs) if alias_pairs else None,
        },
        "annotations": {
            "F01_canonical_exact": _rate(by_form.get("canonical", [])),
            "F02_relevant_cf_exact": _rate(by_form.get("relevant_cf", [])),
            "G05_fresh_alias_exact": _rate(by_form.get("fresh_alias", [])),
            "G06_row_shuffle_exact": _rate(by_form.get("row_shuffle", [])),
            "G07_conflict_exact": _rate(by_cohort.get("E15", [])),
        },
        "a18_repricing_utility": a18_repricing,
        "work_totals": {
            "model_calls": sum(e.model_calls for e in episodes),
            "evaluated_positions": sum(e.evaluated_positions for e in episodes),
            "query_count": sum(e.query_count for e in episodes),
            "retrieve_count": sum(e.retrieve_count for e in episodes),
        },
    }


def summarize_suite_episodes(
    episodes_by_mode: dict[str, list[EpisodeRecord]] | list[EpisodeRecord] | tuple[EpisodeRecord, ...],
    worlds: list[dict[str, object]] | tuple[dict[str, object], ...] | None = None,
) -> dict[str, object]:
    """Summarize exact episode accuracy, isolated-probe separation, counterfactual pairs, A18 repricing, and work costs."""

    world_by_id = {str(w["world_id"]): w for w in (worlds or ())}
    if isinstance(episodes_by_mode, dict):
        return {
            mode: _summarize_episode_list(list(episodes), world_by_id)
            for mode, episodes in episodes_by_mode.items()
        }
    ep_list = list(episodes_by_mode)
    grouped: dict[str, list[EpisodeRecord]] = defaultdict(list)
    for ep in ep_list:
        grouped[ep.mode].append(ep)
    by_mode = {mode: _summarize_episode_list(eps, world_by_id) for mode, eps in sorted(grouped.items())}
    agg = _summarize_episode_list(ep_list, world_by_id)
    agg["by_mode"] = by_mode
    return agg


def select_stratified_probe_steps(
    steps: tuple[Any, ...] | list[Any],
    *,
    max_steps: int | None = None,
) -> list[Any]:
    """Select up to max_steps evenly across every unique family via deterministic round-robin."""

    step_list = list(steps)
    if max_steps is None or len(step_list) <= max_steps:
        return step_list
    if type(max_steps) is not int or max_steps < 0:
        raise ValueError("max_steps must be a non-negative integer or None")
    if max_steps == 0:
        return []

    groups: dict[str, list[Any]] = defaultdict(list)
    for st in step_list:
        fam = getattr(st, "family", None) or getattr(st, "category", None) or "unknown"
        groups[str(fam)].append(st)

    selected: list[Any] = []
    family_names = sorted(groups.keys())
    offsets = {fam: 0 for fam in family_names}
    while len(selected) < max_steps:
        progressed = False
        for fam in family_names:
            idx = offsets[fam]
            pool = groups[fam]
            if idx < len(pool):
                selected.append(pool[idx])
                offsets[fam] = idx + 1
                progressed = True
                if len(selected) >= max_steps:
                    break
        if not progressed:
            break
    return selected


@dataclass(frozen=True, slots=True)
class _NormalizedProbeStep:
    family: str
    tokens: tuple[int, ...]
    eligible: tuple[bool, ...]
    action_pos: int
    gold_op: str
    gold_arg: str | None
    expected_action_tokens: tuple[int, ...]


def _normalize_probe_step(step: Any, *, max_length: int) -> _NormalizedProbeStep:
    family = str(getattr(step, "family", None) or getattr(step, "category", None) or "unknown")
    if hasattr(step, "public_input") and hasattr(step, "target_text"):
        framed = UTF8.frame(step.public_input.text_utf8, step.target_text, max_length=max_length)
        action_pos = framed.answer_span[0] - 1
        expected_tokens = tuple(
            framed.ids[i] for i in range(action_pos + 1, len(framed.ids)) if framed.loss_mask[i]
        )
        return _NormalizedProbeStep(
            family=family,
            tokens=framed.ids,
            eligible=framed.loss_mask,
            action_pos=action_pos,
            gold_op=str(step.gold_op),
            gold_arg=step.gold_arg,
            expected_action_tokens=expected_tokens,
        )
    if hasattr(step, "tokens") and hasattr(step, "eligible") and hasattr(step, "action_position") and hasattr(step, "gold_op"):
        tokens = tuple(step.tokens)
        eligible = tuple(step.eligible)
        action_pos = int(step.action_position)
        op_names = ("ANSWER", "QUERY", "RETRIEVE", "ABSTAIN")
        gold_op = op_names[int(step.gold_op)] if isinstance(step.gold_op, int) else str(step.gold_op)
        gold_arg: str | None = None
        if gold_op == "ANSWER":
            gold_arg = SYMBOLS[int(step.gold_arg)] if isinstance(step.gold_arg, int) else str(step.gold_arg)
        expected_tokens = tuple(t for t, e in zip(tokens, eligible) if e)
        return _NormalizedProbeStep(
            family=family,
            tokens=tokens,
            eligible=eligible,
            action_pos=action_pos,
            gold_op=gold_op,
            gold_arg=gold_arg,
            expected_action_tokens=expected_tokens,
        )
    if hasattr(step, "tokens") and hasattr(step, "eligible"):
        tokens = tuple(step.tokens)
        eligible = tuple(step.eligible)
        try:
            first_elig = eligible.index(True)
        except ValueError as exc:
            raise ValueError("raw sample missing eligible target tokens") from exc
        action_pos = first_elig - 1
        if action_pos < 0:
            raise ValueError("raw sample missing action marker position before target")
        expected_tokens = tuple(t for t, e in zip(tokens, eligible) if e)
        gold_op, gold_arg = parse_action(expected_tokens, ("request-current-status",))
        if gold_op == "INVALID":
            raise ValueError(f"cannot parse expected action tokens: {expected_tokens!r}")
        return _NormalizedProbeStep(
            family=family,
            tokens=tokens,
            eligible=eligible,
            action_pos=action_pos,
            gold_op=gold_op,
            gold_arg=gold_arg,
            expected_action_tokens=expected_tokens,
        )
    raise TypeError(f"unsupported step type for diagnostic_probe: {type(step).__name__}")


def diagnostic_probe(
    model: Any,
    v8_spec_or_steps: Any = None,
    steps: Any = None,
    *,
    max_probe_steps: int | None = None,
    batch_size: int = 32,
    device: str | torch.device | None = None,
    probe_scope: str = "full",
    v8_spec: V8ModelSpec | None = None,
) -> dict[str, object]:
    """Compute teacher-state loss, exact joint action accuracy, op/value separation, pointer diagnostics, and grad norm."""

    if isinstance(v8_spec_or_steps, V8ModelSpec):
        resolved_spec: V8ModelSpec | None = v8_spec_or_steps
        raw_steps = steps
    else:
        resolved_spec = v8_spec if v8_spec is not None else getattr(model, "v8_spec", None)
        raw_steps = steps if steps is not None else v8_spec_or_steps

    if raw_steps is None:
        raise ValueError("diagnostic_probe requires steps")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    all_steps = list(raw_steps)
    if not all_steps:
        raise ValueError("diagnostic_probe requires at least one step")

    probe_steps = (
        select_stratified_probe_steps(all_steps, max_steps=max_probe_steps)
        if max_probe_steps is not None
        else all_steps
    )
    if not probe_steps:
        raise ValueError("diagnostic_probe requires at least one step")

    if resolved_spec is not None:
        arch = resolved_spec.architecture
        max_seq_len = resolved_spec.max_sequence_length
    elif hasattr(model, "forward_action") and hasattr(model, "v8_spec"):
        resolved_spec = model.v8_spec
        arch = resolved_spec.architecture
        max_seq_len = resolved_spec.max_sequence_length
    else:
        arch = "B0"
        core_spec = getattr(model, "spec", None)
        max_seq_len = int(getattr(core_spec, "context_length", SCREEN_MAX_SEQUENCE))

    dev = torch.device(device) if device is not None else next(model.parameters()).device
    norm_items = [_normalize_probe_step(st, max_length=max_seq_len) for st in probe_steps]
    families_covered = sorted({item.family for item in norm_items})
    modes = [(m, m.training) for m in model.modules()]

    try:
        if arch == "B0":
            model.eval()
            total_nll = 0.0
            total_tokens = 0
            answer_nll = 0.0
            first_byte_correct = 0
            exact_action_count = 0
            op_correct = 0
            ans_count = 0
            non_ans_count = 0
            val_correct_all_ans = 0
            val_correct_given_op = 0
            ans_and_op_correct = 0
            non_ans_op_correct = 0
            stop_tokens = {UTF8.eos_id, UTF8.pad_id, UTF8.bos_id, UTF8.special(SpecialToken.ANSWER)}

            with torch.inference_mode():
                for start in range(0, len(norm_items), batch_size):
                    chunk = norm_items[start : start + batch_size]
                    toks, segs, elig_mask = batch_tensors([(it.tokens, it.eligible) for it in chunk], dev)
                    logits = logits_for(model, toks, segs)
                    targets = toks[:, 1:]
                    losses = torch.nn.functional.cross_entropy(
                        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                        targets.reshape(-1),
                        reduction="none",
                    ).reshape_as(targets)
                    elig_f = elig_mask[:, 1:].to(dtype=torch.float32)
                    total_nll += float((losses * elig_f).sum().item())
                    total_tokens += int(elig_f.sum().item())

                    for row_idx, item in enumerate(chunk):
                        pos = item.action_pos
                        answer_nll += float(losses[row_idx, pos].item())
                        tok0 = int(logits[row_idx, pos].argmax().item())
                        if tok0 == int(targets[row_idx, pos].item()):
                            first_byte_correct += 1

                        pred_tokens = [tok0]
                        cur_ids = list(item.tokens[: pos + 1])
                        if tok0 not in stop_tokens:
                            cur_ids.append(tok0)
                            for _ in range(3):
                                if len(cur_ids) >= max_seq_len:
                                    break
                                c_toks = torch.tensor([cur_ids], device=dev, dtype=torch.int64)
                                c_segs = torch.zeros_like(c_toks)
                                c_logits = logits_for(model, c_toks, c_segs)
                                nxt = int(c_logits[0, -1].argmax().item())
                                pred_tokens.append(nxt)
                                if nxt in stop_tokens:
                                    break
                                cur_ids.append(nxt)

                        pred_op, pred_arg = parse_action(pred_tokens, ("request-current-status",))
                        op_match = pred_op == item.gold_op
                        if op_match:
                            op_correct += 1

                        is_ans = item.gold_op == "ANSWER"
                        if is_ans:
                            ans_count += 1
                            val_match = pred_op == "ANSWER" and pred_arg == item.gold_arg
                            if val_match:
                                val_correct_all_ans += 1
                            if op_match:
                                ans_and_op_correct += 1
                                if val_match:
                                    val_correct_given_op += 1
                        else:
                            non_ans_count += 1
                            if op_match:
                                non_ans_op_correct += 1

                        if tuple(pred_tokens) == item.expected_action_tokens:
                            exact_action_count += 1

            # Single-microbatch gradient norm probe
            model.train()
            model.zero_grad(set_to_none=True)
            toks, segs, elig_mask = batch_tensors([(norm_items[0].tokens, norm_items[0].eligible)], dev)
            logits = logits_for(model, toks, segs)
            elig_f = elig_mask[:, 1:].to(dtype=torch.float32)
            loss = (
                torch.nn.functional.cross_entropy(
                    logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                    toks[:, 1:].reshape(-1),
                    reduction="none",
                ).reshape_as(toks[:, 1:])
                * elig_f
            ).sum() / elig_f.sum().clamp_min(1.0)
            loss.backward()
            grad_sq = sum(
                float((p.grad.detach().float() ** 2).sum().item())
                for p in model.parameters()
                if p.grad is not None
            )
            model.zero_grad(set_to_none=True)

            n = len(norm_items)
            teacher_loss = total_nll / max(total_tokens, 1)
            return {
                "architecture": "B0",
                "probe_steps": n,
                "total_steps": n,
                "answer_steps": ans_count,
                "answer_step_count": ans_count,
                "non_answer_step_count": non_ans_count,
                "exact_action_count": exact_action_count,
                "exact_action_accuracy": exact_action_count / n,
                "op_accuracy": op_correct / n,
                "val_accuracy_on_answer_steps": (val_correct_all_ans / ans_count) if ans_count > 0 else None,
                "val_accuracy_given_op_correct": (val_correct_given_op / ans_and_op_correct) if ans_and_op_correct > 0 else None,
                "non_answer_op_accuracy": (non_ans_op_correct / non_ans_count) if non_ans_count > 0 else None,
                "teacher_loss": teacher_loss,
                "mean_total_nll": total_nll / n,
                "byte_nll_per_token": teacher_loss,
                "first_action_byte_nll": answer_nll / n,
                "first_action_byte_accuracy": first_byte_correct / n,
                "grad_l2_norm": math.sqrt(grad_sq),
                "families_covered": families_covered,
                "probe_scope": probe_scope,
            }

        # Typed heads (J, F, P, F-cap, U)
        typed_items: list[tuple[_NormalizedProbeStep, int, int]] = []
        for item in norm_items:
            if item.gold_op not in OP_TO_INDEX:
                raise ValueError(f"typed model {arch!r} cannot evaluate operation {item.gold_op!r}")
            g_op = OP_TO_INDEX[item.gold_op]
            g_arg = SYMBOLS.index(item.gold_arg) if (g_op == 0 and item.gold_arg in SYMBOLS) else 0
            if g_op == 0 and item.gold_arg not in SYMBOLS:
                raise ValueError(f"ANSWER step missing symbol arg: {item.gold_arg!r}")
            typed_items.append((item, g_op, g_arg))

        model.eval()
        total_nll = 0.0
        total_op_nll = 0.0
        total_val_nll = 0.0
        exact_action_count = 0
        op_correct = 0
        val_correct_all_ans = 0
        val_correct_given_op = 0
        ans_count = 0
        non_ans_count = 0
        ans_and_op_correct = 0
        non_ans_op_correct = 0
        ans_mass_sum = 0.0
        gate_vals: list[float] = []
        ent_vals: list[float] = []
        p_ans_correct = 0

        with torch.inference_mode():
            for start in range(0, len(typed_items), batch_size):
                chunk = typed_items[start : start + batch_size]
                tokens, segments, _ = batch_tensors([(it.tokens, it.eligible) for it, _, _ in chunk], dev)
                action_pos = torch.tensor([it.action_pos for it, _, _ in chunk], device=dev, dtype=torch.int64)
                gold_ops = torch.tensor([g_op for _, g_op, _ in chunk], device=dev, dtype=torch.int64)
                gold_args = torch.tensor([g_arg for _, _, g_arg in chunk], device=dev, dtype=torch.int64)
                out = model.forward_action(tokens, segments, action_pos)
                nll_sum, _ = compute_typed_nll(out, gold_ops, gold_args)
                total_nll += float(nll_sum.item())

                op_nll = -out.log_p_op.gather(1, gold_ops.unsqueeze(1)).squeeze(1)
                total_op_nll += float(op_nll.sum().item())

                is_ans = gold_ops == 0
                val_nll = -out.log_p_val.gather(1, gold_args.clamp(0, 15).unsqueeze(1)).squeeze(1)
                if bool(is_ans.any().item()):
                    total_val_nll += float(val_nll[is_ans].sum().item())

                pred_op = out.log_p_op.argmax(dim=-1)
                pred_val = out.log_p_val.argmax(dim=-1)
                op_ok = pred_op == gold_ops
                val_ok_all = pred_val == gold_args
                exact_ok = torch.where(is_ans, op_ok & val_ok_all, op_ok)
                exact_action_count += int(exact_ok.sum().item())
                op_correct += int(op_ok.sum().item())
                ans_mass_sum += float(out.log_p_op[:, 0].exp().sum().item())

                if bool(is_ans.any().item()):
                    ans_count += int(is_ans.sum().item())
                    val_ok = val_ok_all & is_ans
                    val_correct_all_ans += int(val_ok.sum().item())
                    val_correct_given_op += int((val_ok & op_ok).sum().item())
                    ans_and_op_correct += int((op_ok & is_ans).sum().item())

                non_ans_mask = ~is_ans
                if bool(non_ans_mask.any().item()):
                    non_ans_count += int(non_ans_mask.sum().item())
                    non_ans_op_correct += int((op_ok & non_ans_mask).sum().item())

                pred_19 = out.log_probs_19.argmax(dim=-1)
                if bool(is_ans.any().item()):
                    p_ans_correct += int(((pred_19 < 16) & (pred_19 == gold_args) & is_ans).sum().item())

                if out.gate is not None:
                    gate_vals.extend(out.gate.cpu().tolist())
                if out.attention_entropy is not None and out.has_pointer_candidates is not None:
                    mask = out.has_pointer_candidates
                    if bool(mask.any().item()):
                        ent_vals.extend(out.attention_entropy[mask].cpu().tolist())

        # Gradient probe per parameter group on first 4 states without changing weights
        model.train()
        model.zero_grad(set_to_none=True)
        grad_chunk = typed_items[: min(4, len(typed_items))]
        tokens, segments, _ = batch_tensors([(it.tokens, it.eligible) for it, _, _ in grad_chunk], dev)
        action_pos = torch.tensor([it.action_pos for it, _, _ in grad_chunk], device=dev, dtype=torch.int64)
        gold_ops = torch.tensor([g_op for _, g_op, _ in grad_chunk], device=dev, dtype=torch.int64)
        gold_args = torch.tensor([g_arg for _, _, g_arg in grad_chunk], device=dev, dtype=torch.int64)
        out = model.forward_action(tokens, segments, action_pos)
        loss, _ = compute_typed_nll(out, gold_ops, gold_args)
        (loss / float(len(grad_chunk))).backward()
        backbone_sq = 0.0
        head_sq = 0.0
        for name, p in model.named_parameters():
            if p.grad is not None:
                sq = float((p.grad.detach().float() ** 2).sum().item())
                if name.startswith("core."):
                    backbone_sq += sq
                else:
                    head_sq += sq
        model.zero_grad(set_to_none=True)

        n = len(typed_items)
        mean_total_nll = total_nll / n
        return {
            "architecture": arch,
            "probe_steps": n,
            "total_steps": n,
            "answer_steps": ans_count,
            "answer_step_count": ans_count,
            "non_answer_step_count": non_ans_count,
            "exact_action_count": exact_action_count,
            "exact_action_accuracy": exact_action_count / n,
            "teacher_loss": mean_total_nll,
            "mean_total_nll": mean_total_nll,
            "mean_op_nll": total_op_nll / n,
            "mean_val_nll": (total_val_nll / ans_count) if ans_count > 0 else None,
            "op_accuracy": op_correct / n,
            "val_accuracy_on_answer_steps": (val_correct_all_ans / ans_count) if ans_count > 0 else None,
            "val_accuracy_given_op_correct": (val_correct_given_op / ans_and_op_correct) if ans_and_op_correct > 0 else None,
            "non_answer_op_accuracy": (non_ans_op_correct / non_ans_count) if non_ans_count > 0 else None,
            "joint_19_answer_accuracy": (p_ans_correct / ans_count) if ans_count > 0 else None,
            "mean_answer_op_mass": ans_mass_sum / n,
            "mean_gate": (sum(gate_vals) / len(gate_vals)) if gate_vals else None,
            "mean_pointer_entropy": (sum(ent_vals) / len(ent_vals)) if ent_vals else None,
            "backbone_grad_l2": math.sqrt(backbone_sq),
            "head_grad_l2": math.sqrt(head_sq),
            "families_covered": families_covered,
            "probe_scope": probe_scope,
        }
    finally:
        for m, was in modes:
            m.training = was


def run_pointer_interventions(
    model: Any,
    v8_spec: V8ModelSpec,
    steps: tuple[TeacherStep, ...] | list[TeacherStep],
    *,
    seed: int = 1234,
) -> dict[str, object]:
    """Causal pointer-reader interventions at fixed weights: baseline, no-copy, cyclic shift, sham perm."""

    if v8_spec.architecture != "P":
        raise ValueError("run_pointer_interventions requires architecture P")
    samples = [
        s
        for s in teacher_steps_to_typed_samples(tuple(steps), max_length=v8_spec.max_sequence_length)
        if s.gold_op == 0
    ]
    if not samples:
        raise ValueError("pointer intervention requires ANSWER steps")

    device = next(model.parameters()).device
    modes = [(m, m.training) for m in model.modules()]
    model.eval()

    rng = random.Random(seed)
    sham_order = list(range(16))
    rng.shuffle(sham_order)
    sham_perm = torch.tensor(sham_order, device=device, dtype=torch.int64)
    cyclic_perm = torch.tensor([(i + 1) % 16 for i in range(16)], device=device, dtype=torch.int64)

    def _eval(copy_override: bool | None, val_perm: torch.Tensor | None) -> dict[str, float]:
        nll = 0.0
        correct = 0
        with torch.inference_mode():
            for start in range(0, len(samples), 8):
                chunk = samples[start : start + 8]
                tokens, segments, _ = batch_tensors([(s.tokens, s.eligible) for s in chunk], device)
                action_pos = torch.tensor([s.action_position for s in chunk], device=device, dtype=torch.int64)
                gold_ops = torch.tensor([s.gold_op for s in chunk], device=device, dtype=torch.int64)
                gold_args = torch.tensor([s.gold_arg for s in chunk], device=device, dtype=torch.int64)
                out = model.forward_action(
                    tokens,
                    segments,
                    action_pos,
                    copy_enabled_override=copy_override,
                    value_permutation=val_perm,
                )
                step_nll, _ = compute_typed_nll(out, gold_ops, gold_args)
                nll += float(step_nll.item())
                correct += int((out.log_p_val.argmax(dim=-1) == gold_args).sum().item())
        return {"answer_val_nll": nll / len(samples), "answer_val_accuracy": correct / len(samples)}

    try:
        base = _eval(None, None)
        no_copy = _eval(False, None)
        cyclic = _eval(None, cyclic_perm)
        sham = _eval(None, sham_perm)
    finally:
        for m, was in modes:
            m.training = was

    return {
        "answer_steps": len(samples),
        "baseline_pointer": base,
        "no_copy_control": no_copy,
        "cyclic_value_permutation": cyclic,
        "sham_value_permutation": sham,
        "copy_gain_accuracy": base["answer_val_accuracy"] - no_copy["answer_val_accuracy"],
        "cyclic_drop_accuracy": base["answer_val_accuracy"] - cyclic["answer_val_accuracy"],
        "sham_drop_accuracy": base["answer_val_accuracy"] - sham["answer_val_accuracy"],
    }


def evaluate_f03_reverse_extension(
    model: Any,
    v8_spec: V8ModelSpec,
    *,
    model_identity: str,
) -> dict[str, object]:
    """Diagnostic F03 reverse-relation extension: unique inverse vs ambiguous inverse (untrained held-out)."""

    pol = make_policy(model, v8_spec, model_identity=model_identity, mode="raw")
    unique_obs = PublicInput(
        text_utf8=(
            "Latest time per source counts; disagreement needs evidence. Output A-P, Q=query, !=abstain, R=memory.\n"
            "Reverse lookup: which source maps to B? Query=0 Memory=0\n"
            "A=B@1/src1; C=D@1/src1\nAction:"
        ),
        query_available=False,
        memory_available=False,
        action_budget=2,
    )
    ambig_obs = PublicInput(
        text_utf8=(
            "Latest time per source counts; disagreement needs evidence. Output A-P, Q=query, !=abstain, R=memory.\n"
            "Reverse lookup: which source maps to B? Query=0 Memory=0\n"
            "A=B@1/src1; C=B@1/src1\nAction:"
        ),
        query_available=False,
        memory_available=False,
        action_budget=2,
    )
    u_exec, _ = pol.propose(unique_obs)
    a_exec, _ = pol.propose(ambig_obs)
    return {
        "scenario": "F03",
        "unique_inverse_correct": (u_exec.operation == "ANSWER" and u_exec.argument == "A"),
        "ambiguous_inverse_abstained": (a_exec.operation == "ABSTAIN"),
        "unique_proposal": (u_exec.operation, u_exec.argument),
        "ambiguous_proposal": (a_exec.operation, a_exec.argument),
    }


def evaluate_a20_recovery_extension(
    model: Any,
    v8_spec: V8ModelSpec,
    worlds: list[dict[str, object]] | tuple[dict[str, object], ...],
    *,
    model_identity: str,
) -> dict[str, object]:
    """Diagnostic A20 extension: inject wrong earlier claim, then evaluate recovery given corrective evidence."""

    pol = make_policy(model, v8_spec, model_identity=model_identity, mode="raw")
    tested = 0
    recovered = 0
    for w in worlds:
        expected = w["hidden"]["expected"]
        if expected is None or expected not in SYMBOLS:
            continue
        true_val = str(expected)
        wrong_val = next(s for s in SYMBOLS if s != true_val)
        injected = (
            PublicFact(subject=str(w["goal_subject"]), source="src0", tick=0, value=wrong_val),
        )
        ep = run_episode(pol, w, extra_initial_facts=injected)
        tested += 1
        if ep.outcome.correct:
            recovered += 1
    return {
        "scenario": "A20",
        "tested_worlds": tested,
        "recovered_worlds": recovered,
        "recovery_accuracy": (recovered / tested) if tested else None,
    }


__all__ = [
    "EpisodeRecord",
    "EpisodeStepRecord",
    "diagnostic_probe",
    "evaluate_a20_recovery_extension",
    "evaluate_f03_reverse_extension",
    "evaluate_worlds_by_mode",
    "run_episode",
    "run_pointer_interventions",
    "select_stratified_probe_steps",
    "summarize_suite_episodes",
]
