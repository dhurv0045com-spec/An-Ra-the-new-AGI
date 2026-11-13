"""V8 truth-firewalled interactive rollout, four-mode policy evaluation, and head diagnostics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import math
import random
from typing import Any, Callable

import torch

from anra_v5.elendril.representation import UTF8, SpecialToken
from anra_v5.elendril.training import batch_tensors

from .contracts import POLICY_MODES, SCREEN_MAX_SEQUENCE, SYMBOLS, EvaluationOutcome, PolicyProposal, PublicInput, V8ModelSpec
from .data import render_public_v8
from .models import algebraic_null_from_joint, compute_typed_nll
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


def summarize_suite_episodes(
    episodes_by_mode: dict[str, list[EpisodeRecord]],
    worlds: list[dict[str, object]] | tuple[dict[str, object], ...],
) -> dict[str, object]:
    """Summarize exact episode accuracy, counterfactual pairs, A18 repricing, and work costs."""

    world_by_id = {str(w["world_id"]): w for w in worlds}
    mode_metrics: dict[str, dict[str, object]] = {}

    for mode, episodes in episodes_by_mode.items():
        n = len(episodes)
        by_cohort: dict[str, list[EpisodeRecord]] = defaultdict(list)
        by_family: dict[str, list[EpisodeRecord]] = defaultdict(list)
        by_form: dict[str, list[EpisodeRecord]] = defaultdict(list)
        by_base_form: dict[str, dict[str, EpisodeRecord]] = defaultdict(dict)

        for ep in episodes:
            by_cohort[ep.cohort].append(ep)
            by_family[ep.family].append(ep)
            by_form[ep.form].append(ep)
            by_base_form[ep.base_world_id][ep.form] = ep

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
        mode_metrics[mode] = {
            "episodes": n,
            "exact_episode_accuracy": _rate(episodes),
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
        _ = world_by_id
    return mode_metrics


def diagnostic_probe(
    model: Any,
    v8_spec: V8ModelSpec,
    steps: tuple[TeacherStep, ...] | list[TeacherStep],
    *,
    max_probe_steps: int = 32,
) -> dict[str, object]:
    """Compute teacher-state NLL, operation/value separation, pointer diagnostics, and gradient norm probe."""

    probe_steps = tuple(steps[:max_probe_steps])
    if not probe_steps:
        raise ValueError("diagnostic_probe requires at least one TeacherStep")
    device = next(model.parameters()).device
    modes = [(m, m.training) for m in model.modules()]

    if v8_spec.architecture == "B0":
        # Byte NLL and decoded first-action accuracy without mutating weights
        model.eval()
        total_nll = 0.0
        total_tokens = 0
        answer_nll = 0.0
        first_byte_correct = 0
        with torch.inference_mode():
            for st in probe_steps:
                framed = UTF8.frame(st.public_input.text_utf8, st.target_text, max_length=v8_spec.max_sequence_length)
                toks = torch.tensor([framed.ids], device=device, dtype=torch.int64)
                segs = torch.zeros_like(toks)
                logits = model(toks, segs)
                targets = toks[:, 1:]
                losses = torch.nn.functional.cross_entropy(
                    logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                    reduction="none",
                ).reshape_as(targets)
                elig = torch.tensor(framed.loss_mask[1:], device=device, dtype=torch.float32)
                total_nll += float((losses[0] * elig).sum().item())
                total_tokens += int(elig.sum().item())
                pos = framed.answer_span[0] - 1
                answer_nll += float(losses[0, pos].item())
                pred_id = int(logits[0, pos].argmax().item())
                if pred_id == int(targets[0, pos].item()):
                    first_byte_correct += 1
        # Single-microbatch gradient norm probe
        model.train()
        model.zero_grad(set_to_none=True)
        framed = UTF8.frame(probe_steps[0].public_input.text_utf8, probe_steps[0].target_text, max_length=v8_spec.max_sequence_length)
        toks = torch.tensor([framed.ids], device=device, dtype=torch.int64)
        segs = torch.zeros_like(toks)
        logits = model(toks, segs)
        elig = torch.tensor(framed.loss_mask[1:], device=device, dtype=torch.float32)
        loss = (
            torch.nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), toks[:, 1:].reshape(-1), reduction="none").reshape_as(toks[:, 1:])[0]
            * elig
        ).sum() / elig.sum().clamp_min(1.0)
        loss.backward()
        grad_sq = sum(float((p.grad.detach().float() ** 2).sum().item()) for p in model.parameters() if p.grad is not None)
        model.zero_grad(set_to_none=True)
        for m, was in modes:
            m.training = was
        return {
            "architecture": "B0",
            "probe_steps": len(probe_steps),
            "byte_nll_per_token": total_nll / max(total_tokens, 1),
            "first_action_byte_nll": answer_nll / len(probe_steps),
            "first_action_byte_accuracy": first_byte_correct / len(probe_steps),
            "grad_l2_norm": math.sqrt(grad_sq),
        }

    # Typed heads (J, F, P, F-cap, U)
    samples = teacher_steps_to_typed_samples(probe_steps, max_length=v8_spec.max_sequence_length)
    model.eval()
    total_nll = 0.0
    total_op_nll = 0.0
    total_val_nll = 0.0
    op_correct = 0
    val_correct_all_ans = 0
    val_correct_given_op = 0
    ans_count = 0
    ans_and_op_correct = 0
    ans_mass_sum = 0.0
    gate_vals: list[float] = []
    ent_vals: list[float] = []
    p_ans_correct = 0

    with torch.inference_mode():
        for start in range(0, len(samples), 8):
            chunk = samples[start : start + 8]
            tokens, segments, _ = batch_tensors([(s.tokens, s.eligible) for s in chunk], device)
            action_pos = torch.tensor([s.action_position for s in chunk], device=device, dtype=torch.int64)
            gold_ops = torch.tensor([s.gold_op for s in chunk], device=device, dtype=torch.int64)
            gold_args = torch.tensor([s.gold_arg for s in chunk], device=device, dtype=torch.int64)
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
            op_correct += int(op_ok.sum().item())
            ans_mass_sum += float(out.log_p_op[:, 0].exp().sum().item())

            if bool(is_ans.any().item()):
                ans_count += int(is_ans.sum().item())
                val_ok = (pred_val == gold_args) & is_ans
                val_correct_all_ans += int(val_ok.sum().item())
                both_ok = val_ok & op_ok
                val_correct_given_op += int(both_ok.sum().item())
                ans_and_op_correct += int((op_ok & is_ans).sum().item())

            # Full 19-way argmax restricted on ANSWER steps
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
    chunk = samples[: min(4, len(samples))]
    tokens, segments, _ = batch_tensors([(s.tokens, s.eligible) for s in chunk], device)
    action_pos = torch.tensor([s.action_position for s in chunk], device=device, dtype=torch.int64)
    gold_ops = torch.tensor([s.gold_op for s in chunk], device=device, dtype=torch.int64)
    gold_args = torch.tensor([s.gold_arg for s in chunk], device=device, dtype=torch.int64)
    out = model.forward_action(tokens, segments, action_pos)
    loss, _ = compute_typed_nll(out, gold_ops, gold_args)
    (loss / float(len(chunk))).backward()
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
    for m, was in modes:
        m.training = was

    n = len(samples)
    return {
        "architecture": v8_spec.architecture,
        "probe_steps": n,
        "answer_steps": ans_count,
        "mean_total_nll": total_nll / n,
        "mean_op_nll": total_op_nll / n,
        "mean_val_nll": (total_val_nll / ans_count) if ans_count else None,
        "op_accuracy": op_correct / n,
        "val_accuracy_on_answer_steps": (val_correct_all_ans / ans_count) if ans_count else None,
        "val_accuracy_given_op_correct": (val_correct_given_op / ans_and_op_correct) if ans_and_op_correct else None,
        "joint_19_answer_accuracy": (p_ans_correct / ans_count) if ans_count else None,
        "mean_answer_op_mass": ans_mass_sum / n,
        "mean_gate": (sum(gate_vals) / len(gate_vals)) if gate_vals else None,
        "mean_pointer_entropy": (sum(ent_vals) / len(ent_vals)) if ent_vals else None,
        "backbone_grad_l2": math.sqrt(backbone_sq),
        "head_grad_l2": math.sqrt(head_sq),
    }


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
    "summarize_suite_episodes",
]
