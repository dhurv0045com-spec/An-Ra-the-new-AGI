"""Read-only teacher-forced and decoded diagnostics of the active trainer."""
from collections import Counter, defaultdict
import math

import torch

from .representation import UTF8
from .training import batch_tensors


def representative_indices(data, per_kind=16):
    """Select complete world groups, retaining every supervised transition."""
    if type(per_kind) is not int or per_kind < 1:
        raise ValueError("positive per-kind world count required")
    worlds = defaultdict(set)
    for example in data.examples:
        worlds[example.kind].add(example.world_id)
    selected = {world for values in worlds.values() for world in sorted(values)[:per_kind]}
    return tuple(i for i, example in enumerate(data.examples) if example.world_id in selected)


def audit_targets(data):
    """Reject identical visible prompts paired with contradictory action targets."""
    targets = {}
    duplicates = 0
    for example in data.examples:
        previous = targets.get(example.prompt)
        if previous is not None:
            if previous != example.target:
                raise ValueError("identical public prompt has conflicting targets")
            duplicates += 1
        targets[example.prompt] = example.target
    return {"examples": len(data.examples), "unique_prompts": len(targets),
            "duplicate_same_target": duplicates, "conflicting_targets": 0,
            "target_actions": dict(sorted(Counter(e.target for e in data.examples).items()))}


def _summary(rows):
    groups = {}
    for name in ("all", "first_action", "value", "eos"):
        chosen = rows if name == "all" else [r for r in rows if r["role"] == name]
        count = len(chosen)
        groups[name] = {"tokens": count, "correct": sum(r["correct"] for r in chosen),
                        "accuracy": sum(r["correct"] for r in chosen) / count if count else None,
                        "cross_entropy": math.fsum(r["loss"] for r in chosen) / count if count else None,
                        "target_distribution": dict(sorted(Counter(str(r["target"]) for r in chosen).items())),
                        "confusion": dict(sorted(Counter(f'{r["target"]}->{r["prediction"]}' for r in chosen).items()))}
    return groups


def diagnostic(trainer, data, indices=None, *, decode=True):
    """Measure unmodified weights with exact next-token shift and greedy actions.

    Value tokens can also be first-action tokens. Reports retain these two
    overlapping views explicitly; their counts must not be summed.
    """
    audit = audit_targets(data)
    samples = tuple(data.samples)
    selected = tuple(range(len(samples))) if indices is None else tuple(indices)
    if not selected or len(set(selected)) != len(selected) or any(type(i) is not int or not 0 <= i < len(samples) for i in selected):
        raise ValueError("nonempty unique valid measurement indices required")
    modes = [(module, module.training) for module in trainer.model.modules()]
    rows = []
    decoded = []
    exact_teacher = 0
    value_ids = {UTF8.encode(symbol)[0] for symbol in "ABCDEFGHIJKLMNOP"}
    trainer.model.eval()
    try:
        with torch.inference_mode():
            for offset in range(0, len(selected), trainer.recipe.micro_batch_size):
                batch = selected[offset:offset + trainer.recipe.micro_batch_size]
                chosen = [samples[i] for i in batch]
                tokens, segments, _ = batch_tensors([(s.tokens, s.eligible) for s in chosen], trainer.device)
                logits = trainer.backend.logits(tokens, segments).float()
                losses = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                    tokens[:, 1:].reshape(-1), reduction="none").reshape_as(tokens[:, 1:])
                if not torch.isfinite(losses).all():
                    raise FloatingPointError("nonfinite diagnostic loss")
                losses = losses.cpu()
                predictions = logits[:, :-1].argmax(-1).cpu()
                for row_index, (index, sample) in enumerate(zip(batch, chosen)):
                    first = sample.eligible.index(True)
                    if tuple(sample.tokens[first:]) != (*UTF8.encode(data.examples[index].target), UTF8.eos_id):
                        raise ValueError("example target disagrees with framed loss span")
                    exact_teacher += all(int(predictions[row_index, position - 1]) == sample.tokens[position]
                                         for position in range(first, len(sample.tokens)))
                    for position in range(first, len(sample.tokens)):
                        target = sample.tokens[position]
                        predicted = int(predictions[row_index, position - 1])
                        roles = ["eos"] if target == UTF8.eos_id else []
                        if position == first:
                            roles.append("first_action")
                        if target in value_ids:
                            roles.append("value")
                        base = {"category": sample.category, "target": target, "prediction": predicted,
                                "correct": predicted == target, "loss": float(losses[row_index, position - 1])}
                        # A separate unique-token record prevents overlap from biasing all-token CE.
                        rows.append({**base, "role": "token"})
                        rows.extend({**base, "role": role} for role in roles)
                    if decode:
                        completion = trainer.backend.generate((sample.tokens[:first],), max_new_tokens=4)[0]
                        decoded.append({"task_id": sample.task_id, "category": sample.category,
                            "expected": list(sample.tokens[first:]), "generated": list(completion.tokens),
                            "exact": completion.ended_eos and completion.tokens == sample.tokens[first:],
                            "model_calls": completion.forward_calls, "model_positions": completion.evaluated_positions})
    finally:
        for module, mode in modes:
            module.training = mode
    def summarize(subset):
        result = _summary(subset)
        result["all"] = _summary([r for r in subset if r["role"] == "token"])["all"]
        return result
    return {"schema": "elendril-v717-measurement-v1", "audit": audit, "examples": len(selected),
        "teacher_forced": summarize(rows),
        "teacher_forced_exact_actions": {"correct": exact_teacher, "examples": len(selected),
                                         "accuracy": exact_teacher / len(selected)},
        "teacher_minus_decode_exact_accuracy": (exact_teacher / len(selected) - sum(r["exact"] for r in decoded) / len(decoded)) if decoded else None,
        "by_kind": {kind: summarize([r for r in rows if r["category"] == kind]) for kind in sorted({r["category"] for r in rows})},
        "decoded": {"status": "MEASURED" if decode else "NOT_RUN", "examples": len(decoded),
                    "correct": sum(r["exact"] for r in decoded),
                    "accuracy": sum(r["exact"] for r in decoded) / len(decoded) if decoded else None,
                    "records": decoded},
        "qualification": "Teacher-forced token metrics and greedy exact actions on supplied examples; value and first-action views overlap."}


def gradient_probe(trainer):
    """Compare the next batch's microbatch gradients against per-example reference.

    No optimizer step is performed and existing parameter .grad slots survive.
    Comparison is before clipping, using the trainer's actual loss weights.
    """
    if trainer.state.complete or trainer.faulted:
        raise ValueError("active healthy trainer required")
    if trainer.recipe.precision != "fp32":
        raise ValueError("gradient probe currently qualifies FP32 only")
    named = list(trainer.model.named_parameters())
    parameters = [p for _, p in named]
    chosen = [trainer.samples[i] for i in trainer.plan[trainer.state.global_update]]
    if not chosen or any(s.eligible[0] or not all(s.eligible[s.eligible.index(True):])
                         or s.tokens[-1] != UTF8.eos_id for s in chosen):
        raise ValueError("invalid next-target label alignment")
    weights = []
    for sample in chosen:
        row = [float(value) for value in sample.eligible]
        if trainer.recipe.objective == "action-weighted":
            row[sample.eligible.index(True)] = trainer.recipe.action_weight
        weights.append(row)
    denominator = sum(map(sum, weights))
    modes = [(module, module.training) for module in trainer.model.modules()]
    trainer.model.eval()
    def accumulated(chunk_size):
        total = [torch.zeros_like(p) for p in parameters]
        loss_sum = 0.
        for start in range(0, len(chosen), chunk_size):
            chunk = chosen[start:start + chunk_size]
            tokens, segments, _ = batch_tensors([(s.tokens, s.eligible) for s in chunk], trainer.device)
            logits = trainer.backend.logits(tokens, segments).float()
            loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                tokens[:, 1:].reshape(-1), reduction="none").reshape_as(tokens[:, 1:])
            mask = torch.tensor([row + [0.] * (tokens.shape[1] - len(row)) for row in weights[start:start + len(chunk)]],device=trainer.device)
            objective = (loss * mask[:, 1:]).sum() / denominator
            gradients = torch.autograd.grad(objective, parameters)
            if not torch.isfinite(objective) or any(not torch.isfinite(g).all() for g in gradients):
                raise FloatingPointError("nonfinite gradient probe")
            for target, gradient in zip(total, gradients):
                target.add_(gradient)
            loss_sum += float(objective.detach())
        return loss_sum, total
    try:
        reference_loss, reference = accumulated(1)
        actual_loss, actual = accumulated(trainer.recipe.micro_batch_size)
    finally:
        for module, mode in modes:
            module.training = mode
    records = []
    for (name, _), expected, observed in zip(named, reference, actual):
        maximum = float((expected - observed).abs().max())
        scale = float(expected.abs().max())
        records.append({"parameter": name, "max_absolute_difference": maximum,
                        "relative_to_max_gradient": maximum / max(scale, 1e-30),
                        "reference_gradient_norm": float(expected.norm()),
                        "passed": bool(torch.allclose(expected, observed, rtol=1e-4, atol=1e-6))})
    passed = all(r["passed"] for r in records) and math.isclose(reference_loss, actual_loss, rel_tol=1e-5, abs_tol=1e-6)
    return {"status": "PASS" if passed else "FAIL", "reference_loss": reference_loss,
            "microbatch_loss": actual_loss, "weighted_denominator": denominator,
            "label_alignment": "PASS: logits at t predict eligible token t+1, including EOS; prompt excluded",
            "examples": len(chosen), "parameters": records,
            "qualification": "Before-clipping FP32 per-example versus microbatch accumulation; no optimizer update."}
