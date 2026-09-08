from __future__ import annotations

import hashlib
import itertools
import random
import sys
from collections import Counter
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "COLAB"))

from discovery_v6_common import (
    ReceiptWriter,
    RunContext,
    bind_ark11_runtime,
    detect_sustained,
    flat_params,
    generate_indices,
    load_ark11,
    order_sha256,
    sha_json,
)

PLAN_SHA = "59e01a33a11aea1222368553d0819764341e2577"
RUNNER_PATH = Path(__file__)
ACQ_SEED = 2201
CONT_SEEDS = [7701, 7702, 7703]
PERMS = list(itertools.permutations(range(3)))


def fact_signature(facts) -> tuple:
    return tuple(sorted((int(k), int(v)) for k, v in facts))


def render(facts, query: int) -> tuple[str, str]:
    mapping = dict(facts)
    prompt = "+".join(f"{k}={v}" for k, v in facts) + f"/{query}="
    return prompt, str(mapping[query])


def expand_semantic(factsets) -> list[dict]:
    rows = []
    sid = 0
    for facts in factsets:
        facts = tuple((int(k), int(v)) for k, v in facts)
        keys = [k for k, _ in facts]
        mapping = dict(facts)
        for q in keys:
            rows.append({
                "semantic_id": sid,
                "facts": facts,
                "query": int(q),
                "answer": int(mapping[q]),
            })
            sid += 1
    return rows


def eval_variants(factsets) -> dict[str, list[tuple[str, str]]]:
    canonical = []
    order_only = []
    query_only = []
    query_order = []
    for facts0 in factsets:
        facts = tuple((int(k), int(v)) for k, v in facts0)
        keys = [k for k, _ in facts]
        reversed_facts = tuple(reversed(facts))
        for q in keys:
            pos = keys.index(q)
            q2 = keys[(pos + 1) % len(keys)]
            canonical.append(render(facts, q))
            order_only.append(render(reversed_facts, q))
            query_only.append(render(facts, q2))
            query_order.append(render(reversed_facts, q2))
    return {
        "canonical": canonical,
        "order_only": order_only,
        "query_only": query_only,
        "query_order": query_order,
    }


def build_binding_manifest() -> tuple[list[dict], dict[str, list[tuple[str, str]]],
                                      dict[str, list[tuple[str, str]]], dict]:
    keys = range(6)
    vals = range(6)
    factsets = []
    for key_tuple in itertools.combinations(keys, 3):
        for val_tuple in itertools.permutations(vals, 3):
            factsets.append(tuple(zip(key_tuple, val_tuple)))
    rng = random.Random(4242)
    rng.shuffle(factsets)
    train_factsets = factsets[:400]
    test_factsets = factsets[400:500]

    train_sigs = {fact_signature(f) for f in train_factsets}
    test_sigs = {fact_signature(f) for f in test_factsets}
    if train_sigs & test_sigs:
        raise RuntimeError("binding train/test fact-set leakage")

    ordered_test = sorted(test_factsets, key=lambda f: hashlib.sha256(repr(fact_signature(f)).encode()).hexdigest())
    control_factsets = ordered_test[:50]
    sealed_factsets = ordered_test[50:]
    control_sigs = {fact_signature(f) for f in control_factsets}
    sealed_sigs = {fact_signature(f) for f in sealed_factsets}
    if control_sigs & sealed_sigs:
        raise RuntimeError("binding CONTROL/SEALED fact-set leakage")

    train_meta = expand_semantic(train_factsets)
    control = eval_variants(control_factsets)
    sealed = eval_variants(sealed_factsets)

    manifest = {
        "schema": "arkenstone-ark014-binding/v1",
        "task_seed": 4242,
        "train_factsets": len(train_factsets),
        "control_factsets": len(control_factsets),
        "sealed_factsets": len(sealed_factsets),
        "train_semantic_examples": len(train_meta),
        "control_examples_per_variant": len(control["canonical"]),
        "sealed_examples_per_variant": len(sealed["canonical"]),
        "factset_overlap_train_test": 0,
        "factset_overlap_control_sealed": 0,
        "train_factset_sha256": sha_json(sorted(list(train_sigs))),
        "control_factset_sha256": sha_json(sorted(list(control_sigs))),
        "sealed_factset_sha256": sha_json(sorted(list(sealed_sigs))),
        "diagnostics": ["canonical", "order_only", "query_only", "query_order"],
        "train_query_counts": dict(Counter(int(x["query"]) for x in train_meta)),
    }
    manifest["manifest_sha256"] = sha_json(manifest)
    return train_meta, control, sealed, manifest


def augmented_facts(item: dict, seed: int, step: int, batch_position: int):
    key = f"{seed}:{step}:{batch_position}:{item['semantic_id']}".encode()
    idx = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % len(PERMS)
    perm = PERMS[idx]
    facts = item["facts"]
    return tuple(facts[i] for i in perm)


def make_batch(train_meta: list[dict], ids: list[int], *, regime: str, seed: int, step: int):
    rows = []
    for pos, semantic_idx in enumerate(ids):
        item = train_meta[int(semantic_idx)]
        facts = item["facts"] if regime == "CANONICAL_TRAIN" else augmented_facts(item, seed, step, pos)
        rows.append(render(facts, int(item["query"])))
    return rows


def evaluate(ark11, model, vocab, variants: dict[str, list[tuple[str, str]]]) -> dict:
    out = {}
    for key in ["canonical", "order_only", "query_only", "query_order"]:
        exact, _ = ark11.greedy_exact(model, vocab, variants[key], ark11.dev())
        out[key] = exact
    out["qualified"] = out["canonical"] >= 0.90 and out["order_only"] >= 0.85 and out["query_order"] >= 0.85
    return out


def train_regime(ark11, *, regime: str, train_meta, control, sealed, max_steps: int = 24000) -> dict:
    vocab = ark11.CompactVocab()
    torch.manual_seed(ACQ_SEED)
    torch.cuda.manual_seed_all(ACQ_SEED)
    model = ark11.Micro(vocab.size, 128).to(ark11.dev())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1
    )
    indices = generate_indices(ACQ_SEED, max_steps, 64, len(train_meta))
    trajectory = []
    qualified_evals: list[tuple[int, float]] = []
    supervised_tokens = 0

    for step in range(1, max_steps + 1):
        rows = make_batch(train_meta, indices[step - 1], regime=regime, seed=ACQ_SEED, step=step)
        loss, count = ark11.loss_and_positions(model, vocab, rows, ark11.dev())
        if not torch.isfinite(loss):
            raise RuntimeError(f"{regime}: nonfinite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        supervised_tokens += int(count.item())

        if step == 1 or step % 200 == 0:
            metrics = evaluate(ark11, model, vocab, control)
            trajectory.append({"step": step, **metrics, "loss": float(loss.detach().item())})
            qualified_evals.append((step, 1.0 if metrics["qualified"] else 0.0))
            onset, confirm = detect_sustained(qualified_evals, 1.0, 3)
            print(
                f"[ARK-014 {regime}] step={step} can={metrics['canonical']:.3f} "
                f"ord={metrics['order_only']:.3f} qord={metrics['query_order']:.3f}",
                flush=True,
            )
            if confirm is not None:
                sealed_once = evaluate(ark11, model, vocab, sealed)
                return {
                    "regime": regime,
                    "status": "QUALIFIED",
                    "qualification_onset_step": onset,
                    "qualification_confirmation_step": confirm,
                    "supervised_tokens": supervised_tokens,
                    "trajectory": trajectory,
                    "sealed_at_fork": sealed_once,
                    "snapshot": ark11.snapshot_state(model, optimizer),
                    "reference_flat": flat_params(model).detach().cpu(),
                }

    sealed_final = evaluate(ark11, model, vocab, sealed)
    return {
        "regime": regime,
        "status": "NO_QUALIFICATION",
        "supervised_tokens": supervised_tokens,
        "trajectory": trajectory,
        "sealed_final": sealed_final,
    }


def retention_metrics(trajectory: list[dict], prefix: str) -> dict:
    if not trajectory:
        return {"status": "EMPTY"}
    qualified = [bool(x[f"{prefix}_qualified"]) for x in trajectory]
    failure_evals = [(int(x["step"]), 1.0 if not x[f"{prefix}_qualified"] else 0.0) for x in trajectory]
    onset, confirm = detect_sustained(failure_evals, 1.0, 3)
    return {
        "RET_QUALIFIED": sum(qualified) / len(qualified),
        "FINAL_CANONICAL": trajectory[-1][f"{prefix}_canonical"],
        "FINAL_ORDER_ONLY": trajectory[-1][f"{prefix}_order_only"],
        "FINAL_QUERY_ORDER": trajectory[-1][f"{prefix}_query_order"],
        "CANONICAL_AREA": sum(x[f"{prefix}_canonical"] for x in trajectory) / len(trajectory),
        "ORDER_ONLY_AREA": sum(x[f"{prefix}_order_only"] for x in trajectory) / len(trajectory),
        "QUERY_ORDER_AREA": sum(x[f"{prefix}_query_order"] for x in trajectory) / len(trajectory),
        "FAILURE_ONSET": onset,
        "FAILURE_CONFIRM": confirm,
        "collapsed": confirm is not None,
    }


def run_continuation(ark11, *, acq: dict, train_meta, control, sealed,
                     regime: str, order_seed: int, lr: float, steps: int = 6000) -> dict:
    indices = generate_indices(order_seed, steps, 64, len(train_meta))
    vocab, model, optimizer = ark11.load_fork(acq["snapshot"], lr)
    reference = acq["reference_flat"].to(ark11.dev())
    trajectory = []
    supervised_tokens = 0
    base_step = int(acq["qualification_confirmation_step"])

    for step in range(1, steps + 1):
        rows = make_batch(
            train_meta, indices[step - 1], regime=regime, seed=ACQ_SEED, step=base_step + step
        )
        loss, count = ark11.loss_and_positions(model, vocab, rows, ark11.dev())
        if not torch.isfinite(loss):
            raise RuntimeError(f"{regime} retention: nonfinite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        supervised_tokens += int(count.item())

        if step % 200 == 0:
            c = evaluate(ark11, model, vocab, control)
            s = evaluate(ark11, model, vocab, sealed)
            current = flat_params(model)
            disp = float((current - reference).norm().item())
            row = {"step": step, "relative_displacement": disp / max(float(reference.norm().item()), 1e-12)}
            for k, v in c.items():
                row[f"control_{k}"] = v
            for k, v in s.items():
                row[f"sealed_{k}"] = v
            trajectory.append(row)

    return {
        "lr": lr,
        "order_seed": order_seed,
        "order_sha256": order_sha256(indices),
        "supervised_tokens": supervised_tokens,
        "control_retention": retention_metrics(trajectory, "control"),
        "sealed_retention": retention_metrics(trajectory, "sealed"),
        "trajectory": trajectory,
    }


def summarize(acquisitions: list[dict], forks: list[dict]) -> dict:
    by_regime = {a["regime"]: a for a in acquisitions if "regime" in a}
    canonical_q = by_regime.get("CANONICAL_TRAIN", {}).get("status") == "QUALIFIED"
    augmented_q = by_regime.get("ORDER_AUGMENTED", {}).get("status") == "QUALIFIED"

    paired = []
    for regime in ["ORDER_AUGMENTED", "CANONICAL_TRAIN"]:
        for order_seed in CONT_SEEDS:
            high = next((x for x in forks if x.get("regime") == regime and x.get("order_seed") == order_seed and x.get("arm") == "HIGH"), None)
            low = next((x for x in forks if x.get("regime") == regime and x.get("order_seed") == order_seed and x.get("arm") == "LOW"), None)
            if high and low:
                sealed_at_fork = high["sealed_at_fork"]
                fork_qualified = bool(
                    sealed_at_fork["canonical"] >= 0.90
                    and sealed_at_fork["order_only"] >= 0.85
                    and sealed_at_fork["query_order"] >= 0.85
                )
                if fork_qualified:
                    paired.append((regime, order_seed,
                                   bool(high["result"]["sealed_retention"]["collapsed"]),
                                   bool(low["result"]["sealed_retention"]["collapsed"])))

    high_fail = sum(int(h) for _, _, h, _ in paired)
    low_fail = sum(int(l) for _, _, _, l in paired)
    reverse = sum(int((not h) and l) for _, _, h, l in paired)
    summary = {
        "canonical_train_qualified": canonical_q,
        "order_augmented_qualified": augmented_q,
        "sealed_qualified_pairs": len(paired),
        "high_failures": high_fail,
        "low_failures": low_fail,
        "reverse_discordance": reverse,
    }

    if augmented_q and not canonical_q:
        acquisition_verdict = "ORDER_ROBUSTNESS_REPAIRED"
    elif not augmented_q and not canonical_q:
        acquisition_verdict = "ROBUST_BINDING_NOT_ACQUIRED"
    else:
        acquisition_verdict = "ROBUST_BINDING_ACQUIRED"
    summary["acquisition_verdict"] = acquisition_verdict

    if not (canonical_q or augmented_q):
        transfer_verdict = "ROBUST_BINDING_NOT_ACQUIRED"
    elif len(paired) < 2:
        transfer_verdict = "ROBUST_BINDING_ACQUIRED_BUT_TRANSFER_INCONCLUSIVE"
    elif high_fail >= 1 and low_fail == 0 and reverse == 0:
        transfer_verdict = "NONARITHMETIC_LR_PROTECTION_SCREEN"
    elif high_fail + low_fail >= 1:
        transfer_verdict = "TRANSFER_NOT_SUPPORTED_SCREEN"
    else:
        transfer_verdict = "ROBUST_BINDING_ACQUIRED_BUT_TRANSFER_INCONCLUSIVE"
    summary["transfer_verdict"] = transfer_verdict
    return summary


def run_campaign(ctx: RunContext) -> dict:
    writer = ReceiptWriter(ctx, experiment_id="ARK-014", plan_sha=PLAN_SHA, runner_path=RUNNER_PATH)
    ark11 = load_ark11()
    bind_ark11_runtime(ark11, ctx.device, ctx.head)

    train_meta, control, sealed, manifest = build_binding_manifest()
    writer.save("ARK-014_TASK_MANIFEST.json", manifest)

    acquisitions = []
    live_acq = {}
    for regime in ["CANONICAL_TRAIN", "ORDER_AUGMENTED"]:
        if ctx.minutes_left < 12:
            acquisitions.append({"regime": regime, "status": "BUDGET_BLOCKED"})
            continue
        acq = train_regime(ark11, regime=regime, train_meta=train_meta, control=control, sealed=sealed)
        live_acq[regime] = acq
        acquisitions.append({k: v for k, v in acq.items() if k not in {"snapshot", "reference_flat"}})
        writer.save("ARK-014_PARTIAL.json", {"acquisitions": acquisitions, "forks": []})

    forks = []
    eligible = [r for r in ["ORDER_AUGMENTED", "CANONICAL_TRAIN"] if live_acq.get(r, {}).get("status") == "QUALIFIED"]
    for regime in eligible:
        acq = live_acq[regime]
        sealed_at_fork = acq["sealed_at_fork"]
        for order_seed in CONT_SEEDS:
            if ctx.minutes_left < 10:
                forks.append({"regime": regime, "order_seed": order_seed, "status": "BUDGET_BLOCKED"})
                continue
            high = run_continuation(
                ark11, acq=acq, train_meta=train_meta, control=control, sealed=sealed,
                regime=regime, order_seed=order_seed, lr=1e-3
            )
            low = run_continuation(
                ark11, acq=acq, train_meta=train_meta, control=control, sealed=sealed,
                regime=regime, order_seed=order_seed, lr=1e-5
            )
            forks.extend([
                {
                    "regime": regime,
                    "order_seed": order_seed,
                    "arm": "HIGH",
                    "status": "EXECUTED",
                    "sealed_at_fork": sealed_at_fork,
                    "result": high,
                },
                {
                    "regime": regime,
                    "order_seed": order_seed,
                    "arm": "LOW",
                    "status": "EXECUTED",
                    "sealed_at_fork": sealed_at_fork,
                    "result": low,
                },
            ])
            writer.save("ARK-014_PARTIAL.json", {"acquisitions": acquisitions, "forks": forks})

        if regime == "ORDER_AUGMENTED" and ctx.minutes_left < 20:
            break

    payload = {
        "status": "EXECUTED_OR_BUDGETED_PARTIAL",
        "manifest_sha256": manifest["manifest_sha256"],
        "acquisitions": acquisitions,
        "forks": forks,
        "summary": summarize(acquisitions, forks),
    }
    writer.save("ARK-014_RESULT.json", payload)
    return payload
