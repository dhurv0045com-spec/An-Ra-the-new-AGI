"""FORMATION-MUX-001 arm training loop.

One arm = one process = one GPU. The loop is identical for both
experiments: deterministic batch stream from the pregenerated surface,
legacy certified update path (begin/accumulate/finish with the global
clip and the real-token ledger), cadence evaluation, exact-resume
checkpoints with identity rejection, and candidate-free scoring. Arm
treatments differ only through ``formation_mux_model.make_optimizers``
and the M3 forward mask; diagnostics capture row-gradient norms BEFORE
frozen rows are zeroed, so the receipt proves the causal variable state.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from anra_v5 import formation_mux_model as fxm  # noqa: E402
from v5_training.optimizer import build_adamw_optimizer  # noqa: E402
from v5_experiments import formation_mux_protocol as proto  # noqa: E402


def _canonical(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _sha(payload: Any) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


class IdTokenizer:
    """Adapter over latent-ID rows: 'prompt'/'answer' text is the
    space-joined ID list; generation is scored as exact ID sequences with
    a valid EOS stop. Keeps the legacy batched evaluator interface."""

    vocab_size = 24576
    special = {"pad_id": 0, "unk_id": 1, "bos_id": 2, "eos_id": 3}

    class _Identity:
        vocabulary_size = 24576
        special_token_ids = {"pad": 0, "unk": 1, "bos": 2, "eos": 3}

    identity = _Identity()

    @staticmethod
    def encode(text: str) -> list[int]:
        return [int(t) for t in text.split()]

    @staticmethod
    def decode(ids: list[int]) -> str:
        return " ".join(str(int(i)) for i in ids)


def _prompt_answer_ids(row: Mapping[str, Any], experiment: str,
                       arm: str) -> tuple[list[int], list[int]]:
    bos, eos = 2, 3
    if experiment == proto.EXPERIMENT_A or arm == "R1_ISOMORPHIC_RENDERING":
        return [bos, *row["prompt_ids"]], [*row["answer_ids"], eos]
    return list(row["r0_prompt_ids"]), list(row["r0_answer_ids"])


def build_batch(rows: list[Mapping[str, Any]], experiment: str, arm: str, *,
                torch: Any, device: Any) -> tuple[Any, Any, Any, int]:
    bos, pad = 2, 0
    encoded = [_prompt_answer_ids(row, experiment, arm) for row in rows]
    width = max(len(p) + len(a) for p, a in encoded)
    tokens, segments, eligible = [], [], []
    supervised = 0
    for prompt, answer in encoded:
        ids = prompt + answer
        pad_n = width - len(ids)
        tokens.append(ids + [pad] * pad_n)
        segments.append([0] * len(ids) + [-1] * pad_n)
        mask = [False] * len(prompt) + [True] * len(answer) + [False] * pad_n
        eligible.append(mask)
        supervised += len(answer)
    t = torch.tensor(tokens, dtype=torch.long, device=device)
    s = torch.tensor(segments, dtype=torch.long, device=device)
    e = torch.tensor(eligible, dtype=torch.bool, device=device)
    return t, s, e, supervised


def _eval_rates(model: Any, rows: list[Mapping[str, Any]], experiment: str,
                arm: str, *, torch: Any, device: Any,
                max_new_tokens: int = 10) -> dict[str, float]:
    """Candidate-free greedy generation with a valid-EOS complete-exact rate."""

    from v5_model.core import packed_layout
    bos, eos, pad = 2, 3, 0
    was_training = model.training
    model.eval()
    content = complete = eos_stops = caps = 0
    encoded = [_prompt_answer_ids(row, experiment, arm) for row in rows]
    with torch.no_grad():
        for prompt, answer in encoded:
            ids = list(prompt)
            generated: list[int] = []
            stopped = False
            hit_cap = False
            for _ in range(max_new_tokens):
                current = torch.tensor([ids + generated], device=device)
                positions, mask = packed_layout(
                    torch.tensor([[0] * current.shape[1]], device=device),
                    torch_module=torch)
                logits = model(current, positions, mask)[0, -1]
                nxt = int(torch.argmax(logits).item())
                if nxt == eos:
                    stopped = True
                    break
                if nxt == pad:
                    break
                generated.append(nxt)
            else:
                hit_cap = True
            expected = [t for t in answer if t != eos]
            if generated == expected:
                content += 1
                if stopped:
                    complete += 1
            eos_stops += int(stopped)
            caps += int(hit_cap)
    if was_training:
        model.train()
    total = max(len(encoded), 1)
    return {"content_exact": round(content / total, 4),
            "complete_exact_with_valid_stop": round(complete / total, 4),
            "eos_rate": round(eos_stops / total, 4),
            "max_tokens_rate": round(caps / total, 4), "total": total}


def formation_summary(trace: list[dict[str, Any]], updates: int,
                      *, eligible_from: int = 600, g50: float = 0.5) -> dict[str, Any]:
    points = [(e["update"], e["dev_measurement"]["complete_exact_with_valid_stop"])
              for e in trace]
    eligible = [(u, s) for u, s in points if u >= eligible_from]
    sustained = None
    for i in range(len(eligible) - 2):
        if all(s >= g50 for _u, s in eligible[i:i + 3]):
            sustained = eligible[i][0]
            break
    endpoint = next((s for u, s in points if u == updates), None)
    return {"formation_auc": round(sum(s for _u, s in eligible) / len(eligible), 6)
            if eligible else 0.0,
            "sustained_g50_update": sustained,
            "endpoint": endpoint if endpoint is not None else 0.0,
            "eval_count": len(eligible)}


def save_checkpoint(path: Path, *, model: Any, optimizers: Mapping[str, Any],
                    torch: Any, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {"model": model.state_dict(),
            "main_optimizer": optimizers["main"].state_dict(),
            "row_optimizer": (optimizers["rows"].state_dict()
                              if optimizers.get("rows") else None),
            **payload}
    tmp = path.with_suffix(".tmp")
    torch.save(body, tmp)
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
    final = path.with_suffix(".pt")
    final.write_bytes(tmp.read_bytes())
    tmp.unlink(missing_ok=True)
    (path.parent / (path.stem + ".resume.json")).write_text(json.dumps(
        {k: v for k, v in payload.items() if not hasattr(v, "shape")},
        default=str, sort_keys=True), encoding="utf-8")
    return digest


def load_checkpoint(path: Path, *, model: Any, optimizers: Mapping[str, Any],
                    torch: Any, expected: Mapping[str, Any]) -> dict[str, Any]:
    body = torch.load(path, map_location="cpu", weights_only=False)
    for key, value in expected.items():
        if body.get(key) != value:
            raise RuntimeError(f"checkpoint identity mismatch {key}: "
                               f"{body.get(key)!r} != {value!r}")
    model.load_state_dict(body["model"])
    optimizers["main"].load_state_dict(body["main_optimizer"])
    if optimizers.get("rows") is not None and body.get("row_optimizer"):
        optimizers["rows"].load_state_dict(body["row_optimizer"])
    return body


def train_arm(*, experiment: str, arm: str, seed_bundle: int,
              surface: Mapping[str, Any], out_dir: Path, torch: Any,
              device: Any, updates: int | None = None,
              deadline: float | None = None,
              progress: Callable[[str], None] | None = None,
              ) -> dict[str, Any]:
    """Run/resume one official arm. Deterministic in (experiment, arm,
    seed_bundle, surface manifest). Never touches sealed rows."""

    from v5_model.core import initialize
    from v5_training.production_backend import ProductionTrainingBackend
    from v5_training.state import CURSOR_SCHEMA, CursorState

    updates = proto.UPDATES if updates is None else int(updates)
    label = f"{experiment}/{arm}/S{proto.SEED_BUNDLES.index(seed_bundle) + 1}"
    root = Path(out_dir) / experiment / arm / f"S{proto.SEED_BUNDLES.index(seed_bundle) + 1}"
    final_path = root / "ARM_RESULT.json"
    if final_path.exists():
        old = json.loads(final_path.read_text(encoding="utf-8"))
        if (old.get("status") == "COMPLETE"
                and old.get("protocol_sha256") == proto.protocol_sha(experiment)
                and int(old.get("updates", -1)) == updates):
            old["resume_action"] = "SKIPPED_COMPLETED_ARM"
            return old
        if old.get("status") == "COMPLETE" and int(old.get("updates", -1)) > updates:
            raise RuntimeError(f"incompatible completed arm exists: {final_path}")
        # fewer recorded updates than now requested: resume forward from
        # the checkpoint (engineering extension or partitioned execution)
    root.mkdir(parents=True, exist_ok=True)

    if experiment not in proto.EXPERIMENTS or arm not in (
            proto.ARMS_A if experiment == proto.EXPERIMENT_A else proto.ARMS_B):
        raise ValueError(f"arm {arm} not registered for {experiment}")
    if "sealed" in json.dumps(list(surface["splits"].keys())):
        train_rows = [row for row in surface["splits"]["training"]]
        dev_rows = [row for row in surface["splits"]["dev_measurement"]] \
            if "dev_measurement" in surface["splits"] else \
            [row for row in surface["splits"]["development"]]
    else:
        raise RuntimeError("surface manifest splits invalid")
    seed = seed_bundle if experiment == proto.EXPERIMENT_A else proto.b_seed(seed_bundle)
    data_sha = surface["sha256"]
    stream_seed = seed
    order = torch.randperm(len(train_rows), generator=torch.Generator().manual_seed(stream_seed))

    torch.manual_seed(seed)
    model = fxm.build_model(seed, arm, torch=torch, device=device) \
        if experiment == proto.EXPERIMENT_A else \
        initialize(fxm.spec(), int(seed), torch_module=torch).to(device)
    optimizers = fxm.make_optimizers(model, arm, torch=torch, lr=proto.LR) \
        if experiment == proto.EXPERIMENT_A else \
        {"main": build_adamw_optimizer(model, torch_module=torch, lr=proto.LR),
         "rows": None}
    backend = ProductionTrainingBackend(
        model=model, optimizer=optimizers.get("view", optimizers["main"]),
        bos_id=2, pad_id=0,
        device=device, schedule=lambda cumulative_tokens: proto.LR,
        bfloat16_autocast=False, torch_module=torch,
        activation_checkpointing=False)

    identity = {"experiment": experiment, "arm": arm, "seed_bundle": seed_bundle,
                "data_manifest_sha256": data_sha,
                "protocol_sha256": proto.protocol_sha(experiment),
                "target_updates": proto.UPDATES}
    initial_flat = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    initial_sha = hashlib.sha256(initial_flat.numpy().tobytes()).hexdigest()
    checkpoint = root / "resume.pt"
    state = {"updates": 0, "real_tokens": 0, "trace": [], "diagnostics": []}
    if checkpoint.exists():
        saved = load_checkpoint(checkpoint, model=model, optimizers=optimizers,
                                torch=torch, expected=identity)
        if int(saved.get("run_updates", 0)) > updates:
            raise RuntimeError(
                f"checkpoint at {saved.get('run_updates')} updates exceeds "
                f"the requested run target {updates}: refusing to rewind")
        state = {"updates": int(saved["updates"]),
                 "real_tokens": int(saved["real_tokens"]),
                 "trace": list(saved.get("trace", [])),
                 "diagnostics": list(saved.get("diagnostics", []))}
        if saved.get("initial_model_sha256") != initial_sha:
            raise RuntimeError("initial-model identity drift on resume")
        if progress:
            progress(f"RESUME {label}: {state['updates']}/{updates}")

    embedding = model.embedding.weight
    started = time.monotonic()
    status = "RUNNING"
    while state["updates"] < updates:
        if deadline is not None and time.monotonic() >= deadline:
            status = "PARTIAL_TIMEBOX"
            break
        start = state["updates"] * proto.BATCH_ROWS
        indices = order[start % len(train_rows):
                        (start % len(train_rows)) + proto.BATCH_ROWS].tolist()
        if len(indices) < proto.BATCH_ROWS:
            indices = (indices + order.tolist())[:proto.BATCH_ROWS]
        rows = [train_rows[i] for i in indices]
        tokens, segments, eligible, supervised = build_batch(
            rows, experiment, arm, torch=torch, device=device)
        embedding.grad = torch.zeros_like(embedding)
        ctx = backend.begin_update(
            type("S", (), {"cumulative_tokens": state["real_tokens"]})())
        ctx = backend.accumulate_microstep(
            ctx, tokens=tokens, segment_ids=segments, eligible=eligible,
            tokens_by_source={"mux": supervised},
            planned_total=supervised)
        grad = embedding.grad
        extra_index = getattr(model, "_extra_row_index", None)
        if extra_index is None:
            extra_index = torch.tensor(list(fxm.EXTRA_ROWS[:4096]),
                                       dtype=torch.long, device=device)
            model._extra_row_index = extra_index
        diag = {"update": state["updates"] + 1,
                "active_row_grad_norm": float(
                    grad.index_select(0, torch.tensor(
                        list(fxm.ACTIVE_ROWS), dtype=torch.long,
                        device=device)).norm().item()),
                "inactive_row_grad_norm_first4096": float(
                    grad.index_select(0, extra_index).norm().item())}
        fxm.apply_decay_then_zero_frozen(model, arm, torch=torch)
        # The row optimizer steps BEFORE the certified boundary so the
        # certificate observes the embedding's Adam step advancing exactly
        # once. Unclipped embedding grads are part of the declared arm
        # treatment and identical across all four arms (no contrast leak).
        if optimizers["rows"] is not None:
            optimizers["rows"].set_lr(proto.LR)
            optimizers["rows"].step()
        backend.finish_update(
            type("S", (), {"cumulative_tokens": state["real_tokens"]})(), ctx,
            planned_total=supervised,
            cursor=CursorState(CURSOR_SCHEMA, data_sha, state["updates"] + 1, 0, 0))
        state["updates"] += 1
        state["real_tokens"] += supervised
        if progress and state["updates"] % 100 == 0:
            progress(f"{label}: {state['updates']}/{updates}")
        if state["updates"] % proto.EVAL_EVERY == 0 or state["updates"] == updates:
            rates = _eval_rates(model, dev_rows, experiment, arm, torch=torch,
                                device=device)
            state["trace"].append({"update": state["updates"],
                                   "dev_measurement": rates,
                                   **diag})
        if state["updates"] % proto.CHECKPOINT_EVERY == 0:
            save_checkpoint(checkpoint, model=model, optimizers=optimizers,
                            torch=torch, payload={**identity, **state,
                                                  "initial_model_sha256": initial_sha})

    summary = formation_summary(state["trace"], state["updates"])
    body = {"schema": "anra.formation-mux-arm/v1", "experiment": experiment,
            "arm": arm, "seed_bundle": seed_bundle, "status": status,
            "updates": state["updates"], "real_tokens": state["real_tokens"],
            "protocol_sha256": proto.protocol_sha(experiment),
            "data_manifest_sha256": data_sha,
            "initial_model_sha256": initial_sha,
            "formation": summary, "trace": state["trace"],
            "diagnostics": state["diagnostics"],
            "wall_seconds": round(time.monotonic() - started, 1)}
    if status == "RUNNING":
        body["status"] = "COMPLETE"
        final_path.write_text(json.dumps(body, indent=2, default=str) + "\n",
                              encoding="utf-8")
        save_checkpoint(checkpoint, model=model, optimizers=optimizers,
                        torch=torch, payload={**identity, **state,
                                              "initial_model_sha256": initial_sha,
                                              "complete": True})
    else:
        (root / "PARTIAL.json").write_text(json.dumps(body, indent=2,
                                                      default=str) + "\n",
                                           encoding="utf-8")
        save_checkpoint(checkpoint, model=model, optimizers=optimizers,
                        torch=torch, payload={**identity, **state,
                                              "initial_model_sha256": initial_sha})
    return body
