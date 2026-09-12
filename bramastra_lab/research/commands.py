"""CLI command implementations for the integrated build path (B11).

Implements prepare-data -> train -> checkpoint -> fresh-process resume ->
infer -> evaluate -> package over the real modules. Learned work records its
exact optimizer-update count and wall seconds in the cumulative session
ledger; ``--smoke`` refuses to start when the remaining budget cannot cover
the declared work, and every run records its resolved configuration.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

from bramastra_lab.research.errors import CommandError
from bramastra_lab.research.config import BuildConfig, seed_everything
from bramastra_lab.research.contracts.core import content_identity
from bramastra_lab.research.data.manifest import DatasetError, load_dataset
from bramastra_lab.research.experience.codec import (
    SPECIAL_BOUNDARY,
    SPECIAL_EOS,
    codec_identity,
    encode_event,
    encode_text,
)
from bramastra_lab.research.experience.sequences import (
    SequenceRow,
    build_answer_row,
    build_language_row,
    collocate,
)
from bramastra_lab.research.runtime.smoke import SmokeBudgetExhausted

_LEDGER_RELATIVE = os.path.join("engineering", "reports", "B2", "SESSION_LEDGER.json")


def _ledger_path() -> str:
    # bramastra_lab/research/commands.py -> worktree root is two levels up.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(repo_root, _LEDGER_RELATIVE)


def _ledger():
    from bramastra_lab.research.runtime.smoke import SessionLedger

    return SessionLedger(_ledger_path())


def _load_config(config_path: str) -> BuildConfig:
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise CommandError(f"cannot read config {config_path}: {exc}")
    return BuildConfig.from_dict(raw)


# -- prepare-data --------------------------------------------------------------

def _row_provenance(example) -> dict[str, str]:
    return {
        "episode_id": example.example_id,
        "task_semantic_id": example.example_id,
        "split": example.split,
        "source": example.source,
        "collection_policy": "operator-manifest/v1",
        "family": example.family,
    }


def _render_example(example, config: BuildConfig) -> dict[str, Any]:
    max_tokens = config.model.max_seq
    provenance = _row_provenance(example)
    if example.kind == "language":
        row = build_language_row(example.content["text"], provenance=provenance,
                                 max_tokens=max_tokens)
        return _row_record(row, kind="language", prompt_length=None, label=None)
    prompt_events = [(event[0], event[1]) for event in example.content["prompt_events"]]
    row = build_answer_row(prompt_events, example.content["answer"],
                           provenance=provenance, max_tokens=max_tokens)
    prompt_length = 1 + sum(len(encode_event(role, content))
                            for role, content in prompt_events)
    return _row_record(row, kind="trajectory", prompt_length=prompt_length,
                       label=example.content["answer"])


def _row_record(row: SequenceRow, *, kind: str, prompt_length: int | None,
                label: str | None) -> dict[str, Any]:
    return {
        "kind": kind,
        "tokens": list(row.tokens),
        "supervised": [bool(flag) for flag in row.supervised],
        "prompt_length": prompt_length,
        "label": label,
        "pair_group_id": row.pair_group_id,
        "provenance": dict(row.provenance),
    }


def prepare_data(manifest_path: str, out_dir: str, config_path: str) -> int:
    config = _load_config(config_path)
    if os.path.exists(out_dir):
        raise CommandError(f"output directory {out_dir} already exists; "
                           "use a new directory per preparation")
    try:
        handle = load_dataset(manifest_path)
    except DatasetError as exc:
        raise CommandError(f"{exc} (status: {exc.status})")
    os.makedirs(out_dir)
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in sorted({example.split for example in handle.examples}):
        rows = [_render_example(example, config)
                for example in handle.examples_for_split(split)]
        rows_by_split[split] = rows
        with open(os.path.join(out_dir, f"rows-{split}.jsonl"), "w",
                  encoding="utf-8") as out:
            for record in rows:
                out.write(json.dumps(record, sort_keys=True) + "\n")
    prepared = {
        "schema": "bramastra-prepared-data/v1",
        "dataset_identity": handle.identity,
        "dataset_name": handle.name,
        "config_identity": config.identity(),
        "split_inventory": {
            split: {"examples": len(rows),
                    "supervised_targets": sum(sum(record["supervised"][1:])
                                              for record in rows)}
            for split, rows in rows_by_split.items()},
        "created_unix": time.time(),
    }
    prepared["identity"] = content_identity(
        {key: value for key, value in prepared.items()
         if key not in ("identity", "created_unix")})
    with open(os.path.join(out_dir, "prepared.json"), "w", encoding="utf-8") as out:
        json.dump(prepared, out, indent=2, sort_keys=True)
    print(json.dumps({"status": "PREPARED", "out_dir": out_dir,
                      "identity": prepared["identity"],
                      "split_inventory": prepared["split_inventory"]}, indent=2))
    return 0


# -- prepared-row helpers ------------------------------------------------------

class _RowRef:
    __slots__ = ("record", "group_id", "split")

    def __init__(self, record: dict[str, Any]) -> None:
        self.record = record
        self.group_id = record.get("pair_group_id")
        self.split = record["provenance"].get("split", "training")


def _row_from_record(record: dict[str, Any]) -> SequenceRow:
    return SequenceRow(
        tokens=tuple(record["tokens"]),
        supervised=tuple(record["supervised"]),
        provenance=record["provenance"],
        segment_ids=(1,) * len(record["tokens"]),
        pair_group_id=record.get("pair_group_id"))


def _load_rows(data_dir: str, split: str) -> list[dict[str, Any]]:
    path = os.path.join(data_dir, f"rows-{split}.jsonl")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_prepared(data_dir: str, config: BuildConfig) -> dict[str, Any]:
    path = os.path.join(data_dir, "prepared.json")
    if not os.path.exists(path):
        raise CommandError(f"{data_dir} has no prepared.json; run prepare-data first")
    with open(path, "r", encoding="utf-8") as handle:
        prepared = json.load(handle)
    if prepared["config_identity"] != config.identity():
        raise CommandError("prepared data was built under a different config identity; "
                           "prepare again with this config")
    return prepared


def _sampler_units(rows: list[dict[str, Any]]) -> list[tuple[_RowRef, ...]]:
    refs = [_RowRef(record) for record in rows]
    groups: dict[str, list[_RowRef]] = {}
    singles: list[_RowRef] = []
    for ref in refs:
        if ref.group_id:
            groups.setdefault(ref.group_id, []).append(ref)
        else:
            singles.append(ref)
    return ([tuple(bucket) for _, bucket in sorted(groups.items())]
            + [(single,) for single in singles])


def _collocate_records(records: list[dict[str, Any]], max_seq: int):
    return collocate([_row_from_record(record) for record in records], max_seq=max_seq)


def _build_pair_input(batch_rows: list[dict[str, Any]]):
    """Render the counterfactual pair objective inputs from one sampled batch.

    Own batch: each pair member under its own prompt with its own answer.
    Swapped batch: the same prompts with the other member's answer. Pairs
    with legitimately identical answers are excluded here and therefore
    contribute neither loss nor count.
    """
    from bramastra_lab.research.learning.trainer import PairUpdateInput

    groups: dict[str, list[dict[str, Any]]] = {}
    for record in batch_rows:
        group_id = record.get("pair_group_id")
        if group_id:
            groups.setdefault(group_id, []).append(record)
    own_rows: list[SequenceRow] = []
    swapped_rows: list[SequenceRow] = []
    for group_id, members in groups.items():
        if len(members) != 2:
            continue
        first, second = members
        if first["label"] == second["label"]:
            continue
        own_rows.append(_row_from_record(first))
        own_rows.append(_row_from_record(second))
        swapped_first = {**first, "tokens": _swap_answer(first, second),
                         "supervised": _answer_flags(first, second)}
        swapped_second = {**second, "tokens": _swap_answer(second, first),
                          "supervised": _answer_flags(second, first)}
        swapped_rows.append(_row_from_record(swapped_first))
        swapped_rows.append(_row_from_record(swapped_second))
    if not own_rows:
        return None
    max_seq = max(len(row.tokens) for row in own_rows + swapped_rows) - 1
    return PairUpdateInput(own=collocate(own_rows, max_seq=max_seq),
                           swapped=collocate(swapped_rows, max_seq=max_seq))


def _swap_answer(own: dict[str, Any], other: dict[str, Any]) -> list[int]:
    """``own``'s prompt with ``other``'s answer (and EOS) re-appended."""
    prompt = list(own["tokens"][:own["prompt_length"]])
    other_answer = list(other["tokens"][other["prompt_length"]:-1])
    return prompt + other_answer + [own["tokens"][-1]]


def _answer_flags(own: dict[str, Any], other: dict[str, Any]) -> list[bool]:
    other_answer_length = len(other["tokens"]) - other["prompt_length"] - 1
    return [False] * own["prompt_length"] + [True] * (other_answer_length + 1)


def _checkpoint(trainer, run_dir: str, ctx, *, milestone: str | None = None):
    from bramastra_lab.research.runtime.resume import append_event, checkpoint_run

    manifest = checkpoint_run(trainer, run_dir, ctx, milestone=milestone)
    append_event(run_dir, {"event": "checkpoint", "checkpoint_id": manifest.checkpoint_id,
                           "update": trainer.counters.optimizer_updates})
    return manifest


def _source_identity() -> str:
    """Git HEAD plus dirty flag, recorded into every run for provenance."""
    import subprocess

    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_repo_root(), text=True, timeout=10).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=_repo_root(), text=True,
            timeout=10).strip() != ""
        return f"{head}{'-dirty' if dirty else ''}"
    except Exception:
        return "unavailable"


def _write_data_source(run_dir: str, data_dir: str) -> None:
    with open(os.path.join(run_dir, "data_source.json"), "w", encoding="utf-8") as handle:
        json.dump({"data_dir": os.path.abspath(data_dir)}, handle)


def _read_data_source(run_dir: str) -> str:
    path = os.path.join(run_dir, "data_source.json")
    if not os.path.exists(path):
        raise CommandError("run has no data_source.json; cannot relocate its data")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)["data_dir"]


# -- preflight (fail-closed, V5 launch-gate discipline) -----------------------

PREFLIGHT_GATES = ["config", "prepared_data", "training_rows", "updates_declared",
                   "smoke_profile", "smoke_budget", "controller_split", "replay_ledger"]


def _preflight(config: BuildConfig, data_dir: str, max_updates: int | None, *,
               smoke: bool, ledger=None) -> dict[str, Any]:
    """Ordered fail-closed launch gates. Any FAIL refuses the run.

    Gate statuses follow the V5 discipline: a PASS must point at something
    that actually exists; a FAIL is never downgraded; nothing here claims
    more than was checked.
    """
    gates: list[dict[str, Any]] = []
    rows = _load_rows(data_dir, "training")
    prepared_exists = os.path.exists(os.path.join(data_dir, "prepared.json"))

    def gate(name: str, passed: bool, detail: str) -> None:
        gates.append({"id": name, "status": "PASS" if passed else "FAIL",
                      "detail": detail})

    gate("config", config is not None, f"profile={config.model.profile}")
    gate("prepared_data", prepared_exists,
         "prepared.json present" if prepared_exists else "prepared.json missing")
    gate("training_rows", bool(rows), f"{len(rows)} training rows")
    gate("updates_declared", bool(max_updates and max_updates > 0),
         f"max_updates={max_updates}")
    if smoke:
        gate("smoke_profile", config.model.profile == "tiny",
             f"profile={config.model.profile} (smoke requires tiny)")
        if ledger is not None:
            remaining = ledger.remaining("cpu")
            sufficient = remaining["updates"] >= (max_updates or 0) \
                and remaining["seconds"] >= 30.0 + 12.0 * (max_updates or 0)
            gate("smoke_budget", sufficient,
                 f"remaining updates={remaining['updates']:.0f}, "
                 f"seconds={remaining['seconds']:.0f}")
        else:
            gate("smoke_budget", False, "session ledger unavailable")
    else:
        gate("smoke_profile", True, "not a smoke run")
        gate("smoke_budget", True, "not a smoke run")
    if config.controller.mode == "evidence_driven":
        controller_rows = _load_rows(data_dir, "controller")
        gate("controller_split", bool(controller_rows),
             f"{len(controller_rows)} controller rows")
    else:
        gate("controller_split", True,
             f"controller mode {config.controller.mode!r} needs no pool")
    if config.replay.enabled:
        replay_path = os.path.join(data_dir, config.replay.ledger_path)
        gate("replay_ledger", os.path.exists(replay_path), f"ledger at {replay_path}")
    else:
        gate("replay_ledger", True, "replay disabled")
    return {"schema": "bramastra-preflight/v1", "gates": gates,
            "fail_closed": True,
            "ready": all(item["status"] == "PASS" for item in gates)}


def _load_replay_engine(config: BuildConfig, data_dir: str):
    """Build the replay engine from the prepared-data ledger, or None."""
    from bramastra_lab.research.experience.ledger import ExperienceLedger
    from bramastra_lab.research.experience.replay import ReplayEngine

    if not config.replay.enabled:
        return None
    path = os.path.join(data_dir, config.replay.ledger_path)
    if not os.path.exists(path):
        return None
    ledger = ExperienceLedger(path)
    entries = [receipt for _, receipt in ledger.read_all()]
    weights = config.replay.family_weights or {}
    return ReplayEngine(entries, family_weights=weights, batch_size=2,
                        seed=config.training.seed, dataset_identity=ledger.identity())


def _rows_from_receipt(receipt, *, max_tokens: int):
    """Render an episode receipt's public transcript into a supervised row.

    Rendering rule (declared): the full action/feedback transcript is used
    when it fits the configured sequence limit; otherwise only the episode
    endpoints (first and last step) are rendered; if even that does not fit,
    the episode is skipped and counted — nothing is ever truncated silently.
    The supervised answer is the final submitted answer. Receipts without a
    submission render to None.
    """
    transcript = receipt.notes.get("transcript") or {}
    steps = transcript.get("steps") or []
    if len(steps) < 2:
        return None, None
    final = steps[-1]
    submitted = final["feedback"].get("submitted_answer")
    if submitted is None:
        return None, None

    def render(selected):
        tokens: list[int] = [SPECIAL_BOUNDARY]
        supervised: list[bool] = [False]
        for step in selected:
            event = encode_event("action", step["action"])
            tokens += event
            supervised += [False] * len(event)
            event = encode_event("feedback", step["feedback"])
            tokens += event
            supervised += [False] * len(event)
        answer = encode_text(str(submitted))
        tokens += answer
        supervised += [True] * len(answer)
        tokens.append(SPECIAL_EOS)
        supervised.append(True)
        return tokens, supervised

    mode = "full"
    tokens, supervised = render(steps[:-1])
    if len(tokens) > max_tokens:
        mode = "endpoints"
        tokens, supervised = render(steps[:-1][:1] + steps[-2:-1])
    if len(tokens) > max_tokens:
        return None, "oversized"
    row = SequenceRow(
        tokens=tuple(tokens), supervised=tuple(supervised),
        provenance={"kind": "replay", "episode_id": receipt.episode_id,
                    "task_semantic_id": receipt.family, "split": "training",
                    "source": "experience-ledger",
                    "collection_policy": receipt.collection_policy,
                    "family": receipt.family},
        segment_ids=(1,) * len(tokens))
    return row, mode


def _evaluate_controller_families(trainer, config: BuildConfig, controller_rows):
    """Score each controller-pool family by greedy complete-answer accuracy.

    Uses the real inference path (full-vocabulary greedy generation over each
    row's prompt), never dev or sealed data.
    """
    from bramastra_lab.research.models import IntegratedModel  # noqa: F401
    from bramastra_lab.research.runtime.inference import generate_free_form

    by_family: dict[str, list[dict[str, Any]]] = {}
    for record in controller_rows:
        by_family.setdefault(record["provenance"].get("family", "default"), []).append(record)
    scores: dict[str, float] = {}
    for family, rows in sorted(by_family.items()):
        scored = [record for record in rows
                  if record["kind"] == "trajectory" and record.get("label") is not None]
        if not scored:
            continue
        correct = 0
        for record in scored:
            prompt = list(record["tokens"][:record["prompt_length"]])
            report = generate_free_form(trainer.model, config, prompt, max_new_tokens=8)
            if report.stopped_on_eos and report.answer == record["label"]:
                correct += 1
        scores[family] = correct / len(scored)
    return scores


def _controller_boundary(trainer, ctx, config: BuildConfig, data_dir: str,
                         controller_state, *, current_update: int, run_dir: str):
    """Run one controller evaluation+transition at an optimizer boundary."""
    from bramastra_lab.research.learning.plasticity import (
        ControllerMetrics,
        transition,
    )
    from bramastra_lab.research.runtime.resume import append_event

    controller_rows = _load_rows(data_dir, "controller")
    if not controller_rows:
        return controller_state, None
    scores = _evaluate_controller_families(trainer, config, controller_rows)
    if not scores:
        return controller_state, None
    telemetry = trainer.diagnostics.last_telemetry if trainer.diagnostics else {}
    metrics = ControllerMetrics(
        pool_id=config.controller.controller_pool_id or "controller",
        at_update=current_update,
        family_scores=scores,
        parameter_displacement=telemetry.get("parameter_displacement_l2"),
        relative_parameter_displacement=telemetry.get("parameter_displacement_relative"),
    )
    family_request = None
    if controller_state.state == "FORM" and controller_state.acquiring_family is None             and len(scores) == 1:
        # The runner introduces the first acquisition family from the
        # controller pool; further introductions are operator decisions.
        family_request = next(iter(scores))
    controller_state, decision = transition(controller_state, metrics,
                                            config.controller, current_update=current_update,
                                            family_request=family_request)
    trainer.set_controller_multiplier(decision.lr_multiplier, decision.reason)
    if decision.checkpoint_requested:
        _checkpoint(trainer, run_dir, ctx)
    append_event(run_dir, {"event": "controller_boundary",
                           "update": current_update, "decision": decision.to_dict()})
    return controller_state, decision


def _training_loop(trainer, ctx, config: BuildConfig, data_dir: str, run_dir: str,
                   sampler, *, updates: int, controller_state,
                   replay_engine, checkpoint_every: int = 4) -> dict[str, Any]:
    """One shared update loop for train and resume (no parallel paths).

    Each update draws a data batch; updates designated by the replay
    proportion draw from the replay engine instead. At controller evaluation
    boundaries the plasticity controller may change the LR multiplier, request
    a checkpoint, or pause updates (HOLD). Returns an exact report.
    """
    interval = max(1, round(1.0 / config.replay.proportion)) \
        if config.replay.enabled and replay_engine is not None \
        and config.replay.proportion > 0 else 0
    replay_updates_expected = 0
    if interval:
        replay_updates_expected = updates // interval
        replay_engine.declare_planned(replay_updates_expected)
    updates_done = 0
    last_checkpointed = -1
    hold_reason: str | None = None
    replay_batches = 0
    render_modes: dict[str, int] = {}
    for index in range(updates):
        if interval and (index + 1) % interval == 0:
            replay_batch = replay_engine.sample()
            rendered = []
            for receipt in replay_batch.entries:
                row, mode = _rows_from_receipt(receipt, max_tokens=config.model.max_seq)
                if row is not None:
                    rendered.append(row)
                    render_modes[mode] = render_modes.get(mode, 0) + 1
                elif mode == "oversized":
                    render_modes["oversized_skipped"] = render_modes.get("oversized_skipped", 0) + 1
            if rendered:
                trainer.training_step(_collocate_rows(rendered, config.model.max_seq))
                replay_batches += 1
            # A shortfall is recorded even when nothing could be rendered.
            updates_done += 1
        else:
            batch_refs = sampler.take_batch()
            batch = _collocate_records([ref.record for ref in batch_refs],
                                       config.model.max_seq)
            pair_input = _build_pair_input([ref.record for ref in batch_refs]) \
                if config.training.pair_loss_weight > 0 else None
            trainer.training_step(batch, pair_input=pair_input)
            updates_done += 1
        if config.controller.mode == "evidence_driven" \
                and updates_done % config.controller.controller_eval_every == 0:
            controller_state, decision = _controller_boundary(
                trainer, ctx, config, data_dir, controller_state,
                current_update=trainer.counters.optimizer_updates, run_dir=run_dir)
            if decision is not None and decision.pause_updates:
                hold_reason = decision.reason
                break
        if updates_done % checkpoint_every == 0 and updates_done < updates:
            _checkpoint(trainer, run_dir, ctx)
            last_checkpointed = updates_done
    reconciliation = replay_engine.reconciliation() if replay_engine is not None else None
    if replay_engine is not None and reconciliation is not None:
        reconciliation["designated_updates"] = replay_updates_expected
        reconciliation["executed_replay_batches"] = replay_batches
        reconciliation["render_modes"] = dict(render_modes)
        # Planned entries are the designated updates scaled by replay batch
        # size; consumed entries are what the pools actually delivered.
        reconciliation["planned"] = replay_updates_expected * replay_engine.batch_size
    return {
        "updates_done": updates_done,
        "last_checkpointed": last_checkpointed,
        "hold_reason": hold_reason,
        "controller_state": controller_state,
        "replay_reconciliation": reconciliation,
    }


def _collocate_rows(rows: list[SequenceRow], max_seq: int):
    return collocate(rows, max_seq=max_seq)


# -- train / resume ------------------------------------------------------------

def train(config_path: str, data_dir: str, run_dir: str, max_updates: int | None,
          smoke: bool) -> int:
    from bramastra_lab.research.data.sampler import GroupSampler
    from bramastra_lab.research.learning.trainer import Trainer, TrainerDiagnostics
    from bramastra_lab.research.learning.plasticity import ControllerState
    from bramastra_lab.research.models import IntegratedModel
    from bramastra_lab.research.runtime.resume import (
        RunContext,
        append_event,
        create_run,
        read_run_manifest,
    )

    config = _load_config(config_path)
    ledger = _ledger()
    preflight = _preflight(config, data_dir, max_updates, smoke=smoke, ledger=ledger)
    if not preflight["ready"]:
        raise CommandError("preflight failed (fail-closed): "
                           + json.dumps([g for g in preflight["gates"]
                                         if g["status"] == "FAIL"]))
    prepared = _read_prepared(data_dir, config)
    rows = _load_rows(data_dir, "training")
    sampler = GroupSampler(_sampler_units(rows), batch_size=min(4, len(rows)),
                           seed=config.training.seed)
    replay_engine = _load_replay_engine(config, data_dir)

    started = time.monotonic()
    seed_everything(config.training.seed)
    if os.path.exists(run_dir):
        raise CommandError(f"run directory {run_dir} already exists; use a new run id")
    create_run(run_dir, config, data_identity=prepared["identity"])
    with open(os.path.join(run_dir, "config_used.json"), "w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2, sort_keys=True)
    _write_data_source(run_dir, data_dir)
    with open(os.path.join(run_dir, "source.json"), "w", encoding="utf-8") as handle:
        json.dump({"source_identity": _source_identity()}, handle, indent=2)
    with open(os.path.join(run_dir, "preflight.json"), "w", encoding="utf-8") as handle:
        json.dump(preflight, handle, indent=2, sort_keys=True)
    trainer = Trainer(config, IntegratedModel(config))
    diagnostic_batch = _collocate_records(rows[:1], config.model.max_seq)
    trainer.set_diagnostics(TrainerDiagnostics(trainer.model, diagnostic_batch))
    ctx = RunContext(run_dir=run_dir, run_id=read_run_manifest(run_dir)["run_id"],
                     config=config, data_identity=prepared["identity"],
                     writer_token="train", trainer=trainer,
                     sampler_state=None, controller_state=None, replay_cursor=None)
    controller_state = ControllerState()
    result = _training_loop(trainer, ctx, config, data_dir, run_dir, sampler,
                            updates=max_updates, controller_state=controller_state,
                            replay_engine=replay_engine)
    updates_done = result["updates_done"]
    ctx.sampler_state = sampler.state().to_dict()
    ctx.controller_state = result["controller_state"].to_dict()
    milestone = "final" if result["last_checkpointed"] != updates_done else None
    manifest = _checkpoint(trainer, run_dir, ctx, milestone=milestone)
    elapsed = time.monotonic() - started
    if smoke:
        ledger.record(device="cpu", updates=updates_done, seconds=elapsed,
                      what=f"train smoke {os.path.basename(run_dir)}",
                      evidence=manifest.checkpoint_id)
    append_event(run_dir, {"event": "train_complete", "updates": updates_done,
                           "seconds": round(elapsed, 3),
                           "hold_reason": result["hold_reason"]})
    print(json.dumps({
        "status": "TRAINED", "run_dir": run_dir, "updates": updates_done,
        "seconds": round(elapsed, 3), "checkpoint_id": manifest.checkpoint_id,
        "counters": trainer.counters.to_dict(),
        "hold_reason": result["hold_reason"],
        "replay_reconciliation": result["replay_reconciliation"],
    }, indent=2))
    return 0


def resume(run_dir: str, max_updates: int | None, expect_parent: str | None) -> int:
    from bramastra_lab.research.data.sampler import GroupSampler, SamplerState
    from bramastra_lab.research.learning.plasticity import ControllerState

    ledger = _ledger()
    if max_updates is None or max_updates <= 0:
        raise CommandError("--max-updates is required for resume")
    try:
        ledger.check_can_run(device="cpu", updates=max_updates,
                             seconds=30.0 + 12.0 * max_updates,
                             reserve_for_resume=False)
    except SmokeBudgetExhausted as exc:
        raise CommandError(f"smoke budget refuses this resume: {exc}")
    with open(os.path.join(run_dir, "config_used.json"), "r", encoding="utf-8") as handle:
        config = BuildConfig.from_dict(json.load(handle))
    data_dir = _read_data_source(run_dir)
    started = time.monotonic()
    ctx = resume_mod_restore_run(run_dir, config)
    if expect_parent:
        from bramastra_lab.research.runtime import checkpoint as ckpt

        _, latest_manifest = ckpt.load_checkpoint(run_dir)
        if latest_manifest.checkpoint_id != expect_parent:
            raise CommandError(
                f"latest checkpoint {latest_manifest.checkpoint_id[:12]}... is not the "
                f"expected parent {expect_parent[:12]}...; refusing stale or divergent resume")
    rows = _load_rows(data_dir, "training")
    if not rows:
        raise CommandError("prepared training rows are missing; resume needs the "
                           "same prepared data directory")
    sampler = GroupSampler(_sampler_units(rows), batch_size=min(4, len(rows)),
                           seed=config.training.seed)
    if ctx.sampler_state:
        sampler.restore(SamplerState.from_dict(ctx.sampler_state))
    controller_state = ControllerState.from_dict(ctx.controller_state) \
        if ctx.controller_state else ControllerState()
    trainer = ctx.trainer
    updates_before = trainer.counters.optimizer_updates
    result = _training_loop(trainer, ctx, config, data_dir, run_dir, sampler,
                            updates=max_updates, controller_state=controller_state,
                            replay_engine=_load_replay_engine(config, data_dir))
    updates_done = result["updates_done"]
    ctx.sampler_state = sampler.state().to_dict()
    ctx.controller_state = result["controller_state"].to_dict()
    milestone = "resumed" if result["last_checkpointed"] != updates_done else None
    manifest = _checkpoint(trainer, run_dir, ctx, milestone=milestone)
    elapsed = time.monotonic() - started
    ledger.record(device="cpu", updates=updates_done, seconds=elapsed,
                  what=f"fresh-process resume {os.path.basename(run_dir)}",
                  evidence=manifest.checkpoint_id)
    print(json.dumps({
        "status": "RESUMED", "run_dir": run_dir,
        "updates_before": updates_before,
        "updates_after": trainer.counters.optimizer_updates,
        "seconds": round(elapsed, 3), "checkpoint_id": manifest.checkpoint_id,
        "hold_reason": result["hold_reason"],
    }, indent=2))
    return 0


# -- collect (experience collection) -------------------------------------------

COLLECTABLE_ENVIRONMENTS = {
    "switch-world": "SwitchWorld",
    "inventory-world": "InventoryWorld",
    "program-lab": "ProgramLab",
}


def collect(environments: str, ledger_path: str, *, episodes: int, policy: str,
            budget: int, seed: int) -> int:
    """Run real episodes with a collection policy and append receipts."""
    from bramastra_lab.research.collection.runner import collect as run_collection
    from bramastra_lab.research.environments.oracles import failed_baseline_policy
    from bramastra_lab.research.environments.worlds import (
        InventoryWorld,
        ProgramLab,
        SwitchWorld,
    )
    from bramastra_lab.research.experience.ledger import ExperienceLedger
    from bramastra_lab.research.planning.planner import FixedCollectionPolicy

    constructors = {"switch-world": SwitchWorld, "inventory-world": InventoryWorld,
                    "program-lab": ProgramLab}
    names = [name.strip() for name in environments.split(",") if name.strip()]
    unknown = [name for name in names if name not in constructors]
    if unknown:
        raise CommandError(f"unknown environments {sorted(unknown)}; "
                           f"expected {sorted(constructors)}")
    if policy == "fixed":
        collection_policy = FixedCollectionPolicy(seed=seed)
        policy_identity = collection_policy.policy_identity

        def executor(view):
            return collection_policy.select(view)
    elif policy == "failed-baseline":
        executor = failed_baseline_policy
        policy_identity = "failed-baseline/v1"
    else:
        raise CommandError(
            "policy must be 'fixed' or 'failed-baseline'; the winning diagnostic "
            "policy is an environment check, not a collection policy")
    instances = [constructors[name](budget=budget, seed=seed + index)
                 for index, name in enumerate(names)]
    ledger = ExperienceLedger(ledger_path)
    report = run_collection(instances, ledger, executor,
                            policy_identity=policy_identity, episodes_per_environment=episodes)
    print(json.dumps({"status": "COLLECTED", "ledger": ledger_path, "identity":
                      ledger.identity(), **report}, indent=2))
    return 0


def resume_mod_restore_run(run_dir: str, config: BuildConfig):
    from bramastra_lab.research.runtime.resume import restore_run

    return restore_run(run_dir, config)


# -- infer / evaluate / package ------------------------------------------------

def infer(config_path: str, checkpoint: str, input_path: str, out_path: str | None,
          max_new_tokens: int | None) -> int:
    from bramastra_lab.research.models import IntegratedModel
    from bramastra_lab.research.runtime import checkpoint as ckpt
    from bramastra_lab.research.runtime.inference import generate_free_form

    config = _load_config(config_path)
    run_dir = _resolve_run_dir(checkpoint)
    payload, manifest = ckpt.load_checkpoint(run_dir, expect_config_identity=config.identity())
    model = IntegratedModel(config)
    model.load_state_dict(payload["model"])
    model.eval()
    try:
        with open(input_path, "r", encoding="utf-8") as handle:
            request = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise CommandError(f"cannot read inference request {input_path}: {exc}")
    if "prompt_events" in request:
        prompt = [SPECIAL_BOUNDARY]
        for role, content in request["prompt_events"]:
            prompt += encode_event(role, content)
    elif "text" in request:
        prompt = [SPECIAL_BOUNDARY] + encode_text(request["text"])
    else:
        raise CommandError("inference request needs 'prompt_events' or 'text'")
    report = generate_free_form(model, config, prompt, max_new_tokens=max_new_tokens)
    result = {
        "status": "INFERENCE", "checkpoint_id": manifest.checkpoint_id,
        "config_identity": config.identity(), "codec_identity": codec_identity(),
        "generation": report.to_dict(),
    }
    if out_path:
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2))
    return 0


def _resolve_run_dir(checkpoint: str) -> str:
    candidate = os.path.abspath(checkpoint)
    if os.path.exists(os.path.join(candidate, "checkpoints")):
        return candidate
    parent = os.path.dirname(candidate)
    if os.path.basename(parent).startswith("update-") \
            and os.path.isdir(os.path.join(os.path.dirname(parent), "checkpoints")):
        return os.path.dirname(parent)
    raise CommandError(f"cannot locate a run directory for checkpoint {checkpoint!r}")


def evaluate(checkpoint: str, config_path: str, data_dir: str, split: str,
             out_path: str) -> int:
    from bramastra_lab.research.evaluation.scoring import (
        RawOutcome,
        brier_score,
        complete_answer_metrics,
        family_metrics,
    )
    from bramastra_lab.research.evaluation.store import EvaluationStore
    from bramastra_lab.research.models import IntegratedModel
    from bramastra_lab.research.runtime import checkpoint as ckpt
    from bramastra_lab.research.runtime.inference import generate_free_form

    if split in ("sealed", "controller"):
        raise CommandError(
            f"split {split!r} is pool-separated; the CLI evaluates measurement/"
            "confirmation splits only")
    config = _load_config(config_path)
    run_dir = _resolve_run_dir(checkpoint)
    payload, manifest = ckpt.load_checkpoint(run_dir, expect_config_identity=config.identity())
    model = IntegratedModel(config)
    model.load_state_dict(payload["model"])
    model.eval()
    rows = _load_rows(data_dir, split)
    if not rows:
        raise CommandError(f"prepared data {data_dir} has no rows for split {split!r}")
    outcomes = []
    for index, record in enumerate(rows):
        if record["kind"] != "trajectory" or record.get("label") is None:
            continue
        prompt = list(record["tokens"][:record["prompt_length"]])
        report = generate_free_form(model, config, prompt, max_new_tokens=8)
        semantic = record["provenance"].get("task_semantic_id", "fixture")
        outcomes.append(RawOutcome(
            outcome_id=f"{manifest.checkpoint_id[:12]}-{index}",
            pool="measurement", split=split, family=semantic,
            task_semantic_id=semantic,
            prediction=report.answer, stopped_on_eos=report.stopped_on_eos,
            label=record["label"], cost=0.0,
            pair_group_id=record.get("pair_group_id"), role=None))
    if not outcomes:
        raise CommandError("no scoreable trajectory rows in this split")
    report = {
        "checkpoint_id": manifest.checkpoint_id,
        "config_identity": config.identity(),
        "split": split,
        "metrics": complete_answer_metrics(outcomes),
        "families": family_metrics(outcomes),
        "calibration": brier_score(outcomes),
        "note": "tiny-fixture scores are integration evidence, never capability claims",
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    store = EvaluationStore(os.path.dirname(os.path.abspath(out_path)))
    store.record(outcomes, pool="measurement", run_id=manifest.checkpoint_id)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2))
    return 0


def package(run_dir: str, out_path: str) -> int:
    from bramastra_lab.research.runtime import checkpoint as ckpt
    from bramastra_lab.research.runtime.resume import read_run_manifest

    run_manifest = read_run_manifest(run_dir)
    latest = ckpt.read_pointer(run_dir, ckpt.LATEST_POINTER)
    if latest is None:
        raise CommandError("run has no checkpoints to package")
    _, manifest = ckpt.load_checkpoint(run_dir)
    with open(os.path.join(run_dir, "config_used.json"), "r", encoding="utf-8") as handle:
        config_dict = json.load(handle)
    source_path = os.path.join(run_dir, "source.json")
    source_identity = "unavailable"
    if os.path.exists(source_path):
        with open(source_path, "r", encoding="utf-8") as handle:
            source_identity = json.load(handle).get("source_identity", "unavailable")
    ledger = _ledger().snapshot()
    package_manifest = {
        "schema": "bramastra-package/v1",
        "run_id": run_manifest["run_id"],
        "source_identity": source_identity,
        "config_identity": run_manifest["config_identity"],
        "tokenizer_identity": run_manifest["tokenizer_identity"],
        "codec_identity": run_manifest["codec_identity"],
        "data_identity": run_manifest["data_identity"],
        "latest_checkpoint": latest,
        "checkpoint_manifest": manifest.to_dict(),
        "feature_flags": {
            "features": config_dict.get("features", {}),
            "controller": config_dict.get("controller", {}),
            "logit_treatment": config_dict.get("training", {}).get("logit_treatment"),
            "pair_loss_weight": config_dict.get("training", {}).get("pair_loss_weight"),
        },
        "model_parameters": _profile_parameters(config_dict),
        "readiness": {
            "implementation": True,
            "cpu_smoke": ledger["cpu_optimizer_updates"] > 0,
            "gpu_smoke": ledger["gpu_optimizer_updates"] > 0,
            "real_data_supply": "DATA_NOT_READY: operator corpus not supplied; "
                                "tiny fixtures are not a corpus",
            "target_device": "DEVICE_UNVERIFIED (CPU smoke only; accelerator unverified)",
            "scientific_qualification": False,
        },
        "smoke_ledger": ledger,
        "excluded_from_git": ["model weights", "optimizer payloads", "large corpora"],
    }
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(package_manifest, handle, indent=2, sort_keys=True)
    print(json.dumps({"status": "PACKAGED", "out": out_path,
                      "checkpoint_id": manifest.checkpoint_id}, indent=2))
    return 0


def _profile_parameters(config_dict: dict[str, Any]) -> int:
    return BuildConfig.from_dict(config_dict).parameter_count()
