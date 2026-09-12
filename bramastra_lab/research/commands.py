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


# -- train / resume ------------------------------------------------------------

def train(config_path: str, data_dir: str, run_dir: str, max_updates: int | None,
          smoke: bool) -> int:
    from bramastra_lab.research.data.sampler import GroupSampler
    from bramastra_lab.research.learning.trainer import Trainer
    from bramastra_lab.research.models import IntegratedModel
    from bramastra_lab.research.runtime.resume import (
        RunContext,
        append_event,
        create_run,
        read_run_manifest,
    )

    config = _load_config(config_path)
    prepared = _read_prepared(data_dir, config)
    if max_updates is None or max_updates <= 0:
        raise CommandError("--max-updates is required (this build never trains unbounded)")
    ledger = _ledger()
    if smoke:
        if config.model.profile != "tiny":
            raise CommandError("--smoke requires the tiny profile; never disguise a "
                               "reduced run as the declared profile")
        try:
            ledger.check_can_run(device="cpu", updates=max_updates,
                                 seconds=30.0 + 12.0 * max_updates,
                                 reserve_for_resume=False)
        except SmokeBudgetExhausted as exc:
            raise CommandError(f"smoke budget refuses this run: {exc}")
    rows = _load_rows(data_dir, "training")
    if not rows:
        raise CommandError("prepared data has no training rows")
    sampler = GroupSampler(_sampler_units(rows), batch_size=min(4, len(rows)),
                           seed=config.training.seed)

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
    trainer = Trainer(config, IntegratedModel(config))
    ctx = RunContext(run_dir=run_dir, run_id=read_run_manifest(run_dir)["run_id"],
                     config=config, data_identity=prepared["identity"],
                     writer_token="train", trainer=trainer,
                     sampler_state=None, controller_state=None, replay_cursor=None)
    updates_done = 0
    last_checkpointed = -1
    for _ in range(max_updates):
        batch_refs = sampler.take_batch()
        batch = _collocate_records([ref.record for ref in batch_refs], config.model.max_seq)
        pair_input = _build_pair_input([ref.record for ref in batch_refs]) \
            if config.training.pair_loss_weight > 0 else None
        trainer.training_step(batch, pair_input=pair_input)
        updates_done += 1
        if updates_done % 4 == 0 and updates_done < max_updates:
            _checkpoint(trainer, run_dir, ctx)
            last_checkpointed = updates_done
    ctx.sampler_state = sampler.state().to_dict()
    milestone = "final" if last_checkpointed != updates_done else None
    manifest = _checkpoint(trainer, run_dir, ctx, milestone=milestone)
    elapsed = time.monotonic() - started
    if smoke:
        ledger.record(device="cpu", updates=updates_done, seconds=elapsed,
                      what=f"train smoke {os.path.basename(run_dir)}",
                      evidence=manifest.checkpoint_id)
    append_event(run_dir, {"event": "train_complete", "updates": updates_done,
                           "seconds": round(elapsed, 3)})
    print(json.dumps({
        "status": "TRAINED", "run_dir": run_dir, "updates": updates_done,
        "seconds": round(elapsed, 3), "checkpoint_id": manifest.checkpoint_id,
        "counters": trainer.counters.to_dict(),
    }, indent=2))
    return 0


def resume(run_dir: str, max_updates: int | None, expect_parent: str | None) -> int:
    from bramastra_lab.research.data.sampler import GroupSampler, SamplerState

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
    trainer = ctx.trainer
    updates_before = trainer.counters.optimizer_updates
    updates_done = 0
    last_checkpointed = -1
    for _ in range(max_updates):
        batch_refs = sampler.take_batch()
        batch = _collocate_records([ref.record for ref in batch_refs], config.model.max_seq)
        pair_input = _build_pair_input([ref.record for ref in batch_refs]) \
            if config.training.pair_loss_weight > 0 else None
        trainer.training_step(batch, pair_input=pair_input)
        updates_done += 1
        if updates_done % 4 == 0 and updates_done < max_updates:
            _checkpoint(trainer, run_dir, ctx)
            last_checkpointed = updates_done
    ctx.sampler_state = sampler.state().to_dict()
    milestone = "resumed" if last_checkpointed != updates_done else None
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
    }, indent=2))
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
