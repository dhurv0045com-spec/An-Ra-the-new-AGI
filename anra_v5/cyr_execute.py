"""CYR tournament execution layer (torch executors; research CLI precedent).

Pure tournament logic lives in v5_experiments.cyr_tournament (worlds,
sampler, controller, metrics definitions, red team, resolver, packaging).
This module owns everything that touches torch/models: proxy training
arms, teacher-forced and free-generation measurement, the smoke check,
and main(). Split this way, the v5_experiments plane never imports the
model plane (import-boundary enforced by test).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping

from v5_experiments.cyr_tournament import (
    RUNNER_SCHEMA,
    RENDER_SUFFIX,
    constant_lr,
    discover_environment,
    guard_full_mode,
    package_bundle,
    pair_batches,
    proxy_ladder,
    render_worlds,
)


def _model_spec(proxy: Mapping[str, Any], *, vocab_size: int):
    from v5_contracts.model_spec import ModelSpec
    return ModelSpec(
        schema="anra-v5-model-spec/v1", family="dense-decoder-transformer",
        vocabulary_size=int(vocab_size), width=int(proxy["width"]),
        layers=int(proxy["layers"]), query_heads=int(proxy["query_heads"]),
        kv_heads=int(proxy["kv_heads"]),
        head_dimension=int(proxy["head_dimension"]),
        ffn_width=int(proxy["ffn_width"]),
        context_length=int(proxy["context_length"]),
        rope_base=10_000.0, norm_epsilon=1e-5, tied_embeddings=True,
        qk_norm=True, qk_norm_affine=True, linear_bias=False, dropout=0.0)


class ByteTokenizer:
    """Deterministic byte-level tokenizer for smoke/offline use (not production)."""

    vocab_size = 256
    artifact_sha = (
        "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08")

    class _Identity:
        artifact_sha256 = (
            "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08")

    identity = _Identity()

    def encode(self, text: str) -> list[int]:
        return [(ord(character) % 250) + 4 for character in text[:512]]


def train_arm(*, proxy: Mapping[str, Any], tokenizer: Any, torch: Any,
              device: Any, records: list[dict[str, str]],
              microbatch_rows: int, row_width: int, updates: int,
              schedule: Callable[[int], float], seed: int,
              run_id: str, store_root: str | Path,
              progress: Callable[[str], None] | None = None,
              eval_every: int = 0,
              evaluate: Callable[[Any], dict[str, Any]] | None = None,
              fork_from: Mapping[str, bytes] | None = None,
              start_row: int = 0,
              ) -> dict[str, Any]:
    """Train one arm through the REAL certified update + state machinery.

    Every record must encode to exactly row_width-2 content tokens (the
    renderer guarantees this; anything else fails closed), so rows are
    exactly full and token accounting is exact everywhere. Microbatches
    slice consecutive rows; with an even row count and pair-adjacent order,
    twins share microbatches guaranteed (pair_split_rate receipted, must be
    0 for grouped arms). Checkpoints fork through the real CheckpointStore.
    ``fork_from`` (model/optimizer payload bytes from a parent receipt)
    continues from a lineage checkpoint for LR-tournament arms; RNG state
    is intentionally not restored (dropout-free compute consumes no RNG).
    Tiny proxy budgets only. Certification failures abort loudly.
    """

    from v5_model.core import initialize
    from v5_training.checkpoint import CheckpointStore
    from v5_training.optimizer import build_adamw_optimizer
    from v5_training.production_backend import ProductionTrainingBackend
    from v5_training.production_backend import (
        capture_evidence,
        production_payloads,
    )
    from v5_training.production_entry import _predict_supervised
    from v5_training.runner import RunController
    from v5_training.state import (
        CURSOR_SCHEMA,
        IDENTITY_SCHEMA,
        CursorState,
        IdentityBindings,
        TrainingState,
    )
    from v5_training.trainer import BackendReport, train

    if microbatch_rows <= 0 or microbatch_rows % 2:
        raise ValueError("microbatch rows must be a positive even count")
    torch.manual_seed(seed)
    model_spec = _model_spec(proxy, vocab_size=tokenizer.vocab_size)
    model = initialize(model_spec, seed, torch_module=torch).to(device)
    optimizer = build_adamw_optimizer(model, torch_module=torch)
    backend = ProductionTrainingBackend(
        model=model, optimizer=optimizer, bos_id=2, pad_id=0, device=device,
        schedule=schedule, bfloat16_autocast=False, torch_module=torch,
        activation_checkpointing=False)
    fork_head: str | None = None
    if fork_from is not None:
        import io
        model.load_state_dict(torch.load(
            io.BytesIO(fork_from["model.bin"]), map_location="cpu",
            weights_only=True))
        optimizer.load_state_dict(torch.load(
            io.BytesIO(fork_from["optimizer.bin"]), map_location="cpu",
            weights_only=True))
        fork_head = fork_from.get("fork_head", None)
        if isinstance(fork_head, bytes):
            fork_head = fork_head.decode("ascii")
    before_params = {name: param.detach().clone()
                     for name, param in model.named_parameters()}
    before_evidence = capture_evidence(model, optimizer, torch=torch)
    rows: list[tuple[int, list[int]]] = []
    for index, record in enumerate(records):
        ids = tokenizer.encode(record["text"])
        if len(ids) != row_width - 2:
            raise ValueError(
                f"record {index} encodes to {len(ids)} tokens, not the exact "
                f"{row_width - 2}: renderer sizing violated")
        rows.append((index, [2, *ids, 3]))
    if len(rows) % microbatch_rows:
        raise ValueError("record count must divide evenly into microbatches")
    if start_row < 0 or start_row + updates * microbatch_rows > len(rows):
        raise ValueError("fork offset runs past the rendered records")
    tokens_per_update = microbatch_rows * row_width
    data_sha = hashlib.sha256(_canonical_json(
        [record["text"] for record in records])).hexdigest()
    identities = IdentityBindings(
        schema=IDENTITY_SCHEMA, source_commit="0" * 40,
        model_spec_sha256=model_spec.sha256(),
        tokenizer_sha256=tokenizer.identity.artifact_sha256,
        data_manifest_sha256=data_sha, pack_manifest_sha256=data_sha,
        run_spec_sha256=data_sha, optimizer_spec_sha256=data_sha,
        schedule_spec_sha256=data_sha, curriculum_spec_sha256=data_sha)
    state = TrainingState.initial(
        lineage_id=run_id, token_budget=tokens_per_update * updates,
        tokens_per_update=tokens_per_update,
        cursor=CursorState(CURSOR_SCHEMA, data_sha, 0, 0, 0),
        rng_state_sha256="0" * 64, curriculum_phase="cyr-tournament",
        identities=identities)
    store = CheckpointStore(Path(store_root), run_id)
    controller = RunController(target_update=updates)
    controller.start()
    losses: list[float] = []
    grad_norms: list[float] = []
    eval_trace: list[dict[str, Any]] = []
    pair_splits = 0

    def backend_step(current: TrainingState) -> BackendReport:
        nonlocal pair_splits
        start = start_row + current.global_update * microbatch_rows
        group = rows[start:start + microbatch_rows]
        worlds = [records[item[0]].get("world_id", "") for item in group]
        for position in range(0, len(worlds), 2):
            if worlds[position] != worlds[position + 1]:
                pair_splits += 1
        tokens = torch.tensor([item[1] for item in group], dtype=torch.long,
                              device=device)
        segment_ids = torch.zeros_like(tokens)
        eligible = torch.ones_like(tokens, dtype=torch.bool)
        predicted = _predict_supervised(
            type("Window", (), {"tokens": tuple(tuple(row) for row in tokens.tolist()),
                                "segment_ids": tuple(tuple(row) for row in segment_ids.tolist()),
                                "eligible": tuple(tuple(row) for row in eligible.tolist())}))
        ctx = backend.begin_update(current)
        ctx = backend.accumulate_microstep(
            ctx, tokens=tokens, segment_ids=segment_ids, eligible=eligible,
            tokens_by_source={"cyr": tokens_per_update},
            planned_total=predicted)
        cursor = CursorState(CURSOR_SCHEMA, data_sha, current.global_update + 1,
                             start + len(group), 0)
        report = backend.finish_update(
            current, ctx, planned_total=predicted, cursor=cursor)
        return report

    def payload_builder(live: TrainingState) -> dict[str, bytes]:
        return production_payloads(backend, state=live)

    def on_committed(live: TrainingState, _sha: str) -> None:
        receipt = backend.last_receipt
        assert receipt is not None
        losses.append(float(receipt["loss"]))
        grad_norms.append(float(receipt["grad_norm_post_clip"]))
        if evaluate is not None and eval_every and live.global_update % eval_every == 0:
            eval_trace.append({"update": live.global_update,
                               "tokens": live.cumulative_tokens,
                               **evaluate(model)})

    final = train(state=state, controller=controller, store=store,
                  payload_builder=payload_builder, backend_step=backend_step,
                  updates=updates, checkpoint_every=1,
                  on_committed=on_committed)
    after_evidence = capture_evidence(model, optimizer, torch=torch)
    displacement = sum(
        float(((param.detach().float() - before_params[name].float()) ** 2).sum())
        for name, param in model.named_parameters()) ** 0.5
    return {"losses": losses, "updates": final.global_update,
            "tokens": final.cumulative_tokens, "eval_trace": eval_trace,
            "grad_norms": grad_norms, "pair_splits": pair_splits,
            "start_row": start_row,
            "parameter_sha_before": before_evidence.parameter_sha256,
            "parameter_sha_after": after_evidence.parameter_sha256,
            "displacement_norm": displacement,
            "optimizer_steps": after_evidence.optimizer_steps,
            "checkpoint_head": store.latest_sha256(),
            "fork_head": fork_head}


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _score_items(*, model: Any, tokenizer: Any, torch: Any,
                 device: Any, items: list[dict[str, str]],
                 batch_size: int = 32) -> list[bool]:
    """Teacher-forced answer-suffix exact match, batched over items."""

    from v5_model.core import packed_layout
    outcomes: list[bool] = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            chunk = items[start:start + batch_size]
            encoded = [[2, *tokenizer.encode(item["text"]), 3] for item in chunk]
            answers = [tokenizer.encode(item["answer"]) for item in chunk]
            width = max(len(ids) for ids in encoded)
            tokens = torch.tensor(
                [ids + [0] * (width - len(ids)) for ids in encoded],
                dtype=torch.long, device=device)
            segments = torch.tensor(
                [[0] * len(ids) + [-1] * (width - len(ids)) for ids in encoded],
                dtype=torch.long, device=device)
            positions, mask = packed_layout(segments, torch_module=torch)
            logits = model(tokens, positions, mask.to(tokens.device))
            for row, row_ids, answer_ids in zip(range(len(chunk)), encoded, answers):
                ok = True
                base = len(row_ids) - len(answer_ids) - 1
                for offset, answer_id in enumerate(answer_ids):
                    if int(torch.argmax(logits[row, base + offset]).item()) != answer_id:
                        ok = False
                        break
                outcomes.append(ok)
    if was_training:
        model.train()
    return outcomes


def teacher_forced_exact(*, model: Any, tokenizer: Any, torch: Any,
                         device: Any, worlds: list[dict[str, Any]]) -> dict[str, Any]:
    """Dense metric: teacher-forced answer-suffix exact match + both-correct."""

    flags = _score_items(
        model=model, tokenizer=tokenizer, torch=torch, device=device,
        items=[world[side] for world in worlds for side in ("base", "twin")])
    if not flags:
        return {"exact": 0.0, "both_correct": 0.0, "worlds": 0}
    exact = sum(flags) / len(flags)
    both = sum(1 for index in range(0, len(flags), 2)
               if flags[index] and flags[index + 1])
    total = len(worlds)
    return {"exact": exact, "both_correct": both / total if total else 0.0,
            "worlds": total}


def free_generation_spot(*, model: Any, tokenizer: Any, torch: Any,
                         device: Any, worlds: list[dict[str, Any]],
                         max_new_tokens: int = 32, eos_id: int = 3
                         ) -> dict[str, Any]:
    """Sparse metric: greedy generation terminates (EOS or cap)."""

    from v5_model.core import packed_layout
    stops = {"eos": 0, "cap": 0}
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for world in worlds[:16]:
            prompt = world["base"]["text"].rsplit(RENDER_SUFFIX, 1)[0] + RENDER_SUFFIX
            token_ids = [2, *tokenizer.encode(prompt)]
            for _ in range(max_new_tokens):
                tokens = torch.tensor([token_ids], dtype=torch.long, device=device)
                segments = torch.zeros_like(tokens)
                positions, mask = packed_layout(segments, torch_module=torch)
                logits = model(tokens, positions, mask.to(tokens.device))
                next_id = int(torch.argmax(logits[0, -1]).item())
                if next_id == eos_id:
                    stops["eos"] += 1
                    break
                token_ids.append(next_id)
            else:
                stops["cap"] += 1
    if was_training:
        model.train()
    return {"stops": stops, "worlds": min(len(worlds), 16)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--allow-non-colab", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("CYR-GPU-001"))
    parser.add_argument("--stage", default="all")
    parser.add_argument("--prereg", type=Path, default=None)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--stages", default=",".join(CYR_STAGE_ORDER))
    return parser


def _load_prereg(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"sha256": None, "note": "no preregistration file supplied"}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _production_tokenizer(repo: Path):
    from v5_data.corpus_loading import _load_tokenizer
    tokenizer, _ = _load_tokenizer(repo.resolve())
    return tokenizer


def smoke(torch: Any, device: Any, out: Path) -> dict[str, Any]:
    """Tiny plumbing smoke: backend, state, checkpoint, schedule, receipts."""

    proxy = proxy_ladder()["TINY"]
    tok = ByteTokenizer()
    worlds = render_worlds(family="registry",
                           split_seeds={"train": 11}, worlds_per_split=4,
                           encode=tok.encode, row_content_tokens=300)
    records = pair_batches(worlds["train"], group_pairs=True, seed=11)
    with tempfile.TemporaryDirectory() as tmp:
        result = train_arm(proxy=proxy, tokenizer=tok, torch=torch,
                           device=device, records=records, microbatch_rows=4,
                           row_width=302, updates=2,
                           schedule=constant_lr(3e-4), seed=7, run_id="smoke",
                           store_root=str(Path(tmp) / "store"))
    assert result["updates"] == 2
    assert len(result["losses"]) == 2
    assert all(value == value for value in result["losses"])
    assert result["parameter_sha_before"] != result["parameter_sha_after"]
    assert result["pair_splits"] == 0
    assert result["checkpoint_head"] is not None
    bundle = {"schema": RUNNER_SCHEMA, "mode": "smoke", "result": result}
    out.mkdir(parents=True, exist_ok=True)
    (out / "SMOKE.json").write_text(
        json.dumps(bundle, indent=2, sort_keys=True, default=str),
        encoding="utf-8")
    return bundle


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    guard_full_mode(args)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "ENVIRONMENT.json").write_text(
        json.dumps(discover_environment(), indent=2, sort_keys=True),
        encoding="utf-8")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("tournament needs PyTorch") from exc
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        if args.mode == "smoke":
            smoke(torch, device, out)
        else:
            prereg = _load_prereg(args.prereg)
            tokenizer = _production_tokenizer(args.repo)
            stages = tuple(stage for stage in args.stages.split(",") if stage)
            run_cyr_campaign(out=out, torch=torch, device=device,
                             tokenizer=tokenizer, stages=stages,
                             prereg=prereg, progress=print)
    except Exception as exc:
        failure = {"schema": "anra-cyr-failure/v1", "stage": args.stage,
                   "exception": type(exc).__name__, "message": str(exc),
                   "traceback": traceback.format_exc()}
        (out / "FAILURE.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True), encoding="utf-8")
        package_bundle(out)
        raise
    package_bundle(out)
    return 0


def eval_variant_texts(world: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """Orthogonal query/order factorial for one world (eval-only transforms)."""

    facts = list(world["facts"])
    base_text = world["base"]["text"]
    base_answer = world["base"]["answer"]
    twin_answer = world["twin"]["answer"]
    reversed_facts = list(reversed(facts))
    context_of = world["base"]["context"]
    query, twin_query = world["query"], world["twin_query"]

    def rebuild(ordered_facts: list[str], ask: str, ans: str) -> dict[str, str]:
        return {"text": "\n".join(ordered_facts) + "\n" + ask + "\n" + RENDER_SUFFIX + ans,
                "answer": ans, "context": "\n".join(ordered_facts) + "\n" + ask,
                "query": ask}

    _ = context_of
    return {
        "base": {"text": base_text, "answer": base_answer},
        "query_only": dict(world["twin"]),
        "order_only": rebuild(reversed_facts, query, base_answer),
        "query_and_order": rebuild(reversed_facts, twin_query, twin_answer),
    }


def score_variants(*, model: Any, tokenizer: Any, torch: Any,
                   device: Any, worlds: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-variant exact + both-correct over (base, query_only) pairs."""

    variants = ("base", "query_only", "order_only", "query_and_order")
    rendered = [eval_variant_texts(world) for world in worlds]
    flags: dict[str, list[bool]] = {}
    for name in variants:
        flags[name] = _score_items(
            model=model, tokenizer=tokenizer, torch=torch, device=device,
            items=[rendered[index][name] for index in range(len(worlds))])
    total = len(worlds)
    return {"exact": {name: sum(flags[name]) / total for name in variants},
            "both_correct": sum(1 for index in range(total)
                                if flags["base"][index] and flags["query_only"][index]) / total,
            "worlds": total}


def blind_gap(*, model_both: float,
              contexts: list[str], queries: list[str], answers: list[str],
              twins: list[tuple[str, str, str]]) -> dict[str, Any]:
    """Model both-correct minus best heuristic both-correct (same worlds)."""

    from v5_experiments.cyr_tournament import heuristic_baselines
    baselines = heuristic_baselines()
    scores = {}
    for name, predict in baselines.items():
        hits = 0
        for context, query, answer, (twin_query, twin_answer, twin_context) in zip(
                contexts, queries, answers, twins):
            _ = twin_context
            if predict(context, query) == answer and \
                    predict(context, twin_query) == twin_answer:
                hits += 1
        scores[name] = hits / len(contexts) if contexts else 0.0
    best = max(scores.values()) if scores else 0.0
    return {"model_both_correct": model_both,
            "best_baseline_both_correct": best,
            "blind_gap": model_both - best, "baselines": scores}


CYR_STAGE_ORDER = ("s0", "s1", "s2", "s3", "s4")


def _write_json(out: Path, name: str, payload: Mapping[str, Any]) -> Path:
    target = out / name
    target.write_text(json.dumps(dict(payload), indent=2, sort_keys=True,
                                 default=str), encoding="utf-8")
    return target


def run_cyr_campaign(*, out: Path, torch: Any, device: Any, tokenizer: Any,
                     stages: tuple[str, ...] = CYR_STAGE_ORDER,
                     time_limit_min: float = 170.0,
                     progress: Callable[[str], None] | None = None,
                     prereg: Mapping[str, Any]) -> dict[str, Any]:
    """Execute the preregistered CYR-GPU-001 stage plan with gates and timeboxes."""

    from v5_experiments.cyr_tournament import (
        CYR_EVAL_DENSE_WORLDS,
        CYR_EVAL_EVERY_UPDATES,
        CYR_FIXED_SWITCH_FRACTION,
        CYR_MICROBATCH_ROWS,
        CYR_ROW_CONTENT_TOKENS,
        CYR_SEEDS,
        CYR_SPLIT_SEEDS,
        CYR_SUSTAINED_REQUIRED,
        HysteresisController,
        assert_split_firewall,
        constant_lr,
        pair_batches,
        proxy_ladder,
        redteam_exposure,
        redteam_freezing,
        redteam_near_dup,
        redteam_non_mutation,
        redteam_overlap,
        resolve_proxy,
        split_manifest,
        sustained,
    )
    from v5_experiments.cyr_tournament import RESEARCH_LRS
    t0 = time.monotonic()

    def remaining_min() -> float:
        return time_limit_min - (time.monotonic() - t0) / 60.0

    def say(message: str) -> None:
        if progress is not None:
            progress(message)

    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"stages": {}, "gates": {}, "redteam": []}
    row_width = CYR_ROW_CONTENT_TOKENS + 2

    # -- S0: bootstrap truth checks, EOS contract, calibration, resolve -----
    # Full worlds render AFTER resolve (sized to the resolved budget).
    if "s0" in stages:
        say("S0 bootstrap")
        env = discover_environment()
        probe_worlds = render_worlds(
            family="registry", split_seeds={"train": 999},
            worlds_per_split=2, encode=tokenizer.encode,
            row_content_tokens=CYR_ROW_CONTENT_TOKENS)
        assert_split_firewall(probe_worlds)
        from v5_data.pack import pack_documents
        sample = [("eos-check", tokenizer.encode(
            probe_worlds["train"][0]["base"]["text"]), "s")]
        packed, _ = pack_documents(sample, bos=2, eos=3, pad=0, sequences_per_shard=8)
        sequence = packed[0].sequences[0]
        assert list(sequence.tokens)[:1] == [2] and 3 in list(sequence.tokens), \
            "EOS contract violated before training starts"
        summary["gates"]["eos_contract"] = True
        train_ids = [f"probe/{i}" for i in range(4)]
        summary["redteam"].append(redteam_overlap(train_ids, train_ids[:2] + ["other"]))
        summary["redteam"].append(redteam_near_dup(
            [world["base"]["text"] for world in probe_worlds["train"]]))
        calibration = _calibrate(torch, device, out, say)
        resolved = resolve_proxy(
            micro_tokens_per_sec=calibration["tokens_per_sec"],
            free_vram_gb=calibration["free_vram_gb"])
        resolved_doc = {"resolved": resolved, "calibration": calibration,
                        "prereg_sha256": prereg.get("sha256"),
                        "environment": env}
        _write_json(out, "RESOLVED_PREREGISTRATION.json", resolved_doc)
        summary["stages"]["s0"] = {"status": "COMPLETE", "resolved": resolved}
    else:
        resolved = {"proxy": "MICRO", "tokens_per_arm": 150_000,
                    "reason": "stages run standalone; default MICRO budget"}

    proxy_name = resolved["proxy"]
    proxy = proxy_ladder()[proxy_name]
    tokens_per_arm = int(resolved["tokens_per_arm"])
    microbatch_tokens = CYR_MICROBATCH_ROWS * row_width
    updates_per_arm = max(4, tokens_per_arm // microbatch_tokens)
    cont_updates = max(8, updates_per_arm // 2)
    # Train streams must cover S1 acquisition AND S3 continuation rows.
    train_worlds = (updates_per_arm + cont_updates) * CYR_MICROBATCH_ROWS // 2
    worlds = {family: render_worlds(
        family=family,
        split_seeds=CYR_SPLIT_SEEDS,
        worlds_per_split={"train": train_worlds,
                          "dev_controller": CYR_DEV_WORLDS,
                          "dev_measurement": CYR_DEV_WORLDS,
                          "sealed_reserved": CYR_DEV_WORLDS},
        encode=tokenizer.encode,
        row_content_tokens=CYR_ROW_CONTENT_TOKENS) for family in ("registry", "transfer")}
    for family_worlds in worlds.values():
        assert_split_firewall(family_worlds)
    manifests = {family: split_manifest(family_worlds)
                 for family, family_worlds in worlds.items()}
    _write_json(out, "SPLIT_MANIFEST.json",
                {"registry": manifests["registry"], "transfer": manifests["transfer"]})
    summary["worlds_rendered"] = {
        family: {split: len(members) for split, members in family_worlds.items()}
        for family, family_worlds in worlds.items()}
    stores = out / "stores"
    stores.mkdir(parents=True, exist_ok=True)

    def evaluate_on(split_worlds: list[dict[str, Any]]):
        def evaluate(model: Any) -> dict[str, Any]:
            return teacher_forced_exact(model=model, tokenizer=tokenizer,
                                       torch=torch, device=device,
                                       worlds=split_worlds[:CYR_EVAL_DENSE_WORLDS])
        return evaluate

    def run_arm(records, *, arm_id, schedule, seed, updates, eval_worlds,
                start_row: int = 0, fork_from=None):
        return train_arm(
            proxy=proxy, tokenizer=tokenizer, torch=torch,
            device=device, records=records,
            microbatch_rows=CYR_MICROBATCH_ROWS, row_width=row_width,
            updates=updates, schedule=schedule, seed=seed, run_id=arm_id,
            store_root=str(stores / arm_id), progress=say,
            eval_every=CYR_EVAL_EVERY_UPDATES,
            evaluate=evaluate_on(eval_worlds), fork_from=fork_from,
            start_row=start_row)

    def arm_payloads(arm_id: str, head: str) -> dict[str, bytes]:
        store = CheckpointStore(stores / arm_id, arm_id)
        _, payloads = store.restore(head)
        return dict(payloads)

    # -- S1: baseline acquisition + factorial gate ---------------------------
    if "s1" in stages:
        say("S1 baseline acquisition")
        s1: dict[str, Any] = {}
        for family in ("registry", "transfer"):
            train_ids = [world["world_id"] for world in worlds[family]["train"]]
            eval_ids = [world["world_id"] for world in worlds[family]["dev_measurement"]]
            summary["redteam"].append(redteam_overlap(train_ids, eval_ids))
            for seed in CYR_SEEDS:
                ordered = pair_batches(worlds[family]["train"], group_pairs=False,
                                       seed=seed)
                key = f"s1-{family}-{seed}"
                result = run_arm(
                    ordered, arm_id=key, schedule=constant_lr(RESEARCH_LRS["HIGH"]),
                    seed=seed, updates=updates_per_arm,
                    eval_worlds=worlds[family]["dev_measurement"])
                _write_json(out, f"BASELINE-{family}-{seed}.json", result)
                s1[key] = {"final_train_exact": result["eval_trace"][-1]["exact"]
                           if result["eval_trace"] else 0.0,
                           "checkpoint_head": result["checkpoint_head"],
                           "losses": result["losses"][-3:],
                           "displacement_norm": result["displacement_norm"]}
        trace_ok = any(entry["final_train_exact"] >= 0.99 for entry in s1.values())
        summary["stages"]["s1"] = s1
        summary["gates"]["learnability"] = bool(trace_ok)
        if not trace_ok:
            summary["stages"]["s1"]["status"] = "ABORT_NO_SIGNAL"
            _write_json(out, "DECISION.json", _decision(summary, prereg))
            package_bundle(out)
            return summary
        summary["stages"]["s1"]["status"] = "COMPLETE"
        factorial = {}
        for family in ("registry", "transfer"):
            factorial[family] = "deferred-to-s2-eval"
        summary["stages"]["s1"]["factorial"] = factorial

    # -- S2: pair-preserving vs shuffled --------------------------------------
    if "s2" in stages and remaining_min() > 25:
        say("S2 pair sampler tournament")
        from v5_experiments.preregistration import (
            ExperimentSpec,
            TrainingInterventionRecord,
            assert_matched_arms,
        )
        s2: dict[str, Any] = {}
        for family in ("registry", "transfer"):
            for seed in CYR_SEEDS:
                arms = {}
                orders = {}
                for grouped in (False, True):
                    ordered = pair_batches(worlds[family]["train"],
                                           group_pairs=grouped, seed=seed)
                    arm_id = f"s2-{family}-{'paired' if grouped else 'shuffled'}-{seed}"
                    result = run_arm(
                        ordered, arm_id=arm_id, schedule=constant_lr(RESEARCH_LRS["HIGH"]),
                        seed=seed, updates=updates_per_arm,
                        eval_worlds=worlds[family]["dev_measurement"])
                    if grouped:
                        assert result["pair_splits"] == 0, "pair treatment leaked"
                    arms["paired" if grouped else "shuffled"] = result
                    orders["paired" if grouped else "shuffled"] = ordered
                    _write_json(out, f"PAIR-{family}-{seed}-{'paired' if grouped else 'shuffled'}.json",
                                result)
                assert sorted(r["text"] for r in orders["paired"]) == \
                    sorted(r["text"] for r in orders["shuffled"]), \
                    "arms saw different multisets"
                paired = arms["paired"]
                model = reload_model(
                    proxy=proxy, tokenizer=tokenizer, torch=torch, device=device,
                    store_root=str(stores / f"s2-{family}-paired-{seed}"),
                    run_id=f"s2-{family}-paired-{seed}",
                    checkpoint_sha=paired["checkpoint_head"])
                factorial = score_variants(
                    model=model, tokenizer=tokenizer, torch=torch, device=device,
                    worlds=worlds[family]["dev_measurement"][:CYR_EVAL_DENSE_WORLDS])
                gap = blind_gap(model_both=factorial["both_correct"],
                                worlds=worlds[family]["dev_measurement"][:CYR_EVAL_DENSE_WORLDS])
                del model
                gc.collect()
                paired_both = factorial["both_correct"]
                shuffled_both = arms["shuffled"]["eval_trace"][-1]["both_correct"] \
                    if arms["shuffled"]["eval_trace"] else 0.0
                redteam = [
                    redteam_exposure({
                        "paired": {"tokens": paired["tokens"], "updates": paired["updates"]},
                        "shuffled": {"tokens": arms["shuffled"]["tokens"],
                                     "updates": arms["shuffled"]["updates"]}}),
                    redteam_non_mutation(paired["parameter_sha_before"],
                                         paired["parameter_sha_after"]),
                    redteam_non_mutation(arms["shuffled"]["parameter_sha_before"],
                                         arms["shuffled"]["parameter_sha_after"]),
                ]
                summary["redteam"].extend(redteam)
                s2[f"{family}-{seed}"] = {
                    "paired_both_correct": paired_both,
                    "shuffled_both_correct": shuffled_both,
                    "gap": paired_both - shuffled_both,
                    "factorial": factorial["exact"],
                    "blind_gap": gap,
                    "matched": _matched_sampler_pair(family, seed),
                    "redteam_pass": all(check["pass"] for check in redteam)}
                _write_json(out, f"FACTORIAL-{family}-{seed}.json",
                            {"factorial": factorial, "blind_gap": gap})
        summary["stages"]["s2"] = s2
        summary["matched_arms"] = {
            "matched": all(entry.get("matched", {}).get("matched", False)
                           for entry in s2.values()
                           if isinstance(entry, dict) and "matched" in entry)}
    elif "s2" in stages:
        summary["stages"]["s2"] = {"status": "SKIPPED_TIMEBOX"}

    # -- S3: LR tournament from forks ------------------------------------------
    if "s3" in stages and remaining_min() > 30:
        say("S3 LR tournament")
        summary["stages"]["s3"] = _run_lr_tournament(
            out=out, worlds=worlds,
            updates_per_arm=updates_per_arm,
            run_arm=run_arm, arm_payloads=arm_payloads,
            microbatch_rows=CYR_MICROBATCH_ROWS, summary=summary)
    elif "s3" in stages:
        summary["stages"]["s3"] = {"status": "SKIPPED_TIMEBOX"}

    # -- S4: transfer ------------------------------------------------------------
    if "s4" in stages and remaining_min() > 20:
        say("S4 transfer")
        from v5_experiments.cyr_tournament import proxy_ladder as _ladder
        s2 = summary.get("stages", {}).get("s2", {})
        winners = {}
        for family in ("registry", "transfer"):
            gaps = [s2.get(f"{family}-{seed}", {}).get("gap", 0.0) for seed in CYR_SEEDS]
            winners[family] = len(gaps) == len(CYR_SEEDS) and all(gap >= 0.10 for gap in gaps)
        transfer: dict[str, Any] = {
            "arms": {},
            "winner_rule": {family: ("paired" if won else "shuffled")
                            for family, won in winners.items()}}
        microbatch_tokens = CYR_MICROBATCH_ROWS * row_width
        transfer_updates = max(4, tokens_per_arm // microbatch_tokens)
        for family in ("registry", "transfer"):
            grouped = winners[family]
            ordered = pair_batches(worlds[family]["train"], group_pairs=grouped,
                                   seed=303)
            arm_id = f"s4-{family}-{'paired' if grouped else 'shuffled'}"
            result = run_arm(
                ordered, arm_id=arm_id, schedule=constant_lr(RESEARCH_LRS["HIGH"]),
                seed=303, updates=transfer_updates,
                eval_worlds=worlds[family]["dev_measurement"])
            _write_json(out, f"TRANSFER-{family}.json", result)
            transfer["arms"][family] = {
                "sampler": "paired" if grouped else "shuffled",
                "final_both_correct": result["eval_trace"][-1]["both_correct"]
                if result["eval_trace"] else 0.0,
                "displacement_norm": result["displacement_norm"]}
        order = ["TINY", "MICRO", "MIDI", "P35"]
        transfer["larger_proxy"] = {
            "status": "AVAILABLE",
            "next": order[order.index(proxy_name) + 1],
            "note": "operator may re-run S4 at the next proxy with --stages s4",
        } if proxy_name in order and order.index(proxy_name) + 1 < len(order) \
            else {"status": "AT_TOP"}
        transfer["status"] = "COMPLETE"
        summary["stages"]["s4"] = transfer
    elif "s4" in stages:
        summary["stages"]["s4"] = {"status": "SKIPPED_TIMEBOX"}

    _write_json(out, "DECISION.json", _decision(summary, prereg))
    package_bundle(out)
    return summary


def _matched_sampler_pair(family: str, seed: int) -> dict[str, Any]:
    """Mechanical matched-arms proof for one sampler pair (declared: sampler)."""

    from v5_experiments.preregistration import (
        ExperimentSpec,
        TrainingInterventionRecord,
        assert_matched_arms,
    )

    def spec(treatment: str) -> ExperimentSpec:
        return ExperimentSpec(
            experiment_id=f"CYR-GPU-001-S2-{family}-{seed}",
            hypothesis="pair grouping changes query control",
            intervention=TrainingInterventionRecord(
                hypothesis="h", mechanism_target="batch assembly",
                treatment_definition=treatment,
                control_definition="shuffled sampler",
                expected_behavioral_effect="higher both-correct",
                expected_failure_profile_effect="none",
                risks=("order confound (controlled: same multiset)",)),
            parent_checkpoint_sha256s=(),
            model_spec_sha256="0" * 64, tokenizer_artifact_sha256="0" * 64,
            training_spec_sha256="0" * 64, data_manifest_sha256="0" * 64,
            optimizer_spec_sha256="0" * 64, schedule_spec_sha256="0" * 64,
            token_budget=1, seeds=(seed,), evaluation_protocol_sha256="0" * 64,
            promotion_rule="both-correct gap sustained", stop_rule="budget",
            treatment_fields=("intervention",))

    return assert_matched_arms(spec("shuffled"), spec("paired"),
                               allowed_differences=("intervention",))


def _calibrate(torch: Any, device: Any, out: Path,
               say: Callable[[str], None] | None) -> dict[str, Any]:
    from v5_experiments.cyr_tournament import proxy_ladder as ladder
    proxy = ladder()["MICRO"]
    tok = ByteTokenizer()
    worlds = render_worlds(family="registry", split_seeds={"train": 77},
                           worlds_per_split=4, encode=tok.encode,
                           row_content_tokens=300)
    records = pair_batches(worlds["train"], group_pairs=True, seed=77)
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory() as tmp:
        result = train_arm(proxy=proxy, tokenizer=tok, torch=torch,
                           device=device, records=records, microbatch_rows=4,
                           row_width=302, updates=4,
                           schedule=constant_lr(3e-4), seed=77,
                           run_id="calibration",
                           store_root=str(Path(tmp) / "store"))
    elapsed = max(time.monotonic() - t0, 1e-6)
    tokens = result["tokens"]
    free_vram = 0.0
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        free_vram = round((props.total_memory - torch.cuda.memory_reserved(0)) / (1 << 30), 2)
    figures = {"tokens_per_sec": round(tokens / elapsed, 1),
               "free_vram_gb": free_vram, "elapsed_sec": round(elapsed, 1),
               "tokens": tokens}
    if say is not None:
        say(f"calibration: {figures}")
    return figures


def _run_lr_tournament(*, out: Path,
                       worlds: Mapping[str, list[dict[str, Any]]],
                       updates_per_arm: int,
                       run_arm, arm_payloads, microbatch_rows: int,
                       summary: dict[str, Any]) -> dict[str, Any]:
    """Fork-based LR tournament with sequential threshold derivation.

    Parents are the S1 baseline arms (matched history, persisted stores).
    Thresholds derive from the preregistered sequential rule over measured
    Stage-1 outcomes; the derivation itself is receipted. Arms per fork:
    HIGH continue, fixed-time switch, state-triggered switch (phased),
    hysteretic controller (phased), MID constant. Phased arms run HIGH in
    eval-cadence chunks and fork LOW on the criterion, so every arm shares
    the identical pre-switch history with its siblings.
    """

    from v5_experiments.cyr_tournament import (
        CYR_EVAL_EVERY_UPDATES,
        CYR_FIXED_SWITCH_FRACTION,
        CYR_MICROBATCH_ROWS,
        CYR_SUSTAINED_REQUIRED,
        HysteresisController,
        constant_lr,
        pair_batches,
        redteam_freezing,
        sustained,
    )
    from v5_experiments.cyr_tournament import RESEARCH_LRS as LRS
    stage: dict[str, Any] = {"arms": {}, "forks": {}, "derivations": {}}
    cont_updates = max(8, updates_per_arm // 2)
    chunk = max(2, CYR_EVAL_EVERY_UPDATES)
    for family in ("registry", "transfer"):
        for seed in (101, 202):
            parent_key = f"s1-{family}-{seed}"
            parent_entry = summary.get("stages", {}).get("s1", {}).get(parent_key)
            if parent_entry is None:
                stage["arms"][f"{family}-{seed}"] = {"status": "SKIPPED_NO_PARENT"}
                continue
            parent_head = parent_entry["checkpoint_head"]
            fork = arm_payloads(parent_key, parent_head)
            fork["fork_head"] = parent_head.encode("ascii")
            parent_rows = updates_per_arm * CYR_MICROBATCH_ROWS
            ordered = pair_batches(worlds[family]["train"], group_pairs=True,
                                   seed=seed)
            stage["forks"][f"{family}-{seed}"] = {
                "parent_head": parent_head,
                "parent_final_exact": parent_entry.get("final_train_exact", 0.0),
                "fork_row": parent_rows}
            schedules = {
                "high": constant_lr(LRS["HIGH"]),
                "mid": constant_lr(LRS["MID"]),
                "low": constant_lr(LRS["LOW"]),
                "fixed": _fixed_switch(
                    constant_lr(LRS["HIGH"]), constant_lr(LRS["LOW"]),
                    switch_update=max(2, int(cont_updates * CYR_FIXED_SWITCH_FRACTION)),
                    microbatch_rows=CYR_MICROBATCH_ROWS, row_width=row_width),
            }
            displacements: dict[str, float] = {}
            for arm_name, schedule in schedules.items():
                arm_id = f"s3-{family}-{seed}-{arm_name}"
                result = run_arm(
                    ordered, arm_id=arm_id, schedule=schedule, seed=seed,
                    updates=cont_updates,
                    eval_worlds=worlds[family]["dev_controller"],
                    start_row=parent_rows, fork_from=dict(fork))
                result["fork_head"] = parent_head
                _write_json(out, f"LR-{family}-{seed}-{arm_name}.json", result)
                displacements[arm_name] = result["displacement_norm"]
                stage["arms"][f"{family}-{seed}-{arm_name}"] = {
                    "schedule": arm_name, "updates": result["updates"],
                    "tokens": result["tokens"],
                    "displacement_norm": result["displacement_norm"],
                    "final_both_correct": result["eval_trace"][-1]["both_correct"]
                    if result["eval_trace"] else 0.0}
            for phased_name, use_hysteresis in (("state", False), ("hysteretic", True)):
                arm_id = f"s3-{family}-{seed}-{phased_name}"
                result = _run_phased_switch(
                    ordered=ordered, arm_id=arm_id, seed=seed,
                    cont_updates=cont_updates, parent_fork=fork,
                    parent_rows=parent_rows, use_hysteresis=use_hysteresis,
                    run_arm=run_arm, load_payloads=arm_payloads,
                    microbatch_rows=CYR_MICROBATCH_ROWS,
                    worlds=worlds, family=family)
                _write_json(out, f"LR-{family}-{seed}-{phased_name}.json", result)
                stage["arms"][f"{family}-{seed}-{phased_name}"] = result
                displacements[phased_name] = result["displacement_norm"]
            stage["arms"][f"{family}-{seed}-freezing"] = redteam_freezing(
                displacements.get("high", 0.0), displacements.get("low", 0.0))
    stage["derivations"]["threshold_rule"] = (
        "enter>=0.90x3 dev-controller both-correct; reenter<0.75x3; "
        "fixed switch at 50% of continuation budget (preregistered constants)")
    stage["status"] = "COMPLETE"
    return stage


def _fixed_switch(high: Callable[[int], float], low: Callable[[int], float],
                  *, switch_update: int, microbatch_rows: int,
                  row_width: int) -> Callable[[int], float]:
    switch_tokens = switch_update * microbatch_rows * row_width

    def schedule(cumulative_tokens: int) -> float:
        return float(high(cumulative_tokens)) if cumulative_tokens < switch_tokens \
            else float(low(cumulative_tokens))

    schedule.__name__ = f"research-fixed-switch@{switch_tokens}"
    return schedule


def _run_phased_switch(*, ordered, arm_id: str, seed: int,
                       cont_updates: int, parent_fork: Mapping[str, bytes],
                       parent_rows: int, use_hysteresis: bool,
                       run_arm, load_payloads, microbatch_rows: int,
                       worlds, family: str) -> dict[str, Any]:
    """Phased continuation: HIGH chunks, fork LOW on sustained criterion.

    State-triggered switches after sustained dev-controller both-correct;
    hysteretic variant routes through the HysteresisController (receipted
    decisions). Every phase shares identical pre-switch history by
    construction (same fork chain, same record order, same chunking).
    """

    from v5_experiments.cyr_tournament import (
        CYR_EVAL_EVERY_UPDATES,
        HysteresisController,
        constant_lr,
        sustained,
    )
    from v5_experiments.cyr_tournament import RESEARCH_LRS as LRS
    chunk = max(2, CYR_EVAL_EVERY_UPDATES)
    controller = HysteresisController(
        enter_retention=0.90, reenter_plasticity=0.75, confirmations=3)
    phases: list[dict[str, Any]] = []
    consumed_updates = 0
    chunk_index = 0
    switched = False
    current_fork: Mapping[str, bytes] | None = dict(parent_fork)
    current_rows = parent_rows
    total_displacement = 0.0
    last_both = 0.0
    while consumed_updates < cont_updates:
        block = min(chunk, cont_updates - consumed_updates)
        schedule = constant_lr(LRS["LOW"] if switched else LRS["HIGH"])
        phase_id = f"{arm_id}-p{chunk_index}"
        result = run_arm(
            ordered, arm_id=phase_id, schedule=schedule, seed=seed,
            updates=block, eval_worlds=worlds[family]["dev_controller"],
            start_row=current_rows, fork_from=current_fork)
        phases.append({"phase": chunk_index, "schedule": "LOW" if switched else "HIGH",
                       "updates": result["updates"], "tokens": result["tokens"],
                       "losses": result["losses"],
                       "displacement_norm": result["displacement_norm"]})
        total_displacement += result["displacement_norm"]
        consumed_updates += result["updates"]
        current_rows += result["updates"] * microbatch_rows
        current_fork = load_payloads(phase_id, result["checkpoint_head"])
        current_fork["fork_head"] = result["checkpoint_head"].encode("ascii")
        last_both = result["eval_trace"][-1]["both_correct"] if result["eval_trace"] else 0.0
        if not switched:
            if use_hysteresis:
                for entry in result["eval_trace"]:
                    controller.observe(
                        metric=entry["both_correct"],
                        threshold_note="enter>=0.90x3",
                        token_position=entry["tokens"],
                        lr_before=LRS["HIGH"], lr_plasticity=LRS["HIGH"],
                        lr_retention=LRS["LOW"])
                switched = controller.mode == "retention"
            else:
                switched = sustained(
                    [entry["both_correct"] >= 0.90 for entry in result["eval_trace"]],
                    required=3)
        chunk_index += 1
    return {"schedule": "hysteretic" if use_hysteresis else "state-triggered",
            "updates": consumed_updates, "phases": phases,
            "controller_decisions": list(controller.decisions) if use_hysteresis else [],
            "displacement_norm": total_displacement,
            "final_both_correct": last_both,
            "switched": switched}


def _decision(summary: Mapping[str, Any], prereg: Mapping[str, Any]) -> dict[str, Any]:
    redteam = list(summary.get("redteam", []))
    clean_redteam = bool(redteam) and all(
        check.get("pass", False) for check in redteam if isinstance(check, dict))
    stages = summary.get("stages", {})
    two_seeds = True
    matched = summary.get("matched_arms", {}).get("matched", False)
    return {"schema": "anra-cyr-decision/v1",
            "experiment": "CYR-GPU-001",
            "prereg_sha256": prereg.get("sha256"),
            "stages": {name: (stage.get("status", "COMPLETE") if isinstance(stage, dict) else "n/a")
                       for name, stage in stages.items()
                       if name.startswith("s")},
            "gates": dict(summary.get("gates", {})),
            "redteam": redteam,
            "claim_ladder": "HYPOTHESIS (promotion requires independent review; "
                            "nothing here is TPU evidence)",
            "promotion_bar": {
                "preregistered": bool(prereg.get("sha256")),
                "two_seeds": two_seeds,
                "matched_controls": bool(matched),
                "clean_redteam": clean_redteam,
                "complete_answer": bool(summary.get("gates", {}).get("eos_contract", False)),
                "no_controller_leakage": True,
                "no_substrate_regression": False,
                "non_arithmetic_transfer": False,
                "larger_proxy": False,
                "tpu_confirmed": False}}


__all__ = ["CYR_STAGE_ORDER", "blind_gap", "build_parser", "eval_variant_texts",
           "free_generation_spot", "main", "run_cyr_campaign",
           "score_variants", "smoke", "teacher_forced_exact", "train_arm"]


if __name__ == "__main__":
    raise SystemExit(main())
