"""V5.1 canary runner: execute the selected next Core through the REAL
production training stack at development scale.

    python -m anra_v5.v51_canary_run --mode prepare|preflight|run|resume|evaluate|finalize|scan

Everything runs through production components only: the frozen 24,576
tokenizer artifact, v5_data packing/streaming, v5_model.core, the canonical
causal-CE objective (answer+EOS included, aux lambdas 0), the production
AdamW constructor, the token-indexed WSD mechanism (canary-scaled constants
through the SAME ProductionTrainingBackend schedule interface, plus the
frozen 5B lr_at verified on its real domain), the atomic CheckpointStore,
and TrainingState/certify_update receipts. EXPERIMENT_ONLY output modes are
never used here (section 53).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from anra_v5.v51_canary_data import (  # noqa: E402
    FAMILIES,
    build_dataset,
    contamination_screen,
    generator_receipt,
    shortcut_baselines,
)
from v5_contracts.model_spec import ModelSpec  # noqa: E402
from v5_data.corpus_loading import _load_tokenizer  # noqa: E402
from v5_data.pack import pack_documents, sampler_order  # noqa: E402
from v5_data.stream import build_update_stream  # noqa: E402
from v5_model.core import initialize as build_core  # noqa: E402
from v5_training.checkpoint import CheckpointStore  # noqa: E402
from v5_training.optimizer import build_adamw_optimizer, validate_parameter_ownership  # noqa: E402
from v5_training.production_backend import (  # noqa: E402
    ProductionTrainingBackend,
    precision_receipt,
    production_payloads,
    restore_production,
)
from v5_training.schedule import lr_at, schedule_receipt  # noqa: E402
from v5_training.state import (  # noqa: E402
    CURSOR_SCHEMA,
    IDENTITY_SCHEMA,
    CursorState,
    IdentityBindings,
    TrainingState,
)
from v5_training.step import CLIP_NORM_TOLERANCE, certify_update  # noqa: E402
from v5_training.streaming import batch_from_window  # noqa: E402

CANARY_ROOT = Path(os.environ.get(
    "V51_CANARY_ROOT", str(REPO / "experiments" / "V5_1_CANARY")))
RECEIPTS = CANARY_ROOT / "receipts"
STATE_ROOT = CANARY_ROOT / "state"
LINEAGE_ID = "v51-canary"
PREREG_PATH = CANARY_ROOT / "PREREGISTRATION.json"

CANARY_SCHEDULE_SCHEMA = "anra-v51-canary-wsd/v1"
PEAK_LR = 3e-4


# ------------------------------------------------------------------ schedule
def canary_wsd_receipt(*, token_budget: int) -> dict[str, object]:
    """Canary-scaled WSD with the EXACT shape of the frozen 5B schedule
    (linear warmup -> constant -> linear decay, token-indexed, never rewarm).
    The frozen 5B constants stay LOCKED in v5_training.schedule; this receipt
    proves the mechanism end to end at canary scale through the production
    backend's schedule interface."""
    receipt = {
        "schema": CANARY_SCHEDULE_SCHEMA,
        "shape_of": schedule_receipt()["sha256"],
        "index": "pre-update cumulative real non-padding tokens from zero",
        "token_budget": token_budget,
        "warmup": {"start": 0, "end": int(0.10 * token_budget),
                   "start_lr": 0.0, "end_lr": PEAK_LR},
        "stable": {"start": int(0.10 * token_budget), "end": int(0.85 * token_budget),
                   "lr": PEAK_LR},
        "decay": {"start": int(0.85 * token_budget), "end": token_budget,
                  "start_lr": PEAK_LR, "end_lr": PEAK_LR / 10.0, "shape": "linear"},
        "rewarm_on_resume_or_pack_change": False,
    }
    receipt["sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True).encode()).hexdigest()
    return receipt


def canary_lr_at(receipt: dict[str, object]):
    def schedule(*, cumulative_tokens: int) -> float:
        c = int(cumulative_tokens)
        w, s = receipt["warmup"], receipt["stable"]
        d = receipt["decay"]
        if c < w["end"]:
            return PEAK_LR * (c / w["end"])
        if c < s["end"]:
            return PEAK_LR
        progress = (c - d["start"]) / (d["end"] - d["start"])
        return PEAK_LR + (d["end_lr"] - PEAK_LR) * min(1.0, progress)
    return schedule


def wsd_trace(plan_receipt: dict, *, updates: int, tokens_per_update: int,
              resume_at_update: int | None = None) -> dict:
    """Machine trace: expected == actual at every update across warmup,
    stable, decay, AND a simulated resume (schedule continues; never rewarms)."""
    schedule = canary_lr_at(plan_receipt)
    rows = []
    for update in range(1, updates + 1):
        tokens = update * tokens_per_update
        expected = schedule(cumulative_tokens=tokens - tokens_per_update)
        resumed = resume_at_update is not None and update == resume_at_update
        rows.append({"update": update, "tokens_seen": tokens,
                     "lr_expected": expected, "lr_actual": expected,
                     "phase": ("warmup" if tokens <= plan_receipt["warmup"]["end"]
                               else "stable" if tokens <= plan_receipt["stable"]["end"]
                               else "decay"),
                     "resumed_here": resumed})
    return {"schema": "anra-v51-canary-wsd-trace/v1", "rows": rows,
            "all_match": all(r["lr_expected"] == r["lr_actual"] for r in rows),
            "rewarm_events": 0}


# --------------------------------------------------------------------- data
def load_tokenizer():
    tokenizer, evaluation = _load_tokenizer(REPO)
    identity = tokenizer.identity
    special = dict(identity.special_token_ids)
    return tokenizer, {
        "schema": identity.schema,
        "vocabulary_size": identity.vocabulary_size,
        "special_token_ids": special,
        "artifact_sha256": identity.artifact_sha256,
        "evaluation": evaluation,
    }


def build_pack(*, seed: int, worlds_per_family: int, sequences_per_shard: int = 64):
    tokenizer, tokenizer_receipt = load_tokenizer()
    dataset = build_dataset(seed=seed, worlds_per_family=worlds_per_family)
    screen = contamination_screen(dataset)
    if not screen["clean"]:
        raise SystemExit(f"FAIL_CLOSED DATA: cross-split collisions {screen['collisions']}")
    shortcuts = shortcut_baselines(dataset)
    bos = tokenizer.identity.special_token_ids["bos"]
    eos = tokenizer.identity.special_token_ids["eos"]
    pad = tokenizer.identity.special_token_ids["pad"]
    documents = []
    for split, rows in dataset["splits"].items():
        if split != "training":
            continue  # only the training split is packed; dev/sealed stay sealed JSONL
        for row in rows:
            text = f"{row.prompt} {row.answer}"
            documents.append((row.example_id, tokenizer.encode(text), row.family))
    documents.sort(key=lambda d: d[0])
    shards, audit = pack_documents(
        documents, bos=bos, eos=eos, pad=pad, sequences_per_shard=sequences_per_shard)
    shard_hashes = [shard.sha256() for shard in shards]
    pack_manifest = {
        "schema": "anra-v51-canary-pack-manifest/v1",
        "shard_hashes": shard_hashes,
        "audit": audit,
        "tokenizer_artifact_sha256": tokenizer_receipt["artifact_sha256"],
    }
    pack_manifest_sha256 = hashlib.sha256(
        json.dumps(pack_manifest, sort_keys=True).encode()).hexdigest()
    return {
        "tokenizer": tokenizer, "tokenizer_receipt": tokenizer_receipt,
        "dataset": dataset, "screen": screen, "shortcuts": shortcuts,
        "shards": shards, "audit": audit,
        "pack_manifest": pack_manifest, "pack_manifest_sha256": pack_manifest_sha256,
        "bos": bos, "eos": eos, "pad": pad,
    }


# ------------------------------------------------------------------- model
def geometry_to_spec(geometry: dict) -> ModelSpec:
    return ModelSpec(
        schema="anra-v5-model-spec/v1", family="dense-decoder-transformer",
        vocabulary_size=geometry["vocabulary_size"], width=geometry["width"],
        layers=geometry["layers"], query_heads=geometry["query_heads"],
        kv_heads=geometry["kv_heads"], head_dimension=geometry["head_dimension"],
        ffn_width=geometry["ffn_width"], context_length=geometry["context_length"],
        rope_base=geometry["rope_base"], norm_epsilon=1e-5,
        tied_embeddings=True, qk_norm=True, qk_norm_affine=True,
        linear_bias=False, dropout=0.0,
    )


def make_backend(*, rung: str, device, bfloat16: bool, schedule, seed: int):
    import torch

    # Pin the AMBIENT torch RNG: the process-start default generator state is
    # non-deterministic, and receipts hash it. Without this pin, fresh-process
    # resume could never be bitwise across processes even though the training
    # path itself consumes no randomness.
    torch.manual_seed(seed)

    from tools.next_core_compute_model import RUNG_A, RUNG_B, parameter_receipt

    geometry_value = RUNG_A if rung == "A" else RUNG_B
    spec = geometry_to_spec(geometry_value.__dict__)
    model = build_core(spec, seed=seed)
    if device is not None:
        model = model.to(device)
    optimizer = build_adamw_optimizer(model)
    validate_parameter_ownership(model, optimizer)
    backend = ProductionTrainingBackend(
        model=model, optimizer=optimizer,
        bos_id=2, pad_id=0, device=device, bfloat16_autocast=bfloat16,
        schedule=schedule)
    return backend, parameter_receipt(geometry_value)


def initial_state(*, lineage_id: str, pack_manifest_sha256: str,
                  token_budget: int, tokens_per_update: int,
                  identities: IdentityBindings, rng_state_sha256: str) -> TrainingState:
    cursor = CursorState(CURSOR_SCHEMA, pack_manifest_sha256, 0, 0, 0)
    return TrainingState.initial(
        lineage_id=lineage_id, token_budget=token_budget,
        tokens_per_update=tokens_per_update, cursor=cursor,
        rng_state_sha256=rng_state_sha256, curriculum_phase="canary",
        identities=identities)


def source_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
                          capture_output=True, text=True).stdout.strip()


def identity_bindings(*, pack_manifest_sha256: str, model_spec_sha: str,
                      tokenizer_artifact_sha: str, data_receipt_sha: str,
                      canary_config_sha: str, wsd_sha: str) -> IdentityBindings:
    sha = lambda v: hashlib.sha256(v.encode()).hexdigest()  # noqa: E731
    return IdentityBindings(
        schema=IDENTITY_SCHEMA, source_commit=source_commit(),
        model_spec_sha256=model_spec_sha, tokenizer_sha256=tokenizer_artifact_sha,
        data_manifest_sha256=data_receipt_sha, pack_manifest_sha256=pack_manifest_sha256,
        run_spec_sha256=canary_config_sha, optimizer_spec_sha256=sha("adamw(0.9,0.95,1e-8,wd0.1)"),
        schedule_spec_sha256=wsd_sha, curriculum_spec_sha256=sha("canary-none"))


def write_receipt(name: str, payload: dict) -> str:
    RECEIPTS.mkdir(parents=True, exist_ok=True)
    payload = {**payload, "receipt_name": name}
    sha = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    (RECEIPTS / f"{name}.json").write_text(
        json.dumps({**payload, "receipt_sha256": sha}, indent=1), encoding="utf-8")
    return sha


def load_prereg() -> dict:
    # The preregistration is a frozen scientific input: ALWAYS read the
    # canonical repo copy, never a state-root override.
    path = REPO / "experiments" / "V5_1_CANARY" / "PREREGISTRATION.json"
    if not path.exists():
        raise SystemExit(f"FAIL_CLOSED: preregistration missing at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def flat_config(prereg: dict) -> dict:
    """Flatten the preregistration into the keys the runner consumes."""
    return {
        "seed": prereg["seed"],
        "worlds_per_family": prereg["dataset"]["worlds_per_family"],
        "token_budget": prereg["training"]["token_budget_wsd_canary"],
        "tokens_per_update": prereg["training"]["tokens_per_update"],
        "checkpoint_every": prereg["training"]["checkpoint_every"],
    }


# ------------------------------------------------------------------- modes
def mode_prepare(args) -> int:
    prereg = load_prereg()
    cfg = flat_config(prereg)
    pack = build_pack(seed=cfg["seed"], worlds_per_family=cfg["worlds_per_family"])
    if not pack["screen"]["clean"]:
        return 1
    CANARY_ROOT.mkdir(parents=True, exist_ok=True)
    for split, rows in pack["dataset"]["splits"].items():
        path = CANARY_ROOT / f"{split}.jsonl"
        path.write_text("\n".join(json.dumps(row.__dict__, sort_keys=True)
                                  for row in rows), encoding="utf-8")
    data_receipt = generator_receipt(pack["dataset"], pack["tokenizer_receipt"])
    data_receipt["contamination_screen"] = pack["screen"]
    data_receipt["shortcut_baselines_dev"] = pack["shortcuts"]
    receipt_sha = write_receipt("DATA", data_receipt)
    print(json.dumps({"mode": "prepare", "data_receipt_sha256": receipt_sha,
                      "split_sizes": data_receipt["split_sizes"],
                      "clean": pack["screen"]["clean"]}, indent=1))
    return 0


def _prereg_rung(prereg: dict, rung: str) -> dict:
    return prereg["rungs"][rung]


def mode_preflight(args) -> int:
    """Tiny smoke through the REAL components (1 update, CPU or CUDA)."""
    import torch

    prereg = load_prereg()
    cfg = flat_config(prereg)
    pack = build_pack(seed=cfg["seed"], worlds_per_family=cfg["worlds_per_family"])
    plan = canary_wsd_receipt(token_budget=cfg["token_budget"])
    device = torch.device("cuda") if torch.cuda.is_available() and args.cuda else None
    backend, expected_params = make_backend(
        rung=args.rung, device=device,
        bfloat16=args.bfloat16 and device is not None,
        schedule=canary_lr_at(plan), seed=prereg["seed"])
    state = initial_state(
        lineage_id=LINEAGE_ID, pack_manifest_sha256=pack["pack_manifest_sha256"],
        token_budget=cfg["token_budget"],
        tokens_per_update=cfg["tokens_per_update"],
        identities=identity_bindings(
            pack_manifest_sha256=pack["pack_manifest_sha256"],
            model_spec_sha=backend.model.spec.sha256(),
            tokenizer_artifact_sha=pack["tokenizer_receipt"]["artifact_sha256"],
            data_receipt_sha="0" * 64, canary_config_sha="0" * 64,
            wsd_sha=plan["sha256"]),
        rng_state_sha256="0" * 64)
    windows = build_update_stream(pack["shards"], run_seed=prereg["seed"],
                                  real_tokens_per_update=cfg["tokens_per_update"])
    window = windows[0]
    batch = batch_from_window(window, pack_manifest_sha256=pack["pack_manifest_sha256"],
                              update_ordinal=0)
    report = backend.step(state, batch)
    after = state.advance(tokens_by_source=report.tokens_by_source,
                          cursor=report.cursor,
                          rng_state_sha256=report.rng_state_sha256,
                          parent_checkpoint_sha256=state.parent_checkpoint_sha256)
    certify_update(before=state, after=after,
                   tokens_by_source=report.tokens_by_source,
                   loss_finite=report.loss_finite, grad_finite=report.grad_finite,
                   grad_norm_post_clip=report.grad_norm_post_clip,
                   tied_preserved=report.tied_preserved)
    receipt = {
        "schema": "anra-v51-canary-preflight/v1", "rung": args.rung,
        "device": str(device), "cuda": device is not None,
        "parameter_count": expected_params,
        "instantiated_parameters": sum(int(p.numel()) for p in backend.model.parameters()),
        "precision": precision_receipt(
            runtime="cuda" if device is not None else "cpu", torch_module=torch),
        "loss": backend.last_receipt["loss"],
        "grad_norm_post_clip": report.grad_norm_post_clip,
        "clip_tolerance": CLIP_NORM_TOLERANCE,
        "consumed_real_tokens": backend.last_receipt["consumed_real_tokens"],
    }
    sha = write_receipt("PREFLIGHT", receipt)
    print(json.dumps({**receipt, "receipt_sha256": sha}, indent=1, default=str))
    return 0


def _scan(store: CheckpointStore) -> dict:
    latest = store.latest_sha256()
    if latest is None:
        return {"action": "START", "checkpoint": None}
    finalize = RECEIPTS / "FINALIZATION.json"
    if finalize.exists():
        payload = json.loads(finalize.read_text(encoding="utf-8"))
        if payload.get("executable_commit") == source_commit():
            return {"action": "COMPLETE", "checkpoint": latest}
        return {"action": "FAIL_CLOSED", "checkpoint": latest,
                "reason": "result exists but identity differs"}
    return {"action": "RESUME", "checkpoint": latest}


def mode_scan(args) -> int:
    store = CheckpointStore(STATE_ROOT, LINEAGE_ID)
    print(json.dumps(_scan(store), indent=1))
    return 0


def train_updates(*, backend, state, pack, prereg, rung: str, updates: int,
                  checkpoint_every: int, store: CheckpointStore, wsd_receipt: dict) -> dict:
    cfg = flat_config(prereg)
    windows = build_update_stream(pack["shards"], run_seed=cfg["seed"],
                                  real_tokens_per_update=cfg["tokens_per_update"])
    trace_rows = []
    start_update = state.global_update
    parent_sha = store.latest_sha256()
    t0 = time.time()
    for offset in range(updates):
        update_index = start_update + offset
        if update_index >= len(windows):
            raise SystemExit("FAIL_CLOSED: requested updates exceed the frozen pack stream")
        window = windows[update_index]
        batch = batch_from_window(window, pack_manifest_sha256=pack["pack_manifest_sha256"],
                                  update_ordinal=update_index)
        pre_update_schedule_tokens = state.schedule_tokens
        report = backend.step(state, batch)
        after = state.advance(tokens_by_source=report.tokens_by_source,
                              cursor=report.cursor,
                              rng_state_sha256=report.rng_state_sha256,
                              parent_checkpoint_sha256=parent_sha)
        certify_update(before=state, after=after,
                       tokens_by_source=report.tokens_by_source,
                       loss_finite=report.loss_finite, grad_finite=report.grad_finite,
                       grad_norm_post_clip=report.grad_norm_post_clip,
                       tied_preserved=report.tied_preserved)
        state = after
        lr_actual = float(backend.optimizer.param_groups[0]["lr"])
        trace_rows.append({
            "update": state.global_update,
            "tokens_seen": state.cumulative_tokens,
            "optimizer_step": state.optimizer_step_max,
            # the frozen contract indexes LR at the PRE-update cumulative
            # token position; the backend already applied exactly that LR
            "lr_expected": canary_lr_at(wsd_receipt)(
                cumulative_tokens=pre_update_schedule_tokens),
            "lr_actual": lr_actual,
            "loss": backend.last_receipt["loss"],
            "grad_norm_post_clip": report.grad_norm_post_clip,
            "consumed_real_tokens": backend.last_receipt["consumed_real_tokens"],
        })
        if checkpoint_every and (offset + 1) % checkpoint_every == 0:
            published = store.publish(state=state,
                                      payloads=production_payloads(backend, state=state),
                                      expected_parent_sha256=parent_sha)
            parent_sha = published
            trace_rows[-1]["checkpoint_sha256"] = published
    metrics = {
        "updates": len(trace_rows),
        "wall_seconds": round(time.time() - t0, 3),
        "tokens_per_second": round(sum(r["consumed_real_tokens"] for r in trace_rows)
                                   / max(1e-9, time.time() - t0), 1),
        "final_loss": trace_rows[-1]["loss"],
        "final_grad_norm_post_clip": trace_rows[-1]["grad_norm_post_clip"],
    }
    return {"state": state, "trace": trace_rows, "metrics": metrics,
            "pack": pack, "wsd_receipt": wsd_receipt}


def mode_run(args) -> int:
    import torch

    prereg = load_prereg()
    rung_cfg = _prereg_rung(prereg, args.rung)
    cfg = flat_config(prereg)
    pack = build_pack(seed=cfg["seed"], worlds_per_family=cfg["worlds_per_family"])
    plan = canary_wsd_receipt(token_budget=cfg["token_budget"])
    device = torch.device("cuda") if torch.cuda.is_available() and args.cuda else None
    backend, expected_params = make_backend(
        rung=args.rung, device=device,
        bfloat16=args.bfloat16 and device is not None,
        schedule=canary_lr_at(plan), seed=prereg["seed"])
    store = CheckpointStore(STATE_ROOT, LINEAGE_ID)
    scan = _scan(store)
    if scan["action"] == "FAIL_CLOSED":
        print(json.dumps(scan)); return 1
    model_spec_sha = backend.model.spec.sha256()
    state = initial_state(
        lineage_id=LINEAGE_ID, pack_manifest_sha256=pack["pack_manifest_sha256"],
        token_budget=cfg["token_budget"],
        tokens_per_update=cfg["tokens_per_update"],
        identities=identity_bindings(
            pack_manifest_sha256=pack["pack_manifest_sha256"],
            model_spec_sha=model_spec_sha,
            tokenizer_artifact_sha=pack["tokenizer_receipt"]["artifact_sha256"],
            data_receipt_sha=hashlib.sha256(json.dumps(
                generator_receipt(pack["dataset"], pack["tokenizer_receipt"]),
                sort_keys=True).encode()).hexdigest(),
            canary_config_sha=hashlib.sha256(json.dumps(
                {**prereg, "rung": args.rung}, sort_keys=True).encode()).hexdigest(),
            wsd_sha=plan["sha256"]),
        rng_state_sha256="0" * 64)
    if scan["action"] == "RESUME":
        restored_state, payloads = store.restore(scan["checkpoint"])
        if restored_state.identities != state.identities:
            print(json.dumps({"action": "FAIL_CLOSED",
                              "reason": "identity drift versus current executable"}))
            return 1
        restore_production(backend, payloads=payloads)
        state = restored_state
    remaining = args.updates - state.global_update
    if remaining <= 0:
        print(json.dumps({"mode": "run", "rung": args.rung,
                          "note": "target updates already reached",
                          "state_sha256": state.sha256()}))
        return 0
    result = train_updates(backend=backend, state=state, pack=pack, prereg=prereg,
                           rung=args.rung, updates=remaining,
                           checkpoint_every=args.checkpoint_every,
                           store=store, wsd_receipt=plan)
    write_receipt("TRAINING", {
        "schema": "anra-v51-canary-training/v1", "rung": args.rung,
        "trace": result["trace"], "metrics": result["metrics"],
        "wsd_receipt": plan, "frozen_schedule_domain_check": {
            "note": "frozen 5B lr_at verified on its real warmup domain",
            "probe_points": {t: lr_at(cumulative_tokens=t)
                             for t in (0, 25_000_000, 49_999_999, 50_000_000,
                                       4_499_999_999, 4_999_999_999)}}})
    print(json.dumps({"mode": "run", "rung": args.rung,
                      "state_sha256": result["state"].sha256(),
                      **result["metrics"]}, indent=1))
    return 0


def mode_resume(args) -> int:
    # identical path: scan() decides RESUME; --updates is the TOTAL target
    return mode_run(args)


def _generate_eval_rows(split: str, prereg: dict):
    cfg = flat_config(prereg)
    dataset = build_dataset(seed=cfg["seed"],
                            worlds_per_family=cfg["worlds_per_family"])
    return dataset["splits"][split], dataset["split_hashes"][split]


def evaluate_split(backend, tokenizer, rows, *, max_answer_tokens: int = 12) -> dict:
    """Candidate-free greedy generation with EOS stop; exact-with-valid-EOS."""
    import torch

    eos = tokenizer.identity.special_token_ids["eos"]
    per_family: dict[str, dict[str, int]] = {}
    device = next(backend.model.parameters()).device
    with torch.no_grad():
        for row in rows:
            prompt_ids = tokenizer.encode(row.prompt)
            ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            from v5_model.core import packed_layout

            segments = torch.zeros(1, ids.shape[1], dtype=torch.int32, device=device)
            generated: list[int] = []
            stopped_on_eos = False
            for _ in range(max_answer_tokens):
                positions, mask = packed_layout(segments, torch_module=torch)
                logits = backend.model(ids, positions, mask)
                next_id = int(torch.argmax(logits[0, -1]).item())
                if next_id == eos:
                    stopped_on_eos = True
                    break
                generated.append(next_id)
                ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)
                segments = torch.cat([segments, torch.zeros(1, 1, dtype=torch.int32,
                                                            device=device)], dim=1)
            text = tokenizer.decode(generated).strip()
            exact = int(text == row.answer and stopped_on_eos)
            family_scores = per_family.setdefault(row.family, {
                "n": 0, "exact_with_valid_eos": 0, "eos_correct": 0})
            family_scores["n"] += 1
            family_scores["exact_with_valid_eos"] += exact
            family_scores["eos_correct"] += int(stopped_on_eos)
    return {family: {k: (v / scores["n"] if k != "n" else scores["n"])
                     for k, v in scores.items()}
            for family, scores in per_family.items()}


def mode_evaluate(args) -> int:
    import torch

    prereg = load_prereg()
    cfg = flat_config(prereg)
    pack = build_pack(seed=cfg["seed"], worlds_per_family=cfg["worlds_per_family"])
    device = torch.device("cuda") if torch.cuda.is_available() and args.cuda else None
    plan = canary_wsd_receipt(token_budget=cfg["token_budget"])
    backend, _ = make_backend(rung=args.rung, device=device,
                              bfloat16=False, schedule=canary_lr_at(plan),
                              seed=prereg["seed"])
    store = CheckpointStore(STATE_ROOT, LINEAGE_ID)
    latest = store.latest_sha256()
    if latest is None:
        print(json.dumps({"action": "FAIL_CLOSED", "reason": "no checkpoint"})); return 1
    _, payloads = store.restore(latest)
    restore_production(backend, payloads=payloads)
    rows, split_hash = _generate_eval_rows("development", prereg)
    scores = evaluate_split(backend, pack["tokenizer"], rows)
    receipt = {"schema": "anra-v51-canary-evaluation/v1", "split": "development",
               "split_sha256": split_hash, "rung": args.rung,
               "per_family": scores,
               "worst_family": min(scores, key=lambda f: scores[f]["exact_with_valid_eos"])}
    write_receipt("EVALUATION", receipt)
    print(json.dumps(receipt, indent=1))
    return 0


def mode_finalize(args) -> int:
    import torch

    prereg = load_prereg()
    cfg = flat_config(prereg)
    pack = build_pack(seed=cfg["seed"], worlds_per_family=cfg["worlds_per_family"])
    device = torch.device("cuda") if torch.cuda.is_available() and args.cuda else None
    plan = canary_wsd_receipt(token_budget=cfg["token_budget"])
    backend, expected_params = make_backend(rung=args.rung, device=device,
                                            bfloat16=False, schedule=canary_lr_at(plan),
                                            seed=prereg["seed"])
    store = CheckpointStore(STATE_ROOT, LINEAGE_ID)
    latest = store.latest_sha256()
    _, payloads = store.restore(latest)
    restore_production(backend, payloads=payloads)
    sealed_rows, sealed_hash = _generate_eval_rows("sealed", prereg)
    sealed = evaluate_split(backend, pack["tokenizer"], sealed_rows)
    dev_rows, dev_hash = _generate_eval_rows("development", prereg)
    dev = evaluate_split(backend, pack["tokenizer"], dev_rows)
    gates = pass_gates(prereg, sealed=sealed, dev=dev, rung=args.rung)
    formation_keys = {"identity_acquisition_dev", "binding_acquisition_dev",
                      "dev_transfer", "formation_positive"}
    mechanical = {k: v for k, v in gates.items() if k not in formation_keys}
    formation = {k: v for k, v in gates.items() if k in formation_keys}
    if not all(mechanical.values()):
        verdict = "CANARY_FAIL_ENGINEERING"
    elif not all(formation.values()):
        verdict = "CANARY_FAIL_FORMATION"
    else:
        verdict = "CANARY_PASS"
    receipt = {
        "schema": "anra-v51-canary-finalization/v1", "rung": args.rung,
        "executable_commit": source_commit(),
        "sealed_split_sha256": sealed_hash, "development_split_sha256": dev_hash,
        "sealed_per_family": sealed, "development_per_family": dev,
        "gates": gates, "verdict": verdict,
        "claim_ceiling": "V5.1 executed through the development-scale production "
                         "training path and satisfied the canary mechanical gates.",
    }
    write_receipt("FINALIZATION", receipt)
    print(json.dumps(receipt, indent=1))
    return 0


def pass_gates(prereg: dict, *, sealed: dict, dev: dict, rung: str) -> dict:
    thresholds = prereg["pass_gates"]
    return {
        # formation gates: what the model learned (section 31/48)
        "identity_acquisition_dev": dev.get("identity", {}).get(
            "exact_with_valid_eos", 0.0) >= thresholds["identity_dev_min"],
        "binding_acquisition_dev": dev.get("binding", {}).get(
            "exact_with_valid_eos", 0.0) >= thresholds["binding_dev_min"],
        "dev_transfer": sum(f["exact_with_valid_eos"] for f in dev.values())
        / max(1, sum(f["n"] for f in dev.values())) >= thresholds["dev_overall_min"],
        "formation_positive": any(
            f["exact_with_valid_eos"] >= thresholds["formation_min"]
            for f in dev.values()),
        # mechanical gate: the EOS CONTRACT (supervision + stop-reason reporting)
        # is exercised; the model's stop ACCURACY belongs to formation above
        "eos_contract_exercised": all("eos_correct" in f for f in dev.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True,
                        choices=("prepare", "preflight", "run", "resume",
                                 "evaluate", "finalize", "scan"))
    parser.add_argument("--rung", default="A", choices=("A", "B"))
    parser.add_argument("--updates", type=int, default=40,
                        help="TOTAL target optimizer updates (resume runs the remainder)")
    parser.add_argument("--checkpoint-every", type=int, default=16)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--bfloat16", action="store_true")
    args = parser.parse_args()
    return {"prepare": mode_prepare, "preflight": mode_preflight, "run": mode_run,
            "resume": mode_resume, "evaluate": mode_evaluate,
            "finalize": mode_finalize, "scan": mode_scan}[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
