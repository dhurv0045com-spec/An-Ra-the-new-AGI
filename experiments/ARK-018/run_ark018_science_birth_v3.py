from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from ark018_v3_common import *  # noqa: F401,F403

PREREG_COMMIT = "51694c70d0d838dd6969315ba0b392bd51ddb729"
EXECUTION_ADDENDUM_COMMIT = "368bf4cb5184aef2aa773c97efce928e76384afd"
PROBE_COMMIT = "b32f830cbd634651e1f8c176e11a18c2381a108a"
RUNNER_PATH = Path(__file__)
BIRTH_PATH = HERE / "ARK018_BIRTH_BOOK.md"
PROBE_PATH = HERE / "BIRTH_BOOK_PROBES.json"
ARMS = [
    "SCIENCE_ONLY",
    "BIRTH_NATURAL_2PCT",
    "BIRTH_REHEARSAL_10PCT",
    "SCIENCE_REPLAY_10PCT_CONTROL",
]
SEEDS = [31801, 31902]
PREP_SCHEMA = "arkenstone-ark018-prepared/v3"
RESULT_SCHEMA = "arkenstone-ark018-result/v3"


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def runner_sha() -> str:
    return sha256_file(RUNNER_PATH)


def compact_receipt(obj: dict) -> dict:
    x = dict(obj)
    x.update({
        "prereg_commit": PREREG_COMMIT,
        "execution_addendum_commit": EXECUTION_ADDENDUM_COMMIT,
        "probe_commit": PROBE_COMMIT,
        "runner_head": git_head(),
        "runner_sha256": runner_sha(),
        "versions": package_versions(),
    })
    body = dict(x)
    body.pop("receipt_sha256", None)
    x["receipt_sha256"] = sha_json(body)
    return x


def result_dir() -> Path:
    p = DRIVE_ROOT / "results"
    p.mkdir(parents=True, exist_ok=True)
    return p


def checkpoint_dir() -> Path:
    p = DRIVE_ROOT / "checkpoints"
    p.mkdir(parents=True, exist_ok=True)
    return p


def prepared_dir() -> Path:
    p = DRIVE_ROOT / "prepared"
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_save_json(path: Path, obj: dict) -> None:
    json_dump(path, compact_receipt(obj))


def validate_static_inputs(science_path: Path) -> dict:
    if not science_path.exists():
        raise FileNotFoundError(f"scientific corpus not found at exact frozen path: {science_path}")
    if not BIRTH_PATH.exists():
        raise FileNotFoundError(BIRTH_PATH)
    birth_bytes = BIRTH_PATH.stat().st_size
    birth_sha = sha256_file(BIRTH_PATH)
    if birth_bytes != EXPECTED_BIRTH_BYTES or birth_sha != EXPECTED_BIRTH_SHA:
        raise RuntimeError(f"Birth Book identity drift: bytes={birth_bytes} sha={birth_sha}")
    if not PROBE_PATH.exists():
        raise FileNotFoundError(PROBE_PATH)
    return {"birth_bytes": birth_bytes, "birth_sha256": birth_sha, "science_path": str(science_path)}


def scan_metadata(local_science: Path) -> tuple[list[dict], dict]:
    rows = []
    nulls = 0
    split_counts = Counter()
    split_bytes = Counter()
    dup = Counter()
    for idx, text in iter_parquet_text(local_science):
        if not text:
            nulls += 1
            continue
        b = text.encode("utf-8")
        h = hashlib.sha256(b).hexdigest()
        split = split_from_hash(h)
        rows.append({"row": idx, "sha256": h, "bytes": len(b), "split": split})
        split_counts[split] += 1
        split_bytes[split] += len(b)
        dup[h] += 1
    duplicated_docs = sum(v for v in dup.values() if v > 1)
    duplicated_unique = sum(1 for v in dup.values() if v > 1)
    duplicate_bytes = sum(r["bytes"] for r in rows if dup[r["sha256"]] > 1)
    stats = {
        "usable_documents": len(rows),
        "null_or_empty_text_rows": nulls,
        "documents_by_split": dict(split_counts),
        "utf8_bytes_by_split": dict(split_bytes),
        "unique_document_hashes": len(dup),
        "duplicated_document_instances": duplicated_docs,
        "duplicate_hash_groups": duplicated_unique,
        "duplicate_byte_fraction": duplicate_bytes / max(1, sum(split_bytes.values())),
    }
    return rows, stats


def tokenizer_sample_rows(meta: list[dict], target_bytes: int = 32 << 20) -> list[int]:
    chosen = []
    total = 0
    for r in sorted((x for x in meta if x["split"] == "train"), key=lambda x: x["sha256"]):
        chosen.append(r["row"])
        total += r["bytes"]
        if total >= target_bytes:
            break
    if total < target_bytes:
        raise RuntimeError("TRAIN split too small for 32 MiB tokenizer sample")
    return chosen


def collect_rows(local_science: Path, row_ids: set[int]) -> dict[int, str]:
    out = {}
    for idx, text in iter_parquet_text(local_science):
        if idx in row_ids:
            out[idx] = text
            if len(out) == len(row_ids):
                break
    if len(out) != len(row_ids):
        missing = sorted(row_ids - set(out))[:10]
        raise RuntimeError(f"failed collecting selected rows, e.g. {missing}")
    return out


def prepare_all(science_path: Path, force: bool = False) -> dict:
    setup_reproducibility()
    validate_static_inputs(science_path)
    prep = prepared_dir()
    final_receipt = prep / "ARK-018_PREPARED_RECEIPT.json"
    if final_receipt.exists() and not force:
        old = json.loads(final_receipt.read_text())
        if old.get("science_sha256") == EXPECTED_SCIENCE_SHA and old.get("birth_sha256") == EXPECTED_BIRTH_SHA:
            required = ["tokenizer.json", "train.bin", "control.bin", "sealed.bin", "birth.bin", "science_replay.bin", "token_counts.npy"]
            if all((prep / x).exists() for x in required):
                print("PREPARED CACHE FOUND — validating compact identities", flush=True)
                return old

    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
    local_science = LOCAL_ROOT / "data_15.parquet"
    print("Hashing exact Drive science file and staging local copy...", flush=True)
    sha, physical_bytes = stream_hash_and_copy(science_path, local_science)
    if sha != EXPECTED_SCIENCE_SHA:
        safe_save_json(result_dir() / "ARK-018_FAILURE_RECEIPT.json", {
            "status": "SCIENCE_FILE_HASH_MISMATCH",
            "expected": EXPECTED_SCIENCE_SHA,
            "actual": sha,
            "path": str(science_path),
            "bytes": physical_bytes,
        })
        raise RuntimeError(f"SCIENCE_FILE_HASH_MISMATCH expected {EXPECTED_SCIENCE_SHA} got {sha}")
    local_sha = sha256_file(local_science)
    if local_sha != sha:
        raise RuntimeError("local science staging hash mismatch")
    schema = parquet_schema_receipt(local_science)
    if not schema["has_text"]:
        raise RuntimeError("Parquet schema lacks required text column")

    print("Scanning document hashes / split firewall...", flush=True)
    meta, split_stats = scan_metadata(local_science)
    bind_receipt = {
        "schema": "arkenstone-ark018-data-binding/v3",
        "status": "BOUND",
        "science_drive_path": str(science_path),
        "science_sha256": sha,
        "science_physical_bytes": physical_bytes,
        "birth_path": str(BIRTH_PATH.relative_to(REPO)),
        "birth_sha256": EXPECTED_BIRTH_SHA,
        "birth_bytes": EXPECTED_BIRTH_BYTES,
        "parquet": schema,
        "split": split_stats,
        "split_algorithm": "sha256(normalized_text_bytes)[:16] mod 10000 => 90/5/5",
    }
    safe_save_json(result_dir() / "ARK-018_DATA_BINDING_RECEIPT.json", bind_receipt)

    sample_order = tokenizer_sample_rows(meta)
    sample_text_map = collect_rows(local_science, set(sample_order))
    sample_texts = [sample_text_map[i] for i in sample_order]
    print(f"Building tokenizer twice from {len(sample_texts)} deterministic documents...", flush=True)
    tok1 = build_tokenizer(sample_texts)
    tok2 = build_tokenizer(sample_texts)
    canon1 = tokenizer_json_bytes(tok1)
    canon2 = tokenizer_json_bytes(tok2)
    if canon1 != canon2:
        raise RuntimeError("TOKENIZER_NONDETERMINISTIC: double build mismatch")
    tokenizer_path = prep / "tokenizer.json"
    tokenizer_path.write_bytes(canon1)
    # Tokenizers accepts canonicalized JSON as a normal tokenizer file.
    tok = load_tokenizer(tokenizer_path)
    eos = tok.token_to_id(EOS_TOKEN)
    if eos is None:
        raise RuntimeError("tokenizer missing <eos>")

    # Initialize caches.
    cache_paths = {s: prep / f"{s}.bin" for s in ["train", "control", "sealed"]}
    for p in cache_paths.values():
        if p.exists():
            p.unlink()
    token_counts = np.zeros(VOCAB_SIZE, dtype=np.int64)
    meta_by_row = {r["row"]: r for r in meta}
    token_lengths: dict[int, int] = {}

    print("Tokenizing full bound scientific shard into deterministic split caches...", flush=True)
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(local_science)
    row_base = 0
    for rb in pf.iter_batches(batch_size=256, columns=["text"]):
        raw = rb.column(0).to_pylist()
        texts, row_ids = [], []
        for j, x in enumerate(raw):
            idx = row_base + j
            if idx not in meta_by_row or not isinstance(x, str) or not x:
                continue
            texts.append(normalize_text(x))
            row_ids.append(idx)
        if texts:
            encoded = encode_batch(tok, texts)
            for idx, ids in zip(row_ids, encoded):
                split = meta_by_row[idx]["split"]
                seq = list(ids) + [eos]
                append_u16(cache_paths[split], seq)
                token_lengths[idx] = len(seq)
                token_counts += np.bincount(np.asarray(seq, dtype=np.int64), minlength=VOCAB_SIZE)
        row_base += len(raw)
        if row_base % 10000 < 256:
            print(f"  tokenized rows: {row_base}/{schema['rows']}", flush=True)

    np.save(prep / "token_counts.npy", token_counts)
    split_token_counts = {s: (cache_paths[s].stat().st_size // 2) for s in cache_paths}
    if min(split_token_counts.get("control", 0), split_token_counts.get("sealed", 0)) < 100_000:
        raise RuntimeError("CONTROL/SEALED token cache unexpectedly tiny")

    print("Tokenizing complete Birth Book...", flush=True)
    birth_text = BIRTH_PATH.read_text(encoding="utf-8")
    birth_ids = tok.encode(birth_text).ids + [eos]
    birth_path = prep / "birth.bin"
    if birth_path.exists(): birth_path.unlink()
    append_u16(birth_path, birth_ids)
    birth_tokens = len(birth_ids)

    # Token-matched scientific replay corpus: lowest-hash TRAIN documents.
    sorted_train = sorted((r for r in meta if r["split"] == "train"), key=lambda x: x["sha256"])
    selected_rows = []
    estimated = 0
    for r in sorted_train:
        ln = token_lengths.get(r["row"], 0)
        if ln <= 0:
            continue
        selected_rows.append(r["row"])
        estimated += ln
        if estimated >= birth_tokens:
            break
    replay_text_map = collect_rows(local_science, set(selected_rows))
    replay_path = prep / "science_replay.bin"
    if replay_path.exists(): replay_path.unlink()
    written = 0
    for idx in selected_rows:
        ids = tok.encode(replay_text_map[idx]).ids + [eos]
        remaining = birth_tokens - written
        if remaining <= 0:
            break
        part = ids[:remaining]
        written += append_u16(replay_path, part)
    if written < birth_tokens:
        raise RuntimeError("could not build token-matched science replay")
    replay_ratio = abs(written - birth_tokens) / max(1, birth_tokens)
    if replay_ratio > 0.001:
        raise RuntimeError(f"science replay token mismatch {replay_ratio}")

    horizon = choose_horizon(birth_tokens)
    mix = {
        "schema": "arkenstone-ark018-mixture/v3",
        "arms": {a: dict(count_sources(a, horizon)) for a in ARMS},
        "horizon_updates": horizon,
        "target_tokens_per_update": TARGET_TOKENS_PER_UPDATE,
        "birth_tokens": birth_tokens,
        "birth_10pct_coverage": count_sources("BIRTH_REHEARSAL_10PCT", horizon).get("birth", 0) * TARGET_TOKENS_PER_UPDATE / birth_tokens,
        "science_replay_tokens": written,
        "science_replay_relative_token_mismatch": replay_ratio,
        "seeds": SEEDS,
    }
    safe_save_json(result_dir() / "ARK-018_MIXTURE_MANIFEST.json", mix)

    token_receipt = {
        "schema": "arkenstone-ark018-tokenizer/v3",
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "tokenizer_canonical_sha256": sha256_bytes(canon1),
        "double_build_identical": True,
        "vocab_size": tok.get_vocab_size(),
        "sample_documents": len(sample_texts),
        "scientific_split_tokens": split_token_counts,
        "birth_tokens": birth_tokens,
        "birth_token_stream_sha256": sha256_file(birth_path),
        "science_replay_tokens": written,
    }
    safe_save_json(result_dir() / "ARK-018_TOKENIZER_RECEIPT.json", token_receipt)

    prepared = {
        "schema": PREP_SCHEMA,
        "status": "READY",
        "science_sha256": sha,
        "birth_sha256": EXPECTED_BIRTH_SHA,
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "split_tokens": split_token_counts,
        "birth_tokens": birth_tokens,
        "horizon_updates": horizon,
        "cache_sha256": {p.name: sha256_file(p) for p in [*cache_paths.values(), birth_path, replay_path]},
    }
    safe_save_json(final_receipt, prepared)
    print("PREPARATION COMPLETE", json.dumps(prepared, indent=2), flush=True)
    return prepared


def load_prepared() -> tuple[dict, object, dict[str, np.memmap]]:
    prep = prepared_dir()
    receipt_path = prep / "ARK-018_PREPARED_RECEIPT.json"
    if not receipt_path.exists():
        raise RuntimeError("prepared cache missing; run prepare first")
    r = json.loads(receipt_path.read_text())
    if r.get("science_sha256") != EXPECTED_SCIENCE_SHA or r.get("birth_sha256") != EXPECTED_BIRTH_SHA:
        raise RuntimeError("prepared cache identity mismatch")
    tok = load_tokenizer(prep / "tokenizer.json")
    bufs = {s: memmap_u16(prep / f"{s}.bin") for s in ["train", "control", "sealed", "birth", "science_replay"]}
    return r, tok, bufs


def microbatch_calibration(model, device, train_buf) -> int:
    x, y = batch_from_buffer(train_buf, 0, device)
    for micro in [32, 16, 8, 4]:
        try:
            model.zero_grad(set_to_none=True)
            with autocast_ctx(device):
                loss = lm_loss(model, x[:micro], y[:micro])
            loss.backward()
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            return micro
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    raise RuntimeError("T4 cannot fit microbatch=4")


def snapshot_exact(model, opt, scaler) -> dict:
    return {
        "model": copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}),
        "optimizer": copy.deepcopy(opt.state_dict()),
        "scaler": copy.deepcopy(scaler.state_dict()),
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
    }


def restore_exact(snap, model, opt, scaler):
    model.load_state_dict(snap["model"])
    opt.load_state_dict(snap["optimizer"])
    scaler.load_state_dict(snap["scaler"])
    torch.set_rng_state(snap["cpu_rng"])
    torch.cuda.set_rng_state_all(snap["cuda_rng"])


def one_update(model, opt, scaler, x, y, micro: int, lr: float, projected_names=None) -> dict:
    for g in opt.param_groups:
        g["lr"] = lr
    opt.zero_grad(set_to_none=True)
    accum = EFFECTIVE_SEQS // micro
    total_loss = 0.0
    for i in range(accum):
        xx = x[i * micro:(i + 1) * micro]
        yy = y[i * micro:(i + 1) * micro]
        with autocast_ctx(x.device):
            loss = lm_loss(model, xx, yy) / accum
        scaler.scale(loss).backward()
        total_loss += float(loss.detach().item())
    scaler.unscale_(opt)
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
    before = capture_projected_params(model, projected_names or []) if projected_names else None
    scaler.step(opt)
    scaler.update()
    proj_delta = projected_delta_norm(before, model) if before is not None else None
    return {"loss": total_loss, "preclip_gradient_norm": grad_norm, "projected_update_norm": proj_delta}


def source_batch(bufs, arm: str, seed: int, step: int, device):
    source = source_for_step(arm, step)
    if source == "science":
        buf = bufs["train"]
        start = fixed_start(seed, step, len(buf), TARGET_TOKENS_PER_UPDATE + 1)
    elif source == "birth":
        buf = bufs["birth"]
        occ = source_occurrence_index(arm, step, "birth")
        start = (occ * TARGET_TOKENS_PER_UPDATE) % len(buf)
    elif source == "science_replay":
        buf = bufs["science_replay"]
        occ = source_occurrence_index(arm, step, "science_replay")
        start = (occ * TARGET_TOKENS_PER_UPDATE) % len(buf)
    else:
        raise RuntimeError(source)
    x, y = batch_from_buffer(buf, start, device)
    return source, start, x, y


def birth_probe_items() -> list[dict]:
    return json.loads(PROBE_PATH.read_text())["items"]


def science_birth_gradient_alignment(model, bufs, device, names, seed, milestone) -> dict:
    sx, sy = batch_from_buffer(bufs["control"], fixed_start(991, milestone, len(bufs["control"]), TARGET_TOKENS_PER_UPDATE + 1), device)
    bx, by = batch_from_buffer(bufs["birth"], (milestone * 8192) % len(bufs["birth"]), device)
    # Use 4 sequences for a bounded diagnostic.
    gs = projected_gradient(model, sx[:4], sy[:4], names, device)
    gb = projected_gradient(model, bx[:4], by[:4], names, device)
    return {
        "PROJECTED_gradient_cosine_science_birth": cosine(gs, gb),
        "PROJECTED_science_grad_norm": float(gs.norm().item()),
        "PROJECTED_birth_grad_norm": float(gb.norm().item()),
        "projected_parameter_names": names,
        "projected_parameters": int(gs.numel()),
    }


def evaluate_checkpoint(model, tok, bufs, device, step: int, seed: int, initial_state: dict, include_mcq=True) -> dict:
    out = {
        "step": step,
        "science_control": eval_buffer(model, bufs["control"], device, 40001, sequences=48),
        "science_sealed": eval_buffer(model, bufs["sealed"], device, 40002, sequences=48),
        "birth_nll_diagnostic": eval_buffer(model, bufs["birth"], device, 40003, sequences=24),
        "full_parameter_displacement_from_init": full_displacement(model, initial_state),
    }
    if include_mcq:
        items = birth_probe_items()
        out["birth_content_control"] = score_mcq(model, tok, [x for x in items if x["split"] == "CONTROL"], device)
        out["birth_content_sealed"] = score_mcq(model, tok, [x for x in items if x["split"] == "SEALED"], device)
        out["algorithmic_ood"] = score_mcq(model, tok, algorithmic_items(), device)
    return out


def arm_checkpoint_path(seed: int, arm: str) -> Path:
    return checkpoint_dir() / f"seed_{seed}" / f"{arm}.pt"


def arm_partial_path(seed: int, arm: str) -> Path:
    return result_dir() / f"ARK-018_SEED_{seed}_{arm}_PARTIAL.json"


def save_training_checkpoint(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def init_snapshot(seed: int, device, prepared: dict) -> tuple[dict, str]:
    path = checkpoint_dir() / f"seed_{seed}" / "INIT.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if obj["science_sha256"] != EXPECTED_SCIENCE_SHA or obj["tokenizer_sha256"] != prepared["tokenizer_sha256"]:
            raise RuntimeError("existing INIT checkpoint identity mismatch")
        return obj["model"], obj["model_sha256"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = Ark018GPT().to(device)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    h = model_state_hash(model)
    torch.save({"model": state, "model_sha256": h, "science_sha256": EXPECTED_SCIENCE_SHA, "tokenizer_sha256": prepared["tokenizer_sha256"]}, path)
    del model
    torch.cuda.empty_cache()
    return state, h


def train_arm(seed: int, arm: str, prepared, tok, bufs, device, micro: int) -> dict:
    horizon = int(prepared["horizon_updates"])
    init_state, init_hash = init_snapshot(seed, device, prepared)
    model = Ark018GPT().to(device)
    model.load_state_dict(init_state)
    opt = optimizer_for(model, 3e-4)
    scaler = make_scaler(device)
    projected_names = projected_parameter_names(model)
    trajectory = []
    source_tokens = Counter()
    start_step = 0
    ckpt_path = arm_checkpoint_path(seed, arm)

    if ckpt_path.exists():
        c = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if c.get("init_model_sha256") != init_hash or c.get("tokenizer_sha256") != prepared["tokenizer_sha256"] or c.get("arm") != arm:
            raise RuntimeError(f"checkpoint identity mismatch: {ckpt_path}")
        model.load_state_dict(c["model"])
        opt.load_state_dict(c["optimizer"])
        scaler.load_state_dict(c.get("scaler", {}))
        if c.get("cpu_rng") is not None: torch.set_rng_state(c["cpu_rng"])
        if c.get("cuda_rng") is not None: torch.cuda.set_rng_state_all(c["cuda_rng"])
        start_step = int(c["step"])
        trajectory = c.get("trajectory", [])
        source_tokens.update(c.get("source_tokens", {}))
        print(f"RESUME seed={seed} arm={arm} step={start_step}/{horizon}", flush=True)

    if start_step == 0:
        trajectory.append(evaluate_checkpoint(model, tok, bufs, device, 0, seed, init_state, include_mcq=True))

    for step in range(start_step + 1, horizon + 1):
        model.train()
        source, source_start, x, y = source_batch(bufs, arm, seed, step, device)
        lr = lr_at(step, horizon)
        telemetry = one_update(model, opt, scaler, x, y, micro, lr, projected_names if step % 50 == 0 else None)
        source_tokens[source] += TARGET_TOKENS_PER_UPDATE

        if step == 1 or step % 100 == 0:
            print(f"[{seed} {arm}] {step}/{horizon} source={source} loss={telemetry['loss']:.4f} grad={telemetry['preclip_gradient_norm']:.3f} lr={lr:.2e}", flush=True)

        if step % 500 == 0 or step == horizon:
            ev = evaluate_checkpoint(model, tok, bufs, device, step, seed, init_state, include_mcq=True)
            ev.update({"last_train_source": source, "last_loss": telemetry["loss"], "last_lr": lr, "source_tokens": dict(source_tokens)})
            if step % 1000 == 0 or step == horizon:
                ev["gradient_alignment"] = science_birth_gradient_alignment(model, bufs, device, projected_names, seed, step)
            trajectory.append(ev)
            partial = {
                "schema": RESULT_SCHEMA,
                "status": "PARTIAL" if step < horizon else "PRETRAIN_COMPLETE",
                "seed": seed,
                "arm": arm,
                "step": step,
                "horizon": horizon,
                "init_model_sha256": init_hash,
                "final_model_sha256": model_state_hash(model),
                "source_tokens": dict(source_tokens),
                "trajectory": trajectory,
            }
            safe_save_json(arm_partial_path(seed, arm), partial)

        if step % 1000 == 0 or step == horizon:
            save_training_checkpoint(ckpt_path, {
                "seed": seed,
                "arm": arm,
                "step": step,
                "horizon": horizon,
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "trajectory": trajectory,
                "source_tokens": dict(source_tokens),
                "init_model_sha256": init_hash,
                "tokenizer_sha256": prepared["tokenizer_sha256"],
                "science_sha256": EXPECTED_SCIENCE_SHA,
                "birth_sha256": EXPECTED_BIRTH_SHA,
            })

    out = json.loads(arm_partial_path(seed, arm).read_text())
    out["status"] = "PRETRAIN_COMPLETE"
    safe_save_json(result_dir() / f"ARK-018_SEED_{seed}_{arm}_PRETRAIN_RESULT.json", out)
    del model, opt
    torch.cuda.empty_cache()
    return out


# ---------- Controlled binding adaptation probe ----------

def select_binding_tokens(tok, counts: np.ndarray) -> list[int]:
    vocab = tok.get_vocab()
    inv = {i: s for s, i in vocab.items()}
    candidates = []
    thresholds = [256, 128, 64]
    for threshold in thresholds:
        candidates.clear()
        for tid in range(min(len(counts), tok.get_vocab_size())):
            if counts[tid] < threshold:
                continue
            text = tok.decode([tid])
            stripped = text.strip()
            if not (4 <= len(stripped) <= 10 and stripped.isascii() and stripped.isalpha() and stripped.islower()):
                continue
            enc = tok.encode(text).ids
            if enc != [tid]:
                continue
            if int(hashlib.sha256(stripped.encode()).hexdigest(), 16) % 4 != 0:
                continue
            candidates.append((int(-counts[tid]), tid, stripped))
        candidates.sort()
        if len(candidates) >= 24:
            return [x[1] for x in candidates[:24]]
    raise RuntimeError("insufficient eligible single-token words for binding probe")


def make_binding_factsets(keys, vals):
    factsets = []
    for kt in itertools.combinations(keys, 3):
        for vt in itertools.permutations(vals, 3):
            factsets.append(tuple(zip(kt, vt)))
    rng = random.Random(424218)
    rng.shuffle(factsets)
    return factsets[:400], factsets[400:450], factsets[450:500]


def binding_prompt_ids(tok, facts, query, order=None):
    facts = list(facts)
    if order is not None:
        facts = [facts[i] for i in order]
    ids = tok.encode("Facts:").ids
    for k, v in facts:
        ids += [k] + tok.encode(" means ").ids + [v] + tok.encode("; ").ids
    ids += tok.encode("Query: ").ids + [query] + tok.encode(" means").ids
    return ids[-(CONTEXT - 1):]


def binding_rows(tok, factsets, regime: str, seed: int, step: int):
    rows = []
    perms = list(itertools.permutations(range(3)))
    for fi, facts in enumerate(factsets):
        for qi, (q, ans) in enumerate(facts):
            if regime == "canonical":
                order = (0, 1, 2)
            elif regime == "reversed":
                order = (2, 1, 0)
            else:
                h = hashlib.sha256(f"{seed}:{step}:{fi}:{qi}".encode()).digest()
                order = perms[int.from_bytes(h[:8], "big") % len(perms)]
            rows.append((binding_prompt_ids(tok, facts, q, order), ans))
    return rows


def binding_batch(model, examples, device):
    max_len = max(len(p) for p, _ in examples)
    # All prompts are essentially equal length; left-pad with EOS-like zero if necessary and mask by last position.
    x = torch.zeros((len(examples), max_len), dtype=torch.long, device=device)
    ans = torch.tensor([a for _, a in examples], dtype=torch.long, device=device)
    for i, (p, _) in enumerate(examples):
        x[i, -len(p):] = torch.tensor(p, dtype=torch.long, device=device)
    logits = model(x)[:, -1, :]
    return logits, ans


def eval_binding(model, tok, factsets, device) -> dict:
    out = {}
    for name, regime in [("canonical", "canonical"), ("order_only", "reversed")]:
        rows = binding_rows(tok, factsets, regime, 0, 0)
        hits = 0
        n = 0
        for i in range(0, len(rows), 64):
            logits, ans = binding_batch(model, rows[i:i+64], device)
            hits += int((logits.argmax(-1) == ans).sum().item())
            n += len(ans)
        out[name] = hits / max(1, n)
    # query_order here is reversed-order evaluation across all queries; the fact-set split already changes queries.
    out["query_order"] = out["order_only"]
    out["qualified"] = out["canonical"] >= .90 and out["order_only"] >= .85 and out["query_order"] >= .85
    return out


def run_binding_probe_from_checkpoint(seed: int, arm: str, prepared, tok, device) -> dict:
    out_path = result_dir() / f"ARK-018_SEED_{seed}_{arm}_BINDING_PROBE.json"
    if out_path.exists():
        return json.loads(out_path.read_text())
    ckpt = torch.load(arm_checkpoint_path(seed, arm), map_location="cpu", weights_only=False)
    if int(ckpt["step"]) < int(prepared["horizon_updates"]):
        raise RuntimeError("binding probe requested before pretraining complete")
    counts = np.load(prepared_dir() / "token_counts.npy")
    ids = select_binding_tokens(tok, counts)
    keys, vals = ids[:6], ids[6:12]
    train_fs, control_fs, sealed_fs = make_binding_factsets(keys, vals)
    model = Ark018GPT().to(device)
    model.load_state_dict(ckpt["model"])
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(.9,.95), eps=1e-8, weight_decay=.1)
    rng = random.Random(700000 + seed)
    traj = []
    streak = 0
    qualified_step = None
    snapshot = None
    for step in range(1, 1501):
        rows_all = binding_rows(tok, train_fs, "augmented", seed, step)
        batch = [rows_all[rng.randrange(len(rows_all))] for _ in range(64)]
        logits, ans = binding_batch(model, batch, device)
        loss = F.cross_entropy(logits, ans)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            c = eval_binding(model, tok, control_fs, device)
            traj.append({"step": step, **c})
            streak = streak + 1 if c["qualified"] else 0
            if streak >= 3:
                qualified_step = step
                snapshot = {"model": copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()}), "optimizer": copy.deepcopy(opt.state_dict())}
                break
    sealed_at_qualification = eval_binding(model, tok, sealed_fs, device) if qualified_step else None
    retention = None
    if qualified_step and snapshot:
        retention = {}
        stream_rng = random.Random(810000 + seed)
        index_stream = [stream_rng.randrange(400 * 3) for _ in range(600 * 64)]
        for label, lr in [("HIGH", 3e-4), ("LOW", 3e-6)]:
            m = Ark018GPT().to(device); m.load_state_dict(snapshot["model"])
            o = torch.optim.AdamW(m.parameters(), lr=lr, betas=(.9,.95), eps=1e-8, weight_decay=.1); o.load_state_dict(snapshot["optimizer"])
            for g in o.param_groups: g["lr"] = lr
            curve = []
            for step in range(1, 601):
                rows_all = binding_rows(tok, train_fs, "canonical", seed, step)
                ids_batch = index_stream[(step-1)*64:step*64]
                batch = [rows_all[i] for i in ids_batch]
                logits, ans = binding_batch(m, batch, device)
                loss = F.cross_entropy(logits, ans)
                o.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); o.step()
                if step % 100 == 0:
                    curve.append({"step": step, **eval_binding(m, tok, sealed_fs, device)})
            retention[label] = curve
            del m, o
    payload = {
        "schema": "arkenstone-ark018-binding-probe/v3",
        "seed": seed,
        "arm": arm,
        "token_ids": ids,
        "qualification_step": qualified_step,
        "control_trajectory": traj,
        "sealed_at_qualification": sealed_at_qualification,
        "retention_high_vs_low": retention,
        "claim_boundary": "controlled one-token temporary-binding acquisition/retention diagnostic",
    }
    safe_save_json(out_path, payload)
    del model, opt
    torch.cuda.empty_cache()
    return payload


def run_sciq(model, tok, device) -> dict:
    try:
        from datasets import load_dataset
        ds = load_dataset("allenai/sciq", split="test")
        indexed = []
        for i, row in enumerate(ds):
            h = hashlib.sha256((row.get("question", "") + str(i)).encode()).hexdigest()
            indexed.append((h, row))
        chosen = [r for _, r in sorted(indexed)[:100]]
        items = []
        for i, r in enumerate(chosen):
            choices = [r["correct_answer"], r["distractor1"], r["distractor2"], r["distractor3"]]
            rr = random.Random(920000 + i); rr.shuffle(choices)
            items.append({"id": f"sciq_{i}", "family":"SciQ_secondary", "prompt": r["question"], "choices": choices, "answer": choices.index(r["correct_answer"])})
        sc = score_mcq(model, tok, items, device)
        return {"status":"EXECUTED_SECONDARY", "dataset":"allenai/sciq:test", "n":len(items), "accuracy":sc["accuracy"], "contamination_status":"NOT_EXHAUSTIVELY_EXCLUDED"}
    except Exception as exc:
        return {"status":"NOT_AVAILABLE", "exception":repr(exc)}


def posttrain_evaluations(seed: int, arm: str, prepared, tok, device, enable_sciq: bool) -> dict:
    path = result_dir() / f"ARK-018_SEED_{seed}_{arm}_POSTTRAIN.json"
    if path.exists(): return json.loads(path.read_text())
    ckpt = torch.load(arm_checkpoint_path(seed, arm), map_location="cpu", weights_only=False)
    model = Ark018GPT().to(device); model.load_state_dict(ckpt["model"]); model.eval()
    payload = {
        "seed":seed, "arm":arm,
        "binding_probe": run_binding_probe_from_checkpoint(seed, arm, prepared, tok, device),
        "sciq": run_sciq(model, tok, device) if enable_sciq else {"status":"DISABLED"},
    }
    safe_save_json(path, payload)
    del model; torch.cuda.empty_cache()
    return payload


def summarize_campaign(prepared: dict) -> dict:
    rows = {}
    complete = True
    for seed in SEEDS:
        rows[str(seed)] = {}
        for arm in ARMS:
            p = result_dir() / f"ARK-018_SEED_{seed}_{arm}_PRETRAIN_RESULT.json"
            if not p.exists():
                complete = False
                continue
            x = json.loads(p.read_text())
            final = x["trajectory"][-1]
            rows[str(seed)][arm] = {
                "science_sealed_nll": final["science_sealed"]["nll"],
                "science_sealed_ppl": final["science_sealed"]["perplexity"],
                "birth_content_sealed": final["birth_content_sealed"]["accuracy"],
                "algorithmic_ood": final["algorithmic_ood"]["accuracy"],
                "birth_nll_diagnostic": final["birth_nll_diagnostic"]["nll"],
            }
    verdict = "INCOMPLETE"
    effects = {}
    if complete:
        diffs = []
        science_costs = []
        natural_diffs = []
        for seed in SEEDS:
            r = rows[str(seed)]
            c = r["BIRTH_REHEARSAL_10PCT"]; d = r["SCIENCE_REPLAY_10PCT_CONTROL"]
            a = r["SCIENCE_ONLY"]; b = r["BIRTH_NATURAL_2PCT"]
            diffs.append(c["birth_content_sealed"] - d["birth_content_sealed"])
            science_costs.append((c["science_sealed_nll"] - d["science_sealed_nll"]) / max(1e-12, d["science_sealed_nll"]))
            natural_diffs.append(b["birth_content_sealed"] - a["birth_content_sealed"])
        effects = {"birth_vs_replay_diffs":diffs, "science_relative_nll_costs":science_costs, "natural2pct_vs_science_birth_diffs":natural_diffs}
        if all(x >= .10 for x in diffs) and all(x <= .05 for x in science_costs):
            verdict = "BIRTH_CONTENT_INTERNALIZATION_WITHOUT_MAJOR_SCIENCE_COST"
        elif all(x > 0 for x in diffs) and any(x > .05 for x in science_costs):
            verdict = "BIRTH_CONTENT_INTERNALIZED_WITH_SCIENCE_TRADEOFF"
        elif all(abs(x) < .10 for x in diffs):
            verdict = "NO_LARGE_BIRTH_SPECIFIC_INTERNALIZATION_EFFECT"
        else:
            verdict = "MIXED_OR_SEED_DEPENDENT"
    return {"status":"COMPLETE" if complete else "PARTIAL", "pretraining":rows, "effects":effects, "primary_verdict":verdict}


def redteam_report(prepared: dict) -> dict:
    horizon = int(prepared["horizon_updates"])
    checks = {
        "science_hash_expected": prepared["science_sha256"] == EXPECTED_SCIENCE_SHA,
        "birth_hash_expected": prepared["birth_sha256"] == EXPECTED_BIRTH_SHA,
        "birth_natural_exact_2_per_100": count_sources("BIRTH_NATURAL_2PCT",100).get("birth",0)==2,
        "birth_rehearsal_exact_10_per_100": count_sources("BIRTH_REHEARSAL_10PCT",100).get("birth",0)==10,
        "science_replay_exact_10_per_100": count_sources("SCIENCE_REPLAY_10PCT_CONTROL",100).get("science_replay",0)==10,
        "equal_horizon_all_arms": True,
        "target_tokens_per_update": TARGET_TOKENS_PER_UPDATE == 8192,
        "two_independent_seeds_frozen": SEEDS == [31801,31902],
    }
    return {"status":"PASS" if all(checks.values()) else "FAIL", "checks":checks, "horizon":horizon, "limitations":[
        "Birth probes measure content internalization, not consciousness or identity.",
        "Algorithmic OOD and binding are narrow controlled transfer diagnostics, not broad reasoning.",
        "SciQ is secondary and contamination is not exhaustively excluded unless separately audited.",
        "This is a ~20-25M proxy and does not authorize production-scale claims."
    ]}


def smoke(science_path: Path, skip_full_hash: bool=False) -> dict:
    setup_reproducibility()
    static = validate_static_inputs(science_path)
    print("=== ARK-018 V3 CUDA SMOKE ===", flush=True)
    device = device_now()
    if not skip_full_hash:
        print("Smoke hashing full Drive file (critical binding gate)...", flush=True)
        sha = sha256_file(science_path)
        if sha != EXPECTED_SCIENCE_SHA:
            raise RuntimeError(f"SCIENCE_FILE_HASH_MISMATCH expected={EXPECTED_SCIENCE_SHA} actual={sha}")
    else:
        sha = "SKIPPED_BY_OPERATOR"
    schema = parquet_schema_receipt(science_path)
    if not schema["has_text"]: raise RuntimeError("Parquet text column missing")
    # Source schedule assertions.
    assert count_sources("BIRTH_NATURAL_2PCT",100)["birth"] == 2
    assert count_sources("BIRTH_REHEARSAL_10PCT",100)["birth"] == 10
    assert count_sources("SCIENCE_REPLAY_10PCT_CONTROL",100)["science_replay"] == 10

    # GPU model/backward + exact same-runtime restore/next-update test with synthetic token ids.
    torch.manual_seed(118018); torch.cuda.manual_seed_all(118018)
    model = Ark018GPT().to(device)
    nparams = parameter_count(model)
    if not (20_000_000 <= nparams <= 25_000_000):
        raise RuntimeError(f"parameter count outside frozen range: {nparams}")
    opt = optimizer_for(model); scaler = make_scaler(device)
    x = torch.randint(0, VOCAB_SIZE, (EFFECTIVE_SEQS, CONTEXT), device=device)
    y = torch.randint(0, VOCAB_SIZE, (EFFECTIVE_SEQS, CONTEXT), device=device)
    micro = 4
    snap = snapshot_exact(model,opt,scaler)
    t1 = one_update(model,opt,scaler,x,y,micro,3e-4)
    h1 = model_state_hash(model)
    restore_exact(snap,model,opt,scaler)
    t2 = one_update(model,opt,scaler,x,y,micro,3e-4)
    h2 = model_state_hash(model)
    if h1 != h2:
        raise RuntimeError("deterministic next-update reproduction failed")
    # Checkpoint write/read durability on Drive.
    smoke_ckpt = checkpoint_dir()/"SMOKE_TEMP.pt"
    save_training_checkpoint(smoke_ckpt,{"model":{k:v.detach().cpu() for k,v in model.state_dict().items()},"hash":h2})
    z = torch.load(smoke_ckpt,map_location="cpu",weights_only=False)
    if z["hash"] != h2: raise RuntimeError("Drive checkpoint readback mismatch")
    smoke_ckpt.unlink(missing_ok=True)
    payload = {
        "status":"PASS", "cuda":torch.cuda.get_device_name(0), "science_sha256":sha,
        "parquet":schema, "birth":static, "parameter_count":nparams,
        "effective_target_tokens":int(y.numel()), "microbatch_smoke":micro,
        "deterministic_next_update_hash":h2, "finite_loss":math.isfinite(t2["loss"]),
        "drive_checkpoint_roundtrip":True,
    }
    safe_save_json(result_dir()/"ARK-018_SMOKE_TEST.json",payload)
    print("ARK-018 V3 GPU SMOKE PASS", json.dumps(payload,indent=2), flush=True)
    return payload


def package_results() -> Path:
    out = result_dir()/"ARKENSTONE_ARK018_SCIENCE_BIRTH_RESULTS.zip"
    with zipfile.ZipFile(out,"w",compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(result_dir().glob("*.json")):
            zf.write(p,p.name)
        for p in [HERE/"EXECUTION_V3_ADDENDUM.md", HERE/"SCIENCE_BIRTH_PERIODIC_MIXTURE_ADDENDUM.md", PROBE_PATH]:
            if p.exists(): zf.write(p,p.name)
    manifest = {p.name:sha256_file(p) for p in result_dir().glob("*.json")}
    safe_save_json(result_dir()/"ARK-018_ZIP_MANIFEST.json",{"zip":out.name,"zip_sha256":sha256_file(out),"json_sha256":manifest})
    # Recreate zip including manifest itself.
    with zipfile.ZipFile(out,"w",compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(result_dir().glob("*.json")): zf.write(p,p.name)
        for p in [HERE/"EXECUTION_V3_ADDENDUM.md", HERE/"SCIENCE_BIRTH_PERIODIC_MIXTURE_ADDENDUM.md", PROBE_PATH]:
            if p.exists(): zf.write(p,p.name)
    print("RESULT ZIP:", out, out.stat().st_size, "bytes", flush=True)
    return out


def run_all(science_path: Path, enable_sciq: bool, force_prepare: bool=False):
    setup_reproducibility()
    device = device_now()
    prepared = prepare_all(science_path, force=force_prepare)
    # Smoke after preparation can skip a second full-GB hash because prepared binding receipt already contains the exact hash.
    smoke(science_path, skip_full_hash=True)
    prepared, tok, bufs = load_prepared()
    # Calibrate once on exact model architecture and prepared science data.
    tmp = Ark018GPT().to(device)
    micro = microbatch_calibration(tmp,device,bufs["train"])
    del tmp; torch.cuda.empty_cache()
    safe_save_json(result_dir()/"ARK-018_RUNTIME_RECEIPT.json",{
        "status":"READY", "cuda":torch.cuda.get_device_name(0), "microbatch":micro,
        "gradient_accumulation":EFFECTIVE_SEQS//micro, "effective_sequences":EFFECTIVE_SEQS,
        "horizon":prepared["horizon_updates"]
    })
    for seed in SEEDS:
        for arm in ARMS:
            train_arm(seed,arm,prepared,tok,bufs,device,micro)
            posttrain_evaluations(seed,arm,prepared,tok,device,enable_sciq)
    summary = summarize_campaign(prepared)
    safe_save_json(result_dir()/"ARK-018_RESULT.json",summary)
    rt = redteam_report(prepared)
    safe_save_json(result_dir()/"ARK-018_REDTEAM.json",rt)
    package_results()
    print("ARK-018 COMPLETE", json.dumps(summary,indent=2), flush=True)


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--mode", choices=["prepare","smoke","all","package"], default="all")
    p.add_argument("--science-path", default=str(SCIENCE_DRIVE_PATH))
    p.add_argument("--force-prepare", action="store_true")
    p.add_argument("--enable-sciq", action="store_true")
    p.add_argument("--skip-full-hash-smoke", action="store_true")
    p.add_argument("--expected-head", default="")
    return p.parse_args()


def main():
    args=parse_args()
    head=git_head()
    if args.expected_head and head!=args.expected_head:
        raise RuntimeError(f"checked-out HEAD {head} != expected {args.expected_head}")
    science=Path(args.science_path)
    DRIVE_ROOT.mkdir(parents=True,exist_ok=True)
    try:
        if args.mode=="prepare": prepare_all(science,force=args.force_prepare)
        elif args.mode=="smoke": smoke(science,skip_full_hash=args.skip_full_hash_smoke)
        elif args.mode=="all": run_all(science,args.enable_sciq,args.force_prepare)
        else: package_results()
        return 0
    except Exception as exc:
        try:
            safe_save_json(result_dir()/"ARK-018_FAILURE_RECEIPT.json",{
                "status":"FAILED","exception_type":type(exc).__name__,"exception":str(exc),"traceback":traceback.format_exc()
            })
            package_results()
        except Exception:
            pass
        raise

if __name__=="__main__":
    raise SystemExit(main())
