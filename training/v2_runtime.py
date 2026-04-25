from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path

import torch

from anra_paths import (
    DRIVE_DIR,
    DRIVE_V2_CHECKPOINTS,
    OUTPUT_V2_DIR,
    ROOT,
    V2_BRAIN_CHECKPOINT,
    V2_IDENTITY_CHECKPOINT,
    V2_OUROBOROS_CHECKPOINT,
    V2_TOKENIZER_FILE,
    ensure_dirs,
    get_dataset_file,
    get_identity_file,
)
from anra_brain import CausalTransformerV2
from tokenizer.subword_tokenizer import SubwordTokenizer
from training.v2_config import V2_MODEL, V2_REPORT_FILES


ensure_dirs()


def canonical_v2_checkpoint(kind: str = "brain") -> Path:
    mapping = {
        "brain": V2_BRAIN_CHECKPOINT,
        "identity": V2_IDENTITY_CHECKPOINT,
        "ouroboros": V2_OUROBOROS_CHECKPOINT,
    }
    return mapping.get(kind, V2_BRAIN_CHECKPOINT)


def v2_output_file(name: str) -> Path:
    OUTPUT_V2_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_V2_DIR / name


def v2_report_path(key: str) -> Path:
    filename = V2_REPORT_FILES.get(key, key)
    return v2_output_file(filename)


def atomic_save(payload: dict, output_path: Path, *, drive_dir: Path | None = DRIVE_V2_CHECKPOINTS) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(output_path)
    if drive_dir is not None:
        try:
            drive_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output_path, drive_dir / output_path.name)
        except Exception:
            pass


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def sync_v2_artifacts(
    checkpoint_path: Path,
    *,
    tokenizer_path: Path | None = None,
    extra_paths: list[Path] | None = None,
) -> None:
    try:
        DRIVE_V2_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint_path, DRIVE_V2_CHECKPOINTS / checkpoint_path.name)
        tok = tokenizer_path or V2_TOKENIZER_FILE
        meta = tok.with_suffix(tok.suffix + ".meta.json")
        if tok.exists():
            drive_tok = DRIVE_DIR / "v2" / tok.name
            drive_tok.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tok, drive_tok)
            if meta.exists():
                shutil.copy2(meta, drive_tok.with_suffix(drive_tok.suffix + ".meta.json"))
        for extra in extra_paths or []:
            if extra.exists():
                target = DRIVE_DIR / "v2" / extra.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(extra, target)
    except Exception:
        pass


def restore_v2_artifact(local_path: Path, *, remote_name: str | None = None) -> Path | None:
    if local_path.exists():
        return local_path
    remote = (DRIVE_V2_CHECKPOINTS / (remote_name or local_path.name))
    if "tokenizer_v2" in (remote_name or local_path.name):
        remote = DRIVE_DIR / "v2" / (remote_name or local_path.name)
    if remote.exists():
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(remote, local_path)
        meta_remote = remote.with_suffix(remote.suffix + ".meta.json")
        meta_local = local_path.with_suffix(local_path.suffix + ".meta.json")
        if meta_remote.exists():
            shutil.copy2(meta_remote, meta_local)
        return local_path
    return None


def _collect_tokenizer_texts(dataset_path: Path) -> list[str]:
    texts = [dataset_path.read_text(encoding="utf-8", errors="replace")]
    identity_path = get_identity_file()
    if identity_path.exists():
        texts.append(identity_path.read_text(encoding="utf-8", errors="replace"))
    teacher_path = ROOT / "training_data" / "teacher_reasoning_v2.jsonl"
    if teacher_path.exists():
        lines = teacher_path.read_text(encoding="utf-8", errors="replace").splitlines()
        texts.extend(line for line in lines if line.strip())
    return texts


def load_or_build_v2_tokenizer(
    *,
    dataset_path: Path | None = None,
    vocab_size: int = V2_MODEL.vocab_size,
) -> SubwordTokenizer:
    dataset_path = dataset_path or get_dataset_file()
    local = V2_TOKENIZER_FILE
    if local.exists():
        return SubwordTokenizer.load(local)
    restored = restore_v2_artifact(local)
    if restored is not None and restored.exists():
        return SubwordTokenizer.load(restored)

    texts = _collect_tokenizer_texts(dataset_path)
    print(f"[build_brain] Building tokenizer_v2 from {dataset_path} ...", flush=True)
    tokenizer = SubwordTokenizer.train_from_texts(texts, vocab_size=vocab_size)
    tokenizer.save(local)
    try:
        drive_tok = DRIVE_DIR / "v2" / local.name
        drive_tok.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local, drive_tok)
        meta = local.with_suffix(local.suffix + ".meta.json")
        if meta.exists():
            shutil.copy2(meta, drive_tok.with_suffix(drive_tok.suffix + ".meta.json"))
    except Exception:
        pass
    print(
        f"[build_brain] tokenizer_v2 built + mirrored to Drive. vocab_size={tokenizer.vocab_size}",
        flush=True,
    )
    return tokenizer


def build_v2_model(*, vocab_size: int, block_size: int = V2_MODEL.block_size) -> CausalTransformerV2:
    return CausalTransformerV2(
        vocab_size=vocab_size,
        n_embd=V2_MODEL.n_embd,
        n_head=V2_MODEL.n_head,
        n_layer=V2_MODEL.n_layer,
        block_size=block_size,
        rms_norm_eps=V2_MODEL.rms_norm_eps,
        dropout=V2_MODEL.dropout,
    )


def load_checkpoint(
    model: CausalTransformerV2,
    optimizer: torch.optim.Optimizer | None,
    scheduler,
    mp_trainer,
    checkpoint_path: Path,
    *,
    device: torch.device,
    strict: bool = False,
) -> dict[str, float | int | bool | str]:
    state = {
        "loaded": False,
        "global_step": 0,
        "epoch": 0,
        "best_loss": float("inf"),
    }
    ckpt = checkpoint_path
    if not ckpt.exists():
        restored = restore_v2_artifact(ckpt)
        if restored is not None:
            ckpt = restored
    if not ckpt.exists():
        return state

    blob = torch.load(ckpt, map_location=device, weights_only=False)
    model_state = blob.get("model_state_dict", blob) if isinstance(blob, dict) else blob
    model.load_state_dict(model_state, strict=strict)
    if isinstance(blob, dict):
        if optimizer is not None:
            try:
                optimizer.load_state_dict(blob.get("optimizer_state_dict", {}))
            except Exception:
                pass
        if scheduler is not None:
            try:
                scheduler.load_state_dict(blob.get("scheduler_state_dict", {}))
            except Exception:
                pass
        if mp_trainer is not None:
            try:
                scaler_state = blob.get("scaler_state_dict")
                if scaler_state:
                    mp_trainer.load_state_dict(scaler_state)
            except Exception:
                pass
        state["global_step"] = int(blob.get("global_step", 0))
        state["epoch"] = int(blob.get("epoch", 0))
        state["best_loss"] = float(blob.get("best_loss", float("inf")))
    state["loaded"] = True
    return state


@torch.no_grad()
def generate_text(
    model: CausalTransformerV2,
    tokenizer: SubwordTokenizer,
    prompt: str,
    *,
    device: torch.device,
    max_new_tokens: int = 96,
    temperature: float = 0.9,
    top_k: int = 40,
) -> str:
    model.eval()
    ids = [tokenizer.bos_token_id] + tokenizer.encode(prompt, add_special_tokens=False)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        x_cond = x[:, -model.block_size :]
        logits, _ = model(x_cond)
        next_logits = logits[:, -1, :] / max(temperature, 1e-4)
        if top_k > 0:
            values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits = next_logits.masked_fill(next_logits < values[:, [-1]], float("-inf"))
        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_token], dim=1)
        if int(next_token.item()) == tokenizer.eos_token_id:
            break
    prompt_token_count = 1 + len(tokenizer.encode(prompt, add_special_tokens=False))
    answer_ids = x[0].tolist()[prompt_token_count:]
    return tokenizer.decode(answer_ids).strip()


def model_summary(model: torch.nn.Module) -> dict[str, int]:
    return {
        "parameters": sum(param.numel() for param in model.parameters()),
        "trainable_parameters": sum(param.numel() for param in model.parameters() if param.requires_grad),
    }


def load_session_state() -> dict[str, object]:
    path = v2_report_path("session_state")
    if not path.exists():
        return {"successful_sessions": 0, "eval_scores": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"successful_sessions": 0, "eval_scores": []}


def update_session_state(*, eval_score: float | None = None) -> dict[str, object]:
    state = load_session_state()
    state["successful_sessions"] = int(state.get("successful_sessions", 0)) + 1
    scores = list(state.get("eval_scores", []))
    if eval_score is not None and not math.isnan(eval_score):
        scores.append({"score": float(eval_score), "ts": time.time()})
    state["eval_scores"] = scores[-12:]
    write_json(v2_report_path("session_state"), state)
    return state
