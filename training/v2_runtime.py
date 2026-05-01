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
    TEACHER_REASONING_V2_FILE,
    V2_BRAIN_CHECKPOINT,
    V2_IDENTITY_CHECKPOINT,
    V2_OUROBOROS_CHECKPOINT,
    V3_TOKENIZER_FILE,
    ensure_dirs,
    get_dataset_file,
    get_identity_file,
    get_v2_checkpoint,
)
from anra_brain import CausalTransformerV2
from tokenizer.tokenizer_adapter import TokenizerAdapter
from training.v2_config import V2_MODEL, V2_REPORT_FILES
from runtime.drive_session_manager import DriveSessionManager


ensure_dirs()
EXPECTED_TOKENIZER_VOCAB_SIZE = 8192
EXPECTED_SPECIAL_TOKENS = ["<unk>", "<pad>", "<bos>", "<eos>"]

DRIVE_SESSION_MANAGER = DriveSessionManager(DRIVE_DIR)


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


def _read_step(path: Path) -> int:
    """Safely read step number from checkpoint without loading full model."""
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        return int(ckpt.get("step", ckpt.get("global_step", 0)))
    except Exception:
        return 0


def restore_v2_artifact(name: str = "brain") -> bool:
    """
    Check Drive for checkpoint. If found, copy to local output dir.
    Returns True if restored, False if starting fresh.
    """
    local_map = {
        "brain": get_v2_checkpoint("brain"),
        "identity": get_v2_checkpoint("identity"),
        "ouroboros": get_v2_checkpoint("ouroboros"),
        "tokenizer": V3_TOKENIZER_FILE,
        "eval_summary": v2_report_path("eval_summary"),
    }
    local_path = local_map.get(name, get_v2_checkpoint(name))
    local_path.parent.mkdir(parents=True, exist_ok=True)

    drive_filenames = {
        "brain": "anra_v2_brain.pt",
        "identity": "anra_v2_identity.pt",
        "ouroboros": "anra_v2_ouroboros.pt",
        "tokenizer": "tokenizer_v3.json",
    }
    drive_file = DRIVE_V2_CHECKPOINTS / drive_filenames.get(name, f"anra_v2_{name}.pt")
    drive_root_file = DRIVE_DIR / drive_filenames.get(name, f"anra_v2_{name}.pt")

    source = None
    if drive_file.exists():
        source = drive_file
    elif drive_root_file.exists():
        source = drive_root_file

    if source is None:
        print(f"[Restore] {name}: not on Drive — will start fresh")
        return False

    step = _read_step(local_path)
    print(f"[Restore] {name}: restored from Drive (step={step})")
    return True


def sync_to_drive(name: str = "brain") -> bool:
    """
    Copy local checkpoint to Drive. Always overwrites same file.
    Never creates new files. Returns True on success.
    """
    local_map = {
        "brain": get_v2_checkpoint("brain"),
        "identity": get_v2_checkpoint("identity"),
        "ouroboros": get_v2_checkpoint("ouroboros"),
        "tokenizer": V3_TOKENIZER_FILE,
        "eval_summary": v2_report_path("eval_summary"),
    }
    local_path = local_map.get(name, get_v2_checkpoint(name))
    if not local_path.exists():
        print(f"[Drive] {name}: local file not found, skipping")
        return False

    drive_filenames = {
        "brain": "anra_v2_brain.pt",
        "identity": "anra_v2_identity.pt",
        "ouroboros": "anra_v2_ouroboros.pt",
        "tokenizer": "tokenizer_v3.json",
        "eval_summary": "anra_v2_eval_summary.json",
    }
    drive_filename = drive_filenames.get(name, f"anra_v2_{name}.pt")

    DRIVE_V2_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    drive_target = DRIVE_V2_CHECKPOINTS / drive_filename
    drive_root = DRIVE_DIR / drive_filename

    try:
        DRIVE_SESSION_MANAGER.save_family(name, local_path)
        step = _read_step(local_path)
        size_kb = local_path.stat().st_size // 1024
        print(f"[Drive] {name}: saved (step={step}, {size_kb}KB)")
        return True
    except Exception as e:
        print(f"[Drive] {name}: save failed ({e})")
        return False


def sync_v2_artifacts(
    checkpoint_path: Path,
    *,
    tokenizer_path: Path | None = None,
    extra_paths: list[Path] | None = None,
) -> None:
    del checkpoint_path, tokenizer_path
    sync_to_drive("brain")
    sync_to_drive("tokenizer")
    for extra in extra_paths or []:
        if extra.name == "v2_eval_summary.json":
            target = get_v2_checkpoint("brain").parent / "anra_v2_eval_summary.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(extra, target)
            sync_to_drive("eval_summary")


def _collect_tokenizer_texts(dataset_path: Path) -> list[str]:
    texts = [dataset_path.read_text(encoding="utf-8", errors="replace")]
    identity_path = get_identity_file()
    if identity_path.exists():
        texts.append(identity_path.read_text(encoding="utf-8", errors="replace"))
    teacher_path = TEACHER_REASONING_V2_FILE
    if teacher_path.exists():
        lines = teacher_path.read_text(encoding="utf-8", errors="replace").splitlines()
        texts.extend(line for line in lines if line.strip())
    return texts


def load_or_build_v2_tokenizer(
    *,
    dataset_path: Path | None = None,
    vocab_size: int = V2_MODEL.vocab_size,
) -> TokenizerAdapter:
    dataset_path = dataset_path or get_dataset_file()
    local = V3_TOKENIZER_FILE
    if local.exists():
        tokenizer = SubwordTokenizer.load(local)
        assert_tokenizer_contract(local, tokenizer)
        return tokenizer
    restored = restore_v2_artifact("tokenizer")
    if restored and local.exists():
        tokenizer = SubwordTokenizer.load(local)
        assert_tokenizer_contract(local, tokenizer)
        return tokenizer

    texts = _collect_tokenizer_texts(dataset_path)
    print(f"[build_brain] Building tokenizer_v3 from {dataset_path} ...", flush=True)
    tokenizer = SubwordTokenizer.train_from_texts(texts, vocab_size=vocab_size)
    tokenizer.save(local)
    assert_tokenizer_contract(local, tokenizer)
    try:
        drive_tok = DRIVE_DIR / "v2" / local.name
        drive_tok.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local, drive_tok)
    except Exception:
        pass
    print(
        f"[build_brain] tokenizer_v3 built + mirrored to Drive. vocab_size={tokenizer.vocab_size}",
        flush=True,
    )
    return tokenizer


def assert_tokenizer_contract(path: Path, tokenizer: SubwordTokenizer) -> None:
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    vocab_size = int(meta.get("vocab_size", tokenizer.vocab_size))
    special_tokens = list(meta.get("special_tokens", tokenizer.special_tokens))
    if vocab_size != EXPECTED_TOKENIZER_VOCAB_SIZE:
        raise AssertionError(
            f"Tokenizer contract mismatch: vocab_size={vocab_size}, expected={EXPECTED_TOKENIZER_VOCAB_SIZE} "
            f"(meta={meta_path})"
        )
    if special_tokens != EXPECTED_SPECIAL_TOKENS:
        raise AssertionError(
            f"Tokenizer contract mismatch: special_tokens={special_tokens}, expected={EXPECTED_SPECIAL_TOKENS} "
            f"(meta={meta_path})"
        )


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
        kind = "brain"
        if "identity" in ckpt.name:
            kind = "identity"
        elif "ouroboros" in ckpt.name:
            kind = "ouroboros"
        restored = restore_v2_artifact(kind)
        if restored:
            ckpt = get_v2_checkpoint(kind)
    if not ckpt.exists():
        return state

    blob = torch.load(ckpt, map_location=device, weights_only=False)
    model_state = blob.get("model_state_dict", blob.get("model", blob)) if isinstance(blob, dict) else blob
    model.load_state_dict(model_state, strict=strict)
    if isinstance(blob, dict):
        if optimizer is not None:
            try:
                optimizer.load_state_dict(blob.get("optimizer_state_dict", blob.get("optimizer", {})))
            except Exception:
                pass
        if scheduler is not None:
            try:
                scheduler.load_state_dict(blob.get("scheduler_state_dict", blob.get("scheduler", {})))
            except Exception:
                pass
        if mp_trainer is not None:
            try:
                scaler_state = blob.get("scaler_state_dict", blob.get("scaler"))
                if scaler_state:
                    mp_trainer.load_state_dict(scaler_state)
            except Exception:
                pass
        state["global_step"] = int(blob.get("global_step", blob.get("step", 0)))
        state["epoch"] = int(blob.get("epoch", 0))
        state["best_loss"] = float(blob.get("best_loss", float("inf")))
    state["loaded"] = True
    return state


@torch.no_grad()
def generate_text(
    model: CausalTransformerV2,
    tokenizer: TokenizerAdapter,
    prompt: str,
    *,
    device: torch.device,
    max_new_tokens: int = 96,
    temperature: float = 0.9,
    top_k: int = 40,
) -> str:
    model.eval()
    special = tokenizer.special_ids()
    ids = [special["<bos>"]] + tokenizer.encode(prompt)
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
        if int(next_token.item()) == special["<eos>"]:
            break
    prompt_token_count = 1 + len(tokenizer.encode(prompt))
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
