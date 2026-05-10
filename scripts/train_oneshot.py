from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path


REJECT_WORDS = [
    "ChatGPT",
    "GPT-4",
    "GPT4",
    "Claude",
    "Anthropic",
    "OpenAI",
    "As an AI",
    "I am an AI",
]

_ARGS = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", default="1b", choices=["25m", "1b"])
    parser.add_argument("--session-minutes", type=int, default=150)
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip data download if already done",
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Only download data, do not train",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def install_missing_dependencies() -> None:
    required = ["datasets", "boto3", "psutil", "tokenizers"]
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            try:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        pkg,
                        "-q",
                        "--only-binary=:all:",
                    ],
                    check=False,
                    timeout=180,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass


def gpu_check(args: argparse.Namespace):
    import torch

    print("=" * 60)
    print("AN-RA ONESHOT TRAINING")
    print("=" * 60)

    if not torch.cuda.is_available():
        if args.data_only:
            print("CUDA GPU: not found")
            print("Mode: DATA ONLY - GPU check skipped")
            print(f"Mode: {args.model_size.upper()}")
            print(f"Time: {args.session_minutes} minutes")
            return None, 0.0
        print("ERROR: No CUDA GPU found. This script requires a GPU.")
        sys.exit(1)

    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1024**3
    print(f"GPU:  {props.name}")
    print(f"VRAM: {vram_gb:.1f} GB")
    print(f"Mode: {args.model_size.upper()}")
    print(f"Time: {args.session_minutes} minutes")

    if args.model_size == "1b" and vram_gb < 20:
        print(f"ERROR: 1B model needs 20GB+ VRAM. You have {vram_gb:.1f}GB.")
        print("Switch to --model-size 25m or rent a bigger GPU.")
        sys.exit(1)

    if args.model_size == "1b" and vram_gb < 40:
        print(f"WARNING: {vram_gb:.1f}GB VRAM is tight.")
        print("Gradient checkpointing and batch_size=1 will be forced.")
    return props, vram_gb


def _reasoning_path(target_dir: str | Path) -> Path:
    return Path(target_dir) / "reasoning.jsonl"


def _file_mb(path: Path) -> float:
    try:
        return path.stat().st_size / 1024**2
    except OSError:
        return 0.0


def _too_large(path: Path, limit_mb: float = 350.0) -> bool:
    return path.exists() and _file_mb(path) >= limit_mb


def _contains_reject_word(text: str) -> bool:
    lowered = text.lower()
    return any(word.lower() in lowered for word in REJECT_WORDS)


def _append_jsonl(path: Path, rows) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _dataset_iter(ds):
    try:
        return iter(ds)
    except TypeError:
        return iter(list(ds))


def _openhermes_turns(item: dict) -> tuple[str, str] | None:
    conversations = item.get("conversations") or item.get("messages") or []
    if not isinstance(conversations, list):
        return None
    user_turn = ""
    assistant_turn = ""
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("from") or turn.get("role") or "").lower()
        text = str(turn.get("value") or turn.get("content") or "").strip()
        if not text:
            continue
        if role in {"human", "user"} and not user_turn:
            user_turn = text
        elif role in {"gpt", "assistant"} and user_turn:
            assistant_turn = text
            break
    if user_turn and assistant_turn:
        return user_turn, assistant_turn
    return None


def _print_data_summary(path: Path, counts: dict[str, int]) -> None:
    total = sum(counts.values())
    size_mb = _file_mb(path)
    chars = path.stat().st_size if path.exists() else 0
    estimated_tokens = int(chars * 0.25)
    print(f"Total examples: {total}")
    print(f"File size: {size_mb:.1f} MB")
    print(f"Estimated tokens: ~{estimated_tokens:,}")


def download_data(target_dir: str) -> dict:
    args = _ARGS
    path = _reasoning_path(target_dir)
    counts: dict[str, int] = {}
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    if (
        args is not None
        and args.skip_download
        and path.exists()
        and path.stat().st_size > 10 * 1024**2
    ):
        print(f"Skipping download; found existing {path} ({_file_mb(path):.1f} MB)")
        existing_count = 0
        try:
            with path.open(encoding="utf-8") as fh:
                for _ in fh:
                    existing_count += 1
        except Exception:
            existing_count = 0
        counts[path.name] = existing_count
        _print_data_summary(path, counts)
        return counts

    try:
        from datasets import load_dataset
    except Exception as exc:
        print(f"Data download skipped: datasets import failed ({exc})")
        _print_data_summary(path, counts)
        return counts

    print("\nDOWNLOADING TRAINING DATA")
    print("-" * 60)

    # DATASET A - OpenHermes 2.5
    if not _too_large(path):
        try:
            print("Dataset A: OpenHermes 2.5")
            ds = load_dataset(
                "teknium/OpenHermes-2.5",
                split="train",
                trust_remote_code=True,
                streaming=True,
            )

            def rows():
                taken = 0
                for item in _dataset_iter(ds):
                    turns = _openhermes_turns(item)
                    if turns is None:
                        continue
                    prompt, response = turns
                    if _contains_reject_word(response):
                        continue
                    yield {
                        "prompt": prompt,
                        "response": response,
                        "source": "openhermes",
                        "task_type": "reasoning",
                    }
                    taken += 1
                    if taken >= 40000:
                        break

            counts["openhermes"] = _append_jsonl(path, rows())
            print(f"  added {counts['openhermes']:,} examples ({_file_mb(path):.1f} MB)")
        except Exception as exc:
            print(f"  skipped OpenHermes: {exc}")

    # DATASET B - GSM8K
    if not _too_large(path):
        try:
            print("Dataset B: GSM8K")
            ds = load_dataset("openai/gsm8k", "main", split="train", trust_remote_code=True)

            def rows():
                for item in _dataset_iter(ds):
                    yield {
                        "prompt": str(item.get("question", "")).strip(),
                        "response": str(item.get("answer", "")).strip(),
                        "source": "gsm8k",
                        "task_type": "math",
                    }

            counts["gsm8k"] = _append_jsonl(path, rows())
            print(f"  added {counts['gsm8k']:,} examples ({_file_mb(path):.1f} MB)")
        except Exception as exc:
            print(f"  skipped GSM8K: {exc}")

    # DATASET C - MetaMathQA
    if not _too_large(path):
        try:
            print("Dataset C: MetaMathQA")
            ds = load_dataset(
                "meta-math/MetaMathQA",
                split="train",
                trust_remote_code=True,
                streaming=True,
            )

            def rows():
                taken = 0
                for item in _dataset_iter(ds):
                    prompt = str(item.get("query", "")).strip()
                    response = str(item.get("response", "")).strip()
                    if not prompt or not response:
                        continue
                    yield {
                        "prompt": prompt,
                        "response": response,
                        "source": "metamath",
                        "task_type": "math",
                    }
                    taken += 1
                    if taken >= 30000:
                        break

            counts["metamath"] = _append_jsonl(path, rows())
            print(f"  added {counts['metamath']:,} examples ({_file_mb(path):.1f} MB)")
        except Exception as exc:
            print(f"  skipped MetaMathQA: {exc}")

    # DATASET D - WizardCoder
    if not _too_large(path):
        try:
            print("Dataset D: WizardCoder")
            ds = load_dataset(
                "WizardLMTeam/WizardCoder_evol_instruct_110k",
                split="train",
                trust_remote_code=True,
                streaming=True,
            )

            def rows():
                taken = 0
                for item in _dataset_iter(ds):
                    prompt = str(item.get("instruction", "")).strip()
                    response = str(item.get("output", "")).strip()
                    if not prompt or not response:
                        continue
                    yield {
                        "prompt": prompt,
                        "response": response,
                        "source": "wizardcoder",
                        "task_type": "code",
                    }
                    taken += 1
                    if taken >= 15000:
                        break

            counts["wizardcoder"] = _append_jsonl(path, rows())
            print(f"  added {counts['wizardcoder']:,} examples ({_file_mb(path):.1f} MB)")
        except Exception as exc:
            print(f"  skipped WizardCoder: {exc}")

    _print_data_summary(path, counts)
    return counts


def _count_reasoning_examples(reasoning_path: str) -> int:
    path = Path(reasoning_path)
    if not path.exists():
        return 0
    count = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                obj = json.loads(line)
                if str(obj.get("prompt", "")).strip() and str(obj.get("response", "")).strip():
                    count += 1
            except Exception:
                continue
    return count


def patch_data_mix(reasoning_path: str, repo_root: str) -> int:
    v2_mix = Path(repo_root) / "training" / "v2_data_mix.py"
    examples_loaded = _count_reasoning_examples(reasoning_path)
    if not v2_mix.exists():
        print(f"Data mix patch skipped: missing {v2_mix}")
        return examples_loaded

    try:
        text = v2_mix.read_text(encoding="utf-8")
    except Exception as exc:
        print(f"Data mix patch skipped: read failed ({exc})")
        return examples_loaded

    changed = False
    if "_load_reasoning_examples" not in text:
        text += '''


def _load_reasoning_examples(
    path: str = "training_data/reasoning.jsonl",
    max_examples: int = 60000,
) -> list:
    import json
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return []
    examples = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                pr = obj.get("prompt","").strip()
                re = obj.get("response","").strip()
                if pr and re and len(re) > 20:
                    examples.append({
                        "text": f"{pr}\\n\\n{re}",
                        "source": obj.get("source","reasoning"),
                    })
                    if len(examples) >= max_examples:
                        break
            except Exception:
                continue
    return examples
'''
        changed = True

    if "reasoning_examples = [" not in text and "def build_v2_training_examples" in text:
        needle = "    frontier_examples = _load_frontier_dfc_examples(dataset_path)\n"
        replacement = needle + '''
    reasoning_raw = _load_reasoning_examples(str(dataset_path.parent / "reasoning.jsonl"))
    reasoning_examples = []
    for item in reasoning_raw:
        raw_text = str(item.get("text", "")).strip()
        if not raw_text:
            continue
        if "\\n\\n" in raw_text:
            prompt, answer = raw_text.split("\\n\\n", 1)
        else:
            prompt, answer = raw_text[: len(raw_text) // 2], raw_text[len(raw_text) // 2 :]
        reasoning_examples.append(
            TrainingExample(
                bucket="reasoning",
                prompt=prompt.strip(),
                answer=answer.strip(),
                source=str(item.get("source", "reasoning")),
                weight=1.2,
                metadata={"task_type": "reasoning"},
            )
        )
'''
        if needle in text:
            text = text.replace(needle, replacement, 1)
            changed = True

    if 'requested_counts["reasoning"]' not in text and "def build_v2_training_examples" in text:
        needle = '    mixed = []\n'
        replacement = '''
    requested_counts["reasoning"] = 0
    if reasoning_examples:
        reasoning_target = min(int(total_examples * 0.20), len(reasoning_examples))
        remaining = reasoning_target
        for bucket_name in ("replay", "teacher", "symbolic", "own"):
            take = min(requested_counts.get(bucket_name, 0), remaining)
            requested_counts[bucket_name] = requested_counts.get(bucket_name, 0) - take
            remaining -= take
            if remaining <= 0:
                break
        requested_counts["reasoning"] = reasoning_target - remaining

    mixed = []
'''
        if needle in text:
            text = text.replace(needle, replacement, 1)
            changed = True

    if 'requested_counts.get("reasoning", 0)' not in text and "def build_v2_training_examples" in text:
        needle = '    mixed.extend(_sample_bucket(rng, frontier_examples, requested_counts["frontier_dfc"]))\n'
        replacement = needle + '    mixed.extend(_sample_bucket(rng, reasoning_examples, requested_counts.get("reasoning", 0)))\n'
        if needle in text:
            text = text.replace(needle, replacement, 1)
            changed = True

    if changed:
        try:
            v2_mix.write_text(text, encoding="utf-8")
            print(f"Patched data mix with reasoning examples: {examples_loaded:,} available")
        except Exception as exc:
            print(f"Data mix patch failed: {exc}")
    else:
        print(f"Data mix already includes reasoning support: {examples_loaded:,} available")
    return examples_loaded


def _tokenize_to_blocks(tokenizer, text: str, token_buffer: list[int], blocks: list[list[int]], block_size: int) -> None:
    if not text:
        return
    token_buffer.extend(tokenizer.encode(text, add_special_tokens=False))
    while len(token_buffer) >= block_size and len(blocks) < 200000:
        blocks.append(token_buffer[:block_size])
        del token_buffer[:block_size]


def _read_txt_blocks(path: Path, tokenizer, blocks: list[list[int]], block_size: int) -> None:
    token_buffer: list[int] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        while len(blocks) < 200000:
            chunk = fh.read(64 * 1024)
            if not chunk:
                break
            _tokenize_to_blocks(tokenizer, chunk, token_buffer, blocks, block_size)


def _jsonl_text(obj: dict) -> str:
    text = str(obj.get("text", "")).strip()
    if text:
        return text
    prompt = str(obj.get("prompt", "")).strip()
    response = str(obj.get("response", "") or obj.get("answer", "")).strip()
    if prompt and response:
        return f"{prompt}\n\n{response}"
    return ""


def _read_jsonl_blocks(path: Path, tokenizer, blocks: list[list[int]], block_size: int) -> None:
    token_buffer: list[int] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if len(blocks) >= 200000:
                break
            try:
                obj = json.loads(line)
            except Exception:
                continue
            _tokenize_to_blocks(tokenizer, _jsonl_text(obj), token_buffer, blocks, block_size)


def build_token_dataset(tokenizer, repo_root, block_size=512) -> list:
    root = Path(repo_root)
    blocks: list[list[int]] = []
    files = [
        root / "training_data" / "anra_training.txt",
        root / "training_data" / "reasoning.jsonl",
        root / "training_data" / "frontier_dfc.jsonl",
    ]

    print("\nBUILDING TOKEN DATASET")
    print("-" * 60)
    for path in files:
        if not path.exists():
            print(f"Skipping missing data file: {path}")
            continue
        before = len(blocks)
        try:
            if path.suffix == ".txt":
                _read_txt_blocks(path, tokenizer, blocks, block_size)
            elif path.suffix == ".jsonl":
                _read_jsonl_blocks(path, tokenizer, blocks, block_size)
        except Exception as exc:
            print(f"Skipping {path}: {exc}")
            continue
        print(f"{path.name}: {len(blocks) - before:,} blocks")
        if len(blocks) >= 200000:
            break

    random.seed(42)
    random.shuffle(blocks)
    tokens = len(blocks) * block_size
    estimated_minutes = tokens / 5000 / 60 if tokens else 0
    print(f"Total blocks: {len(blocks):,}")
    print(f"Tokens: ~{tokens:,}")
    print(f"Estimated training time at 5000 tok/sec: {estimated_minutes:.1f} minutes")
    return blocks


def save_checkpoint_and_report(
    *,
    args: argparse.Namespace,
    model,
    optimizer,
    step: int,
    epoch: int,
    best_loss: float,
    loss_history: list[float],
    accum_steps: int,
) -> tuple[Path, Path]:
    import torch

    checkpoint_dir = Path("output/v2/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    ckpt_path = checkpoint_dir / f"anra_{args.model_size}_{ts}.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "best_loss": best_loss,
            "loss_history": loss_history[-100:],
            "model_size": args.model_size,
            "session_minutes": args.session_minutes,
            "timestamp": ts,
        },
        ckpt_path,
    )

    size_mb = ckpt_path.stat().st_size / 1024**2
    print(f"\nCheckpoint saved: {ckpt_path} ({size_mb:.0f} MB)")

    report = {
        "run_timestamp": ts,
        "model_size": args.model_size,
        "total_steps": step // accum_steps,
        "epochs_completed": epoch,
        "final_loss": loss_history[-1] if loss_history else None,
        "best_loss": best_loss,
        "first_loss": loss_history[0] if loss_history else None,
        "loss_drop": (loss_history[0] - best_loss) if loss_history else 0,
        "session_minutes": args.session_minutes,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_size_mb": round(size_mb, 1),
    }
    report_path = Path("output/v2/reports") / f"training_{ts}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return ckpt_path, report_path


def print_final_summary(
    args: argparse.Namespace,
    step: int,
    epoch: int,
    best_loss: float,
    loss_history: list[float],
    accum_steps: int,
    ckpt_path: Path,
    report_path: Path,
) -> None:
    print()
    print("=" * 60)
    print("AN-RA TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model:         {args.model_size.upper()}")
    print(f"Steps:         {step // accum_steps:,}")
    print(f"Epochs:        {epoch}")
    if loss_history:
        print(f"First loss:    {loss_history[0]:.4f}")
        print(f"Final loss:    {loss_history[-1]:.4f}")
        print(f"Best loss:     {best_loss:.4f}")
        drop = loss_history[0] - best_loss
        print(f"Loss drop:     {drop:.4f} ({drop / loss_history[0] * 100:.1f}%)")
    print(f"Checkpoint:    {ckpt_path}")
    print(f"Report:        {report_path}")
    print()
    if loss_history and loss_history[-1] < loss_history[0] * 0.95:
        print("✅ Model is learning — loss dropped >5%")
    elif loss_history and loss_history[-1] < loss_history[0]:
        print("⚠️  Model improving slowly — needs more data or more time")
    else:
        print("❌ Loss not dropping — check data pipeline")
    print()
    print("Next steps:")
    print("  Download checkpoint from /output/v2/checkpoints/")
    print("  Run: python anra.py --report")
    print("=" * 60)


def train(args: argparse.Namespace, vram_gb: float) -> None:
    import torch
    from torch.cuda.amp import GradScaler, autocast
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    root = repo_root()
    sys.path.insert(0, str(root))

    from training.v2_config import V2_1B_TRAINING, V2_TRAINING
    from training.v2_runtime import (
        build_frontier_model,
        build_v2_model,
        load_or_build_v2_tokenizer,
    )

    print("\nBUILDING MODEL")
    print("-" * 60)
    if args.model_size == "1b":
        model = build_frontier_model()
        training_cfg = V2_1B_TRAINING
    else:
        model = build_v2_model(vocab_size=8209)
        training_cfg = V2_TRAINING
    del training_cfg

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total / 1e6:.0f}M total, {trainable / 1e6:.0f}M trainable")

    if args.model_size == "1b" and vram_gb < 40:
        model.gradient_checkpointing_enable()
        effective_batch = 1
        grad_accum = 32
    elif args.model_size == "1b":
        model.gradient_checkpointing_enable()
        effective_batch = 2
        grad_accum = 16
    else:
        effective_batch = 8
        grad_accum = 4

    if hasattr(model, "disable_kv_cache"):
        model.disable_kv_cache()
    device = torch.device("cuda")
    model = model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=3e-4 if args.model_size == "1b" else 3e-4,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=100)
    cosine = CosineAnnealingLR(optimizer, T_max=10000, eta_min=3e-5)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[100])

    use_bf16 = torch.cuda.is_bf16_supported()
    scaler = GradScaler(enabled=not use_bf16)
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"Precision: {'bfloat16' if use_bf16 else 'float16'}")

    tokenizer = load_or_build_v2_tokenizer()
    blocks = build_token_dataset(tokenizer, root, block_size=512)
    if not blocks:
        raise RuntimeError("No token blocks available for training.")

    deadline = time.time() + args.session_minutes * 60
    train_until = deadline - 600
    if train_until <= time.time():
        train_until = deadline

    model.train()
    step = 0
    epoch = 0
    loss_history: list[float] = []
    best_loss = float("inf")
    ckpt_path = Path("output/v2/checkpoints/not_saved.pt")
    report_path = Path("output/v2/reports/not_saved.json")

    random.shuffle(blocks)

    batch_size = effective_batch
    accum_steps = grad_accum

    print("\nTRAINING STARTED")
    print("-" * 60)
    print(f"Deadline: {args.session_minutes} minutes from now")
    print(f"Batch size: {batch_size} | Grad accum: {accum_steps}")
    print("-" * 60)

    optimizer.zero_grad()
    accum_loss = 0.0
    start_time = time.time()

    try:
        while time.time() < train_until:
            for block_idx in range(0, len(blocks) - batch_size, batch_size):
                if time.time() >= train_until:
                    break

                batch_blocks = blocks[block_idx : block_idx + batch_size]
                x = torch.tensor([b[:-1] for b in batch_blocks], dtype=torch.long, device=device)
                y = torch.tensor([b[1:] for b in batch_blocks], dtype=torch.long, device=device)

                with autocast(dtype=dtype, enabled=True):
                    logits, loss = model(x, targets=y)
                    del logits
                    if loss is None:
                        raise RuntimeError("Model returned no loss.")
                    loss = loss / accum_steps

                if use_bf16:
                    loss.backward()
                else:
                    scaler.scale(loss).backward()

                accum_loss += loss.item()

                if (step + 1) % accum_steps == 0:
                    if use_bf16:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    else:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()

                    if (step // accum_steps) % 10 == 0:
                        elapsed = time.time() - start_time
                        remaining = train_until - time.time()
                        lr_now = scheduler.get_last_lr()[0]
                        real_loss = accum_loss * accum_steps
                        loss_history.append(real_loss)
                        if real_loss < best_loss:
                            best_loss = real_loss
                        print(
                            f"step {step // accum_steps:>5} | "
                            f"loss {real_loss:.4f} | "
                            f"best {best_loss:.4f} | "
                            f"lr {lr_now:.2e} | "
                            f"elapsed {elapsed / 60:.1f}m | "
                            f"left {remaining / 60:.1f}m"
                        )
                        accum_loss = 0.0

                step += 1

            epoch += 1
            random.shuffle(blocks)
            print(f"--- Epoch {epoch} complete ---")
    finally:
        ckpt_path, report_path = save_checkpoint_and_report(
            args=args,
            model=model,
            optimizer=optimizer,
            step=step,
            epoch=epoch,
            best_loss=best_loss,
            loss_history=loss_history,
            accum_steps=accum_steps,
        )

    print_final_summary(
        args=args,
        step=step,
        epoch=epoch,
        best_loss=best_loss,
        loss_history=loss_history,
        accum_steps=accum_steps,
        ckpt_path=ckpt_path,
        report_path=report_path,
    )


def main() -> None:
    global _ARGS
    args = parse_args()
    _ARGS = args

    os.chdir(repo_root())
    sys.path.insert(0, str(repo_root()))

    _, vram_gb = gpu_check(args)
    install_missing_dependencies()

    target_dir = repo_root() / "training_data"
    download_data(str(target_dir))

    if args.data_only:
        print("\nData-only run complete.")
        return

    reasoning_path = target_dir / "reasoning.jsonl"
    patch_data_mix(str(reasoning_path), str(repo_root()))
    train(args, vram_gb)


if __name__ == "__main__":
    main()
