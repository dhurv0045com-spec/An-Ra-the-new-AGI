"""Train a small native An-Ra model to prove tokenizer and shard correctness."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tokenizer.subword_tokenizer import SubwordTokenizer
from torch.nn import functional
from torch.utils.data import DataLoader
from training.v2_config import V2ModelConfig
from training.v2_data_mix import RawCausalShardDataset
from training.v2_runtime import atomic_save, build_model_from_config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-tokens", type=int, default=32_000_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--drive-root", default="")
    args = parser.parse_args()
    if args.drive_root and (
        not args.train_manifest.is_file() or not args.validation_manifest.is_file()
    ):
        subprocess.run(
            [
                sys.executable,
                "scripts/colab_prepare_data.py",
                "--repo",
                str(Path.cwd()),
                "--profile",
                "30gb",
                "--drive-root",
                args.drive_root,
            ],
            check=True,
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer_hash = sha256_file(args.tokenizer)
    tokenizer = SubwordTokenizer.from_pretrained(args.tokenizer)
    config = V2ModelConfig(
        vocab_size=tokenizer.vocab_size,
        n_embd=256,
        n_head=4,
        n_kv_head=1,
        n_layer=6,
        block_size=512,
        mod_layers=(2, 4),
        base_seq_len=512,
        target_seq_len=512,
        use_hal=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(
        config, block_size=512, vocab_size=tokenizer.vocab_size
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.05)
    train = RawCausalShardDataset(
        args.train_manifest,
        tokenizer,
        512,
        expected_tokenizer_sha256=tokenizer_hash,
    )
    validation = RawCausalShardDataset(
        args.validation_manifest,
        tokenizer,
        512,
        expected_tokenizer_sha256=tokenizer_hash,
    )
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    optimizer.zero_grad(set_to_none=True)
    optimizer_steps = 0
    tokens_seen = 0
    microsteps = 0
    started = time.time()
    model.train()
    while tokens_seen < args.target_tokens:
        for inputs, targets, _weights, _indices in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda" and torch.cuda.is_bf16_supported(),
            ):
                logits, _ = model(inputs)
                loss = functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
                )
                scaled = loss / args.grad_accum
            scaled.backward()
            microsteps += 1
            if microsteps < args.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            microsteps = 0
            optimizer_steps += 1
            tokens_seen += int(targets.numel()) * args.grad_accum
            if optimizer_steps % 100 == 0:
                print(
                    f"step={optimizer_steps} tokens={tokens_seen} loss={float(loss):.4f}",
                    flush=True,
                )
            if tokens_seen >= args.target_tokens:
                break
    model.eval()
    losses = []
    with torch.no_grad():
        for index in range(min(100, len(validation))):
            inputs, targets, _weights, _stable = validation[index]
            logits, _ = model(inputs.unsqueeze(0).to(device))
            losses.append(
                float(
                    functional.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        targets.unsqueeze(0).to(device).reshape(-1),
                    )
                )
            )
    validation_loss = sum(losses) / max(1, len(losses))
    payload = {
        "checkpoint_schema_version": 1,
        "profile": "native-draft-proof",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None,
        "global_step": optimizer_steps,
        "tokens_seen": tokens_seen,
        "accum_micro_steps": 0,
        "completed_optimizer_boundary": True,
        "tokenizer_sha256": tokenizer_hash,
        "model_config": config.__dict__,
        "validation_loss": validation_loss,
        "rng_states": {"torch": torch.get_rng_state(), "numpy": np.random.get_state()},
        "source_commit": os.environ.get("ANRA_SOURCE_COMMIT", "unknown"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(payload, args.output, drive_dir=None)
    report = {
        "passed": math.isfinite(validation_loss),
        "checkpoint": str(args.output),
        "checkpoint_sha256": sha256_file(args.output),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "optimizer_steps": optimizer_steps,
        "tokens_seen": tokens_seen,
        "validation_loss": validation_loss,
        "elapsed_seconds": time.time() - started,
    }
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
