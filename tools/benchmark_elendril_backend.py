"""Bounded CPU-only execution microprobe, never a training/capability campaign."""

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import statistics
import subprocess
import time


SOURCE_FILES = (
    "anra_v5/elendril/backend.py", "anra_v5/elendril/codec.py", "anra_v5/elendril/training.py",
    "v5_contracts/model_spec.py", "v5_model/attention.py", "v5_model/block.py",
    "v5_model/core.py", "v5_model/embedding.py", "v5_model/initialize.py", "v5_model/config.py",
    "v5_checkpointing.py", "tools/benchmark_elendril_backend.py",
)


def source_identity():
    root = Path(__file__).resolve().parents[1]
    return {
        "base_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "dirty_worktree": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()),
        "file_sha256": {name: sha256((root / name).read_bytes()).hexdigest() for name in SOURCE_FILES},
    }


def differential_probe():
    import torch
    from anra_v5.elendril.backend import BackendIdentity, NeuralBackend, set_attention_backend
    from anra_v5.elendril.codec import CODEC_SHA
    from anra_v5.elendril.training import MODEL, configure_cpu
    from v5_model.core import initialize

    configure_cpu()
    identity = BackendIdentity(MODEL.sha256(), CODEC_SHA, MODEL.vocabulary_size, MODEL.context_length)
    reference = NeuralBackend(initialize(MODEL, 17), identity)
    candidate = NeuralBackend(initialize(MODEL, 17), identity)
    set_attention_backend(candidate.model, "native-gqa")
    tokens = torch.tensor([[1, 7, 8, 9, 1, 10, 11, 0], [1, 12, 13, 0, 0, 0, 0, 0]])
    segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, -1], [0, 0, 0, -1, -1, -1, -1, -1]])
    left = reference.logits(tokens, segments)
    right = candidate.logits(tokens, segments, activation_checkpointing=True)
    torch.testing.assert_close(left, right, rtol=2e-5, atol=1e-6)
    weights = torch.linspace(-1, 1, left.numel()).reshape_as(left)
    (left * weights).sum().backward()
    (right * weights).sum().backward()
    gradients = {}
    for (name, a), (_, b) in zip(reference.model.named_parameters(), candidate.model.named_parameters()):
        if a.grad is None or b.grad is None or not torch.isfinite(b.grad).all():
            raise AssertionError("missing/nonfinite differential gradient")
        torch.testing.assert_close(a.grad, b.grad, rtol=3e-4, atol=2e-5)
        gradients[name] = float((a.grad - b.grad).abs().max())
    return {"reference": "repeat-sdpa", "candidate": "native-gqa",
            "candidate_activation_checkpointing": True, "state_dict_keys_unchanged": reference.model.state_dict().keys() == candidate.model.state_dict().keys(),
            "tokens": tokens.tolist(), "segments": segments.tolist(),
            "forward_rtol": 2e-5, "forward_atol": 1e-6, "gradient_rtol": 3e-4, "gradient_atol": 2e-5,
            "max_abs_logit_difference": float((left - right).abs().max().detach()),
            "max_abs_gradient_difference_by_parameter": gradients, "passed": True}


def probe(repeats=21):
    import torch
    from anra_v5.elendril.backend import BackendIdentity, NeuralBackend, set_attention_backend
    from anra_v5.elendril.codec import CODEC_SHA
    from anra_v5.elendril.training import MODEL, configure_cpu, runtime_identity
    from v5_model.core import initialize

    if type(repeats) is not int or not 3 <= repeats <= 101:
        raise ValueError("microprobe repeats must be 3..101")
    configure_cpu()
    model = initialize(MODEL, 17).eval()
    backend = NeuralBackend(model, BackendIdentity(MODEL.sha256(), CODEC_SHA, MODEL.vocabulary_size, MODEL.context_length))
    tokens = torch.full((2, 32), 7, dtype=torch.long)
    segments = torch.zeros_like(tokens)
    records = {}
    with torch.inference_mode():
        for path in ("repeat-sdpa", "native-gqa"):
            set_attention_backend(model, path)
            for _ in range(3):
                backend.logits(tokens, segments)
            times = []
            for _ in range(repeats):
                start = time.perf_counter_ns()
                backend.logits(tokens, segments)
                times.append((time.perf_counter_ns() - start) / 1e6)
            records[path] = {"median_ms": statistics.median(times), "min_ms": min(times), "max_ms": max(times), "samples_ms": times}
    return {"schema": "elendril-backend-micro-v1", "scope": "CPU correctness microbenchmark; no optimizer updates",
            "runtime": runtime_identity(), "model_sha256": MODEL.sha256(), "model_spec": asdict(MODEL),
            "tokenizer_sha256": CODEC_SHA, "seed": 17, "batch": 2, "length": 32, "warmups": 3, "repeats": repeats,
            "measurement_order": ["repeat-sdpa", "native-gqa"], "measurements": records,
            "native_over_reference": records["native-gqa"]["median_ms"] / records["repeat-sdpa"]["median_ms"],
            "gpu": "UNVERIFIED", "peak_memory": "NOT_MEASURED", "source": source_identity(), "parity": differential_probe()}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    record = json.dumps(probe(args.repeats), sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(record, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as output:
            output.write(record)
