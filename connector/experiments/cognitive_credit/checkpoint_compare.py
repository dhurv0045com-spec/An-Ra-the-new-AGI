"""Compare generation quality across checkpoints on identical probes.

Tests the hypothesis: 'training was good until ~20k steps'. Same prompts,
same decode settings, only the checkpoint differs.
"""

import sys

from anra_core.executor import CoreExecutor
from anra_core.generate import generate

PROBES = (
    ("continuation", "Once upon a time, there was a little girl who"),
    ("knowledge", "The capital of France is"),
    ("echo", "Echo exactly this word: ember"),
    ("arithmetic", "Compute (3 + 4) x 2."),
)


def probe(path: str) -> None:
    print(f"\n{'=' * 64}\n{path}\n{'=' * 64}", flush=True)
    try:
        executor = CoreExecutor.from_checkpoint(path, device="cpu")
    except Exception:  # noqa: BLE001 - older schemas mark contracts unavailable
        print("[strict load failed; loading weights forensically]", flush=True)
        import torch

        from anra_core.config import CANONICAL_CONFIG
        from anra_core.model import AnRaCore
        from anra_core.tokenizer import V4Tokenizer

        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload.get("model", payload.get("model_state_dict", payload))
        model = AnRaCore(CANONICAL_CONFIG)
        model.load_state_dict(state)
        raw_step = payload.get("global_step", payload.get("step", -1))
        step = int(raw_step) if raw_step is not None else -1
        executor = CoreExecutor(model, tokenizer=V4Tokenizer.load_canonical())
        print(f"forensic load OK; global_step: {step:,}", flush=True)
    tok = executor.tokenizer
    step = executor.checkpoint_identity.global_step
    if step is None:
        step = -1
    print(f"global_step: {step:,}", flush=True)
    for name, prompt in PROBES:
        try:
            out = generate(executor, tok, prompt, max_new_tokens=20, temperature=0.0)
        except Exception as exc:  # noqa: BLE001 - report and continue
            out = f"<error {type(exc).__name__}: {exc}>"
        print(f"[{name:>12}] {prompt[:34]!r} -> {out!r}", flush=True)
    # One sampled sample to see distribution health.
    try:
        out = generate(
            executor, tok, PROBES[0][1], max_new_tokens=20,
            temperature=0.8, seed=7,
        )
        print(f"[sampled t=0.8] {out!r}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[sampled] <error {exc}>", flush=True)
    del executor


if __name__ == "__main__":
    for ckpt in sys.argv[1:]:
        probe(ckpt)
