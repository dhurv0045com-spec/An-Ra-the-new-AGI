"""Checkpoint soup: weight-average two same-lineage full-resume artifacts.

Rationale from the logit diagnostics: step-20000 and step-30400 share one
lineage but drifted to different greedy attractors ('the' vs 'ditj') during
the constant-LR repeat phase. Averaging weights of points in the same basin
frequently cancels such drift (Wortsman et al., 'Model Soups').

Output: a new checkpoint with the averaged dense tensors, tagged
artifact_class 'full_resume', soup provenance recorded in metrics.
CPU-only; no GPU required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def load_state(path: str) -> tuple[dict[str, torch.Tensor], dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict") or payload.get("model")
    return state, payload


def main(parent_a: str, parent_b: str, out_path: str) -> None:
    print(f"[soup] loading {parent_a}", flush=True)
    state_a, payload_a = load_state(parent_a)
    print(f"[soup] loading {parent_b}", flush=True)
    state_b, payload_b = load_state(parent_b)

    keys_a = set(state_a.keys())
    keys_b = set(state_b.keys())
    shared = keys_a & keys_b
    only_a = keys_a - keys_b
    only_b = keys_b - keys_a
    print(f"[soup] shared tensors: {len(shared)} | only-A: {len(only_a)} | only-B: {len(only_b)}", flush=True)

    # Soup over shared float tensors. Non-shared tensors are kept from the
    # older (better) parent so the result stays loadable by the strict loader.
    soup: dict[str, torch.Tensor] = {}
    averaged = 0
    for key in sorted(shared):
        tensor_a, tensor_b = state_a[key], state_b[key]
        if tensor_a.shape != tensor_b.shape:
            print(f"[soup] shape mismatch at {key}; keeping A", flush=True)
            soup[key] = tensor_a.clone()
            continue
        if not torch.is_floating_point(tensor_a):
            soup[key] = tensor_a.clone()
            continue
        soup[key] = ((tensor_a.float() + tensor_b.float()) / 2.0).to(tensor_a.dtype)
        averaged += 1
    for key in only_a:
        soup[key] = state_a[key].clone()

    base = dict(payload_a)
    base["global_step"] = int(payload_a.get("global_step", 0))
    base["training_stage"] = f"checkpoint_soup(a={Path(parent_a).name}, b={Path(parent_b).name})"
    base["metrics"] = {
        "soup": True,
        "parents": [str(parent_a), str(parent_b)],
        "averaged_tensors": averaged,
        "note": "equal-weight average of same-lineage checkpoints",
    }
    base.pop("optimizer", None)
    base.pop("scheduler", None)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(f".{out.name}.uploading")
    torch.save(base, str(temporary))
    temporary.replace(out)

    import hashlib

    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(
        f"[soup] wrote {out}\n[soup] averaged {averaged} tensors\n[soup] sha256 {digest[:16]}...",
        flush=True,
    )


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: python scripts/make_soup.py PARENT_A PARENT_B OUT")
    main(sys.argv[1], sys.argv[2], sys.argv[3])
