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

    # Tied-weight alias repair: the model ties lm_head to the embedding table,
    # and the historical 181M ABI aliases `_normed_mlp` tensors onto their
    # `norm_2`/`mlp` targets. Independent averaging can make each alias drift
    # by one ULP, which the strict loader correctly rejects ("serialized weight
    # alias drift"). The canonical tensor wins; every registered alias must be
    # its exact copy.
    canonical_re = __import__("re").compile(r"^blocks\.(\d+)\._normed_mlp\.(0|1)(\..+)$")
    repaired_aliases = 0

    def _repair_alias(alias: str, canonical_key: str) -> None:
        nonlocal repaired_aliases
        if alias in soup and canonical_key in soup and not torch.equal(soup[alias], soup[canonical_key]):
            soup[alias] = soup[canonical_key].clone()
            repaired_aliases += 1

    for alias, canonical in {
        "lm_head.weight": "token_embedding_table.weight",
        "token_embedding.weight": "token_embedding_table.weight",
    }.items():
        _repair_alias(alias, canonical)
    for key in list(soup.keys()):
        match = canonical_re.fullmatch(key)
        if not match:
            continue
        layer, member, suffix = match.groups()
        target = f"blocks.{layer}.norm_2{suffix}" if member == "0" else f"blocks.{layer}.mlp{suffix}"
        _repair_alias(key, target)
    if repaired_aliases:
        print(f"[soup] repaired {repaired_aliases} tied-weight alias drift(s)", flush=True)

    base = dict(payload_a)
    # THE FIX (P0): the averaged weights must actually be installed into the
    # serialized payload. The previous version built `soup` and saved `base`
    # without it, producing a byte-identical copy of parent A.
    key = "model_state_dict" if "model_state_dict" in payload_a else "model"
    if key not in payload_a:
        raise SystemExit(f"cannot find model tensor key in parent A payload: {sorted(payload_a)[:10]}")
    base[key] = soup
    # A soup is NOT resumable training state: averaging parameters while
    # keeping one parent's optimizer moments is invalid. Demote honestly.
    base["checkpoint_artifact_class"] = "model_only"
    base["checkpoint_schema_version"] = 1
    base.pop("optimizer", None)
    base.pop("scheduler", None)
    base.pop("optimizer_state_dict", None)
    base["global_step"] = int(payload_a.get("global_step", 0))
    base["pack_step"] = 0
    base["training_stage"] = f"checkpoint_soup(a={Path(parent_a).name}, b={Path(parent_b).name})"
    base["metrics"] = {
        "soup": True,
        "parents": [str(parent_a), str(parent_b)],
        "averaged_tensors": averaged,
        "note": "equal-weight average of same-lineage checkpoints; model_only, not resumable",
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(f".{out.name}.uploading")
    torch.save(base, str(temporary))
    temporary.replace(out)

    import hashlib

    digest = hashlib.sha256(out.read_bytes()).hexdigest()

    # Self-verification: the saved artifact's tensors must differ from BOTH
    # parents (parameter-level), else the soup silently failed again.
    saved = torch.load(out, map_location="cpu", weights_only=False)
    saved_state = saved.get(key)
    identical_a = sum(
        1 for k in soup if k in saved_state and torch.equal(saved_state[k], state_a[k])
    )
    identical_b = sum(
        1
        for k in soup
        if k in saved_state and k in state_b and torch.equal(saved_state[k], state_b[k])
    )
    print(
        f"[soup] wrote {out}\n[soup] averaged {averaged} tensors\n"
        f"[soup] sha256 {digest[:16]}...\n[soup] verification vs parent A: "
        f"{identical_a}/{len(soup)} identical\n[soup] verification vs parent B: "
        f"{identical_b}/{len(soup)} identical",
        flush=True,
    )
    if identical_a == len(soup) or identical_b == len(soup):
        raise SystemExit(
            "SOUP FAILED SELF-CHECK: artifact matches a single parent exactly; "
            "averaging did not take effect."
        )


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: python scripts/make_soup.py PARENT_A PARENT_B OUT")
    main(sys.argv[1], sys.argv[2], sys.argv[3])
