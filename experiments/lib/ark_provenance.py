"""Provenance mechanics adopted from BRAMASTRA (branch relations: read-only source).

Three adopted patterns, adapted to Arkenstone's manifest/ledger style:
1. continuation_probe(): save -> one update -> reload -> same update -> verify
   parameters/optimizer/sampler match exactly (BRAMASTRA experiment.py L180-227).
2. source_snapshot(): capture the exact runner source into the receipt
   (BRAMASTRA source_snapshot/ pattern).
3. nominate_next(): a transparent fixed rule that names the next experiment from
   measured evidence (BRAMASTRA decide() pattern; GAP 1's manual precursor).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def source_snapshot(paths: list[str]) -> dict[str, str]:
    """Capture exact source text of every file the experiment depends on."""
    snapshot = {}
    for path in paths:
        p = Path(path)
        if p.exists():
            snapshot[p.name] = p.read_text(encoding="utf-8")
    return snapshot


def snapshot_sha256(snapshot: dict[str, str]) -> str:
    return hashlib.sha256(
        json.dumps(snapshot, sort_keys=True).encode("utf-8")
    ).hexdigest()


def continuation_probe(
    *,
    model,
    optimizer,
    batch_fn,
    loss_fn,
    checkpoint_path: Path,
    device,
) -> dict:
    """Save -> one update -> reload -> same update -> verify exact reproduction.

    Returns a receipt with parameters_exact / optimizer_exact booleans.
    Raises on checkpoint identity mismatch (BRAMASTRA pattern).
    """
    import torch

    import copy
    def cpu_tree(state_dict):
        def walk(obj):
            if torch.is_tensor(obj):
                return obj.detach().cpu().clone()
            if isinstance(obj, dict):
                return {k: walk(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return type(obj)(walk(v) for v in obj)
            return obj
        result = walk(state_dict)
        # deep-copy any remaining nested structures (optimizer param_groups etc.)
        return copy.deepcopy(result)

    payload = {
        "model": cpu_tree(model.state_dict()),
        "optimizer": cpu_tree(optimizer.state_dict()),
        "torch_rng": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng"] = torch.cuda.get_rng_state_all()
    pending = checkpoint_path.with_suffix(".pt.pending")
    torch.save(payload, pending)
    pending.replace(checkpoint_path)
    saved_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()

    # one update, recorded
    loss, _ = loss_fn(batch_fn())
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    expected_model = cpu_tree(model.state_dict())
    expected_opt = cpu_tree(optimizer.state_dict())

    # reload and repeat the same update
    if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != saved_hash:
        raise RuntimeError("checkpoint changed before load")
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["torch_rng"])
    if torch.cuda.is_available() and saved.get("cuda_rng"):
        torch.cuda.set_rng_state_all(saved["cuda_rng"])

    loss2, _ = loss_fn(batch_fn())
    optimizer.zero_grad(set_to_none=True)
    loss2.backward()
    optimizer.step()
    reproduced_model = cpu_tree(model.state_dict())
    reproduced_opt = cpu_tree(optimizer.state_dict())

    def trees_equal(a, b):
        if torch.is_tensor(a) and torch.is_tensor(b):
            return torch.equal(a, b)
        if isinstance(a, dict) and isinstance(b, dict):
            return a.keys() == b.keys() and all(trees_equal(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            return len(a) == len(b) and all(trees_equal(x, y) for x, y in zip(a, b))
        return a == b

    parameters_exact = all(
        trees_equal(expected_model[k], reproduced_model[k]) for k in expected_model
    )
    optimizer_exact = trees_equal(expected_opt, reproduced_opt)
    receipt = {
        "schema": "arkenstone-continuation-probe/v1",
        "checkpoint_sha256": saved_hash,
        "parameters_exact": parameters_exact,
        "optimizer_exact": optimizer_exact,
        "scope": "local single-process continuation, not remote durability",
    }
    if not (parameters_exact and optimizer_exact):
        raise RuntimeError(f"continuation probe failed: {receipt}")
    return receipt


def nominate_next(evidence: dict) -> dict:
    """Fixed transparent research triage (BRAMASTRA decide() pattern).

    evidence: {"retention_collapse_seen": bool, "precursor_survives": bool,
               "intervention_nulls": [str], "open_tier_boundary": bool}
    Returns {"verdict", "next_experiment", "rule"}.
    """
    if evidence.get("retention_collapse_seen") and not evidence.get("intervention_nulls"):
        return {"verdict": "RETENTION_UNADDRESSED",
                "next_experiment": "ARK-005 retention arms",
                "rule": "decay seen with no intervention tried -> test interventions first"}
    if evidence.get("retention_collapse_seen") and evidence.get("intervention_nulls"):
        return {"verdict": "RETENTION_INTERVENTIONS_EXHAUSTED",
                "next_experiment": "stronger LR intervention on decaying seeds only",
                "rule": "interventions tried and null/weak -> escalate the strongest hint"}
    if evidence.get("open_tier_boundary"):
        return {"verdict": "DOSE_RATIO_UNMAPPED",
                "next_experiment": "tier dose-ratio sweep (T3 carry boundary)",
                "rule": "boundary unmeasured -> map it before spending TPU budget"}
    return {"verdict": "INSUFFICIENT_EVIDENCE",
            "next_experiment": "return to the bottleneck graph",
            "rule": "no measured signal available for a fixed rule"}
