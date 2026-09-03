"""Single entry point for checkpoint qualification (Mission 10/16/18/26/27).

Usage (from repo root):
  python x_factor/qualify_checkpoint.py --checkpoint PATH --mode calibrate
  python x_factor/qualify_checkpoint.py --checkpoint PATH --mode qualify \\
      --protocol x_factor/protocols/cognition_readiness_binding_v2.json \\
      --research-subject PATH   # registry entry with research_subject:true
      # ...or --allow-historical-control for intentional negative-control runs

Arrival contract for a stronger checkpoint (Mission 18):
  1 strict identity -> 2 compat profile -> 3 primitive canaries ->
  4 calibration ladder -> 5 candidate PARTIAL regime -> 6 frozen protocol ->
  7 qualification -> 8+ science only if READY_SCOPED/READY.

Modes: calibrate (cheap; NEVER emits final READY) vs qualify (frozen
protocol; only path to READY). Legacy --level quick|standard maps to
calibrate (full rungs) with a deprecation warning.

Refuses to run when: checkpoint missing (no silent fallback), architecture
unsupported, registry lock violated, or execution policy forbids local
model compute (TRIQUETRA_NO_LOCAL_MODEL_COMPUTE=1).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import sys as _sys

_XF = Path(__file__).resolve().parent
if str(_XF) not in _sys.path:
    _sys.path.insert(0, str(_XF))

from checkpoint_identity import (  # noqa: E402
    UnsupportedArchitecture,
    load_core,
    match_architecture_profile,
    resolve_checkpoint,
)
from execution_policy import assert_local_compute_allowed  # noqa: E402

_LEVEL_MAP = {"quick": "calibrate", "standard": "calibrate"}


def _registry_lookup(path: str) -> dict | None:
    reg_path = _XF / "registry" / "checkpoints.json"
    if not reg_path.exists():
        return None
    try:
        reg = json.loads(reg_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    for c in reg.get("checkpoints", []):
        if c.get("path") == path:
            return c
    return None


def _subject_allowed(entry: dict | None, allow_historical_control: bool) -> dict:
    """Pure research-subject lock check (unit-testable, no I/O)."""
    locked = bool(entry and entry.get("research_subject") is True)
    if locked or allow_historical_control:
        return {"allowed": True, "mode": "research_subject" if locked else "historical_control"}
    return {"allowed": False,
            "reason": "checkpoint not registry-marked research_subject:true; "
                      "pass --research-subject or --allow-historical-control"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mode", default=None, choices=["calibrate", "qualify"])
    ap.add_argument("--level", default=None, choices=["quick", "standard"])
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--research-subject", default=None)
    ap.add_argument("--allow-historical-control", action="store_true")
    ap.add_argument("--seed", type=int, default=42424)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="output/readiness_receipt.json")
    args = ap.parse_args(argv)

    if args.mode is None:
        if args.level is not None:
            print(f"WARNING: --level {args.level} is deprecated; mapping to --mode calibrate",
                  file=sys.stderr)
        args.mode = "calibrate"
    assert_local_compute_allowed("model")

    ckpt = str(resolve_checkpoint(args.checkpoint))  # fails loudly if absent

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, payload = load_core(ckpt, device)
    cfg = payload["model_config"]
    try:
        prof = match_architecture_profile(cfg)
    except UnsupportedArchitecture as e:
        print(str(e), file=sys.stderr)
        return 2
    print(f"profile: {prof['profile']}", flush=True)
    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    entry = _registry_lookup(args.checkpoint) or _registry_lookup(ckpt)
    gate = _subject_allowed(entry, args.allow_historical_control)
    if args.mode == "qualify":
        if not gate["allowed"]:
            print(f"research-subject lock: {gate['reason']}.", file=sys.stderr)
            return 2
        if args.protocol is None:
            print("QUALIFY requires --protocol <frozen JSON>; refusing to invent thresholds.",
                  file=sys.stderr)
            return 2
        try:
            protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            print(f"unreadable protocol: {e}", file=sys.stderr)
            return 2
    from provenance import (  # noqa: E402
        git_head,
        param_sha256_from_state_dict,
        sha256_file,
        sha256_json,
    )
    from readiness.pipeline import run_readiness_v2  # noqa: E402 (canonical v2)

    try:
        tok_ident = tok.identity()
    except Exception:
        tok_ident = {"vocab": "canonical-v4-32k"}
    from anra_core.config import CANONICAL_CONFIG  # noqa: E402

    param_sha = param_sha256_from_state_dict(model.state_dict())
    ckpt_sha = sha256_file(ckpt)
    exp_sha = sha256_file(str(Path(__file__).resolve()))
    commit = git_head(Path(__file__).resolve().parents[1])

    if args.mode == "qualify":
        if args.protocol is None:
            print("QUALIFY requires --protocol <frozen JSON>; refusing to invent thresholds.",
                  file=sys.stderr)
            return 2
        try:
            protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            print(f"unreadable protocol: {e}", file=sys.stderr)
            return 2
        design = protocol.get("design", {})
        rungs = tuple(design.get("rungs", ["B0", "B1", "B2", "B3"]))
        n = int(design.get("n_per_rung", 16))
        seed = int(design.get("seed", args.seed))
        protocol_sha = sha256_file(str(Path(args.protocol).resolve()))
        replication_ref = design.get("replication")
        budget_n = design.get("budget_n")
    else:
        rungs, n, seed = ("B0", "B1", "B2", "B3"), 12, args.seed
        protocol_sha, replication_ref, budget_n = None, None, None
    receipt = run_readiness_v2(
        model, tok, payload, checkpoint=ckpt, param_sha=param_sha,
        ckpt_sha=ckpt_sha, tok_sha=sha256_json(tok_ident), exp_sha=exp_sha,
        commit=commit, seed=seed, rungs=rungs, n_per_rung=n, device=device,
        stage=args.mode, protocol_sha=protocol_sha,
        replication_ref=replication_ref, budget_n=budget_n)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({"research_readiness": receipt["research_readiness"],
                      "phase": receipt["phase"],
                      "candidate_rung": receipt["candidate_rung"],
                      "blockers": receipt["blockers"]}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
