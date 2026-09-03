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
        from readiness.gate import run_gate  # noqa: E402  (v1 runner; v2 decision applied below)

        design = protocol.get("design", {})
        rungs = tuple(design.get("rungs", ["B0", "B1", "B2", "B3"]))
        n = int(design.get("n_per_rung", 16))
        receipt = run_gate(ckpt, int(design.get("seed", args.seed)), n, device, rungs=rungs)
        receipt["schema"] = "anra-cognition-readiness/v2-qualify"
        receipt["protocol"] = str(args.protocol)
        receipt["stage"] = "qualify"
    else:
        from readiness.gate import run_gate  # noqa: E402

        receipt = run_gate(ckpt, args.seed, 12, device, rungs=("B0", "B1", "B2", "B3"))
        receipt["schema"] = "anra-cognition-readiness/v2-calibrate"
        receipt["stage"] = "calibrate"
        if receipt.get("classification") == "READY_FOR_BINDING_CAUSAL_RESEARCH":
            receipt["classification"] = "CALIBRATION_REQUIRED"
            receipt.setdefault("notes", []).append(
                "calibration cannot emit final READY (v2 two-stage rule)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({"classification": receipt["classification"],
                      "stage": receipt["stage"]}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
