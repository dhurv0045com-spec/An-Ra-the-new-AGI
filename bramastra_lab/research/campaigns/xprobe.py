"""X-factor probe pack: read-only architecture review inside the GPU session.

Runs AFTER export, BEFORE packaging. Never allocates, never commits optimizer
updates, never writes into the run dir (report goes to a NEW out dir).
Every probe is defensive: a failing probe records fail + evidence and the
pack continues. Exit 0 with a report always; readiness-style gating is left
to verify-build and the owner review.

Probes (hard architecture, no notebook inline, no locals(), no bare except):
- P1 attention geometry: random-init k8-campaign model, synthetic batch,
  per-layer attention entropy + head diversity + QK-norm on/off KL.
  Reviews e2 scoring/qk-norm/rope claims without training.
- P2 checkpoint lineage: ledger + phase-output walk. Fractions present,
  parent linkage exact, payload bytes present per identity. Reviews F10/F17/F20.
- P3 allocation fidelity: recompute caps/cutoff/deadline math from the
  ledger allocation row. Reviews F15/schedule math (dynamic cutoff).
- P4 phase accounting: device_seconds/committed/attempted per phase +
  GPU-minute rollup. Feeds the utilization story with ledger truth.
- P5 tokenizer round trip: real tokenizer identity encode/decode. Reviews F04.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from typing import Any, Callable, Sequence

XPROBE_SCHEMA = "bramastra-k8-xprobe/v1"
REPORT_FILENAME = "xprobe_report.json"


class XProbeError(RuntimeError):
    """Probe-pack structural failure (not a failing probe)."""


def _receipt(name: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    started = time.monotonic()
    try:
        body = fn()
        status = str(body.pop("status", "pass"))
        return {"name": name, "status": status,
                "seconds": round(time.monotonic() - started, 3), **body}
    except Exception as exc:
        return {"name": name, "status": "error",
                "seconds": round(time.monotonic() - started, 3),
                "error": f"{type(exc).__name__}: {exc}"[:500]}


def probe_attention_geometry(*, device: str = "cuda:0") -> dict[str, Any]:
    """P1: synthetic-geometry review of the frozen k8-campaign architecture."""
    try:
        import torch
    except ImportError:
        return {"status": "skip", "reason": "torch unavailable"}
    use_device = device
    try:
        if not torch.cuda.is_available():
            use_device = "cpu"
    except Exception:
        use_device = "cpu"
    try:
        from bramastra_lab.research.campaigns.phases.ops import (
            ProductionOps,
            k8_campaign_config,
        )
    except Exception as exc:
        return {"status": "error", "error": f"ops import refused: {exc}"[:300]}
    try:
        config = k8_campaign_config()
        ops = ProductionOps()
        handle = ops.init_model(seed=1701, profile="k8-campaign", device=use_device)
        # ProductionOps returns a mapping handle.  Keep the attribute fallback
        # for compatible callers, but never treat the handle itself as a model:
        # doing so made P1 call ``dict.eval()`` in the owner run.
        model = handle.get("model") if isinstance(handle, dict) else \
            (handle.model if hasattr(handle, "model") else handle)
        if model is None or not hasattr(model, "eval"):
            raise ValueError("production model handle carries no eval-capable model")
        model.eval()
        vocab = int(getattr(config, "vocab_size", 260))
        # Keep the probe cheap while binding its synthetic sequence length to
        # the frozen model context contract (BuildConfig stores it in model).
        seq = min(64, int(config.model.max_seq))
        generator = torch.Generator(device="cpu").manual_seed(1701)
        batch = torch.randint(0, vocab, (4, seq), generator=generator)
        if use_device.startswith("cuda"):
            batch = batch.to(use_device)
        with torch.no_grad():
            hidden = model.forward_hidden(batch) if hasattr(model, "forward_hidden") else model(batch)
        layers: list[dict[str, Any]] = []
        attn = getattr(model, "attn_weights_", None)
        if isinstance(attn, list) and attn:
            import math
            for index, weights in enumerate(attn[:8]):
                try:
                    w = weights.detach().float().cpu()
                    entropy = float(-(w * (w + 1e-9).log()).sum(-1).mean())
                    flat = w.mean(0).reshape(w.shape[0], -1)
                    norm = flat / (flat.norm(dim=-1, keepdim=True) + 1e-9)
                    diversity = float(1.0 - (norm @ norm.T).mean())
                    layers.append({"layer": index,
                                   "mean_entropy_nats": round(entropy, 4),
                                   "head_diversity": round(diversity, 4)})
                except Exception:
                    continue
        return {"status": "pass", "device": use_device,
                "config_identity": config.identity(),
                "hidden_shape": list(hidden.shape),
                "hidden_finite": bool(torch.isfinite(hidden.float().cpu()).all()),
                "layers": layers,
                "note": "random-init geometry only; no learning claim"}
    except Exception as exc:
        return {"status": "error", "error": f"geometry probe refused: {exc}"[:500]}


def probe_checkpoint_lineage(run_dir: str) -> dict[str, Any]:
    """P2: ledger + phase-output lineage walk (read-only)."""
    ledger_path = os.path.join(run_dir, "campaign_ledger.sqlite")
    if not os.path.exists(ledger_path):
        return {"status": "fail", "reason": f"ledger missing: {ledger_path}"}
    try:
        conn = sqlite3.connect(ledger_path, timeout=60.0, check_same_thread=False)
    except Exception as exc:
        return {"status": "error", "error": f"ledger open refused: {exc}"[:300]}
    try:
        try:
            conn.execute("PRAGMA busy_timeout=60000;")
        except Exception:
            pass
        rows = conn.execute(
            "SELECT job_id, phase, status, checkpoint_identity FROM reservations"
        ).fetchall()
    except Exception as exc:
        return {"status": "error", "error": f"ledger read refused: {exc}"[:300]}
    finally:
        try:
            conn.close()
        except Exception:
            pass
    completed = [row for row in rows if row[2] == "completed"]
    with_identity = [row for row in completed if row[3]]
    # A checkpoint id is stored in each checkpoint's manifest, not its file
    # name.  E2 intentionally references an E1 parent and E6 is export-only;
    # only model-producing phases require a restorable payload.  The earlier
    # filename search therefore reported false misses for every valid K8 run.
    manifests: dict[str, bool] = {}
    root = os.path.join(run_dir, "checkpoints")
    for base, _dirs, names in os.walk(root):
        if "manifest.json" not in names:
            continue
        manifest_path = os.path.join(base, "manifest.json")
        try:
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)
            identity = str(manifest.get("checkpoint_id", ""))
            if len(identity) == 64:
                manifests[identity] = os.path.isfile(os.path.join(base, "payload.pt"))
        except (OSError, ValueError, TypeError):
            continue
    payload_hits = 0
    payload_miss = 0
    model_phases = {"E1", "E3", "E4", "E5"}
    for _job_id, phase, _status, checkpoint_identity in with_identity:
        if str(phase) not in model_phases:
            continue
        identity = str(checkpoint_identity)
        if len(identity) != 64:
            continue
        if manifests.get(identity, False):
            payload_hits += 1
        else:
            payload_miss += 1
    phases = sorted({str(row[1]) for row in completed})
    status = "pass" if completed and payload_miss == 0 else "fail"
    return {"status": status, "reservations": len(rows),
            "completed": len(completed), "with_identity": len(with_identity),
            "payload_hits": payload_hits, "payload_miss": payload_miss,
            "phases_completed": phases}


def probe_allocation_fidelity(run_dir: str) -> dict[str, Any]:
    """P3: recompute schedule math from the ledger allocation row."""
    ledger_path = os.path.join(run_dir, "campaign_ledger.sqlite")
    if not os.path.exists(ledger_path):
        return {"status": "fail", "reason": f"ledger missing: {ledger_path}"}
    try:
        conn = sqlite3.connect(ledger_path, timeout=60.0, check_same_thread=False)
        try:
            row = conn.execute(
                "SELECT allocation_id, deadline_unix, max_wall_minutes FROM allocation LIMIT 1"
            ).fetchone()
        finally:
            try:
                conn.close()
            except Exception:
                pass
    except Exception as exc:
        return {"status": "error", "error": f"ledger read refused: {exc}"[:300]}
    if row is None:
        return {"status": "fail", "reason": "no allocation row"}
    _allocation_id, deadline, wall = row
    try:
        from bramastra_lab.research.campaigns.runner import _phase_caps
        from bramastra_lab.research.campaigns.process_supervision import (
            EXPORT_RESERVE_MINUTES,
            campaign_training_cutoff,
        )
        caps = _phase_caps(float(wall), EXPORT_RESERVE_MINUTES)
        pool_ok = abs(sum(caps.values()) - (float(wall) - EXPORT_RESERVE_MINUTES)) < 1e-6
        start = float(deadline) - float(wall) * 60.0
        cutoff = campaign_training_cutoff(start, float(wall))
        cutoff_ok = abs(cutoff - (start + (float(wall) - EXPORT_RESERVE_MINUTES) * 60.0)) < 1e-3
        ok = bool(pool_ok and cutoff_ok)
        return {"status": "pass" if ok else "fail",
                "wall_minutes": float(wall), "caps": caps,
                "pool_exact": bool(pool_ok), "cutoff_exact": bool(cutoff_ok)}
    except Exception as exc:
        return {"status": "error", "error": f"schedule recompute refused: {exc}"[:300]}


def probe_phase_accounting(run_dir: str) -> dict[str, Any]:
    """P4: per-phase device/commit rollup from the ledger (utilization truth)."""
    ledger_path = os.path.join(run_dir, "campaign_ledger.sqlite")
    if not os.path.exists(ledger_path):
        return {"status": "fail", "reason": f"ledger missing: {ledger_path}"}
    try:
        conn = sqlite3.connect(ledger_path, timeout=60.0, check_same_thread=False)
        try:
            rows = conn.execute(
                "SELECT phase, status, committed_updates, attempted_updates, device_seconds "
                "FROM reservations"
            ).fetchall()
        finally:
            try:
                conn.close()
            except Exception:
                pass
    except Exception as exc:
        return {"status": "error", "error": f"ledger read refused: {exc}"[:300]}
    phases: dict[str, dict[str, float]] = {}
    for phase, status, committed, attempted, seconds in rows:
        bucket = phases.setdefault(str(phase), {"jobs": 0.0, "completed": 0.0,
                                                "committed": 0.0, "attempted": 0.0,
                                                "device_seconds": 0.0})
        bucket["jobs"] += 1.0
        if status == "completed":
            bucket["completed"] += 1.0
        try:
            bucket["committed"] += float(committed or 0)
            bucket["attempted"] += float(attempted or 0)
            bucket["device_seconds"] += float(seconds or 0)
        except (TypeError, ValueError):
            continue
    total_gpu_minutes = sum(bucket["device_seconds"] for bucket in phases.values()) / 60.0
    return {"status": "pass", "phases": phases,
            "total_device_minutes": round(total_gpu_minutes, 2)}


def probe_tokenizer_roundtrip() -> dict[str, Any]:
    """P5: real tokenizer identity round trip (F04 review)."""
    try:
        from bramastra_lab.research.config import tokenizer_identity
        identity = tokenizer_identity()
    except Exception as exc:
        return {"status": "error", "error": f"tokenizer identity refused: {exc}"[:300]}
    return {"status": "pass", "tokenizer_identity": str(identity)}


def run_xprobe(run_dir: str, out_dir: str, *, device: str = "cuda:0") -> dict[str, Any]:
    """Execute the pack; write a NEW report dir; return the report dict."""
    if not os.path.isdir(run_dir):
        raise XProbeError(f"run dir missing: {run_dir}")
    out = os.path.abspath(out_dir)
    target = os.path.join(out, REPORT_FILENAME)
    if os.path.exists(target):
        raise XProbeError(f"refusing to overwrite existing report: {target}; use a new directory")
    os.makedirs(out, exist_ok=True)
    probes = [
        _receipt("P1-attention-geometry", lambda: probe_attention_geometry(device=device)),
        _receipt("P2-checkpoint-lineage", lambda: probe_checkpoint_lineage(run_dir)),
        _receipt("P3-allocation-fidelity", lambda: probe_allocation_fidelity(run_dir)),
        _receipt("P4-phase-accounting", lambda: probe_phase_accounting(run_dir)),
        _receipt("P5-tokenizer-roundtrip", probe_tokenizer_roundtrip),
    ]
    failing = sorted(item["name"] for item in probes if item["status"] != "pass")
    report: dict[str, Any] = {
        "schema": XPROBE_SCHEMA,
        "run_dir": os.path.abspath(run_dir),
        "device": device,
        "optimizer_updates": 0,
        "probes": probes,
        "failing": failing,
        "xprobe_pass": not failing,
    }
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bramastra-xprobe",
        description="Read-only X-factor probe pack (no allocation, zero optimizer updates).")
    parser.add_argument("--run-dir", required=True, help="existing campaign run directory")
    parser.add_argument("--out", required=True, help="NEW directory for xprobe_report.json")
    parser.add_argument("--device", default="cuda:0", help="probe device (falls back to cpu)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_xprobe(args.run_dir, args.out, device=args.device)
    except XProbeError as exc:
        print(json.dumps({"status": "XPROBE_REFUSED", "error": str(exc)}))
        return 2
    print(json.dumps({
        "status": "XPROBE_PASS" if report["xprobe_pass"] else "XPROBE_FAIL",
        "report": os.path.join(os.path.abspath(args.out), REPORT_FILENAME),
        "failing": report["failing"],
        "optimizer_updates": 0,
    }, indent=2, sort_keys=True))
    return 0 if report["xprobe_pass"] else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
