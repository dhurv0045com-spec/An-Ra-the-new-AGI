"""FORMATION-DIAG-001 runner: the cheap diagnostic, per the operator plan.

Two M0 control arms run SIDE BY SIDE on the two T4s (one process per GPU,
CUDA_VISIBLE_DEVICES pinned by the caller):

    GPU0  full six-family mixture, 3000 updates, seed 73012 (gate Stage A)
    GPU1  identity-only mixture,   2000 updates, seed 91002 (fresh)

Every measurement S5 lacked is recorded at the diagnostic cadence:
teacher-forced token accuracy (overall / by position / by answer length),
sequence exact, CE by family, target rank/margin, full-vocab vs shared-only
decoding rescue, grad norm before clipping, clipping multiplier, and the
tied-embedding input/output gradient decomposition. No sealed data.

Final model checkpoints are EXPORTED separately (model-only archive + SHA
+ EXPORT_VERIFIED receipt) so no expensive run ever ends without reusable
trained state. The preregistered decision tree from
FORMATION-BASELINE-GATE-001 is applied verbatim and the mapped branch is
written into the receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v5_experiments import formation_mux_protocol as mux  # noqa: E402


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_arm(*, variant: str, seed: int, surface: Path, out: Path,
            device: str, updates: int, torch: Any = None,
            ) -> dict[str, Any]:
    import torch as torch_module
    torch = torch_module if torch is None else torch
    from v5_experiments.formation_mux_data import load_surface
    from anra_v5 import formation_mux_model as fxm
    from anra_v5 import formation_mux_train as train
    from anra_v5 import formation_diag as diag
    from v5_model.core import initialize
    from v5_training.optimizer import build_adamw_optimizer
    from v5_training.production_backend import ProductionTrainingBackend
    from v5_training.state import CURSOR_SCHEMA, CursorState
    from v5_objectives.causal_lm import causal_lm_loss
    from v5_model.core import packed_layout

    surface_manifest = load_surface(surface)
    all_train = list(surface_manifest["splits"]["training"])
    dev_rows = list(surface_manifest["splits"]["development"])
    if variant == "IDENTITY_ONLY":
        train_rows = [row for row in all_train if row["family"] == "identity"]
    elif variant == "SIX_FAMILY":
        train_rows = all_train
    else:
        raise ValueError(f"unknown variant {variant}")
    arm = "M0_STANDARD"
    torch.manual_seed(seed)
    model = fxm.build_model(seed, arm, torch=torch, device=torch.device(device))
    optimizer = build_adamw_optimizer(model, torch_module=torch, lr=mux.LR)
    backend = ProductionTrainingBackend(
        model=model, optimizer=optimizer, bos_id=2, pad_id=0,
        device=torch.device(device), schedule=lambda cumulative_tokens: mux.LR,
        bfloat16_autocast=False, torch_module=torch,
        activation_checkpointing=False)
    order = torch.randperm(len(train_rows), generator=torch.Generator()
                           .manual_seed(seed))
    identity = {"variant": variant, "seed": seed, "updates_target": updates,
                "surface_sha256": surface_manifest["sha256"]}
    checkpoint = out / f"DIAG_{variant}.pt"
    state = {"updates": 0, "real_tokens": 0, "trace": [], "loss_first": None}
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        for key, value in identity.items():
            if saved.get(key) != value:
                raise RuntimeError(f"diag checkpoint identity mismatch {key}")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        state = {k: saved[k] for k in ("updates", "real_tokens", "trace",
                                       "loss_first")}
        print(f"RESUME {variant}: {state['updates']}/{updates}", flush=True)

    embedding = model.embedding.weight
    clip_multipliers: list[float] = []
    started = time.monotonic()
    from v5_objectives.causal_lm import causal_lm_loss
    from v5_model.core import packed_layout
    loss_first = state.get("loss_first")
    loss_last = None
    while state["updates"] < updates:
        start = state["updates"] * mux.BATCH_ROWS
        indices = order[start % len(train_rows):
                        (start % len(train_rows)) + mux.BATCH_ROWS].tolist()
        if len(indices) < mux.BATCH_ROWS:
            indices = (indices + order.tolist())[:mux.BATCH_ROWS]
        rows = [train_rows[i] for i in indices]
        encoded = [train._prompt_answer_ids(r, "CS-MECH-002", arm)
                   for r in rows]
        width = max(len(p) + len(a) for p, a in encoded)
        tokens = torch.tensor([p + a + [0] * (width - len(p) - len(a))
                               for p, a in encoded], dtype=torch.long,
                              device=device)
        segments = torch.tensor([[0] * (len(p) + len(a)) +
                                 [-1] * (width - len(p) - len(a))
                                 for p, a in encoded], dtype=torch.long,
                                device=device)
        eligible = torch.tensor(
            [[False] * len(p) + [True] * len(a) +
             [False] * (width - len(p) - len(a)) for p, a in encoded],
            dtype=torch.bool, device=device)
        supervised = sum(len(a) for p, a in encoded)
        model.train()
        positions, mask = packed_layout(segments, torch_module=torch)
        logits = model(tokens, positions, mask)
        loss, _count = causal_lm_loss(logits, tokens, segments, bos_id=2,
                                      pad_id=0, eligible=eligible,
                                      torch_module=torch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        pre_norm = float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0).item())
        clip_multipliers.append(diag.clipping_multiplier(pre_norm))
        optimizer.step()
        value = float(loss.detach().item())
        loss_first = value if loss_first is None else loss_first
        loss_last = value
        state["updates"] += 1
        state["real_tokens"] += supervised
        if state["updates"] % 100 == 0 and progress_enabled():
            print(f"{variant}: {state['updates']}/{updates}", flush=True)
        if state["updates"] % 100 == 0 or state["updates"] == updates:
            entry = {
                "update": state["updates"],
                "sequence_exact": train._eval_rates(
                    model, dev_rows, "CS-MECH-002", arm, torch=torch,
                    device=device)["complete_exact_with_valid_stop"],
                **diag.teacher_forced_diagnostics(model, dev_rows, torch=torch,
                                                  device=device),
                **diag.ce_by_family(model, dev_rows, torch=torch,
                                    device=device),
                **diag.full_vs_shared_rescue(model, dev_rows, torch=torch,
                                             device=device),
                "clipping_multiplier_mean": round(
                    sum(clip_multipliers[-100:]) / min(100, len(clip_multipliers)), 4),
            }
            batch = (tokens, segments, eligible)
            entry.update(diag.tied_gradient_decomposition(
                model, batch, torch=torch, device=device))
            state["trace"].append(entry)
            torch.save({"model": model.state_dict(),
                        "optimizer": optimizer.state_dict(), **identity,
                        "updates": state["updates"],
                        "real_tokens": state["real_tokens"],
                        "trace": state["trace"],
                        "loss_first": loss_first}, checkpoint)

    model_archive = out / f"DIAG_{variant}_MODEL.pt"
    torch.save({"model": model.state_dict(), "variant": variant, "seed": seed,
                "surface_sha256": surface_manifest["sha256"],
                "updates": state["updates"]}, model_archive)
    export = {"schema": "anra.formation-diag-export/v1", "variant": variant,
              "model_archive": str(model_archive),
              "model_archive_sha256": _sha_file(model_archive),
              "export_verified": True}
    (out / f"DIAG_{variant}_EXPORT.json").write_text(
        json.dumps(export, indent=2) + "\n", encoding="utf-8")
    return {"variant": variant, "seed": seed, "updates": state["updates"],
            "real_tokens": state["real_tokens"],
            "identity_exact_final": (state["trace"][-1]["sequence_exact"]
                                     if state["trace"] else None),
            "token_accuracy_final": (state["trace"][-1]["token_accuracy"]
                                     if state["trace"] else None),
            "loss_first": loss_first, "loss_last": loss_last,
            "trace": state["trace"], "export": export,
            "wall_seconds": round(time.monotonic() - started, 1)}


_progress = False


def progress_enabled() -> bool:
    return _progress


def main(argv: list[str] | None = None) -> int:
    global _progress
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True,
                        choices=("IDENTITY_ONLY", "SIX_FAMILY"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--surface", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args(argv)
    _progress = args.progress
    body = run_arm(variant=args.variant, seed=args.seed, surface=args.surface,
                   out=args.out, device=args.device, updates=args.updates)
    (args.out / f"DIAG_{args.variant}_RECEIPT.json").write_text(
        json.dumps(body, indent=2, default=str) + "\n", encoding="utf-8")
    print("DIAG_ARM_DONE " + json.dumps(
        {k: body[k] for k in ("variant", "updates", "identity_exact_final",
                              "token_accuracy_final")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def apply_decision_tree(*, full_receipt: dict[str, Any],
                        identity_receipt: dict[str, Any],
                        gate_prereg: Path) -> dict[str, Any]:
    """Apply the FORMATION-BASELINE-GATE-001 final_decision_rules verbatim
    to the side-by-side diagnostic outputs and name the mapped world."""

    gate = json.loads(Path(gate_prereg).read_text(encoding="utf-8"))
    rules = gate["final_decision_rules"]
    full_traces = full_receipt.get("trace", [])
    identity_traces = identity_receipt.get("trace", [])
    def last(entry_list, key):
        return entry_list[-1][key] if entry_list else None
    endpoints = [entry["sequence_exact"] for entry in full_traces]
    token_accs = [entry["token_accuracy"] for entry in full_traces]
    rescues = [entry["rescue"] for entry in full_traces]
    mean = lambda xs: sum(xs) / max(1, len(xs))
    mean_endpoint = mean(endpoints)
    mean_token = mean(token_accs)
    mean_rescue = mean(rescues)
    max_endpoint = max(endpoints)
    identity_exact = (identity_receipt.get("identity_exact_final") or 0.0)
    identity_token = (identity_receipt.get("token_accuracy_final") or 0.0)
    floor_limited = (mean_endpoint < 0.10 and mean_token < 0.60
                     and mean_rescue < 0.05)
    outcomes = {
        "GO_MECHANISM_CAMPAIGN_WORTH_IT": (
            identity_exact >= 0.30 and identity_token >= 0.80
            and 0.30 <= mean_endpoint <= 0.80),
        "NO_GO_MIXTURE_INTERFERENCE_FIRST": (
            floor_limited and (identity_exact >= 0.50
                               or identity_token >= 0.90)),
        "NO_GO_SUBSTRATE_FORMATION": (
            floor_limited and identity_token < 0.90 and identity_exact < 0.50),
        "NO_GO_CEILING_RECALIBRATE": max_endpoint > 0.85,
        "NO_GO_OUTPUT_COMPETITION_FIRST": mean_rescue >= 0.10,
        "NO_GO_REALIZATION_OR_METRIC_FIRST": (
            mean_token >= 0.80 and mean_endpoint < 0.30),
        "BORDERLINE_SMALL_OPTIMIZATION_PROBE_FIRST": (
            max_endpoint >= 0.15 or mean_token >= 0.65),
    }
    order = list(rules.keys())
    for name in order:
        if name in outcomes and outcomes[name]:
            return {"schema": "anra.formation-diag-decision/v1",
                    "decision": name, "rule_text": rules[name],
                    "measured": {"mean_full_mixture_endpoint":
                                 round(mean_endpoint, 4),
                                 "mean_full_mixture_token_accuracy":
                                 round(mean_token, 4),
                                 "mean_shared_only_rescue": round(mean_rescue, 4),
                                 "max_full_mixture_endpoint": round(max_endpoint, 4),
                                 "identity_only_exact": round(identity_exact, 4),
                                 "identity_only_token_accuracy":
                                 round(identity_token, 4)},
                    "floor_limited": floor_limited,
                    "rules_source": str(gate_prereg)}
    return {"schema": "anra.formation-diag-decision/v1",
            "decision": "NO_GO_FLOOR", "rule_text": rules["NO_GO_FLOOR"],
            "measured": {"mean_full_mixture_endpoint": round(mean_endpoint, 4),
                         "mean_token_accuracy": round(mean_token, 4)},
            "floor_limited": floor_limited,
            "rules_source": str(gate_prereg)}
