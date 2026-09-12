"""ARK-014 local before/after demonstration from recorded checkpoints.

Intended interface (no hosted service, no paid API):

    python experiments/ARK-014/demo.py --run-dir <completed-run-directory> \
        --output <new-demo-directory>

The command loads the actual recorded model checkpoints, verifies their file
hashes against the run receipt and the data identities against the frozen task
manifest, then produces a small HTML report plus a machine-readable examples
JSON. Demonstrated fact-sets are selected by a frozen rule (the 12 BIND_SEALED
fact-sets with the smallest signature hash) that is independent of any model
outcome; they are labeled SEALED measurement examples used only after the
frozen run, and are never fed back into model selection.

If a checkpoint is missing, was modified, or does not support a receipt's
success claim, the command reports the discrepancy and refuses to fabricate a
before/after comparison. Failed model answers are shown verbatim; the symbolic
solver never replaces them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "COLAB"))
sys.path.insert(0, str(REPO / "experiments" / "ARK-001"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ark014_binding as binding  # noqa: E402
from run_ark001 import CompactVocab, Micro  # noqa: E402

DEMO_FACTSET_COUNT = 12
RESULT_FILE = "ARK-014_RESULT.json"
TASK_FILE = "ARK-014_TASK_MANIFEST.json"
CHECKPOINT_DIRNAME = "checkpoints"
AGREE_TOLERANCE = 1e-9


class DemoRefused(RuntimeError):
    """The run directory cannot support an honest before/after report."""


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_rank_key(signature_json: list) -> str:
    return hashlib.sha256(json.dumps(signature_json, separators=(",", ":")).encode("utf-8")).hexdigest()


def load_result(run_dir: Path) -> dict:
    path = run_dir / RESULT_FILE
    if not path.exists():
        for alternative in ("ARK-014_FAILURE.json", "FAILURE_RECEIPT.json"):
            if (run_dir / alternative).exists():
                raise DemoRefused(
                    f"run has no {RESULT_FILE}; found {alternative} — the run did not complete, "
                    "so no before/after comparison is available")
        raise DemoRefused(f"missing {RESULT_FILE} in {run_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_task_identity(run_dir: Path) -> dict:
    recorded_path = run_dir / TASK_FILE
    if not recorded_path.exists():
        raise DemoRefused(f"missing {TASK_FILE} in {run_dir}")
    recorded = json.loads(recorded_path.read_text(encoding="utf-8"))
    task = binding.build_binding_task()
    rebuilt = task["manifest"]
    # The recorded file carries receipt decoration added by ReceiptWriter.save;
    # verify every manifest field the frozen builder defines, field by field.
    for key, value in rebuilt.items():
        if key == "manifest_sha256":
            continue
        if key not in recorded:
            raise DemoRefused(f"task manifest drift: missing field {key}")
        if binding.sha_json(recorded[key]) != binding.sha_json(value):
            raise DemoRefused(f"task manifest drift: field {key} does not match the frozen builder")
    if recorded.get("manifest_sha256") != rebuilt["manifest_sha256"]:
        raise DemoRefused("task manifest drift: recorded manifest does not match the frozen builder")
    return task


def load_model_from_checkpoint(ark_module, path: Path, recorded: dict, device) -> tuple:
    if not path.exists():
        raise DemoRefused(f"checkpoint file missing: {path}")
    actual_sha = _file_sha256(path)
    if recorded.get("sha256") != actual_sha:
        raise DemoRefused(
            f"checkpoint hash mismatch for {path.name}: receipt {recorded.get('sha256')} != file {actual_sha}")
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("parameter_sha256") != recorded.get("parameter_sha256"):
        raise DemoRefused(f"checkpoint parameter hash mismatch for {path.name}")
    vocab = CompactVocab()
    model = Micro(vocab.size, 128).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return vocab, model, payload


@torch.no_grad()
def greedy_rows(model, vocab, rows, device, max_answer: int = 6) -> list[str]:
    """Per-row greedy decode; mirrors run_ark001.greedy_exact grouping exactly."""
    groups: dict[int, list[int]] = {}
    for index, (prompt, _) in enumerate(rows):
        groups.setdefault(len(vocab.encode(prompt)), []).append(index)
    text_of: dict[int, str] = {}
    for same_length, indices in groups.items():
        batch_rows = [rows[i] for i in indices]
        tokens = torch.tensor([vocab.encode(p) for p, _ in batch_rows], device=device)
        batch = len(batch_rows)
        finished = torch.zeros(batch, dtype=torch.bool)
        generated: list[list[int]] = [[] for _ in batch_rows]
        for _ in range(max_answer):
            logits = model(tokens)[:, -1]
            next_ids = torch.argmax(logits, dim=-1)
            tokens = torch.cat(
                [tokens, torch.full((batch, 1), vocab.PAD, dtype=torch.long, device=device)], dim=1)
            all_finished = True
            for i in range(batch):
                if finished[i]:
                    continue
                token = int(next_ids[i].item())
                if token in (vocab.EOS, vocab.PAD):
                    finished[i] = True
                else:
                    generated[i].append(token)
                    tokens[i, -1] = token
                    all_finished = False
            if all_finished:
                break
        for i, index in enumerate(indices):
            text_of[index] = vocab.decode(generated[i]).strip()
    return [text_of[i] for i in range(len(rows))]


def exact_rate(predictions: list[str], rows) -> float:
    return sum(1.0 for p, (_, a) in zip(predictions, rows) if p == a) / len(rows)


def _arm_receipt(arm: dict) -> dict:
    return arm.get("checkpoint", {})


def evaluate_arm(ark_module, arm: dict, task, device, checkpoints_dir: Path) -> dict:
    """Recompute all diagnostics for one acquisition arm from its checkpoint."""
    checkpoint_info = _arm_receipt(arm)
    if not checkpoint_info:
        raise DemoRefused(f"arm {arm.get('regime')} has no recorded checkpoint")
    path = checkpoints_dir / checkpoint_info["filename"]
    vocab, model, payload = load_model_from_checkpoint(ark_module, path, checkpoint_info, device)
    if int(payload.get("step", -1)) != int(checkpoint_info.get("step", -2)):
        raise DemoRefused(f"checkpoint step mismatch for {path.name}")

    per_example = {}
    aggregates = {}
    for split in ("BIND_CONTROL", "BIND_SEALED"):
        for diagnostic in binding.DIAGNOSTICS:
            rows = binding._diagnostic_rows_grouped(task, split, diagnostic)
            predictions = greedy_rows(model, vocab, rows, device)
            aggregates[f"{split}/{diagnostic}"] = {
                "exact": exact_rate(predictions, rows),
                "denominator": len(rows),
            }
    # Cross-check the batched historical evaluator against the per-row decoder
    # on one full group; disagreement would mean the demo's numbers are not the
    # runtime's numbers.
    probe_rows = binding._diagnostic_rows_grouped(task, "BIND_SEALED", "CANONICAL")
    exact_hist, _ = ark_module.greedy_exact(model, vocab, probe_rows, device)
    demo_hist = exact_rate(greedy_rows(model, vocab, probe_rows, device), probe_rows)
    if abs(exact_hist - demo_hist) > AGREE_TOLERANCE:
        raise DemoRefused("demo decoder disagrees with the runtime evaluator")

    _verify_against_receipt(arm, aggregates, path.name)
    return {"aggregates": aggregates, "model": model, "vocab": vocab,
            "checkpoint": checkpoint_info}


def _verify_against_receipt(arm: dict, aggregates: dict, filename: str) -> None:
    """A receipt's success claims must be supported by its own checkpoint."""
    status = arm.get("status")
    step = int(arm.get("checkpoint", {}).get("step", -1))
    comparable = [t for t in arm.get("trajectory", []) if int(t["step"]) == step]
    if comparable:
        trajectory_point = comparable[-1]
        for diagnostic in binding.DIAGNOSTICS:
            recorded = trajectory_point.get(f"control_{diagnostic}")
            if recorded is None:
                continue
            recomputed = aggregates[f"BIND_CONTROL/{diagnostic}"]["exact"]
            if abs(recorded - recomputed) > AGREE_TOLERANCE:
                raise DemoRefused(
                    f"{filename}: BIND_CONTROL {diagnostic} at step {step} "
                    f"receipt={recorded} recomputed={recomputed} — checkpoint/record mismatch")
    if status == "QUALIFIED":
        sealed = arm.get("sealed_at_qualification")
        if not sealed:
            raise DemoRefused(f"{filename}: receipt claims QUALIFIED without a sealed measurement")
        for diagnostic in binding.DIAGNOSTICS:
            recomputed = aggregates[f"BIND_SEALED/{diagnostic}"]["exact"]
            if abs(float(sealed[diagnostic]) - recomputed) > AGREE_TOLERANCE:
                raise DemoRefused(
                    f"{filename}: sealed {diagnostic} receipt={sealed[diagnostic]} "
                    f"recomputed={recomputed} — checkpoint/record mismatch")
        for diagnostic, threshold in binding.QUALIFICATION_THRESHOLDS.items():
            control_value = aggregates[f"BIND_CONTROL/{diagnostic}"]["exact"]
            if control_value < threshold:
                raise DemoRefused(
                    f"{filename}: receipt claims QUALIFIED but BIND_CONTROL {diagnostic}="
                    f"{control_value} does not meet the frozen threshold {threshold} — "
                    "unsupported success badge")


def select_demo_factsets(task) -> list:
    """Frozen rule: smallest signature-hash BIND_SEALED fact-sets."""
    sealed = task["sealed_factsets"]
    ranked = sorted(sealed, key=_hash_rank_key)
    return ranked[:DEMO_FACTSET_COUNT]


def build_example_records(task, baseline: dict, candidate: dict, device) -> list:
    """Per-example predictions on the frozen demonstration fact-sets."""
    demo_factsets = select_demo_factsets(task)
    records = []
    for facts in demo_factsets:
        canonical = [[int(k), int(v)] for k, v in facts]
        for diagnostic in binding.DIAGNOSTICS:
            dict_rows = binding._diagnostic_rows(
                tuple((int(k), int(v)) for k, v in facts), diagnostic)
            pairs = [(row["prompt"], row["answer"]) for row in dict_rows]
            baseline_preds = greedy_rows(baseline["model"], baseline["vocab"], pairs, device)
            candidate_preds = greedy_rows(candidate["model"], candidate["vocab"], pairs, device)
            for row, bp, cp in zip(dict_rows, baseline_preds, candidate_preds):
                records.append({
                    "diagnostic": diagnostic,
                    "canonical_facts": canonical,
                    "shown_facts": row["facts"],
                    "reordered_shown": row["facts"] != canonical,
                    "query": row["query"],
                    "prompt": row["prompt"],
                    "ground_truth": row["answer"],
                    "baseline_prediction": bp,
                    "candidate_prediction": cp,
                    "baseline_correct": bp == row["answer"],
                    "candidate_correct": cp == row["answer"],
                })
    return records


def _fact_string(pairs) -> str:
    return "+".join(f"{k}={v}" for k, v in pairs)


def render_html(payload: dict) -> str:
    """Minimal static template; no framework, no external assets."""
    def esc(value):
        return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    rows_html = []
    for rec in payload["examples"]:
        rows_html.append(
            "<tr>"
            f"<td>{esc(rec['diagnostic'])}</td>"
            f"<td><code>{esc(rec['prompt'])}</code></td>"
            f"<td>{esc(_fact_string(rec['canonical_facts']))}</td>"
            f"<td>{esc(_fact_string(rec['shown_facts']))}</td>"
            f"<td>{esc(rec['ground_truth'])}</td>"
            f"<td>{esc(rec['baseline_prediction'])}</td>"
            f"<td>{esc(rec['candidate_prediction'])}</td>"
            "</tr>")
    agg_rows = []
    for key, values in payload["aggregate"].items():
        agg_rows.append(
            "<tr>"
            f"<td>{esc(key)}</td>"
            f"<td>{values['baseline']['exact']:.4f}</td><td>{values['baseline']['denominator']}</td>"
            f"<td>{values['candidate']['exact']:.4f}</td><td>{values['candidate']['denominator']}</td>"
            "</tr>")
    failure_rows = []
    for rec in payload["examples"]:
        if not (rec["baseline_correct"] and rec["candidate_correct"]):
            failure_rows.append(
                "<tr>"
                f"<td>{esc(rec['diagnostic'])}</td>"
                f"<td><code>{esc(rec['prompt'])}</code></td>"
                f"<td>{esc(rec['ground_truth'])}</td>"
                f"<td>{esc(rec['baseline_prediction'])}</td>"
                f"<td>{esc(rec['candidate_prediction'])}</td>"
                "</tr>")
    arms = payload["run"].get("acquisitions", [])
    arm_rows = "".join(
        "<tr>"
        f"<td>{esc(a.get('regime'))}</td><td>{esc(a.get('status'))}</td>"
        f"<td>{a.get('steps_run')}</td><td>{a.get('supervised_positions')}</td>"
        f"<td><code>{esc((a.get('checkpoint') or {}).get('sha256', '')[:16])}</code></td>"
        "</tr>"
        for a in arms)
    summary = payload["run"].get("summary", {})
    completed = payload["run"].get("status") == "EXECUTED_OR_PARTIAL_BUDGETED"
    scale = payload["run"].get("protocol_scale", "UNKNOWN")
    arms_by_regime = {a.get("regime"): a.get("status") for a in payload["run"].get("acquisitions", [])}
    aug_status = arms_by_regime.get("ORDER_AUGMENTED")
    canon_status = arms_by_regime.get("CANONICAL_TRAIN")
    if aug_status == "QUALIFIED" and canon_status != "QUALIFIED":
        repair_line = ("<b>ORDER_ROBUSTNESS_REPAIRED criterion: MET</b> — the candidate regime "
                       "qualified on BIND_CONTROL while the matched baseline did not.")
    elif aug_status == "QUALIFIED" and canon_status == "QUALIFIED":
        repair_line = ("<b>ORDER_ROBUSTNESS_REPAIRED criterion:</b> both regimes qualified; consult "
                       "the receipt's material-improvement deltas for the frozen secondary rule.")
    else:
        repair_line = "<b>ORDER_ROBUSTNESS_REPAIRED criterion: NOT MET</b> — the candidate regime did not qualify on BIND_CONTROL."
    scale_banner = "" if scale == "PREREGISTERED_FROZEN" else (
        f'<p class="failure"><b>NOT THE FROZEN HORIZON:</b> this run is labeled '
        f'<code>{esc(scale)}</code>. Its numbers are engineering diagnostics and '
        f"must not be read as the preregistered experiment outcome.</p>")
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>ARK-014 before/after demonstration</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; max-width: 1100px; }}
table {{ border-collapse: collapse; margin: 1rem 0; width: 100%; }}
th, td {{ border: 1px solid #bbb; padding: 4px 8px; text-align: left; font-size: 0.9rem; }}
code {{ background: #f4f4f4; }}
.badge {{ display: inline-block; padding: 2px 10px; border-radius: 4px; color: white;
        background: #555; }}
.badge.qualified {{ background: #1a7f37; }}
.badge.failed {{ background: #b35900; }}
.note {{ background: #fff8e1; border: 1px solid #e0c96b; padding: 8px 12px; }}
.failure {{ background: #fdecea; border: 1px solid #d88; padding: 8px 12px; }}
</style></head><body>
<h1>ARK-014 — order-robust non-arithmetic binding: before/after</h1>
{scale_banner}
<p><b>Run:</b> <code>{esc(payload['run_id'])}</code> ·
<b>run completed:</b> {esc(completed)} ·
<b>verdict (from receipt):</b> <code>{esc(summary.get('verdict'))}</code><br>
{repair_line}</p>
<p class="note">Demonstrated examples are the {payload['demo_factset_count']} BIND_SEALED fact-sets
with the smallest signature hash (frozen rule, independent of model outcomes). BIND_SEALED is a
measurement-only split; these examples were not used for checkpoint selection. Failed model
answers are shown verbatim — the symbolic solver never substitutes for them.</p>
<h2>Arms and checkpoints</h2>
<table><tr><th>Regime</th><th>Status</th><th>Steps</th><th>Supervised positions</th><th>Checkpoint sha256 (prefix)</th></tr>
{arm_rows}</table>
<h2>Aggregate exact accuracy (recomputed from checkpoints)</h2>
<table><tr><th>Split/Diagnostic</th><th>Baseline (CANONICAL_TRAIN)</th><th>n</th>
<th>Candidate (ORDER_AUGMENTED)</th><th>n</th></tr>
{''.join(agg_rows)}</table>
<h2>Per-example predictions</h2>
<table><tr><th>Diagnostic</th><th>Prompt</th><th>Canonical facts</th><th>Shown facts</th>
<th>Ground truth</th><th>Baseline</th><th>Candidate</th></tr>
{''.join(rows_html)}</table>
<h2>Failures on the demonstrated examples</h2>
<p class="failure">{len(failure_rows)} of {len(payload['examples'])} displayed rows are incorrect
for at least one model. They are listed below and are not corrected or hidden.</p>
<table><tr><th>Diagnostic</th><th>Prompt</th><th>Ground truth</th><th>Baseline</th><th>Candidate</th></tr>
{''.join(failure_rows)}</table>
<h2>Provenance</h2>
<p>Task manifest sha256: <code>{esc(payload['task_manifest_sha256'])}</code><br>
Checkpoint files verified against receipt hashes: {esc(payload['checkpoint_verification'])}<br>
Generated locally by <code>experiments/ARK-014/demo.py</code> at {esc(payload['generated_at'])}.</p>
</body></html>"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="must be a new directory")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args(argv)

    run_dir = args.run_dir.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    try:
        run = load_result(run_dir)
        task = verify_task_identity(run_dir)
        if run.get("protocol_scale") != "PREREGISTERED_FROZEN":
            print("WARNING: this run is labeled "
                  f"{run.get('protocol_scale', 'UNKNOWN')} — not the frozen horizon; "
                  "the report will say so prominently.", flush=True)
        checkpoints_dir = run_dir / CHECKPOINT_DIRNAME
        device = torch.device(args.device)
        ark_module = _load_ark_module()

        arms = {a.get("regime"): a for a in run.get("acquisitions", [])}
        baseline_arm = arms.get("CANONICAL_TRAIN")
        candidate_arm = arms.get("ORDER_AUGMENTED")
        if not baseline_arm or not candidate_arm:
            raise DemoRefused("run receipt does not contain both acquisition arms")
        verified = []
        baseline = evaluate_arm(ark_module, baseline_arm, task, device, checkpoints_dir)
        verified.append(baseline_arm["checkpoint"]["sha256"])
        candidate = evaluate_arm(ark_module, candidate_arm, task, device, checkpoints_dir)
        verified.append(candidate_arm["checkpoint"]["sha256"])

        examples = build_example_records(task, baseline, candidate, device)
        aggregate = {}
        for split in ("BIND_CONTROL", "BIND_SEALED"):
            for diagnostic in binding.DIAGNOSTICS:
                key = f"{split}/{diagnostic}"
                aggregate[key] = {
                    "baseline": baseline["aggregates"][key],
                    "candidate": candidate["aggregates"][key],
                }

        payload = {
            "run_id": run_dir.name,
            "run": run,
            "task_manifest_sha256": task["manifest"]["manifest_sha256"],
            "checkpoint_verification": verified,
            "demo_factset_count": DEMO_FACTSET_COUNT,
            "demo_selection_rule": ("BIND_SEALED fact-sets ranked by sha256(canonical signature); "
                                    "smallest 12; frozen, outcome-independent"),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "examples": examples,
            "aggregate": aggregate,
        }
        examples_json = {
            "schema": "arkenstone-ark014-demo/v1",
            "run_id": payload["run_id"],
            "run_status": run.get("status"),
            "protocol_scale": run.get("protocol_scale"),
            "verdict": run.get("summary", {}).get("verdict"),
            "task_manifest_sha256": payload["task_manifest_sha256"],
            "baseline_checkpoint": baseline["checkpoint"],
            "candidate_checkpoint": candidate["checkpoint"],
            "demo_selection_rule": payload["demo_selection_rule"],
            "aggregate": aggregate,
            "examples": examples,
        }
        (output / "examples.json").write_text(
            json.dumps(examples_json, indent=2, default=str) + "\n", encoding="utf-8")
        (output / "report.html").write_text(render_html(payload), encoding="utf-8")
        print("demo written:", output / "report.html", flush=True)
        return 0
    except DemoRefused as exc:
        (output / "REFUSED.txt").write_text(
            f"DEMO REFUSED: {exc}\nNo before/after comparison was fabricated.\n", encoding="utf-8")
        print(f"DEMO REFUSED: {exc}", flush=True)
        return 2


def _load_ark_module():
    import importlib.util
    path = REPO / "experiments" / "ARK-001" / "run_ark001.py"
    spec = importlib.util.spec_from_file_location("ark014_demo_ark001", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    raise SystemExit(main())
