"""T1C batched experiment matrix (one session). Platform-neutral TPU backend.

Four frozen arms (PLAN.md): A control / B answer-objective / C narrow-data /
D scale. Shared calibration (one static shape for all arms), deterministic
indexed data, per-arm receipts, hash-verified resume markers, cross-arm
summary with preregistered rules, one result bundle. Cymek code resolves from
the pinned read-only runtime; production loss/optimizer/checkpoint paths are
reused unchanged (answer arming uses the production `eligible` seam).
"""

from __future__ import annotations

import hashlib
import json
import time
import zipfile
from pathlib import Path
from typing import Any


MID_SPEC_KWARGS: dict[str, Any] = {
    "schema": "anra-v5-model-spec/v1",
    "family": "dense-decoder-transformer",
    "vocabulary_size": 24_576,
    "width": 128,
    "layers": 4,
    "query_heads": 8,
    "kv_heads": 4,
    "head_dimension": 16,
    "ffn_width": 256,
    "context_length": 4_096,
    "rope_base": 10_000.0,
    "norm_epsilon": 1e-5,
    "tied_embeddings": True,
    "qk_norm": True,
    "qk_norm_affine": True,
    "linear_bias": False,
    "dropout": 0.0,
}
MID_EXPECTED_PARAMS = 3_737_472
MINI_EXPECTED_PARAMS = 1_647_104

CALIBRATION_SHAPES = [(32, 32), (96, 32), (256, 32)]
CALIBRATION_UPDATES = 5
ARMS: dict[str, dict[str, Any]] = {
    "A": {"spec": "MINI", "objective": "whole",  "data": "rich",   "budget": 8_000_000},
    "B": {"spec": "MINI", "objective": "answer", "data": "rich",   "budget": 8_000_000},
    "C": {"spec": "MINI", "objective": "answer", "data": "narrow", "budget": 8_000_000},
    "D": {"spec": "MID",  "objective": "answer", "data": "rich",   "budget": 8_000_000},
}
AUTO_SCALE_RATE = 5_000.0  # MINI tok/s below this → halve all budgets (recorded)
DEV_SAMPLE_N = 1_000
TRAIN_SAMPLE_N = 500
CKPT_ZIP_BYTES_CAP = 200_000_000


def build_spec(which: str):
    """Construct + validate MINI/MID spec from the pinned runtime (no tensors)."""
    from v5_contracts.model_spec import ModelSpec

    if which == "MINI":
        from anra_v5.miniature_run import MINI_SPEC

        spec, expected = MINI_SPEC, MINI_EXPECTED_PARAMS
    elif which == "MID":
        spec = ModelSpec(**MID_SPEC_KWARGS)
        expected = MID_EXPECTED_PARAMS
    else:
        raise ValueError(f"unknown spec {which!r}")
    spec.assert_valid()
    total = spec.parameter_receipt().total
    if total != expected:
        raise ValueError(f"{which} receipt {total} != expected {expected}")
    return spec


def answer_spans(texts: list[str], length: int) -> list[tuple[int, int]]:
    """Pure (plen, alen) per row; raises if any row does not fit [L]."""
    from citadel_tpu import calculator_eval as cev

    spans = []
    for t in texts:
        prompt, _ = cev.split_prompt_target(t)
        ids = cev.encode(t)
        if len(ids) > length:
            raise ValueError(f"row exceeds fixed length {length}: {t!r}")
        spans.append((len(cev.encode(prompt)), len(ids) - len(cev.encode(prompt))))
    return spans


def _batch_tensors(texts: list[str], *, length: int, objective: str, torch_mod: Any):
    """Encode a batch on host; return (tokens, seg, eligible|None, whole_sup, answer_sup)."""
    from citadel_tpu import calculator_eval as cev

    torch = torch_mod
    encoded = [cev.encode(t) for t in texts]
    for ids in encoded:
        if len(ids) > length:
            raise ValueError("row exceeds fixed batch length")
    tokens = torch.full((len(texts), length), cev.PAD_ID, dtype=torch.long)
    for i, ids in enumerate(encoded):
        tokens[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    seg = torch.zeros_like(tokens)
    whole_sup = sum(max(0, len(ids) - 1) for ids in encoded)
    eligible = None
    answer_sup = 0
    if objective == "answer":
        eligible = torch.zeros_like(tokens, dtype=torch.bool)
        for i, (plen, alen) in enumerate(answer_spans(texts, length)):
            assert alen > 0, f"empty answer span: {texts[i]!r}"
            eligible[i, plen:plen + alen] = True
            answer_sup += alen
    return tokens, seg, eligible, whole_sup, answer_sup


def _train_updates(model, optimizer, feed, *, n_updates: int, start_update: int,
                   device, torch_mod, batch: int, length: int, objective: str,
                   lr: float) -> dict[str, Any]:
    """Fixed-shape update loop with exact accounting; mechanical objective check."""
    from v5_model.core import packed_layout
    from v5_objectives.causal_lm import causal_lm_loss
    from citadel_tpu import xla_backend as xb

    torch = torch_mod
    first_loss, last_loss, gsum, gmax, gn = None, None, 0.0, 0.0, 0
    cap = ans_sup = whole_sup = 0
    for u in range(n_updates):
        texts = feed((start_update + u) * batch, batch)
        tokens, seg, eligible, w_sup, a_sup = _batch_tensors(
            texts, length=length, objective=objective, torch_mod=torch)
        pos, mask = packed_layout(seg, torch_module=torch)  # host (audit A8)
        dev = device
        kwargs: dict[str, Any] = {}
        if eligible is not None:
            kwargs["eligible"] = eligible.to(dev)
        logits = model(tokens.to(dev), pos.to(dev), mask.to(dev))
        loss, count = causal_lm_loss(logits, tokens.to(dev), seg.to(dev),
                                     torch_module=torch, **kwargs)
        if objective == "answer" and int(count) != a_sup:
            raise RuntimeError(
                f"abort ELIGIBLE_MISMATCH: loss count {count} != answer span {a_sup}")
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError(f"abort NONFINITE_LOSS at update {start_update + u}")
        loss.backward()
        xb.mark_step()
        sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                sq += float(p.grad.detach().float().pow(2).sum().to("cpu").item())
        import math as _math

        gnorm = _math.sqrt(sq)
        if not (gnorm == gnorm and gnorm < float("inf")):
            raise RuntimeError(f"abort NONFINITE_GRADIENT at update {start_update + u}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        xb.optimizer_step(optimizer)
        xb.mark_step()
        optimizer.zero_grad()
        v = float(loss.detach().to("cpu").item())
        first_loss = v if first_loss is None else first_loss
        last_loss = v
        gsum, gmax, gn = gsum + gnorm, max(gmax, gnorm), gn + 1
        cap += batch * length
        ans_sup += a_sup if objective == "answer" else int(count)
        whole_sup += w_sup
    return {"first_loss": first_loss, "last_loss": last_loss,
            "capacity_tokens": cap, "answer_supervised": ans_sup,
            "whole_supervised": whole_sup,
            "grad_norm_mean": gsum / max(gn, 1), "grad_norm_max": gmax}


def _rich_feed(cursor: int, n: int) -> list[str]:
    from citadel_tpu import arith_data as ad

    return [ad.row_at("train", cursor + k)[0] for k in range(n)]


def _narrow_feed(cursor: int, n: int, *, train_rows: list[str]) -> list[str]:
    return [train_rows[(cursor + k) % len(train_rows)] for k in range(n)]


def calibrate(*, out: str | None = None, updates: int = CALIBRATION_UPDATES) -> dict[str, Any]:
    """Pick one static (batch, length) for ALL arms: max steady tok/s that fits
    and passes a finite-loss correctness step with exact accounting."""
    from citadel_tpu import environment as env_mod
    from citadel_tpu import runtime_bootstrap as rb
    from citadel_tpu import xla_backend as xb

    rt_root, rt_sha = rb.ensure_cymek_runtime()
    env = env_mod.probe(require_tpu=True)
    if not env.get("probe_pass"):
        raise env_mod.NoTpuError("ABORT_NO_TPU")
    xb.assert_tpu_active(min_devices=1)
    import torch

    from v5_model.core import initialize, packed_layout
    from v5_objectives.causal_lm import causal_lm_loss
    from v5_training.optimizer import build_adamw_optimizer

    from citadel_tpu import arith_data as ad

    spec = build_spec("MINI")
    results = []
    for batch, length in CALIBRATION_SHAPES:
        torch.manual_seed(20260904)
        model = initialize(spec, 20260904)
        device = xb.get_device()
        model = model.to(device)
        opt = build_adamw_optimizer(model, torch_module=torch)
        texts = [ad.row_at("train", k)[0] for k in range(batch)]
        try:
            import time as _time

            t = _time.time()
            rep = _train_updates(model, opt, lambda s, n: texts[:n], n_updates=updates,
                                 start_update=0, device=device, torch_mod=torch,
                                 batch=batch, length=length, objective="whole", lr=3e-4)
            wall = _time.time() - t
            toks = rep["capacity_tokens"] / wall
            ok = rep["first_loss"] is not None and rep["first_loss"] == rep["first_loss"]
            results.append({"batch": batch, "length": length, "tokens_per_second": toks,
                            "correct": bool(ok), "updates": updates})
        except Exception as exc:  # noqa: BLE001 - record, do not propagate
            results.append({"batch": batch, "length": length, "tokens_per_second": 0.0,
                            "correct": False, "error": f"{type(exc).__name__}: {exc}"})
        del model
    feasible = [r for r in results if r["correct"]]
    if not feasible:
        raise RuntimeError("abort CALIBRATION_FAILURE: no candidate shape passed")
    best = max(feasible, key=lambda r: r["tokens_per_second"])
    receipt = {"schema": "citadel-t1c-throughput-calibration/v1",
               "citadel_sha": rb.citadel_sha(), "cymek_runtime_sha": rt_sha,
               "environment": env, "candidates": results,
               "selected": {"batch": best["batch"], "length": best["length"]},
               "selected_tokens_per_second": best["tokens_per_second"]}
    if out is not None:
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    return receipt


def _eval_slice(model, rows, targets, *, device, torch_mod, xb):
    from citadel_tpu import calculator_eval as cev

    recs = cev.generate(rows, model, xb, device=device, torch_mod=torch_mod)
    return recs, cev.summarize([r["prediction"] for r in recs], targets)


def run_arm(tag: str, cfg: dict[str, Any], *, shape: tuple[int, int],
            out_dir: str, seed: int = 20260904) -> dict[str, Any]:
    """Execute one frozen arm end to end; write ARM_<tag>.json + checkpoint + marker."""
    import random as _random

    from citadel_tpu import arith_data as ad
    from citadel_tpu import calculator_data as calc
    from citadel_tpu import calculator_eval as cev
    from citadel_tpu import checkpoint as ckpt_mod
    from citadel_tpu import environment as env_mod
    from citadel_tpu import runtime_bootstrap as rb
    from citadel_tpu import xla_backend as xb

    batch, length = shape
    out_path = Path(out_dir) / f"ARM_{tag}.json"
    marker = Path(out_dir) / f"ARM_{tag}.done.json"
    if marker.is_file() and out_path.is_file():
        prior = json.loads(out_path.read_text(encoding="utf-8"))
        if isinstance(prior, dict) and prior.get("status") in ("PASS", "FAIL",
                                                               "IMPLEMENTATION_FAILURE"):
            prior["resumed"] = True
            return prior
    t0 = time.time()
    rt_root, rt_sha = rb.ensure_cymek_runtime()
    env = env_mod.probe(require_tpu=True)
    if not env.get("probe_pass"):
        raise env_mod.NoTpuError("ABORT_NO_TPU")
    n_devices = xb.assert_tpu_active(min_devices=1)
    import torch

    from v5_contracts.model_spec import QK_NORM_EPSILON
    from v5_model.config import from_spec
    from v5_model.core import initialize
    from v5_training.optimizer import build_adamw_optimizer
    from citadel_tpu.calculator_train import _mean_ce as _whole_ce  # shared CE yardstick

    spec = build_spec(cfg["spec"])
    _ = from_spec(spec, qk_norm_epsilon=QK_NORM_EPSILON)
    budget = int(cfg["budget"])
    updates_total = budget // (batch * length)

    if cfg["data"] == "rich":
        train_n = ad.SPLITS["train"]["n"]
        # Feed takes a ROW cursor (caller passes update_index * batch) and must
        # advance exactly one row per element: no stride gaps, no early wrap.
        feed = lambda s, n: _rich_feed(s, n)
        unique_note = {"kind": "indexed-rich", "unique_rows": train_n}
    else:
        narrow = calc.generate(split="train")
        feed = lambda s, n: _narrow_feed(s, n, train_rows=narrow)
        unique_note = {"kind": "narrow-canary", "unique_rows": len(narrow)}

    dev_sample = [ad.row_at("dev", i)[0] for i in range(DEV_SAMPLE_N)]
    dev_targets = [ad.split_prompt_target(r)[1] for r in dev_sample]
    test_slices = {}
    for name in ("test_core", "test_template", "test_range", "test_composition"):
        rows = [ad.row_at(name, i)[0] for i in range(ad.SPLITS[name]["n"])]
        test_slices[name] = {"rows": rows,
                             "targets": [ad.split_prompt_target(r)[1] for r in rows]}

    if cfg["data"] == "rich":
        # Memorization lens must sample rows this arm actually consumes: the
        # feed is sequential from index 0 with no wrap (asserted), so the
        # consumed prefix is exactly range(updates_total * batch).
        max_rows = updates_total * batch
        assert max_rows <= ad.SPLITS["train"]["n"], "arm would wrap train corpus"
        sample_idx = _random.Random(seed + 7).sample(
            range(max_rows), min(TRAIN_SAMPLE_N, max_rows))
        train_sample = [ad.row_at("train", i)[0] for i in sample_idx]
    else:
        train_sample = _random.Random(seed + 7).sample(
            calc.generate(split="train"), TRAIN_SAMPLE_N)
    train_targets = [ad.split_prompt_target(r)[1] for r in train_sample]

    torch.manual_seed(seed)
    model = initialize(spec, seed)
    device = xb.get_device()
    model = model.to(device)
    param_count = sum(int(p.numel()) for p in model.parameters())
    optimizer = build_adamw_optimizer(model, torch_module=torch)

    def gen_eval(rows, targets):
        recs, summ = _eval_slice(model, rows, targets, device=device,
                                 torch_mod=torch, xb=xb)
        return recs, summ

    untrained = {}
    for name, sl in [("dev", (dev_sample, dev_targets))] + \
            [(k, (v["rows"], v["targets"])) for k, v in test_slices.items()] + \
            [("train_sample", (train_sample, train_targets))]:
        _, summ = gen_eval(*sl)
        untrained[name] = summ
    nulls = cev.heuristic_nulls(test_slices["test_core"]["rows"],
                                [ad.row_at("train", i)[0] for i in range(2_000)]
                                if cfg["data"] == "rich" else calc.generate(split="train"))
    null_summaries = {k: cev.summarize(v, test_slices["test_core"]["targets"])
                      for k, v in nulls.items()}
    null_name, null_best = cev.strongest_null_accuracy(null_summaries)

    checkpoints = sorted({max(1, int(updates_total * f)) for f in (0.10, 0.25, 0.50, 1.0)})
    done, ledgers, inter = 0, [], {}
    first_loss, last_loss = None, None
    cap_total, ans_total, whole_total, gsum, gmax, gn = 0, 0, 0, 0.0, 0.0, 0
    t_train0 = time.time()
    for cp in checkpoints:
        n_new = cp - done
        blk = _train_updates(model, optimizer, feed, n_updates=n_new, start_update=done,
                             device=device, torch_mod=torch, batch=batch, length=length,
                             objective=cfg["objective"], lr=3e-4)
        done = cp
        first_loss = blk["first_loss"] if first_loss is None else first_loss
        last_loss = blk["last_loss"]
        cap_total += blk["capacity_tokens"]
        ans_total += blk["answer_supervised"]
        whole_total += blk["whole_supervised"]
        gsum, gn = gsum + blk["grad_norm_mean"] * n_new, gn + n_new
        gmax = max(gmax, blk["grad_norm_max"])
        ledgers.append({"updates": cp, **{k: blk[k] for k in
                                          ("first_loss", "last_loss", "capacity_tokens",
                                           "answer_supervised", "whole_supervised",
                                           "grad_norm_mean", "grad_norm_max")}})
        _, dev_summ = gen_eval(dev_sample, dev_targets)
        dev_ce = _whole_ce(model, dev_sample, device=device, torch_mod=torch,
                           batch_rows=batch, length=length)
        dev_ans = cev.answer_token_ce(model, dev_sample, device=device, xb=xb,
                                      torch_mod=torch)
        inter[str(cp)] = {"dev_exact": dev_summ["accuracy"],
                          "dev_lcb": dev_summ["wilson_lcb"],
                          "dev_ce": dev_ce,
                          "dev_answer_ce": dev_ans["answer_nll_mean"]}
    train_wall = time.time() - t_train0

    trained, trained_recs = {}, {}
    for name, sl in [(k, (v["rows"], v["targets"])) for k, v in test_slices.items()] + \
            [("train_sample", (train_sample, train_targets))]:
        recs, summ = gen_eval(*sl)
        trained[name], trained_recs[name] = summ, recs
    mem_flag = bool(trained["train_sample"]["accuracy"]
                    - trained["test_core"]["accuracy"] >= 0.30)

    ckpt_path = str(Path(out_dir) / f"t1c_arm_{tag.lower()}.pt")
    ckpt_hash = ckpt_mod.save(model, ckpt_path, {"arm": tag, "seed": seed,
                                                 "spec": cfg["spec"],
                                                 "updates": done})
    pre_sha = cev.sha_predictions([r["prediction"] for r in trained_recs["test_core"]])
    del model
    model2 = initialize(spec, seed).to(device)
    ckpt_mod.load_into(model2, ckpt_path)
    xb.mark_step()
    re_recs, _ = _eval_slice(model2, test_slices["test_core"]["rows"],
                                   test_slices["test_core"]["targets"],
                                   device=device, torch_mod=torch, xb=xb)
    post_sha = cev.sha_predictions([r["prediction"] for r in re_recs])
    reload_ok = bool(pre_sha == post_sha)

    u, t = untrained["test_core"], trained["test_core"]
    rules = {
        "nonoverlap": bool(t["accuracy"] > u["accuracy"] and t["wilson_lcb"] > u["wilson_ucb"]),
        "beats_null": bool(t["wilson_lcb"] > null_best),
        "margin": bool(t["accuracy"] - u["accuracy"] >= 0.10),
        "reload": bool(reload_ok),
    }
    status = "PASS" if all(rules.values()) else "FAIL"
    wall = time.time() - t0
    receipt = {
        "schema": "citadel-t1c-arm/v1", "arm": tag, "config": cfg,
        "citadel_sha": rb.citadel_sha(), "cymek_runtime_sha": rt_sha,
        "environment": env, "batch": batch, "sequence_length": length,
        "model": {"spec": cfg["spec"], "parameter_count": param_count},
        "data": unique_note,
        "training": {"updates": done, "ledgers": ledgers,
                      "first_loss": ledgers[0]["first_loss"] if ledgers else None,
                      "last_loss": ledgers[-1]["last_loss"] if ledgers else None,
                      "capacity_tokens": cap_total,
                      "answer_supervised_tokens": ans_total,
                      "whole_supervised_tokens": whole_total,
                      "grad_norm_mean": gsum / max(gn, 1), "grad_norm_max": gmax,
                      "first_update_includes_compile": True,
                      "train_wall_seconds": train_wall},
        "untrained": untrained, "trained": trained,
        "intermediates": inter,
        "diagnostics": {
            "stop_histogram": cev.stop_histogram(trained_recs["test_core"]),
            "samples": cev.sample_records(trained_recs["test_core"], 20, seed),
            "memorization_flag": mem_flag},
        "heuristic_nulls": null_summaries,
        "strongest_heuristic_null": {"name": null_name, "accuracy": null_best},
        "gate_rules": rules,
        "pre_reload_prediction_sha256": pre_sha,
        "post_reload_prediction_sha256": post_sha,
        "reload_identical": reload_ok,
        "checkpoint": {"path": ckpt_path, "sha256": ckpt_hash},
        "device_count": n_devices, "wall_seconds": wall, "status": status,
    }
    out_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    marker.write_text(json.dumps({"receipt": str(out_path), "status": status,
                                  "checkpoint_sha256": ckpt_hash}, indent=2), encoding="utf-8")
    return receipt


CLASSIFY_ORDER = ("CAPABILITY_LEARNED", "OBJECTIVE_LIMITED", "DATA_LIMITED",
                  "SCALE_LIMITED", "MEMORIZATION", "FORMAT_FAILURE",
                  "BUDGET_LIMITED", "INCONCLUSIVE")


def classify_cross_arm(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Preregistered T1C decision rules over arm receipts. Multiple labels may fire."""
    def acc(tag, sl="test_core", trained=True):
        key = "trained" if trained else "untrained"
        return arms[tag][key][sl]["accuracy"]

    def lcb(tag, sl="test_core"):
        return arms[tag]["trained"][sl]["wilson_lcb"]

    def ucb_u(tag, sl="test_core"):
        return arms[tag]["untrained"][sl]["wilson_ucb"]

    fired: dict[str, str] = {}
    present = set(arms)
    if any(arms[t].get("status") == "PASS" for t in arms):
        fired["CAPABILITY_LEARNED"] = "at least one arm passed its gate"
    if {"A", "B"} <= present:
        b, a = acc("B"), acc("A")
        if b - a >= 0.10 and lcb("B") > ucb_u("A"):
            fired["OBJECTIVE_LIMITED"] = f"B-A test gap {b - a:.3f} with separated intervals"
    if {"B", "C"} <= present:
        b, c = acc("B"), acc("C")
        if b - c >= 0.10 and lcb("B") > arms["C"]["trained"]["test_core"]["wilson_lcb"] and \
                lcb("B") > arms["C"]["untrained"]["test_core"]["wilson_ucb"]:
            fired["DATA_LIMITED"] = f"B-C test gap {b - c:.3f} (same objective, richer data wins)"
    if {"B", "D"} <= present:
        b, d = acc("B"), acc("D")
        if d - b >= 0.10 and lcb("D") > ucb_u("B"):
            fired["SCALE_LIMITED"] = f"D-B test gap {d - b:.3f} with separated intervals"
    for t in arms:
        if "trained" not in arms[t] or "diagnostics" not in arms[t]:
            continue  # infra-failed arms carry no measurements
        tr = arms[t]["trained"]["train_sample"]["accuracy"]
        te = arms[t]["trained"]["test_core"]["accuracy"]
        if tr - te >= 0.30 and te < 0.05:
            fired["MEMORIZATION"] = f"arm {t}: train {tr:.2f} vs test {te:.2f}"
            break
    for t in arms:
        if "trained" not in arms[t] or "diagnostics" not in arms[t]:
            continue
        hist = arms[t]["diagnostics"]["stop_histogram"]
        tot = sum(hist.values()) or 1
        bad = hist.get("NON_ALPHABET", 0) + hist.get("PAD", 0)
        if bad / tot > 0.50 and arms[t]["trained"]["test_core"]["accuracy"] < 0.05:
            fired["FORMAT_FAILURE"] = f"arm {t}: {(bad / tot):.0%} non-content stops"
            break
    for t in arms:
        if "trained" not in arms[t]:
            continue
        inter = arms[t].get("intermediates", {})
        keys = sorted((int(k) for k in inter if k.isdigit()))
        if len(keys) >= 2:
            early, late = inter[str(keys[-2])]["dev_exact"], inter[str(keys[-1])]["dev_exact"]
            if late - early >= 0.05 and arms[t]["trained"]["test_core"]["accuracy"] < 0.10:
                fired["BUDGET_LIMITED"] = f"arm {t}: dev still rising at 100% ({early:.2f}->{late:.2f})"
                break
    if not fired:
        fired["INCONCLUSIVE"] = "no rule fired"
    return {"labels": sorted(fired, key=CLASSIFY_ORDER.index),
            "reasons": fired}


def run_session(session_dir: str, *, seed: int = 20260904) -> dict[str, Any]:
    """One-click T1C session: calibrate → manifest → arms A–D → summary → bundle.

    Idempotent resume: existing calibration receipt, manifest, and per-arm
    markers are verified and reused, never recomputed (TEST accounting
    preserved). Per-arm infra failure is recorded; session aborts on the 2nd.
    Prints progress; returns the session manifest dict.
    """
    from citadel_tpu import arith_data as ad
    from citadel_tpu import environment as env_mod
    from citadel_tpu import runtime_bootstrap as rb
    from citadel_tpu import xla_backend as xb

    root = Path(session_dir)
    root.mkdir(parents=True, exist_ok=True)
    print("T1C session starting", flush=True)
    rt_root, rt_sha = rb.ensure_cymek_runtime()
    env = env_mod.probe(require_tpu=True)
    if not env.get("probe_pass"):
        raise env_mod.NoTpuError("ABORT_NO_TPU")
    xb.assert_tpu_active(min_devices=1)

    cal_path = root / "THROUGHPUT_CALIBRATION.json"
    if cal_path.is_file():
        cal = json.loads(cal_path.read_text(encoding="utf-8"))
        shape = (cal["selected"]["batch"], cal["selected"]["length"])
        rate = float(cal["selected_tokens_per_second"])
        print(f"calibration reused: shape={shape} rate={rate:.0f} tok/s", flush=True)
    else:
        cal = calibrate(out=str(cal_path))
        shape = (cal["selected"]["batch"], cal["selected"]["length"])
        rate = float(cal["selected_tokens_per_second"])
        print(f"calibration selected: shape={shape} rate={rate:.0f} tok/s", flush=True)
    budgets = {t: dict(c) for t, c in ARMS.items()}
    scaled = False
    if rate < AUTO_SCALE_RATE:
        for c in budgets.values():
            c["budget"] //= 2
        scaled = True
        print(f"auto-scale rule fired (rate {rate:.0f} < {AUTO_SCALE_RATE:.0f}): budgets halved",
              flush=True)

    man_path = root / "DATA_MANIFEST.json"
    if man_path.is_file():
        manifest = json.loads(man_path.read_text(encoding="utf-8"))
        assert manifest.get("generator_version") == ad.GENERATOR_VERSION, \
            "stale data manifest version"
        print("data manifest reused", flush=True)
    else:
        manifest = ad.build_manifest(out=str(man_path))
        print(f"data manifest built: {manifest['total_bytes']} bytes", flush=True)
    from citadel_tpu import calculator_eval as cev

    if any(v != 0 for v in manifest["leakage"].values()):
        raise RuntimeError(f"abort LEAKAGE: {manifest['leakage']}")
    if manifest.get("max_row_chars", 0) > cev.EVAL_LENGTH:
        raise RuntimeError(
            f"abort ROW_TOO_LONG: max {manifest['max_row_chars']} chars exceeds L={cev.EVAL_LENGTH}")

    arm_receipts: dict[str, Any] = {}
    infra_failures = 0
    for tag in ("A", "B", "C", "D"):
        try:
            receipt = run_arm(tag, budgets[tag], shape=shape, out_dir=str(root), seed=seed)
            arm_receipts[tag] = receipt
            print(f"arm {tag}: {receipt.get('status')} "
                  f"test={receipt['trained']['test_core']['accuracy']:.4f}", flush=True)
        except Exception as exc:  # noqa: BLE001 - per-arm isolation per plan
            infra_failures += 1
            arm_receipts[tag] = {"arm": tag, "status": "IMPLEMENTATION_FAILURE",
                                 "error": f"{type(exc).__name__}: {exc}"}
            print(f"arm {tag}: IMPLEMENTATION_FAILURE {exc}", flush=True)
            if infra_failures >= 2:
                raise RuntimeError("abort SESSION: 2nd infra failure") from exc
    summary = classify_cross_arm(
        {t: r for t, r in arm_receipts.items() if r.get("status") in ("PASS", "FAIL")})
    (root / "CROSS_ARM_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print("cross-arm labels:", summary["labels"], flush=True)
    session = {"schema": "citadel-t1c-session/v1",
               "citadel_sha": rb.citadel_sha(), "cymek_runtime_sha": rt_sha,
               "shape": list(shape), "calibrated_rate": rate, "budgets_scaled": scaled,
               "budgets": {t: c["budget"] for t, c in budgets.items()},
               "arms": {t: r.get("status") for t, r in arm_receipts.items()},
               "labels": summary["labels"], "bundle": "pending"}
    (root / "SESSION_MANIFEST.json").write_text(
        json.dumps(session, indent=2, sort_keys=True), encoding="utf-8")
    bundle = build_bundle(str(root), out=str(root / "CITADEL_T1C_RESULTS.zip"))
    session["bundle"] = bundle["zip_bytes"]
    (root / "SESSION_MANIFEST.json").write_text(
        json.dumps(session, indent=2, sort_keys=True), encoding="utf-8")
    print("session complete:", session["labels"], flush=True)
    return session


def build_bundle(session_dir: str, *, out: str) -> dict[str, Any]:
    """Assemble CITADEL_T1C_RESULTS.zip (receipts + manifest; binaries per cap rule)."""
    root = Path(session_dir)
    names = ["SESSION_MANIFEST.json", "THROUGHPUT_CALIBRATION.json", "DATA_MANIFEST.json",
             "ARM_A.json", "ARM_B.json", "ARM_C.json", "ARM_D.json", "CROSS_ARM_SUMMARY.json"]
    missing = [n for n in names if not (root / n).is_file()]
    if missing:
        raise RuntimeError(f"bundle incomplete, missing: {', '.join(missing)}")
    ckpts = sorted(str(p) for p in root.glob("t1c_arm_*.pt"))
    ckpt_bytes = sum(Path(p).stat().st_size for p in ckpts)
    bundle = {"files": names, "checkpoints": ckpts, "checkpoint_bytes": ckpt_bytes,
              "checkpoints_bundled": ckpt_bytes <= CKPT_ZIP_BYTES_CAP}
    zp = Path(out)
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
        for n in names:
            zf.write(root / n, n)
        if bundle["checkpoints_bundled"]:
            for p in ckpts:
                zf.write(p, Path(p).name)
    bundle["zip"] = str(zp)
    bundle["zip_bytes"] = zp.stat().st_size
    (root / "BUNDLE_MANIFEST.json").write_text(json.dumps(bundle, indent=2, sort_keys=True),
                                               encoding="utf-8")
    return bundle


__all__ = [
    "ARMS",
    "AUTO_SCALE_RATE",
    "CALIBRATION_SHAPES",
    "DEV_SAMPLE_N",
    "MID_EXPECTED_PARAMS",
    "MID_SPEC_KWARGS",
    "MINI_EXPECTED_PARAMS",
    "TRAIN_SAMPLE_N",
    "answer_spans",
    "build_bundle",
    "build_spec",
    "calibrate",
    "classify_cross_arm",
    "run_arm",
    "run_session",
]
