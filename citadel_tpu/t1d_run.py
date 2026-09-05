"""T1D arm runner: lift-off discriminator (one session). Platform-neutral TPU.

Arms A (flat) / B (curriculum) / C (teacher) / D (SCALE2) / E (masked-softmax
diagnostic). Shared calibration (one static packed shape for all arms),
deterministic tiered data, per-arm receipts with tier lift-off curves,
hash-verified resume markers, machine-evaluated cross-arm rules, one bundle.
Cymek production model/loss/optimizer/checkpoint paths reused unchanged;
answer/teacher supervision uses the production `eligible` seam; packing honors
production segment-isolation semantics (`packed_layout`).
"""

from __future__ import annotations

import hashlib
import json
import time
import zipfile
from pathlib import Path
from typing import Any


SCALE2_SPEC_KWARGS: dict[str, Any] = {
    "schema": "anra-v5-model-spec/v1",
    "family": "dense-decoder-transformer",
    "vocabulary_size": 24_576,
    "width": 192,
    "layers": 8,
    "query_heads": 12,
    "kv_heads": 6,
    "head_dimension": 16,
    "ffn_width": 384,
    "context_length": 4_096,
    "rope_base": 10_000.0,
    "norm_epsilon": 1e-5,
    "tied_embeddings": True,
    "qk_norm": True,
    "qk_norm_affine": True,
    "linear_bias": False,
    "dropout": 0.0,
}
SCALE2_EXPECTED_PARAMS = 7_378_368
TRAIN_LENGTH = 64
EVAL_BATCH = 8
CALIBRATION_SHAPES = [(64, 64), (128, 64), (256, 64), (512, 64), (1024, 64)]
CALIBRATION_UPDATES = 5
ARMS: dict[str, dict[str, Any]] = {
    "A": {"spec": "MID", "mode": "flat", "budget": 8_000_000},
    "B": {"spec": "MID", "mode": "curriculum", "budget": 8_000_000},
    "C": {"spec": "MID", "mode": "teacher", "budget": 8_000_000},
    "D": {"spec": "SCALE2", "mode": "curriculum", "budget": 4_000_000},
    "E": {"spec": "MID", "mode": "masked", "budget": 4_000_000},
}
ARM_TIME_BOX_S = 45 * 60
AUTO_SCALE_RATE = 5_000.0
CKPT_ZIP_BYTES_CAP = 200_000_000
LIFT_THRESHOLD = 0.20
LIFT_MIN_N = 200
TRAIN_SAMPLE_PER_TIER = 200


def build_spec(which: str):
    """Construct + validate MID (T1C-certified builder) / SCALE2 specs."""
    if which == "MID":
        from citadel_tpu import t1c_run as t1c

        return t1c.build_spec("MID")
    if which == "SCALE2":
        from v5_contracts.model_spec import ModelSpec

        spec = ModelSpec(**SCALE2_SPEC_KWARGS)
        spec.assert_valid()
        total = spec.parameter_receipt().total
        if total != SCALE2_EXPECTED_PARAMS:
            raise ValueError(f"SCALE2 receipt {total} != expected {SCALE2_EXPECTED_PARAMS}")
        return spec
    raise ValueError(f"unknown spec {which!r}")


def pack_rows(texts: list[str], length: int) -> tuple[list[list[tuple[int, int]]], list[tuple[int, int, int, int]]]:
    """Deterministic first-fit packing; segment ids are per-sequence ordinals.

    Returns (sequences, placements): sequences[s] = [(seg_id, row_len)],
    placements[i] = (seq, seg_id, start, len) for input row i. Padding tails
    carry -1 per the packed_layout contract. Raises if any row exceeds length.
    """
    seqs: list[list[tuple[int, int]]] = []
    placements: list[tuple[int, int, int, int]] = []
    for t in texts:
        n = len(t)
        if n > length:
            raise ValueError(f"row exceeds fixed length {length}: {t!r}")
        placed = False
        for s, seq in enumerate(seqs):
            used = sum(ln for _, ln in seq)
            if used + n <= length:
                seg_id = len(seq)
                seq.append((seg_id, n))
                placements.append((s, seg_id, used, n))
                placed = True
                break
        if not placed:
            seqs.append([(0, n)])
            placements.append((len(seqs) - 1, 0, 0, n))
    return seqs, placements


def valid_alphabet_ids() -> list[int]:
    """Closed output vocabulary for Arm E: decodable ids + PAD/EOS stop signals."""
    from citadel_tpu import calculator_eval as cev

    return sorted(set(cev.DECODABLE_IDS) | {cev.PAD_ID, cev.EOS_ID})


def assemble_batch(seq_texts: list[list[str]], *, length: int, torch_mod: Any):
    """Build fixed [S, L] tensors from pre-packed sequences of row strings.

    Segment id = row ordinal within its sequence; padding tails carry -1 per
    the packed_layout contract. Eligible covers exactly the answer spans.
    """
    from citadel_tpu import calculator_eval as cev
    from citadel_tpu import t1c_run as t1c

    torch = torch_mod
    n_seq = len(seq_texts)
    tokens = torch.full((n_seq, length), cev.PAD_ID, dtype=torch.long)
    seg_ids = torch.full((n_seq, length), -1, dtype=torch.long)
    eligible = torch.zeros((n_seq, length), dtype=torch.bool)
    real = ans = 0
    for s, rows in enumerate(seq_texts):
        pos = 0
        for j, t in enumerate(rows):
            ids = cev.encode(t)
            n = len(ids)
            if pos + n > length:
                raise ValueError(f"packed sequence overflows fixed length {length}")
            tokens[s, pos:pos + n] = torch.tensor(ids, dtype=torch.long)
            seg_ids[s, pos:pos + n] = j
            plen, alen = t1c.answer_spans([t], length)[0]
            assert alen > 0, f"empty answer span: {t!r}"
            eligible[s, pos + plen:pos + plen + alen] = True
            pos += n
            real += n
            ans += alen
    return tokens, seg_ids, eligible, {"real": real, "pad": n_seq * length - real,
                                       "answer": ans, "sequences": n_seq,
                                       "rows": sum(len(r) for r in seq_texts)}


def _train_updates_packed(model, optimizer, feeder, *, n_updates: int,
                          start_update: int, updates_total: int, device,
                          torch_mod, length: int, masked: bool,
                          valid_ids: list[int] | None,
                          time_box_s: float | None = None,
                          t_start: float | None = None):
    """Packed update loop with exact accounting + mechanical eligible check."""
    from v5_model.core import packed_layout
    from v5_objectives.causal_lm import causal_lm_loss
    from citadel_tpu import xla_backend as xb

    torch = torch_mod
    allow = None
    if masked:
        assert valid_ids is not None
        allow = torch.zeros(24_576, dtype=torch.bool, device=device)
        allow[torch.tensor(valid_ids, dtype=torch.long, device=device)] = True
    first_loss, last_loss, gsum, gmax, gn = None, None, 0.0, 0.0, 0
    cap = ans_sup = whole_sup = real_tok = 0
    for u in range(n_updates):
        if time_box_s is not None and t_start is not None \
                and time.time() - t_start > time_box_s:
            raise TimeoutError(f"arm time box exceeded ({time_box_s:.0f}s)")
        frac = (start_update + u) / max(updates_total, 1)
        seqs = feeder.fill_sequences(frac)
        tokens, seg_ids, eligible, stats = assemble_batch(
            seqs, length=length, torch_mod=torch)
        pos, mask = packed_layout(seg_ids, torch_module=torch)  # host (audit A8)
        logits = model(tokens.to(device), pos.to(device), mask.to(device))
        if masked:
            logits = torch.where(allow, logits, torch.full_like(logits, float("-inf")))
        loss, count = causal_lm_loss(logits, tokens.to(device), seg_ids.to(device),
                                     eligible=eligible.to(device), torch_module=torch)
        if int(count) != stats["answer"]:
            raise RuntimeError(
                f"abort ELIGIBLE_MISMATCH: loss count {count} != answer span {stats['answer']}")
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
        cap += stats["sequences"] * length
        ans_sup += stats["answer"]
        whole_sup += stats["real"] - stats["answer"]
        real_tok += stats["real"]
    return {"first_loss": first_loss, "last_loss": last_loss,
            "capacity_tokens": cap, "answer_supervised": ans_sup,
            "prompt_tokens_unsupervised": whole_sup, "real_tokens": real_tok,
            "grad_norm_mean": gsum / max(gn, 1), "grad_norm_max": gmax}


def _tier_slices():
    """Materialize frozen DEV/TEST tier slices + targets (sizes from tiered_data)."""
    from citadel_tpu import calculator_eval as cev
    from citadel_tpu import tiered_data as td

    out = {}
    for split, n in (("dev", td.EVAL_DEV_N), ("test", td.EVAL_TEST_N)):
        for tier in range(5):
            key = f"{split}_t{tier}"
            rows = [td.tier_row(tier, split, j)[0] for j in range(n)]
            out[key] = {"rows": rows,
                        "targets": [cev.split_prompt_target(r)[1] for r in rows]}
    return out


def _gen_eval(model, rows, targets, *, device, torch_mod, xb, allow_ids=None,
              stats=False):
    from citadel_tpu import calculator_eval as cev

    recs = cev.generate(rows, model, xb, device=device, torch_mod=torch_mod,
                        allow_ids=allow_ids, first_step_stats=stats)
    return recs, cev.summarize([r["prediction"] for r in recs], targets)


def _lift_tier(accuracy: float, n: int) -> bool:
    return n >= LIFT_MIN_N and accuracy >= LIFT_THRESHOLD


def calibrate(*, out: str | None = None, updates: int = CALIBRATION_UPDATES) -> dict[str, Any]:
    """Pick one static packed (batch, length) for ALL arms: max steady tok/s
    that fits and passes a finite-loss correctness step with exact accounting."""
    from citadel_tpu import environment as env_mod
    from citadel_tpu import runtime_bootstrap as rb
    from citadel_tpu import xla_backend as xb

    rt_root, rt_sha = rb.ensure_cymek_runtime()
    env = env_mod.probe(require_tpu=True)
    if not env.get("probe_pass"):
        raise env_mod.NoTpuError("ABORT_NO_TPU")
    xb.assert_tpu_active(min_devices=1)
    import torch

    from v5_model.core import initialize
    from v5_training.optimizer import build_adamw_optimizer

    spec = build_spec("MID")
    results = []
    for batch, length in CALIBRATION_SHAPES:
        torch.manual_seed(20260904)
        model = initialize(spec, 20260904)
        device = xb.get_device()
        model = model.to(device)
        opt = build_adamw_optimizer(model, torch_module=torch)
        feeder = TierFeeder("flat", batch, length)
        try:
            import time as _time

            first_s, second_s, steady_s, mem = None, None, None, "unavailable"
            t = _time.time()
            rep = _train_updates_packed(
                model, opt, feeder, n_updates=1, start_update=0,
                updates_total=updates, device=device, torch_mod=torch,
                length=length, masked=False, valid_ids=None)
            first_s = _time.time() - t
            t = _time.time()
            rep2 = _train_updates_packed(
                model, opt, feeder, n_updates=1, start_update=1,
                updates_total=updates, device=device, torch_mod=torch,
                length=length, masked=False, valid_ids=None)
            second_s = _time.time() - t
            t = _time.time()
            rep3 = _train_updates_packed(
                model, opt, feeder, n_updates=max(updates - 2, 1), start_update=2,
                updates_total=updates, device=device, torch_mod=torch,
                length=length, masked=False, valid_ids=None)
            steady_s = (_time.time() - t) / max(updates - 2, 1)
            try:
                import torch_xla.core.xla_model as _xm

                info = _xm.get_memory_info(device)
                mem = {k: int(v) for k, v in dict(info).items()}
            except Exception:
                mem = "unavailable"
            wall = (first_s or 0) + (second_s or 0) + steady_s * max(updates - 2, 1)
            toks = rep["capacity_tokens"] * updates / max(wall, 1e-9)
            ok = rep["first_loss"] is not None and rep["first_loss"] == rep["first_loss"]
            recompile = bool(steady_s > 0 and second_s > 2 * steady_s)
            results.append({"batch": batch, "length": length,
                            "tokens_per_second": toks, "correct": bool(ok),
                            "first_step_seconds": first_s,
                            "second_step_seconds": second_s,
                            "steady_seconds_per_update": steady_s,
                            "unexpected_recompile": recompile,
                            "memory": mem,
                            "sequences_per_update": batch,
                            "updates": updates})
        except Exception as exc:  # noqa: BLE001 - record, do not propagate
            results.append({"batch": batch, "length": length, "tokens_per_second": 0.0,
                            "correct": False, "error": f"{type(exc).__name__}: {exc}"})
        del model
    feasible = [r for r in results if r["correct"]]
    if not feasible:
        raise RuntimeError("abort CALIBRATION_FAILURE: no candidate shape passed")
    best = max(feasible, key=lambda r: r["tokens_per_second"])
    receipt = {"schema": "citadel-t1d-throughput-calibration/v1",
               "citadel_sha": rb.citadel_sha(), "cymek_runtime_sha": rt_sha,
               "environment": env, "candidates": results,
               "selected": {"batch": best["batch"], "length": best["length"]},
               "selected_tokens_per_second": best["tokens_per_second"]}
    if out is not None:
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    return receipt


def run_arm(tag: str, cfg: dict[str, Any], *, shape: tuple[int, int],
            out_dir: str, seed: int = 20260904) -> dict[str, Any]:
    """Execute one frozen T1D arm end to end; receipt + checkpoint + marker."""
    import random as _random

    from citadel_tpu import calculator_eval as cev
    from citadel_tpu import checkpoint as ckpt_mod
    from citadel_tpu import environment as env_mod
    from citadel_tpu import runtime_bootstrap as rb
    from citadel_tpu import tiered_data as td
    from citadel_tpu import xla_backend as xb

    n_seq, length = shape
    masked = cfg["mode"] == "masked"
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
    from v5_training.optimizer import build_adamw_optimizer
    from v5_model.core import initialize

    spec = build_spec(cfg["spec"])
    _ = from_spec(spec, qk_norm_epsilon=QK_NORM_EPSILON)
    budget = int(cfg["budget"])
    updates_total = budget // (n_seq * length)
    feeder = TierFeeder(cfg["mode"] if cfg["mode"] != "masked" else "curriculum",
                        n_seq, length)

    slices = _tier_slices()
    allow_ids = valid_alphabet_ids() if masked else None

    torch.manual_seed(seed)
    model = initialize(spec, seed)
    device = xb.get_device()
    model = model.to(device)
    param_count = sum(int(p.numel()) for p in model.parameters())
    optimizer = build_adamw_optimizer(model, torch_module=torch)

    def gen(rows, targets):
        return _gen_eval(model, rows, targets, device=device, torch_mod=torch,
                         xb=xb, allow_ids=allow_ids)

    untrained: dict[str, Any] = {}
    for tier in range(5):
        _, summ = gen(slices[f"dev_t{tier}"]["rows"], slices[f"dev_t{tier}"]["targets"])
        untrained[f"dev_t{tier}"] = summ
        _, summ = gen(slices[f"test_t{tier}"]["rows"], slices[f"test_t{tier}"]["targets"])
        untrained[f"test_t{tier}"] = summ
    train_sample_idx: dict[int, list[int]] = {}
    untrained_train: dict[str, Any] = {}
    for tier in range(5):
        consumed = feeder.placed_rows.get(f"tier:{tier}", 0)
        idx = _random.Random(seed + 11).sample(range(max(consumed, 1)),
                                               min(TRAIN_SAMPLE_PER_TIER, max(consumed, 1)))
        train_sample_idx[tier] = idx
        rows = [td.tier_row(tier, "train", i)[0] for i in idx]
        tgts = [cev.split_prompt_target(r)[1] for r in rows]
        _, summ = gen(rows, tgts)
        untrained_train[f"t{tier}"] = summ

    checkpoints = sorted({max(1, int(updates_total * f)) for f in (0.25, 0.50, 0.75, 1.0)})
    done, ledgers, inter = 0, [], {}
    first_loss, last_loss = None, None
    cap_total, ans_total, whole_total = 0, 0, 0
    gsum, gmax, gn = 0.0, 0.0, 0
    t_train0 = time.time()
    for cp in checkpoints:
        n_new = cp - done
        blk = _train_updates_packed(
            model, optimizer, feeder, n_updates=n_new, start_update=done,
            updates_total=updates_total, device=device, torch_mod=torch,
            length=length, masked=masked,
            valid_ids=allow_ids if masked else None,
            time_box_s=ARM_TIME_BOX_S, t_start=t_train0)
        done = cp
        first_loss = blk["first_loss"] if first_loss is None else first_loss
        last_loss = blk["last_loss"]
        cap_total += blk["capacity_tokens"]
        ans_total += blk["answer_supervised"]
        whole_total += blk["prompt_tokens_unsupervised"]
        gsum, gn = gsum + blk["grad_norm_mean"] * n_new, gn + n_new
        gmax = max(gmax, blk["grad_norm_max"])
        ledgers.append({"updates": cp, **{k: blk[k] for k in
                                          ("first_loss", "last_loss", "capacity_tokens",
                                           "answer_supervised", "prompt_tokens_unsupervised",
                                           "real_tokens", "grad_norm_mean", "grad_norm_max")}})
        dev_curve = {}
        for tier in range(5):
            _, summ = gen(slices[f"dev_t{tier}"]["rows"], slices[f"dev_t{tier}"]["targets"])
            dev_curve[f"t{tier}"] = {"exact": summ["accuracy"], "lcb": summ["wilson_lcb"]}
        inter[str(cp)] = dev_curve
    train_wall = time.time() - t_train0

    trained: dict[str, Any] = {}
    trained_recs: dict[str, Any] = {}
    for tier in range(5):
        recs, summ = gen(slices[f"test_t{tier}"]["rows"], slices[f"test_t{tier}"]["targets"])
        trained[f"t{tier}"], trained_recs[f"t{tier}"] = summ, recs
    trained_train: dict[str, Any] = {}
    for tier in range(5):
        idx = train_sample_idx[tier]
        rows = [td.tier_row(tier, "train", i)[0] for i in idx]
        tgts = [cev.split_prompt_target(r)[1] for r in rows]
        _, summ = gen(rows, tgts)
        trained_train[f"t{tier}"] = summ

    all_test_rows, all_test_tgts = [], []
    for tier in range(5):
        all_test_rows.extend(slices[f"test_t{tier}"]["rows"])
        all_test_tgts.extend(slices[f"test_t{tier}"]["targets"])
    ref_rows: list[str] = []
    for key, cnt in feeder.placed_rows.items():
        if key.startswith("tier:") and cnt > 0:
            t = int(key.split(":")[1])
            idx = _random.Random(seed + 13).sample(range(cnt), min(200, cnt))
            ref_rows.extend(td.tier_row(t, "train", i)[0] for i in idx)
    nulls = cev.heuristic_nulls(all_test_rows, ref_rows)
    null_summaries = {k: cev.summarize(v, all_test_tgts) for k, v in nulls.items()}
    null_name, null_best = cev.strongest_null_accuracy(null_summaries)
    nulls_per_tier = {}
    for tier in range(5):
        tn = cev.heuristic_nulls(slices[f"test_t{tier}"]["rows"], ref_rows)
        ts = {k: cev.summarize(v, slices[f"test_t{tier}"]["targets"])
              for k, v in tn.items()}
        nn, nb = cev.strongest_null_accuracy(ts)
    nulls_per_tier[f"t{tier}"] = {"strongest": nn, "accuracy": nb,
                                       "all": {k: v["accuracy"] for k, v in ts.items()}}

    # Failure-cause probes (diagnostic-only; gates unchanged). All pure
    # aggregations except the two extra generation evals below.
    all_trained_recs = [r for t in range(5) for r in trained_recs[f"t{t}"]]
    teacher_eval: dict[str, Any] = {}
    consumed_teacher = max(list(feeder.teacher_cursors.values()) + [0])
    if consumed_teacher < 900_000:
        from citadel_tpu import tiered_data as _td

        t_rows, t_tgts = [], []
        for _k in ("digadd", "digsub", "singlemul", "divmicro"):
            for j in range(50):
                r, _ = _td.teacher_row(_k, 900_000 + j)
                t_rows.append(r)
                t_tgts.append(cev.split_prompt_target(r)[1])
        t_recs, t_summ = _gen_eval(model, t_rows, t_tgts, device=device,
                                   torch_mod=torch, xb=xb, allow_ids=allow_ids)
        teacher_eval = {"n": len(t_rows), "summary": t_summ,
                        "stop_histogram": cev.stop_histogram(t_recs)}
    else:
        teacher_eval = {"skipped": "teacher consumption exceeded held-out band"}
    stats_recs, _ = _gen_eval(model, slices["test_t1"]["rows"][:200],
                              slices["test_t1"]["targets"][:200],
                              device=device, torch_mod=torch, xb=xb,
                              allow_ids=allow_ids, stats=True)
    ent = [r["first_entropy_nats"] for r in stats_recs
           if r.get("first_entropy_nats") is not None]
    digit_ids = {cev.encode_char(c) for c in "0123456789"}
    top1_digit = sum(1 for r in stats_recs
                     if (r.get("first_top5_ids") or [None])[0] in digit_ids)
    first_step = {"n": len(stats_recs),
                  "mean_entropy_nats": (sum(ent) / len(ent)) if ent else None,
                  "top1_digit_rate": (top1_digit / len(stats_recs)) if stats_recs else 0.0}

    def _digits(s: str) -> list[str]:
        v = cev.normalize_answer(s)
        return list(str(abs(v))) if v is not None else []

    pos_acc = cev.position_accuracy(all_trained_recs)
    length_dist = cev.length_distribution(all_trained_recs)
    digit_hist = cev.digit_histograms(all_trained_recs)
    digit_correct = sum(
        1 for r in all_trained_recs if r.get("valid", True)
        for a, b in zip(_digits(r["prediction"]), _digits(r["target"])) if a == b)
    digit_total = sum(
        min(len(_digits(r["prediction"])), len(_digits(r["target"])))
        for r in all_trained_recs if r.get("valid", True))
    easy_mem = {}
    for t in (0, 1, 2):
        tra = trained_train[f"t{t}"]["accuracy"]
        tee = trained[f"t{t}"]["accuracy"]
        easy_mem[f"t{t}"] = {"train": tra, "test": tee, "gap": tra - tee}

    ckpt_path = str(Path(out_dir) / f"t1d_arm_{tag.lower()}.pt")
    ckpt_hash = ckpt_mod.save(model, ckpt_path, {"arm": tag, "seed": seed,
                                                 "spec": cfg["spec"], "updates": done})
    pre_recs = [r for t in range(5) for r in trained_recs[f"t{t}"]]
    pre_sha = cev.sha_predictions([r["prediction"] for r in pre_recs])
    del model
    model2 = initialize(spec, seed).to(device)
    ckpt_mod.load_into(model2, ckpt_path)
    xb.mark_step()
    post_preds = []
    for tier in range(5):
        recs, _ = _gen_eval(model2, slices[f"test_t{tier}"]["rows"],
                            slices[f"test_t{tier}"]["targets"],
                            device=device, torch_mod=torch, xb=xb,
                            allow_ids=allow_ids)
        post_preds.extend(r["prediction"] for r in recs)
    post_sha = cev.sha_predictions(post_preds)
    reload_ok = bool(pre_sha == post_sha)

    lift_train = {f"t{t}": {"accuracy": trained_train[f"t{t}"]["accuracy"],
                            "lift": _lift_tier(trained_train[f"t{t}"]["accuracy"],
                                              trained_train[f"t{t}"]["total"])}
                  for t in range(5) if t != 0}
    lift_test = {f"t{t}": {"accuracy": trained[f"t{t}"]["accuracy"],
                           "lift": _lift_tier(trained[f"t{t}"]["accuracy"],
                                             trained[f"t{t}"]["total"])}
                 for t in range(5) if t != 0}

    def first_lift(d):
        for t in (1, 2, 3, 4):
            if d[f"t{t}"]["lift"]:
                return t
        return None

    wall = time.time() - t0
    receipt = {
        "schema": "citadel-t1d-arm/v1", "arm": tag, "config": cfg,
        "citadel_sha": rb.citadel_sha(), "cymek_runtime_sha": rt_sha,
        "environment": env, "batch_sequences": n_seq, "sequence_length": length,
        "model": {"spec": cfg["spec"], "parameter_count": param_count},
        "data": {"feeder": feeder.ledger()},
        "training": {"updates": done, "ledgers": ledgers,
                      "first_loss": ledgers[0]["first_loss"] if ledgers else None,
                      "last_loss": ledgers[-1]["last_loss"] if ledgers else None,
                      "capacity_tokens": cap_total,
                      "answer_supervised_tokens": ans_total,
                      "prompt_tokens_unsupervised": whole_total,
                      "grad_norm_mean": gsum / max(gn, 1), "grad_norm_max": gmax,
                      "train_wall_seconds": train_wall},
        "untrained": untrained, "untrained_train": untrained_train,
        "trained": trained, "trained_train": trained_train,
        "intermediates": inter,
        "diagnostics": {
            "stop_histogram": cev.stop_histogram(pre_recs),
            "samples": cev.sample_records(pre_recs, 20, seed),
            "first_train_lift_tier": first_lift(lift_train),
            "first_test_lift_tier": first_lift(lift_test),
            "teacher_eval": teacher_eval,
            "first_step": first_step,
            "position_accuracy": pos_acc,
            "length_distribution": length_dist,
            "digit_histograms": digit_hist,
            "digit_level_accuracy": {
                "correct": digit_correct, "total": digit_total,
                "accuracy": (digit_correct / digit_total) if digit_total else 0.0},
            "easy_memorization": easy_mem},
        "heuristic_nulls": null_summaries,
        "nulls_per_tier": nulls_per_tier,
        "strongest_heuristic_null": {"name": null_name, "accuracy": null_best},
        "gate_rules": {"reload": bool(reload_ok)},
        "pre_reload_prediction_sha256": pre_sha,
        "post_reload_prediction_sha256": post_sha,
        "reload_identical": reload_ok,
        "checkpoint": {"path": ckpt_path, "sha256": ckpt_hash},
        "device_count": n_devices, "wall_seconds": wall,
        "status": "PASS" if reload_ok else "IMPLEMENTATION_FAILURE",
    }
    out_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    marker.write_text(json.dumps({"receipt": str(out_path), "status": receipt["status"],
                                  "checkpoint_sha256": ckpt_hash}, indent=2), encoding="utf-8")
    return receipt


def _pooled(arm: dict[str, Any], key: str) -> tuple[int, int]:
    from citadel_tpu import calculator_eval as cev

    k = sum(arm[key][f"t{t}"]["correct"] for t in (1, 2, 3, 4))
    n = sum(arm[key][f"t{t}"]["total"] for t in (1, 2, 3, 4))
    return k, n


CLASSIFY_ORDER = ("CAPABILITY_LIFTED", "CURRICULUM_HELPED", "TEACHER_HELPED",
                  "SCALE_HELPED", "REPRESENTATION_LIMITED", "BELOW_FIT_FLOOR",
                  "GENERALIZATION_LIMITED", "COMPLEXITY_FRONTIER", "FORMAT_FAILURE",
                  "BUDGET_LIMITED", "INCONCLUSIVE")


def classify_cross_arm(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Preregistered T1D decision rules. Multiple labels may fire."""
    from citadel_tpu import calculator_eval as cev

    fired: dict[str, str] = {}
    present = set(arms)
    pooled: dict[str, Any] = {}
    for t in present:
        k, n = _pooled(arms[t], "trained")
        uk, un = _pooled(arms[t], "untrained")
        lcb, ucb = cev.wilson(k, n)
        ulcb, uucb = cev.wilson(uk, un)
        null_ref = max(arms[t]["nulls_per_tier"][f"t{i}"]["accuracy"] for i in (1, 2, 3, 4))
        pooled[t] = {"acc": k / n if n else 0.0, "lcb": lcb, "ucb": ucb,
                     "untrained_acc": uk / un if un else 0.0,
                     "untrained_ucb": uucb, "null": null_ref,
                     "status": arms[t].get("status")}
        gate = (lcb > max(null_ref, uucb) + 0.10
                and arms[t].get("reload_identical") is True)
        pooled[t]["gate"] = bool(gate)
    if any(pooled[t]["gate"] for t in pooled):
        fired["CAPABILITY_LIFTED"] = "an arm beats null+untrained by 0.10 with reload identity"
    if {"A", "B"} <= present:
        diffs = [arms["B"]["trained"][f"t{t}"]["accuracy"]
                 - arms["A"]["trained"][f"t{t}"]["accuracy"] for t in (1, 2, 3, 4)]
        if sum(diffs) / 4 >= 0.15 and sum(1 for d in diffs if d > 0) >= 3:
            fired["CURRICULUM_HELPED"] = f"mean B-A tiers1-4 = {sum(diffs) / 4:.3f}"
    if {"B", "C"} <= present:
        diffs = [arms["C"]["trained"][f"t{t}"]["accuracy"]
                 - arms["B"]["trained"][f"t{t}"]["accuracy"] for t in (1, 2, 3, 4)]
        if sum(diffs) / 4 >= 0.15 and sum(1 for d in diffs if d > 0) >= 3:
            fired["TEACHER_HELPED"] = f"mean C-B tiers1-4 = {sum(diffs) / 4:.3f}"
    if {"B", "D"} <= present:
        diffs = [arms["D"]["trained"][f"t{t}"]["accuracy"]
                 - arms["B"]["trained"][f"t{t}"]["accuracy"] for t in (1, 2, 3, 4)]
        if sum(diffs) / 4 >= 0.15 and sum(1 for d in diffs if d > 0) >= 3:
            fired["SCALE_HELPED"] = f"mean D-B tiers1-4 = {sum(diffs) / 4:.3f}"
    if {"B", "E"} <= present:
        diffs = [arms["E"]["trained"][f"t{t}"]["accuracy"]
                 - arms["B"]["trained"][f"t{t}"]["accuracy"] for t in (1, 2, 3, 4)]
        if sum(diffs) / 4 >= 0.15 and sum(1 for d in diffs if d > 0) >= 3:
            fired["REPRESENTATION_LIMITED"] = f"mean E-B tiers1-4 = {sum(diffs) / 4:.3f} (diagnostic)"
    easy_ok = True
    for t in present:
        for tier in (0, 1, 2):
            if arms[t]["trained_train"][f"t{tier}"]["accuracy"] >= 0.05:
                easy_ok = False
    if easy_ok and present:
        fired["BELOW_FIT_FLOOR"] = "train exact < 0.05 on tiers 0-2 in every arm"
    for t in present:
        tr = max(arms[t]["trained_train"][f"t{i}"]["accuracy"] for i in range(5))
        te = pooled[t]["acc"]
        if tr - te >= 0.30 and te < 0.05:
            fired["GENERALIZATION_LIMITED"] = f"arm {t}: train {tr:.2f} vs pooled test {te:.2f}"
            break
    for t in present:
        e12 = max(arms[t]["trained"][f"t{i}"]["accuracy"] for i in (1, 2))
        e4 = arms[t]["trained"]["t4"]["accuracy"]
        if e12 >= 0.50 and e4 < 0.05:
            fired["COMPLEXITY_FRONTIER"] = f"arm {t}: tiers1-2 {e12:.2f}, tier4 {e4:.2f}"
            break
    for t in present:
        hist = arms[t].get("diagnostics", {}).get("stop_histogram", {})
        tot = sum(hist.values()) or 1
        bad = hist.get("NON_ALPHABET", 0) + hist.get("PAD", 0)
        if bad / tot > 0.50 and pooled[t]["acc"] < 0.05:
            fired["FORMAT_FAILURE"] = f"arm {t}: {(bad / tot):.0%} non-content stops"
            break
    for t in present:
        inter = arms[t].get("intermediates", {})
        keys = sorted(int(k) for k in inter if str(k).isdigit())
        if len(keys) >= 2:
            early = inter[str(keys[-2])].get("dev_exact")
            late = inter[str(keys[-1])].get("dev_exact")
            te = pooled[t]["acc"]
            if early is not None and late is not None and late - early >= 0.05 and te < 0.10:
                fired["BUDGET_LIMITED"] = (
                    f"arm {t}: dev still rising at 100% ({early:.2f}->{late:.2f})")
                break
    if not fired:
        fired["INCONCLUSIVE"] = "no rule fired"
    return {"labels": sorted(fired, key=CLASSIFY_ORDER.index),
            "reasons": fired, "pooled": pooled}


def run_session(session_dir: str, *, seed: int = 20260904) -> dict[str, Any]:
    """One-click T1D session: calibrate → manifest → arms → summary → bundle.

    Idempotent resume: calibration receipt, manifest, and per-arm markers are
    verified and reused, never recomputed (TEST accounting preserved).
    Per-arm infra failure is recorded; session aborts on the 2nd.
    """
    from citadel_tpu import environment as env_mod
    from citadel_tpu import runtime_bootstrap as rb
    from citadel_tpu import tiered_data as td
    from citadel_tpu import xla_backend as xb

    root = Path(session_dir)
    root.mkdir(parents=True, exist_ok=True)
    print("T1D session starting", flush=True)
    rt_root, rt_sha = rb.ensure_cymek_runtime()
    env = env_mod.probe(require_tpu=True)
    if not env.get("probe_pass"):
        raise env_mod.NoTpuError("ABORT_NO_TPU")
    xb.assert_tpu_active(min_devices=1)

    cal_path = root / "CALIBRATION.json"
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
        assert manifest.get("generator_version") == td.GENERATOR_VERSION, \
            "stale data manifest version"
        print("data manifest reused", flush=True)
    else:
        manifest = td.build_manifest(out=str(man_path))
        print(f"data manifest built: {manifest['total_bytes']} bytes", flush=True)
    fatal, _ = td.leakage_verdict(manifest["leakage"])
    if fatal:
        raise RuntimeError(f"abort LEAKAGE: {fatal}")
    if manifest.get("max_row_chars", 0) > 64:
        raise RuntimeError(f"abort ROW_TOO_LONG: {manifest['max_row_chars']}")

    arm_receipts: dict[str, Any] = {}
    infra_failures = 0
    for tag in ("A", "B", "C", "D", "E"):
        try:
            receipt = run_arm(tag, budgets[tag], shape=shape, out_dir=str(root), seed=seed)
            arm_receipts[tag] = receipt
            print(f"arm {tag}: {receipt.get('status')} "
                  f"pooled14={sum(receipt['trained'][f't{t}']['accuracy'] for t in (1, 2, 3, 4)) / 4:.4f}",
                  flush=True)
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
    curves = {}
    for tag, r in arm_receipts.items():
        if r.get("status") not in ("PASS", "FAIL"):
            curves[tag] = {"status": r.get("status")}
            continue
        curves[tag] = {
            "status": r.get("status"),
            "train": {f"t{t}": r["trained_train"][f"t{t}"]["accuracy"] for t in range(5)},
            "dev": {f"t{t}": r["intermediates"][str(max(
                int(k) for k in r["intermediates"] if k.isdigit()))][f"t{t}"]["exact"]
                if r["intermediates"] else None for t in range(5)},
            "test": {f"t{t}": r["trained"][f"t{t}"]["accuracy"] for t in range(5)},
            "untrained_test": {f"t{t}": r["untrained"][f"t{t}"]["accuracy"]
                               for t in range(5)},
            "first_train_lift_tier": r["diagnostics"]["first_train_lift_tier"],
            "first_test_lift_tier": r["diagnostics"]["first_test_lift_tier"]},
    (root / "LIFT_OFF_CURVES.json").write_text(
        json.dumps({"schema": "citadel-t1d-lift-off-curves/v1", "arms": curves},
                   indent=2, sort_keys=True), encoding="utf-8")
    from citadel_tpu import pre50m as p50  # noqa: E402 (phase import)

    pre50m_status: dict[str, Any] = {}
    try:
        from citadel_tpu import t1c_run as t1c

        (root / "PRE50M_TARGET.json").write_text(json.dumps(
            {"schema": "citadel-pre50m-target/v1", **p50.PRE50M_TARGET,
             "citadel_sha": rb.citadel_sha(),
             "cymek_runtime_sha": rt_sha}, indent=2, sort_keys=True), encoding="utf-8")
        smoke = p50.smoke_target_model(out_dir=str(root))
        di = p50.data_interface_cert(out_dir=str(root))
        packing = p50.packing_cert(out_dir=str(root))
        feas = {"MID_3_7M": p50.memory_estimate(t1c.MID_EXPECTED_PARAMS),
                "SCALE2_7_4M": p50.memory_estimate(SCALE2_EXPECTED_PARAMS)}
        try:
            scale2_rate = (smoke["capacity_tokens"] / max(
                smoke.get("train_wall_seconds", 0) or smoke["wall_seconds"], 1e-9))
        except Exception:
            scale2_rate = None
        curve = {"MID": p50.throughput_estimates(rate),
                 "SCALE2": p50.throughput_estimates(scale2_rate) if scale2_rate else None}
        grad_accum = p50.grad_accumulation_status(True, shape[0])
        oom = p50.oom_decision(cal.get("candidates", []))
        feas["grad_accumulation"] = grad_accum
        feas["oom_selection"] = oom
        (root / "PRE50M_FEASIBILITY.json").write_text(json.dumps(
            {"schema": "citadel-pre50m-feasibility/v1",
             "citadel_sha": rb.citadel_sha(), "cymek_runtime_sha": rt_sha,
             "memory": feas, "grad_accumulation": grad_accum,
             "oom_selection": oom}, indent=2, sort_keys=True), encoding="utf-8")
        (root / "PRE50M_THROUGHPUT.json").write_text(json.dumps(
            {"schema": "citadel-pre50m-throughput/v1",
             "citadel_sha": rb.citadel_sha(), "cymek_runtime_sha": rt_sha,
             "curve": curve}, indent=2, sort_keys=True), encoding="utf-8")
        diagnostics = {
            "schema": "citadel-t1d-diagnostics/v1",
            "arms": {t: r.get("diagnostics", {}) for t, r in arm_receipts.items()},
            "pre50m_smoke_losses": smoke.get("losses", []),
        }
        (root / "DIAGNOSTICS.json").write_text(
            json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8")
        decision = p50.build_decision(
            target={"understood": True, "type": p50.PRE50M_TARGET["type"],
                    "parameter_count": None},
            smoke=smoke, feasibility={"verdict": feas["SCALE2_7_4M"]["verdict"]},
            data_interface=di, packing=packing,
            recommended_batch=shape[0], recommended_sequence_length=shape[1],
            rate_tok_s=rate)
        (root / "NEXT_50M_DECISION.json").write_text(
            json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
        pre50m_status = {"status": "PASS", "decision": decision}
        print("pre50m:", decision["ready_for_50m_training"],
              decision["blocking_reasons"], flush=True)
    except Exception as exc:  # noqa: BLE001 - preserve arms, record, continue
        pre50m_status = {"status": "IMPLEMENTATION_FAILURE",
                         "error": f"{type(exc).__name__}: {exc}"}
        print(f"pre50m: IMPLEMENTATION_FAILURE {exc}", flush=True)
    session = {"schema": "citadel-t1d-session/v1",
               "citadel_sha": rb.citadel_sha(), "cymek_runtime_sha": rt_sha,
               "shape": list(shape), "calibrated_rate": rate, "budgets_scaled": scaled,
               "budgets": {t: c["budget"] for t, c in budgets.items()},
               "arms": {t: r.get("status") for t, r in arm_receipts.items()},
               "labels": summary["labels"], "pre50m": pre50m_status,
               "bundle": "pending"}
    (root / "SESSION_MANIFEST.json").write_text(
        json.dumps(session, indent=2, sort_keys=True), encoding="utf-8")
    bundle = build_bundle(str(root), out=str(root / "CITADEL_T1D_RESULTS.zip"))
    session["bundle"] = bundle["zip_bytes"]
    (root / "SESSION_MANIFEST.json").write_text(
        json.dumps(session, indent=2, sort_keys=True), encoding="utf-8")
    print("session complete:", session["labels"], flush=True)
    return session


def build_bundle(session_dir: str, *, out: str) -> dict[str, Any]:
    """Assemble CITADEL_T1D_RESULTS.zip (receipts + manifest; binaries per cap rule)."""
    root = Path(session_dir)
    names = ["SESSION_MANIFEST.json", "DATA_MANIFEST.json", "CALIBRATION.json",
             "ARM_A.json", "ARM_B.json", "ARM_C.json", "ARM_D.json", "ARM_E.json",
             "LIFT_OFF_CURVES.json", "CROSS_ARM_SUMMARY.json",
             "PRE50M_TARGET.json", "PRE50M_FEASIBILITY.json",
             "PRE50M_THROUGHPUT.json", "PRE50M_CHECKPOINT_SMOKE.json",
             "PRE50M_DATA_INTERFACE.json", "PRE50M_PACKING.json",
             "DIAGNOSTICS.json", "NEXT_50M_DECISION.json"]
    missing = [n for n in names if not (root / n).is_file()]
    if missing:
        raise RuntimeError(f"bundle incomplete, missing: {', '.join(missing)}")
    ckpts = sorted(str(p) for p in root.glob("t1d_arm_*.pt"))
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
    "ARM_TIME_BOX_S",
    "AUTO_SCALE_RATE",
    "CALIBRATION_SHAPES",
    "CALIBRATION_UPDATES",
    "CKPT_ZIP_BYTES_CAP",
    "LIFT_MIN_N",
    "LIFT_THRESHOLD",
    "SCALE2_EXPECTED_PARAMS",
    "SCALE2_SPEC_KWARGS",
    "TRAIN_LENGTH",
    "TRAIN_SAMPLE_PER_TIER",
    "TierFeeder",
    "assemble_batch",
    "build_bundle",
    "build_spec",
    "calibrate",
    "classify_cross_arm",
    "pack_rows",
    "run_arm",
    "run_session",
    "valid_alphabet_ids",
]


class TierFeeder:
    """Deterministic per-arm row stream with exact consumption accounting.

    Static shapes demand EXACTLY n_seq full sequences per update despite
    variable row lengths: rows are drawn into a FIFO carry buffer and packed
    first-fit; leftovers carry to the next update (zero waste, zero fake rows).
    Tier cursors advance on draw (reservation); per-tier consumed counts subtract
    carried reservations, so consumed rows are always exact index prefixes —
    the memorization lens samples truly-seen rows. Curriculum arms pick tiers
    by the frozen schedule on global training fraction; flat arm uses the
    frozen uniform mixture; teacher arm interleaves 6 ordinary + 4 teacher
    rows per 10-block (exact token split recorded).
    """

    TEACHER_KINDS = ("digadd", "digsub", "singlemul", "divmicro")

    def __init__(self, mode: str, n_seq: int, length: int):
        from citadel_tpu import tiered_data as td

        self.mode = mode
        self.n_seq = n_seq
        self.length = length
        # Ordinary rows per 10-pattern, driven by the frozen TEACHER_RATIO
        # (0.40 teacher → 6 ordinary + 4 teacher). Exact token split recorded.
        self._ordinary_per_10 = int(round((1.0 - td.TEACHER_RATIO) * 10))
        self.cursors: dict[int, int] = {t: 0 for t in range(5)}
        self.drawn: dict[str, int] = {}
        self.carried: dict[str, int] = {}
        self.placed_rows: dict[str, int] = {}
        self.placed_tokens: dict[str, int] = {}
        self.teacher_cursors: dict[str, int] = {k: 0 for k in self.TEACHER_KINDS}
        self._draw = 0
        self._pattern = 0
        self._carry: list[tuple[str, str, str]] = []  # (text, key, tier_tag)

    def _ordinary_row(self, frac: float) -> tuple[str, str]:
        from citadel_tpu import tiered_data as td

        tier = td.uniform_tier(self._draw) if self.mode == "flat" \
            else td.curriculum_tier(frac, self._draw)
        self._draw += 1
        text, _ = td.tier_row(tier, "train", self.cursors[tier])
        self.cursors[tier] += 1
        key = f"tier:{tier}"
        self.drawn[key] = self.drawn.get(key, 0) + 1
        return text, key

    def _teacher_row(self) -> tuple[str, str]:
        from citadel_tpu import tiered_data as td

        kind = self.TEACHER_KINDS[sum(self.teacher_cursors.values()) % 4]
        text, _ = td.teacher_row(kind, self.teacher_cursors[kind])
        self.teacher_cursors[kind] += 1
        key = f"teacher:{kind}"
        self.drawn[key] = self.drawn.get(key, 0) + 1
        return text, key

    def _refill(self, n: int, frac: float) -> None:
        for _ in range(n):
            if self.mode == "teacher" and (self._pattern % 10) >= self._ordinary_per_10:
                t, key = self._teacher_row()
            else:
                t, key = self._ordinary_row(frac)
            self._pattern += 1
            if len(t) > self.length:
                raise ValueError(f"row exceeds fixed length {self.length}: {t!r}")
            self._carry.append((t, key, key))
            self.carried[key] = self.carried.get(key, 0) + 1

    def fill_sequences(self, frac: float) -> list[list[str]]:
        """Exactly n_seq sequences of row texts; consumption recorded on placement."""
        seqs: list[list[str]] = []
        cur: list[str] = []
        used = 0
        while len(seqs) < self.n_seq:
            if not self._carry:
                self._refill(64, frac)
            text, key, _ = self._carry.pop(0)
            self.carried[key] -= 1
            n = len(text)
            if used + n > self.length:
                seqs.append(cur)
                cur, used = [], 0
                if len(seqs) == self.n_seq:
                    self._carry.insert(0, (text, key, key))
                    self.carried[key] = self.carried.get(key, 0) + 1
                    break
            cur.append(text)
            used += n
            self.placed_rows[key] = self.placed_rows.get(key, 0) + 1
            self.placed_tokens[key] = self.placed_tokens.get(key, 0) + n
        assert len(seqs) == self.n_seq, "packing invariant violated (corpus exhausted?)"
        return [s for s in seqs]

    def consumed_prefix(self, tier: int) -> int:
        """Rows of tier t guaranteed consumed (exact prefix bound)."""
        key = f"tier:{tier}"
        return self.placed_rows.get(key, 0)

    def ledger(self) -> dict[str, Any]:
        return {"placed_rows": dict(self.placed_rows),
                "placed_tokens": dict(self.placed_tokens),
                "carry_pending": sum(self.carried.values())}
