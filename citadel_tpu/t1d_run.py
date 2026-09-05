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
import random
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
    "F": {"spec": "MID", "mode": "self", "budget": 2_000_000},
}
ARM_ORDER = ("A", "B", "C", "D", "E", "F")


MID_STATE_SCHEMA = "citadel-t1d-arm-mid/v1"


def save_mid_state(out_dir: str | Path, tag: str, *, model, optimizer, feeder,
                   payload: dict[str, Any]) -> str:
    """Durably persist resumable mid-arm training state at a preregistered
    checkpoint fraction (model + optimizer + feeder cursors/carry + ledger +
    pre-training baselines). Self-hash recorded in the JSON sidecar; a
    disconnected runtime resumes from here instead of restarting the arm."""
    import torch

    body = {**payload, "plan_sha": plan_identity(),
            "schema": MID_STATE_SCHEMA, "arm": tag}
    base = Path(out_dir)
    model_sha = _ckpt_mod_ref().save(model, str(base / f"t1d_arm_{tag.lower()}_mid.pt"),
                                     {"arm": tag, "update": payload["update"]})
    opt_sha = _ckpt_mod_ref().save_optimizer_state(
        optimizer, str(base / f"t1d_arm_{tag.lower()}_mid.opt"),
        {"arm": tag, "update": payload["update"]})
    doc = {**body, "model_path": str(base / f"t1d_arm_{tag.lower()}_mid.pt"),
           "model_sha256": model_sha,
           "optimizer_path": str(base / f"t1d_arm_{tag.lower()}_mid.opt"),
           "optimizer_sha256": opt_sha,
           "payload_sha256": hashlib.sha256(_canonical_json(body)).hexdigest()}
    (base / f"ARM_{tag}.mid.json").write_text(
        json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    return str(base / f"ARM_{tag}.mid.json")


def _ckpt_mod_ref():
    from citadel_tpu import checkpoint as ckpt_mod

    return ckpt_mod


def load_mid_state(out_dir: str | Path, tag: str, *, expect_cfg: dict,
                   seed: int, shape: tuple[int, int]) -> tuple[dict | None, str]:
    """Hash-verified mid-arm state (model/optimizer loaded by the caller).
    Returns (payload | None, reason). Any mismatch archives the sidecar."""
    path = Path(out_dir) / f"ARM_{tag}.mid.json"
    if not path.is_file():
        return None, ""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        path.rename(path.with_suffix(f".json.corrupt-{stamp}"))
        return None, f"unreadable mid state ({type(exc).__name__})"
    body = {k: v for k, v in doc.items()
            if k not in ("model_path", "model_sha256", "optimizer_path",
                         "optimizer_sha256", "payload_sha256")}
    if hashlib.sha256(_canonical_json(body)).hexdigest() != doc.get("payload_sha256"):
        return None, "mid payload hash mismatch"
    if doc.get("plan_sha") != plan_identity():
        return None, "mid state belongs to a different plan version"
    if doc.get("cfg") != expect_cfg or doc.get("seed") != seed \
            or tuple(doc.get("shape", ())) != tuple(shape):
        return None, "mid state cfg/seed/shape mismatch"
    for key in ("model_path", "model_sha256", "optimizer_path", "optimizer_sha256"):
        if key not in doc:
            return None, f"mid state missing {key}"
    mp, op = Path(doc["model_path"]), Path(doc["optimizer_path"])
    if not mp.is_file() or not op.is_file():
        return None, "mid model/optimizer files missing"
    if hashlib.sha256(mp.read_bytes()).hexdigest() != doc["model_sha256"]:
        return None, "mid model hash mismatch"
    if hashlib.sha256(op.read_bytes()).hexdigest() != doc["optimizer_sha256"]:
        return None, "mid optimizer hash mismatch"
    return doc, f"verified mid state at update {doc.get('update')}"


def plan_identity() -> str:
    """SHA-256 over the frozen scientific plan (arms, thresholds, data
    generators, amendments). Part of resume identity: scientific state can
    never be resumed under a changed plan."""
    from citadel_tpu import self_knowledge as sk
    from citadel_tpu import tiered_data as td

    body = {
        "arms": {t: dict(c) for t, c in sorted(ARMS.items())},
        "lift_threshold": LIFT_THRESHOLD, "lift_min_n": LIFT_MIN_N,
        "train_sample_per_tier": TRAIN_SAMPLE_PER_TIER,
        "arm_time_box_s": ARM_TIME_BOX_S,
        "calibration_shapes": [list(s) for s in CALIBRATION_SHAPES],
        "auto_scale_rate": AUTO_SCALE_RATE,
        "tiered_generator": td.GENERATOR_VERSION,
        "self_knowledge": sk.plan_identity(),
        "amendments": ["PRE50M_ADDENDUM", "RUNTIME_AMENDMENT_001",
                       "SELF_KNOWLEDGE_AMENDMENT"],
    }
    return hashlib.sha256(_canonical_json(body)).hexdigest()
ARM_TIME_BOX_S = 45 * 60


def sk_self_mod() -> int:
    """Frozen self-row cadence for arm F: every 7th draw (self_knowledge.SELF_ROW_FRACTION)."""
    from citadel_tpu import self_knowledge as sk

    return sk.SELF_ROW_FRACTION
TIER_KEYS = tuple(f"t{t}" for t in range(5))
UNTRAINED_SUMMARY_KEYS = ("correct", "total", "accuracy", "wilson_lcb", "wilson_ucb")
PREFINAL_SCHEMA = "citadel-t1d-arm-prefinal/v1"


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _validate_summary(where: str, summ: Any, defects: list[str]) -> None:
    """One tier summary must carry the exact metric contract."""
    if not isinstance(summ, dict):
        defects.append(f"{where} summary is not a dict")
        return
    for key in UNTRAINED_SUMMARY_KEYS:
        if key not in summ:
            defects.append(f"{where} missing {key!r}")
    total = summ.get("total")
    acc = summ.get("accuracy")
    if isinstance(total, bool) or not isinstance(total, (int, float)) \
            or not (total == total) or float(total) < 0:
        defects.append(f"{where}.total not a nonnegative number: {total!r}")
    if isinstance(acc, bool) or not isinstance(acc, (int, float)) \
            or not (acc == acc) or not (0.0 <= float(acc) <= 1.0):
        defects.append(f"{where}.accuracy not finite in [0,1]: {acc!r}")


def normalize_untrained_receipt(untrained: Any) -> dict[str, Any]:
    """Canonicalize the untrained TEST block to t0..t4 (pure; defense in depth).

    REAL TPU FAILURE THIS GUARDS AGAINST (2026-09-05): the producer stored
    untrained results as dev_tN/test_tN while the finalizer read tN — the
    scientific gate died on KeyError: 't1' AFTER expensive training. This
    normalizer accepts the canonical {"t0".."t4"} form or the legacy
    {"test_t0".."test_t4"} form, validates every summary
    (correct/total/accuracy/wilson_lcb/wilson_ucb; total >= 0; accuracy
    finite in [0,1]) and returns the canonical mapping. Raises
    RuntimeError('ARM_SCHEMA_INVALID: ...') — never a raw KeyError.
    """
    if not isinstance(untrained, dict):
        raise RuntimeError("ARM_SCHEMA_INVALID: untrained block is not a dict")
    if all(f"t{t}" in untrained for t in range(5)):
        src = {f"t{t}": untrained[f"t{t}"] for t in range(5)}
    elif all(f"test_t{t}" in untrained for t in range(5)):
        src = {f"t{t}": untrained[f"test_t{t}"] for t in range(5)}
    else:
        missing = [k for k in TIER_KEYS if k not in untrained]
        legacy_missing = [f"test_t{t}" for t in range(5)
                          if f"test_t{t}" not in untrained]
        raise RuntimeError(
            "ARM_SCHEMA_INVALID: untrained block has neither canonical t0-t4 "
            f"(missing {missing}) nor complete test_t0-test_t4 form "
            f"(missing {legacy_missing}); present keys: {sorted(untrained)}")
    defects: list[str] = []
    for t in range(5):
        _validate_summary(f"untrained[t{t}]", src[f"t{t}"], defects)
    if defects:
        raise RuntimeError("ARM_SCHEMA_INVALID: " + "; ".join(defects))
    return src
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


def _verify_shape_on_scale2(batch: int, length: int) -> bool:
    """One correctness update on SCALE2 at the candidate shape (§17).

    A batch that fits MID may fail the larger arm; Arm D must never be the
    first place the session-wide shape is tried. Returns True/False only
    (callers record the verdict); never raises.
    """
    try:
        from citadel_tpu import xla_backend as xb

        import torch

        from v5_model.core import initialize
        from v5_training.optimizer import build_adamw_optimizer

        torch.manual_seed(20260904)
        model = initialize(build_spec("SCALE2"), 20260904)
        device = xb.get_device()
        model = model.to(device)
        opt = build_adamw_optimizer(model, torch_module=torch)
        feeder = TierFeeder("flat", batch, length)
        rep = _train_updates_packed(
            model, opt, feeder, n_updates=1, start_update=0,
            updates_total=1, device=device, torch_mod=torch,
            length=length, masked=False, valid_ids=None)
        ok = rep["first_loss"] is not None and rep["first_loss"] == rep["first_loss"]
        model = opt = feeder = None
        try:
            import gc as _gc

            _gc.collect()
        except Exception:
            pass
        return bool(ok)
    except Exception:
        return False


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
        finally:
            # Drop references so XLA/CPU tensors can release (del-by-name via
            # locals() would be a no-op in optimized scopes; None-assignment
            # actually decrements refcounts).
            model = opt = feeder = None
        try:
            import gc as _gc

            _gc.collect()
        except Exception:
            pass
    best, scale2_note = select_calibrated_shape(
        results, scale2_verifier=_verify_shape_on_scale2)
    if best is None:
        raise RuntimeError(f"abort CALIBRATION_FAILURE: {scale2_note}")
    receipt = {"schema": "citadel-t1d-throughput-calibration/v1",
               "citadel_sha": rb.citadel_sha(), "cymek_runtime_sha": rt_sha,
               "environment": env, "candidates": results,
               "scale2_verification": scale2_note,
               "selected": {"batch": best["batch"], "length": best["length"]},
               "selected_tokens_per_second": best["tokens_per_second"]}
    if out is not None:
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    return receipt


def should_skip_arm(out_dir: str | Path, tag: str) -> tuple[str, str]:
    """Resume predicate for one arm (pure file logic, no device).

    Returns ("skip", reason) when a valid receipt+marker pair exists,
    ("run", reason) when nothing is complete, ("raise", repair-instruction)
    on contradictory state (marker without receipt or unreadable receipt).
    Never silently overwrites or reruns completed TEST work.
    """
    out_path = Path(out_dir) / f"ARM_{tag}.json"
    marker = Path(out_dir) / f"ARM_{tag}.done.json"
    if marker.is_file() and out_path.is_file():
        try:
            prior = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return ("raise", f"receipt {out_path.name} is unreadable ({exc}); "
                             f"delete marker {marker.name} + receipt to force rerun")
        if isinstance(prior, dict) and prior.get("status") == "IMPLEMENTATION_FAILURE":
            # an implementation failure is NOT a completed arm, marker or no
            # marker: retry after software repair (a valid prefinal snapshot
            # resumes finalization only; otherwise the arm retrains)
            return ("run", "IMPLEMENTATION_FAILURE receipt is not completion; "
                           "retrying after software repair")
        if isinstance(prior, dict) and prior.get("status") in ("SCIENTIFIC_PASS",
                                                               "SCIENTIFIC_FAIL",
                                                               "TIMEBOX_ABORT"):
            return ("skip", f"valid {prior.get('status')} receipt+marker present")
        return ("raise", f"receipt {out_path.name} has unknown status "
                         f"{prior.get('status') if isinstance(prior, dict) else '?'}; "
                         f"delete marker {marker.name} + receipt to force rerun")
    if marker.is_file():
        return ("raise", f"marker {marker.name} exists without receipt "
                         f"{out_path.name}; delete the marker to force rerun")
    return ("run", "nothing complete")


def timebox_abort_receipt(*, tag: str, cfg: dict[str, Any], env: dict[str, Any],
                          n_seq: int, length: int, updates_done: int,
                          ledgers: list, feeder_ledger: dict[str, Any],
                          error: str, wall: float) -> dict[str, Any]:
    """Minimal TIMEBOX_ABORT receipt (pure builder; skips all evals)."""
    return {
        "schema": "citadel-t1d-arm/v1", "arm": tag, "config": cfg,
        "environment": env, "batch_sequences": n_seq, "sequence_length": length,
        "training": {"updates": updates_done, "ledgers": ledgers},
        "data": {"feeder": feeder_ledger},
        "status": "TIMEBOX_ABORT",
        "error": error,
        "wall_seconds": wall,
    }


def run_arm(tag: str, cfg: dict[str, Any], *, shape: tuple[int, int],
            out_dir: str, seed: int = 20260904) -> dict[str, Any]:
    """Execute one frozen T1D arm end to end; receipt + checkpoint + marker."""
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
    decision, why = should_skip_arm(out_dir, tag)
    if decision == "skip":
        prior = json.loads(out_path.read_text(encoding="utf-8"))
        prior["resumed"] = True
        return prior
    if decision == "raise":
        raise RuntimeError(f"abort RESUME_CONFLICT: {why}")
    # FINALIZATION-ONLY RECOVERY: a valid pre-finalization snapshot plus a
    # matching checkpoint means the expensive work already completed — finish
    # the PURE finalizer without retraining and without any device.
    snap, snap_why = load_prefinal_snapshot(out_dir, tag, expect_cfg=cfg,
                                            seed=seed, shape=shape)
    if snap is not None:
        print(f"arm {tag}: prefinal recovery — finalization only ({snap_why})",
              flush=True)
        receipt = build_arm_receipt(**snap)
        write_arm_receipt(out_dir, receipt, ckpt_hash=snap["ckpt_hash"])
        (Path(out_dir) / f"ARM_{tag}.prefinal.json").unlink(missing_ok=True)
        return receipt
    if snap_why:
        print(f"arm {tag}: no usable prefinal snapshot ({snap_why}); fresh run",
              flush=True)
    # orphan checkpoint from a failed prior session: archive as forensic
    # artifact, never silently clobbered (invalid results are not results)
    orphan = Path(out_dir) / f"t1d_arm_{tag.lower()}.pt"
    if orphan.is_file():
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        orphan.rename(orphan.with_suffix(f".pt.orphan-{stamp}"))
        print(f"arm {tag}: archived orphan checkpoint {orphan.name}", flush=True)
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
    # Mid-arm resume (§12): a disconnected arm continues from its last
    # preregistered checkpoint fraction instead of restarting from zero.
    mid, mid_why = load_mid_state(out_dir, tag, expect_cfg=cfg, seed=seed,
                                  shape=(n_seq, length))
    if mid is not None:
        print(f"arm {tag}: mid-arm resume from update {mid['update']} ({mid_why})",
              flush=True)

    slices = _tier_slices()
    allow_ids = valid_alphabet_ids() if masked else None

    torch.manual_seed(seed)
    model = initialize(spec, seed)
    device = xb.get_device()
    model = model.to(device)
    param_count = sum(int(p.numel()) for p in model.parameters())
    optimizer = build_adamw_optimizer(model, torch_module=torch)
    if mid is not None:
        ckpt_mod.load_into(model, mid["model_path"])
        ckpt_mod.load_optimizer_state(optimizer, mid["optimizer_path"])

    def gen(rows, targets):
        return _gen_eval(model, rows, targets, device=device, torch_mod=torch,
                         xb=xb, allow_ids=allow_ids)

    # Canonical untrained namespaces (one schema downstream, no guessing):
    # untrained_test[tN] = untrained TEST summary (the receipt's "untrained");
    # untrained_dev[tN]  = untrained DEV summary (explicit separate block).
    # DEV and TEST are each evaluated exactly once per preregistered arm;
    # these are namespace fixes, not extra observations.
    untrained_dev: dict[str, Any] = {}
    untrained_test: dict[str, Any] = {}
    if mid is None:
        for tier in range(5):
            _, summ = gen(slices[f"dev_t{tier}"]["rows"], slices[f"dev_t{tier}"]["targets"])
            untrained_dev[f"t{tier}"] = summ
            _, summ = gen(slices[f"test_t{tier}"]["rows"], slices[f"test_t{tier}"]["targets"])
            untrained_test[f"t{tier}"] = summ
        untrained = untrained_test
    else:
        # baselines are durable pre-training measurements restored verbatim;
        # re-evaluating them against a trained model would be fabrication
        untrained_dev = mid["untrained_dev"]
        untrained = mid["untrained"]
    untrained_test = untrained
    # Frozen TRAIN diagnostic candidates: the FIRST TRAIN_SAMPLE_PER_TIER train
    # indices per tier, fixed BEFORE any training. Consumption is verified
    # against the feeder's exact per-tier consumed prefix after training —
    # never assumed (the feeder has consumed nothing at this point).
    train_candidates = frozen_train_candidates()
    untrained_train: dict[str, Any] = {}
    for tier in range(5):
        if mid is not None:
            untrained_train = mid["untrained_train"]
            break
        rows = [td.tier_row(tier, "train", i)[0] for i in train_candidates[tier]]
        tgts = [cev.split_prompt_target(r)[1] for r in rows]
        _, summ = gen(rows, tgts)
        untrained_train[f"t{tier}"] = summ
    # Self-knowledge probes (DEV-tier diagnostic family, SELF_KNOWLEDGE
    # AMENDMENT): identical frozen probe rows for EVERY arm; text scoring.
    from citadel_tpu import self_knowledge as sk

    self_rows, self_targets, self_meta = sk.self_probe_rows()

    def gen_text(rows, targets):
        recs = cev.generate(rows, model, xb, device=device, torch_mod=torch,
                            allow_ids=allow_ids)
        return recs, sk.summarize_text([r["prediction"] for r in recs], targets)

    if mid is not None:
        untrained_self = mid["untrained_self"]
        self_diag_rest = mid.get("self_baseline", {})
    else:
        _, untrained_self = gen_text(self_rows, self_targets)
        self_diag_rest = {}
    self_null_preds = sk.most_common_null(self_targets)
    untrained_self_null = sk.summarize_text(self_null_preds, self_targets)

    checkpoints = sorted({max(1, int(updates_total * f)) for f in (0.25, 0.50, 0.75, 1.0)})
    done, ledgers, inter = 0, [], {}
    first_loss, last_loss = None, None
    cap_total, ans_total, whole_total = 0, 0, 0
    gsum, gmax, gn = 0.0, 0.0, 0
    if mid is not None:
        done = int(mid["update"])
        ledgers = list(mid["ledgers"])
        inter = dict(mid["inter"])
        first_loss = mid["first_loss"]
        last_loss = mid["last_loss"]
        cap_total = int(mid["cap_total"])
        ans_total = int(mid["ans_total"])
        whole_total = int(mid["whole_total"])
        gsum, gmax, gn = float(mid["gsum"]), float(mid["gmax"]), int(mid["gn"])
        checkpoints = [cp for cp in checkpoints if cp > done]
    t_train0 = time.time()
    try:
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
            print(f"arm {tag}: update {cp}/{updates_total} "
                  f"({100 * cp / max(updates_total, 1):.0f}%) "
                  f"last_loss={last_loss:.3f} "
                  f"dev_t1={dev_curve['t1']['exact']:.3f}", flush=True)
            # durable mid-arm checkpoint (disconnect recovery, §12)
            if cp < checkpoints[-1]:
                save_mid_state(out_dir, tag, model=model, optimizer=optimizer,
                               feeder=feeder,
                               payload={"update": cp, "cfg": dict(cfg),
                                        "seed": seed, "shape": [n_seq, length],
                                        "feeder_state": feeder.state(),
                                        "untrained": untrained,
                                        "untrained_dev": untrained_dev,
                                        "untrained_self": untrained_self,
                                        "untrained_train": untrained_train,
                                        "ledgers": ledgers, "inter": inter,
                                        "first_loss": first_loss,
                                        "last_loss": last_loss,
                                        "cap_total": cap_total,
                                        "ans_total": ans_total,
                                        "whole_total": whole_total,
                                        "gsum": gsum, "gmax": gmax, "gn": gn})
    except TimeoutError as exc:
        wall = time.time() - t0
        abort = timebox_abort_receipt(
            tag=tag, cfg=cfg, env=env, n_seq=n_seq, length=length,
            updates_done=done,
            ledgers=ledgers + [{"train_wall_seconds": time.time() - t_train0}],
            feeder_ledger=feeder.ledger(),
            error=f"{type(exc).__name__}: {exc}", wall=wall)
        abort["citadel_sha"] = rb.citadel_sha()
        abort["cymek_runtime_sha"] = rt_sha
        abort["model"] = {"spec": cfg["spec"], "parameter_count": param_count}
        abort["device_count"] = n_devices
        out_path.write_text(json.dumps(abort, indent=2, sort_keys=True), encoding="utf-8")
        marker.write_text(json.dumps({"receipt": str(out_path), "status": "TIMEBOX_ABORT",
                                      "updates_done": done}, indent=2), encoding="utf-8")
        return abort
    train_wall = time.time() - t_train0

    trained: dict[str, Any] = {}
    trained_recs: dict[str, Any] = {}
    for tier in range(5):
        recs, summ = gen(slices[f"test_t{tier}"]["rows"], slices[f"test_t{tier}"]["targets"])
        trained[f"t{tier}"], trained_recs[f"t{tier}"] = summ, recs
    # Train memorization lens: score ONLY rows verified inside the exact
    # consumed prefix for that tier. Frozen candidates were fixed before
    # training; the plan verifies consumption, never assumes it.
    trained_train: dict[str, Any] = {}
    train_memorization: dict[str, Any] = {}
    plan = train_memorization_plan(feeder, candidates=train_candidates)
    for tier in range(5):
        entry = plan[tier]
        verified = entry["verified_indices"]
        rows = [td.tier_row(tier, "train", i)[0] for i in verified]
        tgts = [cev.split_prompt_target(r)[1] for r in rows]
        if rows:
            _, summ = gen(rows, tgts)
        else:
            summ = {"correct": 0, "total": 0, "accuracy": 0.0,
                    "wilson_lcb": 0.0, "wilson_ucb": 0.0}
        trained_train[f"t{tier}"] = summ
        train_memorization[f"t{tier}"] = {
            "consumed_prefix": entry["consumed_prefix"],
            "n_frozen_candidates": entry["n_candidates"],
            "n_verified_consumed": entry["n_verified"],
            "evaluated_rows": len(rows),
            "status": entry["status"],
            "lift_eligible": bool(entry["status"] == "OK"
                                  and summ["accuracy"] >= LIFT_THRESHOLD),
        }

    # Failure-cause probes (diagnostic-only; gates unchanged). The two extra
    # generation evals run here; every pure aggregation happens in
    # build_arm_receipt (device-free, simulated by tests).
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

    self_recs, trained_self = gen_text(self_rows, self_targets)
    per_domain = {}
    for m, r in zip(self_meta, self_recs):
        d = m["domain"]
        agg = per_domain.setdefault(d, {"correct": 0, "total": 0})
        agg["total"] += 1
        agg["correct"] += int(sk.text_exact(r["prediction"], r["target"]))
    for d, agg in per_domain.items():
        agg["accuracy"] = agg["correct"] / agg["total"] if agg["total"] else 0.0
    self_diag = {"n": len(self_rows),
                 "trained": trained_self,
                 "untrained": untrained_self,
                 "most_common_null": untrained_self_null,
                 "per_domain": per_domain}

    wall = time.time() - t0
    # Durable PRE-FINALIZATION RECOVERY SNAPSHOT: training, TEST evaluation,
    # diagnostics, checkpoint, and reload identity are complete and durable —
    # everything the PURE finalizer needs. If finalization ever throws, a
    # rerun resumes FINALIZATION ONLY (no retraining, no device).
    snapshot_kwargs = dict(
        tag=tag, cfg=dict(cfg), env=dict(env), n_seq=n_seq, length=length,
        param_count=param_count, citadel_sha=rb.citadel_sha(), cymek_sha=rt_sha,
        seed=seed,
        feeder_placed_rows={k: int(v) for k, v in feeder.placed_rows.items()},
        feeder_ledger=feeder.ledger(),
        ledgers=ledgers, done=done, first_loss=first_loss, last_loss=last_loss,
        cap_total=cap_total, ans_total=ans_total, whole_total=whole_total,
        gsum=gsum, gmax=gmax, gn=gn, train_wall=train_wall,
        untrained=untrained, untrained_dev=untrained_dev,
        untrained_train=untrained_train,
        untrained_self=untrained_self, trained_self=trained_self,
        self_diagnostics=self_diag,
        trained=trained, trained_recs=trained_recs,
        trained_train=trained_train, train_memorization=train_memorization,
        inter=inter, teacher_eval=teacher_eval, first_step=first_step,
        ckpt_path=ckpt_path, ckpt_hash=ckpt_hash,
        pre_sha=pre_sha, post_sha=post_sha, reload_ok=reload_ok,
        device_count=n_devices, wall=wall)
    sidecar = write_prefinal_snapshot(out_dir, snapshot_kwargs)
    # Pure post-training path: nulls (global + per-tier t0-t4), diagnostic
    # aggregation, lift tiers, scientific gate, receipt assembly. No device.
    receipt = build_arm_receipt(**load_finalizer_kwargs(snapshot_kwargs))
    write_arm_receipt(out_dir, receipt, ckpt_hash=ckpt_hash)
    # finalization succeeded: the recovery sidecar is consumed
    Path(sidecar).unlink(missing_ok=True)
    return receipt


def frozen_train_candidates() -> dict[int, list[int]]:
    """Frozen TRAIN memorization candidates: the FIRST TRAIN_SAMPLE_PER_TIER
    train indices per tier, fixed before any training. Every tier has far more
    than 200 train rows (tiered_data.TRAIN_N: min 20,000), so the frozen
    prefix always exists. Consumption is verified later, never assumed."""
    return {t: list(range(TRAIN_SAMPLE_PER_TIER)) for t in range(5)}


def train_memorization_plan(feeder, *,
                            candidates: dict[int, list[int]] | None = None
                            ) -> dict[int, dict[str, Any]]:
    """Verify frozen candidates against the feeder's EXACT consumed prefix.

    The feeder consumes tier rows strictly in index order, so a row counts as
    consumed iff its index < consumed_prefix for that tier. Status is "OK"
    only with at least LIFT_MIN_N verified rows; otherwise the tier is
    INSUFFICIENT_CONSUMPTION and FIRST_TRAIN_LIFT_TIER can never fire on it.
    Unseen train rows are never used as memorization evidence.
    """
    cand_map = frozen_train_candidates() if candidates is None else candidates
    plan: dict[int, dict[str, Any]] = {}
    for tier in range(5):
        cand = list(cand_map[tier])
        consumed = feeder.consumed_prefix(tier)
        verified = [i for i in cand if i < consumed]
        plan[tier] = {"consumed_prefix": consumed,
                      "n_candidates": len(cand),
                      "verified_indices": verified,
                      "n_verified": len(verified),
                      "status": ("OK" if len(verified) >= LIFT_MIN_N
                                 else "INSUFFICIENT_CONSUMPTION")}
    return plan


NULL_TIER_KEYS = {"t0", "t1", "t2", "t3", "t4"}


def validate_null_block(receipt: dict[str, Any]) -> list[str]:
    """Receipt schema validation for the heuristic-null block (pure).

    Requires: heuristic_nulls non-empty dict; nulls_per_tier exactly t0-t4;
    each tier carrying strongest/accuracy/all; every accuracy finite in
    [0, 1]. Returns defect strings (empty list = valid).
    """
    defects: list[str] = []
    hn = receipt.get("heuristic_nulls")
    if not isinstance(hn, dict) or not hn:
        defects.append("heuristic_nulls missing or empty")
    npt = receipt.get("nulls_per_tier")
    if not isinstance(npt, dict) or set(npt) != NULL_TIER_KEYS:
        got = sorted(npt) if isinstance(npt, dict) else type(npt).__name__
        defects.append(f"nulls_per_tier keys {got!r} != {sorted(NULL_TIER_KEYS)}")
        return defects
    for tier_key in sorted(NULL_TIER_KEYS):
        entry = npt[tier_key]
        if not isinstance(entry, dict):
            defects.append(f"nulls_per_tier[{tier_key}] not a dict")
            continue
        for key in ("strongest", "accuracy", "all"):
            if key not in entry:
                defects.append(f"nulls_per_tier[{tier_key}] missing {key!r}")
        acc = entry.get("accuracy")
        if isinstance(acc, bool) or not isinstance(acc, (int, float)) \
                or not (acc == acc) or not (0.0 <= float(acc) <= 1.0):
            defects.append(f"nulls_per_tier[{tier_key}].accuracy not finite in [0,1]: {acc!r}")
        all_accs = entry.get("all")
        if isinstance(all_accs, dict):
            for k, v in all_accs.items():
                if isinstance(v, bool) or not isinstance(v, (int, float)) \
                        or not (v == v) or not (0.0 <= float(v) <= 1.0):
                    defects.append(
                        f"nulls_per_tier[{tier_key}].all[{k}] not finite in [0,1]: {v!r}")
    return defects


class _PrefinalFeeder:
    """Reconstructs the feeder view the pure finalizer needs (placed rows for
    the heuristic reference sample + the exact consumption ledger)."""

    def __init__(self, placed_rows: dict[str, int], ledger: dict[str, Any]):
        self.placed_rows = dict(placed_rows)
        self._ledger = dict(ledger)

    def ledger(self) -> dict[str, Any]:
        return self._ledger


# kwargs of build_arm_receipt that the finalizer-only path reconstructs from
# the snapshot instead of trusting verbatim (slices are deterministic code;
# feeder is rebuilt from the exact recorded consumption state).
_FINALIZER_RECONSTRUCTED = ("feeder", "slices")
_FINALIZER_SNAPSHOT_KEYS = (
    "tag", "cfg", "env", "n_seq", "length", "param_count", "citadel_sha",
    "cymek_sha", "seed", "feeder_placed_rows", "feeder_ledger", "ledgers",
    "done", "first_loss", "last_loss", "cap_total", "ans_total",
    "whole_total", "gsum", "gmax", "gn", "train_wall", "untrained",
    "untrained_dev", "untrained_train", "untrained_self", "trained_self",
    "self_diagnostics", "trained", "trained_recs",
    "trained_train", "train_memorization", "inter", "teacher_eval",
    "first_step", "ckpt_path", "ckpt_hash", "pre_sha", "post_sha",
    "reload_ok", "device_count", "wall")


def load_finalizer_kwargs(snapshot_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Bridge snapshot-shaped kwargs -> build_arm_receipt kwargs: rebuild the
    feeder view from the recorded consumption state and recompute the
    deterministic evaluation slices (never trusted from the snapshot)."""
    kwargs = dict(snapshot_kwargs)
    kwargs["feeder"] = _PrefinalFeeder(kwargs.pop("feeder_placed_rows"),
                                       kwargs.pop("feeder_ledger"))
    kwargs["slices"] = _tier_slices()
    return kwargs


def write_prefinal_snapshot(out_dir: str | Path, kwargs: dict[str, Any]) -> str:
    """Durably store every serializable input the PURE post-training
    finalizer needs, so an expensive arm can never be lost to a receipt
    bug again. Self-hash protected (payload_sha256 over the canonical body)."""
    body = {k: kwargs[k] for k in _FINALIZER_SNAPSHOT_KEYS if k in kwargs}
    missing = [k for k in _FINALIZER_SNAPSHOT_KEYS if k not in body]
    if missing:
        raise RuntimeError(f"prefinal snapshot incomplete, missing: {missing}")
    doc = {"schema": PREFINAL_SCHEMA, "payload_sha256":
           hashlib.sha256(_canonical_json(body)).hexdigest(), **body}
    path = Path(out_dir) / f"ARM_{kwargs['tag']}.prefinal.json"
    path.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


def load_prefinal_snapshot(out_dir: str | Path, tag: str, *, expect_cfg: dict,
                           seed: int, shape: tuple[int, int]
                           ) -> tuple[dict[str, Any] | None, str]:
    """Return (finalizer_kwargs | None, reason).

    Hash-verifies EVERYTHING before trusting it: sidecar self-hash, required
    keys, cfg/seed/shape agreement with the requested run, and the checkpoint
    file's SHA-256 against the recorded checkpoint hash. Any mismatch moves
    the sidecar aside (.invalid-<ts>) and returns None — a corrupt snapshot
    can never be silently trusted, and never blocks a fresh rerun.
    """
    import shutil

    path = Path(out_dir) / f"ARM_{tag}.prefinal.json"
    if not path.is_file():
        return None, ""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        path.rename(path.with_suffix(f".json.corrupt-{stamp}"))
        return None, f"unreadable sidecar ({type(exc).__name__}): archived"
    if not isinstance(doc, dict) or doc.get("schema") != PREFINAL_SCHEMA:
        return None, "unknown sidecar schema"
    body = {k: doc[k] for k in _FINALIZER_SNAPSHOT_KEYS if k in doc}
    missing = [k for k in _FINALIZER_SNAPSHOT_KEYS if k not in body]
    if missing:
        return None, f"sidecar missing keys: {missing}"
    digest = hashlib.sha256(_canonical_json(body)).hexdigest()
    if digest != doc.get("payload_sha256"):
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        path.rename(path.with_suffix(f".json.invalid-{stamp}"))
        return None, "payload hash mismatch: sidecar archived"
    if body["cfg"] != expect_cfg or body["seed"] != seed \
            or (body["n_seq"], body["length"]) != tuple(shape):
        return None, ("sidecar belongs to a different run "
                      f"(cfg/seed/shape mismatch); archived for the fresh run")
    ckpt = Path(body["ckpt_path"])
    if not ckpt.is_file():
        return None, "checkpoint file from snapshot is missing"
    actual = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    if actual != body["ckpt_hash"]:
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        path.rename(path.with_suffix(f".json.invalid-{stamp}"))
        return None, ("checkpoint hash mismatch: snapshot archived "
                      "(checkpoint changed since snapshot)")
    kwargs = load_finalizer_kwargs(body)
    return kwargs, f"verified snapshot (checkpoint {body['ckpt_hash'][:12]}...)"


def validate_arm_receipt(receipt: dict[str, Any]) -> list[str]:
    """Terminal arm receipt validator (pure).

    SCIENTIFIC_PASS / SCIENTIFIC_FAIL receipts must carry the complete
    contract: untrained/trained/trained_train/train_memorization/nulls_per_tier
    each exactly t0-t4 with valid summaries, non-empty intermediates/
    diagnostics/gate_rules, checkpoint path+sha256, reload hashes + boolean,
    training and data ledgers, and a valid status. Returns defect strings.
    IMPLEMENTATION_FAILURE / TIMEBOX_ABORT receipts carry partial data by
    design and are validated only for a known status.
    """
    defects: list[str] = []
    status = receipt.get("status")
    if status not in ("SCIENTIFIC_PASS", "SCIENTIFIC_FAIL"):
        if status not in ("IMPLEMENTATION_FAILURE", "TIMEBOX_ABORT"):
            defects.append(f"unknown status {status!r}")
        return defects
    for block in ("untrained", "trained", "trained_train",
                  "train_memorization", "nulls_per_tier"):
        d = receipt.get(block)
        if not isinstance(d, dict) or set(d) != set(TIER_KEYS):
            got = sorted(d) if isinstance(d, dict) else type(d).__name__
            defects.append(f"{block} must have exactly t0-t4 (got {got})")
            continue
        if block in ("untrained", "trained", "trained_train"):
            for tk in TIER_KEYS:
                _validate_summary(f"{block}[{tk}]", d[tk], defects)
    defects.extend(validate_null_block(receipt))
    for key in ("intermediates", "diagnostics", "gate_rules"):
        if not isinstance(receipt.get(key), dict) or not receipt[key]:
            defects.append(f"{key} missing or empty")
    ckpt = receipt.get("checkpoint")
    if not (isinstance(ckpt, dict) and ckpt.get("path") and ckpt.get("sha256")):
        defects.append("checkpoint path+sha256 required")
    for key in ("pre_reload_prediction_sha256", "post_reload_prediction_sha256"):
        v = receipt.get(key)
        if not isinstance(v, str) or len(v) != 64:
            defects.append(f"{key} must be a sha256 hex string")
    if not isinstance(receipt.get("reload_identical"), bool):
        defects.append("reload_identical must be boolean")
    training = receipt.get("training")
    if not (isinstance(training, dict) and training.get("ledgers")
            and training.get("updates") is not None):
        defects.append("training ledger required")
    data = receipt.get("data")
    if not (isinstance(data, dict) and isinstance(data.get("feeder"), dict)):
        defects.append("data ledger required")
    return defects


def build_arm_receipt(*, tag: str, cfg: dict[str, Any], env: dict[str, Any],
                      n_seq: int, length: int, param_count: int,
                      citadel_sha: str, cymek_sha: str, feeder, seed: int,
                      slices: dict[str, Any], ledgers: list, done: int,
                      first_loss, last_loss, cap_total: int, ans_total: int,
                      whole_total: int, gsum: float, gmax: float, gn: int,
                      train_wall: float, untrained: dict[str, Any],
                      untrained_dev: dict[str, Any] | None = None,
                      untrained_self: dict[str, Any] | None = None,
                      trained_self: dict[str, Any] | None = None,
                      self_diagnostics: dict[str, Any] | None = None,
                      untrained_train: dict[str, Any], trained: dict[str, Any],
                      trained_recs: dict[str, Any], trained_train: dict[str, Any],
                      train_memorization: dict[str, Any], inter: dict[str, Any],
                      teacher_eval: dict[str, Any], first_step: dict[str, Any],
                      ckpt_path: str, ckpt_hash: str, pre_sha: str,
                      post_sha: str, reload_ok: bool, device_count: int,
                      wall: float) -> dict[str, Any]:
    """Pure post-training arm finalization (no device; simulation-covered).

    Computes: heuristic nulls (global + per-tier t0-t4, assignment inside the
    loop), diagnostic aggregations, easy-tier memorization lens, lift tiers
    (train lift only from verified-consumed rows), the pooled scientific gate
    (reload alone can NEVER pass), and the full receipt dict. Raises
    RuntimeError on any null-block schema defect: an incomplete receipt can
    never be written.
    """
    from citadel_tpu import calculator_eval as cev
    from citadel_tpu import tiered_data as td

    # Defense in depth: never depend on the producer being correct. The
    # untrained TEST block is canonicalized (t0..t4) and validated here, so a
    # producer/consumer key mismatch surfaces as ARM_SCHEMA_INVALID — never
    # as a raw KeyError after expensive training (real TPU failure class).
    untrained = normalize_untrained_receipt(untrained)

    all_test_rows, all_test_tgts = [], []
    for tier in range(5):
        all_test_rows.extend(slices[f"test_t{tier}"]["rows"])
        all_test_tgts.extend(slices[f"test_t{tier}"]["targets"])
    ref_rows: list[str] = []
    for key, cnt in feeder.placed_rows.items():
        if key.startswith("tier:") and cnt > 0:
            t = int(key.split(":")[1])
            idx = random.Random(seed + 13).sample(range(cnt), min(200, cnt))
            ref_rows.extend(td.tier_row(t, "train", i)[0] for i in idx)
    nulls = cev.heuristic_nulls(all_test_rows, ref_rows)
    null_summaries = {k: cev.summarize(v, all_test_tgts) for k, v in nulls.items()}
    null_name, null_best = cev.strongest_null_accuracy(null_summaries)
    nulls_per_tier: dict[str, Any] = {}
    for tier in range(5):
        tn = cev.heuristic_nulls(slices[f"test_t{tier}"]["rows"], ref_rows)
        ts = {k: cev.summarize(v, slices[f"test_t{tier}"]["targets"])
              for k, v in tn.items()}
        nn, nb = cev.strongest_null_accuracy(ts)
        # Assignment INSIDE the loop: every tier t0-t4 is recorded. (The
        # pre-fix version wrote only the final tier here and crashed the
        # scientific gate and the cross-arm classifier AFTER expensive
        # training.)
        nulls_per_tier[f"t{tier}"] = {"strongest": nn, "accuracy": nb,
                                      "all": {k: v["accuracy"] for k, v in ts.items()}}
    defects = validate_null_block({"heuristic_nulls": null_summaries,
                                   "nulls_per_tier": nulls_per_tier})
    if defects:
        raise RuntimeError("abort NULL_BLOCK_INVALID: " + "; ".join(defects))

    all_trained_recs = [r for t in range(5) for r in trained_recs[f"t{t}"]]

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

    lift_train = {f"t{t}": {"accuracy": trained_train[f"t{t}"]["accuracy"],
                            "n_verified_consumed":
                                train_memorization[f"t{t}"]["n_verified_consumed"],
                            "status": train_memorization[f"t{t}"]["status"],
                            "lift": bool(train_memorization[f"t{t}"]["lift_eligible"])}
                  for t in range(1, 5)}
    lift_test = {f"t{t}": {"accuracy": trained[f"t{t}"]["accuracy"],
                           "lift": _lift_tier(trained[f"t{t}"]["accuracy"],
                                              trained[f"t{t}"]["total"])}
                 for t in range(1, 5)}

    def first_lift(d):
        for t in (1, 2, 3, 4):
            if d[f"t{t}"]["lift"]:
                return t
        return None

    # Per-arm scientific gate (pooled tiers 1-4): reload alone can NEVER pass.
    _k = sum(trained[f"t{t}"]["correct"] for t in (1, 2, 3, 4))
    _n = sum(trained[f"t{t}"]["total"] for t in (1, 2, 3, 4))
    _uk = sum(untrained[f"t{t}"]["correct"] for t in (1, 2, 3, 4))
    _un = sum(untrained[f"t{t}"]["total"] for t in (1, 2, 3, 4))
    _lcb, _ = cev.wilson(_k, _n)
    _, _uucb = cev.wilson(_uk, _un)
    _null_ref = max([null_best] + [nulls_per_tier[f"t{t}"]["accuracy"]
                                   for t in (1, 2, 3, 4)])
    _rules = {
        "nonoverlap": bool((_k / _n if _n else 0.0) > (_uk / _un if _un else 0.0)
                           and _lcb > _uucb),
        "beats_null": bool(_lcb > _null_ref),
        "margin": bool((_k / _n if _n else 0.0) - (_uk / _un if _un else 0.0) >= 0.10),
        "loss": bool(first_loss is not None and last_loss is not None
                     and last_loss < first_loss),
        "reload": bool(reload_ok),
    }
    _status = "SCIENTIFIC_PASS" if all(_rules.values()) else "SCIENTIFIC_FAIL"
    receipt = {
        "schema": "citadel-t1d-arm/v1", "arm": tag, "config": cfg,
        "citadel_sha": citadel_sha, "cymek_runtime_sha": cymek_sha,
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
        "untrained": untrained,
        "untrained_dev": (untrained_dev
                          if isinstance(untrained_dev, dict) and untrained_dev
                          else {}),
        "untrained_self": (untrained_self
                           if isinstance(untrained_self, dict) else {}),
        "trained_self": (trained_self
                         if isinstance(trained_self, dict) else {}),
        "self_diagnostics": (self_diagnostics
                             if isinstance(self_diagnostics, dict) else {}),
        "untrained_train": untrained_train,
        "trained": trained, "trained_train": trained_train,
        "train_memorization": train_memorization,
        "intermediates": inter,
        "diagnostics": {
            "stop_histogram": cev.stop_histogram(
                [r for t in range(5) for r in trained_recs[f"t{t}"]]),
            "samples": cev.sample_records(
                [r for t in range(5) for r in trained_recs[f"t{t}"]], 20, seed),
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
        "gate_rules": _rules,
        "pre_reload_prediction_sha256": pre_sha,
        "post_reload_prediction_sha256": post_sha,
        "reload_identical": reload_ok,
        "checkpoint": {"path": ckpt_path, "sha256": ckpt_hash},
        "device_count": device_count, "wall_seconds": wall,
        "status": _status,
    }
    terminal_defects = validate_arm_receipt(receipt)
    if terminal_defects:
        raise RuntimeError("ARM_SCHEMA_INVALID: " + "; ".join(terminal_defects))
    return receipt


def write_arm_receipt(out_dir: str | Path, receipt: dict[str, Any], *,
                      ckpt_hash: str) -> None:
    """Durably write the arm receipt + completion marker.

    Receipt first, marker second: every terminal state (SCIENTIFIC_PASS,
    SCIENTIFIC_FAIL, TIMEBOX_ABORT, IMPLEMENTATION_FAILURE) always lands as
    an ARM_<tag>.json on disk — the bundle guarantee. Interrupted pairs fail
    loudly via the resume predicate, never silently.
    """
    root = Path(out_dir)
    out_path = root / f"ARM_{receipt['arm']}.json"
    out_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    (root / f"ARM_{receipt['arm']}.done.json").write_text(
        json.dumps({"receipt": str(out_path), "status": receipt["status"],
                    "checkpoint_sha256": ckpt_hash}, indent=2), encoding="utf-8")


def select_calibrated_shape(results: list[dict[str, Any]], *,
                            scale2_verifier) -> tuple[dict[str, Any] | None, str]:
    """Pure shape selection: max tok/s among correctness-passing candidates
    whose shape ALSO verifies on SCALE2 (Arm D must never be the first place
    the session-wide shape is tried). A failed candidate is marked failed IN
    PLACE — no duplicate dicts left behind that keep a rejected shape
    accidentally selectable. Returns (selected | None, note)."""
    feasible = [r for r in results if r.get("correct")]
    if not feasible:
        return None, "no candidate passed correctness"
    ordered = sorted(feasible, key=lambda r: r.get("tokens_per_second", 0.0),
                     reverse=True)
    for cand in ordered:
        if scale2_verifier(cand["batch"], cand["length"]):
            return cand, "pass"
        cand["correct"] = False
        cand["error"] = "SCALE2_VERIFICATION_FAILED"
    return None, "no shape safe for SCALE2"


def _pooled(arm: dict[str, Any], key: str) -> tuple[int, int]:
    from citadel_tpu import calculator_eval as cev

    k = sum(arm[key][f"t{t}"]["correct"] for t in (1, 2, 3, 4))
    n = sum(arm[key][f"t{t}"]["total"] for t in (1, 2, 3, 4))
    return k, n


CLASSIFY_ORDER = ("CAPABILITY_LIFTED", "CURRICULUM_HELPED", "TEACHER_HELPED",
                  "SCALE_HELPED", "REPRESENTATION_LIMITED", "SELF_KNOWLEDGE_ACQUIRED",
                  "SELF_PROBE_LEAKAGE", "BELOW_FIT_FLOOR",
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
            def _dev_mean(cp: int):
                block = inter[str(cp)]
                vals = [block[f"t{tier}"]["exact"] for tier in (1, 2, 3, 4)
                        if isinstance(block, dict) and f"t{tier}" in block
                        and isinstance(block[f"t{tier}"], dict)
                        and block[f"t{tier}"].get("exact") is not None]
                return (sum(vals) / len(vals)) if vals else None
            early = _dev_mean(keys[-2])
            late = _dev_mean(keys[-1])
            te = pooled[t]["acc"]
            if (early is not None and late is not None
                    and late - early >= 0.05 and te < 0.10):
                fired["BUDGET_LIMITED"] = (
                    f"arm {t}: mean dev tiers1-4 still rising between the last "
                    f"two checkpoints ({early:.2f}->{late:.2f}); pooled test {te:.2f}")
                break
    # Self-knowledge rules (SELF_KNOWLEDGE AMENDMENT). Acquisition needs arm
    # F; leakage is checked on every arm that carries self probes.
    if any(arms[t].get("trained_self") for t in present):
        def _self_block(t: str):
            return arms[t].get("trained_self") or {}
        f_self = _self_block("F")
        f_untr = arms.get("F", {}).get("untrained_self") or {}
        f_null = (arms.get("F", {}).get("self_diagnostics") or {}).get(
            "most_common_null") or {}
        if f_self and f_untr and f_null and "F" in present:
            acquired = (f_self.get("wilson_lcb", 0.0)
                        >= f_untr.get("wilson_lcb", 0.0) + 0.10
                        and f_self.get("wilson_lcb", 0.0)
                        > f_null.get("wilson_lcb", 0.0) + 0.10)
            if acquired:
                fired["SELF_KNOWLEDGE_ACQUIRED"] = (
                    f"arm F probe LCB {f_self.get('wilson_lcb', 0.0):.2f} "
                    f"> untrained {f_untr.get('wilson_lcb', 0.0):.2f} + 0.10 "
                    f"and > null {f_null.get('wilson_lcb', 0.0):.2f} + 0.10")
        for t in sorted(present):
            if t == "F":
                continue
            s_self, s_untr = _self_block(t), arms[t].get("untrained_self") or {}
            if s_self and s_untr and (s_self.get("wilson_lcb", 0.0)
                                      >= s_untr.get("wilson_lcb", 0.0) + 0.10):
                fired["SELF_PROBE_LEAKAGE"] = (
                    f"arm {t} passes the self-probe bar without self data - "
                    "probe disjointness is broken; no acquisition claim")
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
    for tag in ARM_ORDER:
        try:
            receipt = run_arm(tag, budgets[tag], shape=shape, out_dir=str(root), seed=seed)
            arm_receipts[tag] = receipt
            print(f"arm {tag}: {receipt.get('status')} "
                  f"pooled14={sum(receipt['trained'][f't{t}']['accuracy'] for t in (1, 2, 3, 4)) / 4:.4f}",
                  flush=True)
        except Exception as exc:  # noqa: BLE001 - per-arm isolation per plan
            infra_failures += 1
            # Durable terminal receipt on disk for EVERY arm, including this
            # one (bundle guarantee). No marker: a rerun retries the arm.
            receipt = arm_failure_receipt(tag, exc, citadel_sha=rb.citadel_sha(),
                                          cymek_sha=rt_sha)
            (root / f"ARM_{tag}.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
            arm_receipts[tag] = receipt
            print(f"arm {tag}: IMPLEMENTATION_FAILURE {exc}", flush=True)
            if infra_failures >= 2:
                raise RuntimeError("abort SESSION: 2nd infra failure") from exc
    return _assemble_session_results(
        root, arm_receipts, shape=shape, rate=rate, scaled=scaled,
        budgets={t: c["budget"] for t, c in budgets.items()}, rt_sha=rt_sha)


def arm_failure_receipt(tag: str, exc: Exception, *, citadel_sha: str,
                        cymek_sha: str) -> dict[str, Any]:
    """Terminal IMPLEMENTATION_FAILURE receipt for one arm (pure builder)."""
    return {"schema": "citadel-t1d-arm/v1", "arm": tag,
            "status": "IMPLEMENTATION_FAILURE",
            "error": f"{type(exc).__name__}: {exc}",
            "citadel_sha": citadel_sha, "cymek_runtime_sha": cymek_sha}


def _write_pre50m_failure_receipts(root: Path, exc: Exception, *,
                                   citadel_sha: str, cymek_sha: str) -> None:
    """PRE50M failed: NEVER lose the T1D arms or the bundle.

    Every still-missing required PRE50M/diagnostics artifact gets an explicit
    failure receipt, and NEXT_50M_DECISION.json is (re)written fail-closed
    with ready_for_50m_training=false and a precise blocker.
    """
    for name in ("PRE50M_TARGET.json", "PRE50M_FEASIBILITY.json",
                 "PRE50M_THROUGHPUT.json", "PRE50M_CHECKPOINT_SMOKE.json",
                 "PRE50M_DATA_INTERFACE.json", "PRE50M_PACKING.json",
                 "DIAGNOSTICS.json"):
        if (root / name).is_file():
            continue
        (root / name).write_text(json.dumps(
            {"status": "IMPLEMENTATION_FAILURE",
             "phase": name,
             "error": f"{type(exc).__name__}: {exc}",
             "citadel_sha": citadel_sha, "cymek_runtime_sha": cymek_sha},
            indent=2, sort_keys=True), encoding="utf-8")
    (root / "NEXT_50M_DECISION.json").write_text(json.dumps(
        {"schema": "citadel-pre50m-decision/v1",
         "ready_for_50m_training": False,
         "blocking_reasons": [f"PRE50M implementation failure: "
                              f"{type(exc).__name__}: {exc}"],
         "status": "IMPLEMENTATION_FAILURE",
         "citadel_sha": citadel_sha, "cymek_runtime_sha": cymek_sha},
        indent=2, sort_keys=True), encoding="utf-8")


def _run_pre50m_phase(root: Path, arm_receipts: dict[str, Any], *, rt_sha: str,
                      rate: float, shape: tuple[int, int]) -> dict[str, Any]:
    """The real PRE50M phase (device-bound); writes every required receipt."""
    from citadel_tpu import pre50m as p50  # noqa: E402 (phase import)
    from citadel_tpu import runtime_bootstrap as rb
    from citadel_tpu import t1c_run as t1c

    citadel_sha = rb.citadel_sha()
    (root / "PRE50M_TARGET.json").write_text(json.dumps(
        {"schema": "citadel-pre50m-target/v1", **p50.PRE50M_TARGET,
         "citadel_sha": citadel_sha, "cymek_runtime_sha": rt_sha},
        indent=2, sort_keys=True), encoding="utf-8")
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
    oom = p50.oom_decision(cal_candidates(root))
    feas["grad_accumulation"] = grad_accum
    feas["oom_selection"] = oom
    (root / "PRE50M_FEASIBILITY.json").write_text(json.dumps(
        {"schema": "citadel-pre50m-feasibility/v1",
         "citadel_sha": citadel_sha, "cymek_runtime_sha": rt_sha,
         "memory": feas, "grad_accumulation": grad_accum,
         "oom_selection": oom}, indent=2, sort_keys=True), encoding="utf-8")
    (root / "PRE50M_THROUGHPUT.json").write_text(json.dumps(
        {"schema": "citadel-pre50m-throughput/v1",
         "citadel_sha": citadel_sha, "cymek_runtime_sha": rt_sha,
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
                "value_tokens": p50.PRE50M_TARGET["value_tokens"],
                "parameter_count": None},
        smoke=smoke, feasibility={"verdict": feas["SCALE2_7_4M"]["verdict"]},
        data_interface=di, packing=packing,
        recommended_batch=shape[0], recommended_sequence_length=shape[1],
        rate_tok_s=rate)
    (root / "NEXT_50M_DECISION.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    return {"status": "PASS", "decision": decision}


def cal_candidates(root: Path) -> list[dict[str, Any]]:
    """Calibration candidates from the session receipt (fail-closed default)."""
    cal_path = Path(root) / "CALIBRATION.json"
    if not cal_path.is_file():
        return []
    try:
        return list(json.loads(cal_path.read_text(encoding="utf-8")).get("candidates", []))
    except Exception:
        return []


def build_lift_curves(arm_receipts: dict[str, Any]) -> dict[str, Any]:
    """Pure LIFT_OFF_CURVES assembly. Scientific arms contribute full curves;
    every other terminal status is represented by status alone — the curve
    file never indexes scientific fields a failed arm does not carry."""
    curves: dict[str, Any] = {}
    for tag, r in arm_receipts.items():
        if r.get("status") not in ("SCIENTIFIC_PASS", "SCIENTIFIC_FAIL"):
            curves[tag] = {"status": r.get("status")}
            continue
        latest = str(max((int(k) for k in r["intermediates"] if str(k).isdigit()),
                         default=0))
        curves[tag] = {
            "status": r.get("status"),
            "train": {f"t{t}": r["trained_train"][f"t{t}"]["accuracy"]
                      for t in range(5)},
            "dev": {f"t{t}": (r["intermediates"][latest][f"t{t}"]["exact"]
                              if r["intermediates"]
                              and f"t{t}" in r["intermediates"][latest]
                              else None) for t in range(5)},
            "test": {f"t{t}": r["trained"][f"t{t}"]["accuracy"] for t in range(5)},
            "untrained_test": {f"t{t}": r["untrained"][f"t{t}"]["accuracy"]
                               for t in range(5)},
            "first_train_lift_tier": r["diagnostics"].get("first_train_lift_tier"),
            "first_test_lift_tier": r["diagnostics"].get("first_test_lift_tier")}
        if r.get("trained_self"):
            curves[tag]["self_probe"] = r["trained_self"].get("accuracy")
    return curves


def producer_consumer_contract_probe(*, legacy_untrained_keys: bool = True,
                                     n_updates: int = 400) -> list[str]:
    """THE producer -> finalizer -> classifier -> curves contract bridge.

    Builds the EXACT producer-shaped data run_arm constructs (including the
    pre-fix untrained keying when legacy_untrained_keys=True — the shape that
    caused the real TPU failure KeyError: 't1'), runs the full pure path —
    normalize -> build_arm_receipt -> validate_arm_receipt ->
    classify_cross_arm -> build_lift_curves — on REAL generator rows and REAL
    feeder consumption, and returns defect strings (empty = the whole
    producer/consumer contract holds). No device anywhere.
    """
    from citadel_tpu import calculator_eval as cev
    from citadel_tpu import tiered_data as td

    defects: list[str] = []
    try:
        feeder = TierFeeder("curriculum", 8, 64)
        for u in range(n_updates):
            feeder.fill_sequences(u / n_updates)
        slices = _tier_slices()
        cands = frozen_train_candidates()
        plan = train_memorization_plan(feeder, candidates=cands)

        def s(acc, n=500):
            return {"correct": int(acc * n), "total": n, "accuracy": acc,
                    "wilson_lcb": max(0.0, acc - 0.04),
                    "wilson_ucb": min(1.0, acc + 0.04)}

        # EXACT producer shape. legacy_untrained_keys=True reproduces the
        # pre-fix run_arm producer: dev_tN/test_tN keys in one dict.
        untrained_producer: dict[str, Any] = {}
        for tier in range(5):
            dev_row = slices[f"dev_t{tier}"]["targets"][0]
            test_row = slices[f"test_t{tier}"]["targets"][0]
            if legacy_untrained_keys:
                untrained_producer[f"dev_t{tier}"] = s(0.0, n=200)
                untrained_producer[f"test_t{tier}"] = s(0.0)
            else:
                untrained_producer[f"t{tier}"] = s(0.0)
        untrained_dev = {f"t{t}": s(0.0, n=200) for t in range(5)} \
            if not legacy_untrained_keys else \
            {f"t{t}": untrained_producer[f"dev_t{t}"] for t in range(5)}
        accs = {0: 0.9, 1: 0.5, 2: 0.3, 3: 0.1, 4: 0.02}
        trained = {f"t{t}": s(accs[t]) for t in range(5)}
        trained_recs = {f"t{t}": [
            {"prompt": "probe", "target": tg, "prediction": tg,
             "correct": True, "stop_reason": "EOS",
             "generated_token_count": len(tg), "valid": True}
            for tg in slices[f"test_t{t}"]["targets"]] for t in range(5)}
        trained_train, train_memorization = {}, {}
        for tier in range(5):
            entry = plan[tier]
            rows = [td.tier_row(tier, "train", i)[0]
                    for i in entry["verified_indices"]]
            tgts = [cev.split_prompt_target(r)[1] for r in rows]
            summ = s(0.6, n=len(rows)) if rows else s(0.0, n=0)
            trained_train[f"t{tier}"] = summ
            train_memorization[f"t{tier}"] = {
                "consumed_prefix": entry["consumed_prefix"],
                "n_frozen_candidates": entry["n_candidates"],
                "n_verified_consumed": entry["n_verified"],
                "evaluated_rows": len(rows),
                "status": entry["status"],
                "lift_eligible": bool(entry["status"] == "OK"
                                      and summ["accuracy"] >= LIFT_THRESHOLD)}
        inter = {str(cp): {f"t{tier}": {"exact": dev, "lcb": max(0.0, dev - 0.03)}
                           for tier in range(5)}
                 for cp, dev in ((25, 0.05), (50, 0.12), (75, 0.18), (100, 0.22))}
        if legacy_untrained_keys:
            # the finalizer MUST canonicalize the legacy producer shape
            untrained_arg = untrained_producer
        else:
            untrained_arg = untrained_producer
        receipt = build_arm_receipt(
            tag="A", cfg=dict(ARMS["A"]), env={"probe_pass": True},
            n_seq=8, length=64, param_count=3_737_472,
            citadel_sha="0" * 40, cymek_sha="1" * 64,
            feeder=feeder, seed=20260904, slices=slices,
            ledgers=[{"updates": 100, "first_loss": 9.0, "last_loss": 6.0}],
            done=100, first_loss=9.0, last_loss=6.0, cap_total=100 * 8 * 64,
            ans_total=1234, whole_total=42_000, gsum=3.0, gmax=0.9, gn=100,
            train_wall=60.0, untrained=untrained_arg,
            untrained_dev=untrained_dev,
            untrained_train={f"t{t}": s(0.0, n=200) for t in range(5)},
            trained=trained, trained_recs=trained_recs,
            trained_train=trained_train,
            train_memorization=train_memorization, inter=inter,
            teacher_eval={"skipped": "n/a"}, first_step={"n": 0},
            ckpt_path="probe.pt", ckpt_hash="a" * 64,
            pre_sha="p" * 64, post_sha="p" * 64, reload_ok=True,
            device_count=1, wall=90.0)
        receipt_defects = validate_arm_receipt(receipt)
        defects.extend(receipt_defects)
        summary = classify_cross_arm({"A": receipt})
        if not summary.get("labels"):
            defects.append("classifier produced no labels")
        curves = build_lift_curves({"A": receipt})
        curve = curves["A"]
        for tier_key in TIER_KEYS:
            for block in ("train", "dev", "test", "untrained_test"):
                if tier_key not in curve[block]:
                    defects.append(f"curves[{block}] missing {tier_key}")
        if set(receipt["untrained"]) != set(TIER_KEYS):
            defects.append("final receipt untrained block is not canonical t0-t4")
        if legacy_untrained_keys and "test_t0" in receipt["untrained"]:
            defects.append("legacy producer keys leaked into the final receipt")
    except Exception as exc:
        defects.append(f"{type(exc).__name__}: {exc}")
    return defects


def _assemble_session_results(root: Path, arm_receipts: dict[str, Any], *,
                              shape: tuple[int, int], rate: float, scaled: bool,
                              budgets: dict[str, int], rt_sha: str,
                              pre50m_runner=None) -> dict[str, Any]:
    """Post-arm session assembly (device-free except the injected PRE50M
    runner): cross-arm classification, lift curves, PRE50M phase, session
    manifest, bundle. PRE50M failure never destroys T1D evidence.
    """
    from citadel_tpu import runtime_bootstrap as rb

    root = Path(root)
    citadel_sha = rb.citadel_sha()
    # Session-boundary schema gate: a receipt claiming a scientific verdict
    # must satisfy the terminal contract, or it is NOT a scientific result.
    # Demote it (durably) to IMPLEMENTATION_FAILURE instead of letting the
    # classifier crash on a missing tier key — no classifier KeyError is
    # acceptable, whatever wrote the receipt.
    scientific: dict[str, Any] = {}
    for t, r in arm_receipts.items():
        if r.get("status") not in ("SCIENTIFIC_PASS", "SCIENTIFIC_FAIL"):
            continue
        schema_defects = validate_arm_receipt(r)
        if schema_defects:
            demoted = {**r, "status": "IMPLEMENTATION_FAILURE",
                       "schema_defects": schema_defects}
            (root / f"ARM_{t}.json").write_text(
                json.dumps(demoted, indent=2, sort_keys=True), encoding="utf-8")
            arm_receipts[t] = demoted
            print(f"arm {t}: receipt failed terminal schema validation -> "
                  f"IMPLEMENTATION_FAILURE ({'; '.join(schema_defects[:2])})",
                  flush=True)
            continue
        scientific[t] = r
    runner = _run_pre50m_phase if pre50m_runner is None else pre50m_runner
    try:
        pre50m_status = runner(root, arm_receipts, rt_sha=rt_sha, rate=rate,
                               shape=tuple(shape))
    except Exception as exc:  # noqa: BLE001 - preserve arms, record, continue
        pre50m_status = {"status": "IMPLEMENTATION_FAILURE",
                         "error": f"{type(exc).__name__}: {exc}"}
        _write_pre50m_failure_receipts(root, exc, citadel_sha=citadel_sha,
                                       cymek_sha=rt_sha)
        print(f"pre50m: IMPLEMENTATION_FAILURE {exc}", flush=True)
    else:
        decision = pre50m_status.get("decision", {})
        print("pre50m:", decision.get("ready_for_50m_training"),
              decision.get("blocking_reasons"), flush=True)
    # classification/curves/manifest/bundle happen in summarize_session
    return summarize_session(root, arm_receipts, shape=shape, rate=rate,
                             scaled=scaled, budgets=budgets, rt_sha=rt_sha,
                             pre50m_status=pre50m_status)


def summarize_session(root: Path, arm_receipts: dict[str, Any], *, shape,
                      rate: float, scaled: bool, budgets: dict[str, int],
                      rt_sha: str,
                      pre50m_status: dict[str, Any] | None = None
                      ) -> dict[str, Any]:
    """Post-arm session assembly WITHOUT the PRE50M phase (already run):
    cross-arm classification, lift curves, session manifest, bundle."""
    from citadel_tpu import runtime_bootstrap as rb

    root = Path(root)
    citadel_sha = rb.citadel_sha()
    if pre50m_status is None:  # orchestrator runs PRE50M as its own phase
        dpath = root / "NEXT_50M_DECISION.json"
        pre50m_status = ({"status": "PASS",
                          "decision": json.loads(dpath.read_text(encoding="utf-8"))}
                         if dpath.is_file() else
                         {"status": "NOT_RUN"})
    scientific = {t: r for t, r in arm_receipts.items()
                  if r.get("status") in ("SCIENTIFIC_PASS", "SCIENTIFIC_FAIL")}
    summary = classify_cross_arm(scientific)
    (root / "CROSS_ARM_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print("cross-arm labels:", summary["labels"], flush=True)
    curves = build_lift_curves(arm_receipts)
    (root / "LIFT_OFF_CURVES.json").write_text(
        json.dumps({"schema": "citadel-t1d-lift-off-curves/v1", "arms": curves},
                   indent=2, sort_keys=True), encoding="utf-8")
    session = {"schema": "citadel-t1d-session/v1",
               "citadel_sha": citadel_sha, "cymek_runtime_sha": rt_sha,
               "shape": list(shape), "calibrated_rate": rate, "budgets_scaled": scaled,
               "budgets": budgets,
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


BUNDLE_FILES = ["SESSION_MANIFEST.json", "DATA_MANIFEST.json", "CALIBRATION.json",
                "ARM_A.json", "ARM_B.json", "ARM_C.json", "ARM_D.json", "ARM_E.json",
                "ARM_F.json",
                "LIFT_OFF_CURVES.json", "CROSS_ARM_SUMMARY.json",
                "PRE50M_TARGET.json", "PRE50M_FEASIBILITY.json",
                "PRE50M_THROUGHPUT.json", "PRE50M_CHECKPOINT_SMOKE.json",
                "PRE50M_DATA_INTERFACE.json", "PRE50M_PACKING.json",
                "DIAGNOSTICS.json", "NEXT_50M_DECISION.json"]


BUNDLE_KNOWN_STATUSES = {"SCIENTIFIC_PASS", "SCIENTIFIC_FAIL", "TIMEBOX_ABORT",
                         "IMPLEMENTATION_FAILURE", "PASS", "FAIL",
                         "COMPLETE", "FAILED", "RUNNING", "NOT_RUN"}


def verify_bundle(session_dir: str) -> dict[str, Any]:
    """Reopen + validate the result bundle BEFORE download (Cell F gate).

    Checks: every required file present and JSON-parseable; every receipt
    that carries a status carries a KNOWN terminal status (implementation
    failures and PRE50M failure placeholders included — they must exist, not
    be absent); checkpoint SHAs in arm receipts match the bundled binaries
    when the binaries were bundled; the zip contains every member and
    decompresses cleanly. Raises with an exact defect list otherwise. Pure
    file I/O (no device).
    """
    import hashlib as _hl

    root = Path(session_dir)
    required = list(BUNDLE_FILES)
    defects: list[str] = []
    loaded: dict[str, Any] = {}
    for name in required:
        p = root / name
        if not p.is_file():
            defects.append(f"missing {name}")
            continue
        try:
            loaded[name] = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            defects.append(f"{name} unparseable: {type(exc).__name__}")
    for name, doc in loaded.items():
        if isinstance(doc, dict) and "status" in doc \
                and doc.get("status") not in BUNDLE_KNOWN_STATUSES:
            defects.append(f"{name} unknown status {doc.get('status')!r}")
    for tag in ARM_ORDER:
        r = loaded.get(f"ARM_{tag}.json")
        if isinstance(r, dict) and r.get("status") in ("SCIENTIFIC_PASS",
                                                       "SCIENTIFIC_FAIL"):
            for defect in validate_arm_receipt(r):
                defects.append(f"ARM_{tag}.json {defect}")
    zp = root / "CITADEL_T1D_RESULTS.zip"
    zip_bytes: dict[str, bytes] = {}
    if not zp.is_file():
        defects.append("CITADEL_T1D_RESULTS.zip missing")
    else:
        try:
            with zipfile.ZipFile(zp) as zf:
                names = set(zf.namelist())
                for name in required:
                    if name not in names:
                        defects.append(f"zip missing {name}")
                bad = zf.testzip()
                if bad is not None:
                    defects.append(f"zip corrupt at {bad}")
                for ckpt in [n for n in names if n.endswith(".pt")]:
                    zip_bytes[ckpt] = zf.read(ckpt)
        except Exception as exc:
            defects.append(f"zip unreadable: {type(exc).__name__}: {exc}")
    # checkpoint identity: when binaries are bundled, each arm receipt's
    # checkpoint sha256 must match the bundled bytes exactly
    if zip_bytes:
        for tag in ARM_ORDER:
            r = loaded.get(f"ARM_{tag}.json")
            if not isinstance(r, dict):
                continue
            ckpt = r.get("checkpoint") or {}
            sha = ckpt.get("sha256")
            member = f"t1d_arm_{tag.lower()}.pt"
            if sha and member in zip_bytes:
                actual = _hl.sha256(zip_bytes[member]).hexdigest()
                if actual != sha:
                    defects.append(f"ARM_{tag}.json checkpoint sha mismatch "
                                   f"({member})")
    if defects:
        raise RuntimeError(f"abort BUNDLE_INVALID: {'; '.join(defects)}")
    return {"schema": "citadel-t1d-bundle-verify/v1", "files": len(required),
            "status": "VALID"}


def build_bundle(session_dir: str, *, out: str) -> dict[str, Any]:
    """Assemble CITADEL_T1D_RESULTS.zip (receipts + manifest; binaries per cap rule)."""
    root = Path(session_dir)
    names = list(BUNDLE_FILES)
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
    "BUNDLE_FILES",
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
    "build_arm_receipt",
    "build_spec",
    "calibrate",
    "classify_cross_arm",
    "arm_failure_receipt",
    "build_lift_curves",
    "frozen_train_candidates",
    "load_finalizer_kwargs",
    "load_prefinal_snapshot",
    "normalize_untrained_receipt",
    "producer_consumer_contract_probe",
    "validate_arm_receipt",
    "write_prefinal_snapshot",
    "pack_rows",
    "run_arm",
    "run_session",
    "select_calibrated_shape",
    "should_skip_arm",
    "train_memorization_plan",
    "validate_null_block",
    "write_arm_receipt",
    "timebox_abort_receipt",
    "valid_alphabet_ids",
    "verify_bundle",
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

        if mode not in ("flat", "curriculum", "teacher", "masked", "self"):
            raise ValueError(f"unknown feeder mode {mode!r}")
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
        self.self_cursor = 0
        self._draw = 0
        self._pattern = 0
        self._carry: list[tuple[str, str, str]] = []  # (text, key, tier_tag)

    # -- state serialization (mid-arm training checkpoints, §12 of the
    # one-shot hardening: a disconnected arm resumes, never restarts)
    def state(self) -> dict[str, Any]:
        return {"mode": self.mode, "n_seq": self.n_seq, "length": self.length,
                "cursors": dict(self.cursors), "drawn": dict(self.drawn),
                "carried": dict(self.carried),
                "placed_rows": dict(self.placed_rows),
                "placed_tokens": dict(self.placed_tokens),
                "teacher_cursors": dict(self.teacher_cursors),
                "self_cursor": self.self_cursor,
                "draw": self._draw, "pattern": self._pattern,
                "carry": [list(t) for t in self._carry]}

    def load_state(self, state: dict[str, Any]) -> None:
        self.mode = state["mode"]
        self.cursors = {int(k): v for k, v in state["cursors"].items()}
        self.drawn = dict(state["drawn"])
        self.carried = dict(state["carried"])
        self.placed_rows = dict(state["placed_rows"])
        self.placed_tokens = dict(state["placed_tokens"])
        self.teacher_cursors = dict(state["teacher_cursors"])
        self.self_cursor = int(state.get("self_cursor", 0))
        self._draw = int(state["draw"])
        self._pattern = int(state["pattern"])
        self._carry = [tuple(t) for t in state["carry"]]

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

    def _self_row(self) -> tuple[str, str]:
        from citadel_tpu import self_knowledge as sk

        text, _ = sk.self_row(self.self_cursor, split="train")
        self.self_cursor = (self.self_cursor + 1) % sk.SELF_TRAIN_N
        key = "self:train"
        self.drawn[key] = self.drawn.get(key, 0) + 1
        return text, key

    def _refill(self, n: int, frac: float) -> None:
        for _ in range(n):
            if self.mode == "teacher" and (self._pattern % 10) >= self._ordinary_per_10:
                t, key = self._teacher_row()
            elif self.mode == "self" and (self._pattern % sk_self_mod()) == 0:
                t, key = self._self_row()
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
