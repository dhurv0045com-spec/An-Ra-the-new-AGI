"""Generation-based calculator evaluator (T1 primary metric). No torch at import.

Implements AMENDMENT_001 §§A3–A10: prompt/target split, strict integer
normalization, static-shape greedy generation ([B=8, L=32], index scatter, one
compiled graph), stop rules, exact-match + Wilson reporting, mechanical
heuristic nulls, prediction hashing, and the machine-readable data receipt.
Torch/XLA enter only inside `generate()` via injected modules (same host/device
split as T0: layout on CPU, step on XLA, per-step host sync intentional and
eval-only). `selftest()` runs the full torch-free validation suite (also used
by calculator_preflight and the unit tests).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


ENCODING_VERSION = "char-byte-offset/1.0"
ALPHABET = "0123456789+-*/= \n"
MAX_ANSWER_TOKENS = 8
EVAL_BATCH = 8
EVAL_LENGTH = 32
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
NEWLINE_ID = (ord("\n") % 250) + 2  # 12

_ANSWER_RE = re.compile(r"^-?\d+$")


def encode_char(c: str) -> int:
    """char-byte-offset/1.0: id = (ord(c) % 250) + 2."""
    return (ord(c) % 250) + 2


def encode(s: str) -> list[int]:
    return [encode_char(c) for c in s]


def _decodable_table() -> dict[int, str]:
    table: dict[int, str] = {}
    for c in ALPHABET:
        i = encode_char(c)
        if i in table:
            raise RuntimeError(f"ENCODING_COLLISION: id {i} for {c!r} and {table[i]!r}")
        table[i] = c
    return table


DECODABLE_IDS: dict[int, str] = _decodable_table()


def decode_ids(ids: list[int]) -> str | None:
    """Decode only fully decodable sequences; None if any id is outside the alphabet."""
    try:
        return "".join(DECODABLE_IDS[i] for i in ids)
    except KeyError:
        return None


def roundtrip_ok(s: str) -> bool:
    return decode_ids(encode(s)) == s


def parse_row(row: str) -> tuple[int, str, int, str]:
    """'a op b = c' -> (a, op, b, target). Raises on malformed rows."""
    left, sep, right = row.partition("=")
    if not sep:
        raise ValueError(f"malformed calculator row (no '='): {row!r}")
    parts = left.split()
    if len(parts) != 3:
        raise ValueError(f"malformed calculator row (prompt): {row!r}")
    a_s, op, b_s = parts
    if op not in ("+", "-", "*", "/"):
        raise ValueError(f"malformed calculator row (op): {row!r}")
    return int(a_s), op, int(b_s), right.strip()


def split_prompt_target(row: str) -> tuple[str, str]:
    """Frozen A3 convention: PROMPT keeps '=', TARGET is stripped."""
    if "=" not in row:
        raise ValueError(f"malformed calculator row: {row!r}")
    left, right = row.rsplit("=", 1)
    return left + "=", right.strip()


def normalize_answer(s: str) -> int | None:
    """Frozen A4: strip, cut at newline, strict ^-?\\d+$ → int; else None.

    Documented choice: leading zeros accepted ("007" == 7); commentary, units,
    empty strings, and multi-number text are INCORRECT.
    """
    head = s.strip().split("\n", 1)[0].strip()
    if not _ANSWER_RE.match(head):
        return None
    try:
        return int(head)
    except ValueError:
        return None


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval (clamped). k=0 → LCB exactly 0.0; k=n → UCB exactly 1.0."""
    if n <= 0:
        return (0.0, 0.0)
    if not 0 <= k <= n:
        raise ValueError("k must satisfy 0 <= k <= n")
    z2 = z * z
    denom = 1.0 + z2 / n
    p = k / n
    center = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    lcb, ucb = max(0.0, center - half), min(1.0, center + half)
    if k == 0:  # exact boundary values (float rounding would give 1-eps/eps)
        lcb = 0.0
    if k == n:
        ucb = 1.0
    return (lcb, ucb)


def summarize(predictions: list[str], targets: list[str],
              valid: list[bool] | None = None) -> dict[str, Any]:
    """Exact-match summary over valid rows with Wilson 95% interval."""
    if len(predictions) != len(targets):
        raise ValueError("predictions/targets length mismatch")
    mask = valid if valid is not None else [True] * len(targets)
    pairs = [(p, t) for p, t, v in zip(predictions, targets, mask) if v]
    correct = sum(1 for p, t in pairs if normalize_answer(p) == normalize_answer(t))
    total = len(pairs)
    lcb, ucb = wilson(correct, total)
    return {"correct": correct, "total": total,
            "accuracy": (correct / total) if total else 0.0,
            "wilson_lcb": lcb, "wilson_ucb": ucb}


def sha_predictions(predictions: list[str]) -> str:
    """Order-sensitive identity hash of a prediction vector (reload gate)."""
    return hashlib.sha256(("\n".join(predictions) + "\n").encode("utf-8")).hexdigest()


def heuristic_nulls(rows: list[str], train_rows: list[str]) -> dict[str, list[str]]:
    """Four mechanical nulls (§A9). Computed from data only, never from trained results."""
    parsed = [parse_row(r) for r in rows]
    train_targets = [parse_row(r)[3] for r in train_rows]
    common = Counter(train_targets).most_common(1)[0][0] if train_targets else "0"
    return {
        "always_zero": ["0"] * len(rows),
        "copy_first_operand": [str(a) for a, _, _, _ in parsed],
        "copy_second_operand": [str(b) for _, _, b, _ in parsed],
        "most_common_train_answer": [common] * len(rows),
    }


def strongest_null_accuracy(null_summaries: dict[str, dict[str, Any]]) -> tuple[str, float]:
    """STRONGEST_HEURISTIC_NULL = max accuracy over the four nulls."""
    name, best = max(((k, v["accuracy"]) for k, v in null_summaries.items()),
                     key=lambda kv: kv[1])
    return name, best


def _comm_key(a: int, op: str, b: int) -> tuple:
    if op in ("+", "*"):
        return (op, frozenset((a, b)))
    return (op, a, b)


def split_overlap_report(splits: dict[str, list[str]]) -> dict[str, int]:
    """Exact + commutative-key + triple overlap counts across named splits."""
    sets = {k: set(v) for k, v in splits.items()}
    keys = {k: {_comm_key(*parse_row(r)[:3]) for r in v} for k, v in splits.items()}
    names = sorted(splits)
    out: dict[str, int] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            out[f"exact_{a}_x_{b}"] = len(sets[a] & sets[b])
            out[f"commkey_{a}_x_{b}"] = len(keys[a] & keys[b])
    return out


def _code_hash() -> str:
    h = hashlib.sha256()
    here = Path(__file__).resolve().parent
    for name in ("calculator_data.py", "calculator_eval.py"):
        h.update(Path(here / name).read_bytes())
    return h.hexdigest()


def build_data_receipt(*, out: str | None = None) -> dict[str, Any]:
    """Machine-readable canary data receipt (§A8). Deterministic; no torch."""
    from citadel_tpu import calculator_data as calc

    splits = {s: calc.generate(split=s) for s in ("train", "development", "test")}  # type: ignore[arg-type]
    slices = calc.generalization_slices()
    test_set = set(splits["test"])
    scored_slices = {k: [r for r in v if r not in test_set] for k, v in slices.items()}
    dropped = {k: len(v) - len(scored_slices[k]) for k, v in slices.items()}
    op_dist = {}
    for name, rows in splits.items():
        c: Counter = Counter()
        for r in rows:
            c[parse_row(r)[1]] += 1
        op_dist[name] = dict(sorted(c.items()))
    receipt: dict[str, Any] = {
        "schema": "citadel-calculator-data-receipt/v1",
        "generator_version": calc.GENERATOR_VERSION,
        "generator_code_sha256": _code_hash(),
        "counts": {k: len(v) for k, v in splits.items()},
        "seeds": {k: calc.SPLITS[k]["seed"] for k in splits},
        "ranges": {k: [calc.SPLITS[k]["lo"], calc.SPLITS[k]["hi"]] for k in splits},
        "op_distribution": op_dist,
        "split_sha256": {k: hashlib.sha256(("\n".join(v) + "\n").encode()).hexdigest()
                         for k, v in splits.items()},
        "overlap": split_overlap_report(splits),
        "generalization_slices": {k: len(v) for k, v in slices.items()},
        "slice_test_duplicates_dropped": dropped,
        "scored_slice_counts": {k: len(v) for k, v in scored_slices.items()},
        "encoding_version": ENCODING_VERSION,
    }
    if out is not None:
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    return receipt


def generate(rows: list[str], model: Any, xb: Any, *, device: Any, torch_mod: Any,
             batch: int = EVAL_BATCH, length: int = EVAL_LENGTH) -> list[dict[str, Any]]:
    """Static-shape greedy generation (§A5). Host/device split mirrors T0.

    Fixed [batch, length] every step; per-row cursors advance via index scatter
    (one compiled graph). Per-step host sync (.tolist) is intentional and
    eval-only — never inside a training update. Caller ensures the Cymek
    runtime is importable (runtime_bootstrap) before calling.
    """
    torch = torch_mod
    from v5_model.core import packed_layout

    prompts = [split_prompt_target(r)[0] for r in rows]
    targets = [split_prompt_target(r)[1] for r in rows]
    encoded = [encode(p) for p in prompts]
    for p, ids in zip(prompts, encoded):
        if any(i in (PAD_ID, UNK_ID, BOS_ID, EOS_ID) for i in ids):
            raise ValueError(f"prompt encodes to reserved id: {p!r}")
        if len(ids) > length - MAX_ANSWER_TOKENS:
            raise ValueError(f"prompt too long for fixed buffer: {p!r}")
    records: list[dict[str, Any]] = []
    arange_b = torch.arange(batch, device=device)
    for start in range(0, len(rows), batch):
        chunk = list(range(start, min(start + batch, len(rows))))
        nb = len(chunk)
        buf = torch.full((batch, length), PAD_ID, dtype=torch.long)
        for j, i in enumerate(chunk):
            buf[j, : len(encoded[i])] = torch.tensor(encoded[i], dtype=torch.long)
        plen = torch.tensor([len(encoded[i]) for i in chunk] + [0] * (batch - nb),
                            dtype=torch.long)
        seg = torch.zeros((batch, length), dtype=torch.long)
        positions, mask = packed_layout(seg, torch_module=torch)  # host tensors (audit A8)
        buf_d, pos_d, mask_d = buf.to(device), positions.to(device), mask.to(device)
        cursor = plen.clone()
        done = [False] * batch
        reasons: list[str | None] = [None] * batch
        answers: list[list[str]] = [[] for _ in range(batch)]
        steps = [0] * batch
        for _ in range(MAX_ANSWER_TOKENS):
            logits = model(buf_d, pos_d, mask_d)
            xb.mark_step()
            nxt = logits[arange_b, (cursor - 1).clamp(min=0).to(device)].argmax(-1)
            ids = nxt.to("cpu").tolist()
            write = []
            for j in range(batch):
                real = j < nb
                if done[j] or not real:
                    write.append(PAD_ID)
                    continue
                steps[j] += 1
                v = ids[j]
                if v == PAD_ID:
                    done[j], reasons[j] = True, "PAD"
                    write.append(PAD_ID)
                elif v == EOS_ID:
                    done[j], reasons[j] = True, "EOS"
                    write.append(PAD_ID)
                elif v == NEWLINE_ID:
                    done[j], reasons[j] = True, "NEWLINE"
                    write.append(PAD_ID)
                elif v not in DECODABLE_IDS:
                    done[j], reasons[j] = True, "NON_ALPHABET"
                    write.append(PAD_ID)
                else:
                    answers[j].append(DECODABLE_IDS[v])
                    write.append(v)
                    if len(answers[j]) >= MAX_ANSWER_TOKENS:
                        done[j], reasons[j] = True, "MAX_TOKENS"
            buf_d.scatter_(1, cursor.unsqueeze(1).to(device),
                           torch.tensor(write, dtype=torch.long, device=device).unsqueeze(1))
            cursor = cursor + 1
            if all(done[j] or j >= nb for j in range(batch)):
                break
        for k, i in enumerate(chunk):
            pred = "".join(answers[k])
            records.append({
                "prompt": prompts[i], "target": targets[i], "prediction": pred,
                "correct": normalize_answer(pred) == normalize_answer(targets[i]),
                "stop_reason": reasons[k] if reasons[k] is not None else "MAX_TOKENS",
                "generated_token_count": steps[k], "valid": True,
            })
    return records


def selftest() -> None:
    """Torch-free validation suite (used by unit tests + calculator_preflight)."""
    for c in ALPHABET:
        assert roundtrip_ok(c), f"round-trip failed for {c!r}"
    assert roundtrip_ok("72 / 8 = 9") and roundtrip_ok("18 - 7 = 11")
    assert normalize_answer("12") == 12
    assert normalize_answer("  -39 \n") == -39
    assert normalize_answer("007") == 7
    assert normalize_answer("") is None
    assert normalize_answer("The answer is 12") is None
    assert normalize_answer("12 because") is None
    assert normalize_answer("3+4") is None
    assert normalize_answer("1 2") is None
    lcb, ucb = wilson(0, 10)
    assert lcb == 0.0 and 0.0 < ucb <= 1.0
    lcb, ucb = wilson(10, 10)
    assert ucb == 1.0 and 0.0 <= lcb < 1.0
    lcb, ucb = wilson(50, 100)
    assert lcb < 0.5 < ucb and (ucb - lcb) < 0.25
    _, ucb10 = wilson(6, 10)
    _, ucb100 = wilson(60, 100)
    assert (ucb100 - 0.6) < (ucb10 - 0.6)  # interval narrows with n
    assert wilson(0, 0) == (0.0, 0.0)
    try:
        wilson(11, 10)
        raise SystemExit("wilson accepted k>n")
    except ValueError:
        pass
    gold = ["7", "11", "48"]
    s = summarize(gold, gold)
    assert s["correct"] == 3 and s["accuracy"] == 1.0 and s["wilson_lcb"] > 0.3
    s = summarize(["0", "0", "0"], gold)
    assert s["correct"] == 0 and s["accuracy"] == 0.0 and s["wilson_lcb"] == 0.0
    s = summarize(["7", "0", "48", "0"], ["7", "0", "48", "9"], valid=[True, True, False, True])
    assert (s["correct"], s["total"]) == (2, 3)
    assert sha_predictions(gold) == sha_predictions(list(gold))
    assert sha_predictions(gold) != sha_predictions(["11", "7", "48"])
    assert parse_row("72 / 8 = 9") == (72, "/", 8, "9")
    assert split_prompt_target("18 - 7 = 11") == ("18 - 7 =", "11")
    assert _comm_key(3, "+", 7) == _comm_key(7, "+", 3)
    assert _comm_key(3, "-", 7) != _comm_key(7, "-", 3)
    rows = ["3 + 4 = 7", "18 - 7 = 11", "6 * 8 = 48"]
    nulls = heuristic_nulls(rows, rows)
    assert set(nulls) == {"always_zero", "copy_first_operand", "copy_second_operand",
                          "most_common_train_answer"}
    assert nulls["copy_first_operand"] == ["3", "18", "6"]
    assert nulls["copy_second_operand"] == ["4", "7", "8"]
    acc = {k: summarize(v, [parse_row(r)[3] for r in rows])["accuracy"] for k, v in nulls.items()}
    name, best = strongest_null_accuracy({k: {"accuracy": v} for k, v in acc.items()})
    assert best == max(acc.values()) and name in acc


__all__ = [
    "ALPHABET",
    "DECODABLE_IDS",
    "ENCODING_VERSION",
    "EVAL_BATCH",
    "EVAL_LENGTH",
    "MAX_ANSWER_TOKENS",
    "NEWLINE_ID",
    "build_data_receipt",
    "decode_ids",
    "encode",
    "encode_char",
    "generate",
    "heuristic_nulls",
    "normalize_answer",
    "parse_row",
    "roundtrip_ok",
    "selftest",
    "sha_predictions",
    "split_overlap_report",
    "split_prompt_target",
    "strongest_null_accuracy",
    "summarize",
    "wilson",
]
