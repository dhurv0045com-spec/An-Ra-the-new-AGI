"""Rich synthetic arithmetic corpus (T1C). Deterministic indexed generation.

`row_at(split, i)` is a pure function of (version, split, i): O(1) memory, no
stored lists, resumable, hashable. 5M-row TRAIN streams in chunks; eval slices
materialize fully (thousands of rows). No downloads, no git artifacts (manifest
only). Splits are structurally disjoint by operand range + template set; the
manifest mechanically verifies eval slices + a train audit sample.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


GENERATOR_VERSION = "arith-canary/1.0"
MASK64 = (1 << 64) - 1
_CHUNK = 50_000
AUDIT_SAMPLE_EVERY = 50

OPS = ("+", "-", "*", "/")
SYM = {"+": "+", "-": "-", "*": "*", "/": "/"}
TEMPLATES = ("canon", "compact", "arrow", "words")

SPLITS: dict[str, dict[str, Any]] = {
    # name: n, salt, operand range, mult cap, div_b range, templates
    "train":          {"n": 5_000_000, "salt": 201, "lo": 0,      "hi": 999,
                       "mult_hi": 99, "div_b_lo": 1, "div_b_hi": 12,
                       "templates": ("canon", "compact", "arrow")},
    "dev":            {"n": 150_000,   "salt": 202, "lo": 1000,   "hi": 1999,
                       "mult_hi": 1999, "div_b_lo": 1, "div_b_hi": 12,
                       "templates": ("canon", "compact", "words")},
    "test_core":      {"n": 1000,      "salt": 203, "lo": 2000,   "hi": 2999,
                       "mult_hi": 2999, "div_b_lo": 1, "div_b_hi": 12,
                       "templates": ("canon",)},
    "test_template":  {"n": 500,       "salt": 204, "lo": 2000,   "hi": 2999,
                       "mult_hi": 2999, "div_b_lo": 1, "div_b_hi": 12,
                       "templates": ("words",)},
    "test_range":     {"n": 500,       "salt": 205, "lo": 100000, "hi": 999999,
                       "mult_hi": 999999, "div_b_lo": 1, "div_b_hi": 99,
                       "templates": ("canon",)},
    "test_composition":{"n": 500,      "salt": 206, "lo": 100,    "hi": 999,
                       "mult_hi": 999, "div_b_lo": 1, "div_b_hi": 12,
                       "templates": ("compact", "arrow"),
                       "mult_only": True},
}


def _splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return (z ^ (z >> 31)) & MASK64


def _draw(salt: int, i: int, stream: int) -> int:
    return _splitmix64((salt << 32) ^ (i * 2654435761 + stream * 40503))


def _carries(a: int, b: int) -> int:
    n, c = 0, 0
    while a > 0 or b > 0:
        s = (a % 10) + (b % 10) + c
        c, n = (1, n + 1) if s >= 10 else (0, n)
        a, b = a // 10, b // 10
    return n


def _borrows(a: int, b: int) -> int:
    n, c, x, y = 0, 0, abs(a), abs(b)
    while x > 0 or y > 0:
        d = (x % 10) - c - (y % 10)
        if d < 0:
            d, c, n = d + 10, 1, n + 1
        else:
            c = 0
        x, y = x // 10, y // 10
    return n


def render(a: int, op: str, b: int, c: int, template: str) -> str:
    s = SYM[op]
    if template == "canon":
        return f"{a} {s} {b} = {c}"
    if template == "compact":
        return f"{a}{s}{b}={c}"
    if template == "arrow":
        return f"{a} {s} {b} -> {c}"
    if template == "words":
        if op == "+":
            return f"add {a} and {b} = {c}"
        if op == "-":
            return f"subtract {b} from {a} = {c}"
        if op == "*":
            return f"multiply {a} by {b} = {c}"
        return f"divide {a} by {b} = {c}"
    raise ValueError(f"unknown template {template!r}")


def row_at(split: str, i: int) -> tuple[str, dict[str, Any]]:
    """Pure deterministic row i of split. Raises on out-of-range i."""
    cfg = SPLITS[split]
    if not 0 <= i < cfg["n"]:
        raise ValueError(f"index {i} out of range for split {split}")
    salt = cfg["salt"]
    ops = OPS if not cfg.get("mult_only") else ("*",)
    op = ops[_draw(salt, i, 1) % len(ops)]
    lo, hi = cfg["lo"], cfg["hi"]
    span = hi - lo + 1
    if op == "/":
        b = cfg["div_b_lo"] + _draw(salt, i, 2) % (cfg["div_b_hi"] - cfg["div_b_lo"] + 1)
        q = lo + _draw(salt, i, 3) % span
        a, c = b * q, q
    elif op == "*" and cfg.get("mult_hi") is not None:
        mhi = cfg["mult_hi"]
        a = lo + _draw(salt, i, 2) % span
        b = cfg["lo"] + _draw(salt, i, 3) % span
        a, b = min(a, mhi), min(b, mhi)
        c = a * b
    else:
        a = lo + _draw(salt, i, 2) % span
        b = lo + _draw(salt, i, 3) % span
        c = a + b if op == "+" else a - b
    tpls = cfg["templates"]
    template = tpls[_draw(salt, i, 4) % len(tpls)]
    text = render(a, op, b, c, template)
    meta = {"op": op, "a": a, "b": b, "c": c, "template": template,
            "digits": (len(str(abs(a))), len(str(abs(b))), len(str(abs(c)))),
            "carries": _carries(a, b) if op == "+" else 0,
            "borrows": _borrows(a, b) if op == "-" else 0}
    return text, meta


def split_prompt_target(row: str) -> tuple[str, str]:
    """Prompt/target split for canon/compact/words (=) and arrow (->) rows."""
    for delim in ("->", "="):
        if delim in row:
            left, right = row.rsplit(delim, 1)
            return left + delim, right.strip()
    raise ValueError(f"malformed arithmetic row: {row!r}")


def _eval_rows(split: str) -> list[str]:
    return [row_at(split, i)[0] for i in range(SPLITS[split]["n"])]


def _stream_hash(split: str, n: int) -> tuple[str, int, int]:
    """Incremental sha256 + byte count over the first n rows (chunked)."""
    h = hashlib.sha256()
    total = 0
    buf: list[str] = []
    for i in range(n):
        buf.append(row_at(split, i)[0])
        if len(buf) >= _CHUNK:
            chunk = ("\n".join(buf) + "\n").encode("utf-8")
            h.update(chunk)
            total += len(chunk)
            buf = []
    if buf:
        chunk = ("\n".join(buf) + "\n").encode("utf-8")
        h.update(chunk)
        total += len(chunk)
    return h.hexdigest(), total, n


def parse_arith(text: str) -> tuple[int, str, int, int]:
    """Parse any template row -> (a, op, b, c); asserts the arithmetic holds."""
    if "->" in text:
        left, right = text.rsplit("->", 1)
    elif "=" in text:
        left, right = text.rsplit("=", 1)
    else:
        raise ValueError(f"malformed arithmetic row: {text!r}")
    c = int(right.strip())
    core = left.strip()
    nospace = core.replace(" ", "")
    import re as _re

    m = _re.fullmatch(r"(-?\d+)([+\-*/])(-?\d+)", nospace)
    if m:
        a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
    elif core.startswith("add "):
        nums = [int(p) for p in core.split() if p.lstrip("-").isdigit()]
        a, op, b = nums[0], "+", nums[1]
    elif core.startswith("subtract "):
        nums = [int(p) for p in core.split() if p.lstrip("-").isdigit()]
        a, op, b = nums[1], "-", nums[0]  # "subtract B from A"
    elif core.startswith("multiply "):
        nums = [int(p) for p in core.split() if p.lstrip("-").isdigit()]
        a, op, b = nums[0], "*", nums[1]
    elif core.startswith("divide "):
        nums = [int(p) for p in core.split() if p.lstrip("-").isdigit()]
        a, op, b = nums[0], "/", nums[1]
    else:
        raise ValueError(f"cannot parse arithmetic row: {text!r}")
    expect = a + b if op == "+" else a - b if op == "-" else a * b if op == "*" else a // b
    if expect != c or (op == "/" and a % b):
        raise ValueError(f"arithmetic mismatch in row: {text!r}")
    return a, op, b, c


def _triple_key_of(text: str) -> tuple:
    prompt, target = split_prompt_target(text)
    return (prompt, target)


def _comm_key_of(text: str) -> tuple:
    a, op, b, _ = parse_arith(text)
    if op in ("+", "*"):
        return (op, frozenset((a, b)))
    return (op, a, b)


def build_manifest(*, out: str | None = None, audit_sample_n: int = 100_000) -> dict[str, Any]:
    """Stream the corpus once: hashes, bytes, audit sample, leakage checks."""
    from citadel_tpu import calculator_eval as cev

    manifest: dict[str, Any] = {
        "schema": "citadel-arith-manifest/v1",
        "generator_version": GENERATOR_VERSION,
        "splits": {},
    }
    eval_text: dict[str, list[str]] = {}
    for name in ("dev", "test_core", "test_template", "test_range", "test_composition"):
        rows = _eval_rows(name)
        eval_text[name] = rows
        blob = ("\n".join(rows) + "\n").encode("utf-8")
        manifest["splits"][name] = {
            "n": len(rows), "bytes": len(blob),
            "sha256": hashlib.sha256(blob).hexdigest(),
        }
    train_hash, train_bytes, train_n = _stream_hash("train", SPLITS["train"]["n"])
    manifest["splits"]["train"] = {"n": train_n, "bytes": train_bytes, "sha256": train_hash}
    # audit sample for leakage (deterministic stride over train space)
    step = max(1, SPLITS["train"]["n"] // audit_sample_n)
    sample = [row_at("train", i)[0] for i in range(0, SPLITS["train"]["n"], step)]
    manifest["audit_sample"] = {"stride": step, "n": len(sample),
                                "sha256": hashlib.sha256(
                                    ("\n".join(sample) + "\n").encode()).hexdigest()}
    # leakage: exact + commutative/template keys across eval slices and audit sample
    sample_set, sample_keys = set(sample), {_triple_key_of(r) for r in sample}
    sample_comm = set()
    for r in sample:
        try:
            sample_comm.add(_comm_key_of(r))
        except ValueError:
            pass
    leak: dict[str, int] = {}
    eval_sets = {k: set(v) for k, v in eval_text.items()}
    eval_comms = {}
    for k, v in eval_text.items():
        comm = set()
        for r in v:
            try:
                comm.add(_comm_key_of(r))
            except ValueError:
                pass
        eval_comms[k] = comm
    names = sorted(eval_text)
    for x in range(len(names)):
        for y in range(x + 1, len(names)):
            a, b = names[x], names[y]
            leak[f"exact_{a}_x_{b}"] = len(eval_sets[a] & eval_sets[b])
            leak[f"commkey_{a}_x_{b}"] = len(eval_comms[a] & eval_comms[b])
        leak[f"exact_train-sample_x_{names[x]}"] = len(sample_set & eval_sets[names[x]])
        leak[f"commkey_train-sample_x_{names[x]}"] = len(sample_comm & eval_comms[names[x]])
    manifest["leakage"] = leak
    manifest["total_bytes"] = sum(s["bytes"] for s in manifest["splits"].values())
    if out is not None:
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


__all__ = [
    "GENERATOR_VERSION",
    "OPS",
    "SPLITS",
    "TEMPLATES",
    "build_manifest",
    "render",
    "row_at",
    "split_prompt_target",
]
