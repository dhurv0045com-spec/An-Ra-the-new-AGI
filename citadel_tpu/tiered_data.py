"""Tiered difficulty-ladder corpus + micro-teacher tasks (T1D). Indexed, O(1).

`tier_row(tier, split, i)` is pure in (version, tier, split, i). Tiers carry
frozen difficulty semantics (T0 trivial … T4 multi-digit); splits are
independent index spaces per (tier, split) with different salts. T0/T1 spaces
are inherently tiny — their TEST slices are LABELED in-distribution
memorization probes (overlap reported, never gated); T2+ slices are
structurally held-out with zero-leak gates. Teacher rows (digit/subproblem
supervision) are train-only and never enter eval slices.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


GENERATOR_VERSION = "tiered-arith/1.0"
MASK64 = (1 << 64) - 1
_CHUNK = 50_000

# Frozen per-split operand bands: TEST rows come from ranges TRAIN never
# consumes, so T2+ held-out claims are structural. Tiers 0/1 have no bands
# (spaces too small) — their slices are labeled memorization probes.
BANDS: dict[int, dict[str, dict[str, int]]] = {
    0: {"train": {"x_lo": 0, "x_hi": 5999},
        "dev": {"x_lo": 6000, "x_hi": 7999},
        "test": {"x_lo": 8000, "x_hi": 9999}},
    1: {},
    2: {"train": {"ha_lo": 1, "ha_hi": 5,
                  "mult_a_lo": 10, "mult_a_hi": 59, "div_q_lo": 10, "div_q_hi": 399},
        "dev": {"ha_lo": 6, "ha_hi": 7,
                "mult_a_lo": 60, "mult_a_hi": 79, "div_q_lo": 400, "div_q_hi": 699},
        "test": {"ha_lo": 8, "ha_hi": 9,
                 "mult_a_lo": 80, "mult_a_hi": 99, "div_q_lo": 700, "div_q_hi": 999}},
    3: {"train": {"lo": 10, "hi": 59, "mult_a_lo": 10, "mult_a_hi": 59,
                  "div_q_lo": 10, "div_q_hi": 59},
        "dev": {"lo": 60, "hi": 79, "mult_a_lo": 60, "mult_a_hi": 79,
                "div_q_lo": 60, "div_q_hi": 79},
        "test": {"lo": 80, "hi": 99, "mult_a_lo": 80, "mult_a_hi": 99,
                 "div_q_lo": 80, "div_q_hi": 99}},
    4: {"train": {"lo": 100, "hi": 5999, "mult_a_lo": 100, "mult_a_hi": 599,
                  "div_q_lo": 100, "div_q_hi": 3999},
        "dev": {"lo": 6000, "hi": 7999, "mult_a_lo": 600, "mult_a_hi": 799,
                "div_q_lo": 4000, "div_q_hi": 6999},
        "test": {"lo": 8000, "hi": 9999, "mult_a_lo": 800, "mult_a_hi": 999,
                 "div_q_lo": 7000, "div_q_hi": 9999}},
}

# Frozen per-tier TRAIN index-space sizes.
TRAIN_N = {0: 96_000, 1: 20_000, 2: 150_000, 3: 150_000, 4: 6_000_000}
EVAL_DEV_N = 200
EVAL_TEST_N = 500
AUDIT_PER_TIER = 20_000

TEMPLATES = ("canon", "compact", "arrow", "words")

# tier: ops config. lo/hi = operand band; mult/div caps keep answers in L.
TIERS: dict[int, dict[str, Any]] = {
    0: {"forms": ("add0", "sub0", "mul1", "div1"), "lo": 0, "hi": 9999,
        "templates": ("canon", "compact", "arrow", "words")},
    1: {"lo": 0, "hi": 9,     "div_b_lo": 1, "div_b_hi": 9, "div_q_lo": 0, "div_q_hi": 9,
        "templates": ("canon", "compact", "arrow", "words")},
    2: {"lo": 0, "hi": 999, "easy": True, "mult_a_lo": 10, "mult_a_hi": 99,
        "mult_b_lo": 2, "mult_b_hi": 9, "div_b_lo": 2, "div_b_hi": 9,
        "div_q_lo": 10, "div_q_hi": 999,
        "templates": ("canon", "compact", "arrow", "words")},
    3: {"lo": 10, "hi": 99, "div_b_lo": 10, "div_b_hi": 99,
        "div_q_lo": 10, "div_q_hi": 99,
        "templates": ("canon", "compact", "arrow", "words")},
    4: {"lo": 100, "hi": 9999, "div_b_lo": 100, "div_b_hi": 999,
        "div_q_lo": 100, "div_q_hi": 9999,
        "templates": ("canon", "compact", "arrow", "words")},
}

# Frozen curriculum: phase boundaries (% of budget) -> tier weights.
CURRICULUM: tuple[tuple[float, dict[int, float]], ...] = (
    (0.00, {0: 0.50, 1: 0.50}),
    (0.15, {1: 0.50, 2: 0.50}),
    (0.35, {2: 0.50, 3: 0.50}),
    (0.60, {1: 0.20, 2: 0.25, 3: 0.30, 4: 0.25}),
)
UNIFORM_MIXTURE: dict[int, float] = {0: 0.10, 1: 0.15, 2: 0.25, 3: 0.25, 4: 0.25}

TEACHER_RATIO = 0.40  # arm C: 6 ordinary + 4 teacher rows per 10 (row split)


def _splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return (z ^ (z >> 31)) & MASK64


def _draw(salt: int, i: int, stream: int) -> int:
    return _splitmix64((salt << 32) ^ (i * 2654435761 + stream * 40503))


def _nodigit_pair_2d(r1: int, r2: int, r3: int, kind: str) -> tuple[int, int]:
    """Two-digit no-carry (add) / no-borrow (sub) pair with second operand >= 2.

    Tens digits nonzero-capable (either may be 0); units drawn so that
    u2 >= 2 always: add draws u1 in 0..7 then u2 in 2..9-u1; sub draws
    u1 in 2..9 then u2 in 2..u1. This keeps T2+ rows out of the T0 trivial
    forms (second operand 0/1) while preserving digitwise constraints.
    """
    if kind == "add":
        t1 = 1 + r1 % 9
        t2 = r2 % (10 - t1)
        u1 = r3 % 8
        u2 = 2 + _draw(r3, 7, 1) % (8 - u1)
    else:
        t1 = 1 + r1 % 9
        t2 = r2 % (t1 + 1)
        u1 = 2 + r3 % 8
        u2 = 2 + _draw(r3, 7, 1) % (u1 - 1)
    return t1 * 10 + u1, t2 * 10 + u2


def _nodigit_pair(r1: int, r2: int, nd: int, kind: str) -> tuple[int, int]:
    """Deterministic no-carry (add) / no-borrow (sub) operand pair, nd digits.

    Per digit: add constrains b_d <= 9 - a_d; sub constrains b_d <= a_d.
    Pure function of the two draws; uniform over the valid subspace.
    """
    a_digits, b_digits = [], []
    v1, v2 = r1, r2
    for _ in range(nd):
        da, v1 = v1 % 10, v1 // 10
        if kind == "add":
            db = v2 % (10 - da)
        else:
            db = v2 % (da + 1)
        v2 //= 10
        a_digits.append(da)
        b_digits.append(db)
    a = sum(d * (10 ** k) for k, d in enumerate(a_digits))
    b = sum(d * (10 ** k) for k, d in enumerate(b_digits))
    return a, b


def _render(a: int, op: str, b: int, c: int, template: str) -> str:
    s = {"+": "+", "-": "-", "*": "*", "/": "/"}[op]
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


def tier_row(tier: int, split: str, i: int) -> tuple[str, dict[str, Any]]:
    """Pure deterministic tier row with frozen per-split operand bands.

    Bands (not salts) isolate splits: TEST rows come from operand ranges TRAIN
    never consumes, so T2+ held-out claims are structural, not probabilistic.
    Tiers 0/1 cannot be band-isolated (tiny spaces) — their slices are labeled
    memorization probes and excluded from the zero-leak gate (see
    leakage_verdict). Templates are shared across splits (covariate, recorded).
    """
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier}")
    if split not in ("train", "dev", "test"):
        raise ValueError(f"unknown split {split}")
    cfg = dict(TIERS[tier])
    cfg.update(BANDS.get(tier, {}).get(split, {}))
    salt = 1000 + tier * 10 + {"train": 1, "dev": 2, "test": 3}[split]
    tpls = cfg.get("templates", TEMPLATES)
    template = tpls[_draw(salt, i, 1) % len(tpls)]
    if tier == 0:
        forms = cfg["forms"]
        form = forms[_draw(salt, i, 2) % len(forms)]
        x = cfg["x_lo"] + _draw(salt, i, 3) % (cfg["x_hi"] - cfg["x_lo"] + 1)
        if form == "add0":
            a, op, b, c = x, "+", 0, x
        elif form == "sub0":
            a, op, b, c = x, "-", 0, x
        elif form == "mul1":
            a, op, b, c = x, "*", 1, x
        else:
            a, op, b, c = x, "/", 1, x
        meta = {"op": op, "a": a, "b": b, "c": c, "template": template,
                "tier": 0, "digits": (len(str(a)), 1, len(str(c))),
                "carries": 0, "borrows": 0}
        return _render(a, op, b, c, template), meta
    ops = ("+", "-", "*", "/")
    op = ops[_draw(salt, i, 2) % 4]
    lo, hi = cfg["lo"], cfg["hi"]
    span = hi - lo + 1
    if op == "/":
        b = cfg["div_b_lo"] + _draw(salt, i, 3) % (cfg["div_b_hi"] - cfg["div_b_lo"] + 1)
        q = cfg["div_q_lo"] + _draw(salt, i, 4) % (cfg["div_q_hi"] - cfg["div_q_lo"] + 1)
        a, c = b * q, q
    elif op == "*":
        mhi = cfg.get("mult_a_hi", cfg.get("mult_hi", hi))
        mlo = cfg.get("mult_a_lo", lo)
        a = mlo + _draw(salt, i, 3) % (mhi - mlo + 1)
        b = cfg.get("mult_b_lo", lo) + _draw(salt, i, 4) % (
            cfg.get("mult_b_hi", hi) - cfg.get("mult_b_lo", lo) + 1)
        c = a * b
    elif tier == 2 and op in ("+", "-"):
        # Band-isolated hundreds digit + 2-digit no-carry/borrow pair with
        # second operand >= 2 (never a T0 trivial form, never T1 range).
        ha = cfg["ha_lo"] + _draw(salt, i, 5) % (cfg["ha_hi"] - cfg["ha_lo"] + 1)
        x, y = _nodigit_pair_2d(_draw(salt, i, 3), _draw(salt, i, 4),
                                _draw(salt, i, 6), "add" if op == "+" else "sub")
        a, b = ha * 100 + x, y
        c = a + b if op == "+" else a - b
    elif cfg.get("easy"):
        nd = max(1, len(str(hi)))  # T2 lo=0 so nd-digit values stay in band
        a, b = _nodigit_pair(_draw(salt, i, 3), _draw(salt, i, 4),
                             nd, "add" if op == "+" else "sub")
        c = a + b if op == "+" else a - b
    else:
        a = lo + _draw(salt, i, 3) % span
        b = lo + _draw(salt, i, 4) % span
        c = a + b if op == "+" else a - b
    meta = {"op": op, "a": a, "b": b, "c": c, "template": template, "tier": tier,
            "digits": (len(str(abs(a))), len(str(abs(b))), len(str(abs(c)))),
            "carries": _carries(a, b) if op == "+" else 0,
            "borrows": _borrows(a, b) if op == "-" else 0}
    return _render(a, op, b, c, template), meta


def teacher_row(kind: str, i: int) -> tuple[str, dict[str, Any]]:
    """Micro-teacher supervision rows (train-only, never evaluated)."""
    if kind not in ("digadd", "digsub", "singlemul", "divmicro"):
        raise ValueError(f"unknown teacher kind {kind!r}")
    salt = 5000 + {"digadd": 1, "digsub": 2, "singlemul": 3, "divmicro": 4}[kind]
    if kind == "digadd":
        a = _draw(salt, i, 1) % 10
        b = _draw(salt, i, 2) % 10
        c0 = _draw(salt, i, 3) % 2
        d, c1 = (a + b + c0) % 10, (a + b + c0) // 10
        text = f"digadd {a} {b} carry{c0} = digit{d} carry{c1}"
        meta = {"kind": kind, "a": a, "b": b, "carry_in": c0, "digit": d, "carry_out": c1}
    elif kind == "digsub":
        a = _draw(salt, i, 1) % 10
        b = _draw(salt, i, 2) % 10
        b0 = _draw(salt, i, 3) % 2
        diff = a - b - b0
        d, b1 = diff % 10, 1 if diff < 0 else 0
        text = f"digsub {a} {b} borrow{b0} = digit{d} borrow{b1}"
        meta = {"kind": kind, "a": a, "b": b, "borrow_in": b0, "digit": d, "borrow_out": b1}
    elif kind == "singlemul":
        a = 2 + _draw(salt, i, 1) % 8
        b = 2 + _draw(salt, i, 2) % 8
        text = f"singlemul {a} {b} = {a * b}"
        meta = {"kind": kind, "a": a, "b": b, "c": a * b}
    elif kind == "divmicro":
        b = 2 + _draw(salt, i, 1) % 11
        q = _draw(salt, i, 2) % 13
        text = f"divmicro {b * q} {b} = {q}"
        meta = {"kind": kind, "a": b * q, "b": b, "c": q}
    else:
        raise ValueError(f"unknown teacher kind {kind!r}")
    meta["template"] = "teacher"
    return text, meta


def curriculum_tier(frac: float, draw: int) -> int:
    """Frozen schedule: training fraction + deterministic draw -> tier.

    Draws are hashed before bucketing so SEQUENTIAL draws (the production
    feeder counter) sample the mixture uniformly instead of sweeping it.
    """
    active = CURRICULUM[0][1]
    for bound, weights in CURRICULUM:
        if frac >= bound:
            active = weights
    total = sum(active.values())
    x = (_splitmix64(draw & MASK64) / 2**64) * total
    acc = 0.0
    for tier in sorted(active):
        acc += active[tier]
        if x < acc:
            return tier
    return sorted(active)[-1]


def uniform_tier(draw: int) -> int:
    total = sum(UNIFORM_MIXTURE.values())
    x = (_splitmix64(draw & MASK64) / 2**64) * total
    acc = 0.0
    for tier in sorted(UNIFORM_MIXTURE):
        acc += UNIFORM_MIXTURE[tier]
        if x < acc:
            return tier
    return sorted(UNIFORM_MIXTURE)[-1]


def tier_index(tier: int, split: str, j: int) -> str:
    """j-th row actually consumed from a tier/split (O(1), no lists)."""
    return tier_row(tier, split, j)[0]


def _stream_rows(tier: int, split: str, n: int):
    for i in range(n):
        yield tier_row(tier, split, i)[0]


def _hash_stream(rows_iter, chunk_rows: int = _CHUNK) -> tuple[str, int, int]:
    h = hashlib.sha256()
    total, count, buf = 0, 0, []
    for r in rows_iter:
        buf.append(r)
        count += 1
        if len(buf) >= chunk_rows:
            chunk = ("\n".join(buf) + "\n").encode("utf-8")
            h.update(chunk)
            total += len(chunk)
            buf = []
    if buf:
        chunk = ("\n".join(buf) + "\n").encode("utf-8")
        h.update(chunk)
        total += len(chunk)
    return h.hexdigest(), total, count


def leakage_verdict(leak: dict[str, int]) -> tuple[dict[str, int], dict[str, int]]:
    """Split a leakage dict into (fatal, reported).

    Fatal: any nonzero overlap where every tier named in the key is >= 2
    (structurally held-out by operand bands). Reported (labeled memorization
    probes, never gated): pairs involving tier 0/1 on either side, plus the
    designed T1C-style transfer pair if present. Keys name tiers as _t<N>.
    """
    import re as _re

    fatal: dict[str, int] = {}
    reported: dict[str, int] = {}
    for key, count in leak.items():
        tiers = [int(x) for x in _re.findall(r"_t(\d)", key)]
        if not tiers or min(tiers) <= 1:
            reported[key] = count
        elif count != 0:
            fatal[key] = count
    return fatal, reported


def eval_pair_leakage() -> dict[str, int]:
    """Fast leakage over materialized eval slices only (preflight gate)."""
    eval_text: dict[str, list[str]] = {}
    for tier in range(5):
        for split, sn, tag in (("dev", EVAL_DEV_N, "dev"), ("test", EVAL_TEST_N, "test")):
            eval_text[f"{tag}_t{tier}"] = [tier_row(tier, split, j)[0] for j in range(sn)]
    leak: dict[str, int] = {}
    names = sorted(eval_text)
    for x in range(len(names)):
        for y in range(x + 1, len(names)):
            a, b = names[x], names[y]
            leak[f"exact_{a}_x_{b}"] = len(set(eval_text[a]) & set(eval_text[b]))
    return leak


def build_manifest(*, out: str | None = None) -> dict[str, Any]:
    """Stream the tiered corpus once: hashes, leakage gates, dup rates, max len."""
    manifest: dict[str, Any] = {
        "schema": "citadel-tiered-manifest/v1",
        "generator_version": GENERATOR_VERSION,
        "tiers": {},
    }
    eval_text: dict[str, list[str]] = {}
    max_len = 0
    for tier in range(5):
        n = TRAIN_N[tier]
        digest, nbytes, _ = _hash_stream(_stream_rows(tier, "train", n))
        manifest["tiers"][str(tier)] = {"train_n": n, "train_bytes": nbytes,
                                        "train_sha256": digest}
        for split, sn, tag in (("dev", EVAL_DEV_N, "dev"), ("test", EVAL_TEST_N, "test")):
            rows = [tier_row(tier, split, j)[0] for j in range(sn)]
            eval_text[f"{tag}_t{tier}"] = rows
            max_len = max(max_len, max(len(r) for r in rows))
            blob = ("\n".join(rows) + "\n").encode("utf-8")
            manifest["tiers"][str(tier)][f"{tag}_sha256"] = hashlib.sha256(blob).hexdigest()
    # audit samples per tier (stride) with exact-duplicate rates
    audit: dict[str, list[str]] = {}
    for tier in range(5):
        n = TRAIN_N[tier]
        step = max(1, n // AUDIT_PER_TIER)
        sample = [tier_row(tier, "train", i)[0] for i in range(0, n, step)]
        audit[f"t{tier}"] = sample
        max_len = max(max_len, max(len(r) for r in sample))
        manifest["tiers"][str(tier)]["audit"] = {
            "stride": step, "n": len(sample),
            "sha256": hashlib.sha256(("\n".join(sample) + "\n").encode()).hexdigest(),
            "exact_duplicate_rate": 1.0 - len(set(sample)) / len(sample)}
    manifest["max_row_chars"] = max_len
    # leakage: eval slices vs each other and vs every tier audit sample.
    # T0/T1 slices are labeled memorization probes (overlap reported, not gated).
    leak: dict[str, int] = {}
    probe_note: dict[str, str] = {}
    names = sorted(eval_text)
    for x in range(len(names)):
        for y in range(x + 1, len(names)):
            a, b = names[x], names[y]
            leak[f"exact_{a}_x_{b}"] = len(set(eval_text[a]) & set(eval_text[b]))
    for tier in range(5):
        sample_set = set(audit[f"t{tier}"])
        for name in names:
            key = f"exact_train-t{tier}_x_{name}"
            leak[key] = len(sample_set & set(eval_text[name]))
            stag = int(name.split("_t")[1])
            if stag <= 1 or tier <= 1:
                probe_note[key] = "memorization-probe slice (overlap allowed, reported)"
    manifest["leakage"] = leak
    manifest["probe_slices"] = probe_note
    fatal, _ = leakage_verdict(leak)
    manifest["leakage_fatal"] = fatal
    manifest["teacher_inventory"] = {
        "digadd": 200, "digsub": 200, "singlemul": 64, "divmicro": 156}
    manifest["total_bytes"] = sum(t.get("train_bytes", 0) for t in manifest["tiers"].values())
    if out is not None:
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


__all__ = [
    "AUDIT_PER_TIER",
    "BANDS",
    "CURRICULUM",
    "EVAL_DEV_N",
    "EVAL_TEST_N",
    "GENERATOR_VERSION",
    "TEACHER_RATIO",
    "TIERS",
    "TEMPLATES",
    "TRAIN_N",
    "UNIFORM_MIXTURE",
    "build_manifest",
    "curriculum_tier",
    "eval_pair_leakage",
    "leakage_verdict",
    "parse_arith",
    "render",
    "teacher_row",
    "tier_index",
    "tier_row",
    "uniform_tier",
]
