"""CITADEL-EVAL-001 executable attack suite (stdlib only, deterministic).

Three mechanical layers, each producing numbers an auditor can re-run:

1. SHORTCUT ATTACKS — cheap adversarial solvers scored against the SAME real
   tiered-arithmetic rows used for evaluation. If a trivial solver matches the
   gold at ceiling, the evaluation cannot certify the skill it claims.
2. CAUSAL-DATA CHECKS — does the corpus itself distinguish causally relevant
   mutations (operation swap) and hold gold machine-consistent (reference
   solver)?
3. FIREWALL CHECKS — static/behavioral: the corpus generator must not import
   evaluation or scoring code; the contamination gate must fail closed on a
   planted leak; a tamper digest must flip when any gate result changes.

Exit code: 0 when every attack/mechanism behaved as designed (including
"attack succeeded" — that is the expected finding). Exit 1 only when a
mechanism itself is broken (e.g. gold is not machine-consistent, a firewall
check that should hold does not).

Run:  python tools/citadel_eval_attack.py [--n 300] [--json PATH]
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from citadel_tpu import tiered_data as td  # noqa: E402
from citadel_tpu.data_gate import _ngrams, _norm_text, gate_contamination  # noqa: E402

TOOL_SCHEMA = "citadel-eval-attack/v1"

_NUM_RE = re.compile(r"-?\d+")

# Modules that may never be imported by the pure corpus generator (evaluation
# or receipt/scoring code — the generator must be usable to build sealed
# fixtures without evaluation logic in its import closure).
_EVAL_OR_SCORING_MODULES = {
    "calculator_eval", "data_gate", "self_knowledge", "pre500m", "pre50m",
    "milestones", "runtime_bootstrap",
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _gold(meta: dict[str, Any]) -> str:
    """Gold answer derived from operands, never from the rendered text."""
    return str(meta["c"])


def _solve_reference(meta: dict[str, Any]) -> str:
    """Independent reference solver: recompute a op b from operands."""
    a, b, op = int(meta["a"]), int(meta["b"]), meta["op"]
    if op == "+":
        return str(a + b)
    if op == "-":
        return str(a - b)
    if op == "*":
        return str(a * b)
    return str(a // b)


def _numbers(text: str) -> list[str]:
    return _NUM_RE.findall(text)


def _accuracy(pred: Callable[[str, dict[str, Any]], str],
              rows: list[tuple[str, dict[str, Any]]]) -> float:
    if not rows:
        return 0.0
    hits = sum(1 for text, meta in rows if pred(text, meta) == _gold(meta))
    return hits / len(rows)


# --------------------------------------------------------------------------
# 1. shortcut attacks
# --------------------------------------------------------------------------

def attack_latest_position(text: str, meta: dict[str, Any]) -> str:
    nums = _numbers(text)
    return nums[-1] if nums else ""


def attack_first_position(text: str, meta: dict[str, Any]) -> str:
    nums = _numbers(text)
    return nums[0] if nums else ""


def attack_constant(value: str) -> Callable[[str, dict[str, Any]], str]:
    def pred(text: str, meta: dict[str, Any]) -> str:
        return value
    return pred


def attack_most_common(rows: list[tuple[str, dict[str, Any]]]) -> Callable[..., str]:
    counts: dict[str, int] = {}
    for _, meta in rows:
        g = _gold(meta)
        counts[g] = counts.get(g, 0) + 1
    mode = max(counts, key=counts.get) if counts else "0"

    def pred(text: str, meta: dict[str, Any]) -> str:
        return mode
    return pred


def attack_random(seed: int = 1234) -> Callable[[str, dict[str, Any]], str]:
    """Deterministic per-row random guess among the operands and zero.
    Ignores the rendered text entirely (query-blind by construction)."""
    def pred(text: str, meta: dict[str, Any]) -> str:
        del text
        rng = random.Random(seed + int(meta["a"]) * 1_000_003 + int(meta["b"]))
        return rng.choice([str(int(meta["a"])), str(int(meta["b"])), "0"])
    return pred


def attack_reference(text: str, meta: dict[str, Any]) -> str:
    return _solve_reference(meta)


ATTACKS = ("latest_position", "first_position", "constant_zero",
           "most_common_answer", "random_guess", "reference_solver")


def attack_tier(tier: int, n: int) -> dict[str, Any]:
    """Score every attack against n real test rows of one tier."""
    rows = [td.tier_row(tier, "test", i) for i in range(n)]
    scores = {
        "latest_position": _accuracy(attack_latest_position, rows),
        "first_position": _accuracy(attack_first_position, rows),
        "constant_zero": _accuracy(attack_constant("0"), rows),
        "most_common_answer": _accuracy(attack_most_common(rows), rows),
        "random_guess": _accuracy(attack_random(), rows),
        "reference_solver": _accuracy(attack_reference, rows),
    }
    return {"tier": tier, "n": n,
            "scores": {k: round(v, 4) for k, v in scores.items()}}


def run_shortcut_attacks(n_per_tier: int = 300) -> dict[str, Any]:
    per_tier = [attack_tier(t, n_per_tier) for t in range(5)]
    return {"schema": TOOL_SCHEMA, "attack": "shortcut_solvers",
            "split": "test", "n_per_tier": n_per_tier, "per_tier": per_tier}


# --------------------------------------------------------------------------
# 2. causal-data checks
# --------------------------------------------------------------------------

def check_gold_self_consistency(tier: int, n: int) -> dict[str, Any]:
    """The reference solver must reproduce gold on every row: gold is claimed
    to be computed from operands. Any miss means the corpus gold is broken."""
    rows = [td.tier_row(tier, "test", i) for i in range(n)]
    misses = [(i, meta) for i, (_, meta) in enumerate(rows)
              if _solve_reference(meta) != _gold(meta)]
    return {"tier": tier, "n": n, "misses": len(misses),
            "self_consistent": not misses}


def check_operation_swap_sensitivity(tier: int, n: int) -> dict[str, Any]:
    """Pair each row with its operation-swapped twin (a+b vs a-b, same
    operands). A corpus whose answers barely move under the swap cannot
    distinguish an addition model from a subtraction model. Division is
    excluded (no clean inverse twin)."""
    rows = [td.tier_row(tier, "test", i) for i in range(n)]
    pairs = compared = differ = 0
    for _, meta in rows:
        if meta["op"] not in ("+", "-"):
            continue
        pairs += 1
        swapped = dict(meta, op="+" if meta["op"] == "-" else "-")
        if _solve_reference(swapped) != _solve_reference(meta):
            compared += 1
            if int(_solve_reference(swapped)) != int(_solve_reference(meta)):
                differ += 1
    return {"tier": tier, "n": n, "addsub_rows": pairs,
            "swap_answer_changes": differ,
            "sensitivity": round(differ / pairs, 4) if pairs else None}


def run_causal_checks(n_per_tier: int = 300) -> dict[str, Any]:
    gold = [check_gold_self_consistency(t, n_per_tier) for t in range(5)]
    swap = [check_operation_swap_sensitivity(t, n_per_tier) for t in range(1, 5)]
    return {"schema": TOOL_SCHEMA, "check": "causal_data",
            "gold_self_consistency": gold,
            "operation_swap_sensitivity": swap}


# --------------------------------------------------------------------------
# 3. firewall checks
# --------------------------------------------------------------------------

def _module_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module.split(".")[0])
    return names


def check_generator_isolation() -> dict[str, Any]:
    """The corpus generator must not import evaluation/scoring modules, and
    evaluation helpers must not import the training loop. Verified on real
    source files in this repository."""
    pkg = ROOT / "citadel_tpu"
    violations: list[str] = []
    gen_imports = _module_imports(pkg / "tiered_data.py")
    bad = gen_imports & _EVAL_OR_SCORING_MODULES
    if bad:
        violations.append(f"tiered_data imports eval/scoring modules: {sorted(bad)}")
    for eval_mod in ("calculator_eval", "self_knowledge"):
        imports = _module_imports(pkg / f"{eval_mod}.py")
        bad = imports & {"calculator_train", "t1d_run", "t1c_run", "one_update"}
        if bad:
            violations.append(f"{eval_mod} imports training-loop modules: {sorted(bad)}")
    return {"check": "generator_isolation", "violations": violations,
            "pass": not violations}


def check_contamination_gate_fails_closed() -> dict[str, Any]:
    """A verbatim eval row planted into train MUST be reported. Silently
    passing would mean the firewall is decorative."""
    base = "7319 + 246 = 7565"
    train = [{"text": base}, {"text": "12 + 9 = 21"}]
    clean = gate_contamination(train, [{"text": "404 + 51 = 455"}])
    dirty = gate_contamination(train, [{"text": base},
                                       {"text": "some totally unrelated eval row"}])
    ok = (not clean) and len(dirty) >= 1
    return {"check": "contamination_gate_fail_closed",
            "clean_defects": clean, "planted_defects": dirty,
            "pass": ok}


def _gate_digest(gates: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(gates, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def check_tamper_digest() -> dict[str, Any]:
    """A flipped gate result must change the receipt digest: promotion logic
    consuming immutable receipts can only work if receipts are tamper-evident."""
    a = {"gate_a": {"defects": [], "pass": True},
         "gate_b": {"defects": ["x"], "pass": False}}
    b = {"gate_a": {"defects": [], "pass": False},
         "gate_b": {"defects": ["x"], "pass": False}}
    c = json.loads(json.dumps(a))
    c["gate_b"]["defects"] = []
    d1, d2, d3 = _gate_digest(a), _gate_digest(b), _gate_digest(c)
    ok = len({d1, d2, d3}) == 3 and d1 != d2 and d1 != d3
    return {"check": "receipt_tamper_evidence",
            "digests_differ": [d1 != d2, d1 != d3], "pass": ok}


def run_firewall_checks() -> dict[str, Any]:
    checks = [check_generator_isolation(),
              check_contamination_gate_fails_closed(),
              check_tamper_digest()]
    return {"schema": TOOL_SCHEMA, "check": "firewall", "results": checks,
            "all_pass": all(c["pass"] for c in checks)}


# --------------------------------------------------------------------------
# verdict + main
# --------------------------------------------------------------------------

def build_readiness(attacks: dict, causal: dict, firewall: dict) -> dict[str, Any]:
    """Machine-readable verdict consistent with EVALUATION_INVENTORY.md."""
    latest = [t["scores"]["latest_position"] for t in attacks["per_tier"]]
    ref_ok = all(t["scores"]["reference_solver"] == 1.0
                 for t in attacks["per_tier"])
    gold_ok = all(not g["misses"] for g in causal["gold_self_consistency"])
    blockers = []
    if max(latest) >= 0.99:
        blockers.append("LATEST_POSITION_SHORTCUT: trivial solver scores "
                        f"{max(latest):.3f} at ceiling on every tier")
    blockers.append("EOS_TERMINATION_UNSUPERVISED: training never supervised "
                    "the EOS token (T1D evidence, 15000/15000 MAX_TOKENS)")
    blockers.append("CROSS_SPLIT_CONTAMINATION: 173 dev + 357 test texts "
                    "appear verbatim in train (CITADEL-DATA-001)")
    blockers.append("NO_SEALED_FIXTURE: no sealed evaluation fixture exists; "
                    "all certificates are development-tier")
    return {
        "schema": TOOL_SCHEMA,
        "branch": "eval-integrity-001",
        "generated_utc_source": "deterministic (no wall clock in verdict fields)",
        "surfaces": {
            "t1d_tiered_arithmetic": {
                "classification": "SHORTCUT_COMPROMISED + LEAKAGE_COMPROMISED",
                "latest_position_per_tier": latest,
                "gold_machine_consistent": ref_ok,
            },
            "e0_cognition_dev_certificate": {
                "classification": "PROVISIONALLY_VALID (dev tier only)",
            },
            "triquetra_binding": {"classification": "PROVISIONALLY_VALID"},
            "pre50m_smoke": {"classification": "VALIDATED (mechanical only)"},
            "pre500m_decision": {"classification": "NOT_EXECUTED"},
            "data_readiness_gate": {"classification": "PROVISIONALLY_VALID"},
        },
        "causal": {"gold_self_consistent": gold_ok,
                   "swap_check": causal["operation_swap_sensitivity"]},
        "firewall_all_pass": firewall["all_pass"],
        "verdict": "NOT_READY",
        "blockers": blockers,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=300, help="rows per tier")
    ap.add_argument("--json", type=str, default=None, help="optional output path")
    args = ap.parse_args(argv)

    attacks = run_shortcut_attacks(args.n)
    causal = run_causal_checks(args.n)
    firewall = run_firewall_checks()
    readiness = build_readiness(attacks, causal, firewall)

    broken_mechanisms = [c for c in firewall["results"] if not c["pass"]]
    gold_broken = [g for g in causal["gold_self_consistency"] if g["misses"]]
    report = {"attacks": attacks, "causal": causal, "firewall": firewall,
              "readiness": readiness,
              "tool_status": "BROKEN_MECHANISM" if (broken_mechanisms or gold_broken)
                             else "OK"}
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.json:
        Path(args.json).write_text(payload, encoding="utf-8")
    print(payload)
    return 1 if report["tool_status"] != "OK" else 0


if __name__ == "__main__":
    raise SystemExit(main())
