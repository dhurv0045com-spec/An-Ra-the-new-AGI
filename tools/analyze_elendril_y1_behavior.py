"""Supplemental Y1 generator-cue and recorded-behavior audit; no neural inference."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import zipfile

from anra_v5.elendril.y1_data import load_y1_suite


def analyze(archive: Path) -> dict:
    result = {"schema": "codex-y1-behavior-audit-v1", "scope": "Public fixture cue audit and recorded predictions; not fresh inference or proof of neural mechanism", "splits": {}, "jobs": {}}
    with zipfile.ZipFile(archive) as z:
        suites = {}
        for split, member in (("development", "development_suite"), ("confirmation", "confirmation_suite"), ("capacity_followup", "followup_suite")):
            suite = load_y1_suite(json.loads(z.read(f"prepared/{member}.json")))
            suites[split] = suite
            cells = defaultdict(Counter)
            gold_counts = defaultdict(Counter)
            for c in suite["cases"]:
                ev = c.environment.initial
                if c.panel in ("main-base", "main-row-perm") and c.solvability == "answerable":
                    # Uses only public hop count and public timestamps; ignores query identity.
                    ranked = sorted(ev, key=lambda e: e.tick)[c.hops - 1].value
                    cells["timestamp_rank_primary"]["total"] += 1
                    cells["timestamp_rank_primary"]["correct"] += ranked == c.v6_case.truth.expected
                if c.panel in ("main-base", "missing-bridge-panel") and c.solvability == "query_needed":
                    payload = [e for events in c.environment.channels.values() for e in events]
                    cells["query_terminal_payload"]["total"] += 1
                    cells["query_terminal_payload"]["one_event"] += len(payload) == 1
                    cells["query_terminal_payload"]["last_payload_object_is_gold"] += bool(payload) and payload[-1].value == c.v6_case.truth.expected
                if c.panel == "main-terminal-value-change" and c.applicable:
                    gold_counts["main_terminal_change"][str(c.v6_case.truth.expected)] += 1
                if c.panel == "edge-change-panel" and c.factor == "fixed_pos":
                    gold_counts["edge_panel_changed"][str(c.v6_case.truth.expected)] += 1
            result["splits"][split] = {"cue_counts": {k: dict(v) for k, v in cells.items()}, "changed_gold_counts": {k: dict(v) for k, v in gold_counts.items()}}

        for name in sorted(z.namelist()):
            if not name.startswith("evidence/predictions/") or name.endswith("_hist.json"):
                continue
            doc = json.loads(z.read(name))
            jid = Path(name).stem
            split = "capacity_followup" if "t3_" in jid else "confirmation" if "t2_" in jid else "development"
            cases = {c.case_id: c for c in suites[split]["cases"]}
            groups = defaultdict(list)
            counts = Counter()
            examples = []
            for r in doc["predictions"]:
                c = cases[r["case_id"]]
                if c.panel == "main-base" and c.solvability == "answerable":
                    counts["answerable_base_total"] += 1
                    counts["answerable_base_false_abstain"] += r["first_action"] == "ABSTAIN"
                    counts["answerable_base_answer_first"] += r["first_action"] == "ANSWER"
                    counts["answerable_base_correct"] += bool(r["correct"])
                if c.panel in ("main-base", "main-row-perm") and c.solvability == "answerable":
                    counts["primary_total"] += 1
                    expected_rule = sorted(c.environment.initial, key=lambda e: e.tick)[c.hops - 1].value
                    counts["primary_final_answer_matches_timestamp_rule"] += r["final_answer"] == expected_rule
                if c.panel == "dual-chain-query-panel":
                    groups[c.family_id].append(r)
            for family, rows in sorted(groups.items()):
                assert len(rows) == 2, (jid, family, len(rows))
                a, b = rows
                counts["dual_pairs"] += 1
                a_out = tuple(tuple(d["raw_tokens"]) for d in a["decisions"])
                b_out = tuple(tuple(d["raw_tokens"]) for d in b["decisions"])
                counts["dual_identical_complete_raw_outputs"] += a_out == b_out
                counts["dual_both_correct"] += a["correct"] and b["correct"]
                counts["dual_both_abstain_first"] += a["first_action"] == b["first_action"] == "ABSTAIN"
                counts["dual_both_answer_first"] += a["first_action"] == b["first_action"] == "ANSWER"
                counts["dual_same_nonnull_answer"] += a["final_answer"] is not None and a["final_answer"] == b["final_answer"]
                if len(examples) < 3 and a_out == b_out:
                    examples.append({"family_id": family, "cases": [a["case_id"], b["case_id"]], "gold": [cases[r["case_id"]].v6_case.truth.expected for r in rows], "predicted": [a["final_answer"], b["final_answer"]], "raw_tokens": [list(x) for x in a_out]})
            d1name = f"analysis/jobs/{jid}/d1_summary.json"
            d1 = json.loads(z.read(d1name)) if d1name in z.namelist() else None
            result["jobs"][jid] = {"split": split, "counts": dict(counts), "identical_output_examples": examples, "d1_summary": d1}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({"jobs": len(result["jobs"]), "splits": result["splits"], "bytes": args.output.stat().st_size}))


if __name__ == "__main__":
    main()
