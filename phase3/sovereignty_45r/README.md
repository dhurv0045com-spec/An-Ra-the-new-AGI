# 45R — Sovereignty

**Component 19/19 · `sovereignty`**

Nothing gets promoted because it "felt better." Sovereignty is the **governance layer**: audit source health, catch regressions, sweep dead code, compare benchmarks, and **promote or quarantine** checkpoints.

---

## What runs in an audit

1. **Source health** — all 19 registered paths exist  
2. **Import health** — components load without degradation  
3. **Dead code sweep** — unreachable registered modules  
4. **Benchmark deltas** — new vs baseline scores  
5. **Identity stability** — CIVGuard cosine check  
6. **Regression** — `EvalHarness` compare vs stored baseline  

Fail any gate → checkpoint stays quarantined; report explains why.

---

## Operator commands

```bash
python anra.py --sovereignty-report
python anra.py --sovereignty-run
python anra.py --briefing          # often includes sovereignty summary
python scripts/run_sovereignty_audit.py
```

**Daemon / API** (optional local install): see `service.py`, `daemon.py`, `api.py` in this folder.

---

## Key files

| File | Role |
| --- | --- |
| `sovereignty_bridge.py` | Main integration surface |
| `auditor.py` | Audit orchestration |
| `benchmarks.py` | Benchmark suite |
| `reporter.py` | Nightly / milestone reports |
| `improver.py` | Improvement pipeline hooks |
| `dead_code.py` | Reachability analysis |

---

## Promotion story

```text
train → eval → sovereignty audit
  → pass → anra_v2_brain.pt (production)
  → fail → hold + report in output/v2/reports/
```

**45R does not train models.** It judges whether training output is allowed to become production.

---

## Dependencies

- `psutil` recommended for resource monitoring  
- Works offline for core audit paths  
- Pairs with `engine/eval_harness.py` and `engine/report.py`

See [`PHASE3_INTEGRATION.md`](../PHASE3_INTEGRATION.md) for pipeline placement.
