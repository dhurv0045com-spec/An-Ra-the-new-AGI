# 45R - Sovereignty

**Layer 19/19: `sovereignty`**

Sovereignty is the governance and promotion layer. It audits code, checks dead code, benchmarks behavior, writes reports, and gives An-Ra a way to reject shallow progress.

The core principle is simple:

> Newer checkpoint does not automatically mean better checkpoint.

## Current Role

```text
source / checkpoint / reports
  -> audit
  -> dead-code sweep
  -> benchmark deltas
  -> report
  -> promote / hold / repair recommendation
```

## Main Files

| File | Role |
| --- | --- |
| `sovereignty_bridge.py` | Mainline bridge |
| `auditor.py` | AST quality and complexity checks |
| `dead_code.py` | Unused/dead-code sweep |
| `benchmarks.py` | Performance and behavior benchmark pass |
| `reporter.py` | Human-readable report generation |
| `improver.py` | Pipeline orchestration |
| `daemon.py`, `service.py` | Local daemon runtime |
| `api.py`, `auth.py` | Local authenticated API |
| `resource_monitor.py`, `watchdog.py` | Runtime health checks |

## Main Commands

From the repo root:

```bash
python anra.py --sovereignty-report
python anra.py --sovereignty-run
python scripts/run_sovereignty_audit.py
```

From this folder:

```bash
python demo.py
python service.py start
```

## Report Meaning

A good sovereignty report should answer:

- what improved?
- what regressed?
- which files or behaviors are risky?
- did benchmarks get slower or better?
- should a checkpoint be promoted, held, or repaired?

## Boundary

Sovereignty may recommend actions. It should not silently rewrite the project or promote artifacts without a visible audit trail. The system earns autonomy by making its judgment inspectable.
