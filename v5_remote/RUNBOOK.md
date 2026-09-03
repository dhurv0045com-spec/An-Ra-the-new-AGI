# cymek remote runbook — E1–E6 as RemoteJobs (planned, not executed)

Nothing in this file has been submitted or run. Each row is a `RemoteJob`
template: freeze it with `v5_remote.job_spec`, transport the
`submission_envelope` to the accelerator host out of band, execute the pinned
command there, and collect with `v5_remote.collect`.

Update arithmetic uses the frozen 131,072 real tokens/update:

| Slice | Full updates | Partial tail | Total updates |
|---|---:|---:|---:|
| 100M | 762 × 131,072 = 99,876,864 | 123,136 | 763 |
| 200M | 1,525 × 131,072 = 199,884,800 | 115,200 | 1,526 |
| 600M | 4,577 × 131,072 = 599,916,544 | 83,456 | 4,578 |
| 1B | 7,629 × 131,072 = 999,948,288 | 51,712 | 7,630 |

## Order (frozen: E0 → E1/E2 → E3 → E4 → E5 → freeze review)

| Gate | Job | Remote command | Token slice |
|---|---|---|---:|
| E6-pre | `e6-preflight` | `esoes-v5-target-preflight --output artifacts/v5/target_preflight.json` | 0 (probe only) |
| E1 | `e1-p35-16k`, `e1-p35-24k`, `e1-p35-32k` | learned P35 runner at 16k/24k/32k (matched raw bytes + FLOPs) | 100–200M each |
| E2 | `e2-screen-*` fractional arms, then top-2 × 3 seeds | learned P35 2:1-GQA screen + replication | 200M each |
| E3-A | `e3-mix-05`, `e3-mix-15`, `e3-mix-30` (+ finalists) | CE-only cognition mixtures | 200M each |
| E3-B | `e3-qswap-005`, `e3-qswap-015` at winner + neighbor | query-swap challenger, FLOP-matched | gated on E3-A |
| E4 | `e4-lr-*`, `e4-curriculum-staged` (≤4 arms) | LR/batch + staged/replay challenger | bounded |
| E5 | `e5-winner-seed{1,2}`, `e5-control` | M102 winner vs CE/general-data control | 600M–1B each |
| E6 | `e6-durability`, `e6-sealed-replication` | remote upload/redownload/restore + sealed/fresh eval | — |

## Per-job freezing checklist

- `accelerator` + `replicas`: exact host slice (e.g. `tpu-v5e-8`, 8).
- `runtime_image_sha256`: pinned container that produced a local
  `BLOCKED_TORCH_XLA`-free preflight on the same host family.
- `code_commit`: full SHA-1 of the frozen tree the command runs from.
- `command`: exact argv; stdout receipt path inside the job sandbox.
- `identities`: fill the six slots from `blueprint/LAUNCH_GATES.json` as each
  gate freezes them; `null` only for genuinely future identities.
- `seed`: distinct per replication arm; never reuse a development seed for a
  confirmation arm.
- Collect with `--job/--result/--receipt-dir`: the collector re-hashes every
  receipt file and the remote log before `bind_result` accepts anything.

## Abort rules (mirror the frozen spec)

A `failed` result, a log/receipt mismatch, or an envelope edit stops the
dependent gate. Two consecutive Tier-1 worst-family declines >5 points while
LM loss improves pauses training and preserves the earlier milestone. No
retrospective policy selection from trained-model outcomes.
