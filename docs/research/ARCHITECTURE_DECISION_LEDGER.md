# ARCHITECTURE DECISION LEDGER

**Phase 2 · 2026-09-13.** The legal state of every design decision for the next-Core architecture agent. Machine source: [`ARCHITECTURE_DECISION_LEDGER.json`](ARCHITECTURE_DECISION_LEDGER.json) (schema `anra.architecture-decision-ledger/v1`).

**Completion test (Phase-2 §27):** for every row, the answer to *"why is this in the next Core?"* is one of: evidence locks it / simplest baseline while evidence is absent / provisional / blocked on experiment X / rejected. No row answers "seemed like a good idea."

Statuses: `LOCKED_BY_EVIDENCE` · `DEFAULT_BASELINE` · `PROVISIONAL` · `BLOCKED_ON_EXPERIMENT` · `REJECTED` · `UNJUSTIFIED`.

## Summary table

| ID | Component | Decision | Status | Evidence | Blocked by |
|---|---|---|---|---|---|
| AD01 | model family | dense causal decoder-only Transformer | DEFAULT_BASELINE | — (baseline argument; B01) | — |
| AD02 | parameter scale | ~250M V5-A; 500M deferred | PROVISIONAL | closure canaries; T1-series confounded null | U03 |
| AD03 | depth | 26 layers | PROVISIONAL | signal-propagation canary | — |
| AD04 | width | 896 | PROVISIONAL | — | — |
| AD05 | FFN ratio | SwiGLU ~2.64× | DEFAULT_BASELINE | — | — |
| AD06 | attention | full causal; GQA 2:1; head 64 | DEFAULT_BASELINE | repeat-KV equivalence receipt | — |
| AD07 | QK norm | affine per-head RMS | **LOCKED_BY_EVIDENCE** (mechanism prior D-025) | Q/K stress invariance ≤1.0001 | — |
| AD08 | residual init | 0.02/√(2L) on residual outputs | **LOCKED_BY_EVIDENCE** (D-024) | growth ratio 0.12–0.23 | — |
| AD09 | position/context | RoPE 10k; ctx 4,096 | PROVISIONAL | conformance only | — |
| AD10 | norm placement | pre-RMSNorm + final | DEFAULT_BASELINE | — | — |
| AD11 | bias/dropout | none / 0 | DEFAULT_BASELINE | — | — |
| AD12 | tokenizer family | byte-BPE + byte fallback | PROVISIONAL | compression prior only | U01+U03 |
| AD13 | vocabulary size | 24,576 frozen | **BLOCKED_ON_EXPERIMENT** | class-space effect 0%↔100% (R1/R1B) | **U01+U03 (R1C + CS-TRANSFER-001)** |
| AD14 | numeric atomization | default BPE numbers | **BLOCKED_ON_EXPERIMENT** | inside composite treatment (CYR-011) | U01+U03 |
| AD15 | tied embedding/output | tied | **BLOCKED_ON_EXPERIMENT** | effect moves tied matrix jointly | U01 |
| AD16 | output softmax | full vocab | **BLOCKED_ON_EXPERIMENT** | MASK/OFFSET arms frozen unexecuted (R1C) | U01 |
| AD17 | precision | BF16 compute + FP32 master/moments | **LOCKED_BY_EVIDENCE** | parity 0.000118; native-BF16 violation (D-026/D-028) | — |
| AD17b | optimizer | AdamW (0.9, 0.95), wd 0.1 selective | DEFAULT_BASELINE | production contract | — |
| AD18 | LR policy | HIGH acquire/recover; LOW/CAP1X+support retain | PROVISIONAL (micro law) | ARK-007R/010/011/017-V2 | U05 |
| AD18b | LR schedule | token-indexed WSD | PROVISIONAL | never executed end-to-end | entry-point gate |
| AD19 | gradient clipping | global L2 1.0, cert tolerance 1e-4 | **LOCKED_BY_EVIDENCE** | R1C reduction-order derivation | — |
| AD20 | replay | external, treatment-exact, sparse; dose NOT universal; no internalization | PROVISIONAL | ARK-017-V2 both-levers; BRM replay null (provenance-weak) | U05 |
| AD21 | continual controller | none in production | **BLOCKED_ON_EXPERIMENT** | V3.1 NOT_SUPPORTED; V4 transcribed-only | **U02** |
| AD22 | objective | causal CE, answer+EOS supervised; aux lambdas 0 | **LOCKED_BY_EVIDENCE** | EOS cross-program; margin falsified | — |
| AD23 | data mixture | 65/20/15 + cognition families | PROVISIONAL | unmeasured; zero-control missing | U08 + E3 |
| AD24 | exposure policy | never stop at train saturation | **LOCKED_BY_EVIDENCE** | ARK-002B + rho 0.00 (ARK-004A) | — |
| AD25 | evaluation architecture | candidate-free primary; orthogonal axes; sealed firewall; attack screens | **LOCKED_BY_EVIDENCE** | scorer failures; ARK-015; CITADEL-EVAL-001 | — |
| AD26 | cognition generators | executable-truth + attack screens; tiered surface retired | **BLOCKED_ON_EXPERIMENT** | CITADEL-DATA-001 (shortcut/leakage/supply) | U08 |
| AD27 | external memory/tools | stay outside the Core | DEFAULT_BASELINE | structural asset | — |
| AD28 | MoE / SSM / latent-thought / learned memory | none | **UNJUSTIFIED** | no bottleneck evidence | — |
| AD29 | checkpoint/promotion | fail-closed content-addressed pipeline | **LOCKED_BY_EVIDENCE** | canaries; restore ≡ uninterrupted | — |
| AD30 | learned self-model heads | none | **REJECTED** | X1 invalidated; no qualified subject | U09 (re-entry gate) |

## Count by status

LOCKED_BY_EVIDENCE: 7 (AD07, AD08, AD17, AD19, AD22, AD24, AD25, AD29 — 8 including AD29) · DEFAULT_BASELINE: 6 · PROVISIONAL: 8 · BLOCKED_ON_EXPERIMENT: 6 (AD13, AD14, AD15, AD16, AD21, AD26) · REJECTED: 1 (AD30) · UNJUSTIFIED: 1 (AD28).

## Notes for the architecture agent

1. The **only** architecture components with LOCKED status are: scale-control mechanisms (QK norm, residual init), precision layout, clipping invariant, objective/EOS contract, exposure policy, evaluation architecture, and the promotion pipeline. Everything exotic is UNJUSTIFIED; everything popular-but-untested here is DEFAULT.
2. **AD13/AD14/AD15/AD16 form a single evidence bundle**: the R1C + CS-TRANSFER-001 pair decides them together. Deciding any one of them early is UNJUSTIFIED.
3. AD21 (controller) re-opens only through the U02 audit path; AD30 (self-model) re-opens only through the U09 qualification path.
4. PROVISIONAL items carry their revisit trigger in the JSON `would_change_if` field — no provisional choice may be promoted without its named evidence.
