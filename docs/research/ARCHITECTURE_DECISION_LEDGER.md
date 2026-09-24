# ARCHITECTURE DECISION LEDGER

**Generated:** 2026-09-24
**Re-freeze phase:** 3
**Historical phase-2 snapshot:** 2026-09-13 (preserved in the machine ledger metadata)
**Machine source:** [`ARCHITECTURE_DECISION_LEDGER.json`](ARCHITECTURE_DECISION_LEDGER.json) (schema `anra.architecture-decision-ledger/v1`)
**Authority:** [`EVIDENCE_SOURCE_MANIFEST_2026-09-24.json`](EVIDENCE_SOURCE_MANIFEST_2026-09-24.json), SHA-256 `c164c735ec4e628a6311fe4c52117c8b58370321df61b00d4976c11769f0215f`
**Pre-consolidation tip:** `research/evidence-consolidation-2026-09-25` @ `90f77b7fa6ffd99f5a982263f03b2908298805ec`; parent of the consolidated head.

**Authorization ceiling:** no row authorizes production vocabulary or tokenizer replacement, PRE500M, 250M, 500M, cognition, AGI, tool learning, TPU qualification, or RSI.

Statuses: `LOCKED_BY_EVIDENCE` · `DEFAULT_BASELINE` · `PROVISIONAL` · `BLOCKED_ON_EXPERIMENT` · `REJECTED` · `UNJUSTIFIED`.

## Summary table

| ID | Component | Decision | Status | Evidence | Blocked by |
|---|---|---|---|---|---|
| AD01 | model family | dense causal decoder-only Transformer | DEFAULT_BASELINE | closure canaries; B01 | — |
| AD02 | parameter scale | ~250M V5-A candidate; 500M deferred | PROVISIONAL | closure canaries; confounded T1-series; CS development-only | U07 |
| AD03 | depth | 26 layers | PROVISIONAL | signal-propagation canary | — |
| AD04 | width | 896 | PROVISIONAL | — | — |
| AD05 | FFN ratio | SwiGLU ~2.64x | DEFAULT_BASELINE | — | — |
| AD06 | attention | full causal; GQA 2:1; head 64 | DEFAULT_BASELINE | repeat-KV equivalence | — |
| AD07 | QK norm | affine per-head RMS | LOCKED_BY_EVIDENCE | Q/K stress invariance ≤1.0001 | — |
| AD08 | residual init | 0.02/√(2L) on residual outputs | LOCKED_BY_EVIDENCE | growth ratio 0.12–0.23 | — |
| AD09 | position/context | RoPE 10k; context 4,096 | PROVISIONAL | conformance only | — |
| AD10 | norm placement | pre-RMSNorm + final | DEFAULT_BASELINE | — | — |
| AD11 | bias/dropout | none / 0 | DEFAULT_BASELINE | — | — |
| AD12 | tokenizer family | byte-BPE + byte fallback | PROVISIONAL | planning prior; CS does not establish natural-language transfer | U07 |
| AD13 | vocabulary size | 24,576 conservative frozen default | BLOCKED_ON_EXPERIMENT | completed R1C + CS; geometry, initialization, optimizer/WD, and transfer open | U03+U07 |
| AD14 | numeric/symbol representation | default BPE number segmentation | BLOCKED_ON_EXPERIMENT | completed R1C + CS; segmentation still correlated | U03+U07 |
| AD15 | tied embedding/output | tied currently; no mechanism lock | BLOCKED_ON_EXPERIMENT | R1C/CS leave the interaction open; S5 is floor-limited; preregistered Role-Transfer is unexecuted and blocked | U03 |
| AD16 | masked/intermediate softmax remedy | do not use as the production remedy | **REJECTED** | R1C `SOFTMAX_COMPETITION_NOT_SUFFICIENT`; CS `PARTIAL_OR_INTERACTION`; no V24576-optimality claim | — |
| AD17 | precision | BF16 compute + FP32 master/moments | LOCKED_BY_EVIDENCE | parity 0.000118; native-BF16 violation | — |
| AD17b | optimizer | AdamW (0.9, 0.95), selective wd 0.1 | DEFAULT_BASELINE | production contract | — |
| AD18 | LR policy | HIGH acquire/recover; LOW/CAP1X plus support retain | PROVISIONAL | ARK-007R/010/011/017-V2 | U05 |
| AD18b | LR schedule | token-indexed WSD | PROVISIONAL | never executed end-to-end | N-ENTRY-POINT |
| AD19 | gradient clipping | global L2 1.0; certified tolerance 1e-4 | LOCKED_BY_EVIDENCE | R1C reduction-order derivation | — |
| AD20 | replay | external treatment-exact sparse replay; dose not universal; no internalization | PROVISIONAL | ARK-017-V2; ARK-019-V3.1 | U05 |
| AD21 | continual controller | no controller in production | BLOCKED_ON_EXPERIMENT | V3.1 not supported; V4 raw Guardian gap; ARK-020 DO_NOT_RUN | U02+U04 |
| AD22 | objective | causal CE; answer+EOS; auxiliary lambdas 0 | LOCKED_BY_EVIDENCE | EOS contract; margin falsified | — |
| AD23 | data mixture | 65/20/15 plus cognition families | PROVISIONAL | unmeasured; zero-control missing | U08 |
| AD24 | exposure policy | never stop at train saturation | LOCKED_BY_EVIDENCE | ARK-002B + ARK-004A | — |
| AD25 | evaluation architecture | candidate-free primary; orthogonal axes; sealed firewall; robust-min | LOCKED_BY_EVIDENCE | scorer failures; ARK-015; CITADEL-EVAL-001 | — |
| AD26 | cognition generators | executable truth + attack screens; tiered surface retired | BLOCKED_ON_EXPERIMENT | data/evaluation audits | U08 |
| AD27 | external memory/tools | stay outside the Core | DEFAULT_BASELINE | structural asset | — |
| AD28 | MoE/SSM/latent thought/memory | none | UNJUSTIFIED | no bottleneck evidence | — |
| AD29 | checkpoint/promotion | fail-closed content-addressed pipeline | LOCKED_BY_EVIDENCE | restore equals uninterrupted; tamper rejection | — |
| AD30 | learned self-model heads | none | REJECTED | X1 invalidated; no qualified subject | U09 |
| AD31 | Formation-Mux TIE-role frontier | no promotion; blocked provisional branch | BLOCKED_ON_EXPERIMENT | S5 floor-limited; v12 2/24 with no sealed/final/checkpoint payload | U12 |

## Count by status

- `LOCKED_BY_EVIDENCE`: **8** — AD07, AD08, AD17, AD19, AD22, AD24, AD25, AD29
- `DEFAULT_BASELINE`: **7** — AD01, AD05, AD06, AD10, AD11, AD17b, AD27
- `PROVISIONAL`: **9** — AD02, AD03, AD04, AD09, AD12, AD18, AD18b, AD20, AD23
- `BLOCKED_ON_EXPERIMENT`: **6** — AD13, AD14, AD15, AD21, AD26, AD31
- `REJECTED`: **2** — AD16, AD30
- `UNJUSTIFIED`: **1** — AD28

Total: **33 decisions**.

## Architecture-agent notes

1. R1C and CS-TRANSFER-001 are complete evidence, not pending gates. AD13/AD14/AD15 cite them as completed constraints. Preregistered `ROLE-TRANSFER-001` is the blocked design of record for the narrower geometry/interaction question; it is not positive evidence.
2. AD16 rejects masked/intermediate softmax as a production remedy based on R1C, without claiming V24576 optimality.
3. AD21 cannot unlock from the transcribed Guardian claim: the raw bundle is unresolved and ARK-020 is `DO_NOT_RUN/NOT_EXECUTED` with confirmed resume/identity defects.
4. AD31 remains blocked because floor-limited S5 NULLs and a 2/24 partial frontier cannot lock or exonerate the TIE-role mechanism.
5. K8 and TPU records are engineering-only and do not authorize cognition, AGI, tool learning, TPU qualification, or RSI.
