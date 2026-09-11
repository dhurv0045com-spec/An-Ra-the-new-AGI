# ARK-020 V3 — CORRECTNESS REPAIR AND OPERATOR FREEZE

**Status: PREREGISTERED BEFORE IMPLEMENTATION EXECUTION.** V1 (`experiments/ARK-020/`)
and V2 (`experiments/ARK-020-V2/`) are immutable historical records. V3 changes ONLY
what the confirmed V2 defects require; the broad architecture is unchanged.

## Question (unchanged)

Can one real-text-pretrained model sequentially acquire several genuinely different
capabilities (relational binding x2, two-hop in-context composition, inverse retrieval)
while retaining earlier capabilities and real-text competence, with adaptive protection
using less rehearsal than permanent replay?

## V2 defects confirmed live (each repaired in V3; evidence in PREEXECUTION_AUDIT.md)

1. **Exact-resume smoke could not execute**: `state_hashes(*a)` received 11 args against
   a 10-param signature (global_confirm missing); post-load reconstruction indexed
   `c2[3..5]` of a 4-tuple+None. No test ever executed the production function.
   REPAIR: dict-based state; hashes cover 12 state classes incl. global_confirm, phase
   identity, all three phase seeds, stream receipts; the production
   `exact_resume_smoke` is executed directly by tests and writes
   EXACT_RESUME_SMOKE_V3.json with per-field PASS/FAIL.
2. **Skill C positional shortcut**: XM and MY blocks used the SAME permutation, so
   "pick Y at the queried X's ordinal position" solved the task with no intermediate
   matching (measured: 1.000 under the tied renderer). REPAIR: independent hash
   namespaces per block and mode (`c_orders`); measured same-ordinal match rate
   0.3377 (~1/3 chance) across 3000 canonical prompts; X->M->Y oracle 1.0000.
3. **Skill D was not inverse**: facts were pre-reversed to (value, key), making the
   query a forward lookup of a presented adjacency. REPAIR: facts presented forward as
   (owner, object); query is the object; answer its owner; the ordered (object, owner)
   pair is never presented (mutation test proves the detector catches the old build).
4. **Resume scan before Drive mount + identity overclaim**: notebook mounted Drive in
   cell 3 while cell 0 scanned; scan printed PASS when keys merely existed. REPAIR:
   cell order = mount -> clone/checkout -> scan; scan verifies schema/version/seeds/arm/
   task-hash/parent-hash/dose against ENTRY_RECEIPT and DOSE_SELECTION, prints
   PARTIAL_IDENTITY_CHECK where substrate fields cannot be verified, and emits exactly
   one SAFE ACTION per the frozen rules.
5. **Vacuous construct**: `len(schema_keys) == 7 or True` in the V2 smoke. REPAIR:
   deleted; a meta-test forbids the construct in V3 sources.
6. **Weak lock**: 6h timestamp file. REPAIR: CAMPAIGN_LOCK.json with experiment,
   executable commit, session UUID, timestamp, host; classified ACTIVE/STALE/MALFORMED;
   ACTIVE never silently overridden.

## Unchanged from V2 (audited good)

- Skills A/B (V4-identical constructions), phases B 2000 / C 1500 / D 1500 updates,
  task slots dose-selected(B)/12/12, 4 matched sets, 7 arms, thresholds, controller
  state machine, predictive trigger, replay allocation (risk-ordered, max 2 slots),
  efficiency gate, formation/interference gates, phase-relative median acquisition
  rule, per-phase order seeds bound into identity, V4 integration shapes,
  SEALED firewall.

## Frozen execution identity

- Drive root `/content/drive/MyDrive/genisis-arkenstone/ARK020_V3_CONTINUAL` (never V2's).
- Bundles ARKENSTONE_ARK020_V3_CONTINUAL_{PARTIAL,RESULTS}.zip + .sha256 sidecars.
- Launcher `experiments/COLAB/arkenstone_ark020_v3.ipynb`; cell 0 mounts Drive FIRST,
  then clones and detach-checkouts the frozen executable commit, then runs the
  read-only scan.
- Claim ceiling: controlled real-text-proxy evidence for a multi-capability
  continual-learning controller candidate. Two-hop composition is a *proxy* — never
  "reasoning demonstrated". Inverse retrieval is a *proxy* — never "world-model
  bidirectionality". No universal laws, no PRE500M/500M, no AGI or consciousness claims.
