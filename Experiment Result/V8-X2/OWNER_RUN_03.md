# X2 owner run 3 · Standalone recurrence (U-1, U-2, U-4 s7) collapses, and all 36 core runs reproduce bit-for-bit

**22 November 2026.** Owner Run 3 executed all 43 admitted S20 jobs cleanly across 22 waves with zero worker failures (`status: "COMPLETE"`, session slice finished within its 555-minute usable budget). Because Run 3 started without mounted prior owner ZIPs (`VERIFIED_RESUME_RECEIPT.json` imported `0` prior receipts), the runner re-executed all 36 non-A0 X1 core jobs alongside **7 new `IIIa_ORIGINAL_U` recurrent jobs**: `U-1` (seeds 7, 19, 41), `U-2` (seeds 7, 19, 41), and `U-4` (seed 7; seeds 19 and 41 deferred cleanly to Session 4). Together with `A0` (`6` jobs) and the `36` verified core jobs (`I_X1_CORE`), **`49` of `192` obligation rows are now complete (`42/42` X1 core + `7/12` `IIIa_ORIGINAL_U`)**. **The wider X2 campaign remains `PARTIAL` (`x3_unlocked: false`):** `U-4` (`s19`, `s41`), `U2-untied` (`3` seeds), `BII-FROZEN`, `IIIb` (`DM`, `P2`, `R2`, `U-P`, `PS`), `T13`, `T14`, `H4736`, and confirmation cohorts have no training outcomes in this archive. [Run 1 report](OWNER_RUN_01.md) · [Run 2 report](OWNER_RUN_02.md) · [A0 results](../V8-X1/A0/RESULTS.md) · [Run 3 machine-readable metrics](OWNER_RUN_03_METRICS.json)

## What the experiment measured

Sessions `U0` (`U-1`, `U-2`) and `U1` (`U-4`) evaluate standalone weight-tied recurrence (`X1TypedModel` with `_URecurrentStepBlock`, `21,589,396` parameters = S20 core + `F` + `1` recurrent cross-attention + GRU cell block, `use_pointer=False`) trained on the identical `D1-H` training stream, `2,368` updates, and `37,888` logged states as `A0` (`P`, `R`), `A1` (`F`, `F-cap`), `A2` (`P-capR`, `R-noedge`), and `A3` (`B0`, `J`). At update `2,368`, each seed was evaluated on the same 384 base development worlds across six linked presentations (`2,304` episodes per seed) plus the 320-world depth stress suite (`640` episodes per seed across 1–5 hops).

The table below places the new `U` models alongside every non-paired (`D1-H`) comparator from the verified Run 3 archive (and `A0` references from `A0/RESULTS.md`). Figures are percentages; `conjunction` requires all three invariance views (canonical, entity-code permutation, irrelevant facts) of a base world correct simultaneously, `edit pair` requires both sides of `320` eligible evidence-change pairs correct, and `five-hop` is the 5-hop slice (`128` episodes) of depth stress.

| Session and model | Seeds | Mean acc | Canonical | Narrative | Edit pair | 3-view conj | Five-hop |
|---|---|---:|---:|---:|---:|---:|---:|
| **U0 U-1** (`k=1`) | 7, 19, 41 | **19.66** | 21.44 | 11.55 | 1.67 | 12.07 | 6.25 |
| **U0 U-2** (`k=2`) | 7, 19, 41 | **17.14** | 19.36 | 7.20 | 0.10 | 11.81 | 1.56 |
| **U1 U-4** (`k=4`, `s7` only) | 7 *(n=1)* | **14.11** | 16.67 | 0.26 | 0.00 | 16.41 | 0.00 |
| A1 F *(no copy, D1-H)* | 7, 19, 41 | 27.65 | 30.21 | 18.92 | 10.21 | 11.81 | 21.09 |
| A1 F-cap *(capacity-matched)* | 7, 19, 41 | 28.10 | 31.16 | 10.76 | 11.35 | 9.90 | 22.14 |
| A3 B0 *(byte decoder)* | 7, 19, 41 | 29.99 | 31.60 | 20.31 | 11.77 | 11.11 | 22.14 |
| A3 J *(joint action head)* | 7, 19, 41 | 33.16 | 34.20 | 26.22 | 14.58 | 11.20 | 32.29 |
| R (A0 ref, copy + edges) | 7, 19, 41 | 61.65 | 69.79 | 25.43 | 75.21 | 52.34 | 58.33 |
| P (A0 ref, pointer) | 7, 19, 41 | **62.02** | 74.39 | 16.15 | 80.94 | 51.30 | 58.59 |
| A2 P-capR *(width-matched P)* | 7, 19, 41 | 59.07 | 68.66 | 16.23 | 74.79 | 53.47 | 55.73 |
| A2 R-noedge *(copy, no edges)* | 7, 19, 41 | 61.91 | 72.22 | 16.75 | 78.44 | 54.95 | 59.90 |

*(Run 3 also reproduced `D0 P` 42.13%, `D0 R` 40.68%, `D1 P` 41.04%, `D1 R` 45.11%, `D2 P` 39.99%, and `D2 R` 47.64% to 0.0000 percentage points against Run 2; see reproducibility section below.)*

## Findings and limits

### 1. Standalone recurrence performed worse than every baseline and degraded monotonically with more steps

Rather than helping iterative multi-hop reasoning, wrapping an extra recurrent GRU-attention loop on top of the non-pointer `F` decoder harmed performance across every seed (`U-1`: `19.66%` vs `F`: `27.65%`, paired mean **−7.99 pts** `[s7: −9.46, s19: −12.15, s41: −2.34]`; vs `F-cap`: **−8.43 pts**; vs `B0`: **−10.33 pts**; vs `J`: **−13.50 pts**; vs `A0 P`: **−42.36 pts**). Adding a second recurrent step (`U-2` vs `U-1`) dropped mean accuracy another **−2.52 pts** (`[s7: −7.99, s19: −1.48, s41: +1.91]`), cut narrative accuracy from `11.55%` to `7.20%`, cut edit-pair resolution from `1.67%` to `0.10%`, and pushed 5-hop accuracy down to `1.56%`. At four recurrent steps (`U-4`), the sole completed seed (`s7`) fell to `14.11%` (`−7.81` pts below `U-1 s7`), `0.26%` narrative, `0.00%` edit pairs, and `0.00%` on both 1-hop and 5-hop depth stress. Because `U-4` has only `n = 1` completed seed (`s19` and `s41` pending), `U-4` seed variability remains open; the pattern across `U-1` (`n = 3`), `U-2` (`n = 3`), and `U-4` (`n = 1`) is nonetheless consistent.

Per-seed metrics reveal an important trap in aggregate accuracy and 3-view invariance: almost all remaining accuracy in `U` comes from abstaining on the two unanswerable families (`X10`, `X12`, which together form `2/12 = 16.67%` of base worlds), while accuracy on the ten **answerable families (`X01–X09, X11`, `1,920` episodes per seed)** collapses toward zero:

| Model & seed | Overall | Answerable (`X01–09,11`) | Unans (`X10,12`) | Canon | Narr | Perm | Irrel | Row | Edit pair | 3-view conj | 5-hop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| U-1, s7 | 21.92 | **15.21** | 55.47 | 23.18 | 9.90 | 26.82 | 22.66 | 25.00 | 5.00 | 10.68 | 11.72 |
| U-1, s19 | 20.49 | **8.33** | 81.25 | 22.14 | 15.36 | 21.88 | 19.53 | 20.83 | 0.00 | 11.72 | 4.69 |
| U-1, s41 | 16.58 | **3.23** | 83.33 | 19.01 | 9.38 | 17.71 | 17.45 | 17.97 | 0.00 | 13.80 | 2.34 |
| U-2, s7 | 13.93 | **2.03** | 73.44 | 17.19 | 0.00 | 16.93 | 16.41 | 17.19 | 0.00 | 12.76 | 0.78 |
| U-2, s19 | 19.01 | **9.06** | 68.75 | 20.83 | 7.81 | 21.61 | 21.35 | 20.83 | 0.31 | 10.94 | 3.91 |
| U-2, s41 | 18.49 | **5.78** | 82.03 | 20.05 | 13.80 | 18.75 | 19.27 | 19.27 | 0.00 | 11.72 | 0.00 |
| U-4, s7 | 14.11 | **0.36** | 82.81 | 16.67 | 0.26 | 16.93 | 17.19 | 16.93 | 0.00 | 16.41 | 0.00 |

Notice `U-4 s7`: its `16.41%` (`63/384`) three-view conjunction score nearly equals the `16.67%` (`64/384`) fraction of unanswerable worlds while answerable accuracy is `0.36%` (`7/1,920`). Mean per-family accuracy confirms near-uniform collapse outside `X10`/`X12`: `U-1` averaged `6.94%–12.50%` across `X01–X09, X11` vs `72.92%` (`X10`) and `73.78%` (`X12`); `U-2` averaged `3.82%–7.29%` vs `74.83%`/`74.65%`; and `U-4 s7` scored `0.00%` on seven answerable families (`0.52%` on `X01`/`X04`, `2.60%` on `X08`) vs `81.77%` (`X10`) and `83.85%` (`X12`).

### 2. Test-time recurrence sweeps (`k = 1, 2, 4, 8`) and learning trajectories

Running each trained `U` checkpoint at mismatched test-time step counts (`u_depth_sweep` on the depth suite) exposes an unmistakable pathology: **running more recurrent steps actively destroys accuracy, and `k = 1` outperforms the model's own trained step count (`k = 2` on all three `U-2` seeds, `k = 4` on `U-4 s7`)**. Conversely, at `k = 4` and `k = 8`, accuracy plunges to **0.00%** on every single `U-1` and `U-2` run:

| Model & seed | Milestone `u0000→u0592→u1184→u1472→u2368` (576 views) | `u_depth_sweep` `k=1` | `k=2` | `k=4` | `k=8` |
|---|---|---:|---:|---:|---:|
| U-1, s7 *(trained k=1)* | 1.39 → 16.15 → 22.92 → 22.05 → 29.51 | **16.56** | 1.56 | 0.00 | 0.00 |
| U-1, s19 *(trained k=1)* | 0.00 → 16.49 → 16.32 → 17.01 → 20.31 | **7.03** | 0.31 | 0.00 | 0.00 |
| U-1, s41 *(trained k=1)* | 3.47 → 16.32 → 13.89 → 13.54 → 15.80 | **2.34** | 0.00 | 0.00 | 0.00 |
| U-2, s7 *(trained k=2)* | 2.08 → 16.67 → 10.76 → 16.49 → 15.28 | **4.06** | 2.03 | 0.00 | 0.00 |
| U-2, s19 *(trained k=2)* | 0.00 → 13.54 → 17.19 → 14.76 → 18.92 | **10.78** | 7.34 | 0.00 | 0.00 |
| U-2, s41 *(trained k=2)* | 5.38 → 15.45 → 15.62 → 14.06 → 16.49 | **7.50** | 4.06 | 0.00 | 0.00 |
| U-4, s7 *(trained k=4)* | 5.73 → 15.80 → 13.89 → 13.89 → 14.76 | **4.69** | 3.44 | 0.47 | 0.00 |

Every model jumped from ~0–6% to ~14–17% by update `592`—right around the `16.67%` unanswerable baseline—and stalled (`U-1 s41` peaked at `592`, `U-2 s7` at `592`, `U-4 s7` at `592`; only `U-1 s7` reached `29.51%` on the 576-view subset and `11.72%` at 5 hops).

### 3. Action collapse, gradient attenuation, and mechanistic diagnosis

First-operation, terminal-operation, and update-journal traces establish *how* the `U` models fail:

- **Defaulting to `ABSTAIN` on answerable worlds:** Across `1,920` answerable episodes per seed (where ideal `ABSTAIN` count is `0`), `U-1` chose `ABSTAIN` as its *first* action on `316–664` episodes (mean `489.0`) and as its *terminal* action on `564–1,187` (mean `927.0`); `U-2` abstained initially on `449–625` (mean `537.7`) and terminally on `807–1,435` (mean `1,081.7`); and `U-4 s7` abstained initially on `614` (`32.0%`) and terminally on **`1,383` of `1,920` answerable episodes (`72.0%`)**.
- **Single-symbol collapse on `ANSWER`:** When `U-2` and `U-4` did issue `ANSWER` on the first step, they mostly emitted one fixed symbol regardless of context: `U-2 s7` emitted `'J'` in `101/117` (`86.3%`) of first-step answers; `U-2 s41` emitted `'D'` in `161/180` (`89.4%`); and `U-4 s7` emitted `ANSWER` only `29` times out of `2,304` episodes (`ideal = 960`), emitting `'F'` on `27/29` (`93.1%`). Contrast `A1 F` seed `7` (`796` `ANSWER` actions spread across symbols, top `'J': 106`) and `A3 J` seed `7` (`840` `ANSWER`, top `'H': 86`).
- **Attenuated gradient norms and elevated training loss floor:** Across updates `1–2,368`, mean cross-entropy loss plateaued high above non-recurrent `F`: `U-1` `2.553 → 1.608`, `U-2` `2.528 → 1.660`, `U-4 s7` `2.435 → 1.664`, compared to `F` `2.581 → 1.317`, `F-cap` `2.581 → 1.263`, and `A0 P` `1.830 → 0.441`. Mean pre-clip gradient norms were consistently smaller and decayed further with depth: `U-1` `7.09 → 5.34`, `U-2` `6.18 → 5.54`, `U-4 s7` `6.64 → 3.85`, compared to `F` `11.46 → 9.44` and `F-cap` `11.26 → 10.77`.
- **Code-level architectural hypothesis (not yet causally isolated by intervention):** Inspection of `anra_v5/elendril/v8x/models.py` shows `_URecurrentStepBlock.step` initializes `z_state = h` (the Transformer hidden state at the decision token) and iteratively overwrites it via an unresidualized `nn.GRUCell(self.u_out(read_vec), z_state)` without skip connections back to `h`, normalization, or pointer copy readout (`use_pointer = False`, `val_head(z_state)`). In effect, `k` steps of `GRUCell` replacement without a residual identity path likely washes out token-specific information in `h` needed by `op_head` and `val_head`, making `ABSTAIN` and a single frequent value token the lowest-loss Bayes-optimal refuge. Whether decoupling weights (`U2-untied`), adding a pointer readout (`U-P`), or switching to gated residual delta updates with a persistent backbone anchor (`P-R-Iter`) cures this regression cannot be settled from Run 3 alone and is the exact causal question queued for `IIIa`/`IIIb` and V8-X3.

### 4. Exact cross-run reproducibility across all 36 X1 core jobs

Every one of the 36 overlapping jobs (`A1`, `A2`, `A3`, `D0`, `D1`, `D2` across seeds 7, 19, 41) reproduced its Run 1 or Run 2 development accuracy to **0.0000 percentage points** (`36/36` exact match across all `82,944` final dev episodes), and all `24/24` Run 2 jobs reproduced **identical `latest_checkpoint_sha256` hashes** on fresh Kaggle GPUs:

| Block | Re-run jobs | Prior Run | Prior Acc == Run 3 Acc | Max \|diff\| (pts) | Exact `checkpoint_sha256` matches |
|---|---:|---|---:|---:|---|
| `A1` (`F`, `F-cap`) | 6 | Run 1 | 6 / 6 | 0.0000 | 0 / 6 *(source metadata `8e62ca0f` → `98b9e331`)* |
| `A2` (`P-capR`, `R-noedge`) | 6 | Run 1 | 6 / 6 | 0.0000 | 0 / 6 *(source metadata `8e62ca0f` → `98b9e331`)* |
| `A3` (`B0`, `J`) | 6 | Run 2 | 6 / 6 | 0.0000 | **6 / 6** |
| `D0` (`P`, `R` on `PAIR-CE`) | 6 | Run 2 | 6 / 6 | 0.0000 | **6 / 6** |
| `D1` (`P`, `R` on `PAIR-EQ`) | 6 | Run 2 | 6 / 6 | 0.0000 | **6 / 6** |
| `D2` (`P`, `R` on `PAIR-EQ-CF`) | 6 | Run 2 | 6 / 6 | 0.0000 | **6 / 6** |
| **Total core overlap** | **36** | **Run 1 + Run 2** | **36 / 36** | **0.0000** | **24 / 24** (`98b9e331`-matched) |

*(The 12 Run 1 checkpoint hashes differ solely because `IdentityBindings.source_commit` is hashed inside the checkpoint manifest and updated cleanly from `8e62ca0f` to the repaired worker commit `98b9e331` via `legacy_compatibility_disclosure`; all numerical evaluations are bit-for-bit identical.)*

## Verification, compute, and custody

- **Archive integrity:** `ELENDRIL_V8_X2_ANALYSIS (2).zip` is `70,185,498` bytes, SHA-256 `84b73494ec17ddaeb5cb6da2fc26398794cdd7552294a1fbe62d77b080304074`. All `916` manifested entries in `BUNDLE_MANIFEST.json` matched byte size and SHA-256 (`0` mismatches, `0` unmanifested files among `917` total entries), and all `915` lines of `SHA256SUMS` verified cleanly.
- **Provenance bindings:** Campaign source `9ffc60e3e16adf9cf68e8f1551b19a5d5d5dbf02`, legacy worker source `98b9e3318bfa3c02024685bd8668323065f62811` (disclosed compatibility map from `8e62ca0f`), registry SHA-256 `df00cb63528389d1d14b71bd898e5f902aee5032c2686dc5c5866a2fd477fc35`, protocol SHA-256 `fd133dc2948827d6efe9e9acf6718f3bd34c824c141ab6adf124563ad22112c8`, and audit commit `ec9b742f8803c1d477bffe2505b6dd75c6f349a2`.
- **Independent episode & loss replay:** I regenerated the `D1-H` (`8e897a19...`) and `D2` (`5a130299...`) training dataset identities and all evaluation world cohorts, verified every world order hash (`digest(world_ids)`), and re-executed `verify_x1_episode` across all **225,664 recorded action sequences** (`99,072` early milestone + `99,072` final development + `27,520` depth stress episodes across `43` jobs). Every terminal verdict and all `258` milestone/depth summaries matched (`max_macro_diff = 2.220e-16`). All **101,824 optimizer update rows** (`43 × 2,368`) were contiguous with finite losses, and weighted `CE + 0.1·EQ + 0.1·CF` total loss reconstruction matched within `1.629e-07`. `PREFLIGHT_REPLAY_RECEIPT.json` recorded `verified=True, MATCHED`.
- **Compute & weights boundary:** The 43 job timers sum to **`50,798.54` worker-seconds (`14.11` worker-hours)**, including `9,574.56` s (`2.66` worker-hours) for the 7 new `U0`/`U1` runs (`U-1`: `1,338.6 s` mean, `U-2`: `1,374.8 s` mean, `U-4 s7`: `1,434.3 s`). Summed concurrent `2×T4` wave durations across the `22` waves equal **`26,705.28` s (`7.42` hours)** against the `555`-minute session slice limit. Verification here replays recorded actions; fresh neural weight inference and frozen probes require production checkpoint tensors stored outside this compact analysis ZIP.

## Global campaign coverage & next decisions

`X2_COMPLETION_AUDIT.json` records `49/192` completed obligations (`0` invalid, `0` negative, `143` pending = `125` mandatory + `18` conditional, `global_x2_status = "PARTIAL"`, `x3_unlocked = False`):

| Block | Eligible | Completed | Pending | Status & contents |
|---|---:|---:|---:|---|
| `I_X1_CORE` | 42 | **42** | 0 | `A0` (6) + `A1` (6) + `A2` (6) + `A3` (6) + `D0–D2` (18) complete |
| `IIIa_ORIGINAL_U` | 12 | **7** | 5 | `U-1` (3/3), `U-2` (3/3), `U-4` (`1/3` s7 done; `s19, s41`), `U2-untied` (`0/3`) |
| `II_FROZEN_A0_DIAGNOSIS` | 6 | 0 | 6 | Frozen A0 pointer/edge probes (deferred behind independent jobs) |
| `IIIb_NEW_MECHANISMS` | 30 | 0 | 30 | `DM/DM-add`, `P2/P2-dense`, `R2/R2-shuf`, `U-P/U-P-ctrl`, `PS/PS-ctrl` |
| `IV_HARD_TASKS` | 24 | 0 | 24 | `BIV-T13` (12) + `BIV-T14` (12) |
| `V_HORIZON_4736` | 6 | 0 | 6 | 4,736-update horizon extension (`P`, `DM`) |
| `EXTRA_T14_ROUTING_PREP` | 6 | 0 | 6 | `PS` vs `PS-ctrl` routing prep |
| `EXTRA_T13_TRANSFER_PARENTS` | 48 | 0 | 48 | `HIST_A`, `HIST_APRIME`, `HIST_DISJOINT_PRETRAIN`, `HIST_FRESH_CTRL` |
| `CONF_SOURCE` / `CONF_T13` / `CONF_T14` | 18 | 0 | 18 | Conditional confirmation cohorts (`not triggered` until base finishes) |
| **Total** | **192** | **49** | **143** | **Session slice `43/43 COMPLETE`; next queue ID `X1-U1-S20-U-4-standard-s19`** |

**Fresh-session import test (critical operational update):** Running the repository importer on `ELENDRIL_V8_X2_ANALYSIS (2).zip` **alone** recovered all **43 completed job IDs** with **0 rejections** (`36` X1 core + `7` `U0`/`U1`). Mounting Run 1 + Run 2 + Run 3 together also yielded `43` unique accepted IDs with `0` rejections. Because Run 3 is a strict superset of Run 1 + Run 2 on `9ffc60e3` (`36/36` exact match + `7` new `U` runs), **mounting ONLY `ELENDRIL_V8_X2_ANALYSIS (2).zip` in Kaggle Input for Session 4 resumes immediately at `X1-U1-S20-U-4-standard-s19`** and skips repeating all 43 completed jobs (assuming `~0.65 s/update` admits `13` jobs up to `X2-IIIb-PS-S20-PS-s7`, or `~40+` jobs under faster measured preflight timing).

### Next decisions

1. **Execution (Session 4 Resume):** Attach `ELENDRIL_V8_X2_ANALYSIS (2).zip` as a Kaggle Input dataset when running [ELENDRIL_V8_X2_KAGGLE_T4X2.ipynb](../../notebooks/ELENDRIL_V8_X2_KAGGLE_T4X2.ipynb) so verified-resume skips all `43` completed jobs (`~8+` GPU hours saved!) and immediately executes `U-4` (`s19`, `s41`), `U2-untied` (`s7`, `s19`, `s41`), and Block `IIIb` (`DM`, `P2`, `R2`, `U-P`, `PS`). Preserve `A0` weights if/when running `BII-FROZEN`; otherwise `run_admitted_waves` now records `BLOCKED_MISSING_PARENT` cleanly without stalling independent blocks.
2. **Science:** Standalone non-pointer recurrence (`U-1`, `U-2`, `U-4 s7`) fails decisively below non-recurrent `F` (`19.66%` / `17.14%` / `14.11%` vs `27.65%`) and loses all answerable competence beyond `1` step in test-time sweeps (`0.00%` at `k = 4, 8`). Wait for `U2-untied` (`3` seeds) and `U-4` (`s19`, `s41`) to verify whether layer untying helps without pointers, and treat `IIIb` `U-P` (iterative attention + pointer copy) and V8-X3's residual delta-gated `P-R-Iter` as the decisive test of whether keeping a pointer readout + skip-preserved working memory avoids GRU state washout.
