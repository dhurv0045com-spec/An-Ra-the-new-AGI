# An-Ra PGE/Core checkpoint audit

Status: **Behavioral audit complete for all five checkpoints and the frozen EXP policy; campaign token-provenance integrity gate FAILED and is reported explicitly below.**

Audit date: 2026-08-28
Evaluation schema: `anra-pge-capability-audit/v1`
Execution: local RTX 4050, FP32, canonical stateful prefill + incremental decoding
Validation: 8,192 fixed held-out tokens from the exact continuation pack (2,048/domain)
Binding: 48 frozen four-fact nonce queries; four observed candidates/query

## Executive result

The step-22,517 checkpoint is real. Its parameters and both Adam moment families differ from the step-20,000 parent, Adam advanced to 22,517, and the full continuation pack completed. Training materially improved held-out language-model loss across every measured domain.

It did **not** produce the tested query-conditioned cognitive capabilities. Exact copy, nonce context use, free multi-fact binding, and two-hop composition all remain 0 on the PGE battery. Candidate scoring exposes only a chance-level signal (12/48 with four choices), but free decoding never realizes a correct answer. True counterfactual query normalization does not transfer as a reliable repair to the final PGE checkpoint: it scores 10/48 versus RAW 12/48.

The strongest behavioral checkpoint among the available PGE lineage is **step 21,800**, not the final. It combines most of the final's LM gain with far less raw degeneration. SFT6 remains strongest for exact-copy/context behavior, while no checkpoint demonstrates composition on this fresh battery.

## Checkpoint reality and provenance

| Evidence | Parent 20,000 | Intermediate 21,800 | Final 22,517 |
|---|---:|---:|---:|
| Full-resume | yes | yes | yes |
| Model/trainer/Adam step | 20,000 | 21,800 | 22,517 |
| Canonical parameter SHA-256 | `c836262df01171cc6813a804c14315fc25b4c7efa7bce39e4f771b6b562a4e19` | `cd5605568d817a8e0898ac772db5598f010fe5d04198c67f83f4a521f9cf51b1` | `de1ca6b813897b5ddd6a66fc68e1a690c3cd307cd48d1fc2913a5ff2544b452a` |
| Adam `exp_avg` SHA-256 | `574ca093f5ffdb2cf1d802f3a35986576e45052250309806aef965e9d28395b7` | `1486a3014660952965d6b7578e75d25ce98a83cb124d9c74dd6c7e98d5723b7f` | `f2e96211a184b7af2ae42f92863bfc8c9b7c2476ea334dbddad78110ec9daee5` |
| Adam `exp_avg_sq` SHA-256 | `04ed4981a27d8a369912e409c94a198ada42b2230f1d57b9f8f1c76001233be6` | `71eb98fdf36ee1a96fa7d855525492c5a65c02cb8188abe3f83acf2c17e5d57c` | `1abc7b2c7616cb76c6ee6ea0d640be02fe16ece70d6ab00b861c9be7df5b2526` |
| File SHA-256 | `8b39323abe855f5164bf21851bc2a3dece2157ae38389e59f727db47ab63dd4c` | `ae64a43839e344e966c5619bde337943577005d48bf5ec35ed45acfa84af371a` | `2ea76d47ed89fd50d6118e280a3a0cfc80eb9b45f6fd8041d1cd331dd655272d` |

All three step triplets are internally consistent. Parent→intermediate and intermediate→final both change parameters, first moments, and second moments. The final tokenizer contract loads strictly; the parent requires declared legacy-forensic mode because its old contract lacks the modern usability field, although its vocabulary/probe identity matches.

### Token finding

Do **not** call this a verified 2.7B-token checkpoint.

- The TPU continuation contract is 2,517 optimizer steps × 131,072 tokens/step = **329,908,224 certified continuation tokens**.
- The pack manifest declares 330,000,384 effective pack tokens and a 170M→500M campaign.
- The step-20,000 parent itself records `tokens_seen=327,827,071`, which conflicts with the pack's claimed 170M starting point.
- The parent also records a token window `start_token=324,550,271, end_token=500,000,000` and the same 500M phase recipe. The downloaded pack is named `170m_to_500m` and declares itself an additional 330M-token continuation. These are overlapping/stale campaign labels, not a single consistent lifetime counter.
- Applying the TPU tokens/step contract retroactively to all 20,000 historical steps is invalid. That is the calculation that produced the false ~2.95B figure.
- If the parent's `tokens_seen` is cumulative and additive, the implied total is 657,735,295; if the signed pack campaign lineage is authoritative, the intended total is ~500M. Because those sources disagree, **total lifetime tokens are not certifiable from the present artifacts**.
- `lineage.json` therefore reports `checkpoint_is_real=true` for the weight/Adam/step evidence but `campaign_provenance_consistent=false` for the lifetime-token claim.

The exact training-manifest SHA recorded by intermediate/final is `87cf95bef97ba76cc8af40939f3a8fba09be8df24cbf674975f3d7ffa89bc5ab`. The held-out validation manifest used here hashes to `03587987b9a4d153d61041406543e81b5eb70451b8fee56e1f2d13c1242a35cd`, matching the signed pack manifest.

## Capability matrix

`RAW` means learned behavior without decode penalties. `ASSISTED` reports the best declared intervention for that row, never silently mixing it with raw behavior.

| Capability | Parent 20,000 | Intermediate best 21,800 | Final 22,517 | SFT6 | SFT7 |
|---|---:|---:|---:|---:|---:|
| Held-out loss ↓ | 2.1884 | 2.0156 | **1.9710** | not run | not run |
| Perplexity ↓ | 8.9206 | 7.5054 | **7.1779** | not run | not run |
| Exact copy, RAW | 0/6 | 0/6 | 0/6 | **5/6** | 4/6 |
| Nonce context use, RAW | 0/8 | 0/8 | 0/8 | **7/8** | **7/8** |
| Multi-fact FREE | 0/48 | 0/48 | 0/48 | 3/48 | 2/48 |
| Candidate selection RAW | **12/48** | 11/48 | **12/48** | **13/48** | **13/48** |
| Exact constrained realization | 12/48 | 11/48 | 12/48 | **13/48** | **13/48** |
| Counterfactual normalization | 12/48 | **12/48** | 10/48 | 9/48 | 9/48 |
| Composition FREE | 0/12 | 0/12 | 0/12 | 0/12 | 0/12 |
| Composition sampled/assisted | 0/12 | 0/12 | 0/12 | 0/12 | 0/12 |
| Raw degenerate continuations ↓ | 3/4 | **1/4** | 3/4 | 0/4 | 0/4 |
| Assisted degenerate continuations ↓ | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 |

With four candidates, chance expectation is 12/48. No PGE RAW selection score exceeds chance on this frozen set; SFT6/SFT7 are only one case above chance (13/48). The final's 10/48 normalized score is below its own RAW score.

## Causal decomposition of final multi-fact failures

- **FREE:** 0/48. Canonical greedy generation never begins with the gold value.
- **RAW candidate choice:** 12/48. Candidate scoring is at chance; selection failures are 36/48 (75%).
- **CONSTRAINED realization:** 12/48. Deterministically emitting the RAW-selected observed value (without another model forward pass) rescues all twelve latent correct selections.
- **COUNTERFACTUAL NORMALIZATION:** 10/48. It repairs four RAW misses but harms six RAW successes, for a net loss of two.
- **Realization failure:** 12/48 overall (25%), or 12/12 (100%) conditional on RAW having selected the correct value.

This cleanly separates two failures: the selector is wrong on 75% of cases, and the decoder fails to realize every correct latent selection that remains. Constrained decoding repairs realization only; it cannot repair the dominant selection error.

## What PGE training bought

Every validation domain improves monotonically from parent to final:

| Domain | Parent | Step 21,800 | Final |
|---|---:|---:|---:|
| FineMath | 1.8624 | 1.6821 | **1.6386** |
| FineWeb-Edu | 2.5781 | 2.3771 | **2.3157** |
| Permissive code | 2.2742 | 2.0994 | **2.0552** |
| Science/technical | 2.0388 | 1.9039 | **1.8746** |

Overall loss falls 9.93% and perplexity falls 19.54% parent→final. PGE therefore learned ordinary next-token distribution structure. Step 21,800 also sharply improves unassisted diversity (repeated 3-gram ratio 0.442→0.134), but the final partially regresses to 0.359. Decode controls eliminate measured repetition for all three, showing a useful runtime repair that does not create task correctness.

No tested SFT-like capability appears natively: exact copy, nonce use, free query binding, and composition remain at zero; candidate selection remains chance.

## Required conclusions

**BEST CURRENT CHECKPOINT:** Step 21,800 (`anra-v4-tpu-latest (2).pt`) for the PGE lineage; SFT6 is the best exact-memory specialist (`5/6` copy, `7/8` nonce context).
**WHAT 2.7B PGE GAINED:** The run is not verified as 2.7B. The certified ~329.9M-token continuation lowered held-out LM loss in all four domains and temporarily reduced raw degeneration.
**WHAT IT LOST:** Final-vs-intermediate raw degeneration worsened (1/4→3/4), and normalized candidate selection worsened (12/48→10/48).
**WHICH OLD SFT CAPABILITIES IT NOW HAS NATIVELY:** None of SFT6's exact-copy or nonce-context gains transferred to PGE; PGE remains `0/6` and `0/8`.
**WHICH STILL REQUIRE RUNTIME REPAIR:** Repetition is repaired by decode controls; exact realization is repaired by candidate-constrained emission. Selection is not reliably repaired by true counterfactual normalization.
**SELECTION FAILURE RATE:** 36/48 = 75% at the final checkpoint.
**REALIZATION FAILURE RATE:** 12/48 = 25% overall; 12/12 = 100% conditional on a correct RAW selection.
**COMPOSITION STATUS:** Not demonstrated: 0/12 FREE and 0/12 assisted at all three PGE checkpoints.
**DOES EXP INTERVENTION POLICY TRANSFER:** No robust transfer on this frozen distribution. The SFT6-trained v7 policy scores parent `13/48`, step 21,800 `12/48`, final `11/48`, SFT6 `9/48`, and SFT7 `9/48`; fixed deterministic candidate emission scores `12/48`, `11/48`, `12/48`, `13/48`, and `13/48` respectively.
**TRAIN MORE NOW:** NO. More undirected PGE improved LM loss without improving the target cognitive behaviors and the final regressed on degeneration/normalization.
**HIGHEST-LEVERAGE NEXT EXPERIMENT:** Run a small, preregistered query-conditioned contrastive-SFT ablation from step 21,800, evaluated on this frozen battery plus the proven EXP QIM fixture, with constrained realization held identical across RAW and normalized arms.
**WHY:** The five-way comparison shows that PGE improved ordinary LM loss while SFT specifically improved exact memory; the next experiment must target selection while preserving the clean selection/realization separation and avoid another undirected large token run.

## EXP comparison and policy transfer

The exact SFT6 and SFT7 model-only checkpoints were recovered from `C:\Users\ankit\An-Ra-xexp\checkpoints`, and the frozen v7 policy from `C:\Users\ankit\An-Ra-xexp\output\self_model_v7.json`. Their file and parameter hashes are recorded in `sft_artifact_inventory.json`.

The policy transfer matrix is in `policy_transfer_matrix.json`. It is a valid observed-state transfer test, but not evidence that the policy has learned a universal PGE selector: on the fresh four-word distribution it is no better than deterministic candidate emission and is worse on the final PGE and both SFT checkpoints. The historical EXP QIM/MC wins remain valid for their own sealed fixtures; they do not override this same-probe result.

For context, the same frozen v7 policy did transfer on its native EXP fixtures: MC-v7 `310/480` vs always-normalized `274/480`, MC-v8 `291/480` vs `248/480`, and SFT7 MC-v9 transfer `320/480` vs `280/480`. Those historical receipts are captured in `exp_policy_reference.json`; the contrast with the new 48-query battery is evidence of distribution/format sensitivity, not evidence that the old EXP result never existed.

| Checkpoint | Always FREE | Always CONSTRAINED | Always NORMALIZED | Frozen v7 policy |
|---|---:|---:|---:|---:|
| Parent 20,000 | 0/48 | 12/48 | 12/48 | **13/48** |
| Intermediate 21,800 | 0/48 | 11/48 | **12/48** | 12/48 |
| Final 22,517 | 0/48 | **12/48** | 10/48 | 11/48 |
| SFT6 | 3/48 | **13/48** | 9/48 | 9/48 |
| SFT7 | 2/48 | **13/48** | 9/48 | 9/48 |

## Raw receipts

- `lineage.json`: file/parameter/Adam hashes, steps, and pack-token provenance.
- `parent_step20000_cf.json`, `intermediate_step21800_cf.json`, `final_step22517_cf.json`: corrected true-counterfactual receipts.
- `sft6_cf.json`, `sft7_cf.json`: direct SFT6/SFT7 same-probe receipts.
- `policy_transfer_matrix.json`: fixed baselines versus frozen EXP policy.
- `sft_artifact_inventory.json`: checkpoint and policy inventory/hashes.
- `token_provenance_ledger.json`: corrected token accounting and the failed campaign-provenance invariant.
- `exp_policy_reference.json`: immutable references to the native EXP policy-transfer receipts.
- `pack/.../validation/manifest.json`: exact held-out shard inventory and hashes (large `.npy` shards remain external).
- `step22517/verification.json`: prior strict final-download verification.

The reusable scripts are `scripts/audit_checkpoint_lineage.py` and `scripts/audit_pge_capabilities.py`.
