# ARK-018 FINAL CROSS-BRANCH AUDIT

**Status:** EXECUTED / COMPLETE / independently rechecked from returned bundle  
**Bundle:** `ARKENSTONE_ARK018_SCIENCE_BIRTH_RESULTS.zip`  
**Bundle SHA-256:** `cea50622b1725eb12308c54355616e2188b7956761db46289c6da33c09c31f7c`

This file is the `cymek-500m-readiness` branch's distilled authority for the completed ARK-018 result. It updates older cross-branch prose that still described ARK-018 as unexecuted. It does **not** change Arkenstone history or claim AGI, consciousness, identity, or broad reasoning.

## Integrity

Independent local audit of the returned ZIP found:

- ZIP readable and internally consistent;
- **40/40** JSON `receipt_sha256` values recomputed successfully using canonical sorted compact JSON after removing the receipt field;
- **39/39** members listed in `ARK-018_ZIP_CONTENT_MANIFEST.json` matched their recorded SHA-256;
- peS2o science shard bound to `b397427cd5964b7cc2a41264ca8789a0c020f96d4e403b314900798711a2ead5`;
- Birth Book bound to `8f17d092897f41df45100d227ecf7a2391ed5a5cfd3d3bcf373ad86c94b0c4f4`;
- REDTEAM status `PASS`;
- CUDA runtime Tesla T4, torch `2.11.0+cu128`;
- both frozen seeds `31801` and `31902`, all four arms, **8000/8000 updates** complete.

The data receipt reports 110,597 usable science documents, 99,635 TRAIN / 5,568 CONTROL / 5,394 SEALED, with split assignment by SHA-256 of normalized text. This is a ~20–25M controlled proxy, not target-scale evidence.

## Primary preregistered result

Official primary verdict: **`NO_LARGE_BIRTH_SPECIFIC_INTERNALIZATION_EFFECT`**.

| Seed | Arm | Science SEALED NLL | Birth NLL diagnostic | Birth SEALED content score |
|---|---|---:|---:|---:|
| 31801 | SCIENCE_ONLY | 4.2607 | 3.8309 | 0.4000 |
| 31801 | BIRTH_NATURAL_2PCT | 4.3054 | 1.6572 | 0.4667 |
| 31801 | BIRTH_REHEARSAL_10PCT | 4.4372 | 1.3214 | 0.4000 |
| 31801 | SCIENCE_REPLAY_10PCT_CONTROL | 4.2795 | 3.9301 | 0.4000 |
| 31902 | SCIENCE_ONLY | 4.2929 | 3.7534 | 0.4000 |
| 31902 | BIRTH_NATURAL_2PCT | 4.3053 | 1.4926 | 0.4000 |
| 31902 | BIRTH_REHEARSAL_10PCT | 4.4360 | 1.2387 | 0.4667 |
| 31902 | SCIENCE_REPLAY_10PCT_CONTROL | 4.2692 | 3.6504 | 0.4000 |

The preregistered Birth-specific C-vs-D SEALED content-score differences were `0.0000` and `+0.0667`, below the required `+0.10` in both seeds. Heavy Birth rehearsal therefore **strongly changed corpus prediction** but did not demonstrate the preregistered large transferable content-internalization effect.

The 10% Birth arm's science NLL cost relative to the token-matched 10% science-replay control was **+3.6846%** and **+3.9066%**, both below the preregistered 5% major-cost boundary but clearly nonzero.

## Most important exploratory result: later plasticity changed

| Seed | SCIENCE_ONLY | Birth 2% | Birth 10% | Science replay 10% |
|---|---:|---:|---:|---:|
| 31801 | 400 | 400 | **1200** | 300 |
| 31902 | 400 | 300 | **not sustained by 1500** | 300 |

The heavy Birth arm reached a raw qualifying point at step 1500 in seed 31902 but could not satisfy the required sustained-confirmation sequence before the 1500-step cap. The matched 10% science-replay control qualified at 300 steps in both seeds.

**Interpretation:** this is a **replicated exploratory plasticity signal**, not yet a demonstrated universal mechanism. It argues that training content can change not only what a model predicts, but also the ease with which the resulting checkpoint acquires a later controlled skill. The matched replay control makes “any repeated small corpus causes the slowdown” an inadequate explanation, but ARK-018 was not preregistered primarily to establish this plasticity mechanism.

## What ARK-018 changes in the master picture

- Real-text substrate evidence is no longer “unexecuted”; a ~21M science-pretrained proxy has now been studied causally.
- Heavy specialized rehearsal can buy strong distribution-specific prediction while imposing measurable science-modeling cost and an apparent later-learning cost.
- A low-dose arm (2%) captured a large Birth-NLL gain with much smaller apparent general/plasticity cost, but **2% is not proven optimal**.
- The result strengthens the need for a future continual-learning controller to optimize **retention and future acquisition simultaneously**, not merely prevent forgetting.
- It does not repair the separate Cymek production-representation formation failure exposed by CYR-GPU-011.

## Claim boundary

**DEMONSTRATED:** corpus-specific prediction change under controlled matched arms; measurable science-NLL tradeoff.  
**SUPPORTED / exploratory replicated signal:** heavy 10% Birth rehearsal slowed later temporary-binding acquisition relative to matched controls.  
**NOT_DEMONSTRATED:** better reasoning, broad semantic transfer, AGI, consciousness, identity, universal plasticity law, or target-scale transfer.
